"""
SentryAML - RAG 层 R5b/R6a：检索接口（稠密 / BM25 / 混合 + 条款编号精确路由）+ 冒烟测试
==========================================================================================
search(question, as_of, variant, mode) 是之后 R6 评测、R8 Agent 工具共用的入口。

三种模式：
  dense   只用 bge-m3 向量：按意思匹配，措辞不同、中英文都能对上
  bm25    只用 BM25：按关键词匹配，对编号（6.14C）、专有名词（COSMIC、STRO）最准
  hybrid  两路各取前 N 名，用 RRF（倒数排名融合）合并：
              RRF 分数 = Σ_各路  1 / (k + 该块在这一路的名次)        （Qdrant 的名次从 0 开始）
          只看名次、不看原始分数：两路的分数尺度完全不同（余弦相似度在 0–1，BM25 分数没有上限），
          直接相加会被尺度大的一路主导；名次则天然可比。两路都排得靠前的块得分最高。
          k 越大，名次之间的差距越小，越看重"两路共识"；k 越小，越看重"某一路的第一名"。
          注意：Qdrant 默认 k = 2，不是 RRF 原论文的 60（R5b 冒烟测试的分数 0.8333 = 1/2 + 1/3 暴露了这一点）。
          这里显式传入 rrf_k，默认 60；R6 的消融会比较 2 和 60。

条款编号精确路由（exact，D39 规划的"条款编号精确表"）：
  问题里明确写了 "paragraph 12.5 of MAS Notice 626"、"Guidelines 14-6"、"Case Study G" 时，
  直接按（文件族, 条款编号）取出包含该条款的块，排在最前面，其余名额由混合检索补足。
  原因：编号对向量模型几乎没有语义（"12.5" 和 "12.3" 的向量很接近），BM25 也只把它当作一个普通词。

按调查日期过滤（D45）：只返回 effective_from ≤ 调查日期 < effective_to 的版本。
过滤和搜索在 Qdrant 内部同时执行，而不是先搜前 N 个再删：后者可能把有效版本删到不够 N 个。
在混合模式里，过滤条件要放进每一路（prefetch），否则某一路会先选出过期版本，挤占名额。

两阶段检索（R6b，rerank=True）：
  1. 第一阶段：混合检索（+ 编号路由）取 pool 个候选，求"别漏"
  2. 引用扩展（expand=True）：对前 expand_top 个候选，沿交叉引用（引用 / 被引用）补入相邻的块
     （SQL 函数 kb_expand_chunk，017）。多跳题的第二条答案往往和第一条没有相似的措辞，
     只能靠引用关系找到，例如 Notice 6.22 ← TM 指引脚注 3
  3. 重排：bge-reranker-v2-m3 把"问题 + 块"放在一起读，给每个候选打分，按分数重新排序
  编号路由命中的块仍然固定在最前（D50）：最初改为"由重排决定名次"，但实测重排模型把只是提到
  "paragraph 12.5"的 12.3 排在了真正的 12.5 前面，按编号查的题 MRR 从 1.0 跌到 0.39。
  重排模型判断的是正文语义，而条款编号存在于元数据里，不在正文中。

同分的确定性（D70，R9b-1 回归测试中发现）：
  RRF 只看名次，不同的名次组合常常算出完全相同的分数（"只在向量第 20 名"与"只在 BM25 第 20 名"）。
  同分块的先后由 Qdrant 内部决定，每次请求可能不同；同分跨在候选池截断线（第 30 名）上时，
  进入候选池的块时而是这个、时而是那个，同一个问题的检索结果因此不可复现（dev 上 72 题中 15 题的候选池会变）。
  修正：每一步截断都由我们自己按"分数降序、同分按块编号升序"排序后再截，向 Qdrant 多要 TIE_MARGIN 个候选
  保证截断线附近的同分块都拿得到；重排之后的排序、引用扩展补入的顺序也使用同样的规则。
  剩余风险：向量 / BM25 两路各取前 per_route 个是在 Qdrant 内部完成的，那里的同分无法从外部决定；
  dev 72 题 × 3 次的扫描中两路都没有出现不稳定，记录在案。

用法：
    python scripts/kb_search.py                      # 用 R4 评测集里的 8 道题做冒烟测试
    python scripts/kb_search.py "What is COSMIC?"    # 查一个问题（默认调查日期为今天，完整两阶段）
"""

from __future__ import annotations

import json
import re
import sys
import threading
from datetime import date
from pathlib import Path

from qdrant_client import models

import kb_bm25
from db import connect
from kb_embed import get_embedder
from kb_index import point_id, qdrant, variant_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_client, _embedder = None, None
TIE_MARGIN = 20      # 第一阶段多要的候选数：截断线附近的同分块都拿到手，再由我们按确定的规则截断（D70）


def _cut(points: list, limit: int, fetched: int) -> list:
    """按（分数降序，块编号升序）排序后取前 limit 个。同分时由块编号决定先后，而不是由 Qdrant 的处理顺序决定。
    如果同分一直延续到多要的最后一个，说明 TIE_MARGIN 不够，截断仍可能不确定：发出警告而不是悄悄继续。"""
    pts = sorted(points, key=lambda p: (-p.score, p.payload["chunk_id"]))
    if len(pts) >= fetched > limit and pts[limit - 1].score == pts[-1].score:
        import warnings
        warnings.warn(f"同分块从第 {limit} 名一直延续到第 {fetched} 名：TIE_MARGIN 不够，候选池截断可能不确定")
    return pts[:limit]


_lock = threading.Lock()


def _clients():
    """惰性创建 Qdrant 客户端和向量模型。加锁：R7 评测并发调用时，多个线程不能同时各建一个（D53）。"""
    global _client, _embedder
    with _lock:
        if _client is None:
            _client, _embedder = qdrant(), get_embedder()
    return _client, _embedder


def as_of_filter(as_of: date) -> models.Filter:
    t = f"{as_of.isoformat()}T00:00:00Z"
    return models.Filter(must=[
        models.FieldCondition(key="effective_from", range=models.DatetimeRange(lte=t)),
        models.FieldCondition(key="effective_to", range=models.DatetimeRange(gt=t)),
    ])


# ---------------------------------------------------------------------------
# 条款编号精确路由
# ---------------------------------------------------------------------------
DOC_HINTS = [(re.compile(r"\bguidelines?\b", re.I), "mas_n626_guidelines"),
             (re.compile(r"\b(?:tm|transaction monitoring) guidance\b", re.I), "mas_tm_guidance"),
             (re.compile(r"\bnotice\b", re.I), "mas_n626")]
DOT_ID = r"\d{1,2}[A-Z]?\.\d{1,2}[A-Z]?"
HYPHEN_ID = r"\d{1,2}(?:-(?:\d{1,2}[A-Z]?|[A-Z])){1,3}"


def parse_clause_refs(question: str) -> list[tuple[str, str]]:
    """从问题里找出明确写出的条款编号，返回 [(文件族, unit_id)]。宁缺毋滥：没有把握的写法不识别。"""
    fam_hint = next((fam for rx, fam in DOC_HINTS if rx.search(question)), None)
    refs = []
    for m in re.finditer(r"\bcase study ([A-N])\b", question, re.I):
        refs.append(("mas_tm_guidance", f"case-{m.group(1).upper()}"))
    for m in re.finditer(r"\bfootnote (\d{1,2})\b", question, re.I):
        if fam_hint:
            refs.append((fam_hint, f"fn{m.group(1)}"))
    for m in re.finditer(r"\b(?:appendix\s+)?([AB]-\d{1,2})\b", question):
        refs.append(("mas_n626_guidelines", m.group(1)))
    for m in re.finditer(rf"\b(?:paragraphs?|para\.?)\s+({DOT_ID}|{HYPHEN_ID})\b", question, re.I):
        uid = m.group(1).upper() if re.search(r"[a-z]$", m.group(1)) else m.group(1)
        # 编号语法指明文件（与 kb_xrefs 同一规则）：连字符编号只属于 Guidelines；点号编号默认 Notice，
        # 但问题里点名了 TM 指引时归 TM 指引
        if re.fullmatch(HYPHEN_ID, uid):
            fam = "mas_n626_guidelines"
        else:
            fam = "mas_tm_guidance" if fam_hint == "mas_tm_guidance" else "mas_n626"
        refs.append((fam, uid))
    return list(dict.fromkeys(refs))


def exact_lookup(question: str, as_of: date | None, collection: str, use_filter: bool = True) -> list[dict]:
    client, _ = _clients()
    out = []
    for fam, uid in parse_clause_refs(question):
        must = [models.FieldCondition(key="family", match=models.MatchValue(value=fam)),
                models.FieldCondition(key="unit_ids", match=models.MatchValue(value=uid))]
        if use_filter:
            must += as_of_filter(as_of or date.today()).must
        pts, _ = client.scroll(collection, scroll_filter=models.Filter(must=must), limit=5, with_payload=True)
        # 被切成多段的长条款会有多个块；按块编号排序，保证第 1 段在前
        out += sorted((p.payload for p in pts), key=lambda p: p["chunk_id"])
    return out


def expand(hits: list[dict], as_of: date, collection: str, top: int = 5, cap: int = 10
           ) -> tuple[list[dict], dict[str, set[str]]]:
    """
    沿交叉引用找前 top 个候选的相邻块。返回：
      新补入的块（不在候选里的，最多 cap 个）
      links：{块: 这一块里承载引用的条款}，对已在候选里的邻居也记录 —— 重排时要单独给这些条款打分
    """
    have = {h["chunk_id"] for h in hits}
    new: dict[str, str] = {}
    links: dict[str, set[str]] = {}
    with connect() as conn, conn.cursor() as cur:
        for h in hits[:top]:
            cur.execute("SELECT direction, chunk_id, link_unit FROM kb_expand_chunk(%s, %s)", (h["chunk_id"], as_of))
            for direction, cid, unit in cur.fetchall():
                if cid in have or cid in new or len(new) < cap:
                    links.setdefault(cid, set()).add(unit)
                if cid not in have and cid not in new and len(new) < cap:
                    new[cid] = f"graph:{direction}"
    if not new:
        return [], links
    client, _ = _clients()
    pts = client.retrieve(collection, ids=[point_id(c) for c in new], with_payload=True)
    order = {c: i for i, c in enumerate(new)}          # Qdrant 按 id 取点时不保证返回顺序：按补入的先后排好（D70）
    pts = sorted(pts, key=lambda p: order[p.payload["chunk_id"]])
    return [{"score": None, "route": new[p.payload["chunk_id"]], **p.payload} for p in pts], links


def unit_texts(pairs: list[tuple[str, str]]) -> list[str]:
    """单条条款的文字（带文件名和编号），给"聚焦重排"用。pairs = [(chunk_id, unit_id)]"""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT k.chunk_id, u.unit_id, d.title, d.version_label, u.text
                       FROM kb_chunks k JOIN kb_documents d USING (doc_id)
                       JOIN kb_units u ON u.doc_id = k.doc_id AND u.unit_id = ANY(k.unit_ids)
                       WHERE k.chunk_id = ANY(%s)""", (list({c for c, _ in pairs}),))
        rows = {(c, u): f"{t} ({v}) | {u}\n\n{txt}" for c, u, t, v, txt in cur.fetchall()}
    return [rows.get(p, "") for p in pairs]


def rerank_texts(chunk_ids: list[str], variant: str) -> list[str]:
    """重排看到的文字与建索引时一致（raw / hdr / ctx），保证消融对比公平。"""
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT c.chunk_id, d.title, d.version_label, c.unit_ids, c.heading_path, c.text, c.context
                       FROM kb_chunks c JOIN kb_documents d USING (doc_id) WHERE c.chunk_id = ANY(%s)""",
                    (chunk_ids,))
        cols = [d.name for d in cur.description]
        rows = {r[0]: dict(zip(cols, r)) for r in cur.fetchall()}
    return [variant_text(rows[c], variant) for c in chunk_ids]


def search(question: str, as_of: date | None = None, variant: str = "ctx", mode: str = "hybrid",
           k: int = 10, per_route: int = 50, rrf_k: int = 60, exact: bool = False,
           use_filter: bool = True, rerank: bool = False, pool: int = 30,
           expand_graph: bool = False, expand_top: int = 5, collection: str | None = None,
           rerank_query: str | None = None) -> list[dict]:
    """
    rerank_query（R8c）：检索用 question，重排和条款编号路由用 rerank_query（默认相同）。
    HyDE 时 question 是模型写的假想段落，但"是否回答了问题"必须对照用户的原问题判断；
    编号路由也只认原问题里写明的编号，不能认假想段落里模型自己编出来的编号。
    use_filter=False 只用于消融（量化"按日期过滤"本身的价值），正式检索永远按日期过滤。
    rerank=True 时第一阶段取 pool 个候选；expand_graph=True 时再沿交叉引用补入相邻的块。
    collection 默认是 kb_<variant>；正式检索用别名 kb_current（见 retrieve）。
    """
    as_of = as_of or date.today()
    first_k = pool if rerank else k
    client, embedder = _clients()
    collection = collection or f"kb_{variant}"
    flt = as_of_filter(as_of or date.today()) if use_filter else None
    si, sv = kb_bm25.query_vector(question)
    sparse = models.SparseVector(indices=si, values=sv) if si else None

    if mode == "bm25" and sparse is None:
        return []                                        # 中文问题切不出英文词，BM25 这一路为空
    fetch = first_k + TIE_MARGIN
    if mode == "dense" or (mode == "hybrid" and sparse is None):
        dense = embedder.embed([question], use_cache=CACHE)[0]
        res = client.query_points(collection, query=dense, using="dense", query_filter=flt,
                                  limit=fetch, with_payload=True)
    elif mode == "bm25":
        res = client.query_points(collection, query=sparse, using="bm25", query_filter=flt,
                                  limit=fetch, with_payload=True)
    elif mode == "hybrid":
        dense = embedder.embed([question], use_cache=CACHE)[0]
        res = client.query_points(
            collection,
            prefetch=[models.Prefetch(query=dense, using="dense", filter=flt, limit=per_route),
                      models.Prefetch(query=sparse, using="bm25", filter=flt, limit=per_route)],
            query=models.RrfQuery(rrf=models.Rrf(k=rrf_k)),
            query_filter=flt, limit=fetch, with_payload=True)
    else:
        raise ValueError(f"未知模式 {mode}")
    hits = [{"score": p.score, "route": mode, **p.payload} for p in _cut(res.points, first_k, fetch)]
    if exact:
        pinned = [{"score": None, "route": "exact", **p}
                  for p in exact_lookup(rerank_query or question, as_of, collection, use_filter)]
        seen = {h["chunk_id"] for h in pinned}
        hits = pinned + [h for h in hits if h["chunk_id"] not in seen]
    if not rerank:
        return [{"rank": i + 1, **h} for i, h in enumerate(hits[:k])]

    links: dict[str, set[str]] = {}
    if expand_graph:
        extra, links = expand(hits, as_of, collection, top=expand_top)
        hits = hits + extra
    import kb_rerank
    # 聚焦重排（D50）：经由引用关联到的条款（例如长块末尾的脚注 3）单独打一次分，取两者较高者。
    # 整块读起来讲的是别的事时，承载引用的那一条本身可能正是答案。
    # 整块文字和单条文字放进同一次重排请求（D51）：分两次请求会多一次网络往返，延迟几乎翻倍
    focus = [(h["chunk_id"], u) for h in hits for u in sorted(links.get(h["chunk_id"], ()))]
    docs = rerank_texts([h["chunk_id"] for h in hits], variant) + (unit_texts(focus) if focus else [])
    all_scores = kb_rerank.rerank(rerank_query or question, docs, use_cache=CACHE)
    scores, focus_scores = all_scores[:len(hits)], dict(zip(focus, all_scores[len(hits):]))
    for i, (h, sc) in enumerate(zip(hits, scores)):
        h["first_stage_rank"] = i + 1          # 在候选列表里的位置（引用扩展补入的排在第一阶段候选之后）
        fs = [v for (c, _), v in focus_scores.items() if c == h["chunk_id"]]
        h["score"] = max([sc] + fs)
        if fs and max(fs) > sc and h["route"] != "exact":
            h["route"] += "+focus"
    # 编号路由命中的块固定在最前（D50）：重排模型判断的是文字语义，条款编号不在正文里；
    # 实测把"Subject to paragraph 12.5 …"（12.3）排到了真正的 12.5 前面
    pinned = [h for h in hits if h["route"] == "exact"]
    rest = sorted((h for h in hits if h["route"] != "exact"), key=lambda h: (-h["score"], h["chunk_id"]))
    return [{"rank": i + 1, **h} for i, h in enumerate((pinned + rest)[:k])]


# ---------------------------------------------------------------------------
# 正式检索入口（D51：按 D50 预先写定的规则，在开发集上选定）
# ---------------------------------------------------------------------------
CACHE = True          # 评测时 --cold 会关掉缓存读取，用来测真实延迟
FINAL = dict(variant="hdr", mode="hybrid", exact=True, rerank=True, expand_graph=True, pool=30, rrf_k=60)


def retrieve(question: str, as_of: date | None = None, k: int = 8, query: str | None = None) -> list[dict]:
    """
    Agent 检索工具调用的入口：最终配置，查询别名 kb_current（指向哪个集合由 kb_index.py --set-current 决定）。
    query（R8c 的 HyDE 消融）：用另一段文字去检索，重排和编号路由仍按 question。
    """
    if query is None:
        return search(question, as_of, k=k, collection="kb_current", **FINAL)
    return search(query, as_of, k=k, collection="kb_current", rerank_query=question, **FINAL)


def smoke():
    path = PROJECT_ROOT / "reports" / "rag_golden_resolved.json"
    qs = {q["id"]: q for q in json.loads(path.read_text(encoding="utf-8"))}
    # 每类挑一道：改写过的查找题、场景题、按编号查、定义、中文、两道版本题、无答案题
    picks = ["q002", "q042", "q063", "q034", "q068", "q019", "q020", "q075"]
    print(f"{'题号':<6}{'类型':<13}{'dense':>7}{'bm25':>7}{'hybrid':>8}   问题")
    for qid in picks:
        q = qs[qid]
        gold = {c for cs in q["gold_chunks"] for c in cs}
        ranks = []
        for mode in ("dense", "bm25", "hybrid"):
            hits = search(q["question"], date.fromisoformat(q["as_of"]), "ctx", mode, k=10)
            r = next((h["rank"] for h in hits if h["chunk_id"] in gold), None)
            ranks.append("—" if not gold else (str(r) if r else ">10"))
        print(f"{qid:<6}{q['type']:<13}{ranks[0]:>7}{ranks[1]:>7}{ranks[2]:>8}   {q['question'][:60]}")
    print("\n数字是标准答案块在前 10 名里的名次（>10 表示没进前 10，— 表示无答案题）。")
    print("这里只是确认索引和过滤能正常工作；系统的评测在 R6 用全部 54 道开发集题做。")

    q = qs["q019"]
    print(f"\n版本过滤检查（{q['question'][:50]}…）：")
    for d in ("2025-03-01", "2026-09-30"):
        top = search(q["question"], date.fromisoformat(d), "ctx", "hybrid", k=3)
        print(f"  调查日期 {d}：" + "，".join(h["chunk_id"] for h in top))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="检索一个问题（最终配置）；不带问题时运行冒烟测试")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--as-of", type=date.fromisoformat, default=None, help="调查日期，默认今天")
    ap.add_argument("--collection", default=None, help="默认按最终配置查 kb_hdr；kb_current 设好后可传 kb_current")
    a = ap.parse_args()
    if a.question:
        for h in search(" ".join(a.question), a.as_of, k=a.k, collection=a.collection, **FINAL):
            print(f"{h['rank']}. [{h['chunk_id']}] 重排分 {h['score']:.3f}  来源 {h['route']:<12} "
                  f"候选第 {h['first_stage_rank']} 位  含条款 {', '.join(h['unit_ids'])}")
            print(f"      {h['text'][:110]!r}")
    else:
        smoke()
    if _client is not None:
        _client.close()