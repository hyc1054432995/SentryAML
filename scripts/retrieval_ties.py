"""
SentryAML - R9b-1：检索不确定性的根因确认
==========================================
regress_diag.py 的结果：v229 的首轮检索在两种结果之间来回切换（5 个进程里 3 次 / 2 次）。
两种结果的唯一区别是候选池里第 30 个位置：一次是 B-9，一次是 6-15-6（"第一阶段名次 30"）。
重排分数每次完全相同（来自缓存），所以变化发生在重排之前，即第一阶段的混合检索。

第一阶段（kb_search.search，mode=hybrid）分三步，每一步都有一个"截断"：
  1. 向量检索取前 50（per_route）
  2. BM25 取前 50
  3. 两路用 RRF 融合，取前 30（pool）进入重排
RRF 只看名次：score = 1/(60 + 向量名次) + 1/(60 + BM25 名次)。不同的名次组合可能得到完全相同的分数，
比如"向量第 20、BM25 没有" 和 "BM25 第 20、向量没有" 同分。同分的块谁排在前面，由 Qdrant 内部决定，
不保证每次一样。如果同分正好跨在第 30 名的截断线上，进入候选池的就会时而是这个、时而是那个。

假设：v229 的第 30 名处有 RRF 同分（或者某一路第 50 名处有同分），同分的块被 Qdrant 不确定地排序。
这个脚本直接验证：在 5 个新进程里分别取三份列表（向量前 50、BM25 前 50、融合前 40，多取 10 个看截断线附近），
比较各次结果，找出第一个不同的位置，并打印那附近的分数。

--scan：对开发集全部题目（v1 + v2）各跑 3 个进程，统计有多少题的首轮候选池不稳定。
  只看检索结果是否每次相同，不看 gold、不看回答，所以不涉及评测结论。
--scan --final：走完整的正式检索（kb_search.retrieve，R7、R8 实际用的那一个），统计前 8 名 / 前 30 名会变的题数。
  修正前跑一次（量影响面），修正后再跑一次（应为 0）。
  --split test 只在修正前跑一次：量的是"测试集当时报告的结果里，有多少题的检索本身就可能是另一个样子"，
  不读 gold、不生成回答、不算指标，不改变也不重做任何测试集结论。

用法：
    python scripts/retrieval_ties.py                 # v229 详细
    python scripts/retrieval_ties.py --qid v229 --runs 5
    python scripts/retrieval_ties.py --scan
    python scripts/retrieval_ties.py --scan --final                 # 正式检索，开发集
    python scripts/retrieval_ties.py --scan --final --split test    # 正式检索，测试集（只在修正前跑一次）
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = {"v1": PROJECT_ROOT / "reports" / "rag_golden_resolved.json",
          "v2": PROJECT_ROOT / "reports" / "rag_golden_v2_resolved.json"}

# 子进程：复现 kb_search.search 第一阶段的三份列表（参数与 FINAL 相同：hdr、per_route 50、rrf_k 60、pool 30）
CHILD = r"""
import json, sys
from datetime import date
sys.path.insert(0, sys.argv[1])
from qdrant_client import models
import kb_bm25
from kb_search import _clients, as_of_filter, CACHE, FINAL
items = json.loads(sys.stdin.read())
client, embedder = _clients()
out = []
for question, as_of in items:
    flt = as_of_filter(date.fromisoformat(as_of))
    si, sv = kb_bm25.query_vector(question)
    sparse = models.SparseVector(indices=si, values=sv) if si else None
    dense = embedder.embed([question], use_cache=CACHE)[0]
    def pts(res):
        return [[p.payload["chunk_id"], round(p.score, 9)] for p in res.points]
    d = pts(client.query_points("kb_current", query=dense, using="dense", query_filter=flt, limit=50, with_payload=True))
    s = pts(client.query_points("kb_current", query=sparse, using="bm25", query_filter=flt, limit=50,
                               with_payload=True)) if sparse else []
    if sparse:
        f = pts(client.query_points("kb_current",
            prefetch=[models.Prefetch(query=dense, using="dense", filter=flt, limit=50),
                      models.Prefetch(query=sparse, using="bm25", filter=flt, limit=50)],
            query=models.RrfQuery(rrf=models.Rrf(k=FINAL["rrf_k"])), query_filter=flt, limit=40, with_payload=True))
    else:
        f = d[:40]
    out.append({"dense": d, "bm25": s, "fused": f})
print(json.dumps(out))
"""


# 子进程（--final）：走完整的正式检索（kb_search.retrieve：混合检索 + 编号路由 + 引用扩展 + 重排），取前 30
CHILD_FINAL = r"""
import json, sys, warnings
from datetime import date
sys.path.insert(0, sys.argv[1])
from kb_search import retrieve
items = json.loads(sys.stdin.read())
out = []
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    for question, as_of in items:
        hits = retrieve(question, date.fromisoformat(as_of), 30)
        out.append({"final": [[h["chunk_id"], None if h["score"] is None else round(h["score"], 9)] for h in hits]})
print(json.dumps({"out": out, "warnings": [str(x.message) for x in w]}))
"""


def run_child(items: list[list[str]], final: bool = False) -> list[dict]:
    code = CHILD_FINAL if final else CHILD
    p = subprocess.run([sys.executable, "-c", code, str(Path(__file__).parent)], input=json.dumps(items),
                       capture_output=True, text=True, encoding="utf-8")
    if p.returncode != 0:
        print(p.stderr[-3000:])
        raise SystemExit("子进程出错")
    data = json.loads(p.stdout.strip().splitlines()[-1])
    if final:
        for msg in data["warnings"]:
            print(f"  警告：{msg}")
        return data["out"]
    return data


def first_diff(lists: list[list]) -> int | None:
    n = min(len(x) for x in lists)
    for i in range(n):
        if len({x[i][0] for x in lists}) > 1:
            return i
    return None if len({len(x) for x in lists}) == 1 else n


def membership_changes(lists: list[list], cut: int) -> set[str]:
    """截断到前 cut 个之后，各次的成员集合是否相同；返回只在部分运行里出现的块。"""
    sets = [{c for c, _ in x[:cut]} for x in lists]
    return set.union(*sets) - set.intersection(*sets)


def detail(q: dict, runs: int):
    print(f"{q['id']}（调查日期 {q['as_of']}）：{q['question']}\n")
    outs = [run_child([[q["question"], q["as_of"]]])[0] for _ in range(runs)]
    for name, cut in (("dense", 50), ("bm25", 50), ("fused", 30)):
        lists = [o[name] for o in outs]
        i = first_diff(lists)
        moved = membership_changes(lists, cut)
        label = {"dense": "向量前 50", "bm25": "BM25 前 50", "fused": "RRF 融合（候选池取前 30）"}[name]
        if i is None:
            print(f"[{label}] {runs} 次完全相同")
            continue
        print(f"[{label}] 从第 {i + 1} 名开始出现不同；截断到前 {cut} 后，成员有变化的块：{sorted(moved) or '无（只是顺序不同）'}")
        lo, hi = max(0, i - 2), min(i + 6, max(len(x) for x in lists))
        for r, x in enumerate(lists, 1):
            row = "  ".join(f"{j + 1}:{c.split('::')[-1]}({s:.6f})" for j, (c, s) in enumerate(x[lo:hi], lo))
            print(f"   第 {r} 次  {row}")
        for j in range(lo, hi):
            scores = {round(x[j][1], 9) for x in lists if j < len(x)}
            names = {x[j][0] for x in lists if j < len(x)}
            if len(names) > 1 and len(scores) == 1:
                print(f"   → 第 {j + 1} 名：不同的块、完全相同的分数 {scores.pop():.9f}（同分，顺序由 Qdrant 决定）")
        print()


def load_questions(split: str) -> list[dict]:
    qs = []
    for ver, path in GOLDEN.items():
        if path.exists():
            qs += [dict(q, ver=ver) for q in json.loads(path.read_text(encoding="utf-8")) if q["split"] == split]
    return qs


def scan_final(runs: int, split: str):
    """正式检索（与 R7、R8 实际使用的完全相同）是否可复现：前 8 名（回答 / Agent 首轮实际看到的）和前 30 名。"""
    qs = load_questions(split)
    items = [[q["question"], q["as_of"]] for q in qs]
    print(f"{'开发集' if split == 'dev' else '测试集'} v1 + v2 共 {len(qs)} 题，正式检索各跑 {runs} 个进程"
          f"（只比较各次结果是否相同，不读 gold、不生成回答、不计算任何指标）")
    outs = [run_child(items, final=True) for _ in range(runs)]
    rows = []
    for i, q in enumerate(qs):
        lists = [[c for c, _ in o[i]["final"]] for o in outs]
        top8_set = len({frozenset(x[:8]) for x in lists}) > 1
        top8_order = len({tuple(x[:8]) for x in lists}) > 1
        top30_set = len({frozenset(x[:30]) for x in lists}) > 1
        if top8_set or top8_order or top30_set:
            moved = sorted(set().union(*[set(x[:8]) for x in lists]) - set.intersection(*[set(x[:8]) for x in lists]))
            rows.append({"id": q["id"], "ver": q["ver"], "as_of": q["as_of"], "top8_set": top8_set,
                         "top8_order": top8_order, "top30_set": top30_set,
                         "top8_moved": [c.split("::")[-1] for c in moved]})
    n8 = sum(r["top8_set"] for r in rows)
    print(f"\n  前 8 名成员会变：{n8} / {len(qs)} 题；只是前 8 名顺序会变：{sum(r['top8_order'] and not r['top8_set'] for r in rows)} 题；"
          f"前 30 名成员会变：{sum(r['top30_set'] for r in rows)} 题")
    for r in rows:
        print(f"    {r['ver']} {r['id']}（{r['as_of']}）  "
              + ("前 8 名成员变：" + str(r["top8_moved"]) if r["top8_set"] else
                 "前 8 名顺序变" if r["top8_order"] else "前 30 名成员变"))
    out = PROJECT_ROOT / "reports" / f"retrieval_stability_{split}.json"
    out.write_text(json.dumps({"runs": runs, "n": len(qs), "unstable": rows}, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"  结果写到 {out.relative_to(PROJECT_ROOT)}")


def scan(runs: int):
    qs = load_questions("dev")
    items = [[q["question"], q["as_of"]] for q in qs]
    print(f"开发集 v1 + v2 共 {len(qs)} 题，每题 {runs} 个进程（只看检索结果是否稳定，不看 gold 和回答）")
    outs = [run_child(items) for _ in range(runs)]
    unstable = []
    for i, q in enumerate(qs):
        res = {}
        for name, cut in (("dense", 50), ("bm25", 50), ("fused", 30)):
            moved = membership_changes([o[i][name] for o in outs], cut)
            if moved:
                res[name] = sorted(c.split("::")[-1] for c in moved)
        if res:
            unstable.append((q, res))
    print(f"\n候选池成员不稳定的题：{len(unstable)} / {len(qs)}")
    for q, res in unstable:
        print(f"  {q['ver']} {q['id']}（{q['as_of']}）  " + "；".join(f"{k}: {v}" for k, v in res.items()))
    out = PROJECT_ROOT / "reports" / "retrieval_ties_scan.json"
    out.write_text(json.dumps([{"id": q["id"], "ver": q["ver"], "as_of": q["as_of"], **res} for q, res in unstable],
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"结果写到 {out.relative_to(PROJECT_ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default="v229")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--final", action="store_true", help="与 --scan 一起用：检查完整的正式检索是否可复现")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    a = ap.parse_args()
    if a.scan and a.final:
        scan_final(min(a.runs, 3), a.split)
        return
    if a.scan:
        scan(min(a.runs, 3))
        return
    q = next(x for x in json.loads(GOLDEN["v2"].read_text(encoding="utf-8")) if x["id"] == a.qid)
    if q["split"] != "dev":
        raise SystemExit("只诊断开发集题目")
    detail(q, a.runs)


if __name__ == "__main__":
    main()