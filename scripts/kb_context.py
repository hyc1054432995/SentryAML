"""
SentryAML - RAG 层 R5a：检索块的上下文说明（Contextual Retrieval）
=====================================================================
问题：块被切出来以后，丢掉了它在文件里的位置和指代。例如 FATF CEF 附录 A 的一块只有一句
"Rapid or immediate, high or low value transactions after opening of an account"，
单看不知道这是"钱骡账户的风险信号"；Notice 6.23 写"the transactions in paragraph 6.21"，
单看不知道是"异常交易"。向量化和关键词检索都只能看到块本身。

做法（Anthropic, Contextual Retrieval, 2024）：把整份文件和这一块一起交给大模型，
让它写 40–80 词的说明：这块讲什么、属于文件哪部分、指代的是什么、业内还怎么称呼。
说明拼在块前面再做索引；它只用于"找到"，引用和回答永远只用原文。

两层上下文：
  1. 结构化抬头（确定性，不花钱）：文件名、版本、标题路径、条款编号 —— structural_header()
  2. 大模型说明（本脚本生成）：语义层面的位置和指代
  R6 的消融阶梯会分别量化这两层的贡献。

成本控制：
  - 整份文件放在提示词最前面、每块都完全相同 → DeepSeek 自动前缀缓存命中，重复部分按缓存价计费
  - 结果按内容哈希缓存（015_kb_context.sql）：重新入库、新版本中没变的块都不再调用
  - 先单独跑每份文件的第一块"预热"缓存，再并发；命中率过低时自动停止，避免按全价烧钱

防止大模型编造：
  - 说明里出现的数字（金额、天数、条款号）必须出现在块原文或抬头里，否则要求重写一次，仍不通过则丢弃
  - 长度 15–110 词

用法：
    python scripts/kb_context.py --dry-run          # 只估算：要生成多少块、多少 token、多少钱；写出一份样例提示词
    python scripts/kb_context.py --limit 20         # 试跑 20 块，看命中率和说明质量
    python scripts/kb_context.py                    # 全量（已缓存的跳过），最后回填 kb_chunks.context
    python scripts/kb_context.py --apply-only       # 只从缓存回填（kb_load 之后用）
    python scripts/kb_context.py --retry-rejected   # 重新生成之前被拒绝的块（覆盖旧的拒绝记录）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from db import connect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_VERSION = "ctx-v1"
DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
BASE_URL = "https://api.deepseek.com"
# 美元 / 百万 token。按 DeepSeek 官方价格页的高峰时段价格做保守估算，价格以官网为准
PRICE = {"hit": 0.006, "miss": 0.30, "out": 1.20}
CHARS_PER_TOKEN = 3.8
MIN_WORDS, MAX_WORDS = 15, 110
MAX_OUTPUT_TOKENS = 1200          # 说明本身约 150 token；其余留给可能存在的"思考"过程
EST_OUTPUT_PER_CALL = 450         # 按 ctx-v1 实测（44.4 万输出 token / 约 1,300 次调用）估算
WARMUP_CHECK_AFTER, MIN_HIT_RATIO = 8, 0.5

SYSTEM = ("You write short retrieval contexts for chunks of anti-money-laundering regulations, "
          "supervisory guidance and typology reports. Compliance analysts and an investigation agent "
          "search these chunks with plain-language questions. Your context is prepended to the chunk "
          "before indexing and is never shown as a citation.")

INSTRUCTION = """Here is a chunk from the document above:
<chunk>
{chunk}
</chunk>

Write a succinct context (40-80 words) that situates this chunk within the document to improve search retrieval. Cover:
- what the chunk is about in plain terms, and which part of the document it belongs to;
- what references in the chunk point to (e.g. "these measures", "the transactions in paragraph 6.21", "the Notice"), when the document makes that clear;
- one or two terms a practitioner might use for this topic, if they differ from the chunk's own wording.
Rules: do not repeat or paraphrase the whole chunk; do not state any number, amount, date, time limit or paragraph number unless it appears in the chunk itself; do not add facts that are not in the document.
Return JSON: {{"context": "..."}}"""

NUM_RE = re.compile(r"\d+(?:[.,:\-]\d+)*[A-Z]?")


# ---------------------------------------------------------------------------
# 确定性部分：抬头、文件全文、缓存键
# ---------------------------------------------------------------------------
def structural_header(doc: dict, chunk: dict) -> str:
    """结构化抬头：R5b 建索引时拼在块前面，不依赖大模型。"""
    path = " > ".join(chunk["heading_path"])
    body_units = [u for u in chunk["unit_ids"] if not re.fullmatch(r"fn\d+", u)]
    return f"{doc['title']} ({doc['version_label']}) | {path} | {', '.join(body_units)}"


def norm(s: str) -> str:
    s = s.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    return re.sub(r"\s+", " ", s).strip()


def cache_key(model: str, family: str, chunk_text: str) -> str:
    raw = f"{PROMPT_VERSION}|{model}|{family}|{norm(chunk_text)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def document_text(units: list[dict]) -> str:
    out = []
    for u in units:
        out.append(f"## {u['text']}" if u["kind"] == "heading" else u["text"])
    return "\n\n".join(out)


def check_context(ctx: str, chunk_text: str, header: str) -> str | None:
    """返回拒绝原因；通过则返回 None。"""
    words = len(ctx.split())
    if words < MIN_WORDS or words > MAX_WORDS:
        return f"长度 {words} 词，超出 {MIN_WORDS}–{MAX_WORDS}"
    allowed = norm(chunk_text) + " " + header
    bad = sorted({n for n in NUM_RE.findall(ctx) if n not in allowed})
    if bad:
        return f"出现原文没有的数字：{', '.join(bad)}"
    return None


# ---------------------------------------------------------------------------
# 大模型调用
# ---------------------------------------------------------------------------
def make_client():
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass
    from openai import OpenAI
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("没有找到 DEEPSEEK_API_KEY：请在项目根目录的 .env 里设置（.env 不要提交到 git）")
    return OpenAI(api_key=key, base_url=BASE_URL, timeout=120, max_retries=3)


def call_llm(client, model: str, messages: list[dict]) -> tuple[str, dict]:
    # max_tokens 必须留足：如果模型先"思考"，思考的 token 也算在输出里（ctx-v1 用 300，约三分之一被截断）
    resp = client.chat.completions.create(model=model, messages=messages, temperature=0.0,
                                          max_tokens=MAX_OUTPUT_TOKENS, response_format={"type": "json_object"})
    u = resp.usage
    details = getattr(u, "completion_tokens_details", None)
    usage = {"prompt": u.prompt_tokens, "completion": u.completion_tokens,
             "hit": getattr(u, "prompt_cache_hit_tokens", 0) or 0,
             "reasoning": (getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
             "finish": resp.choices[0].finish_reason}
    return resp.choices[0].message.content or "", usage


def parse_context(raw: str, finish: str | None) -> tuple[str, str | None]:
    """从模型输出里取出 context。容忍 ```json 代码块和前后多余文字；失败时给出具体原因。"""
    if finish == "length":
        return "", "输出被截断（finish_reason=length）"
    if not raw.strip():
        return "", "返回内容为空"
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    m = re.search(r"\{.*\}", text, re.S)
    try:
        ctx = json.loads(m.group(0) if m else text)["context"]
        if not isinstance(ctx, str):
            raise TypeError
        return norm(ctx), None
    except (json.JSONDecodeError, KeyError, TypeError):
        return "", f"JSON 解析失败：{raw[:80]!r}"


def generate_one(client, model: str, doc_prompt: str, chunk_text: str, header: str) -> dict:
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": doc_prompt + "\n\n" + INSTRUCTION.format(chunk=chunk_text)}]
    total = {"prompt": 0, "completion": 0, "hit": 0, "reasoning": 0}
    reason, finish = None, None
    for attempt in (1, 2):
        raw, usage = call_llm(client, model, messages)
        finish = usage["finish"]
        for k in total:
            total[k] += usage[k]
        ctx, reason = parse_context(raw, finish)
        if reason is None:
            reason = check_context(ctx, chunk_text, header)
        if reason is None:
            return {"context": ctx, "status": "ok", "reason": None, "attempts": attempt, "finish": finish, **total}
        # 第二次：把问题告诉模型，让它重写（文件部分不变，仍能命中缓存）
        messages += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": f"Your context was rejected: {reason}. "
                                                 "Rewrite it following all the rules. Return JSON only."}]
    return {"context": None, "status": "rejected", "reason": reason, "attempts": 2, "finish": finish, **total}


# ---------------------------------------------------------------------------
def load(only: list[str] | None):
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, family, title, version_label FROM kb_documents ORDER BY doc_id")
        docs = {d: {"doc_id": d, "family": f, "title": t, "version_label": v} for d, f, t, v in cur.fetchall()}
        cur.execute("SELECT doc_id, unit_id, kind, text FROM kb_units ORDER BY doc_id, ord")
        units = defaultdict(list)
        for d, uid, kind, text in cur.fetchall():
            units[d].append({"unit_id": uid, "kind": kind, "text": text})
        cur.execute("SELECT chunk_id, doc_id, unit_ids, heading_path, text FROM kb_chunks ORDER BY doc_id, chunk_id")
        chunks = [dict(zip(("chunk_id", "doc_id", "unit_ids", "heading_path", "text"), r)) for r in cur.fetchall()]
        cur.execute("SELECT cache_key, status FROM kb_chunk_context")
        cached = dict(cur.fetchall())
    if only:
        chunks = [c for c in chunks if c["doc_id"] in only]
    return docs, units, chunks, cached


def save(cur, key: str, model: str, family: str, chunk_id: str, r: dict) -> None:
    cur.execute(
        """INSERT INTO kb_chunk_context (cache_key, prompt_version, model, family, sample_chunk_id, context,
                                         status, reject_reason, attempts, prompt_tokens, cache_hit_tokens,
                                         completion_tokens, reasoning_tokens, finish_reason)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           ON CONFLICT (cache_key) DO UPDATE SET            -- 只会覆盖被拒绝的旧结果（见 --retry-rejected）
               context = EXCLUDED.context, status = EXCLUDED.status, reject_reason = EXCLUDED.reject_reason,
               attempts = EXCLUDED.attempts, prompt_tokens = EXCLUDED.prompt_tokens,
               cache_hit_tokens = EXCLUDED.cache_hit_tokens, completion_tokens = EXCLUDED.completion_tokens,
               reasoning_tokens = EXCLUDED.reasoning_tokens, finish_reason = EXCLUDED.finish_reason,
               created_at = now()
           WHERE kb_chunk_context.status = 'rejected'""",
        (key, PROMPT_VERSION, model, family, chunk_id, r["context"], r["status"], r["reason"],
         r["attempts"], r["prompt"], r["hit"], r["completion"], r["reasoning"], r["finish"]))


def apply_cache(model: str) -> None:
    """把缓存里的说明回填到 kb_chunks.context；没有（或被拒绝）的置为 NULL。"""
    docs, _, chunks, _ = load(None)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT cache_key, context FROM kb_chunk_context WHERE status = 'ok'")
        ctx = dict(cur.fetchall())
        rows = [(ctx.get(cache_key(model, docs[c["doc_id"]]["family"], c["text"])), c["chunk_id"]) for c in chunks]
        cur.executemany("UPDATE kb_chunks SET context = %s WHERE chunk_id = %s", rows)
        conn.commit()
    n = sum(r[0] is not None for r in rows)
    print(f"\n回填 kb_chunks.context：{n}/{len(rows)} 块有上下文说明（模型 {model}，提示词 {PROMPT_VERSION}）")


def main():
    ap = argparse.ArgumentParser(description="为检索块生成上下文说明")
    ap.add_argument("--dry-run", action="store_true", help="只估算，不调用大模型")
    ap.add_argument("--limit", type=int, help="最多生成多少块（试跑用）")
    ap.add_argument("--only", nargs="+", help="只处理这几份文件")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--apply-only", action="store_true", help="只从缓存回填 kb_chunks.context")
    ap.add_argument("--retry-rejected", action="store_true", help="重新生成之前被拒绝的块")
    args = ap.parse_args()

    if args.apply_only:
        apply_cache(args.model)
        return

    docs, units, chunks, cached = load(args.only)
    # 待生成：按缓存键去重（同一族不同版本里文字相同的块，只生成一次）
    pending, seen, n_cached, n_dup = defaultdict(list), set(), 0, 0
    for c in chunks:
        k = cache_key(args.model, docs[c["doc_id"]]["family"], c["text"])
        if k in cached and not (args.retry_rejected and cached[k] == "rejected"):
            n_cached += 1
        elif k in seen:
            n_dup += 1
        else:
            seen.add(k)
            pending[c["doc_id"]].append((k, c))
    n_pending = sum(len(v) for v in pending.values())

    # ---- 估算
    print(f"检索块 {len(chunks)}：已有缓存 {n_cached}，与其他版本文字相同可复用 {n_dup}，"
          f"待生成 {n_pending}（模型 {args.model}，提示词 {PROMPT_VERSION}）")
    est = {"hit": 0, "miss": 0, "out": 0}
    print(f"\n  {'文件':<26}{'待生成':>7}{'文件 token':>11}")
    doc_prompts = {}
    for d, items in sorted(pending.items()):
        doc_prompts[d] = f'<document title="{docs[d]["title"]}">\n{document_text(units[d])}\n</document>'
        t_doc = len(doc_prompts[d]) / CHARS_PER_TOKEN
        t_tail = sum(len(c["text"]) for _, c in items) / CHARS_PER_TOKEN + 250 * len(items)
        est["hit"] += t_doc * (len(items) - 1)
        est["miss"] += t_doc + t_tail
        est["out"] += EST_OUTPUT_PER_CALL * len(items)
        print(f"  {d:<26}{len(items):>7}{t_doc:>11,.0f}")
    cost = sum(est[k] * PRICE[k] for k in est) / 1e6
    worst = ((est["hit"] + est["miss"]) * PRICE["miss"] + est["out"] * PRICE["out"]) / 1e6
    print(f"\n  估算输入 {(est['hit'] + est['miss']) / 1e6:.1f}M token（其中可命中缓存 {est['hit'] / 1e6:.1f}M），"
          f"输出 {est['out'] / 1e6:.2f}M")
    print(f"  估算费用：缓存正常约 ${cost:.2f}；缓存完全失效约 ${worst:.2f}")

    if args.dry_run:
        if pending:
            d, items = next(iter(sorted(pending.items())))
            k, c = items[0]
            sample = PROJECT_ROOT / "reports" / "kb_context_sample_prompt.txt"
            sample.parent.mkdir(exist_ok=True)
            body = doc_prompts[d]
            sample.write_text(f"[system]\n{SYSTEM}\n\n[user]\n{body[:3000]}\n……（中间省略 {len(body) - 6000:,} 字）……\n"
                              f"{body[-3000:]}\n\n{INSTRUCTION.format(chunk=c['text'])}\n\n"
                              f"[结构化抬头]\n{structural_header(docs[d], c)}\n", encoding="utf-8")
            print(f"\n  样例提示词已写到 {sample.relative_to(PROJECT_ROOT)}")
        return
    if not pending:
        apply_cache(args.model)
        return

    # ---- 生成
    client = make_client()
    budget = args.limit or n_pending
    stats = defaultdict(int)
    t0 = time.perf_counter()
    with connect() as conn, conn.cursor() as cur:
        for d, items in sorted(pending.items()):
            if budget <= 0:
                break
            items = items[:budget]
            fam = docs[d]["family"]
            run = lambda kc: generate_one(client, args.model, doc_prompts[d], kc[1]["text"],
                                          structural_header(docs[d], kc[1]))
            d_stats = defaultdict(int)

            def record(kc, r):
                save(cur, kc[0], args.model, fam, kc[1]["chunk_id"], r)
                conn.commit()
                for key in ("prompt", "hit", "completion", "reasoning"):
                    stats[key] += r[key]
                    d_stats[key] += r[key]
                stats[r["status"]] += 1
                d_stats["n"] += 1

            # 预热：第一块单独跑，让服务端把"系统提示 + 整份文件"写进前缀缓存
            record(items[0], run(items[0]))
            time.sleep(2)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futs = {pool.submit(run, kc): kc for kc in items[1:]}
                for f in as_completed(futs):
                    record(futs[f], f.result())
                    if d_stats["n"] == WARMUP_CHECK_AFTER + 1:
                        ratio = d_stats["hit"] / max(d_stats["prompt"], 1)
                        if ratio < MIN_HIT_RATIO:
                            for other in futs:
                                other.cancel()
                            raise SystemExit(f"\n{d}：前 {d_stats['n']} 次调用缓存命中率只有 {ratio:.0%}，"
                                             f"低于 {MIN_HIT_RATIO:.0%}，为避免按全价计费已停止。已生成的结果保留在缓存表里。")
            budget -= len(items)
            hr = d_stats["hit"] / max(d_stats["prompt"], 1)
            print(f"  {d:<26} 生成 {d_stats['n']:>4} 块，缓存命中率 {hr:.0%}，累计 {time.perf_counter() - t0:.0f}s")

    miss = stats["prompt"] - stats["hit"]
    cost = (stats["hit"] * PRICE["hit"] + miss * PRICE["miss"] + stats["completion"] * PRICE["out"]) / 1e6
    print(f"\n完成：通过 {stats['ok']}，拒绝 {stats['rejected']}；输入 {stats['prompt']:,} token"
          f"（命中缓存 {stats['hit'] / max(stats['prompt'], 1):.0%}），输出 {stats['completion']:,}"
          f"（其中思考 {stats['reasoning']:,}）；约 ${cost:.2f}")
    apply_cache(args.model)

    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT reject_reason, COUNT(*) FROM kb_chunk_context WHERE status = 'rejected' "
                    "GROUP BY 1 ORDER BY 2 DESC LIMIT 5")
        rej = cur.fetchall()
        cur.execute("SELECT c.chunk_id, c.context, left(c.text, 160) FROM kb_chunks c "
                    "WHERE c.context IS NOT NULL ORDER BY random() LIMIT 3")
        samples = cur.fetchall()
    if rej:
        print("\n被拒绝的原因：" + "；".join(f"{r}（{n}）" for r, n in rej))
    print("\n随机抽查 3 块：")
    for cid, ctx, head in samples:
        print(f"\n  [{cid}]\n  说明：{ctx}\n  原文：{head}…")


if __name__ == "__main__":
    main()