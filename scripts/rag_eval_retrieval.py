"""
SentryAML - RAG 层 R6a/R6b：检索评测与消融阶梯
================================================
用 R4 的评测集，逐级量化每一项检索技术的贡献。规则与 ML 层相同（D27、D28）：
  - 一次只改一个变量：阶梯里每一行只和它的"对照行"相差一处
  - 先量噪声：开发集只有约 49 道有答案的题，用配对 bootstrap 给出差值的 95% 置信区间，
    区间跨过 0 的提升不算数
  - 只在开发集上做选择；测试集只对最终选定的那一行跑一次（--split test 需要 --final）

指标（只对有答案的题计算）：
  Recall@10  严格：每条 gold 条款是否出现在前 10 名，取平均（多条 gold 的题要全部找到才是 1）
  Hit@1/@5   宽松：前 1 / 前 5 名里有没有任何可用答案（gold 或 also_ok）
  MRR        宽松：第一个可用答案名次的倒数（第 1 名 = 1，第 2 名 = 0.5，没进前 10 = 0）
为什么既有严格又有宽松：Recall 衡量"回答所需的条款是否都拿到了"，Hit/MRR 衡量"最先看到的是不是能用的"。
also_ok 只算进宽松指标，见 R4 测验第 1 题。

用法：
    python scripts/rag_eval_retrieval.py                         # 开发集，全部阶梯
    python scripts/rag_eval_retrieval.py --only E I              # 只跑其中几行
    python scripts/rag_eval_retrieval.py --split test --config I --final   # 最终配置在测试集上跑一次
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

import kb_search
from kb_search import search

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOLVED = PROJECT_ROOT / "reports" / "rag_golden_resolved.json"
K = 10

# 行号, 名称, 检索参数, 对照行（与对照行只差一个变量）
LADDER = [
    ("A", "raw · BM25",                 dict(variant="raw", mode="bm25"),                         None),
    ("B", "raw · dense",                dict(variant="raw", mode="dense"),                        None),
    ("C", "raw · hybrid",               dict(variant="raw", mode="hybrid"),                       "B"),
    ("D", "hdr · hybrid（+抬头）",       dict(variant="hdr", mode="hybrid"),                       "C"),
    ("E", "ctx · hybrid（+说明）",       dict(variant="ctx", mode="hybrid"),                       "D"),
    ("F", "ctx · dense",                dict(variant="ctx", mode="dense"),                        "B"),
    ("G", "ctx · BM25",                 dict(variant="ctx", mode="bm25"),                         "A"),
    ("H", "ctx · hybrid, RRF k=2",      dict(variant="ctx", mode="hybrid", rrf_k=2),              "E"),
    ("I", "ctx · hybrid + 编号路由",     dict(variant="ctx", mode="hybrid", exact=True),           "E"),
    ("J", "I 去掉日期过滤",              dict(variant="ctx", mode="hybrid", exact=True, use_filter=False), "I"),
    # R6b：两阶段
    ("K", "I + 重排（候选 30）",          dict(variant="ctx", mode="hybrid", exact=True, rerank=True),        "I"),
    ("L", "K + 引用扩展",                dict(variant="ctx", mode="hybrid", exact=True, rerank=True,
                                             expand_graph=True),                                     "K"),
    ("M", "L 换成 hdr（不要说明）",       dict(variant="hdr", mode="hybrid", exact=True, rerank=True,
                                             expand_graph=True),                                     "L"),
]


def in_force(h: dict, as_of: str) -> bool:
    return h["effective_from"][:10] <= as_of < h["effective_to"][:10]


def score_question(q: dict, hits: list[dict]) -> dict:
    ids = [h["chunk_id"] for h in hits]
    groups = [set(g) for g in q["gold_chunks"]]
    ok_units = {(u["doc_id"], u["unit_id"]) for u in q["gold"] + q["also_ok"]}
    usable = [i for i, h in enumerate(hits)
              if any(h["chunk_id"] in g for g in groups)
              or any((h["doc_id"], u) in ok_units for u in h["unit_ids"])]
    first = usable[0] + 1 if usable else None
    return {
        "recall": float(np.mean([any(c in g for c in ids[:K]) for g in groups])),
        "hit1": float(first == 1), "hit5": float(first is not None and first <= 5),
        "mrr": 1.0 / first if first else 0.0,
        "first_rank": first,
        "expired": sum(not in_force(h, q["as_of"]) for h in hits),
        "top": ids[:5],
        "top_score": next((h["score"] for h in hits if h["score"] is not None and h["route"] != "exact"), None),
    }


def bootstrap_ci(x: np.ndarray, n: int = 4000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def run(row, qs):
    _, _, params, _ = row
    out = {}
    for q in qs:
        t = time.perf_counter()
        hits = search(q["question"], date.fromisoformat(q["as_of"]), k=K, **params)
        ms = (time.perf_counter() - t) * 1000
        out[q["id"]] = (score_question(q, hits) if q["answerable"] else {
            "top": [h["chunk_id"] for h in hits[:5]],
            "top_score": next((h["score"] for h in hits if h["score"] is not None and h["route"] != "exact"), None)}
        ) | {"ms": ms}
    return out


def main():
    ap = argparse.ArgumentParser(description="检索评测与消融阶梯")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--only", nargs="+", help="只跑这几行（行号）")
    ap.add_argument("--config", help="测试集上要跑的那一行（必须是开发集上选定的最终配置）")
    ap.add_argument("--final", action="store_true", help="确认这是最终配置，测试集只跑这一次")
    ap.add_argument("--cold", action="store_true",
                    help="不读向量和重排缓存（会真实调用 API），用来测真实延迟；结果不受影响")
    args = ap.parse_args()

    kb_search.CACHE = not args.cold
    qs_all = json.loads(RESOLVED.read_text(encoding="utf-8"))
    qs = [q for q in qs_all if q["split"] == args.split]
    ans = [q for q in qs if q["answerable"]]
    rows = LADDER
    out_path = PROJECT_ROOT / "reports" / f"r6_retrieval_{args.split}{'_cold' if args.cold else ''}.json"
    if args.split == "test":
        if not (args.config and args.final):
            raise SystemExit("测试集只对开发集上选定的最终配置跑一次：需要 --config <行号> --final")
        if out_path.exists():
            raise SystemExit(f"{out_path.name} 已存在：测试集已经跑过，不再重跑")
        rows = [r for r in LADDER if r[0] == args.config]
    elif args.only:
        rows = [r for r in LADDER if r[0] in args.only]
    print(f"{args.split} 集：{len(qs)} 题，其中有答案 {len(ans)} 题，无答案 {len(qs) - len(ans)} 题；取前 {K} 名\n")

    results, t0 = {}, time.perf_counter()
    for row in rows:
        results[row[0]] = run(row, qs)

    # ---- 总表
    print(f"  {'行':<3}{'配置':<24}{'Recall@10':>10}{'Hit@1':>8}{'Hit@5':>8}{'MRR':>7}{'毫秒/题':>9}   {'Recall@10 的 95% 区间':<18}")
    summary = {}
    for code, name, _, _ in rows:
        r = results[code]
        m = {k: np.array([r[q["id"]][k] for q in ans]) for k in ("recall", "hit1", "hit5", "mrr")}
        lo, hi = bootstrap_ci(m["recall"])
        ms = float(np.median([r[q["id"]]["ms"] for q in qs]))
        summary[code] = {k: float(v.mean()) for k, v in m.items()} | {"recall_ci": [lo, hi], "name": name, "ms": ms}
        print(f"  {code:<3}{name:<24}{m['recall'].mean():>10.3f}{m['hit1'].mean():>8.3f}{m['hit5'].mean():>8.3f}"
              f"{m['mrr'].mean():>7.3f}{ms:>9.0f}   [{lo:.3f}, {hi:.3f}]")

    # ---- 逐级对比：配对 bootstrap
    pairs = [(c, ref) for c, _, _, ref in rows if ref and ref in results]
    if pairs:
        print("\n  逐级对比（同一批题配对重抽样；区间不含 0 才算真实差异）：")
        for c, ref in pairs:
            for metric in ("recall", "mrr"):
                d = np.array([results[c][q["id"]][metric] - results[ref][q["id"]][metric] for q in ans])
                lo, hi = bootstrap_ci(d)
                verdict = "提升" if lo > 0 else "下降" if hi < 0 else "不显著"
                print(f"    {c} vs {ref}  {metric:<7} 差 {d.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  {verdict}"
                      f"（变好 {int((d > 0).sum())} 题，变差 {int((d < 0).sum())} 题）")

    # ---- 日期过滤的价值
    if "J" in results:
        exp = np.mean([results["J"][q["id"]]["expired"] for q in ans])
        print(f"\n  去掉日期过滤（J）：前 10 名里平均有 {exp:.1f} 个过期版本的块；有过滤时恒为 0")

    # 细分分析看哪一行：测试集就是唯一那一行；开发集默认看完整配置 I（没跑 I 时看 Recall 最高的一行）
    best = rows[0][0] if args.split == "test" else next(
        (c for c in ("L", "K", "I") if c in results),
        max(summary, key=lambda c: (summary[c]["recall"], summary[c]["mrr"])))
    r = results[best]

    # ---- 按题型、按语言
    print(f"\n  按题型（配置 {best}）：")
    by = defaultdict(list)
    for q in ans:
        by[q["type"]].append(r[q["id"]])
    for t, xs in sorted(by.items()):
        print(f"    {t:<12} {len(xs):>3} 题  Recall@10 {np.mean([x['recall'] for x in xs]):.3f}  "
              f"MRR {np.mean([x['mrr'] for x in xs]):.3f}")
    gold_key = lambda q: (q["as_of"], tuple(sorted((g["doc_id"], g["unit_id"]) for g in q["gold"])))
    en = {gold_key(q): q for q in ans if q["lang"] == "en"}
    zh_pairs = [(q, en[gold_key(q)]) for q in ans if q["lang"] == "zh" and gold_key(q) in en]
    if zh_pairs:
        print(f"\n  中英配对（{len(zh_pairs)} 对，同一标准答案）：")
        for code in [c for c in ("B", "A", best) if c in results]:
            mz = np.mean([results[code][z["id"]]["mrr"] for z, _ in zh_pairs])
            me = np.mean([results[code][e["id"]]["mrr"] for _, e in zh_pairs])
            print(f"    {code} {summary[code]['name']:<24} 中文 MRR {mz:.3f}   英文 MRR {me:.3f}")

    # ---- 无答案题：给 R7 的拒答阈值留数据
    for code, label in (("F", "余弦相似度"), ("L", "重排分数")):
        if code not in results:
            continue
        a = [results[code][q["id"]]["top_score"] for q in ans if results[code][q["id"]]["top_score"] is not None]
        u = [results[code][q["id"]]["top_score"] for q in qs if not q["answerable"]
             and results[code][q["id"]]["top_score"] is not None]
        if a and u:
            print(f"\n  第 1 名的{label}（{code}）：有答案题中位数 {np.median(a):.3f}（最低 {min(a):.3f}），"
                  f"无答案题中位数 {np.median(u):.3f}（最高 {max(u):.3f}）"
                  f"；有答案题中低于无答案题最高分的有 {sum(x <= max(u) for x in a)} 道")
            if code == "L":
                for q in qs:
                    if not q["answerable"]:
                        x = results[code][q["id"]]
                        print(f"    无答案 {q['id']} 第 1 名重排分 {x['top_score']:.3f} → {x['top'][0] if x['top'] else '—'}"
                              f"    问：{q['question'][:60]}")

    # ---- 失败清单：下一步错误分析的输入
    fails = [q for q in ans if r[q["id"]]["recall"] < 1]
    print(f"\n  配置 {best} 没有找全 gold 的题（{len(fails)} 道）：")
    for q in fails:
        x = r[q["id"]]
        gold = "；".join(f"{g['doc_id']}:{g['unit_id']}" for g in q["gold"])
        print(f"    {q['id']} [{q['type']}/{q['lang']}] 首个可用答案名次 {x['first_rank'] or '>10'}  gold {gold}")
        print(f"         问：{q['question'][:90]}")
        print(f"         前 3：{'，'.join(x['top'][:3])}")

    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"split": args.split, "k": K, "summary": summary, "per_question": results},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n用时 {time.perf_counter() - t0:.0f}s；逐题结果已写到 {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()