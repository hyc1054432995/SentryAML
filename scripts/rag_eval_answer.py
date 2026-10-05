"""
SentryAML - RAG 层 R7：带引用回答的评测
========================================
用 R4 的评测集评测"回答"而不是"检索"。所有指标都是确定性的，不用大模型当裁判：

有答案的题：
  误拒率        该答却拒答的比例
  要点覆盖率    answer_keys（逐字取自 gold 原文的要点）有多少出现在回答或通过核验的引文里
  版本陷阱      must_not（答成另一版本时才会出现的说法）出现在回答或引文里的题数
  引用准确率    通过核验的论断 / 全部论断（第一次生成、重写后各算一次，看重写的作用）
  引用命中率    至少有一条通过核验的论断引用了 gold 或 also_ok 条款的题的比例
  gold 覆盖率   gold 条款中被引用到的比例（多跳题需要全部引到）
无答案的题：
  拒答率        以及拒答原因（超出范围 / 资料不足 / 引用无法核实）

要点算"覆盖"的标准：整句逐字出现，或者要点里 80% 以上的实词（去掉 the、should、was 等功能词）
  出现在回答 + 引文里。
  只认逐字会冤枉合理的引法：要点是 "should not exceed 5 business days after suspicion was first established"，
  模型引了 "5 business days after suspicion was first established"，意思完全覆盖，逐字匹配却会判为没覆盖。
  两种口径都输出：逐字（严格）和词级（主指标）。
为什么要点覆盖率在"回答 + 引文"里找：answer_keys 是英文原文的逐字片段；中文问题的回答是中文，
只能在引文里找到；英文回答也可能改写措辞，引文是逐字的，判断最可靠。

两种配置：
  full       完整流程（范围检查 + 生成 + 核验）
  no_scope   去掉范围检查，量化第 1 步的价值。生成部分与 full 的提示词完全相同，命中本地缓存，几乎不花钱

用法：
    python scripts/rag_eval_answer.py                                   # 开发集，两种配置
    python scripts/rag_eval_answer.py --split test --config full --final  # 最终配置在测试集上跑一次
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np

import llm
from kb_answer import answer, norm
from kb_bm25 import STOPWORDS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOLVED = PROJECT_ROOT / "reports" / "rag_golden_resolved.json"
CONFIGS = {"full": dict(use_scope=True), "no_scope": dict(use_scope=False)}


JUDGED = {"out_of_scope", "insufficient"}       # 由判断得出的拒答原因；其余为兜底


def words(s: str) -> list[str]:
    """实词（去掉 the、should、was 这类功能词），用于词级覆盖判断。"""
    return [w for w in re.findall(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*", norm(s)) if w not in STOPWORDS]


def key_hit(key: str, pool: str, pool_words: set[str], strict: bool = False) -> bool:
    if norm(key) in pool:
        return True
    if strict:
        return False
    kw = words(key)
    return bool(kw) and sum(w in pool_words for w in kw) / len(kw) >= 0.8


def score(q: dict, rec: dict) -> dict:
    abstained = rec["status"] == "abstained"
    verified = [c for c in rec.get("claims", []) if c["verified"]]
    out = {"status": rec["status"], "reason": rec.get("abstain_reason"), "abstained": abstained,
           "n_claims": len(rec.get("claims", [])), "n_verified": len(verified),
           "first_pass": rec.get("first_pass"), "cost": rec["cost_usd"], "ms": rec["ms"].get("total"),
           "calls": rec["usage"]["calls"], "cached_calls": rec["usage"]["cached_calls"]}
    if not q["answerable"]:
        return out
    pool = norm(rec.get("answer", "")) + " " + " ".join(norm(c["quote"]) for c in verified)
    keys = q["answer_keys"]
    cited = {(c["doc_id"], c["unit"]) for c in verified}
    gold = {(g["doc_id"], g["unit_id"]) for g in q["gold"]}
    ok = gold | {(g["doc_id"], g["unit_id"]) for g in q["also_ok"]}
    pw = set(words(pool))
    src_units = {(sr["doc_id"], u) for sr in rec.get("sources", []) for u in sr["units"]}
    out |= {
        "key_cov": 0.0 if abstained else float(np.mean([key_hit(k, pool, pw) for k in keys])),
        "key_exact": 0.0 if abstained else float(np.mean([key_hit(k, pool, pw, strict=True) for k in keys])),
        "keys_missed": [] if abstained else [k for k in keys if not key_hit(k, pool, pw)],
        "must_not_hit": [m for m in q["must_not"] if norm(m) in pool],
        "cite_ok": bool(cited & ok),
        "gold_cov": len(cited & gold) / len(gold),
        # 误拒细分（测验 R7-1 的建议）：gold 条款根本不在来源里 → 检索的问题；在来源里仍拒答 → 生成的问题
        "gold_in_sources": bool(gold & src_units),
    }
    return out


def bootstrap_ci(x, n=4000, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser(description="带引用回答的评测")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--config", choices=list(CONFIGS))
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="不读大模型缓存（真实调用，用于测延迟和费用）")
    ap.add_argument("--workers", type=int, default=4, help="同时处理几道题（每道题约 10–20 秒，大部分时间在等 API）")
    args = ap.parse_args()

    qs = [q for q in json.loads(RESOLVED.read_text(encoding="utf-8")) if q["split"] == args.split]
    out_path = PROJECT_ROOT / "reports" / f"r7_answer_{args.split}.json"
    configs = CONFIGS
    if args.split == "test":
        if not (args.config and args.final):
            raise SystemExit("测试集只对开发集上选定的最终配置跑一次：需要 --config <名称> --final")
        if out_path.exists():
            raise SystemExit(f"{out_path.name} 已存在：测试集已经跑过，不再重跑")
        configs = {args.config: CONFIGS[args.config]}
    elif args.config:
        configs = {args.config: CONFIGS[args.config]}

    ans_q = [q for q in qs if q["answerable"]]
    una_q = [q for q in qs if not q["answerable"]]
    print(f"{args.split} 集：有答案 {len(ans_q)} 题，无答案 {len(una_q)} 题；模型 {llm.MODEL}\n")

    results, records, errors, t0 = {}, {}, {}, time.perf_counter()
    from concurrent.futures import ThreadPoolExecutor, as_completed
    for name, params in configs.items():
        results[name], records[name] = {}, {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(answer, q["question"], date.fromisoformat(q["as_of"]),
                                use_cache=not args.no_cache, **params): q for q in qs}
            for i, fut in enumerate(as_completed(futs), 1):
                q = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:                       # 单题出错不中断整批；已完成的题都在缓存里，重跑几乎零成本
                    errors.setdefault(name, []).append((q["id"], f"{type(e).__name__}: {e}"))
                    continue
                results[name][q["id"]] = score(q, rec)
                records[name][q["id"]] = rec
                if i % 10 == 0:
                    print(f"  [{name}] {i}/{len(qs)}，{time.perf_counter() - t0:.0f}s")

    if errors:
        print("\n  出错的题（不计入指标；修复后重跑，已完成的题命中缓存）：")
        for name, errs in errors.items():
            for qid, msg in errs:
                print(f"    [{name}] {qid}：{msg[:120]}")
        ans_q = [q for q in ans_q if all(q["id"] in results[n] for n in configs)]
        una_q = [q for q in una_q if all(q["id"] in results[n] for n in configs)]
        qs = ans_q + una_q

    # ---- 总表
    print(f"\n  {'配置':<10}{'误拒':>6}{'要点覆盖':>9}{'(逐字)':>8}{'版本陷阱':>9}{'引用准确':>9}{'首次准确':>9}"
          f"{'引用命中':>9}{'gold覆盖':>9}{'无答案拒答':>11}{'费用$':>8}{'中位秒':>8}")
    summary = {}
    for name in configs:
        r = results[name]
        A = [r[q["id"]] for q in ans_q]
        U = [r[q["id"]] for q in una_q]
        claims = sum(x["n_claims"] for x in A)
        fp = [x["first_pass"] for x in A if x.get("first_pass")]
        s = {
            "false_abstain": float(np.mean([x["abstained"] for x in A])),
            "key_cov": float(np.mean([x["key_cov"] for x in A])),
            "key_cov_ci": bootstrap_ci([x["key_cov"] for x in A]),
            "key_exact": float(np.mean([x["key_exact"] for x in A])),
            "must_not": sum(bool(x["must_not_hit"]) for x in A),
            "cite_precision": sum(x["n_verified"] for x in A) / claims if claims else None,
            "first_precision": (sum(f["verified"] for f in fp) / sum(f["claims"] for f in fp)
                                if fp and sum(f["claims"] for f in fp) else None),
            "cite_ok": float(np.mean([x["cite_ok"] for x in A if not x["abstained"]] or [0])),
            "gold_cov": float(np.mean([x["gold_cov"] for x in A])),
            "abstain_unanswerable": float(np.mean([x["abstained"] for x in U])) if U else None,
            # 测验 R7 收尾第 1 题的建议：无答案题的拒答按原因拆开。
            #   判断拒答：范围检查拦下（out_of_scope）或模型判断资料不足（insufficient），是系统"看出来答不了"；
            #   兜底拒答：生成失败（generation_error）或没有一条引文通过核验（unverifiable），是系统"出错时不乱答"。
            # 只有前者能证明系统识别得了无答案的问题
            "abstain_judged": sum(x["reason"] in JUDGED for x in U),
            "abstain_fallback": sum(x["abstained"] and x["reason"] not in JUDGED for x in U),
            "cost": sum(x["cost"] for x in A + U),
            "median_s": float(np.median([x["ms"] for x in A + U])) / 1000,
        }
        summary[name] = s
        f = lambda v, p=3: "—" if v is None else f"{v:.{p}f}"
        print(f"  {name:<10}{f(s['false_abstain']):>6}{f(s['key_cov']):>9}{f(s['key_exact']):>8}{s['must_not']:>9}"
              f"{f(s['cite_precision']):>9}{f(s['first_precision']):>9}{f(s['cite_ok']):>9}{f(s['gold_cov']):>9}"
              f"{f(s['abstain_unanswerable']):>11}{s['cost']:>8.3f}{s['median_s']:>8.1f}")
    for name, s in summary.items():
        lo, hi = s["key_cov_ci"]
        print(f"    {name} 要点覆盖率 95% 区间 [{lo:.3f}, {hi:.3f}]；无答案 {len(una_q)} 题：判断拒答 "
              f"{s['abstain_judged']}，兜底拒答 {s['abstain_fallback']}（生成失败或引文全部未通过，不算识别能力）")
    print("  说明：费用只计实际调用 API 的部分；命中本地缓存的调用不计费，也几乎不耗时。")

    main_cfg = "full" if "full" in configs else next(iter(configs))
    r = results[main_cfg]

    # ---- 无答案题逐题
    print(f"\n  无答案题（{main_cfg}{' vs no_scope' if 'no_scope' in configs and main_cfg != 'no_scope' else ''}）：")
    for q in una_q:
        line = f"    {q['id']} {r[q['id']]['status']:<10}{str(r[q['id']]['reason']):<16}"
        if "no_scope" in configs and main_cfg != "no_scope":
            x = results["no_scope"][q["id"]]
            line += f"| 无范围检查：{x['status']:<10}{str(x['reason']):<16}"
        print(line + f"  {q['question'][:50]}")

    # ---- 有答案题被误拒（范围检查是否误伤）
    fa = [q for q in ans_q if r[q["id"]]["abstained"]]
    if fa:
        print(f"\n  有答案却被拒答（{len(fa)} 题）：")
        for q in fa:
            rec = records[main_cfg][q["id"]]
            where = "检索没找到" if not r[q["id"]].get("gold_in_sources") else "来源里有，仍拒答"
            print(f"    {q['id']} [{q['type']}/{q['lang']}] {where}，原因 {rec['abstain_reason']}："
                  f"{(rec.get('missing') or '')[:80]}")

    # ---- 按题型
    print(f"\n  按题型（{main_cfg}）：")
    by = defaultdict(list)
    for q in ans_q:
        by[q["type"]].append(r[q["id"]])
    for t, xs in sorted(by.items()):
        print(f"    {t:<12}{len(xs):>3} 题  要点覆盖 {np.mean([x['key_cov'] for x in xs]):.3f}  "
              f"gold 覆盖 {np.mean([x['gold_cov'] for x in xs]):.3f}  拒答 {sum(x['abstained'] for x in xs)}")

    # ---- 核验情况
    st = Counter(c["status"] for q in qs for c in records[main_cfg][q["id"]].get("claims", []))
    print(f"\n  论断核验结果（{main_cfg}，重写之后）：" + "，".join(f"{k} {v}" for k, v in st.most_common()))
    print(f"  触发引文重写的题：{sum(bool(records[main_cfg][q['id']].get('retried')) for q in qs)}；"
          f"触发格式重写的题：{sum(bool(records[main_cfg][q['id']].get('schema_retry')) for q in qs)}；"
          f"\"不能回答\"却给了回答的题：{sum(bool(records[main_cfg][q['id']].get('answerable_conflict')) for q in qs)}")

    # ---- 要点没覆盖全的题
    miss = [q for q in ans_q if not r[q["id"]]["abstained"] and r[q["id"]]["key_cov"] < 1]
    if miss:
        print(f"\n  回答了但要点没覆盖全（{len(miss)} 题）：")
        for q in miss:
            rec = records[main_cfg][q["id"]]
            print(f"    {q['id']} [{q['type']}/{q['lang']}] 缺：{'；'.join(k[:50] for k in r[q['id']]['keys_missed'])}")
            print(f"         引用：{'，'.join(sorted({c['chunk_id'] + ':' + c['unit'] for c in rec['claims'] if c['verified']}))[:150]}")
    vt = [q for q in ans_q if r[q["id"]].get("must_not_hit")]
    for q in vt:
        print(f"    ⚠ 版本陷阱 {q['id']}：出现了 {r[q['id']]['must_not_hit']}")

    if errors and args.split == "test":
        raise SystemExit("测试集有题出错，不写结果文件（这样修复后还能重跑，且没有任何结果被用于选择）")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps({"split": args.split, "summary": summary, "scores": results,
                                    "records": records}, ensure_ascii=False, indent=1, default=str),
                        encoding="utf-8")
    print(f"\n用时 {time.perf_counter() - t0:.0f}s；完整回答记录已写到 {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()