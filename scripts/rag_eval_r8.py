"""
SentryAML - R8c 第 2 阶段：在评测集 v2 上比较四种做法
=====================================================
  r7            检索一次就回答（R7 原样，基线）
  two_round     固定两轮：R7 的"未覆盖"不为空时，用它再检索一次，补入最多 4 段，重新生成（规则写死）
  agent         Agent：系统发起首轮检索（与 R7 的 8 段相同），之后由模型决定是否再查、按编号取条款
  agent_noseed  Agent：没有首轮检索，第一次查询也由模型来写

指标沿用 R7（rag_eval_answer.score，全部确定性），另加三项：
  未核验引文    回答和"未覆盖"里加了引号的英文原文（≥5 个词），如果在本题看过的任何条款里都找不到，就记一处。
                逐字核验只检查 claims，D62 发现 q076 把引文写进了"未覆盖"，绕过了核验
  工具调用      Agent 平均调用了几次检索 / 取条款；固定两轮有多少题进入了第二轮
  按 pattern    每种"检索一次答不好"的情形分别统计要点覆盖率

选择规则（D64，运行前写定；D65 修订）见 select()。

D65 的两处修订（dev2 第一次运行后发现，test2 运行之前写定）：
  1. 引文检查改用 kb_answer.unverified_quotes：成对取引号（修正把两段引文之间的叙述当成引文的错误），
     去掉法律文书的方括号改写（"[s]ubject"）
  2. 无答案题"答出"不再一律算失败：答出的回答由人按 D65 写定的标准判定（config/r8_adjudication.yaml），
     只有"给出了知识库里没有的具体事实"才算编造。没判定的一律按编造处理

用法：
    python scripts/rag_eval_r8.py                                        # dev2，四种做法
    python scripts/rag_eval_r8.py --split test --configs r7 agent --final  # test2：基线 + dev2 选出的做法，只跑一次
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import numpy as np

from db import connect
from kb_agent import research, two_round
import yaml

from kb_answer import answer, unverified_quotes
from rag_eval_answer import JUDGED, bootstrap_ci, score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOLVED = PROJECT_ROOT / "reports" / "rag_golden_v2_resolved.json"
CONFIGS = {
    "r7": lambda q, d, uc: answer(q, d, use_cache=uc),
    "two_round": lambda q, d, uc: two_round(q, d, use_cache=uc),
    "agent": lambda q, d, uc: research(q, d, use_cache=uc),
    "agent_noseed": lambda q, d, uc: research(q, d, use_cache=uc, seed=False),
}
ADJUDICATION = PROJECT_ROOT / "config" / "r8_adjudication.yaml"


def quote_audit(rec: dict, unit_text: dict) -> list[str]:
    """回答和"未覆盖"里带引号的英文原文片段，必须逐字出现在本题看过的条款里。"""
    texts = [unit_text.get((s["doc_id"], u), "") for s in rec.get("sources", []) for u in s["units"]]
    return unverified_quotes({"answer": rec.get("answer") or "", "missing": rec.get("missing") or ""}, texts)


def load_adjudication(split: str) -> dict:
    """{做法: {题号: acceptable | fabricated}}，人工判定无答案题的"答出"是否属于编造（D65）。"""
    if not ADJUDICATION.exists():
        return {}
    return (yaml.safe_load(ADJUDICATION.read_text(encoding="utf-8")) or {}).get(split, {}) or {}


def summarize(qs: list[dict], res: dict, recs: dict, adj: dict) -> dict:
    A = [res[q["id"]] for q in qs if q["answerable"]]
    U = [res[q["id"]] for q in qs if not q["answerable"]]
    claims = sum(x["n_claims"] for x in A)
    R = [recs[q["id"]] for q in qs]
    return {
        "key_cov": float(np.mean([x["key_cov"] for x in A])), "key_cov_ci": bootstrap_ci([x["key_cov"] for x in A]),
        "gold_cov": float(np.mean([x["gold_cov"] for x in A])),
        "false_abstain": int(sum(x["abstained"] for x in A)),
        "cite_precision": sum(x["n_verified"] for x in A) / claims if claims else None,
        "version_traps": int(sum(bool(x["must_not_hit"]) for x in A)),
        "unans_judged": int(sum(x["abstained"] and x["reason"] in JUDGED for x in U)),
        "unans_fallback": int(sum(x["abstained"] and x["reason"] not in JUDGED for x in U)),
        "unans_answered": int(sum(not x["abstained"] for x in U)),
        # 答出的无答案题中被判为编造的（含尚未判定的）
        "unans_fabricated": int(sum(not res[q["id"]]["abstained"] and adj.get(q["id"]) != "acceptable"
                                    for q in qs if not q["answerable"])),
        "unverified_quote_qs": int(sum(bool(res[q["id"]]["unverified_quotes"]) for q in qs)),
        "conflicts": int(sum(bool(r.get("answerable_conflict")) for r in R)),
        "cost": float(sum(x["cost"] for x in res.values())),
        "median_s": float(np.median([x["ms"] for x in res.values()])) / 1000,
        "tool_calls": float(np.mean([sum((r.get("tool_calls") or {}).get(t, 0) for t in
                                         ("search_regulations", "get_clause")) for r in R])),
        "second_round": int(sum(r.get("rounds") == 2 for r in R)),
    }


def select(summary: dict) -> tuple[str, list[str]]:
    """
    D64 的选择规则（运行前写定）：
      护栏（全部满足才有资格）：无答案题被编造回答的题数不多于 r7（D65：原为"答出"）；引用准确率 ≥ 0.98；
                                 版本陷阱 0；未核验引文 0 题
      在有资格的做法里取要点覆盖率最高者；与最高者相差不超过 0.03 的，取费用最低的（r7 最便宜）
    """
    base = summary["r7"]
    notes, ok = [], []
    for name, s in summary.items():
        fails = []
        if s["unans_fabricated"] > base["unans_fabricated"]:
            fails.append("无答案题有编造回答（或尚未判定）")
        if s["cite_precision"] is not None and s["cite_precision"] < 0.98:
            fails.append("引用准确率 < 0.98")
        if s["version_traps"]:
            fails.append("版本陷阱")
        if s["unverified_quote_qs"]:
            fails.append("有未核验引文")
        notes.append(f"{name}：{'通过' if not fails else '未通过（' + '，'.join(fails) + '）'}")
        if not fails:
            ok.append(name)
    if not ok:
        return "r7", notes + ["没有做法通过护栏，保持 r7"]
    best = max(summary[n]["key_cov"] for n in ok)
    close = [n for n in ok if summary[n]["key_cov"] >= best - 0.03]
    return min(close, key=lambda n: summary[n]["cost"]), notes


def main():
    ap = argparse.ArgumentParser(description="R8c：四种做法在评测集 v2 上的比较")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--rescore", action="store_true",
                    help="不调用任何模型：读取已保存的记录，按当前的人工判定重新计分（test2 跑过之后补判定用）")
    ap.add_argument("--show", nargs="+", metavar="题号", help="打印已保存记录中这些题的完整回答和论断（供人工判定）")
    a = ap.parse_args()

    qs = [q for q in json.loads(RESOLVED.read_text(encoding="utf-8")) if q["split"] == a.split]
    out_path = PROJECT_ROOT / "reports" / f"r8_answer_{a.split}.json"
    if a.show:
        saved = json.loads(out_path.read_text(encoding="utf-8"))["records"]
        for n, recs in saved.items():
            for qid in a.show:
                rec = recs.get(qid)
                if rec is None:
                    continue
                print(f"{'=' * 90}\n[{n}] {qid} 状态 {rec['status']} {rec.get('abstain_reason') or ''}\n"
                      f"答：{rec.get('answer')}\n未覆盖：{rec.get('missing')}")
                for c in rec.get("claims", []):
                    print(f"  {'✓' if c.get('verified') else '✗'} [{c.get('source')}] {c.get('statement', '')[:120]}"
                          f"\n      引文：{c.get('quote', '')[:200]}")
        return
    if a.split == "test" and not a.rescore:
        if not a.final or "r7" not in a.configs or len(a.configs) != 2:
            raise SystemExit("test2 只跑一次：--configs r7 <dev2 选出的做法> --final")
        if out_path.exists():
            raise SystemExit(f"{out_path.name} 已存在：test2 已经跑过，不再重跑")
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, unit_id, text FROM kb_units")
        unit_text = {(d, u): t for d, u, t in cur.fetchall()}

    n_ans = sum(q["answerable"] for q in qs)
    print(f"v2 {a.split}：有答案 {n_ans} 题，无答案 {len(qs) - n_ans} 题；做法：{', '.join(a.configs)}\n")
    results, records, errors, t0 = {}, {}, [], time.perf_counter()
    if a.rescore:
        saved = json.loads(out_path.read_text(encoding="utf-8"))
        if not saved.get("complete", True):
            raise SystemExit("保存的结果不完整，不能重新计分")
        by_id = {q["id"]: q for q in qs}
        for name, recs in saved["records"].items():
            results[name] = {}
            records[name] = recs
            for qid, rec in recs.items():
                results[name][qid] = score(by_id[qid], rec) | {"unverified_quotes": quote_audit(rec, unit_text)}
        a.configs = list(saved["records"])
    for name in ([] if a.rescore else a.configs):
        results[name], records[name] = {}, {}
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futs = {pool.submit(CONFIGS[name], q["question"], date.fromisoformat(q["as_of"]), not a.no_cache): q
                    for q in qs}
            for fut in as_completed(futs):
                q = futs[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    errors.append(f"[{name}] {q['id']}：{type(e).__name__}: {e}"[:200])
                    continue
                s = score(q, rec)
                s["unverified_quotes"] = quote_audit(rec, unit_text)
                results[name][q["id"]], records[name][q["id"]] = s, rec
        print(f"  [{name}] 完成，{time.perf_counter() - t0:.0f}s")
    n_all = len(qs)
    if errors:
        print("\n  出错的题：\n    " + "\n    ".join(errors))
        if a.split == "test":
            raise SystemExit("test2 有题出错，不写结果文件；修复后重跑")
        # 只在全部做法都成功的题上计分，便于查看；但这是不完整的子集，不能用来做选择（见文件末尾）
        qs = [q for q in qs if all(q["id"] in results[n] for n in a.configs)]
        print(f"\n  ⚠ 以下指标只基于 {len(qs)}/{n_all} 道全部做法都成功的题，仅供排查，不用于选择")

    adj = load_adjudication(a.split)
    summary = {n: summarize(qs, results[n], records[n], adj.get(n, {})) for n in a.configs}
    f = lambda v, p=3: "—" if v is None else f"{v:.{p}f}"
    print(f"\n  {'做法':<13}{'要点覆盖':>8}{'gold覆盖':>9}{'误拒':>5}{'引用准确':>9}{'版本陷阱':>9}"
          f"{'无答案:判断/兜底/答出(编造)':>21}{'未核验引文':>10}{'工具/题':>8}{'费用$':>8}{'中位秒':>7}")
    for n, s in summary.items():
        print(f"  {n:<13}{f(s['key_cov']):>8}{f(s['gold_cov']):>9}{s['false_abstain']:>5}{f(s['cite_precision']):>9}"
              f"{s['version_traps']:>9}{s['unans_judged']:>10}/{s['unans_fallback']}/{s['unans_answered']}"
              f"({s['unans_fabricated']})"
              f"{s['unverified_quote_qs']:>10}{s['tool_calls']:>8.1f}{s['cost']:>8.3f}{s['median_s']:>7.1f}")
    for n, s in summary.items():
        lo, hi = s["key_cov_ci"]
        extra = f"；进入第二轮 {s['second_round']} 题" if n == "two_round" else ""
        print(f"    {n} 要点覆盖 95% 区间 [{lo:.3f}, {hi:.3f}]；自相矛盾标记 {s['conflicts']}{extra}")

    # ---- 与 r7 配对比较
    ans = [q for q in qs if q["answerable"]]
    if "r7" in summary and len(summary) > 1:
        print("\n  与 r7 配对比较（同一批题；区间不含 0 才算真实差异）：")
        rng = np.random.default_rng(0)
        for n in summary:
            if n == "r7":
                continue
            d = np.array([results[n][q["id"]]["key_cov"] - results["r7"][q["id"]]["key_cov"] for q in ans])
            boots = d[rng.integers(0, len(d), size=(4000, len(d)))].mean(axis=1)
            better = [q["id"] for q, x in zip(ans, d) if x > 0]
            worse = [q["id"] for q, x in zip(ans, d) if x < 0]
            print(f"    {n:<13} 差 {d.mean():+.3f} [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]"
                  f"  变好 {better}  变差 {worse}")

    # ---- 按 pattern
    pats = sorted({q["pattern"] for q in ans})
    print(f"\n  要点覆盖率按 pattern：{'':<4}" + "".join(f"{n:>14}" for n in summary))
    for p in pats:
        ids = [q["id"] for q in ans if q["pattern"] == p]
        print(f"    {p:<12}{len(ids):>3} 题" + "".join(
            f"{np.mean([results[n][i]['key_cov'] for i in ids]):>14.3f}" for n in summary))

    # ---- 逐题：要点没覆盖全 / 无答案题 / 未核验引文
    print("\n  逐题（要点覆盖；无答案题显示状态）：")
    for q in qs:
        cells = []
        for n in summary:
            x = results[n][q["id"]]
            cells.append(("拒答" if x["abstained"] else f"{x['key_cov']:.2f}") if q["answerable"]
                         else (f"拒:{x['reason']}" if x["abstained"] else "答出"))
        print(f"    {q['id']} [{q['pattern']:<12}] " + "  ".join(f"{n}={c}" for n, c in zip(summary, cells)))
    for n in summary:
        for q in qs:
            for b in results[n][q["id"]]["unverified_quotes"]:
                print(f"    ⚠ 未核验引文 [{n}] {q['id']} {b}")

    # ---- 答出的无答案题：打印回答，供人工判定（D65 的判定标准写在 decisions.md）
    pending = []
    for q in qs:
        if q["answerable"]:
            continue
        for n in summary:
            if results[n][q["id"]]["abstained"]:
                continue
            verdict = adj.get(n, {}).get(q["id"])
            rec = records[n][q["id"]]
            print(f"\n  【无答案题被答出】[{n}] {q['id']} 判定：{verdict or '未判定'}\n    问：{q['question']}"
                  f"\n    答：{(rec.get('answer') or '')[:700]}\n    未覆盖：{(rec.get('missing') or '')[:300]}")
            if not verdict:
                pending.append(f"{n}.{q['id']}")
    if pending:
        print(f"\n  待判定：{', '.join(pending)} → 写入 config/r8_adjudication.yaml 后重跑（全部命中缓存）")

    if a.split == "dev" and errors:
        # D66：第一次修订后的重跑因 API 余额不足，agent_noseed 有 9 题出错；脚本当时仍在剩下的 9 题上做了选择，
        # 而剩下的题恰好不含任何无答案题，护栏形同虚设。不完整的结果一律不做选择
        print(f"\n  ✗ 有 {len(errors)} 题出错，结果不完整：不执行 D64 选择规则。修复原因后重跑（已完成的部分命中缓存）")
    elif a.split == "dev":
        chosen, notes = select(summary)
        print("\n  D64 选择规则：\n    " + "\n    ".join(notes) + f"\n    → 选定：{chosen}")
    out_path.write_text(json.dumps({"split": a.split, "complete": not errors, "errors": errors, "summary": summary,
                                    "scores": results, "records": records},
                                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n用时 {time.perf_counter() - t0:.0f}s；记录已写到 {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()