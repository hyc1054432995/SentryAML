"""
SentryAML - R9c：分诊 Agent 的评测（加权还原到整个告警队列 + 与"按分数升级同样多条"的基线比较）
=================================================================================================
先在 dev 上把评测代码写好、跑通、看懂（这一步可以反复跑）；比较方案与采用规则写定之后（D75），test 只跑一次。

1. 每层的升级比例：样本按 m 加权（D72：先等概率抽分量、再在分量里抽 1 条，大分量里的告警被抽中的概率是 1/m）
     比例_h = Σ(升级_i × m_i) / Σ m_i
   Agent 出错（生成失败、没有提交）按"升级"计：分诊失败的告警留给人工处理，这是运行时的真实后果（D75）。
2. 还原到整个队列（4,226 条）：各层的比例 × 这一层的总条数，加起来
     Agent 升级数 E、其中洗钱 TP；精确率 = TP / E；洗钱召回 = TP / 627；减少的误报 = 非洗钱里被关闭的条数
3. 基线：按模型分数升级同样多的 E 条（告警编号就是分数名次）。它不需要抽样，直接在全部 4,226 条上精确计算。
   比较 Agent 与基线在同样的升级数下各抓到多少洗钱：差值 = TP_agent − TP_baseline。
4. 两种用法（D75，在看到 dev 加权结果之前写定）：
   方案 A：Agent 分诊全部告警（T1 也可能被关闭）；基线 = 全部告警里分数最高的 E 条。
   方案 B：T1 全部直接升级，Agent 只分诊 T2；基线 = T1 全部 + T2 里分数最高的 E2 条。
           两边的 T1 部分完全相同，差值只来自 T2，所以方案 B 与基线的比较 = 只在 T2 内部比较。
   dev 上按差值的点估计选一个方案（写进 r9c_eval_dev.json），test 上把选中的方案作为主要结果报告。
5. 不确定性：在每一层内对样本有放回地重抽（带权重 m），重复 4,000 次，得到 TP、E、差值的 95% 区间；
   每次重抽的 E 不同，基线跟着在新的 E 处重新计算。
6. 其他：手法判断（有模式标签的洗钱告警里，Agent 升级时给出的手法与标签是否一致）、核验、费用与耗时。

用法：
    python scripts/r9c_eval.py --split dev
    python scripts/r9c_eval.py --split test        # 只在 D75 的规则写定、test 跑完之后
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from db import connect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT = PROJECT_ROOT / "reports" / "r9c_split.json"
RUNS = {"dev": PROJECT_ROOT / "reports" / "r9b_dev_runs.json", "test": PROJECT_ROOT / "reports" / "r9c_test_runs.json"}
STRATA = ["T1/L", "T1/N", "T2/L", "T2/N"]
PATTERN = {"FAN-IN": "fan_in", "FAN-OUT": "fan_out", "GATHER-SCATTER": "gather_scatter",
           "SCATTER-GATHER": "scatter_gather", "CYCLE": "cycle", "RANDOM": "random_walk",
           "BIPARTITE": "bipartite", "STACK": "stack"}
B = 4000


def load(split: str):
    sp = json.loads(SPLIT.read_text(encoding="utf-8"))
    sample = sp[split]
    runs = json.loads(RUNS[split].read_text(encoding="utf-8"))
    pop = sp["strata_population"]
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT a.alert_id, a.tier, l.is_laundering, l.pattern_type
                       FROM alerts a JOIN alert_labels l USING (alert_id) ORDER BY a.alert_id""")
        allrows = cur.fetchall()
    return sample, runs, pop, allrows


def escalated(rec: dict | None) -> int:
    """Agent 的结论；出错按升级计（分诊失败的告警留给人工处理）。"""
    if rec is None or rec.get("status") != "done":
        return 1
    return int(rec["decision"] == "escalate")


def weighted_rate(esc: np.ndarray, m: np.ndarray) -> float:
    return float((esc * m).sum() / m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    a = ap.parse_args()
    sample, runs, pop, allrows = load(a.split)

    missing = [x["alert_id"] for x in sample if str(x["alert_id"]) not in runs]
    versions = Counter(runs[str(x["alert_id"])].get("prompt_version") for x in sample if str(x["alert_id"]) in runs)
    print(f"{a.split}：样本 {len(sample)} 条，有运行记录 {len(sample) - len(missing)} 条；提示词版本 {dict(versions)}")
    if missing:
        print(f"  缺少运行记录：{missing}（按出错计，即升级）")
    if len(versions) > 1:
        print("  注意：样本里混有不同版本的运行记录，结论不对应单一版本")

    # 1. 每层
    by = defaultdict(list)
    for x in sample:
        rec = runs.get(str(x["alert_id"]))
        by[x["stratum"]].append((escalated(rec), x["m"], rec))
    print(f"\n[1] 各层（样本按 m 加权；出错按升级计）")
    print(f"  {'层':<6}{'样本':>5}{'升级':>5}{'出错':>5}{'加权升级比例':>12}{'总体条数':>9}")
    rate = {}
    for h in STRATA:
        esc = np.array([e for e, _, _ in by[h]], dtype=float)
        m = np.array([mm for _, mm, _ in by[h]], dtype=float)
        rate[h] = weighted_rate(esc, m)
        err = sum(1 for _, _, r in by[h] if r is None or r.get("status") != "done")
        print(f"  {h:<6}{len(esc):>5}{int(esc.sum()):>5}{err:>5}{rate[h]:>12.1%}{pop[h]:>9,}")

    # 2–3. 还原到整个队列，与基线比较
    lab = np.array([r[2] for r in allrows])                     # 按告警编号 = 分数名次排好
    tier = np.array([r[1] for r in allrows])
    cum = np.concatenate([[0], np.cumsum(lab)])
    lab2 = lab[tier == "T2"]
    cum2 = np.concatenate([[0], np.cumsum(lab2)])
    n_l = int(lab.sum())

    def summarize(rt: dict) -> dict:
        E = sum(pop[h] * rt[h] for h in STRATA)
        TP = pop["T1/L"] * rt["T1/L"] + pop["T2/L"] * rt["T2/L"]
        K = int(round(E))
        E2 = pop["T2/L"] * rt["T2/L"] + pop["T2/N"] * rt["T2/N"]
        TP2 = pop["T2/L"] * rt["T2/L"]
        K2 = int(round(E2))
        return {"E": E, "TP": TP, "base_TP": float(cum[K]), "diff": TP - float(cum[K]),
                "E2": E2, "TP2": TP2, "base_TP2": float(cum2[K2]), "diff2": TP2 - float(cum2[K2]),
                "closed_N": pop["T1/N"] * (1 - rt["T1/N"]) + pop["T2/N"] * (1 - rt["T2/N"])}
    point = summarize(rate)

    # 5. 分层带权重的自助法
    rng = np.random.default_rng(0)
    boots = []
    arrays = {h: (np.array([e for e, _, _ in by[h]], float), np.array([mm for _, mm, _ in by[h]], float))
              for h in STRATA}
    for _ in range(B):
        rt = {}
        for h, (esc, m) in arrays.items():
            idx = rng.integers(0, len(esc), len(esc))
            rt[h] = weighted_rate(esc[idx], m[idx])
        boots.append(summarize(rt))
    ci = lambda k: (np.percentile([b[k] for b in boots], 2.5), np.percentile([b[k] for b in boots], 97.5))

    print(f"\n[2] 方案 A：Agent 分诊全部告警（{len(allrows):,} 条，其中洗钱 {n_l}）")
    E, TP = point["E"], point["TP"]
    print(f"  Agent 升级 {E:,.0f} 条 [{ci('E')[0]:,.0f}, {ci('E')[1]:,.0f}]，其中洗钱 {TP:,.0f} 条 "
          f"[{ci('TP')[0]:,.0f}, {ci('TP')[1]:,.0f}]")
    print(f"  精确率 {TP / E:.1%}；洗钱召回 {TP / n_l:.1%}；关闭的非洗钱告警 {point['closed_N']:,.0f} 条"
          f"（占非洗钱 {point['closed_N'] / (len(allrows) - n_l):.1%}）")
    print(f"  其中 T1：关闭 {pop['T1/N'] * (1 - rate['T1/N']):,.0f} 条非洗钱，代价是关掉 "
          f"{pop['T1/L'] * (1 - rate['T1/L']):,.0f} 条洗钱")
    K = int(round(E))
    print(f"\n[3] 基线：按模型分数升级同样多的 {K:,} 条 → 洗钱 {point['base_TP']:,.0f} 条，精确率 "
          f"{point['base_TP'] / K:.1%}")
    lo, hi = ci("diff")
    print(f"  Agent − 基线（同样升级数下多抓到的洗钱）：{point['diff']:+,.0f} 条，95% 区间 [{lo:+,.0f}, {hi:+,.0f}]"
          + ("  ← 区间不含 0" if lo > 0 or hi < 0 else "  ← 区间含 0：与基线没有可分辨的差别"))

    print(f"\n[4] 方案 B：T1 全部直接升级，Agent 只分诊 T2（{pop['T2/L'] + pop['T2/N']:,} 条，洗钱 {pop['T2/L']}）")
    print(f"  T1 两边相同（{pop['T1/L'] + pop['T1/N']:,} 条全部升级），差别只在 T2；整体升级 "
          f"{pop['T1/L'] + pop['T1/N'] + point['E2']:,.0f} 条，其中洗钱 {pop['T1/L'] + point['TP2']:,.0f}，"
          f"洗钱召回 {(pop['T1/L'] + point['TP2']) / n_l:.1%}")
    E2, TP2 = point["E2"], point["TP2"]
    K2 = int(round(E2))
    lo2, hi2 = ci("diff2")
    print(f"  Agent 在 T2 升级 {E2:,.0f} 条，其中洗钱 {TP2:,.0f}（精确率 {TP2 / E2:.1%}，T2 洗钱召回 "
          f"{TP2 / pop['T2/L']:.1%}）")
    print(f"  基线：T2 里分数最高的 {K2:,} 条 → 洗钱 {point['base_TP2']:,.0f}（精确率 {point['base_TP2'] / K2:.1%}）")
    print(f"  Agent − 基线：{point['diff2']:+,.0f} 条，95% 区间 [{lo2:+,.0f}, {hi2:+,.0f}]"
          + ("  ← 区间不含 0" if lo2 > 0 or hi2 < 0 else "  ← 区间含 0"))

    # 6. 手法、核验、费用
    lab_by_id = {r[0]: (r[2], r[3]) for r in allrows}
    typ = []
    for x in sample:
        rec = runs.get(str(x["alert_id"]))
        l, p = lab_by_id[x["alert_id"]]
        if l and p and rec and rec.get("status") == "done" and rec["decision"] == "escalate":
            typ.append((PATTERN.get(p), rec["typology"]))
    exact = sum(t == g for g, t in typ)
    kl = sum(t == "known_launderer_integration" for g, t in typ)
    print(f"\n[5] 手法：有模式标签、被升级的洗钱告警 {len(typ)} 条；手法与标签一致 {exact} 条；"
          f"判为'已知洗钱账户的后续交易' {kl} 条（这是对证据的描述，不对应数据集的模式标签）")
    for (g, t), c in Counter(typ).most_common():
        print(f"    标签 {g:<16} → Agent {t:<28} {c}")
    recs = [runs[str(x["alert_id"])] for x in sample if str(x["alert_id"]) in runs]
    done = [r for r in recs if r.get("status") == "done"]
    fp = sum(1 for r in done if r.get("first_pass") and r["first_pass"]["verified"] == r["first_pass"]["facts"]
             and not r["first_pass"]["ungrounded"])
    print(f"\n[6] 核验：完成 {len(done)}/{len(recs)}；第一次提交即全部通过 {fp}/{len(done)}；"
          f"最终全部通过 {sum(1 for r in done if r.get('verified'))}/{len(done)}；"
          f"因输出截断被提醒 {sum(1 for r in recs if r.get('length_nudged'))} 次")
    # 费用与耗时只看真实调用：从缓存回放的运行（全部或部分调用命中缓存）费用记为 0、耗时只有几百毫秒，
    # 混进来会把中位数压到 0（D76）
    fresh = [r for r in recs if r["usage"].get("cached_calls", 0) == 0]
    print(f"  模型补查次数 {dict(sorted(Counter(r.get('tool_calls', 0) for r in recs).items()))}")
    if fresh:
        cost = [r["cost_usd"] for r in fresh]
        secs = [r["ms"]["total"] / 1000 for r in fresh]
        print(f"  费用与耗时（{len(fresh)} 条真实运行）：合计 ${sum(cost):.3f}、每条中位数 ${statistics.median(cost):.4f}、"
              f"最大 ${max(cost):.4f}；耗时中位数 {statistics.median(secs):.0f}s、最大 {max(secs):.0f}s")
    if len(fresh) < len(recs):
        print(f"  另有 {len(recs) - len(fresh)} 条全部或部分来自缓存回放：它们的费用、耗时不代表真实运行，不计入上面的统计")
    # 7. 选方案（D75）：dev 上按"同样升级数下多抓到的洗钱"的点估计选；test 读 dev 的选择，不重新选
    res = {"rate": rate, "point": point, "versions": dict(versions),
           "ci": {k: ci(k) for k in ("E", "TP", "diff", "E2", "TP2", "diff2")}}
    if a.split == "dev":
        res["selected"] = "A" if point["diff"] > point["diff2"] else "B"
        print(f"\n[7] 选方案（dev）：A 的差值 {point['diff']:+,.0f}，B 的差值 {point['diff2']:+,.0f} → 选 "
              f"{res['selected']}（相等时选 B：T1 不让 Agent 关闭，更保守）")
    else:
        dev_file = PROJECT_ROOT / "reports" / "r9c_eval_dev.json"
        dev_res = json.loads(dev_file.read_text(encoding="utf-8")) if dev_file.exists() else {}
        sel = dev_res.get("selected")
        if sel is None:
            raise SystemExit("找不到 dev 的方案选择（reports/r9c_eval_dev.json）：先在 dev 上运行")
        if set(dev_res.get("versions", {})) != set(versions):
            print(f"  注意：test 的版本 {dict(versions)} 与 dev 评测的版本 {dev_res.get('versions')} 不一致")
        res["selected"] = sel
        k = "diff" if sel == "A" else "diff2"
        lo, hi = ci(k)
        print(f"\n[7] 主要结果（dev 选定的方案 {sel}）：Agent − 基线 {point[k]:+,.0f} 条洗钱，"
              f"95% 区间 [{lo:+,.0f}, {hi:+,.0f}]")
    out = PROJECT_ROOT / "reports" / f"r9c_eval_{a.split}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n结果写到 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()