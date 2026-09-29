"""
SentryAML - M6a：用 SHAP 解释 M4b 模型
======================================
全部在【验证集】（9/8）上做。测试集的纪律见 decisions.md D29：
测试集只用来报最终成绩，解释和诊断放在验证集上。

内容：
  0. 打分、抽样、检验 SHAP 的可加性
  1. 全局重要性：SHAP vs 训练增益（gain），普通交易 vs 告警
  2. 特征方向与形状：取值变大时，贡献怎么变（金额是否呈"区间"形状）
  3. 告警内部：真洗钱 vs 误报，是什么让正常交易看起来像洗钱
  4. 漏报：有模式标签 vs 无模式标签（D32 的盲区在验证集上是否也存在）
  5. 三个单笔解释：最高分的真洗钱、最高分的误报、一笔漏报

用法：
    python scripts/shap_analysis.py
    python scripts/shap_analysis.py --sample 30000
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_spec import LABEL_COL
from model_explain import (check_additivity, describe, display_value, explain_rows,
                           load_booster, print_explanation, shap_values)
from train_lgbm import load_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
REPORT_DIR = PROJECT_ROOT / "reports"

BUDGETS = (0.001, 0.005, 0.01)
# 金额按美元分档（D13：洗钱集中在 2,000–25,000 美元），看模型是否学出了"区间"
AMOUNT_EDGES_USD = [0, 100, 1_000, 3_000, 10_000, 25_000, 100_000, 1e13]


def top_k_mask(score: np.ndarray, frac: float) -> np.ndarray:
    k = max(1, math.ceil(frac * len(score)))
    idx = np.argpartition(-score, k - 1)[:k]
    m = np.zeros(len(score), dtype=bool)
    m[idx] = True
    return m


def dependence(name: str, v: pd.Series, s: np.ndarray) -> list[dict]:
    """把特征取值分档，看每档的平均 SHAP。"""
    if name == "f_log_amount":
        bins = pd.cut(v, np.log1p(AMOUNT_EDGES_USD), right=False)
    elif v.nunique() <= 6:
        bins = v
    else:
        bins = pd.qcut(v, q=8, duplicates="drop")
    df = pd.DataFrame({"bin": bins, "v": v.to_numpy(), "s": s})
    rows = []
    for _, g in df.groupby("bin", observed=True, sort=True):
        if len(g) == 0:
            continue
        lo, hi = g["v"].min(), g["v"].max()
        label = (display_value(name, lo) if lo == hi
                 else f"{display_value(name, lo)} – {display_value(name, hi)}")
        rows.append({"range": label, "n": int(len(g)),
                     "mean_shap": float(g["s"].mean()),
                     "share_positive": float((g["s"] > 0).mean())})
    return rows


def main():
    ap = argparse.ArgumentParser(description="M6a：SHAP 解释（验证集）")
    ap.add_argument("--model", choices=["m4b", "optuna"], default="m4b")
    ap.add_argument("--sample", type=int, default=20000, help="代表性随机样本的行数")
    ap.add_argument("--dep-top", type=int, default=5, help="画方向表的特征个数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    booster, meta = load_booster(args.model)
    num, cat = meta["features_num"], meta["features_cat"]
    cols = num + cat
    names = booster.feature_name()
    assert names == cols, "模型里的特征顺序与元数据不一致"

    # ======================================================================
    # 0. 打分、抽样、可加性检验
    # ======================================================================
    t0 = time.perf_counter()
    X, y = load_split("val", cols, cat, None, args.seed)
    extra = pq.read_table(FEATURE_DIR / "val.parquet",
                          columns=["txn_id", "pattern_type", LABEL_COL]).to_pandas()
    assert (extra[LABEL_COL].to_numpy() == y).all(), "元数据与特征行顺序不一致"
    raw = booster.predict(X, raw_score=True)
    pct = pd.Series(raw).rank(pct=True).to_numpy()          # 分数在验证集中的百分位
    n = len(y)
    print(f"验证集 {n:,} 行（洗钱 {int(y.sum())}），打分用时 {time.perf_counter() - t0:.1f}s")

    rng = np.random.default_rng(args.seed)
    uni = np.sort(rng.choice(n, size=min(args.sample, n), replace=False))
    alert = top_k_mask(raw, 0.001)
    top1 = top_k_mask(raw, 0.01)
    idx_alert = np.flatnonzero(alert)
    idx_pos = np.flatnonzero(y == 1)
    idx_missed = np.flatnonzero((y == 1) & ~top1)            # 连前 1% 都没进
    need = np.unique(np.concatenate([uni, idx_alert, idx_pos]))

    t1 = time.perf_counter()
    X_need = X.iloc[need]
    contrib_need, base_need = shap_values(booster, X_need)
    err = check_additivity(booster, X_need, contrib_need, base_need)
    pos_of = {r: i for i, r in enumerate(need)}             # 原始行号 -> contrib 中的行号

    def C(rows):                                             # 取若干原始行的 SHAP
        return contrib_need[[pos_of[r] for r in rows]]

    base = float(base_need[0])
    print(f"SHAP 计算 {len(need):,} 行，用时 {time.perf_counter() - t1:.1f}s")
    print(f"可加性检验：|基准值 + 贡献之和 − 原始分数| 最大 {err:.2e}（应接近 0）")
    print(f"基准值 {base:+.3f}（log-odds），约等于模型在【下采样后】训练数据上的平均分数；"
          f"下采样带来的偏移是 ln(0.2) = {math.log(0.2):+.3f}")

    report = {"model": args.model, "base_value": base, "additivity_max_err": err}

    # ======================================================================
    # 1. 全局重要性
    # ======================================================================
    print("\n" + "=" * 70)
    print("1. 全局重要性：平均 |SHAP|（普通交易 vs 告警）与训练增益")
    print("=" * 70)
    imp_uni = np.abs(C(uni)).mean(axis=0)
    imp_alert = np.abs(C(idx_alert)).mean(axis=0)
    gain = booster.feature_importance("gain")
    gain = gain / gain.sum()
    g = pd.DataFrame({"feature": names, "uni": imp_uni, "alert": imp_alert, "gain": gain})
    for c in ("uni", "alert", "gain"):
        g[f"{c}_rank"] = g[c].rank(ascending=False, method="min").astype(int)
    g = g.sort_values("uni", ascending=False).reset_index(drop=True)

    print(f"  {'特征':<40}{'随机样本':>9}{'排名':>5}{'告警':>9}{'排名':>5}{'增益占比':>9}{'排名':>5}")
    for _, r in g.head(15).iterrows():
        print(f"  {describe(r['feature'])[:20]:<40}{r['uni']:>9.3f}{r['uni_rank']:>5}"
              f"{r['alert']:>9.3f}{r['alert_rank']:>5}{r['gain']:>9.1%}{r['gain_rank']:>5}")
    print(f"\n  随机样本 {len(uni):,} 笔代表「普通交易」；告警为分数前 0.1%（{len(idx_alert)} 笔）")
    report["global_importance"] = g.to_dict(orient="records")

    # ======================================================================
    # 2. 特征方向与形状
    # ======================================================================
    print("\n" + "=" * 70)
    print("2. 特征方向与形状：每档取值的平均 SHAP（正 = 推高分数）")
    print("=" * 70)
    dep_feats = [f for f in g["feature"] if f not in cat][:args.dep_top]
    if "f_log_amount" not in dep_feats:
        dep_feats.append("f_log_amount")
    X_uni, C_uni = X.iloc[uni], C(uni)
    report["dependence"] = {}
    for f in dep_feats:
        j = names.index(f)
        rows = dependence(f, X_uni[f].astype(float), C_uni[:, j])
        report["dependence"][f] = rows
        print(f"\n  {describe(f)}（{f}）")
        print(f"  {'取值区间':<34}{'笔数':>8}{'平均 SHAP':>11}{'推高的比例':>11}")
        for r in rows:
            print(f"  {r['range']:<34}{r['n']:>8,}{r['mean_shap']:>+11.3f}{r['share_positive']:>11.0%}")

    # ======================================================================
    # 3. 告警内部：真洗钱 vs 误报
    # ======================================================================
    print("\n" + "=" * 70)
    tp = idx_alert[y[idx_alert] == 1]
    fp = idx_alert[y[idx_alert] == 0]
    print(f"3. 告警内部（前 0.1%）：真洗钱 {len(tp)} 笔 vs 误报 {len(fp)} 笔")
    print("=" * 70)
    s_tp, s_fp = C(tp).mean(axis=0), C(fp).mean(axis=0)
    t = pd.DataFrame({"feature": names, "tp": s_tp, "fp": s_fp})
    t = t.sort_values("fp", ascending=False).reset_index(drop=True)
    print("  误报被哪些特征推高（按误报组平均 SHAP 排序）：")
    print(f"  {'特征':<40}{'误报 SHAP':>10}{'真洗钱 SHAP':>12}{'误报中位值':>14}{'真洗钱中位值':>14}")
    tp_rows = []
    for _, r in t.head(10).iterrows():
        f = r["feature"]
        if f in cat:
            m_fp = X.iloc[fp][f].mode().iat[0] if len(fp) else "-"
            m_tp = X.iloc[tp][f].mode().iat[0] if len(tp) else "-"
        else:
            m_fp = display_value(f, float(X.iloc[fp][f].median())) if len(fp) else "-"
            m_tp = display_value(f, float(X.iloc[tp][f].median())) if len(tp) else "-"
        print(f"  {describe(f)[:20]:<40}{r['fp']:>+10.3f}{r['tp']:>+12.3f}{str(m_fp):>14}{str(m_tp):>14}")
        tp_rows.append({"feature": f, "fp_mean_shap": float(r["fp"]),
                        "tp_mean_shap": float(r["tp"]), "fp_median": str(m_fp),
                        "tp_median": str(m_tp)})
    report["alert_tp_vs_fp"] = {"n_tp": int(len(tp)), "n_fp": int(len(fp)), "top": tp_rows}

    # ======================================================================
    # 4. 漏报：有模式标签 vs 无模式标签
    # ======================================================================
    print("\n" + "=" * 70)
    print("4. 盲区检查（验证集）：有模式标签 vs 无模式标签")
    print("=" * 70)
    tagged = extra["pattern_type"].notna().to_numpy()
    print(f"  {'':<16}{'洗钱笔数':>8}" + "".join(f"{f'前{b:.1%}召回':>12}" for b in BUDGETS))
    blind = {}
    for label, grp in (("有模式标签", tagged), ("无模式标签", ~tagged)):
        m = (y == 1) & grp
        recalls = [float((top_k_mask(raw, b) & m).sum() / max(m.sum(), 1)) for b in BUDGETS]
        blind[label] = {"n": int(m.sum()), "recall": recalls}
        print(f"  {label:<16}{int(m.sum()):>8}" + "".join(f"{r:>12.1%}" for r in recalls))
    report["blind_spot_val"] = blind

    caught = np.flatnonzero((y == 1) & alert)
    if len(idx_missed) and len(caught):
        d = C(idx_missed).mean(axis=0) - C(caught).mean(axis=0)
        order = np.argsort(d)[:8]
        n_untag = int((~tagged[idx_missed]).sum())
        print(f"\n  漏报（连前 1% 都没进）{len(idx_missed)} 笔，其中无模式标签 {n_untag} 笔")
        print("  漏报与抓到的洗钱相比，哪些特征的贡献差得最多：")
        print(f"  {'特征':<40}{'漏报 SHAP':>10}{'抓到 SHAP':>10}{'差':>9}")
        miss_rows = []
        for j in order:
            m_s = C(idx_missed)[:, j].mean()
            c_s = C(caught)[:, j].mean()
            print(f"  {describe(names[j])[:20]:<40}{m_s:>+10.3f}{c_s:>+10.3f}{d[j]:>+9.3f}")
            miss_rows.append({"feature": names[j], "missed": float(m_s),
                              "caught": float(c_s), "diff": float(d[j])})
        report["missed_vs_caught"] = {"n_missed": int(len(idx_missed)),
                                      "n_missed_untagged": n_untag, "top": miss_rows}

    # ======================================================================
    # 5. 单笔解释
    # ======================================================================
    print("\n" + "=" * 70)
    print("5. 单笔解释（Agent 处理告警时拿到的就是这种格式）")
    print("=" * 70)
    cases = []
    if len(tp):
        cases.append(("分数最高的真洗钱", tp[np.argmax(raw[tp])]))
    if len(fp):
        cases.append(("分数最高的误报", fp[np.argmax(raw[fp])]))
    if len(idx_missed):
        mid = idx_missed[np.argsort(raw[idx_missed])[len(idx_missed) // 2]]
        cases.append(("一笔漏报（漏报中分数居中）", mid))
    report["examples"] = []
    for title, r in cases:
        e = explain_rows(names, X.iloc[[r]], C([r]), np.array([base]))[0]
        pt = extra["pattern_type"].iat[r]
        head = (f"{title}｜txn_id {int(extra['txn_id'].iat[r])}｜标签 {int(y[r])}"
                f"｜模式 {pt if isinstance(pt, str) else '无'}｜验证集百分位 {pct[r]:.4%}")
        print_explanation(e, head)
        report["examples"].append({"title": title, "txn_id": int(extra["txn_id"].iat[r]),
                                   "label": int(y[r]), "pattern_type": pt if isinstance(pt, str) else None,
                                   "percentile": float(pct[r]), **e})

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"m6a_shap_{args.model}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已保存：{out}")


if __name__ == "__main__":
    main()