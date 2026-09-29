"""
SentryAML - M6c：事后描述性分析
===============================
【性质】看过测试集结果之后追加的分析。结论只用来理解模型和数据，
       不用来修改模型、特征或阈值（decisions.md D29 的纪律）。

四项内容，分别回答前面留下的四个问题：
  1. D36 的机制：F 在 9/9 的误报，是不是来自"近期安静、但和老对象交易"的账户？
  2. D34 的假设：无标签洗钱是不是账户级标签（这些账户几乎每笔交易都被标为洗钱）？
  3. D31 的高估：M4b 的概率高估是不是集中在 9/9？
  4. D29 的反转："训练期出现过"那组洗钱，是不是无标签的更多？

用法：
    python scripts/posthoc_analysis.py
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from evaluate_test import acct_key, budget_threshold, load_meta_cols, load_model
from feature_spec import LABEL_COL
from train_lgbm import load_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

QUIET = "p_out_cnt_3d"          # 付款方前 3 天转出笔数 = 0 → "近期安静"
ACTIVE = "p_active_days_3d"     # 付款方前 3 天活跃天数 = 0 → 前 3 天完全没有交易
PAIR = "f_pair_prev_cnt"        # 双方此前来往笔数 = 0 → 首次来往
BUDGET = 0.01                   # 统一用前 1% 作为"高分"


def pct(x) -> str:
    return f"{x:.1%}" if x == x else "  -"        # NaN 显示为 -


def main():
    t0 = time.perf_counter()
    report = {"note": "看过测试集之后的描述性分析，不用于修改模型"}

    b_m, meta_m = load_model("m4b")
    b_f, meta_f = load_model("ablate_F")
    cols_m = meta_m["features_num"] + meta_m["features_cat"]
    cols_f = meta_f["features_num"] + meta_f["features_cat"]
    cat = meta_m["features_cat"]

    X_te, y_te = load_split("test", cols_m, cat, None, 42)
    X_va, y_va = load_split("val", cols_m, cat, None, 42)
    m_te = load_meta_cols("test")
    m_va = load_meta_cols("val")
    s_m = b_m.predict(X_te[cols_m])
    s_f = b_f.predict(X_te[cols_f])
    day_te = pd.to_datetime(m_te["txn_ts"]).dt.date.astype(str).to_numpy()
    print(f"读取与打分用时 {time.perf_counter() - t0:.1f}s")

    # ======================================================================
    # 1. D36 的机制
    # ======================================================================
    print("\n" + "=" * 78)
    print("1. D36 的机制：F 在 9/9 的误报来自哪里")
    print("=" * 78)

    # 1a. 每天的交易构成
    print("\n  1a. 每天的交易构成（付款方视角）")
    print(f"  {'日期':<12}{'交易数':>10}{'前3天无转出':>12}{'前3天无任何交易':>16}{'首次来往':>10}"
          f"{'安静且非首次':>14}")
    frames = [("2022-09-08（验证）", X_va, np.ones(len(y_va), bool))]
    for d in sorted(set(day_te)):
        frames.append((d, X_te, day_te == d))
    comp = {}
    for name, X, mk in frames:
        q = (X.loc[mk, QUIET] == 0).to_numpy()
        a = (X.loc[mk, ACTIVE] == 0).to_numpy()
        fc = (X.loc[mk, PAIR] == 0).to_numpy()
        qn = q & ~fc
        comp[name] = {"n": int(mk.sum()), "quiet": float(q.mean()), "inactive": float(a.mean()),
                      "first_contact": float(fc.mean()), "quiet_not_first": float(qn.mean()),
                      "quiet_not_first_n": int(qn.sum())}
        print(f"  {name:<12}{int(mk.sum()):>10,}{q.mean():>12.1%}{a.mean():>16.1%}"
              f"{fc.mean():>10.1%}{qn.mean():>14.1%}")
    report["daily_composition"] = comp

    # 1b. 两个模型的前 1% 告警里，误报的构成
    thr_m, thr_f = budget_threshold(s_m, BUDGET), budget_threshold(s_f, BUDGET)
    top_m, top_f = s_m >= thr_m, s_f >= thr_f
    quiet = (X_te[QUIET] == 0).to_numpy()
    first = (X_te[PAIR] == 0).to_numpy()
    print(f"\n  1b. 测试集前 1% 告警（{int(top_f.sum()):,} 条）中的误报构成")
    print(f"  {'模型':<6}{'日期':<12}{'告警':>8}{'误报':>8}{'误报中安静':>12}{'误报中非首次':>14}"
          f"{'误报中安静且非首次':>18}")
    mech = {}
    for mname, top in (("F", top_f), ("M4b", top_m)):
        for d in sorted(set(day_te)):
            mk = top & (day_te == d)
            fp = mk & (y_te == 0)
            n_fp = int(fp.sum())
            r = {"alerts": int(mk.sum()), "fp": n_fp,
                 "fp_quiet": float(quiet[fp].mean()) if n_fp else float("nan"),
                 "fp_not_first": float((~first[fp]).mean()) if n_fp else float("nan"),
                 "fp_quiet_not_first": float((quiet & ~first)[fp].mean()) if n_fp else float("nan")}
            mech[f"{mname}_{d}"] = r
            print(f"  {mname:<6}{d:<12}{r['alerts']:>8,}{n_fp:>8,}{pct(r['fp_quiet']):>12}"
                  f"{pct(r['fp_not_first']):>14}{pct(r['fp_quiet_not_first']):>18}")
    report["top1pct_fp_profile"] = mech

    # 1c. "安静且非首次"这一群，在两个模型里分别被告警多少
    grp = quiet & ~first
    print(f"\n  1c. \"付款方前 3 天无转出、且与老对象交易\"的交易，被两个模型告警的比例")
    print(f"  {'日期':<12}{'笔数':>10}{'其中洗钱':>10}{'F 告警比例':>12}{'M4b 告警比例':>14}")
    grp_rows = {}
    for d in sorted(set(day_te)):
        mk = grp & (day_te == d)
        n = int(mk.sum())
        r = {"n": n, "pos": int(y_te[mk].sum()),
             "f_alert_rate": float(top_f[mk].mean()) if n else float("nan"),
             "m4b_alert_rate": float(top_m[mk].mean()) if n else float("nan")}
        grp_rows[d] = r
        print(f"  {d:<12}{n:>10,}{r['pos']:>10,}{pct(r['f_alert_rate']):>12}{pct(r['m4b_alert_rate']):>14}")
    report["quiet_not_first_group"] = grp_rows

    # ======================================================================
    # 2. D34：无标签洗钱是不是账户级标签
    # ======================================================================
    print("\n" + "=" * 78)
    print("2. D34：无标签洗钱是否是账户级标签（训练 + 验证 + 测试全部交易）")
    print("=" * 78)
    cols = ["from_bank", "from_account", "to_bank", "to_account", "pattern_type", LABEL_COL]
    allm = pd.concat([pq.read_table(FEATURE_DIR / f"{s}.parquet", columns=cols).to_pandas()
                      for s in ("train", "val", "test")], ignore_index=True)
    allm[LABEL_COL] = allm[LABEL_COL].astype("int32")
    allm["tag_l"] = ((allm[LABEL_COL] == 1) & allm["pattern_type"].notna()).astype("int32")
    acct_rows = {}
    print(f"  {'视角':<8}{'洗钱类型':<10}{'洗钱笔数':>9}{'涉及账户':>9}{'前10账户占比':>13}"
          f"{'账户总笔数中位':>15}{'账户洗钱占比中位':>17}{'占比≥90%的账户':>15}")
    for side, (bk, ac) in (("付款方", ("from_bank", "from_account")),
                           ("收款方", ("to_bank", "to_account"))):
        per = allm.groupby([bk, ac], sort=False).agg(
            total=(LABEL_COL, "size"), laund=(LABEL_COL, "sum"), tag_l=("tag_l", "sum"))
        per["untag_l"] = per["laund"] - per["tag_l"]
        per["share"] = per["laund"] / per["total"]
        for kind, col in (("无标签", "untag_l"), ("有标签", "tag_l")):
            sub = per[per[col] > 0]
            n_l = int(sub[col].sum())
            top10 = float(sub[col].nlargest(10).sum() / max(n_l, 1))
            r = {"laundering_txns": n_l, "accounts": int(len(sub)), "top10_share": top10,
                 "median_total_txns": float(sub["total"].median()),
                 "median_laund_share": float(sub["share"].median()),
                 "accounts_share_ge_90": float((sub["share"] >= 0.9).mean())}
            acct_rows[f"{side}_{kind}"] = r
            print(f"  {side:<8}{kind:<10}{n_l:>9,}{len(sub):>9,}{top10:>13.1%}"
                  f"{r['median_total_txns']:>15,.0f}{r['median_laund_share']:>17.1%}"
                  f"{r['accounts_share_ge_90']:>15.1%}")
    print("  （账户洗钱占比 = 该账户在该视角下的全部交易中，被标为洗钱的比例；"
          "只统计特征表里的交易：不含自环、9/1–9/2 与 9/11 之后）")
    report["account_level_labels"] = acct_rows

    # ======================================================================
    # 3. D31：M4b 的高估是否集中在 9/9
    # ======================================================================
    print("\n" + "=" * 78)
    print("3. D31：M4b 校准后的概率，按天拆开看")
    print("=" * 78)
    cal_path = MODEL_DIR / "calibrator_m4b.joblib"
    if cal_path.exists():
        iso = joblib.load(cal_path)
        p = iso.predict(s_m)
        bins = [0, 0.001, 0.01, 0.05, 0.2, 0.5, 1.0001]
        labels = ["<0.1%", "0.1–1%", "1–5%", "5–20%", "20–50%", "≥50%"]
        cal_rows = {}
        for d in sorted(set(day_te)):
            mk = day_te == d
            print(f"\n  {d}：平均预测 {p[mk].mean():.4%}，实际 {y_te[mk].mean():.4%}")
            print(f"  {'区间':<10}{'交易数':>10}{'平均预测':>10}{'实际比例':>10}")
            cut = pd.cut(p[mk], bins=bins, labels=labels, right=False)
            rel = (pd.DataFrame({"b": cut, "p": p[mk], "y": y_te[mk]})
                     .groupby("b", observed=False).agg(n=("y", "size"), mp=("p", "mean"), r=("y", "mean")))
            cal_rows[d] = {"mean_pred": float(p[mk].mean()), "actual": float(y_te[mk].mean()),
                           "bins": {str(b): {"n": int(r["n"]),
                                             "mean_pred": float(r["mp"]) if r["n"] else None,
                                             "actual": float(r["r"]) if r["n"] else None}
                                    for b, r in rel.iterrows()}}
            for b, r in rel.iterrows():
                if r["n"]:
                    print(f"  {b:<10}{int(r['n']):>10,}{r['mp']:>10.2%}{r['r']:>10.2%}")
        report["m4b_calibration_by_day"] = cal_rows
    else:
        print("  （跳过：没有找到 calibrator_m4b.joblib，需要先运行过 evaluate_test.py --model m4b）")

    # ======================================================================
    # 4. D29：出现过 / 没出现过 × 有标签 / 无标签
    # ======================================================================
    print("\n" + "=" * 78)
    print("4. D29 的反转：M4b 在 \"训练期出现过 / 没出现过\" × \"有 / 无模式标签\" 上的召回")
    print("=" * 78)
    tr = pq.read_table(FEATURE_DIR / "train.parquet",
                       columns=["from_bank", "from_account", "to_bank", "to_account", LABEL_COL]).to_pandas()
    tr = tr[tr[LABEL_COL] == 1]
    seen_accts = set(acct_key(tr["from_bank"], tr["from_account"])) | \
                 set(acct_key(tr["to_bank"], tr["to_account"]))
    seen = (acct_key(m_te["from_bank"], m_te["from_account"]).isin(seen_accts) |
            acct_key(m_te["to_bank"], m_te["to_account"]).isin(seen_accts)).to_numpy()
    tagged = m_te["pattern_type"].notna().to_numpy()
    thr = {f: budget_threshold(s_m, f) for f in (0.001, 0.005, 0.01)}
    print(f"  {'分组':<24}{'洗钱笔数':>9}{'前0.1%召回':>12}{'前0.5%召回':>12}{'前1.0%召回':>12}")
    seen_rows = {}
    for sname, smask in (("出现过", seen), ("没出现过", ~seen)):
        n_all = int((smask & (y_te == 1)).sum())
        n_un = int((smask & ~tagged & (y_te == 1)).sum())
        print(f"  {sname}：洗钱 {n_all} 笔，其中无标签 {n_un} 笔（{n_un / max(n_all, 1):.1%}）")
        for tname, tmask in (("有标签", tagged), ("无标签", ~tagged)):
            pos = smask & tmask & (y_te == 1)
            n = int(pos.sum())
            rec = [float((s_m[pos] >= thr[f]).mean()) if n else float("nan") for f in thr]
            seen_rows[f"{sname}_{tname}"] = {"n": n, "recall": rec}
            print(f"    {sname + ' × ' + tname:<22}{n:>9}" + "".join(f"{pct(v):>12}" for v in rec))
    report["seen_x_tagged"] = seen_rows

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "m6c_posthoc.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n报告已保存：{out}   总用时 {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()