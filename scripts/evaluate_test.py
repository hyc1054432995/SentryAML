"""
SentryAML - M5：测试集评测（只运行一次）
=======================================
测试集 = 9/9 ~ 9/10，从 M1 开始一直封存。本脚本是它第一次、也是唯一一次被使用。
【运行前先决定用哪个模型，运行后不再更换。】

一次完成六件事：
  1. 概率校准：在验证集上拟合 isotonic regression，在测试集上检验
  2. 测试集主指标 + bootstrap 95% 置信区间
  3. 规则引擎在【同一批测试交易】上重算，同告警量对比
  4. 按天拆分：9/9（有分布偏移）vs 9/10
  5. 按"洗钱账户是否在训练期出现过"拆分：以"未出现过"那组为准
  6. 按洗钱手法拆分：模型 vs 规则引擎

另外报告逻辑回归在测试集上的成绩，作为事先确定的 baseline（不参与选择）。

用法：
    python scripts/evaluate_test.py --model m4b       # 使用 M4b 手工最优（lgbm_best.txt）
    python scripts/evaluate_test.py --model optuna    # 使用 Optuna 最优（lgbm_optuna.txt）
    python scripts/evaluate_test.py --model ablate_F  # 保守版本：去掉全部来往历史特征（M6b）

保守版本属于 D24 事先登记的"有 / 无首次来往两个版本"，不是看过测试集后新挑的模型。
每个模型的报告单独保存（m4b 为 m5_test.json，其他为 m5_test_<model>.json），各自只跑一次。
"""

import argparse
import json
import math
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from eval_utils import correct_prob, evaluate, print_report
from feature_spec import LABEL_COL
from train_lgbm import load_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODELS = {
    "m4b":    ("lgbm_best.txt",   "lgbm_best_meta.joblib"),
    "optuna": ("lgbm_optuna.txt", "lgbm_optuna_meta.json"),
    "ablate_E": ("lgbm_ablate_E.txt", "lgbm_ablate_E_meta.joblib"),
    "ablate_F": ("lgbm_ablate_F.txt", "lgbm_ablate_F_meta.joblib"),
}
BUDGETS = (0.001, 0.005, 0.01)


# ---------------------------------------------------------------------------
# 读模型与数据
# ---------------------------------------------------------------------------
def load_model(key: str):
    model_file, meta_file = MODELS[key]
    booster = lgb.Booster(model_file=str(MODEL_DIR / model_file))
    meta_path = MODEL_DIR / meta_file
    meta = (json.load(open(meta_path, encoding="utf-8")) if meta_path.suffix == ".json"
            else joblib.load(meta_path))
    return booster, meta


def load_meta_cols(split: str) -> pd.DataFrame:
    """读取评测分组需要的元数据列。"""
    cols = ["txn_id", "txn_ts", "from_bank", "from_account",
            "to_bank", "to_account", "pattern_type", LABEL_COL]
    return pq.read_table(FEATURE_DIR / f"{split}.parquet", columns=cols).to_pandas()


def acct_key(bank: pd.Series, acct: pd.Series) -> pd.Series:
    return bank.astype(str) + ":" + acct.astype(str)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def budget_threshold(score: np.ndarray, frac: float) -> float:
    """告警预算 frac 对应的分数门槛：分数 >= 门槛的交易被告警。"""
    k = max(1, math.ceil(frac * len(score)))
    return float(np.partition(score, len(score) - k)[len(score) - k])


def bootstrap_ap(y: np.ndarray, s: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    """有放回重抽样 n 次，返回 AUC-PR 的 95% 置信区间。"""
    rng = np.random.default_rng(seed)
    aps = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        aps.append(average_precision_score(y[idx], s[idx]))
    return float(np.percentile(aps, 2.5)), float(np.percentile(aps, 97.5))


def fetch_rule_alerts(txn_ids: np.ndarray) -> pd.DataFrame | None:
    """从数据库取测试期交易上的规则告警（按 txn_id 对齐到同一批交易）。"""
    try:
        from db import connect
    except Exception as e:
        print(f"  （无法导入数据库连接：{e}）")
        return None
    lo, hi = int(txn_ids.min()), int(txn_ids.max())
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT txn_id, rule_code FROM rule_alerts "
                        "WHERE txn_id BETWEEN %s AND %s", (lo, hi))
            df = pd.DataFrame(cur.fetchall(), columns=["txn_id", "rule_code"])
    except Exception as e:
        print(f"  （读取规则告警失败：{e}）")
        return None
    return df[df["txn_id"].isin(set(txn_ids.tolist()))]


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="测试集评测（只运行一次）")
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = REPORT_DIR / ("m5_test.json" if args.model == "m4b"
                             else f"m5_test_{args.model}.json")
    if out_path.exists():
        print(f"注意：{out_path} 已存在，说明测试集已经评测过一次。")
        print("      如果这是为了换模型重跑，那测试集就不再是干净的了。继续请输入 yes：")
        if input().strip().lower() != "yes":
            raise SystemExit("已取消。")

    booster, meta = load_model(args.model)
    num, cat = meta["features_num"], meta["features_cat"]
    cols = num + cat
    neg_rate = meta["neg_rate"]
    print(f"模型：{MODELS[args.model][0]}（{args.model}），特征 {len(cols)} 个，"
          f"训练时负样本保留 {neg_rate}")

    # ---------------- 读数据 ----------------
    t0 = time.perf_counter()
    X_va, y_va = load_split("val", cols, cat, None, args.seed)
    X_te, y_te = load_split("test", cols, cat, None, args.seed)
    m_te = load_meta_cols("test")
    assert (m_te[LABEL_COL].to_numpy() == y_te).all(), "元数据与特征行顺序不一致"
    print(f"验证集 {len(y_va):,} 行（洗钱 {int(y_va.sum())}），"
          f"测试集 {len(y_te):,} 行（洗钱 {int(y_te.sum())}），读取 {time.perf_counter() - t0:.1f}s")

    s_va = booster.predict(X_va)
    s_te = booster.predict(X_te)
    report = {"model": args.model, "model_file": MODELS[args.model][0]}

    # ======================================================================
    # 1. 概率校准
    # ======================================================================
    print("\n" + "=" * 64)
    print("1. 概率校准：验证集上拟合 isotonic，测试集上检验")
    print("=" * 64)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(s_va, y_va)
    p_te_iso = iso.predict(s_te)
    p_te_formula = correct_prob(s_te, neg_rate)

    print(f"  测试集实际洗钱比例      {y_te.mean():.4%}")
    print(f"  原始分数平均            {s_te.mean():.4%}   （下采样导致偏高）")
    print(f"  公式校正后平均          {p_te_formula.mean():.4%}   （M3/M4 的方法）")
    print(f"  isotonic 校准后平均     {p_te_iso.mean():.4%}")

    bins = [0, 0.001, 0.01, 0.05, 0.2, 0.5, 1.0001]
    labels = ["<0.1%", "0.1–1%", "1–5%", "5–20%", "20–50%", "≥50%"]
    cut = pd.cut(p_te_iso, bins=bins, labels=labels, right=False)
    rel = (pd.DataFrame({"bin": cut, "p": p_te_iso, "y": y_te})
             .groupby("bin", observed=False)
             .agg(n=("y", "size"), mean_p=("p", "mean"), rate=("y", "mean")))
    print("\n  --- 可靠性表：校准后的概率 vs 实际发生率（两列越接近越好）---")
    print(f"  {'校准概率区间':<12}{'交易数':>10}{'平均预测':>12}{'实际比例':>12}")
    for b, r in rel.iterrows():
        if r["n"] > 0:
            print(f"  {b:<12}{int(r['n']):>10,}{r['mean_p']:>12.2%}{r['rate']:>12.2%}")
    report["calibration"] = {
        "actual_rate": float(y_te.mean()),
        "mean_formula": float(p_te_formula.mean()),
        "mean_isotonic": float(p_te_iso.mean()),
        "reliability": [{"bin": str(b), "n": int(r["n"]),
                         "mean_pred": float(r["mean_p"]) if r["n"] else None,
                         "actual": float(r["rate"]) if r["n"] else None}
                        for b, r in rel.iterrows()],
    }
    joblib.dump(iso, MODEL_DIR / f"calibrator_{args.model}.joblib")

    # ======================================================================
    # 2. 测试集主指标
    # ======================================================================
    print("\n" + "=" * 64)
    print("2. 测试集主指标")
    print("=" * 64)
    m_test = evaluate(y_te, s_te, neg_rate)
    print_report(m_test, neg_rate, split_name="测试集", rough_rule_compare=False)
    t1 = time.perf_counter()
    lo, hi = bootstrap_ap(y_te, s_te, args.n_boot, args.seed)
    print(f"\n  AUC-PR 95% 置信区间（bootstrap {args.n_boot} 次）：[{lo:.4f}, {hi:.4f}]"
          f"   用时 {time.perf_counter() - t1:.0f}s")
    m_test["auc_pr_ci95"] = [lo, hi]
    report["test"] = m_test

    # 验证集 vs 测试集：差距反映验证集上的乐观偏差 + 分布变化
    m_val = evaluate(y_va, s_va, neg_rate)
    print(f"\n  验证集 AUC-PR {m_val['auc_pr']:.4f}  ->  测试集 {m_test['auc_pr']:.4f}"
          f"   （变化 {m_test['auc_pr'] - m_val['auc_pr']:+.4f}）")
    report["val_auc_pr"] = m_val["auc_pr"]

    # 逻辑回归：事先确定的 baseline，一起报告
    lr_path = MODEL_DIR / "lr_baseline.joblib"
    if lr_path.exists():
        lr = joblib.load(lr_path)
        lr_cols = lr["features_num"] + lr["features_cat"]
        # 逻辑回归有自己的特征集合（可能包含被评测模型没用的列），单独读取
        X_lr, _ = load_split("test", lr_cols, lr["features_cat"], None, args.seed)
        for c in lr["features_cat"]:
            X_lr[c] = X_lr[c].astype(str)
        s_lr = lr["pipeline"].predict_proba(X_lr)[:, 1]
        ap_lr = average_precision_score(y_te, s_lr)
        print(f"  逻辑回归 baseline 测试集 AUC-PR {ap_lr:.4f}")
        report["lr_test_auc_pr"] = float(ap_lr)

    # ======================================================================
    # 3. 规则引擎：同一批测试交易
    # ======================================================================
    print("\n" + "=" * 64)
    print("3. 规则引擎 vs 模型（同一批 9/9 ~ 9/10 交易）")
    print("=" * 64)
    txn_ids = m_te["txn_id"].to_numpy()
    rules = fetch_rule_alerts(txn_ids)
    if rules is not None and len(rules):
        is_pos = pd.Series(y_te, index=txn_ids)
        print(f"  {'规则':<8}{'告警数':>10}{'命中':>8}{'精确率':>10}{'召回率':>10}")
        per_rule = {}
        for code, g in rules.groupby("rule_code"):
            ids = g["txn_id"].unique()
            tp = int(is_pos.loc[ids].sum())
            per_rule[code] = {"alerts": len(ids), "tp": tp}
            print(f"  {code:<8}{len(ids):>10,}{tp:>8,}{tp / len(ids):>10.3%}"
                  f"{tp / y_te.sum():>10.2%}")
        union = rules["txn_id"].unique()
        tp_rule = int(is_pos.loc[union].sum())
        n_rule = len(union)
        print(f"  {'合计':<8}{n_rule:>10,}{tp_rule:>8,}{tp_rule / n_rule:>10.3%}"
              f"{tp_rule / y_te.sum():>10.2%}")

        # 同告警量：模型也只取分数最高的 n_rule 条
        top = np.argpartition(-s_te, n_rule - 1)[:n_rule]
        tp_model_same = int(y_te[top].sum())
        print(f"\n  同样打 {n_rule:,} 条告警：规则抓到 {tp_rule}，模型抓到 {tp_model_same}")

        # 同召回：模型要达到规则的召回，需要多少条告警
        order = np.argsort(-s_te)
        cum_tp = np.cumsum(y_te[order])
        need = int(np.searchsorted(cum_tp, tp_rule) + 1)
        if need <= n_rule:
            cmp = f"模型只需要 {need:,} 条（少 {1 - need / n_rule:.1%}）"
        else:
            cmp = f"模型需要 {need:,} 条（多 {need / n_rule - 1:.1%}）"
        print(f"  达到规则同样的召回（{tp_rule} 笔）：规则需要 {n_rule:,} 条告警，{cmp}")
        report["rules"] = {"per_rule": per_rule, "alerts": n_rule, "tp": tp_rule,
                           "precision": tp_rule / n_rule, "recall": tp_rule / y_te.sum(),
                           "model_tp_same_volume": tp_model_same,
                           "model_alerts_same_recall": need}
        rule_hit = set(union.tolist())
    else:
        print("  （跳过：未取到规则告警。确认数据库已启动且 006 已运行）")
        rule_hit = None

    # ======================================================================
    # 按预算门槛拆分：4 / 5 / 6 共用
    # ======================================================================
    thr = {f: budget_threshold(s_te, f) for f in BUDGETS}

    def recall_row(mask: np.ndarray) -> list[float]:
        pos = mask & (y_te == 1)
        n = int(pos.sum())
        return [n] + [float((s_te[pos] >= thr[f]).mean()) if n else float("nan")
                      for f in BUDGETS]

    head = f"{'':<22}{'洗钱笔数':>8}" + "".join(f"{'前' + format(f, '.1%') + '召回':>12}"
                                             for f in BUDGETS)

    # ======================================================================
    # 4. 按天
    # ======================================================================
    print("\n" + "=" * 64)
    print("4. 按天拆分（9/9 有分布偏移：多出十几万个平时不活跃的账户）")
    print("=" * 64)
    day = pd.to_datetime(m_te["txn_ts"]).dt.date.astype(str).to_numpy()
    print(f"  {'日期':<10}{'交易数':>10}{'洗钱':>6}{'AUC-PR':>10}{'AUC-ROC':>10}")
    report["by_day"] = {}
    for d in sorted(set(day)):
        mk = day == d
        a = average_precision_score(y_te[mk], s_te[mk])
        r = roc_auc_score(y_te[mk], s_te[mk])
        print(f"  {d:<10}{int(mk.sum()):>10,}{int(y_te[mk].sum()):>6}{a:>10.4f}{r:>10.4f}")
        report["by_day"][d] = {"n": int(mk.sum()), "pos": int(y_te[mk].sum()),
                               "auc_pr": float(a), "auc_roc": float(r)}

    # ======================================================================
    # 5. 洗钱账户在训练期是否出现过
    # ======================================================================
    print("\n" + "=" * 64)
    print("5. 洗钱账户在训练期是否出现过（'未出现过' 才是模型在新面孔上的真实水平）")
    print("=" * 64)
    tr = pq.read_table(FEATURE_DIR / "train.parquet",
                       columns=["from_bank", "from_account", "to_bank", "to_account",
                                LABEL_COL]).to_pandas()
    tr = tr[tr[LABEL_COL] == 1]
    seen_accts = set(acct_key(tr["from_bank"], tr["from_account"])) | \
                 set(acct_key(tr["to_bank"], tr["to_account"]))
    seen = (acct_key(m_te["from_bank"], m_te["from_account"]).isin(seen_accts) |
            acct_key(m_te["to_bank"], m_te["to_account"]).isin(seen_accts)).to_numpy()
    print(f"  训练期出现过的洗钱账户：{len(seen_accts):,} 个")
    print("  " + head)
    report["seen_vs_unseen"] = {}
    for name, mk in [("收付方出现过", seen), ("收付方都没出现过", ~seen)]:
        row = recall_row(mk)
        print(f"  {name:<20}{row[0]:>8}" + "".join(f"{v:>12.1%}" for v in row[1:]))
        report["seen_vs_unseen"][name] = row

    # ======================================================================
    # 6. 按洗钱手法
    # ======================================================================
    print("\n" + "=" * 64)
    print("6. 按洗钱手法：模型（按预算）vs 规则引擎")
    print("=" * 64)
    pt = m_te["pattern_type"].fillna("（无模式标签）").to_numpy()
    rule_col = f"{'规则召回':>10}" if rule_hit is not None else ""
    print("  " + head + rule_col)
    report["by_pattern"] = {}
    for p in sorted(set(pt[y_te == 1])):
        mk = pt == p
        row = recall_row(mk)
        line = f"  {p:<22}{row[0]:>8}" + "".join(f"{v:>12.1%}" for v in row[1:])
        entry = {"n": row[0], **{f"top_{f:.1%}": v for f, v in zip(BUDGETS, row[1:])}}
        if rule_hit is not None:
            pos_ids = txn_ids[mk & (y_te == 1)]
            rr = float(np.mean([i in rule_hit for i in pos_ids])) if len(pos_ids) else float("nan")
            line += f"{rr:>10.1%}"
            entry["rule_recall"] = rr
        print(line)
        report["by_pattern"][p] = entry
    print("\n  规则引擎打了几十万条告警才有上面的召回；模型只用了 0.1% / 0.5% / 1% 的告警量。")

    # ---------------- 保存 ----------------
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n报告已保存：{out_path}")
    print(f"校准器已保存：{MODEL_DIR / f'calibrator_{args.model}.joblib'}")


if __name__ == "__main__":
    main()