"""
SentryAML - M6b：泄漏审计（消融实验）
=====================================
问题：模型的成绩里，有多少来自"数据生成器的痕迹"，而不是可迁移到真实世界的规律？

两个嫌疑特征：
  f_payment_format  支付方式。洗钱 86.6% 是 ACH，电汇 0 笔（D07），主模型已排除
  f_pair_prev_cnt   双方此前来往笔数。首次来往在 SHAP 里一家独大（D33），
                    而正常交易 87% 在 7 天内已来往 8 次以上，不像真实行为

做法：M4b 的参数全部固定，只改变特征集合，在同一份训练数据上重新训练，
     在【验证集】上比较。每组只改一件事，便于归因（D27 的教训）。

  A  基准：M4b 的 49 个特征（重训一次，同时检验可复现性）
  B  只用支付方式                  它单独能走多远
  C  基准 + 支付方式                加回泄漏特征，成绩会虚高多少
  D  只用"双方此前来往笔数"          首次来往单独能走多远
  E  基准 − 双方此前来往笔数         去掉它，其他特征能补回多少
  F  基准 − 同向和反向来往笔数       连同高度相关的反向来往一起去掉

用法：
    python scripts/leak_audit.py
    python scripts/leak_audit.py --only A E F       # 只跑其中几组
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from eval_utils import RULE_PRECISION, RULE_RECALL, evaluate
from feature_spec import CAT_FEATURES, LABEL_COL, NUM_FEATURES, model_features
from model_explain import describe, load_booster
from train_lgbm import load_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

PF = "f_payment_format"
PAIR = "f_pair_prev_cnt"
PAIR_REV = "f_pair_rev_prev_cnt"


def variants(base: list[str]) -> dict[str, tuple[str, list[str]]]:
    return {
        "A": ("基准（M4b 特征）",          base),
        "B": ("只用支付方式",              [PF]),
        "C": ("基准 + 支付方式",           base + [PF]),
        "D": ("只用双方此前来往笔数",       [PAIR]),
        "E": ("基准 − 此前来往笔数",        [c for c in base if c != PAIR]),
        "F": ("基准 − 同向与反向来往",      [c for c in base if c not in (PAIR, PAIR_REV)]),
    }


def train_one(params, cols, cat_all, X_tr, y_tr, X_va, y_va, max_rounds, patience):
    cat = [c for c in cols if c in cat_all]
    dtr = lgb.Dataset(X_tr[cols], label=y_tr, categorical_feature=cat, free_raw_data=False)
    dva = lgb.Dataset(X_va[cols], label=y_va, categorical_feature=cat, reference=dtr)
    evals = {}
    booster = lgb.train(params, dtr, num_boost_round=max_rounds,
                        valid_sets=[dtr, dva], valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(patience, verbose=False),
                                   lgb.record_evaluation(evals)])
    best = booster.best_iteration
    return booster, {"best_iteration": best,
                     "train_ap": evals["train"]["average_precision"][best - 1],
                     "val_ap": evals["val"]["average_precision"][best - 1]}


def recall_at(score, mask, frac):
    """前 frac 预算下对 mask 这组洗钱的召回；并列按期望值计算（见 eval_utils.alert_budget）。"""
    k = max(1, int(np.ceil(frac * len(score))))
    thr = np.partition(score, len(score) - k)[len(score) - k]
    above, at = score > thr, score == thr
    need = k - int(above.sum())
    hit = float((above & mask).sum()) + need * float((at & mask).sum()) / max(int(at.sum()), 1)
    return hit / max(int(mask.sum()), 1)


def main():
    ap = argparse.ArgumentParser(description="M6b：泄漏审计")
    ap.add_argument("--only", nargs="+", choices=list("ABCDEF"), default=list("ABCDEF"))
    ap.add_argument("--max-rounds", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=100)
    args = ap.parse_args()

    # ---------------- M4b 的参数与特征 ----------------
    _, meta = load_booster("m4b")
    params = dict(meta["params"])
    neg_rate, seed = meta["neg_rate"], params.get("seed", 42)
    base_num, base_cat = model_features(exclude_known_leaky=True)
    base = base_num + base_cat
    all_cols = NUM_FEATURES + CAT_FEATURES            # 含支付方式
    print(f"M4b 参数：叶子 {params['num_leaves']}，每叶至少 {params['min_data_in_leaf']}，"
          f"学习率 {params['learning_rate']}，L2 {params['lambda_l2']}；负样本保留 {neg_rate}")

    # ---------------- 一次读入全部列，各组只取子集，保证训练行完全相同 ----------------
    t0 = time.perf_counter()
    X_tr, y_tr = load_split("train", all_cols, CAT_FEATURES, neg_rate, seed)
    X_va, y_va = load_split("val", all_cols, CAT_FEATURES, None, seed)
    tagged = pq.read_table(FEATURE_DIR / "val.parquet",
                           columns=["pattern_type"]).column(0).is_valid().to_numpy(zero_copy_only=False)
    pos_tag, pos_untag = (y_va == 1) & tagged, (y_va == 1) & ~tagged
    print(f"训练 {len(y_tr):,} 行（洗钱 {int(y_tr.sum()):,}），验证 {len(y_va):,} 行"
          f"（洗钱 {int(y_va.sum())}：有标签 {int(pos_tag.sum())}，无标签 {int(pos_untag.sum())}），"
          f"读取 {time.perf_counter() - t0:.1f}s")

    # ======================================================================
    # 0. 直觉：两条"单特征规则"在验证集上的表现
    # ======================================================================
    print("\n" + "=" * 78)
    print("0. 单特征规则（不训练模型，直接用一个条件报警）")
    print("=" * 78)
    rules = {"支付方式 = ACH": (X_va[PF].astype(str) == "ACH").to_numpy(),
             "首次来往（此前来往 0 笔）": (X_va[PAIR] == 0).to_numpy()}
    print(f"  {'规则':<28}{'告警占交易':>12}{'召回':>10}{'精确率':>10}{'精确率/基础比例':>16}")
    rule_rows = {}
    for name, m in rules.items():
        rec = float((m & (y_va == 1)).sum() / y_va.sum())
        prec = float((m & (y_va == 1)).sum() / max(m.sum(), 1))
        lift = prec / y_va.mean()
        rule_rows[name] = {"alert_share": float(m.mean()), "recall": rec,
                           "precision": prec, "lift": lift}
        print(f"  {name:<28}{m.mean():>12.1%}{rec:>10.1%}{prec:>10.2%}{lift:>16.1f}")
    print(f"  （对照：7 条规则合计召回 {RULE_RECALL:.1%}，精确率 {RULE_PRECISION:.2%}）")

    # ======================================================================
    # 1. 消融实验
    # ======================================================================
    print("\n" + "=" * 78)
    print("1. 消融实验：固定 M4b 参数，只改变特征集合")
    print("=" * 78)
    V = variants(base)
    results, boosters = {}, {}
    for key in args.only:
        name, cols = V[key]
        t1 = time.perf_counter()
        booster, r = train_one(params, cols, CAT_FEATURES, X_tr, y_tr, X_va, y_va,
                               args.max_rounds, args.patience)
        p = booster.predict(X_va[cols], num_iteration=booster.best_iteration)
        m = evaluate(y_va, p, neg_rate)
        r.update({
            "name": name, "n_features": len(cols),
            "auc_roc": m["auc_roc"],
            "precision_at_rule_recall": m["precision_at_rule_recall"],
            "budget": m["budget"],
            "recall_tagged_1pct": recall_at(p, pos_tag, 0.01),
            "recall_untagged_1pct": recall_at(p, pos_untag, 0.01),
            "seconds": time.perf_counter() - t1,
        })
        gain = booster.feature_importance("gain")
        order = np.argsort(gain)[::-1][:5]
        r["top_gain"] = [{"feature": cols[i], "share": float(gain[i] / gain.sum())} for i in order]
        results[key], boosters[key] = r, booster
        print(f"  {key} {name:<22}完成：{r['best_iteration']} 轮，验证 AP {r['val_ap']:.4f}，"
              f"用时 {r['seconds']:.0f}s")

    # ---------------- 汇总表 ----------------
    ref = results.get("A", {}).get("val_ap")
    print(f"\n  {'组':<3}{'特征集合':<22}{'特征数':>6}{'轮数':>6}{'训练AP':>8}{'验证AP':>8}"
          f"{'相对A':>8}{'ROC':>7}{'前0.1%召回':>11}{'前1%召回':>9}{'规则召回下精确率':>16}")
    for key, r in results.items():
        delta = f"{r['val_ap'] - ref:+.4f}" if ref is not None else "-"
        print(f"  {key:<3}{r['name']:<22}{r['n_features']:>6}{r['best_iteration']:>6}"
              f"{r['train_ap']:>8.4f}{r['val_ap']:>8.4f}{delta:>8}{r['auc_roc']:>7.4f}"
              f"{r['budget']['top_0.1%']['recall']:>11.1%}{r['budget']['top_1.0%']['recall']:>9.1%}"
              f"{r['precision_at_rule_recall']:>16.2%}")
    print(f"  （M4b 原始验证 AP 0.5998；相同参数重训的噪声约 0.003，见 D28）")

    # ---------------- 有 / 无模式标签 ----------------
    print(f"\n  前 1% 预算下的召回，按洗钱是否有模式标签拆开：")
    print(f"  {'组':<3}{'特征集合':<22}{'有标签':>10}{'无标签':>10}")
    for key, r in results.items():
        print(f"  {key:<3}{r['name']:<22}{r['recall_tagged_1pct']:>10.1%}{r['recall_untagged_1pct']:>10.1%}")

    # ---------------- 去掉之后，谁顶替了它 ----------------
    for key in ("A", "E", "F"):
        if key in results:
            print(f"\n  {key}（{results[key]['name']}）增益前 5：")
            for t in results[key]["top_gain"]:
                print(f"      {describe(t['feature'])[:26]:<30}{t['share']:>7.1%}")

    # ======================================================================
    # 保存
    # ======================================================================
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    for key in ("E", "F"):                            # 保守版本留着，D24 预先登记过"有/无"两个版本
        if key in boosters:
            cols = V[key][1]
            boosters[key].save_model(str(MODEL_DIR / f"lgbm_ablate_{key}.txt"),
                                     num_iteration=boosters[key].best_iteration)
            joblib.dump({"features_num": [c for c in cols if c not in CAT_FEATURES],
                         "features_cat": [c for c in cols if c in CAT_FEATURES],
                         "neg_rate": neg_rate, "params": params, "config_name": V[key][0],
                         "best_iteration": boosters[key].best_iteration},
                        MODEL_DIR / f"lgbm_ablate_{key}_meta.joblib")
    out = REPORT_DIR / "m6b_leak_audit.json"
    merged = {}
    if out.exists():                                  # 用 --only 重跑部分组时，保留其他组的结果
        with open(out, encoding="utf-8") as f:
            merged = json.load(f).get("variants", {})
    merged.update(results)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"params": params, "single_feature_rules": rule_rows,
                   "variants": dict(sorted(merged.items()))},
                  f, ensure_ascii=False, indent=2, default=str)
    print(f"\n报告已保存：{out}")


if __name__ == "__main__":
    main()