"""
SentryAML - M4b：LightGBM 参数收紧实验
======================================
M4 的默认参数在验证集上 AUC-PR 0.5849，但训练集 0.9928，差距 0.41，
说明模型容量偏高、在记训练集的细节。这里试几组更保守的配置。

每组配置只改少数几个参数，便于看出是哪一项起了作用：
  num_leaves        每棵树的叶子数上限，越小树越简单
  min_data_in_leaf  每个叶子至少覆盖多少样本，越大越不容易记住个别样本
  learning_rate     每棵树的修正幅度，越小需要越多棵树，但泛化通常更好
  lambda_l2         对叶子输出值的 L2 惩罚，抑制极端分数
  feature_fraction  每棵树随机使用多少比例的特征，降低树之间的相关性

所有配置都用【相同的早停规则】在验证集上比较，最后保存最好的一组。

用法：
    python scripts/sweep_lgbm.py
    python scripts/sweep_lgbm.py --neg-rate 0.2 --patience 100
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import lightgbm as lgb

from eval_utils import compare, evaluate, print_report
from feature_spec import LABEL_COL, model_features
from train_lgbm import load_split          # 复用 M4 的数据读取，保证两边完全一致

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# 每组只动一两个参数，便于归因
CONFIGS = [
    ("M4 默认",          dict(num_leaves=63, min_data_in_leaf=200,  learning_rate=0.05,
                             lambda_l2=0.0,  feature_fraction=0.8)),
    ("叶子减半",         dict(num_leaves=31, min_data_in_leaf=200,  learning_rate=0.05,
                             lambda_l2=0.0,  feature_fraction=0.8)),
    ("叶子很少",         dict(num_leaves=15, min_data_in_leaf=200,  learning_rate=0.05,
                             lambda_l2=0.0,  feature_fraction=0.8)),
    ("叶子样本数加大",    dict(num_leaves=31, min_data_in_leaf=1000, learning_rate=0.05,
                             lambda_l2=0.0,  feature_fraction=0.8)),
    ("加 L2 惩罚",       dict(num_leaves=31, min_data_in_leaf=500,  learning_rate=0.05,
                             lambda_l2=10.0, feature_fraction=0.8)),
    ("学习率减半",       dict(num_leaves=31, min_data_in_leaf=500,  learning_rate=0.025,
                             lambda_l2=10.0, feature_fraction=0.8)),
    ("特征采样更狠",     dict(num_leaves=31, min_data_in_leaf=500,  learning_rate=0.05,
                             lambda_l2=10.0, feature_fraction=0.5)),
    ("全面收紧",         dict(num_leaves=15, min_data_in_leaf=1000, learning_rate=0.03,
                             lambda_l2=20.0, feature_fraction=0.6)),
]


def run_one(cfg: dict, X_tr, y_tr, X_va, y_va, cat, seed, max_rounds, patience):
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "num_threads": 0,
        "verbosity": -1,
        **cfg,
    }
    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, categorical_feature=cat, reference=dtrain)

    evals = {}
    t0 = time.perf_counter()
    booster = lgb.train(
        params, dtrain,
        num_boost_round=max_rounds,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(patience, verbose=False),
                   lgb.record_evaluation(evals)],
    )
    secs = time.perf_counter() - t0
    best = booster.best_iteration
    ap_tr = evals["train"]["average_precision"][best - 1]
    ap_va = evals["val"]["average_precision"][best - 1]
    return booster, params, {"best_iteration": best, "train_ap": ap_tr,
                             "val_ap": ap_va, "gap": ap_tr - ap_va, "seconds": secs}


def main():
    p = argparse.ArgumentParser(description="LightGBM 参数收紧实验")
    p.add_argument("--neg-rate", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rounds", type=int, default=3000)
    p.add_argument("--patience", type=int, default=100)
    args = p.parse_args()

    num, cat = model_features(exclude_known_leaky=True)
    cols = num + cat

    t0 = time.perf_counter()
    X_tr, y_tr = load_split("train", cols, cat, args.neg_rate, args.seed)
    X_va, y_va = load_split("val", cols, cat, None, args.seed)
    print(f"训练集 {len(y_tr):,} 行（洗钱 {int(y_tr.sum()):,}），"
          f"验证集 {len(y_va):,} 行（洗钱 {int(y_va.sum()):,}），"
          f"读取 {time.perf_counter() - t0:.1f}s")

    print(f"\n{'配置':<16}{'叶子':>6}{'叶最小样本':>12}{'学习率':>8}{'L2':>6}"
          f"{'特征比':>8}{'轮数':>7}{'训练AP':>9}{'验证AP':>9}{'差距':>8}{'用时':>8}")
    print("-" * 104)

    results = []
    best = None
    for name, cfg in CONFIGS:
        booster, params, r = run_one(cfg, X_tr, y_tr, X_va, y_va, cat,
                                     args.seed, args.max_rounds, args.patience)
        print(f"{name:<16}{cfg['num_leaves']:>6}{cfg['min_data_in_leaf']:>12}"
              f"{cfg['learning_rate']:>8}{cfg['lambda_l2']:>6}{cfg['feature_fraction']:>8}"
              f"{r['best_iteration']:>7}{r['train_ap']:>9.4f}{r['val_ap']:>9.4f}"
              f"{r['gap']:>8.4f}{r['seconds']:>7.0f}s")
        results.append({"name": name, "config": cfg, **r})
        if best is None or r["val_ap"] > best[0]:
            best = (r["val_ap"], name, booster, params, r)

    _, best_name, best_booster, best_params, best_r = best
    print(f"\n验证集 AUC-PR 最高的配置：{best_name}（{best_r['val_ap']:.4f}，"
          f"差距 {best_r['gap']:.4f}）")

    # ---------------- 最好配置的完整报告 ----------------
    p_va = best_booster.predict(X_va, num_iteration=best_booster.best_iteration)
    print("\n" + "=" * 60)
    print(f"最优配置「{best_name}」在验证集上的表现")
    print("=" * 60)
    metrics = evaluate(y_va, p_va, args.neg_rate)
    print_report(metrics, args.neg_rate)

    m4_path = REPORT_DIR / "m4_lgbm_val.json"
    if m4_path.exists():
        with open(m4_path, encoding="utf-8") as f:
            m4 = json.load(f)
        print("\n  --- 与 M4 默认配置对比 ---")
        compare("M4 默认", m4["val_metrics"], best_name, metrics)

    # ---------------- 保存 ----------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    best_booster.save_model(str(MODEL_DIR / "lgbm_best.txt"),
                            num_iteration=best_booster.best_iteration)
    joblib.dump({"features_num": num, "features_cat": cat, "neg_rate": args.neg_rate,
                 "params": best_params, "config_name": best_name,
                 "best_iteration": best_booster.best_iteration},
                MODEL_DIR / "lgbm_best_meta.joblib")
    with open(REPORT_DIR / "m4b_sweep.json", "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "sweep": results,
                   "best": {"name": best_name, "params": best_params, **best_r},
                   "val_metrics": metrics}, f, ensure_ascii=False, indent=2)
    print(f"\n最优模型已保存：{MODEL_DIR / 'lgbm_best.txt'}")
    print(f"报告已保存：{REPORT_DIR / 'm4b_sweep.json'}")


if __name__ == "__main__":
    main()