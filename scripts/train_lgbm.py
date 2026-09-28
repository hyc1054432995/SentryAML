"""
SentryAML - M4：LightGBM
========================
读取 data/features/{train,val}.parquet，训练梯度提升树，在【验证集】上评估，
并与 M3 的逻辑回归结果并排比较。测试集这一步仍然一行都不碰。

与逻辑回归的区别：
  - 不取对数、不标准化：树只比较"大于 / 小于某个切分点"，单调变换不影响任何切分
  - 分类特征直接交给 LightGBM 处理，不用 one-hot
  - 用【早停】决定树的棵数：每加一棵就在验证集上看 AUC-PR，
    连续 --patience 棵没有提升就停下

用法：
    python scripts/train_lgbm.py
    python scripts/train_lgbm.py --neg-rate 1.0        # 用全部训练数据（更吃内存）
    python scripts/train_lgbm.py --num-leaves 127 --min-data-in-leaf 100
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

from eval_utils import compare, evaluate, print_report
from feature_spec import LABEL_COL, model_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


def load_split(split: str, cols: list[str], cat: list[str],
               neg_rate: float | None, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    table = pq.read_table(FEATURE_DIR / f"{split}.parquet", columns=cols + [LABEL_COL])

    if neg_rate is not None and neg_rate < 1.0:
        rng = np.random.default_rng(seed)
        y = table[LABEL_COL].to_numpy()
        table = table.filter((y == 1) | (rng.random(len(y)) < neg_rate))

    df = table.to_pandas()
    y = df.pop(LABEL_COL).to_numpy()
    # LightGBM 原生支持 category 类型：内部按类别分组做切分，不需要 one-hot
    for c in cat:
        df[c] = df[c].astype("category")
    return df[cols], y


def main():
    p = argparse.ArgumentParser(description="训练 LightGBM")
    p.add_argument("--neg-rate", type=float, default=0.2, help="训练集负样本保留比例")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=63)
    p.add_argument("--min-data-in-leaf", type=int, default=200)
    p.add_argument("--feature-fraction", type=float, default=0.8)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--max-rounds", type=int, default=3000)
    p.add_argument("--patience", type=int, default=100)
    args = p.parse_args()

    num, cat = model_features(exclude_known_leaky=True)
    cols = num + cat
    print(f"特征：数值 {len(num)} 个 + 分类 {len(cat)} 个（不取对数、不标准化）")
    print("已排除的已知泄漏特征：f_payment_format")

    t0 = time.perf_counter()
    X_tr, y_tr = load_split("train", cols, cat, args.neg_rate, args.seed)
    X_va, y_va = load_split("val", cols, cat, None, args.seed)     # 验证集不下采样
    print(f"\n训练集：{len(y_tr):,} 行，洗钱 {int(y_tr.sum()):,}，"
          f"正负比约 1:{(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1):.0f}")
    print(f"验证集：{len(y_va):,} 行，洗钱 {int(y_va.sum()):,}")
    print(f"读取用时 {time.perf_counter() - t0:.1f}s  "
          f"训练集内存 {X_tr.memory_usage(deep=True).sum() / 1024**2:.0f} MB")

    params = {
        "objective": "binary",
        "metric": "average_precision",     # 早停直接看 AUC-PR，与我们的评价指标一致
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": 1,
        "seed": args.seed,
        "num_threads": 0,                  # 0 = 用满可用核数
        "verbosity": -1,
    }

    dtrain = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat, free_raw_data=False)
    dval = lgb.Dataset(X_va, label=y_va, categorical_feature=cat, reference=dtrain)

    print("\n--- 训练（早停：验证集 AUC-PR 连续 "
          f"{args.patience} 轮无提升即停）---")
    t1 = time.perf_counter()
    evals = {}
    booster = lgb.train(
        params, dtrain,
        num_boost_round=args.max_rounds,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(args.patience, verbose=False),
            lgb.log_evaluation(period=100),
            lgb.record_evaluation(evals),
        ],
    )
    train_time = time.perf_counter() - t1
    print(f"\n  最佳轮数 {booster.best_iteration}（上限 {args.max_rounds}），"
          f"训练用时 {train_time:.1f}s")

    # 训练集与验证集的 AUC-PR 差距：差得越大，过拟合越明显
    ap_tr = evals["train"]["average_precision"][booster.best_iteration - 1]
    ap_va = evals["val"]["average_precision"][booster.best_iteration - 1]
    print(f"  最佳轮处 AUC-PR：训练集 {ap_tr:.4f}   验证集 {ap_va:.4f}   "
          f"（差距 {ap_tr - ap_va:+.4f}）")

    # ---------------- 评测 ----------------
    p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    print("\n" + "=" * 60)
    print("LightGBM 在验证集上的表现")
    print("=" * 60)
    metrics = evaluate(y_va, p_va, args.neg_rate)
    print_report(metrics, args.neg_rate)

    # ---------------- 特征重要性 ----------------
    # gain：该特征参与的所有切分带来的总增益。比 split（切分次数）更能反映实际贡献
    gain = booster.feature_importance(importance_type="gain")
    imp = sorted(zip(booster.feature_name(), gain), key=lambda kv: -kv[1])
    total = sum(gain) or 1
    print("\n  --- 特征重要性（按 gain，前 15）---")
    print(f"  {'特征':<28}{'gain 占比':>12}")
    for name, g in imp[:15]:
        print(f"  {name:<28}{g / total:>11.1%}")

    # ---------------- 与逻辑回归比较 ----------------
    lr_path = REPORT_DIR / "m3_lr_val.json"
    if lr_path.exists():
        with open(lr_path, encoding="utf-8") as f:
            lr = json.load(f)
        print("\n  --- 与 M3 逻辑回归对比（同一验证集）---")
        compare("逻辑回归", lr["val_metrics"], "LightGBM", metrics)
    else:
        print(f"\n  （未找到 {lr_path}，跳过与逻辑回归的对比）")

    # ---------------- 保存 ----------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_DIR / "lgbm.txt"), num_iteration=booster.best_iteration)
    joblib.dump({"features_num": num, "features_cat": cat,
                 "neg_rate": args.neg_rate, "params": params,
                 "best_iteration": booster.best_iteration},
                MODEL_DIR / "lgbm_meta.joblib")
    with open(REPORT_DIR / "m4_lgbm_val.json", "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "params": params,
                   "best_iteration": booster.best_iteration,
                   "train_ap_at_best": ap_tr, "val_ap_at_best": ap_va,
                   "train_seconds": train_time,
                   "val_metrics": metrics,
                   "importance_gain": [{"feature": n, "gain": float(g),
                                        "share": float(g / total)} for n, g in imp]},
                  f, ensure_ascii=False, indent=2)
    print(f"\n模型已保存：{MODEL_DIR / 'lgbm.txt'}")
    print(f"报告已保存：{REPORT_DIR / 'm4_lgbm_val.json'}")


if __name__ == "__main__":
    main()