"""
SentryAML - M3 第二步：逻辑回归 baseline
========================================
读取 data/features/{train,val}.parquet，训练逻辑回归，在【验证集】上评估。
测试集这一步一行都不碰，留到最后只用一次。

流程：
  1. 训练集负样本下采样（保留全部洗钱样本 + 5% 正常样本）
  2. Pipeline：计数与金额取 log1p -> 标准化 -> 分类特征 one-hot -> 逻辑回归
     预处理放进 Pipeline，保证：
       - 均值 / 标准差只用训练集计算（不把验证集信息混进训练）
       - 上线时对新数据自动执行完全相同的变换（避免 training-serving skew）
  3. 在验证集上比较一组正则化强度 C，选 AUC-PR 最好的
  4. 概率校正（抵消下采样带来的偏高），并验证校正是否正确
  5. 报告：AUC-PR、AUC-ROC、告警预算下的召回与精确率、系数解释

用法：
    python scripts/train_lr.py
    python scripts/train_lr.py --neg-rate 0.1 --seed 7
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from eval_utils import evaluate, print_report
from feature_spec import LABEL_COL, model_features

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

# 0/1 标志：不取对数（log1p(0/1) 没有意义）
BINARY_FEATURES = {"f_cross_currency", "f_intra_branch", "f_same_entity"}
# 已经是对数的特征
ALREADY_LOG = {"f_log_amount"}

# ---------------------------------------------------------------------------
# 读数据
# ---------------------------------------------------------------------------
def load_split(split: str, cols: list[str], neg_rate: float | None, seed: int) -> pd.DataFrame:
    """
    只读需要的列（列式存储的好处）。
    neg_rate 不为 None 时：保留全部正样本，负样本按 neg_rate 随机保留。
    过滤在 pyarrow 层完成，避免先把 180 万行全部转成 pandas。
    """
    table = pq.read_table(FEATURE_DIR / f"{split}.parquet", columns=cols + [LABEL_COL])

    if neg_rate is not None:
        rng = np.random.default_rng(seed)                     # 固定种子，结果可复现
        y = table[LABEL_COL].to_numpy()
        keep = (y == 1) | (rng.random(len(y)) < neg_rate)
        table = table.filter(keep)

    df = table.to_pandas()
    return df


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
def build_pipeline(num_log: list[str], num_plain: list[str], cat: list[str], C: float) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            # 重尾的计数和金额：先 log1p，再标准化
            ("log", Pipeline([
                ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]), num_log),
            # 0/1 标志和已取对数的金额：只标准化
            ("plain", StandardScaler(), num_plain),
            # 分类特征：one-hot；验证集 / 上线时遇到训练集没见过的类别，编码为全 0
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
        ],
        verbose_feature_names_out=False,
    )
    clf = LogisticRegression(C=C, max_iter=3000)   # 默认 L2 正则
    return Pipeline([("pre", pre), ("clf", clf)])


def print_coefficients(pipe: Pipeline, top_n: int = 10) -> list[dict]:
    """
    所有数值特征都已标准化，所以系数可以直接比较：
    特征增加 1 个标准差，洗钱赔率乘以 e^系数。
    """
    names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pipe.named_steps["clf"].coef_.ravel()
    order = np.argsort(coefs)

    rows = [{"feature": str(names[i]), "coef": float(coefs[i]),
             "odds_ratio_per_std": float(np.exp(coefs[i]))} for i in order]

    print(f"\n  --- 系数：让洗钱赔率上升最多的 {top_n} 个特征 ---")
    print(f"  {'特征':<28}{'系数':>9}{'赔率倍数/标准差':>18}")
    for r in reversed(rows[-top_n:]):
        print(f"  {r['feature']:<28}{r['coef']:>9.3f}{r['odds_ratio_per_std']:>18.2f}")

    print(f"\n  --- 系数：让洗钱赔率下降最多的 {top_n} 个特征 ---")
    for r in rows[:top_n]:
        print(f"  {r['feature']:<28}{r['coef']:>9.3f}{r['odds_ratio_per_std']:>18.2f}")
    return rows


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="训练逻辑回归 baseline")
    parser.add_argument("--neg-rate", type=float, default=0.05, help="训练集负样本保留比例")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-grid", type=float, nargs="+", default=[0.001, 0.01, 0.1, 1.0, 10.0])
    args = parser.parse_args()

    num, cat = model_features(exclude_known_leaky=True)
    num_log = [c for c in num if c not in BINARY_FEATURES | ALREADY_LOG]
    num_plain = [c for c in num if c in BINARY_FEATURES | ALREADY_LOG]
    cols = num + cat
    print(f"特征：数值 {len(num)} 个（其中取对数 {len(num_log)} 个），分类 {len(cat)} 个")
    print("已排除的已知泄漏特征：f_payment_format")

    # ---------------- 读数据 ----------------
    t0 = time.perf_counter()
    train = load_split("train", cols, neg_rate=args.neg_rate, seed=args.seed)
    val = load_split("val", cols, neg_rate=None, seed=args.seed)       # 验证集不下采样
    for df in (train, val):
        for c in cat:
            df[c] = df[c].astype(str)
    X_tr, y_tr = train[cols], train[LABEL_COL].to_numpy()
    X_va, y_va = val[cols], val[LABEL_COL].to_numpy()
    print(f"\n训练集（下采样后）：{len(y_tr):,} 行，洗钱 {int(y_tr.sum()):,}，"
          f"正负比约 1:{(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1):.0f}")
    print(f"验证集（原样）：    {len(y_va):,} 行，洗钱 {int(y_va.sum()):,}")
    print(f"读取用时 {time.perf_counter() - t0:.1f}s")

    # ---------------- 在验证集上选 C ----------------
    print("\n--- 正则化强度 C 的选择（在验证集上比较）---")
    print(f"  {'C':>8}{'AUC-PR':>10}{'AUC-ROC':>10}{'训练用时':>10}")
    results = []
    best = None
    for C in args.c_grid:
        t1 = time.perf_counter()
        pipe = build_pipeline(num_log, num_plain, cat, C)
        pipe.fit(X_tr, y_tr)
        p = pipe.predict_proba(X_va)[:, 1]
        ap = average_precision_score(y_va, p)
        auc = roc_auc_score(y_va, p)
        dt = time.perf_counter() - t1
        results.append({"C": C, "auc_pr": float(ap), "auc_roc": float(auc)})
        print(f"  {C:>8}{ap:>10.4f}{auc:>10.4f}{dt:>9.1f}s")
        if best is None or ap > best[0]:
            best = (ap, C, pipe, p)

    _, best_C, best_pipe, p_va = best
    print(f"\n  选定 C = {best_C}")

    # ---------------- 报告 ----------------
    print("\n" + "=" * 60)
    print(f"逻辑回归 baseline（C = {best_C}）在验证集上的表现")
    print("=" * 60)
    metrics = evaluate(y_va, p_va, args.neg_rate)
    print_report(metrics, args.neg_rate)
    coef_rows = print_coefficients(best_pipe)

    # ---------------- 保存 ----------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": best_pipe, "neg_rate": args.neg_rate,
                 "features_num": num, "features_cat": cat},
                MODEL_DIR / "lr_baseline.joblib")
    with open(REPORT_DIR / "m3_lr_val.json", "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "best_C": best_C, "c_search": results,
                   "val_metrics": metrics, "coefficients": coef_rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\n模型已保存：{MODEL_DIR / 'lr_baseline.joblib'}")
    print(f"报告已保存：{REPORT_DIR / 'm3_lr_val.json'}")


if __name__ == "__main__":
    main()