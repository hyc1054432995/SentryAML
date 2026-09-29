"""
SentryAML - 模型解释工具（SHAP）
================================
把"模型为什么给这笔交易打高分"拆成每个特征的贡献。

两个用途：
  1. M6 分析：全局重要性、特征方向、误报原因（shap_analysis.py 调用）
  2. 后续 Agent 层：处理每条告警时调用 explain_rows()，把贡献最大的几个特征
     作为"证据"交给 LLM 组织成调查说明。所以这里只放可复用的函数，不放分析逻辑。

实现：LightGBM 自带 TreeSHAP（predict(..., pred_contrib=True)），
     不需要安装 shap 库，结果与 shap.TreeExplainer 相同。

单位：SHAP 值在【对数赔率（log-odds）】空间里，可以直接相加：
    基准值 + 所有特征的 SHAP 值 = 这笔交易的原始分数（log-odds）
"""

import json
import math
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"

MODELS = {
    "m4b":    ("lgbm_best.txt",   "lgbm_best_meta.joblib"),
    "optuna": ("lgbm_optuna.txt", "lgbm_optuna_meta.json"),
}

# ---------------------------------------------------------------------------
# 特征的中文说明：给人看、给 Agent 看
# ---------------------------------------------------------------------------
_SNAP = {
    "out_cnt_1d": "前 1 天转出笔数",   "out_cp_1d": "前 1 天转出对手方数",
    "out_amt_1d": "前 1 天转出金额",   "in_cnt_1d": "前 1 天转入笔数",
    "in_cp_1d": "前 1 天转入对手方数", "in_amt_1d": "前 1 天转入金额",
    "out_cnt_3d": "前 3 天转出笔数",   "out_cp_3d": "前 3 天转出对手方数",
    "out_amt_3d": "前 3 天转出金额",   "out_amt_max_3d": "前 3 天单笔最大转出额",
    "in_cnt_3d": "前 3 天转入笔数",    "in_cp_3d": "前 3 天转入对手方数",
    "in_amt_3d": "前 3 天转入金额",    "in_amt_max_3d": "前 3 天单笔最大转入额",
    "active_days_3d": "前 3 天活跃天数",
}
FEATURE_DESC = {
    "f_log_amount": "本笔金额",
    "f_cross_currency": "跨币种",
    "f_intra_branch": "同一银行内转账",
    "f_payment_format": "支付方式",
    "p_entity_type": "付款方实体类型",
    "r_entity_type": "收款方实体类型",
    "p_today_out_cnt": "付款方今天此前已转出笔数",
    "p_today_out_amt": "付款方今天此前已转出金额",
    "p_today_out_cp": "付款方今天此前已转给几个对手方",
    "r_today_in_cnt": "收款方今天此前已收到笔数",
    "r_today_in_amt": "收款方今天此前已收到金额",
    "r_today_in_cp": "收款方今天此前已从几个对手方收款",
    "f_hist_days": "可用历史天数",
    "f_pair_prev_cnt": "双方此前同方向来往笔数（0 = 首次来往）",
    "f_pair_rev_prev_cnt": "此前反方向来往笔数（资金曾回流）",
    "p_ent_n_accounts": "付款实体名下账户数",
    "p_ent_n_banks": "付款实体涉及银行数",
    "r_ent_n_accounts": "收款实体名下账户数",
    "r_ent_n_banks": "收款实体涉及银行数",
    "f_same_entity": "收付双方是同一实体",
}
for _k, _v in _SNAP.items():
    FEATURE_DESC[f"p_{_k}"] = "付款方" + _v
    FEATURE_DESC[f"r_{_k}"] = "收款方" + _v

_BINARY = {"f_cross_currency", "f_intra_branch", "f_same_entity"}


def describe(name: str) -> str:
    return FEATURE_DESC.get(name, name)


def display_value(name: str, v) -> str:
    """把特征值转成人能读懂的形式：对数金额还原成美元，0/1 转成是/否。"""
    if isinstance(v, str) or v is None:
        return str(v)
    if isinstance(v, float) and math.isnan(v):
        return "缺失"
    if name == "f_log_amount":
        return f"{math.expm1(v):,.0f} 美元"
    if name in _BINARY:
        return "是" if v >= 0.5 else "否"
    if "_amt" in name:
        return f"{v:,.0f} 美元"
    return f"{v:,.0f}"


# ---------------------------------------------------------------------------
# 模型与 SHAP
# ---------------------------------------------------------------------------
def load_booster(key: str = "m4b"):
    model_file, meta_file = MODELS[key]
    booster = lgb.Booster(model_file=str(MODEL_DIR / model_file))
    meta_path = MODEL_DIR / meta_file
    if meta_path.suffix == ".json":
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = joblib.load(meta_path)
    return booster, meta


def shap_values(booster: lgb.Booster, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    返回 (contrib, base)：
      contrib  形状 (行数, 特征数)，每个特征对这一行的贡献（log-odds）
      base     形状 (行数,)，基准值：模型在训练数据上的平均原始分数，每行都一样
    """
    out = booster.predict(X, pred_contrib=True)
    return out[:, :-1], out[:, -1]


def check_additivity(booster: lgb.Booster, X: pd.DataFrame,
                     contrib: np.ndarray, base: np.ndarray) -> float:
    """SHAP 的核心性质：基准值 + 贡献之和 = 原始分数。返回最大误差，应在 1e-6 量级。"""
    raw = booster.predict(X, raw_score=True)
    return float(np.abs(base + contrib.sum(axis=1) - raw).max())


def explain_rows(names: list[str], X: pd.DataFrame, contrib: np.ndarray,
                 base: np.ndarray, k_up: int = 5, k_down: int = 3) -> list[dict]:
    """
    每一行给出：推高分数最多的 k_up 个特征、拉低最多的 k_down 个特征。
    这是后续 Agent 工具 `explain_alert` 的返回格式。
    """
    results = []
    for i in range(len(X)):
        c = contrib[i]
        row = X.iloc[i]

        def item(j):
            n = names[j]
            v = row[n]
            v = v.item() if hasattr(v, "item") else v
            return {"feature": n, "desc": describe(n),
                    "value": display_value(n, v), "shap": round(float(c[j]), 4)}

        order = np.argsort(c)
        up = [item(j) for j in order[::-1][:k_up] if c[j] > 0]
        down = [item(j) for j in order[:k_down] if c[j] < 0]
        raw = float(base[i] + c.sum())
        results.append({"base": round(float(base[i]), 4),
                        "raw_score": round(raw, 4),
                        "pushes_up": up, "pushes_down": down})
    return results


def print_explanation(e: dict, title: str) -> None:
    print(f"\n  【{title}】")
    print(f"  基准值 {e['base']:+.3f}  →  最终 {e['raw_score']:+.3f}（log-odds）")
    print(f"  {'推高分数':<34}{'取值':>18}{'SHAP':>9}")
    for it in e["pushes_up"]:
        print(f"    {it['desc']:<32}{it['value']:>18}{it['shap']:>+9.3f}")
    if e["pushes_down"]:
        print(f"  {'拉低分数':<34}")
        for it in e["pushes_down"]:
            print(f"    {it['desc']:<32}{it['value']:>18}{it['shap']:>+9.3f}")