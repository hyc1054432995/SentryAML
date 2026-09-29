"""
SentryAML - 评测工具（逻辑回归与 LightGBM 共用）
================================================
两个模型必须用【完全相同】的指标计算方式，比较才有意义，
所以把评测逻辑抽到这里，两边都从这里导入。
和 feature_spec.py 一样，是"唯一真相来源"的做法。
"""

import math

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

# 规则引擎 baseline（decisions.md D14），用作参照
RULE_RECALL = 0.5567
RULE_PRECISION = 0.001508


def correct_prob(p: np.ndarray, r: float) -> np.ndarray:
    """
    负样本下采样后的概率校正。
    负样本只保留了比例 r，模型眼中的赔率被放大了 1/r 倍，乘回 r：
        真实概率 = p·r / (p·r + 1 − p)
    这是单调变换，不改变排序，所以 AUC-PR / AUC-ROC 不受影响。
    """
    return p * r / (p * r + 1 - p)


def precision_at_recall(y: np.ndarray, score: np.ndarray, target: float) -> float:
    """召回率至少达到 target 时，能取得的最高精确率。"""
    prec, rec, _ = precision_recall_curve(y, score)
    mask = rec >= target
    return float(prec[mask].max()) if mask.any() else float("nan")


def alert_budget(y: np.ndarray, score: np.ndarray, frac: float) -> dict:
    """
    把分数最高的 frac 比例交易标为告警，能抓到多少洗钱。

    并列处理：如果第 k 名的分数有很多交易并列（单特征模型、规则类分数很常见），
    "取哪几笔"不能交给排序算法随手决定——那样结果取决于行的物理顺序，而不是模型。
    这里按"并列者中随机抽取"的期望值计算：
        分数 > 门槛的全部计入；等于门槛的，按需要的名额比例计入。
    分数连续、几乎没有并列时，结果与直接取前 k 名相同。
    """
    k = max(1, math.ceil(frac * len(y)))
    thr = np.partition(score, len(score) - k)[len(score) - k]   # 第 k 大的分数
    above = score > thr
    at = score == thr
    need = k - int(above.sum())                                 # 并列组里还要取几笔
    tp = float(y[above].sum()) + need * float(y[at].mean())
    return {"alerts": k, "tp": tp,
            "precision": tp / k,
            "recall": tp / int(y.sum())}


def evaluate(y: np.ndarray, p_raw: np.ndarray, r: float) -> dict:
    base_rate = float(y.mean())
    ap = float(average_precision_score(y, p_raw))
    p_cal = correct_prob(p_raw, r)
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": base_rate,
        "auc_pr": ap,
        "auc_pr_lift": ap / base_rate,
        "auc_roc": float(roc_auc_score(y, p_raw)),
        "mean_prob_raw": float(p_raw.mean()),
        "mean_prob_corrected": float(p_cal.mean()),
        "precision_at_rule_recall": precision_at_recall(y, p_raw, RULE_RECALL),
        "precision_at_recall_80": precision_at_recall(y, p_raw, 0.80),
        "budget": {f"top_{frac:.1%}": alert_budget(y, p_raw, frac)
                   for frac in (0.001, 0.005, 0.01)},
    }


def print_report(m: dict, r: float, split_name: str = "验证集", rough_rule_compare: bool = True) -> None:
    print(f"\n  {split_name}：{m['n']:,} 笔，洗钱 {m['positives']:,} 笔，"
          f"基础比例 {m['base_rate']:.4%}")

    print("\n  --- 排序能力 ---")
    print(f"  AUC-PR   {m['auc_pr']:.4f}   （随机猜测 = {m['base_rate']:.4f}，"
          f"提升 {m['auc_pr_lift']:.0f} 倍）")
    print(f"  AUC-ROC  {m['auc_roc']:.4f}   （极度不平衡时通常远高于 AUC-PR）")

    print("\n  --- 概率校正是否正确 ---")
    print(f"  校正前平均概率 {m['mean_prob_raw']:.4%}   "
          f"校正后 {m['mean_prob_corrected']:.4%}   实际洗钱比例 {m['base_rate']:.4%}")
    print(f"  （负样本保留比例 r = {r}）")

    print("\n  --- 告警预算：只审查分数最高的一部分交易 ---")
    print(f"  {'预算':<10}{'告警数':>10}{'抓到':>8}{'召回率':>10}{'精确率':>10}")
    for name, b in m["budget"].items():
        print(f"  {name:<10}{b['alerts']:>10,}{b['tp']:>8,.0f}"
              f"{b['recall']:>10.1%}{b['precision']:>10.2%}")

    if not rough_rule_compare:
        return
    print("\n  --- 与规则引擎粗略对比（口径不同，正式对比见 M5）---")
    print(f"  规则引擎：召回 {RULE_RECALL:.2%} 时，精确率 {RULE_PRECISION:.2%}（全时段）")
    print(f"  本模型：  召回 {RULE_RECALL:.2%} 时，精确率 {m['precision_at_rule_recall']:.2%}")
    print(f"  本模型：  召回 80% 时，精确率 {m['precision_at_recall_80']:.2%}")


def compare(name_a: str, m_a: dict, name_b: str, m_b: dict) -> None:
    """并排比较两个模型的关键指标。"""
    print(f"\n  {'指标':<24}{name_a:>14}{name_b:>14}")
    rows = [
        ("AUC-PR", "auc_pr", "{:.4f}"),
        ("AUC-ROC", "auc_roc", "{:.4f}"),
        ("召回 55.67% 时精确率", "precision_at_rule_recall", "{:.2%}"),
        ("召回 80% 时精确率", "precision_at_recall_80", "{:.2%}"),
    ]
    for label, key, fmt in rows:
        print(f"  {label:<24}{fmt.format(m_a[key]):>14}{fmt.format(m_b[key]):>14}")
    for frac in ("top_0.1%", "top_0.5%", "top_1.0%"):
        a = m_a["budget"][frac]
        b = m_b["budget"][frac]
        print(f"  {frac + ' 召回':<24}{a['recall']:>13.1%}{b['recall']:>14.1%}")