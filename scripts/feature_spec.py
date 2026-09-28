"""
SentryAML - 特征清单（唯一真相来源）
====================================
哪些列是特征、哪些是元数据、哪些是分类变量、哪些正在接受泄漏审计，
只在这个文件里定义。导出脚本和训练脚本都从这里导入，
避免"改了一处、忘了另一处"导致的隐蔽 bug。

对应数据库表：feat_txn（由 sql/011_feature_table.sql 生成）
"""

# ---------------------------------------------------------------------------
# 元数据：不是特征。用于切分、评测分组、事后分析
# ---------------------------------------------------------------------------
META_COLS = [
    "txn_id",
    "txn_ts",
    "split",
    "from_bank", "from_account",
    "to_bank", "to_account",
    "pattern_type",     # 事后分析用。它是调查后才有的标注，绝不能当特征（M2 Q7）
]

LABEL_COL = "label"

# ---------------------------------------------------------------------------
# 分类特征：取值是有限的几个文本类别
# ---------------------------------------------------------------------------
CAT_FEATURES = [
    "f_payment_format",
    "p_entity_type",
    "r_entity_type",
]

# ---------------------------------------------------------------------------
# 数值特征
# ---------------------------------------------------------------------------
_SNAPSHOT_SUFFIXES = [
    "out_cnt_1d", "out_cp_1d", "out_amt_1d",
    "in_cnt_1d",  "in_cp_1d",  "in_amt_1d",
    "out_cnt_3d", "out_cp_3d", "out_amt_3d", "out_amt_max_3d",
    "in_cnt_3d",  "in_cp_3d",  "in_amt_3d",  "in_amt_max_3d",
    "active_days_3d",
]

NUM_FEATURES = (
    # ① 交易本身
    ["f_log_amount", "f_cross_currency", "f_intra_branch"]
    # ② 付款方：离线快照 + 当日实时
    + [f"p_{s}" for s in _SNAPSHOT_SUFFIXES]
    + ["p_today_out_cnt", "p_today_out_amt", "p_today_out_cp"]
    # ③ 收款方：离线快照 + 当日实时
    + [f"r_{s}" for s in _SNAPSHOT_SUFFIXES]
    + ["r_today_in_cnt", "r_today_in_amt", "r_today_in_cp"]
    # 可用历史天数
    + ["f_hist_days"]
    # ④ 账户对关系
    + ["f_pair_prev_cnt", "f_pair_rev_prev_cnt"]
    # ⑤ 实体资料
    + ["p_ent_n_accounts", "p_ent_n_banks",
       "r_ent_n_accounts", "r_ent_n_banks",
       "f_same_entity"]
)

# ---------------------------------------------------------------------------
# 泄漏审计（M6）
# ---------------------------------------------------------------------------
# 已知泄漏：数据生成器的痕迹，主模型不使用（D07）
KNOWN_LEAKY = ["f_payment_format"]

# 可疑、待审计：主模型暂时保留，M6 做"有 / 无"对比（D24）
UNDER_AUDIT = ["f_pair_prev_cnt"]


def is_amount_col(col: str) -> bool:
    """金额类特征：导出为 float32。其余数值特征是计数或 0/1 标志，导出为 int32。"""
    return "_amt" in col or col == "f_log_amount"


def model_features(exclude_known_leaky: bool = True) -> tuple[list[str], list[str]]:
    """返回主模型使用的 (数值特征, 分类特征)。"""
    num = list(NUM_FEATURES)
    cat = list(CAT_FEATURES)
    if exclude_known_leaky:
        num = [c for c in num if c not in KNOWN_LEAKY]
        cat = [c for c in cat if c not in KNOWN_LEAKY]
    return num, cat


ALL_COLS = META_COLS + [LABEL_COL] + CAT_FEATURES + NUM_FEATURES

# 导入时自检：不允许重复列名
assert len(ALL_COLS) == len(set(ALL_COLS)), "feature_spec 中存在重复列名"