"""
SentryAML - R9a：生成告警队列
==============================
用 M4b 给测试期（9/9–9/10）全部交易打分，取分数最高的 0.5% 作为告警队列，写入 alerts 表；
真实标签单独写入 alert_labels（调查工具的数据库角色读不到它）。

为什么用模型告警而不是规则告警（R9 设计，D68）：
  规则引擎误报率 99.85%（每 662 条告警才有 1 笔洗钱），Agent 会淹没在误报里，测不出价值；
  "模型打分 → 人工分诊"也是今天银行交易监控的常见流程。

为什么是 0.5%：D29 的告警预算之一。分两档（D31：对分析员展示档位而不是概率）：
  T1  前 0.1%       测试集实际命中率 53.6%（453/846）
  T2  0.1%–0.5%     测试集实际命中率 5.1%（174/3,380）—— 分诊真正要发挥作用的地方
脚本会重新算出这两组数字，与 D31 不一致就报错：说明模型文件或特征数据和当时评测的不是同一份。

每条告警附带 SHAP 解释（推高分数最多的 5 个特征、拉低最多的 3 个），Agent 的首轮从这里开始。

用法：
    python scripts/alert_queue.py
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from db import connect
from feature_spec import LABEL_COL
from model_explain import explain_rows, load_booster, shap_values

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
BUDGET, T1 = 0.005, 0.001
EXPECTED = {"T1": (846, 453), "T2": (3380, 174)}         # D31 记录的 (告警数, 洗钱数)


def main():
    t0 = time.perf_counter()
    booster, meta = load_booster("m4b")
    num, cat = meta["features_num"], meta["features_cat"]
    cols = num + cat
    meta_cols = ["txn_id", "txn_ts", "from_bank", "from_account", "to_bank", "to_account", "pattern_type", LABEL_COL]
    df = pq.read_table(FEATURE_DIR / "test.parquet", columns=meta_cols + [c for c in cols if c not in meta_cols]
                       ).to_pandas()
    X = df[cols].copy()
    for c in cat:
        X[c] = X[c].astype("category")
    score = booster.predict(X)
    n = len(score)
    print(f"测试期交易 {n:,} 笔，打分用时 {time.perf_counter() - t0:.0f}s")

    k, k1 = math.ceil(BUDGET * n), math.ceil(T1 * n)
    order = np.argsort(-score, kind="stable")[:k]
    # 分位：在全部测试交易中，分数低于它的比例
    pct = 100.0 * (1 - (np.arange(k) + 1) / n)
    sel = df.iloc[order].reset_index(drop=True)
    sel["score"], sel["score_pct"] = score[order], pct
    sel["tier"] = np.where(np.arange(k) < k1, "T1", "T2")

    # 自检：与 D31 的数字一致
    for t, (n_exp, pos_exp) in EXPECTED.items():
        g = sel[sel["tier"] == t]
        got = (len(g), int(g[LABEL_COL].sum()))
        print(f"  {t}：告警 {got[0]:,}，其中洗钱 {got[1]}（命中率 {got[1] / got[0]:.1%}）；D31 记录 {n_exp:,} / {pos_exp}")
        # 告警数必须完全一致；洗钱数允许差几笔：D31 用"分数 ≥ 门槛"取告警，这里按排名取前 k 笔，
        # 门槛上分数并列的交易谁进谁出可能不同（D29 记录过并列问题）
        if got[0] != n_exp or abs(got[1] - pos_exp) > 3:
            raise SystemExit("与 D31 不一致：模型文件或特征数据不是当时评测的那一份，先查清原因再继续")
        if got[1] != pos_exp:
            print(f"    （洗钱数差 {got[1] - pos_exp}：门槛上的并列造成，可以接受）")

    Xs = X.iloc[order].reset_index(drop=True)
    contrib, base = shap_values(booster, Xs)
    expl = explain_rows(cols, Xs, contrib, base)
    amount = np.expm1(Xs["f_log_amount"].to_numpy())

    with connect() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE alerts CASCADE")
        cur.executemany(
            """INSERT INTO alerts (alert_id, txn_id, txn_ts, from_bank, from_account, to_bank, to_account,
                                   amount_usd, score, score_pct, tier, shap, model)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'm4b')""",
            [(i + 1, int(r.txn_id), r.txn_ts, int(r.from_bank), str(r.from_account), int(r.to_bank),
              str(r.to_account), round(float(amount[i]), 2), float(r.score), round(float(r.score_pct), 4), r.tier,
              json.dumps(expl[i], ensure_ascii=False)) for i, r in enumerate(sel.itertuples())])
        cur.executemany(
            "INSERT INTO alert_labels (alert_id, is_laundering, pattern_type) VALUES (%s, %s, %s)",
            [(i + 1, int(r[LABEL_COL]), None if pd.isna(r.pattern_type) else r.pattern_type)
             for i, r in sel.iterrows()])
        conn.commit()
        cur.execute("""SELECT a.tier, l.pattern_type, COUNT(*) FROM alerts a JOIN alert_labels l USING (alert_id)
                       WHERE l.is_laundering = 1 GROUP BY 1, 2 ORDER BY 1, 3 DESC""")
        rows = cur.fetchall()
    print(f"\n已写入 alerts / alert_labels：{k:,} 条告警；用时 {time.perf_counter() - t0:.0f}s")
    print("  告警中的洗钱，按档位和手法（无标签 = 没有模式标注的洗钱，D32）：")
    for tier, pt, c in rows:
        print(f"    {tier}  {pt or '（无模式标签）':<16}{c:>5}")


if __name__ == "__main__":
    main()