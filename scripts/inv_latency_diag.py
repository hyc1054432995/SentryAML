"""
SentryAML - R9a：调查工具超时的根因诊断
========================================
冒烟测试的延迟部分（30 条随机告警）里，inv_pass_through 和 inv_trace_cycle 各超时 1 次（15 秒），
而中位数只有几毫秒。中位数和最大值差了 3 个数量级，说明慢的不是"普遍偏慢"，而是少数输入特别重。

待验证的假设：耗时由焦点账户的交易量决定，而交易量是重尾分布（D16）。
  inv_pass_through  把 72 小时内每一笔转入和之后 72 小时内每一笔转出两两比较，成本约为 转入笔数 × 转出笔数
  inv_trace_cycle   从起点向外逐跳展开，起点的对手方越多，第一跳的分支越多，后面每跳再分叉
  inv_fan_profile   要数告警时刻之前全部历史里的不同对手方，账户历史越长越慢
如果假设成立：最重的账户应该集中在超时和慢调用里，普通账户都很快。
如果不成立（重账户也很快，或者慢的是普通账户），根因在别处（比如执行计划没用上索引），修法完全不同。

做法（不看标签）：
  1. 主连接直接查 txn_norm，算出全部 4,226 条告警两端账户在告警时刻的活跃度
     （72 小时收 / 付笔数、7 天付款笔数和对手方数、历史对手方数、是否清算层）
  2. 挑三组账户：转入 × 转出最大的 10 个、7 天付款对手方最多的 10 个、随机 10 个作对照
  3. 用 Agent 角色（15 秒超时）在这些账户上逐个计时，把耗时和活跃度放在同一张表里看

用法：
    python scripts/inv_latency_diag.py
"""

from __future__ import annotations

import json
import random
import statistics
import time

from psycopg import errors
from psycopg.rows import dict_row

from db import PROJECT_ROOT, connect, connect_agent

TOOLS = {
    "pass": "SELECT * FROM inv_pass_through(%s, %s, %s)",
    "cycle": "SELECT * FROM inv_trace_cycle(%s, %s, %s)",
    "fan": "SELECT * FROM inv_fan_profile(%s, %s, %s)",
    "activity": "SELECT * FROM inv_account_activity(%s, %s, %s)",
    "two_hop": "SELECT * FROM inv_two_hop(%s, %s, %s)",
}

ACTIVITY_SQL = """
WITH acc AS (
    SELECT alert_id, txn_ts AS t, 'payer' AS side, from_bank AS bank, from_account AS acct FROM alerts
    UNION ALL
    SELECT alert_id, txn_ts, 'payee', to_bank, to_account FROM alerts
)
SELECT acc.alert_id, acc.t, acc.side, acc.bank, acc.acct,
       COALESCE(dg.total_cp_count, 0) AS total_cp,
       i72.n AS in_72h, o72.n AS out_72h, o7.n AS out_7d, o7.cp AS out_cp_7d
FROM acc
LEFT JOIN account_degrees dg ON dg.bank_id = acc.bank AND dg.account_number = acc.acct
CROSS JOIN LATERAL (SELECT COUNT(*)::int AS n FROM txn_norm
                    WHERE to_bank = acc.bank AND to_account = acc.acct
                      AND txn_ts > acc.t - INTERVAL '72 hours' AND txn_ts <= acc.t) i72
CROSS JOIN LATERAL (SELECT COUNT(*)::int AS n FROM txn_norm
                    WHERE from_bank = acc.bank AND from_account = acc.acct
                      AND txn_ts > acc.t - INTERVAL '72 hours' AND txn_ts <= acc.t) o72
CROSS JOIN LATERAL (SELECT COUNT(*)::int AS n, COUNT(DISTINCT (to_bank, to_account))::int AS cp FROM txn_norm
                    WHERE from_bank = acc.bank AND from_account = acc.acct
                      AND txn_ts > acc.t - INTERVAL '7 days' AND txn_ts <= acc.t) o7
ORDER BY acc.alert_id, acc.side
"""


def pct(v: list[float], q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]


def main():
    t0 = time.perf_counter()
    with connect() as main_conn, main_conn.cursor(row_factory=dict_row) as cur:
        cur.execute(ACTIVITY_SQL)
        rows = cur.fetchall()
    for r in rows:
        r["pairs"] = r["in_72h"] * r["out_72h"]       # 过桥检查要比较的配对数（上限）
        r["clearing"] = r["total_cp"] > 200
    print(f"[1] 告警两端账户的活跃度（{len(rows):,} 个 告警×账户，{time.perf_counter() - t0:.0f}s）")
    print(f"    {'指标':<22}{'中位数':>8}{'P90':>8}{'P99':>9}{'最大':>10}")
    for k, name in (("in_72h", "72h 收款笔数"), ("out_72h", "72h 付款笔数"), ("pairs", "收 × 付（过桥成本）"),
                    ("out_cp_7d", "7 天付款对手方数"), ("total_cp", "历史对手方数")):
        v = [r[k] for r in rows]
        print(f"    {name:<20}{statistics.median(v):>8.0f}{pct(v, .9):>8.0f}{pct(v, .99):>9.0f}{max(v):>10,}")
    for side in ("payer", "payee"):
        n = sum(r["clearing"] for r in rows if r["side"] == side)
        print(f"    {side} 是清算层账户（历史对手方 > 200，D17）的告警：{n}")

    # 挑账户：同一账户可能出现在多条告警里，按 (方向, 账户) 去重，保留最重的那次
    def top(key, k, non_clearing=False):
        seen, out = set(), []
        for r in sorted(rows, key=lambda r: -r[key]):
            if non_clearing and r["clearing"]:
                continue
            if (r["bank"], r["acct"]) not in seen:
                seen.add((r["bank"], r["acct"]))
                out.append(r)
            if len(out) == k:
                break
        return out
    picked, keys = [], set()
    # 021 之后，清算层上的路径工具直接拒绝；真正决定最坏耗时的变成"最重的普通账户"，单独挑出来
    for group, rs in (("收×付最大", top("pairs", 10)), ("对手方最多", top("out_cp_7d", 10)),
                      ("普通·收×付", top("pairs", 10, True)), ("普通·对手方", top("out_cp_7d", 10, True)),
                      ("随机对照", random.Random(7).sample(rows, min(10, len(rows))))):
        for r in rs:
            if (r["bank"], r["acct"]) not in keys:
                keys.add((r["bank"], r["acct"]))
                picked.append({**r, "group": group})

    print(f"\n[2] 在 {len(picked)} 个账户上逐个计时（Agent 角色，15 秒超时；'超时' = 15 秒内没跑完）")
    print(f"    {'组':<8}{'告警':>6} {'方向':<6}{'历史cp':>7}{'收72h':>7}{'付72h':>7}{'7天cp':>7}  "
          + "".join(f"{k:>10}" for k in TOOLS))
    with connect_agent() as agent:
        for r in picked:
            r["ms"] = {}
            for k, sql in TOOLS.items():
                t1 = time.perf_counter()
                try:
                    with agent.cursor() as c:
                        c.execute(sql, (r["bank"], r["acct"], r["t"]))
                        c.fetchall()
                    r["ms"][k] = round((time.perf_counter() - t1) * 1000)
                except errors.QueryCanceled:
                    r["ms"][k] = None
                except errors.RaiseException as e:          # 021：清算层账户上的路径工具明确拒绝
                    if not str(e).startswith("clearing_layer"):
                        raise
                    r["ms"][k] = "refused"
            cells = "".join(f"{'超时' if v is None else '拒绝' if v == 'refused' else f'{v:,}ms':>10}"
                            for v in r["ms"].values())
            flag = "*" if r["clearing"] else " "
            print(f"    {r['group']:<6}{r['alert_id']:>8} {r['side']:<6}{r['total_cp']:>6}{flag}{r['in_72h']:>7}"
                  f"{r['out_72h']:>7}{r['out_cp_7d']:>7}  {cells}")
    print("    （* = 清算层账户；拒绝 = 021 起路径工具对清算层账户明确拒绝）")

    out = PROJECT_ROOT / "reports" / "r9a_latency_diag.json"
    out.write_text(json.dumps({"picked": picked}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n结果写到 {out.relative_to(PROJECT_ROOT)}；总用时 {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()