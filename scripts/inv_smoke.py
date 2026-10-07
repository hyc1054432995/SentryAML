"""
SentryAML - R9a：调查工具冒烟测试
==================================
在 R9b 把这些 SQL 函数包装成 Agent 工具之前，先确认三件事：

  1. 看不到答案   静态审计：所有 inv_* 函数的源码、inv_txn 视图的列里，不出现含标签的表或列；
                  权限测试：用 sentry_agent 角色直接读 txn_norm / alert_labels 等表，必须被数据库拒绝。
  2. 时点正确     工具返回的每个时间都不晚于告警时刻；而且"未来"交易确实存在、只是看不到；
                  已确认案件在 9/8 23:59:59 看不到、在告警时刻看得到。
  3. 结果正确     用主连接直接查 txn_norm 算出同样的数，和工具的结果逐项对比；
                  回流追踪返回的每条路径逐笔核对（首尾相接、时间递增、7 天内、回到起点）。

清算层账户：021 起，路径类工具（过桥、回流、两跳）对清算层账户明确拒绝，单独验证拒绝和不误拒。
最后测一遍延迟：随机抽 30 条告警，每个工具跑一次，看中位数、P95 和超时次数。
延迟测试不看标签。为什么：告警队列就是 R9c 要评测的数据，按标签统计"哪个信号在洗钱告警里更常出现"，
等于在定规则之前偷看测试集（D64 的教训）。
冒烟测试逐条展示的 3 条告警会看到标签（为了人工核对），它们的编号写进 reports/r9a_smoke_alerts.json，
R9c 抽样时连同所在的账户连通分量一起排除。

用法：
    python scripts/inv_smoke.py               # 默认延迟测试抽 30 条
    python scripts/inv_smoke.py --latency 0   # 跳过延迟测试
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from datetime import date, datetime, timedelta

import psycopg
from psycopg import errors
from psycopg.rows import dict_row

from db import PROJECT_ROOT, connect, connect_agent

TOOLS = ["inv_account_activity", "inv_fan_profile", "inv_counterparty_history", "inv_pass_through",
         "inv_trace_cycle", "inv_two_hop", "inv_entity_profile", "inv_case_history", "inv_amount_profile"]
# 工具源码里不允许出现的名字：含标签的表、标签列、支付方式列（D07：数据生成器的痕迹）
FORBIDDEN = ["txn_norm", "is_laundering", "alert_labels", "pattern_transactions", "laundering_patterns",
             "payment_format"]
INV_TXN_COLS = {"txn_id", "txn_ts", "from_bank", "from_account", "to_bank", "to_account", "amount_usd",
                "is_cross_currency"}
# Agent 角色必须读不到的表（第二项是"表不存在"时是否也算通过：本项目里这些表都应该存在）
DENIED_TABLES = ["txn_norm", "alert_labels", "transactions", "pattern_transactions", "laundering_patterns"]
ALLOWED_TABLES = ["inv_txn", "accounts", "entities", "account_degrees", "inv_cases", "alerts"]
CASE_CUTOFF = datetime(2022, 9, 8, 23, 59, 59)

results: list[dict] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"  {'✓' if ok else '✗'} {name}" + (f"   {detail}" if detail else ""))
    return ok


def tool_calls(a: dict, bank: int, acct: str) -> list[tuple[str, str, tuple]]:
    """一条告警、一个焦点账户上的全部工具调用。as_of 一律是告警时刻（R9b 里由包装层绑定，模型填不了）。"""
    t = a["txn_ts"]
    return [
        ("inv_account_activity", "SELECT * FROM inv_account_activity(%s, %s, %s)", (bank, acct, t)),
        ("inv_fan_profile", "SELECT * FROM inv_fan_profile(%s, %s, %s)", (bank, acct, t)),
        ("inv_counterparty_history", "SELECT * FROM inv_counterparty_history(%s, %s, %s, %s, %s)",
         (a["from_bank"], a["from_account"], a["to_bank"], a["to_account"], t)),
        ("inv_pass_through", "SELECT * FROM inv_pass_through(%s, %s, %s)", (bank, acct, t)),
        ("inv_trace_cycle", "SELECT * FROM inv_trace_cycle(%s, %s, %s)", (bank, acct, t)),
        ("inv_two_hop", "SELECT * FROM inv_two_hop(%s, %s, %s)", (bank, acct, t)),
        ("inv_entity_profile", "SELECT * FROM inv_entity_profile(%s, %s)", (bank, acct)),
        ("inv_case_history", "SELECT * FROM inv_case_history(%s, %s, %s)", (bank, acct, t)),
        ("inv_amount_profile", "SELECT * FROM inv_amount_profile(%s)", (a["txn_id"],)),
    ]


def run(conn, sql: str, args: tuple) -> tuple[list[dict], float]:
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql, args)
        cols = [d.name for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return rows, (time.perf_counter() - t0) * 1000


PATH_TOOLS = {"inv_pass_through", "inv_trace_cycle", "inv_two_hop"}   # 021：对清算层账户明确拒绝


def is_clearing_refusal(e: Exception) -> bool:
    """021 的拒绝：RAISE EXCEPTION，消息以 clearing_layer 开头。和超时、真正的错误分开处理。"""
    return isinstance(e, errors.RaiseException) and str(e).startswith("clearing_layer")


def one(conn, sql: str, args: tuple = ()):
    with conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchone()


def fmt(v) -> str:
    if isinstance(v, datetime):
        return v.strftime("%m-%d %H:%M")
    if isinstance(v, list):
        return "[" + ", ".join(map(str, v[:4])) + (", …" if len(v) > 4 else "") + "]"
    return str(v)


def show(name: str, rows: list[dict], ms: float, limit: int = 4):
    print(f"    {name}  ({len(rows)} 行, {ms:.0f} ms)")
    for r in rows[:limit]:
        print("      " + "  ".join(f"{k}={fmt(v)}" for k, v in r.items()))
    if len(rows) > limit:
        print(f"      … 另有 {len(rows) - limit} 行")


def time_bound_ok(name: str, rows: list[dict], t: datetime) -> tuple[bool, str]:
    """工具结果里所有时间都不晚于告警时刻；账户对的既往来往要严格早于告警时刻。"""
    strict = name == "inv_counterparty_history"
    for r in rows:
        for k, v in r.items():
            if isinstance(v, datetime) and (v >= t if strict else v > t):
                return False, f"{k}={v} {'不早于' if strict else '晚于'}告警时刻 {t}"
            if isinstance(v, date) and not isinstance(v, datetime) and v > t.date():
                return False, f"{k}={v} 晚于告警日期"
    return True, ""


# ---------------------------------------------------------------------------
def static_audit(main):
    print("\n[1] 静态审计：工具源码和视图里没有标签")
    with main.cursor() as cur:
        cur.execute("""SELECT p.proname, p.prosrc FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
                       WHERE n.nspname = 'public' AND p.proname LIKE 'inv\\_%'""")
        funcs = dict(cur.fetchall())
        cur.execute("""SELECT column_name FROM information_schema.columns
                       WHERE table_schema = 'public' AND table_name = 'inv_txn'""")
        cols = {r[0] for r in cur.fetchall()}
    check("9 个调查函数都已创建", set(TOOLS) <= set(funcs), f"缺少 {sorted(set(TOOLS) - set(funcs))}"
          if not set(TOOLS) <= set(funcs) else "")
    extra = sorted(set(funcs) - set(TOOLS))
    if extra:   # 以后新增的 inv_* 函数也要过同样的审计
        print(f"    （另有 inv_* 函数 {extra}，一并审计）")
    bad = {f: [w for w in FORBIDDEN if w in src.lower()] for f, src in funcs.items()}
    bad = {f: ws for f, ws in bad.items() if ws}
    check("函数源码不引用含标签的表 / 列", not bad, json.dumps(bad, ensure_ascii=False) if bad else "")
    check("inv_txn 视图只有约定的 8 列", cols == INV_TXN_COLS,
          "" if cols == INV_TXN_COLS else f"多出 {sorted(cols - INV_TXN_COLS)}，缺少 {sorted(INV_TXN_COLS - cols)}")


def permission_test(agent):
    print("\n[2] 权限：用 sentry_agent 角色连接")
    user, is_super = one(agent, "SELECT current_user, rolsuper FROM pg_roles WHERE rolname = current_user")
    check("连接角色是 sentry_agent、不是超级用户", user == "sentry_agent" and not is_super, f"current_user={user}")
    for tbl in DENIED_TABLES:
        try:
            one(agent, f"SELECT 1 FROM {tbl} LIMIT 1")
            check(f"读 {tbl} 被拒绝", False, "居然读到了！检查 GRANT")
        except errors.InsufficientPrivilege:
            check(f"读 {tbl} 被拒绝", True, "permission denied")
        except errors.UndefinedTable:
            check(f"读 {tbl} 被拒绝", False, "表不存在：本项目里它应该存在，检查连的是不是 sentryaml 库")
    for tbl in ALLOWED_TABLES:
        try:
            one(agent, f"SELECT 1 FROM {tbl} LIMIT 1")
            check(f"读 {tbl} 允许", True)
        except psycopg.Error as e:
            check(f"读 {tbl} 允许", False, type(e).__name__)
    try:   # UPDATE 权限在执行计划阶段就检查，WHERE false 保证即使有权限也不改任何数据
        with agent.cursor() as cur:
            cur.execute("UPDATE alerts SET tier = tier WHERE false")
        check("写 alerts 被拒绝", False, "居然有写权限！")
    except errors.InsufficientPrivilege:
        check("写 alerts 被拒绝", True, "permission denied")


def pick_alerts(main) -> list[tuple[str, dict, tuple[int, str]]]:
    """挑 3 条告警逐条展示（用主连接读标签，只用于人工核对）：
       T1 洗钱（优先有手法标签的）、T1 误报、涉及已确认案件账户的告警（D37 盲区，测 case_history 的时间边界；
       与前两条不重复）。"""
    q = """SELECT a.alert_id, a.txn_id, a.txn_ts, a.from_bank, a.from_account, a.to_bank, a.to_account,
                  a.amount_usd, a.tier, a.score_pct, l.is_laundering, l.pattern_type
           FROM alerts a JOIN alert_labels l USING (alert_id) """
    picks = []
    with main.cursor(row_factory=dict_row) as cur:
        cur.execute(q + "WHERE a.tier = 'T1' AND l.is_laundering = 1 ORDER BY (l.pattern_type IS NULL), a.alert_id LIMIT 1")
        r = cur.fetchone()
        if r:
            picks.append(("T1 洗钱", r, (r["from_bank"], r["from_account"])))
        cur.execute(q + "WHERE a.tier = 'T1' AND l.is_laundering = 0 ORDER BY a.alert_id LIMIT 1")
        r = cur.fetchone()
        if r:
            picks.append(("T1 误报", r, (r["from_bank"], r["from_account"])))
        cur.execute(q + """, LATERAL (SELECT c.bank_id, c.account_number FROM inv_cases c
                                      WHERE (c.bank_id = a.from_bank AND c.account_number = a.from_account)
                                         OR (c.bank_id = a.to_bank AND c.account_number = a.to_account)
                                      ORDER BY (c.bank_id = a.from_bank AND c.account_number = a.from_account) DESC
                                      LIMIT 1) c
                           WHERE a.alert_id <> ALL(%s)
                           ORDER BY a.alert_id LIMIT 1""", ([p[1]["alert_id"] for p in picks],))
        r = cur.fetchone()
        if r:
            cur.execute("""SELECT 1 FROM inv_cases WHERE bank_id = %s AND account_number = %s""",
                        (r["from_bank"], r["from_account"]))
            focus = (r["from_bank"], r["from_account"]) if cur.fetchone() else (r["to_bank"], r["to_account"])
            picks.append(("涉及已确认案件", r, focus))
    for want in ("T1 洗钱", "T1 误报", "涉及已确认案件"):
        check(f"找到「{want}」的示例告警", any(p[0] == want for p in picks))
    return picks


def verify_cycle(main, cyc: dict, start: str, t: datetime) -> tuple[bool, str]:
    """逐笔核对一条回流路径：首尾相接、时间不减、都不晚于告警时刻、7 天内、回到起点。"""
    with main.cursor() as cur:
        cur.execute("""SELECT txn_id, txn_ts, from_bank || ':' || from_account, to_bank || ':' || to_account
                       FROM txn_norm WHERE txn_id = ANY(%s)""", (cyc["txn_path"],))
        by_id = {r[0]: r[1:] for r in cur.fetchall()}
    if len(by_id) != len(cyc["txn_path"]):
        return False, "路径里有交易在 txn_norm 中找不到"
    legs = [by_id[i] for i in cyc["txn_path"]]
    if legs[0][1] != start or legs[-1][2] != start:
        return False, "起点或终点不是告警账户"
    for (ts1, _, to1), (ts2, fr2, _) in zip(legs, legs[1:]):
        if to1 != fr2 or ts2 < ts1:
            return False, f"在 {to1} 处断开或时间倒退"
    if legs[-1][0] > t:
        return False, "用到了告警时刻之后的交易"
    if legs[-1][0] - legs[0][0] > timedelta(days=7):
        return False, "跨度超过 7 天"
    return True, ""


def walk_through(main, agent, picks):
    print("\n[3] 逐条运行（Agent 角色）+ 时点检查 + 与主连接直接查询对账")
    shown = []
    for label, a, (bank, acct) in picks:
        t = a["txn_ts"]
        shown.append(a["alert_id"])
        print(f"\n  ── 告警 #{a['alert_id']}（{label}；标签仅供人工核对：is_laundering={a['is_laundering']}, "
              f"pattern={a['pattern_type']}）")
        print(f"     {a['from_bank']}:{a['from_account']} → {a['to_bank']}:{a['to_account']}  "
              f"${float(a['amount_usd']):,.2f}  {t}  {a['tier']}  分位 {a['score_pct']:.3f}；焦点账户 {bank}:{acct}")
        out = {}
        for name, sql, args in tool_calls(a, bank, acct):
            try:
                rows, ms = run(agent, sql, args)
            except errors.QueryCanceled:
                check(f"#{a['alert_id']} {name} 在超时内完成", False, "超时")
                continue
            except errors.RaiseException as e:
                if is_clearing_refusal(e):
                    print(f"    {name}  拒绝：焦点账户是清算层（021）")
                    continue
                raise
            out[name] = rows
            if label != "涉及已确认案件" or name in ("inv_case_history", "inv_fan_profile"):
                show(name, rows, ms)
            ok, why = time_bound_ok(name, rows, t)
            if not ok:
                check(f"#{a['alert_id']} {name} 不晚于告警时刻", False, why)
        bound_fail = [r for r in results if r["check"].startswith(f"#{a['alert_id']} ") and not r["ok"]]
        check(f"#{a['alert_id']} 全部工具结果的时间都不晚于告警时刻", not bound_fail)

        # 未来的交易确实存在，但工具看不到：把窗口放到 40 天（覆盖全部历史），工具的计数应等于"截至告警时刻"的直接计数
        n_in, n_out, n_future = one(main, """
            SELECT COUNT(*) FILTER (WHERE to_bank = %(b)s AND to_account = %(a)s AND txn_ts <= %(t)s),
                   COUNT(*) FILTER (WHERE from_bank = %(b)s AND from_account = %(a)s AND txn_ts <= %(t)s),
                   COUNT(*) FILTER (WHERE txn_ts > %(t)s)
            FROM txn_norm WHERE (to_bank = %(b)s AND to_account = %(a)s) OR (from_bank = %(b)s AND from_account = %(a)s)
            """, {"b": bank, "a": acct, "t": t})
        fan_all, _ = run(agent, "SELECT * FROM inv_fan_profile(%s, %s, %s, %s)", (bank, acct, t, 24 * 40))
        got = (fan_all[0]["in_cnt"], fan_all[0]["out_cnt"])
        check(f"#{a['alert_id']} 全历史计数 = 直接查询截至告警时刻的计数", got == (n_in, n_out),
              f"工具 收{got[0]}/付{got[1]}，直接查询 收{n_in}/付{n_out}；告警时刻之后还有 {n_future} 笔（工具看不到）")

        if "inv_fan_profile" in out:   # 72 小时窗口的计数和金额
            d = one(main, """
                SELECT COUNT(*) FILTER (WHERE to_bank = %(b)s AND to_account = %(a)s),
                       ROUND(COALESCE(SUM(amount_usd) FILTER (WHERE to_bank = %(b)s AND to_account = %(a)s), 0), 2),
                       COUNT(*) FILTER (WHERE from_bank = %(b)s AND from_account = %(a)s),
                       ROUND(COALESCE(SUM(amount_usd) FILTER (WHERE from_bank = %(b)s AND from_account = %(a)s), 0), 2)
                FROM txn_norm WHERE ((to_bank = %(b)s AND to_account = %(a)s) OR (from_bank = %(b)s AND from_account = %(a)s))
                  AND txn_ts > %(t)s - INTERVAL '72 hours' AND txn_ts <= %(t)s""", {"b": bank, "a": acct, "t": t})
            f = out["inv_fan_profile"][0]
            got = (f["in_cnt"], f["in_amt"], f["out_cnt"], f["out_amt"])
            check(f"#{a['alert_id']} 72 小时收付笔数和金额与直接查询一致", got == d, f"工具 {got}，直接查询 {d}")

        if "inv_account_activity" in out:   # 逐日数字：直接查询按天分组，没有交易的日子补 0
            with main.cursor() as cur:
                cur.execute("""
                    SELECT txn_ts::date,
                           COUNT(*) FILTER (WHERE to_bank = %(b)s AND to_account = %(a)s),
                           ROUND(COALESCE(SUM(amount_usd) FILTER (WHERE to_bank = %(b)s AND to_account = %(a)s), 0), 2),
                           COUNT(*) FILTER (WHERE from_bank = %(b)s AND from_account = %(a)s),
                           ROUND(COALESCE(SUM(amount_usd) FILTER (WHERE from_bank = %(b)s AND from_account = %(a)s), 0), 2)
                    FROM txn_norm WHERE ((to_bank = %(b)s AND to_account = %(a)s) OR (from_bank = %(b)s AND from_account = %(a)s))
                      AND txn_ts >= %(t)s::date - 6 AND txn_ts <= %(t)s
                    GROUP BY 1""", {"b": bank, "a": acct, "t": t})
                direct = {r[0]: tuple(r[1:]) for r in cur.fetchall()}
            zero = (0, 0, 0, 0)
            bad = [str(r["day"]) for r in out["inv_account_activity"]
                   if (r["in_cnt"], r["in_amt"], r["out_cnt"], r["out_amt"]) != direct.get(r["day"], zero)]
            check(f"#{a['alert_id']} 逐日收付笔数和金额与直接查询一致", not bad, f"不一致的日期 {bad}" if bad else "")

        if "inv_counterparty_history" in out:   # 严格早于告警时刻：告警这一笔本身不算"既往来往"
            n_ab = one(main, """SELECT COUNT(*) FROM txn_norm WHERE from_bank = %s AND from_account = %s
                                AND to_bank = %s AND to_account = %s AND txn_ts < %s""",
                       (a["from_bank"], a["from_account"], a["to_bank"], a["to_account"], t))[0]
            got = next(r["n"] for r in out["inv_counterparty_history"] if r["direction"] == "a_to_b")
            check(f"#{a['alert_id']} 既往来往不含告警这一笔", got == n_ab, f"工具 {got}，直接查询 {n_ab}")

        for cyc in out.get("inv_trace_cycle", []):
            ok, why = verify_cycle(main, cyc, f"{bank}:{acct}", t)
            check(f"#{a['alert_id']} 回流路径逐笔核对（{cyc['hops']} 跳）", ok, why)

        for p in out.get("inv_pass_through", []):
            ok = (p["out_ts"] >= p["in_ts"] and p["lag_hours"] <= 72 and p["diff_pct"] <= 10.05)
            if not ok:
                check(f"#{a['alert_id']} 过桥配对符合定义", False, json.dumps(p, default=str))

        if label == "涉及已确认案件":
            before, _ = run(agent, "SELECT * FROM inv_case_history(%s, %s, %s)", (bank, acct, CASE_CUTOFF))
            check(f"#{a['alert_id']} 已确认案件在 9/8 23:59:59 看不到", not before, f"{len(before)} 行")
            check(f"#{a['alert_id']} 已确认案件在告警时刻看得到", bool(out.get("inv_case_history")),
                  f"{len(out.get('inv_case_history', []))} 行")
    return shown


def clearing_test(main, agent):
    """021：付款方是清算层账户的告警上，三个路径工具必须明确拒绝（报错），而不是返回 0 行；
    同一条告警的收款方（普通账户）上不能误拒；描述性工具照常返回，对手方画像标出 clearing_layer。
    不读标签，这条告警不计入"看过标签"的名单。"""
    print("\n[4] 清算层账户：路径工具明确拒绝，描述性工具照常")
    with main.cursor(row_factory=dict_row) as cur:
        cur.execute("""SELECT a.alert_id, a.txn_id, a.txn_ts, a.from_bank, a.from_account, a.to_bank, a.to_account
                       FROM alerts a JOIN account_degrees d
                         ON d.bank_id = a.from_bank AND d.account_number = a.from_account
                       WHERE d.total_cp_count > 200 ORDER BY a.alert_id LIMIT 1""")
        a = cur.fetchone()
    if not check("找到付款方是清算层账户的告警", a is not None):
        return
    print(f"    告警 #{a['alert_id']}：{a['from_bank']}:{a['from_account']} → {a['to_bank']}:{a['to_account']}")
    for name, sql, args in tool_calls(a, a["from_bank"], a["from_account"]):
        try:
            rows, ms = run(agent, sql, args)
            if name in PATH_TOOLS:
                check(f"付款方（清算层）{name} 明确拒绝", False, f"没有拒绝，返回了 {len(rows)} 行")
            elif name == "inv_fan_profile":
                check("付款方（清算层）inv_fan_profile 标出 clearing_layer", rows and rows[0]["clearing_layer"] is True,
                      f"{ms:.0f} ms")
            else:
                check(f"付款方（清算层）{name} 照常返回", True, f"{len(rows)} 行，{ms:.0f} ms")
        except errors.RaiseException as e:
            ok = name in PATH_TOOLS and is_clearing_refusal(e)
            check(f"付款方（清算层）{name} 明确拒绝" if name in PATH_TOOLS else f"付款方（清算层）{name} 照常返回",
                  ok, str(e).splitlines()[0][:90])
        except errors.QueryCanceled:
            check(f"付款方（清算层）{name} 在超时内完成", False, "超时")
    for name, sql, args in tool_calls(a, a["to_bank"], a["to_account"]):
        if name not in PATH_TOOLS:
            continue
        try:
            run(agent, sql, args)
            check(f"收款方（普通账户）{name} 没有被误拒", True)
        except errors.RaiseException as e:
            check(f"收款方（普通账户）{name} 没有被误拒", False, str(e)[:90])
        except errors.QueryCanceled:
            check(f"收款方（普通账户）{name} 在超时内完成", False, "超时")


def latency(agent, main, n: int, seed: int = 42) -> dict:
    print(f"\n[5] 延迟：随机抽 {n} 条告警（不看标签），每个工具在付款账户上跑一次")
    with main.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT alert_id, txn_id, txn_ts, from_bank, from_account, to_bank, to_account FROM alerts "
                    "ORDER BY alert_id")   # 固定顺序：不排序时行的返回顺序没有保证，同一个种子可能抽到不同的告警
        pool = cur.fetchall()
    sample = random.Random(seed).sample(pool, min(n, len(pool)))
    ms: dict[str, list[float]] = {t: [] for t in TOOLS}
    timeouts: dict[str, int] = {t: 0 for t in TOOLS}
    refused: dict[str, int] = {t: 0 for t in TOOLS}
    calls = []                       # (耗时, 告警, 工具)；超时记为 inf，用于列出最慢的调用
    for a in sample:
        for name, sql, args in tool_calls(a, a["from_bank"], a["from_account"]):
            t1 = time.perf_counter()
            try:
                _, m = run(agent, sql, args)
                ms[name].append(m)
            except errors.QueryCanceled:
                timeouts[name] += 1
                m = float("inf")
            except errors.RaiseException as e:
                if not is_clearing_refusal(e):
                    raise
                refused[name] += 1          # 拒绝本身的耗时也计入：模型等的就是这么久
                m = (time.perf_counter() - t1) * 1000
                ms[name].append(m)
            calls.append((m, a["alert_id"], f"{a['from_bank']}:{a['from_account']}", name))
    print(f"    {'工具':<26}{'中位数':>9}{'P95':>9}{'最大':>9}{'超时':>6}{'清算层拒绝':>8}")
    summary = {}
    for name in TOOLS:
        v = sorted(ms[name])
        if not v:
            print(f"    {name:<26}{'—':>9}{'—':>9}{'—':>9}{timeouts[name]:>6}{refused[name]:>8}")
            continue
        p95 = v[min(len(v) - 1, int(0.95 * len(v)))]
        summary[name] = {"p50_ms": round(statistics.median(v)), "p95_ms": round(p95), "max_ms": round(v[-1]),
                         "timeouts": timeouts[name], "clearing_refused": refused[name]}
        print(f"    {name:<26}{statistics.median(v):>8.0f}ms{p95:>7.0f}ms{v[-1]:>7.0f}ms{timeouts[name]:>6}"
              f"{refused[name]:>8}")
    print("    最慢的 5 次调用：")
    for m, aid, acct, name in sorted(calls, reverse=True)[:5]:
        print(f"      告警 #{aid}  付款账户 {acct}  {name}  {'超时' if m == float('inf') else f'{m:,.0f} ms'}")
    check("延迟测试中没有工具超时（15 秒）", not any(timeouts.values()),
          ", ".join(f"{k} {v} 次" for k, v in timeouts.items() if v))
    slow = [k for k, s in summary.items() if s["p95_ms"] > 2000]
    if slow:
        print(f"    注意：{slow} 的 P95 超过 2 秒。不算失败，但 R9b 里一次调查调用 6 次工具，会直接拖慢分诊")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency", type=int, default=30, help="延迟测试抽样的告警数；0 = 跳过")
    args = ap.parse_args()

    with connect() as main_conn, connect_agent() as agent:
        main_conn.autocommit = True
        n_alerts = one(main_conn, "SELECT COUNT(*) FROM alerts")[0]
        if not n_alerts:
            raise SystemExit("alerts 表是空的：先运行 python scripts/alert_queue.py")
        print(f"告警队列 {n_alerts:,} 条")
        static_audit(main_conn)
        permission_test(agent)
        shown = walk_through(main_conn, agent, pick_alerts(main_conn))
        clearing_test(main_conn, agent)
        lat = latency(agent, main_conn, args.latency) if args.latency > 0 else {}

    n_fail = sum(not r["ok"] for r in results)
    print(f"\n共 {len(results)} 项检查，失败 {n_fail} 项")
    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    (PROJECT_ROOT / "reports" / "r9a_smoke.json").write_text(
        json.dumps({"checks": results, "latency": lat}, ensure_ascii=False, indent=1), encoding="utf-8")
    # 看过标签的告警：R9c 抽评测集时连同账户连通分量一起排除
    (PROJECT_ROOT / "reports" / "r9a_smoke_alerts.json").write_text(
        json.dumps({"alert_ids": shown, "reason": "R9a 冒烟测试中看过标签，R9c 评测抽样排除"}, ensure_ascii=False),
        encoding="utf-8")
    print("  结果写到 reports/r9a_smoke.json；看过标签的告警编号写到 reports/r9a_smoke_alerts.json")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()