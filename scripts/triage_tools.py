"""
SentryAML - R9b-2：分诊 Agent 的工具层
======================================
把 R9a 的 9 个调查函数（sql/020、021）包装成模型可以调用的工具。这一层不调用大模型；
循环、预算、重复检测、留痕都在 agent_core.py（R9b-1），这里只定义"分诊特有"的部分。

模型和数据库之间隔着这一层包装，它做六件事（每一件对应一个具体的风险）：
  1. 校验参数（pydantic，extra="forbid"）：参数名写错、多给参数、窗口超出范围，都作为错误返回给模型改正。
  2. 绑定告警时刻：所有函数的 p_as_of 由这里填入 ctx.as_of（告警发生的那一分钟），模型的参数里根本没有这一项，
     想看告警之后的交易也无从下手（D68 设计原则一）。
  3. 解析账户：模型不能直接写账户号，只能写 "payer"、"payee"，或者前面某次工具结果里出现过的账户。
     原因见 resolve_account 的说明：写错的账户号会查出 0 行，而 0 行会被读成"没有活动"。
  4. 区分三种结果：查了、有发现 / 查了、没有发现（0 行）/ 没能查（清算层拒绝、超时）。
     第三种作为带 "error" 的观察返回，绝不伪装成 0 行（D69）。
  5. 统一结果格式：金额保留两位小数、时间写成 "YYYY-MM-DD HH:MM"，结构固定、可以按路径定位
     （例如 "result.out_amt"、"matches[0].lag_hours"），并按步骤号登记到 ctx.results——
     R9b-4 核验分诊理由里的数字时，就按 {step, path, value} 到这里取值比对。
  6. 用最小权限角色执行（connect_agent，sentry_agent）：即使这一层写错，也读不到标签（D68 设计原则二）。

用法：
    python scripts/triage_tools.py --schemas                # 打印模型看到的工具定义
    python scripts/triage_tools.py --smoke                  # 在 R9a 冒烟测试用过的告警上逐个调用工具
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from psycopg import errors
from pydantic import BaseModel, ConfigDict, Field

import agent_core
from agent_core import RunContext, Tool
from db import connect, connect_agent

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAX_CALLS = 8


# ---------------------------------------------------------------------------
# 运行上下文
# ---------------------------------------------------------------------------
@dataclass
class TriageContext(RunContext):
    """
    一条告警的一次调查。as_of = 告警交易的时刻（datetime），由 load() 从 alerts 表读出后绑定。
      accounts  账户池：模型可以查的账户。开始只有 payer / payee 两个，工具结果里出现的账户陆续加入
      results   事实登记表：步骤号 → 这一步交给模型的完整结果（只登记成功的结果）
    """
    kind: str = "triage"
    max_calls: int = DEFAULT_MAX_CALLS
    alert: dict = field(default_factory=dict)
    accounts: dict[str, tuple[int, str]] = field(default_factory=dict)    # "70:1004286A8" → (70, "1004286A8")
    results: dict[int, dict] = field(default_factory=dict)
    conn: Any = None
    budget_note: str = ("Submit your triage decision now with the facts you have, and state what you "
                        "could not check.")
    duplicate_hint: str = "Change the account or the window."

    @classmethod
    def load(cls, alert_id: int, **kw) -> "TriageContext":
        conn = connect_agent()
        with conn.cursor() as cur:
            cur.execute("""SELECT alert_id, txn_id, txn_ts, from_bank, from_account, to_bank, to_account,
                                  amount_usd, score, score_pct, tier, shap FROM alerts WHERE alert_id = %s""",
                        (alert_id,))
            row = cur.fetchone()
            if row is None:
                conn.close()
                raise ValueError(f"告警 #{alert_id} 不存在")
            cols = [d.name for d in cur.description]
        alert = dict(zip(cols, row))
        ctx = cls(question=f"alert #{alert_id}", as_of=alert["txn_ts"], alert=alert, conn=conn, **kw)
        ctx.accounts = {"payer": (alert["from_bank"], alert["from_account"]),
                        "payee": (alert["to_bank"], alert["to_account"])}
        for b, a in ctx.accounts.copy().values():
            ctx.accounts[f"{b}:{a}"] = (b, a)
        return ctx

    def start(self) -> "TriageContext":
        if self.log:
            with connect() as conn, conn.cursor() as cur:
                cur.execute("""INSERT INTO agent_runs (run_id, kind, question, as_of, alert_id)
                               VALUES (%s, %s, %s, %s, %s)""",
                            (self.run_id, self.kind, self.question, self.as_of, self.alert.get("alert_id")))
                conn.commit()
        return self

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def summary_extra(self) -> dict:
        return {"alert_id": self.alert.get("alert_id"), "accounts_seen": len(set(self.accounts.values()))}


# ---------------------------------------------------------------------------
# 账户解析：模型只能查"账户池"里的账户
# ---------------------------------------------------------------------------
class UnknownAccount(Exception):
    pass


def resolve_account(ctx: TriageContext, ref: str) -> tuple[str, int, str]:
    """
    "payer" / "payee" / "bank:account" → (标准写法, bank, account)。不在账户池里就报错，由模型改正。

    为什么不让模型直接写任意账户号：
      a) 写错一位（或凭印象编一个看起来像的账户号），数据库里没有这个账户，所有工具都返回 0 行；
         0 行在模型眼里是"这个账户没有活动、没有回流、没有案件"——一个查错了的账户会被当成排除嫌疑的证据。
      b) 调查范围要可审计：每一个被查的账户都能追溯到"它是从哪一步的结果里出现的"。
      c) 与 R8 的证据池同理：回答只能引用看过的条款，分诊只能调查出现过的账户。
    """
    key = ref.strip()
    if key.lower() in ("payer", "payee"):
        key = key.lower()
    if key not in ctx.accounts:
        raise UnknownAccount(key)
    bank, acct = ctx.accounts[key]
    return f"{bank}:{acct}", bank, acct


# 结果里哪些字段是账户：出现过的账户加入账户池
_ACCOUNT_FIELDS = ("in_from", "out_to", "endpoint", "intermediaries", "path", "other_accounts")
_ACCOUNT_RE = re.compile(r"^(\d+):([0-9A-Za-z]+)$")


def _collect_accounts(ctx: TriageContext, obj) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in _ACCOUNT_FIELDS:
                for s in (v if isinstance(v, list) else [v]):
                    m = _ACCOUNT_RE.match(s) if isinstance(s, str) else None
                    if m:
                        ctx.accounts.setdefault(s, (int(m.group(1)), m.group(2)))
            else:
                _collect_accounts(ctx, v)
    elif isinstance(obj, list):
        for v in obj:
            _collect_accounts(ctx, v)


# ---------------------------------------------------------------------------
# 结果格式：固定、可按路径定位
# ---------------------------------------------------------------------------
def _norm(v):
    """金额统一两位小数（四舍五入到分）、时间到分钟。模型看到的和登记表里存的是同一个值，核验时才能逐字比较。"""
    if isinstance(v, Decimal):
        return float(v.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, list):
        return [_norm(x) for x in v]
    return v


def _rows(ctx: TriageContext, sql: str, args: tuple) -> list[dict]:
    with ctx.conn.cursor() as cur:
        cur.execute(sql, args)
        cols = [d.name for d in cur.description]
        return [{c: _norm(v) for c, v in zip(cols, r)} for r in cur.fetchall()]


_PATH_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")


def resolve_path(obj, path: str):
    """按路径取值："result.out_amt"、"matches[0].lag_hours"、"a_to_b.n"。路径不存在时抛 KeyError（R9b-4 的核验用）。"""
    cur = obj
    for name, idx in _PATH_RE.findall(path.replace(".", " ")):
        if name:
            if not isinstance(cur, dict) or name not in cur:
                raise KeyError(path)
            cur = cur[name]
        else:
            i = int(idx)
            if not isinstance(cur, list) or i >= len(cur):
                raise KeyError(path)
            cur = cur[i]
    return cur


# ---------------------------------------------------------------------------
# 参数定义
# ---------------------------------------------------------------------------
ACCOUNT_DESC = ("Which account: 'payer' or 'payee' of the alerted transaction, or an account id exactly as it "
                "appeared in an earlier tool result (format 'bank:account', e.g. '1292:800432D80').")


class AccountArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account: str = Field(description=ACCOUNT_DESC)


class ActivityArgs(AccountArgs):
    days: int = Field(default=7, ge=1, le=14, description="How many days back, including the alert day (1-14).")


class WindowArgs(AccountArgs):
    hours: int = Field(default=72, ge=6, le=168, description="Look-back window in hours before the alert (6-168).")


class PairArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_a: str = Field(default="payer", description=ACCOUNT_DESC)
    account_b: str = Field(default="payee", description=ACCOUNT_DESC)


class NoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# 工具实现：每个函数返回交给模型的 dict；预期内的问题放在 "error" 里
# ---------------------------------------------------------------------------
CLEARING_NOTE = ("This is a clearing-layer account (more than 200 historical counterparties). Path analysis on it "
                 "is noise: it matches unrelated payments by chance. Investigate the other party of the alert, or "
                 "an account that appeared in earlier results.")


def _unknown(ctx: TriageContext, e: UnknownAccount) -> dict:
    known = [k for k in ctx.accounts if k not in ("payer", "payee")]
    return {"error": f"unknown account '{e}'. You can investigate 'payer', 'payee', or an account that "
                     f"appeared in an earlier result: {', '.join(known[:30])}"}


def _run(ctx: TriageContext, tool: str, account: str | None, body) -> dict:
    """公共外壳：解析账户 → 执行 → 把可以预期的数据库情况转成观察 → 登记结果、收集新账户。"""
    try:
        acct = resolve_account(ctx, account) if account is not None else None
    except UnknownAccount as e:
        return _unknown(ctx, e)
    try:
        res = body(*(acct[1:] if acct else ()))
    except errors.RaiseException as e:
        if str(e).startswith("clearing_layer"):
            return {"error": "not run: clearing-layer account", "account": acct[0], "refused": "clearing_layer",
                    "note": CLEARING_NOTE}
        raise
    except errors.QueryCanceled:
        return {"error": "not run: the query took longer than the time limit", "account": acct[0] if acct else None,
                "refused": "timeout", "note": "Try a shorter window, or another account."}
    out = {"step": ctx.step, "tool": tool, "as_of": ctx.as_of.strftime("%Y-%m-%d %H:%M")}
    if acct:
        out["account"] = acct[0]
    out.update(res)
    ctx.results[ctx.step] = out
    _collect_accounts(ctx, res)
    return out


def account_activity(ctx: TriageContext, a: ActivityArgs) -> dict:
    def body(bank, acct):
        days = _rows(ctx, "SELECT * FROM inv_account_activity(%s, %s, %s, %s)", (bank, acct, ctx.as_of, a.days))
        active = sum(1 for d in days if d["in_cnt"] or d["out_cnt"])
        return {"window_days": a.days, "active_days": active, "days": days}
    return _run(ctx, "account_activity", a.account, body)


def fan_profile(ctx: TriageContext, a: WindowArgs) -> dict:
    def body(bank, acct):
        r = _rows(ctx, "SELECT * FROM inv_fan_profile(%s, %s, %s, %s)", (bank, acct, ctx.as_of, a.hours))
        return {"result": r[0]}
    return _run(ctx, "fan_profile", a.account, body)


def counterparty_history(ctx: TriageContext, a: PairArgs) -> dict:
    try:
        ka, ba, aa = resolve_account(ctx, a.account_a)
        kb, bb, ab = resolve_account(ctx, a.account_b)
    except UnknownAccount as e:
        return _unknown(ctx, e)

    def body():
        rows = _rows(ctx, "SELECT * FROM inv_counterparty_history(%s, %s, %s, %s, %s)", (ba, aa, bb, ab, ctx.as_of))
        by = {r.pop("direction"): r for r in rows}
        return {"account_a": ka, "account_b": kb, "a_to_b": by["a_to_b"], "b_to_a": by["b_to_a"],
                "note": "Counts only transactions strictly before the alert time; the alerted transaction itself "
                        "is not included."}
    return _run(ctx, "counterparty_history", None, body)


def pass_through(ctx: TriageContext, a: WindowArgs) -> dict:
    def body(bank, acct):
        m = _rows(ctx, "SELECT * FROM inv_pass_through(%s, %s, %s, %s)", (bank, acct, ctx.as_of, a.hours))
        return {"window_hours": a.hours, "tolerance_pct": 10, "n_matches": len(m), "matches": m}
    return _run(ctx, "pass_through", a.account, body)


def trace_cycle(ctx: TriageContext, a: AccountArgs) -> dict:
    def body(bank, acct):
        c = _rows(ctx, "SELECT * FROM inv_trace_cycle(%s, %s, %s)", (bank, acct, ctx.as_of))
        return {"window_days": 7, "max_hops": 12, "n_cycles": len(c), "cycles": c}
    return _run(ctx, "trace_cycle", a.account, body)


def two_hop(ctx: TriageContext, a: WindowArgs) -> dict:
    def body(bank, acct):
        rows = _rows(ctx, "SELECT * FROM inv_two_hop(%s, %s, %s, %s)", (bank, acct, ctx.as_of, a.hours))
        return {"window_hours": a.hours,
                "gather_out": [{k: v for k, v in r.items() if k != "direction"} for r in rows
                               if r["direction"] == "gather_out"],
                "gather_in": [{k: v for k, v in r.items() if k != "direction"} for r in rows
                              if r["direction"] == "gather_in"]}
    return _run(ctx, "two_hop", a.account, body)


def entity_profile(ctx: TriageContext, a: AccountArgs) -> dict:
    def body(bank, acct):
        r = _rows(ctx, "SELECT * FROM inv_entity_profile(%s, %s)", (bank, acct))
        return {"result": r[0] if r else None}
    return _run(ctx, "entity_profile", a.account, body)


def case_history(ctx: TriageContext, a: AccountArgs) -> dict:
    def body(bank, acct):
        c = _rows(ctx, "SELECT * FROM inv_case_history(%s, %s, %s)", (bank, acct, ctx.as_of))
        return {"n_cases": len(c), "cases": c,
                "note": "Only cases confirmed before the alert time are visible."}
    return _run(ctx, "case_history", a.account, body)


def amount_profile(ctx: TriageContext, a: NoArgs) -> dict:
    def body():
        r = _rows(ctx, "SELECT * FROM inv_amount_profile(%s)", (ctx.alert["txn_id"],))
        return {"txn_id": ctx.alert["txn_id"], "result": r[0]}
    return _run(ctx, "amount_profile", None, body)


# ---------------------------------------------------------------------------
# 工具登记表：说明是写给模型的，它完全依靠这几句话决定什么时候用哪个工具、怎么读结果
# ---------------------------------------------------------------------------
_TIME = "All tools see only transactions up to the alert time; nothing after it is visible."
TOOLS = {t.name: t for t in [
    Tool("account_activity",
         "Day-by-day incoming and outgoing counts, distinct counterparties and amounts of an account for the last "
         "N days up to the alert. Use it to see whether the account was dormant and suddenly became active, or "
         "whether the alert fits a regular pattern (the same payments every day). " + _TIME,
         ActivityArgs, account_activity),
    Tool("fan_profile",
         "Incoming and outgoing counts, distinct counterparties and amounts of an account in the last N hours, "
         "plus its number of distinct counterparties over its whole history before the alert and whether it is "
         "a clearing-layer account. Use it to check fan-in (many senders) or fan-out (many receivers). " + _TIME,
         WindowArgs, fan_profile),
    Tool("counterparty_history",
         "Earlier transactions between two accounts, in each direction, strictly before the alert. Defaults to the "
         "payer and payee of the alert. n = 0 in both directions means this is their first contact. " + _TIME,
         PairArgs, counterparty_history),
    Tool("pass_through",
         "Checks whether money received by the account was sent on within N hours in a single payment of a "
         "similar amount (within 10%), matched one incoming payment to one outgoing payment. Returning the money to "
         "the sender does not count. n_matches = 0 means no such pair was found. Refused for clearing-layer "
         "accounts. " + _TIME,
         WindowArgs, pass_through),
    Tool("trace_cycle",
         "Follows money forward from the account, in time order, within 7 days and up to 12 hops, and reports "
         "paths that come back to the account (round-tripping). n_cycles = 0 means no such path was found. "
         "Refused for clearing-layer accounts. " + _TIME,
         AccountArgs, trace_cycle),
    Tool("two_hop",
         "Two-hop structure around the account in the last N hours. gather_out: money the account sent to "
         "different intermediaries that then reached the same endpoint (scatter-gather). gather_in: money from one "
         "source that reached the account through different intermediaries. Only endpoints reached through at "
         "least 2 intermediaries are listed. Refused for clearing-layer accounts. " + _TIME,
         WindowArgs, two_hop),
    Tool("entity_profile",
         "The registered entity that owns the account: entity type, how many accounts and banks it has, and its "
         "other accounts. Use it to see whether money moves between accounts of the same owner.",
         AccountArgs, entity_profile),
    Tool("case_history",
         "Confirmed money-laundering cases involving the account (as payer or payee), confirmed before the alert "
         "time, with the laundering patterns involved. n_cases = 0 means no confirmed case is on record.",
         AccountArgs, case_history),
    Tool("amount_profile",
         "The alerted transaction itself: amount in USD, whether it is cross-currency, and whether the amount "
         "falls in the range where most laundering amounts in the training period fell (2,154-24,075 USD).",
         NoArgs, amount_profile),
]}


def tool_schemas() -> list[dict]:
    return [t.schema() for t in TOOLS.values()]


def call_tool(ctx: TriageContext, name: str, raw_args: str | dict) -> dict:
    return agent_core.call_tool(ctx, TOOLS, name, raw_args)


# ---------------------------------------------------------------------------
# 冒烟测试：只用 R9a 冒烟测试已经看过标签、R9c 会排除的告警（不在评测候选告警上看工具输出）
# ---------------------------------------------------------------------------
def smoke() -> None:
    shown = json.loads((PROJECT_ROOT / "reports" / "r9a_smoke_alerts.json").read_text(encoding="utf-8"))["alert_ids"]
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"  {'✓' if ok else '✗'} {name}" + (f"   {detail}" if detail else ""))

    def brief(res):
        if "error" in res:
            return f"error: {res['error']}"
        keys = [k for k in res if k not in ("step", "tool", "as_of", "account", "note")]
        return json.dumps({k: res[k] for k in keys}, ensure_ascii=False)[:220]

    aid = shown[0]
    ctx = TriageContext.load(aid, log=False, max_calls=20)
    print(f"[1] 告警 #{aid}（R9a 已看过标签、R9c 排除）：告警时刻 {ctx.as_of}，"
          f"付款方 {ctx.accounts['payer']}，收款方 {ctx.accounts['payee']}")
    for name, args in [("amount_profile", {}), ("counterparty_history", {}),
                       ("fan_profile", {"account": "payer"}), ("account_activity", {"account": "payer", "days": 7}),
                       ("pass_through", {"account": "payer"}), ("trace_cycle", {"account": "payer"}),
                       ("two_hop", {"account": "payer"}), ("entity_profile", {"account": "payee"}),
                       ("case_history", {"account": "payer"})]:
        t0 = time.perf_counter()
        r = call_tool(ctx, name, json.dumps(args))
        print(f"    #{ctx.step} {name}{args}  {(time.perf_counter() - t0) * 1000:.0f} ms\n        → {brief(r)}")
        check(f"{name} 成功并登记", "error" not in r and ctx.results.get(ctx.step) is r)

    print("\n[2] 账户池")
    found = [k for k in ctx.accounts if k not in ("payer", "payee")]
    check("工具结果里出现的账户加入了账户池", len(found) >= 2, f"共 {len(found)} 个，例如 {found[:4]}")
    r = call_tool(ctx, "fan_profile", json.dumps({"account": "999:FAKE00001"}))
    check("不在账户池里的账户被拒绝", "unknown account" in r.get("error", ""), r.get("error", "")[:90])
    discovered = next((k for k in found if k != f"{ctx.alert['from_bank']}:{ctx.alert['from_account']}"
                       and k != f"{ctx.alert['to_bank']}:{ctx.alert['to_account']}"), None)
    if discovered:
        r = call_tool(ctx, "fan_profile", json.dumps({"account": discovered}))
        check("结果里出现过的账户可以继续查", "error" not in r, f"{discovered}: {brief(r)[:120]}")

    print("\n[3] 参数校验与时点绑定")
    r = call_tool(ctx, "fan_profile", json.dumps({"account": "payer", "as_of": "2022-09-15 00:00"}))
    check("模型不能传入 as_of（多余参数被拒绝）", r.get("error") == "invalid arguments", str(r.get("details"))[:100])
    r = call_tool(ctx, "fan_profile", json.dumps({"account": "payer", "hours": 1000}))
    check("窗口超出范围被拒绝", r.get("error") == "invalid arguments", str(r.get("details"))[:100])
    late = [s for s, res in ctx.results.items() if res["as_of"] != ctx.as_of.strftime("%Y-%m-%d %H:%M")]
    check("所有登记的结果都绑定在告警时刻", not late)

    print("\n[4] 事实登记表与路径定位（R9b-4 核验用）")
    step_fan = next(s for s, res in ctx.results.items() if res["tool"] == "fan_profile")
    v = resolve_path(ctx.results[step_fan], "result.out_amt")
    check("按 {step, path} 取得数值", isinstance(v, (int, float)), f"step {step_fan} result.out_amt = {v}")
    try:
        resolve_path(ctx.results[step_fan], "result.no_such_field")
        check("不存在的路径报错", False)
    except KeyError:
        check("不存在的路径报错", True)
    ctx.close()

    print("\n[5] 清算层账户：拒绝是带 error 的观察，不是 0 行")
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT a.alert_id FROM alerts a JOIN account_degrees d
                         ON d.bank_id = a.from_bank AND d.account_number = a.from_account
                       WHERE d.total_cp_count > 200 ORDER BY a.alert_id LIMIT 1""")
        row = cur.fetchone()
    if row:
        c2 = TriageContext.load(row[0], log=False)
        for name in ("pass_through", "trace_cycle", "two_hop"):
            r = call_tool(c2, name, json.dumps({"account": "payer"}))
            check(f"清算层付款方 {name} → 明确拒绝", r.get("refused") == "clearing_layer" and c2.step not in c2.results,
                  r.get("error", ""))
        r = call_tool(c2, "pass_through", json.dumps({"account": "payee"}))
        check("同一条告警的收款方照常执行", "error" not in r, brief(r)[:100])
        c2.close()
        print(f"    （告警 #{row[0]} 只用于测试拒绝，没有读标签）")
    print(f"\n共 {len(results)} 项，失败 {results.count(False)} 项")
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="分诊 Agent 的工具层")
    ap.add_argument("--schemas", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.schemas:
        print(json.dumps(tool_schemas(), ensure_ascii=False, indent=2))
    elif a.smoke:
        smoke()
    else:
        ap.print_help()