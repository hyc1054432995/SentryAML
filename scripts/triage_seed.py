"""
SentryAML - R9b-3：分诊的确定性首轮（系统先查一轮，再交给模型）
================================================================
模型开始调查之前，系统按固定规则跑一组工具，把结果整理成一份"案卷"交给模型。和 R8 的系统首轮检索同理
（D67：agent 比 agent_noseed 更省、更准）：每条告警都要查的基础项，由代码保证查到、查对，模型的预算留给真正需要判断的补查。

四步：
  1. 选焦点账户：付款方是清算层就选收款方（D69：271 条告警的付款方是清算层，在它上面做路径分析是噪音）。
  2. 固定的一组查询（10 次）：告警这一笔（金额）、两端的来往史、两端的已知案件、焦点账户的收付画像 / 逐日活动 /
     过桥 / 回流 / 两跳结构、付款方的实体。外加最多 3 次"按模型理由补查"：SHAP 里推高分数的特征如果指向
     另一端账户（例如收款方前 3 天转入笔数），就在那一端也跑对应的工具。
  3. 由工具结果判定风险信号（规则写在 SIGNAL_RULES，每条都带出处 {step, path, value}）。
     每个信号有三种状态：lit（查实）/ checked（查了、没有）/ not_checked（没能查：被拒绝、超时、或首轮没有对应的工具）。
  4. 由查实的信号找可能的手法：kb_typologies_for_signals（特异度加权，D44），并标出每种手法还缺哪些信号、缺的是"查了没有"还是"没查"。

模型拿到的是：告警本身、模型给的理由（SHAP）、10 来次查询的原样结果（已登记，可被核验引用）、信号判定、候选手法。
SHAP 只是"模型为什么给这个分数"，是线索不是证据；证据是工具结果。两者在案卷里分开写。

首轮查询不占模型的工具预算，但照常登记（ctx.results）和留痕（agent_tool_calls），模型重复调用会被识别为重复。

用法（只能看 dev 告警或已排除的告警；test 冻结）：
    python scripts/triage_seed.py --alert 123        # 看一条告警的案卷
    python scripts/triage_seed.py --dev               # dev 60 条：信号在各层的出现比例（dev 可以看标签）
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

from agent_core import log_call
from triage_tools import TOOLS, TriageContext

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT = PROJECT_ROOT / "reports" / "r9c_split.json"
MAX_SHAP_EXTRA = 3
FAN_CP = 3            # 72 小时内不同对手方 ≥ 3 才算扇入 / 扇出（IBM 的定义是 k ≥ 2；≥ 3 排除"一来一回两个对手方"这种普通情形）。
                      # 事先写定，不在标签上调（D73）
# 知识图谱里的工具编号 → 分诊工具名（R3 的 kb_tools 与 R9a 的 inv_* 一一对应）
KG_TOOL = {"counterparty_history": "counterparty_history", "amount_profile": "amount_profile",
           "fan_profile": "fan_profile", "pass_through_check": "pass_through", "trace_cycle": "trace_cycle",
           "two_hop_structure": "two_hop", "account_activity_history": "account_activity",
           "entity_profile": "entity_profile", "case_history": "case_history"}
# 这些工具有 account 参数，可以针对另一端补查
PER_ACCOUNT = {"fan_profile", "account_activity", "pass_through", "trace_cycle", "two_hop", "entity_profile",
               "case_history"}


# ---------------------------------------------------------------------------
# 1–2. 选焦点、跑固定查询
# ---------------------------------------------------------------------------
def _kb(ctx: TriageContext, sql: str, args: tuple = ()) -> list[tuple]:
    with ctx.conn.cursor() as cur:
        cur.execute(sql, args)
        return cur.fetchall()


def run_seed_call(ctx: TriageContext, name: str, args: dict) -> dict:
    """系统发起的调用：不计入模型预算，照常登记、留痕；记下指纹，模型再发同样的调用会被识别为重复。"""
    ctx.step += 1
    t0 = time.perf_counter()
    tool = TOOLS[name]
    try:
        parsed = tool.args_model.model_validate(args)
    except ValidationError as e:              # 首轮的参数是代码写的，出错就是代码的错误：直接抛出
        raise RuntimeError(f"首轮参数错误 {name} {args}: {e}") from e
    result = tool.fn(ctx, parsed)
    ctx.calls[json.dumps([name, args], sort_keys=True, ensure_ascii=False)] = ctx.step
    log_call(ctx, f"seed:{name}", args, result, "tool_error" if "error" in result else None, t0)
    return result


def shap_signals(ctx: TriageContext) -> list[dict]:
    """SHAP 推高分数的特征 → 风险信号（kb_signals.features）。特征前缀 p_ / r_ / f_ 表示付款方 / 收款方 / 这一笔交易。"""
    feat2sig = defaultdict(list)
    for sid, name, feats in _kb(ctx, "SELECT signal_id, name, features FROM kb_signals"):
        for f in feats:
            feat2sig[f].append((sid, name))
    out = []
    for it in ctx.alert["shap"].get("pushes_up", []):
        f = it["feature"]
        party = {"p": "payer", "r": "payee"}.get(f[:1]) if f[1:2] == "_" else None
        for sid, name in feat2sig.get(f, []):
            out.append({"signal": sid, "name": name, "feature": f, "value": it["value"], "shap": it["shap"],
                        "party": party or "transaction"})
    return out


def build_seed(ctx: TriageContext) -> dict:
    a = ctx.alert
    clearing = {p: bool(_kb(ctx, "SELECT inv_is_clearing(%s, %s)", ctx.accounts[p])[0][0]) for p in ("payer", "payee")}
    focus = "payee" if clearing["payer"] and not clearing["payee"] else "payer"
    other = "payee" if focus == "payer" else "payer"

    plan = [("amount_profile", {}),
            ("counterparty_history", {}),
            ("case_history", {"account": "payer"}),
            ("case_history", {"account": "payee"}),
            ("fan_profile", {"account": focus}),
            ("account_activity", {"account": focus}),
            ("pass_through", {"account": focus}),
            ("trace_cycle", {"account": focus}),
            ("two_hop", {"account": focus}),
            ("entity_profile", {"account": "payer"})]
    # 按模型理由补查：SHAP 指向另一端账户的信号，在另一端跑它的工具（另一端是清算层时不补查路径类工具）
    sigs = shap_signals(ctx)
    tools_of = defaultdict(list)
    for sid, tid in _kb(ctx, "SELECT signal_id, tool_id FROM kb_signal_tools ORDER BY signal_id, tool_id"):
        tools_of[sid].append(KG_TOOL[tid])
    extra = []
    for s in sigs:
        if s["party"] != other or clearing[other]:
            continue
        for t in tools_of[s["signal"]]:
            if t in PER_ACCOUNT and (t, {"account": other}) not in plan + extra and len(extra) < MAX_SHAP_EXTRA:
                args = {"account": other}
                extra.append((t, args))
    calls = []
    for name, args in plan + extra:
        res = run_seed_call(ctx, name, args)
        calls.append({"step": ctx.step, "tool": name, "args": args, "result": res,
                      "why": "standard" if (name, args) in plan else "shap"})

    signals = judge_signals(ctx, calls)
    lit = sorted({e["signal"] for e in signals if e["status"] == "lit"})
    typ = []
    if lit:
        rows = _kb(ctx, "SELECT * FROM kb_typologies_for_signals(%s)", (lit,))
        rank = {"lit": 2, "checked": 1, "not_checked": 0}      # 同一信号有多条判定时，取最强的那条
        status: dict[str, str] = {}
        for e in signals:
            if rank[e["status"]] >= rank[status.get(e["signal"], "not_checked")]:
                status[e["signal"]] = e["status"]
        for tid, name, stage, matched, missing, score, cov in rows[:5]:
            typ.append({"typology": tid, "name": name, "stage": stage, "matched": matched,
                        "missing": [{"signal": m, "status": status.get(m, "not_checked")} for m in (missing or [])],
                        "score": float(score), "coverage": float(cov)})
    return {"alert_id": a["alert_id"], "focus": focus, "clearing": clearing, "shap_signals": sigs,
            "calls": calls, "signals": signals, "typologies": typ}


# ---------------------------------------------------------------------------
# 3. 由工具结果判定信号：每条判定都带出处，R9b-4 核验时可以原样引用
# ---------------------------------------------------------------------------
def judge_signals(ctx: TriageContext, calls: list[dict]) -> list[dict]:
    ev: list[dict] = []
    checked: set[tuple] = set()

    def add(signal, party, c, path, value, lit):
        ev.append({"signal": signal, "party": party, "status": "lit" if lit else "checked",
                   "step": c["step"], "path": path, "value": value})
        checked.add((signal, party))

    payee_key = f"{ctx.alert['to_bank']}:{ctx.alert['to_account']}"
    for c in calls:
        r, t = c["result"], c["tool"]
        party = c["args"].get("account", "transaction")
        if "error" in r:                                # 被拒绝或超时：这一项"没能查"
            for s in {"pass_through": ["s_pass_through"], "trace_cycle": ["s_cycle_3plus"],
                      "two_hop": ["s_common_intermediaries"]}.get(t, []):
                ev.append({"signal": s, "party": party, "status": "not_checked", "step": c["step"],
                           "path": "refused", "value": r.get("refused")})
            continue
        if t == "amount_profile":
            add("s_amount_band", "transaction", c, "result.in_laundering_band", r["result"]["in_laundering_band"],
                bool(r["result"]["in_laundering_band"]))
        elif t == "counterparty_history":
            add("s_first_contact", "pair", c, "a_to_b.n", r["a_to_b"]["n"], r["a_to_b"]["n"] == 0)
            add("s_round_trip_2hop", "pair", c, "b_to_a.n", r["b_to_a"]["n"], r["b_to_a"]["n"] > 0)
        elif t == "case_history":
            add("s_known_launderer", party, c, "n_cases", r["n_cases"], r["n_cases"] > 0)
        elif t == "fan_profile":
            add("s_fan_in_cp", party, c, "result.in_cp", r["result"]["in_cp"], r["result"]["in_cp"] >= FAN_CP)
            add("s_fan_out_cp", party, c, "result.out_cp", r["result"]["out_cp"], r["result"]["out_cp"] >= FAN_CP)
        elif t == "account_activity":
            add("s_dormant_reactivation", party, c, "txns_earlier_days", r["txns_earlier_days"],
                r["txns_earlier_days"] == 0 and r["txns_last_2_days"] > 0)
        elif t == "pass_through":
            add("s_pass_through", party, c, "n_matches", r["n_matches"], r["n_matches"] > 0)
        elif t == "trace_cycle":
            i3 = next((i for i, x in enumerate(r["cycles"]) if x["hops"] >= 3), None)
            if i3 is None:
                add("s_cycle_3plus", party, c, "n_cycles", r["n_cycles"], False)
            else:
                add("s_cycle_3plus", party, c, f"cycles[{i3}].hops", r["cycles"][i3]["hops"], True)
            i2 = next((i for i, x in enumerate(r["cycles"]) if x["hops"] == 2), None)
            if i2 is not None:
                add("s_round_trip_2hop", party, c, f"cycles[{i2}].hops", 2, True)
        elif t == "two_hop":
            add("s_common_intermediaries", party, c, "n_gather_out", r["n_gather_out"], r["n_gather_out"] > 0)
            add("s_common_intermediaries", party, c, "n_gather_in", r["n_gather_in"], r["n_gather_in"] > 0)
        elif t == "entity_profile" and r["result"]:
            others = r["result"]["other_accounts"] or []
            add("s_same_owner", party, c, "result.other_accounts", others, payee_key in others)
    return ev


# ---------------------------------------------------------------------------
# 命令行：只看 dev（或已排除）的告警
# ---------------------------------------------------------------------------
def allowed_ids() -> tuple[set[int], dict[int, str]]:
    sp = json.loads(SPLIT.read_text(encoding="utf-8"))
    return {x["alert_id"] for x in sp["dev"]} | set(sp["excluded_alerts"]), \
        {x["alert_id"]: x["stratum"] for x in sp["dev"]}


def show(seed: dict, ctx: TriageContext) -> None:
    a = ctx.alert
    print(f"告警 #{a['alert_id']}  {a['tier']}  分位 {a['score_pct']:.3f}  ${float(a['amount_usd']):,.2f}  {a['txn_ts']}")
    print(f"  付款方 {ctx.accounts['payer']}{'（清算层）' if seed['clearing']['payer'] else ''} → "
          f"收款方 {ctx.accounts['payee']}{'（清算层）' if seed['clearing']['payee'] else ''}；焦点 {seed['focus']}")
    print("  模型理由（SHAP 推高分数）→ 对应信号：")
    for it in a["shap"].get("pushes_up", []):
        sig = [s["signal"] for s in seed["shap_signals"] if s["feature"] == it["feature"]]
        print(f"    {it['feature']:<22}{it['value']:>16}  {it['shap']:+.3f}  {sig or ''}")
    print("  首轮查询：")
    for c in seed["calls"]:
        r = c["result"]
        brief = r.get("error") or json.dumps({k: v for k, v in r.items()
                                              if k not in ("step", "tool", "as_of", "account", "note", "days")},
                                             ensure_ascii=False)[:150]
        print(f"    #{c['step']:<2} {c['tool']}{c['args'] or ''}{'  [SHAP 补查]' if c['why'] == 'shap' else ''}\n"
              f"        {brief}")
    print("  信号判定：")
    for e in seed["signals"]:
        mark = {"lit": "●", "checked": "○", "not_checked": "?"}[e["status"]]
        print(f"    {mark} {e['signal']:<26}{e['party']:<12}step {e['step']}  {e['path']} = {e['value']}")
    print("  候选手法（由查实的信号，特异度加权）：")
    for t in seed["typologies"]:
        miss = ", ".join(f"{m['signal']}({m['status']})" for m in t["missing"])
        print(f"    {t['typology']:<28} score {t['score']:.2f} coverage {t['coverage']:.2f}  缺：{miss}")
    if not seed["typologies"]:
        print("    （没有查实的信号对应到任何手法）")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--alert", type=int)
    g.add_argument("--dev", action="store_true")
    a = ap.parse_args()
    ok, strata = allowed_ids()
    if a.alert is not None:
        if a.alert not in ok:
            raise SystemExit(f"告警 #{a.alert} 不在 dev 里（或者属于冻结的 test）：首轮调试只看 dev 告警")
        ctx = TriageContext.load(a.alert, log=False)
        show(build_seed(ctx), ctx)
        ctx.close()
        return
    # dev 60 条：信号在各层的出现比例（只读 dev 的标签）
    from db import connect
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT alert_id, is_laundering FROM alert_labels WHERE alert_id = ANY(%s)", (list(strata),))
        label = dict(cur.fetchall())
    count, n, ms = defaultdict(Counter), Counter(), []
    typ_top = defaultdict(Counter)
    for aid in sorted(strata):
        t0 = time.perf_counter()
        ctx = TriageContext.load(aid, log=False)
        seed = build_seed(ctx)
        ctx.close()
        ms.append((time.perf_counter() - t0) * 1000)
        st = strata[aid]
        n[st] += 1
        for s in {e["signal"] for e in seed["signals"] if e["status"] == "lit"}:
            count[st][s] += 1
        count[st]["(焦点是收款方)"] += seed["focus"] == "payee"
        count[st]["(有路径工具被拒绝)"] += any(e["status"] == "not_checked" for e in seed["signals"])
        typ_top[st][seed["typologies"][0]["typology"] if seed["typologies"] else "(无)"] += 1
    sts = ["T1/L", "T1/N", "T2/L", "T2/N"]
    print(f"dev {sum(n.values())} 条；首轮每条耗时中位数 {sorted(ms)[len(ms) // 2]:.0f} ms、最大 {max(ms):.0f} ms\n")
    print(f"  {'信号（查实的比例）':<30}" + "".join(f"{s:>9}" for s in sts))
    names = sorted({k for c in count.values() for k in c}, key=lambda k: (k.startswith("("), k))
    for k in names:
        print(f"  {k:<30}" + "".join(f"{count[s][k] / n[s]:>9.0%}" if n[s] else f"{'—':>9}" for s in sts))
    print("\n  排第一的候选手法：")
    for s in sts:
        print(f"    {s}：" + "，".join(f"{k} {v}" for k, v in typ_top[s].most_common(4)))
    print(f"\n  读这张表的方式：各层样本只有 10–20 条，单条就是 5–10 个百分点。它用来检查首轮"
          f"有没有把证据查出来，不用来调规则（D73）。")


if __name__ == "__main__":
    main()