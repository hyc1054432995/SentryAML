"""
SentryAML - R9b-4：告警分诊 Agent（提交格式、数字事实核验、完整循环）
=====================================================================
triage(alert_id) 调查一条告警，给出分诊结论：升级（escalate）还是关闭（close）、可能的手法、理由和事实。

一次运行：
  0. 把告警本身登记为第 0 步（金额、时间、档位、分位）：理由里提到告警的数字也要有出处。
  1. 确定性首轮（triage_seed，D73）：系统跑 10 来次查询、判定信号、找候选手法，整理成"案卷"作为第一条用户消息。
  2. 循环（agent_core.run_loop，D70）：模型读案卷，可以再调用工具补查（最多 MAX_CALLS 次），最后调用 submit_triage。
  3. 提交时核验（与 R8 的逐字核验对应，D68 设计原则四）：
       facts      每条 {step, path, value}：第 step 步的登记结果里，按 path 取出的值必须等于 value
                  （数字按两位小数比较，字符串逐字比较）。
       rationale  理由里出现的每个数字，都必须能在这次调查登记过的结果里找到（允许按写出的精度四舍五入，
                  例如 64,430 对应 64429.61）。日期、时刻、账户号、"step 3" 这类编号不算数字。
     有不通过的 → 把问题清单作为这次工具调用的结果交给模型，允许改正一次；第二次仍不通过就照实记录（标记，不伪装成通过）。

为什么理由里的数字也要核验：facts 是模型自己挑出来的"证据清单"，理由才是分析员真正读的文字。
  只核验 facts、不管理由，模型可以在 facts 里写对的数字、在理由里写一个记错的数字，而读者只看理由。

用法（只能跑 dev 告警或已排除的告警；test 冻结）：
    python scripts/triage_agent.py --selftest          # 用脚本化的假模型测核验逻辑：不调用 API
    python scripts/triage_agent.py --alert 3437        # 跑一条 dev 告警（调用 API）
    python scripts/triage_agent.py --dev 8             # 跑 8 条 dev 告警（四层轮流取），打印结论与标签对照
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import llm
from agent_core import add_usage, inline_schema, new_usage, run_loop
from db import connect
from triage_seed import build_seed
from triage_tools import TOOLS, TriageContext, call_tool, resolve_path, tool_schemas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT = PROJECT_ROOT / "reports" / "r9c_split.json"
AGENT_VERSION = "triage-v2.1"   # v2.1（D75）：输出截断时提醒一次再判失败；提示词与 v2 相同。v2（D74）：澄清"规律性模式"指告警这一笔本身在重复，首次来往不可能是规律性付款
MAX_CALLS = 6          # 模型自己发起的工具调用（不含首轮、不含提交）
MAX_TURNS = 12
MAX_SUBMITS = 2
TYPOLOGIES = ["fan_in", "fan_out", "gather_scatter", "scatter_gather", "cycle", "random_walk", "bipartite", "stack",
              "dormant_activation", "structuring", "known_launderer_integration", "none", "unclear"]


# ---------------------------------------------------------------------------
# 提交格式
# ---------------------------------------------------------------------------
class FactArg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    step: int = Field(description="Step number of the tool result the fact comes from (0 = the alert itself).")
    path: str = Field(description="Path of the field in that result, e.g. 'result.out_cp', 'n_matches', 'a_to_b.n', "
                                  "'matches[0].lag_hours', 'cases[0].pattern_types'.")
    value: str | float | int | bool | list | None = Field(description="The value of that field, copied exactly.")


class SubmitTriage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["escalate", "close"] = Field(
        description="escalate = send for full investigation; close = not suspicious.")
    typology: Literal[tuple(TYPOLOGIES)] = Field(
        description="The laundering pattern that best fits the evidence; 'none' when closing; 'unclear' when "
                    "escalating without a clear pattern.")
    confidence: Literal["low", "medium", "high"]
    facts: list[FactArg] = Field(min_length=1, max_length=10,
                                 description="The evidence the decision rests on, each checked automatically.")
    rationale: str = Field(description="2 to 5 sentences for the analyst. Every number must come from the tool "
                                       "results of this investigation.")
    not_checked: str = Field(description="What could not be checked that could have changed the decision; "
                                         "empty string if nothing.")


SUBMIT_TOOL = {"type": "function", "function": {
    "name": "submit_triage",
    "description": "Submit the triage decision and finish. Each fact is checked against the stored tool results, "
                   "and every number in the rationale must appear in the tool results of this investigation. "
                   "If something fails, the result lists it and you may correct and resubmit once.",
    "parameters": inline_schema(SubmitTriage.model_json_schema())}}


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------
SYSTEM = """You are an AML alert triage analyst at a bank. A transaction-monitoring model raised an alert on one \
transaction. Decide whether to ESCALATE the alert for full investigation or CLOSE it as not suspicious, based on \
evidence from tools.

What you receive:
- The alert (step 0): the transaction, its amount and time, the model's tier and score percentile.
- The model's reasons (SHAP): features that pushed the score up. They are clues about where to look, not evidence: \
the model learned statistical associations, and some of them (for example first contact between two accounts) are \
also common in normal payments.
- The results of a standard first round of checks run by the system (steps 1 onward), and the risk signals judged \
from them: lit (found), checked (looked, not found), not_checked (could not look).
- Candidate laundering patterns that fit the lit signals, with the signals each pattern is still missing.

How to investigate:
1. Read the first-round results before calling any tool.
2. Call tools only to fill gaps that could change your decision: a strong candidate pattern with not_checked signals, \
an account that appeared in the results and may be part of the scheme (for example the sender in a pass-through \
match), or the other party of the alert. You have at most {max_calls} tool calls besides submit_triage; fewer is \
better.
3. All tools see only transactions up to the alert time. A result of 0 means the tool looked and found nothing. A \
result with "refused" or "timeout" means nothing was checked: never treat it as evidence of innocence.

How to decide:
4. Strong evidence: a confirmed laundering case on either account before the alert; pass-through (money received and \
sent on in a matching amount); cycles of 3 or more hops; money moving through several intermediaries to the same \
endpoint. Weak evidence on its own: first contact, an amount in the typical laundering range, a single fan-in or \
fan-out count. Weak evidence appears in many normal alerts.
5. ESCALATE if strong evidence is found, or if a strong candidate pattern cannot be ruled out because key checks \
could not be run. CLOSE if the checks that matter were run and found nothing beyond weak evidence, or if the alerted \
payment itself is part of a regular pattern: the same payer has paid the same payee similar amounts repeatedly before \
the alert. A first contact (no earlier payment from payer to payee) is never a regular pattern, whatever else the \
accounts do.
6. typology: the pattern that best fits the evidence; "none" when closing; "unclear" when escalating without a clear \
pattern.

How to submit (call submit_triage once, or twice if the first submission fails the check):
7. facts: the evidence your decision rests on (1 to 10 items). Each fact is a step number, the path of a field in that \
step's result, and its value copied exactly.
8. rationale: 2 to 5 sentences. Every number in it must come from the tool results of this investigation; rounding is \
allowed (64,430 for 64429.61). Do not use numbers from outside knowledge.
9. not_checked: what could not be checked that could have changed the decision; empty string if nothing."""

USER = """Alert (step 0):
{alert}

Model reasons (SHAP; clues, not evidence):
{shap}

First-round results:
{calls}

Signals judged from the first round:
{signals}

Candidate patterns (from lit signals, weighted by how specific each signal is to the pattern):
{typologies}"""

NUDGE = "Continue: call a tool, or call submit_triage to finish."
LENGTH_NUDGE = ("Your previous reply ran over the length limit before reaching a decision. Do not re-examine "
                "everything again: decide now with the evidence already collected and call submit_triage. If the "
                "evidence is not enough to rule out a strong pattern, escalate and say what is unresolved in "
                "not_checked.")


def case_file(ctx: TriageContext, seed: dict) -> str:
    j = lambda x: json.dumps(x, ensure_ascii=False)
    shap = [{k: it[k] for k in ("feature", "value", "shap")} for it in ctx.alert["shap"].get("pushes_up", [])]
    calls = "\n".join(j(c["result"]) for c in seed["calls"])
    sig = [{k: e[k] for k in ("signal", "party", "status", "step", "path", "value")} for e in seed["signals"]]
    typ = [{k: t[k] for k in ("typology", "matched", "missing", "score", "coverage")} for t in seed["typologies"]]
    return USER.format(alert=j(ctx.results[0]), shap=j(shap), calls=calls, signals=j(sig),
                       typologies=j(typ) if typ else "(no lit signal matches a pattern)")


# ---------------------------------------------------------------------------
# 核验
# ---------------------------------------------------------------------------
def _num(x) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip().replace(",", "").removeprefix("$").removesuffix("%")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def same_value(claimed, actual) -> bool:
    """数字按两位小数比较（金额已经统一到分），布尔和字符串逐字比较，列表逐项比较。"""
    if isinstance(actual, bool) or isinstance(claimed, bool):
        return actual == claimed or str(claimed).lower() == str(actual).lower()
    if isinstance(actual, (int, float)):
        c = _num(claimed)
        return c is not None and round(c, 2) == round(float(actual), 2)
    if isinstance(actual, list):
        return isinstance(claimed, list) and len(claimed) == len(actual) and \
            all(same_value(c, a) for c, a in zip(claimed, actual))
    if actual is None:
        return claimed is None or str(claimed).lower() in ("null", "none", "")
    return str(claimed).strip() == str(actual).strip()


def check_facts(ctx: TriageContext, facts: list[FactArg]) -> list[dict]:
    out = []
    for i, f in enumerate(facts, 1):
        res = ctx.results.get(f.step)
        if res is None:
            out.append({"fact": i, "verified": False, "problem": "bad_step",
                        "detail": f"step {f.step} is not a stored tool result"})
            continue
        try:
            actual = resolve_path(res, f.path)
        except KeyError:
            out.append({"fact": i, "verified": False, "problem": "bad_path",
                        "detail": f"step {f.step} ({res['tool']}) has no field '{f.path}'"})
            continue
        ok = same_value(f.value, actual)
        out.append({"fact": i, "verified": ok, "step": f.step, "path": f.path, "value": f.value,
                    **({} if ok else {"problem": "value_mismatch", "detail": f"stored value is {json.dumps(actual)}"})})
    return out


_STRIP = [re.compile(r"\d{4}-\d{2}-\d{2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?"),   # 日期、日期时刻
          re.compile(r"\b\d{1,2}:\d{2}\b"),                                    # 时刻
          re.compile(r"\b\d+:[0-9A-Za-z]+\b"),                                 # 账户号 bank:account
          re.compile(r"(?i)\b(?:step|steps)\s*\d+(?:\s*(?:,|and|-|–)\s*\d+)*"),  # "step 3"、"steps 5 and 7"
          re.compile(r"#\d+"), re.compile(r"(?i)\btxn[_ ]?\w*\s*\d+")]
_NUM = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def _strip(text: str) -> str:
    for rx in _STRIP:
        text = rx.sub(" ", text)
    return text


def number_pool(ctx: TriageContext) -> list[float]:
    """这次调查登记过的所有数值（含字符串字段里写着的数字，如金额区间说明里的 2,154–24,075）。"""
    pool: list[float] = []

    def walk(x):
        if isinstance(x, bool) or x is None:
            return
        if isinstance(x, (int, float)):
            pool.append(float(x))
        elif isinstance(x, str):
            pool.extend(float(m.replace(",", "")) for m in _NUM.findall(_strip(x)))
        elif isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    for r in ctx.results.values():
        walk(r)
    return pool


def ungrounded_numbers(text: str, pool: list[float]) -> list[str]:
    """理由里找不到出处的数字。允许按写出的精度四舍五入；整千以上的整数允许 1% 以内的约数（"about 64,000"）。"""
    bad = []
    for tok in _NUM.findall(_strip(text)):
        n = float(tok.replace(",", ""))
        dec = len(tok.split(".")[1]) if "." in tok else 0
        tol = 0.5 * 10 ** (-dec) + 1e-9
        ok = any(abs(n - v) <= tol for v in pool)
        if not ok and dec == 0 and n >= 1000 and tok.replace(",", "").endswith("000"):
            ok = any(v and abs(n - v) / abs(v) <= 0.01 for v in pool)
        if not ok:
            bad.append(tok)
    return bad


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def triage(alert_id: int, use_cache: bool = True, log: bool = True, max_calls: int = MAX_CALLS) -> dict:
    t0 = time.perf_counter()
    ctx = TriageContext.load(alert_id, log=log, max_calls=max_calls).start()
    a = ctx.alert
    ctx.results[0] = {"step": 0, "tool": "alert", "as_of": ctx.as_of.strftime("%Y-%m-%d %H:%M"), "result": {
        "alert_id": a["alert_id"], "txn_id": a["txn_id"], "time": ctx.as_of.strftime("%Y-%m-%d %H:%M"),
        "payer": f"{a['from_bank']}:{a['from_account']}", "payee": f"{a['to_bank']}:{a['to_account']}",
        "amount_usd": round(float(a["amount_usd"]), 2), "tier": a["tier"], "score_percentile": round(a["score_pct"], 3)}}
    rec = {"alert_id": alert_id, "prompt_version": AGENT_VERSION, "model": llm.MODEL, "run_id": ctx.run_id,
           "usage": new_usage(), "trace": [], "ms": {}}
    try:
        seed = build_seed(ctx)
        rec["seed"] = {"focus": seed["focus"], "calls": len(seed["calls"]),
                       "lit": sorted({e["signal"] for e in seed["signals"] if e["status"] == "lit"}),
                       "top_typology": seed["typologies"][0]["typology"] if seed["typologies"] else None}
        rec["ms"]["seed"] = round((time.perf_counter() - t0) * 1000)
        messages = [{"role": "system", "content": SYSTEM.format(max_calls=max_calls)},
                    {"role": "user", "content": case_file(ctx, seed)}]
        tools = tool_schemas() + [SUBMIT_TOOL]

        def run_tool(name: str, raw: str) -> dict:
            result = call_tool(ctx, name, raw)
            rec["trace"].append({"step": ctx.step, "tool": name, "args": raw, "error": result.get("error")})
            return result

        try:
            out = run_loop(messages, tools, run_tool=run_tool, submit_name="submit_triage",
                           on_submit=lambda raw, n: _submit(ctx, rec, raw, n), usage=rec["usage"],
                           use_cache=use_cache, max_turns=MAX_TURNS, nudge=NUDGE, purpose="triage",
                           length_nudge=LENGTH_NUDGE)
        except llm.LLMError as e:
            if e.raw:
                rec["raw_failed"] = e.raw[:8000]
            rec.update(status="error", error=f"生成失败：{e}", turns=getattr(e, "turn", 0))
            return _finish(ctx, rec, t0)
        rec["turns"] = out.turns
        if out.length_nudged:
            rec["length_nudged"] = True
        if out.final is None:
            rec.update(status="error", error=f"在 {out.turns} 轮内没有提交（提醒 {out.nudges} 次）")
            return _finish(ctx, rec, t0)
        data, report = out.final
        rec.update(status="done", decision=data.decision, typology=data.typology, confidence=data.confidence,
                   rationale=data.rationale, not_checked=data.not_checked, facts=report["facts"],
                   ungrounded_numbers=report["ungrounded_numbers"],
                   verified=not report["ungrounded_numbers"] and all(f["verified"] for f in report["facts"]))
        return _finish(ctx, rec, t0)
    finally:
        ctx.close()


def _submit(ctx: TriageContext, rec: dict, raw: str, submits: int):
    ctx.step += 1
    try:
        data = SubmitTriage.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as e:
        msg = e.msg if isinstance(e, json.JSONDecodeError) else "; ".join(
            f"{'.'.join(map(str, d['loc']))}: {d['msg']}" for d in e.errors())
        rec["trace"].append({"step": ctx.step, "tool": "submit_triage", "error": "invalid_args"})
        if submits + 1 >= MAX_SUBMITS:
            raise llm.LLMError(f"提交格式两次不正确：{msg}")
        return {"status": "invalid", "error": f"submission does not follow the required format: {msg}"}, None
    facts = check_facts(ctx, data.facts)
    bad_nums = ungrounded_numbers(data.rationale, number_pool(ctx))
    failed = [f for f in facts if not f["verified"]]
    if submits == 0:
        rec["first_pass"] = {"facts": len(facts), "verified": len(facts) - len(failed), "ungrounded": len(bad_nums)}
    report = {"status": "accepted" if not (failed or bad_nums) else "rejected",
              "facts_verified": len(facts) - len(failed), "failed_facts": failed,
              "ungrounded_numbers_in_rationale": bad_nums}
    done = not (failed or bad_nums) or submits + 1 >= MAX_SUBMITS
    if not done:
        report["instruction"] = ("bad_step / bad_path: cite a step and a field that exist in the stored results; "
                                 "value_mismatch: copy the stored value exactly; ungrounded numbers: every number in "
                                 "the rationale must come from the tool results of this investigation — correct it "
                                 "or remove it. Then call submit_triage again.")
    elif failed or bad_nums:
        report["status"] = "recorded_with_unverified"          # 第二次仍不通过：照实记录，不伪装成通过
    rec["trace"].append({"step": ctx.step, "tool": "submit_triage", "status": report["status"],
                         "failed": len(failed), "ungrounded": len(bad_nums)})
    if submits > 0:
        rec["retried"] = True
    return report, ((data, {"facts": facts, "ungrounded_numbers": bad_nums}) if done else None)


def _finish(ctx: TriageContext, rec: dict, t0: float) -> dict:
    rec["tool_calls"] = sum(1 for t in rec["trace"] if t["tool"] != "submit_triage")
    rec["ms"]["total"] = round((time.perf_counter() - t0) * 1000)
    rec["cost_usd"] = round(llm.cost_usd(rec["usage"]), 5)
    ctx.finish("done" if rec["status"] == "done" else "error",
               {k: rec.get(k) for k in ("decision", "typology", "confidence", "verified", "turns", "tool_calls",
                                        "cost_usd")})
    return rec


# ---------------------------------------------------------------------------
# 离线自检：脚本化的假模型（不调用 API），在已排除的告警 #1 上测核验
# ---------------------------------------------------------------------------
def selftest(alert_id: int = 1) -> None:
    script = []

    def fake(messages, tools, purpose, max_tokens=8000, use_cache=True):
        n = sum(m["role"] == "assistant" for m in messages)
        u = {"prompt": 0, "completion": 0, "hit": 0, "reasoning": 0, "cached": True}
        return {"content": "", "reasoning_content": "", "tool_calls": script[n] if n < len(script) else []}, u

    llm.chat_tools = fake
    # 先跑一遍首轮，拿到真实的步骤号和数值，再据此写剧本
    probe = TriageContext.load(alert_id, log=False)
    probe.results[0] = {"step": 0}
    seed = build_seed(probe)
    probe.close()
    fan = next(c for c in seed["calls"] if c["tool"] == "fan_profile")
    pt = next(c for c in seed["calls"] if c["tool"] == "pass_through")
    out_cp = fan["result"]["result"]["out_cp"]
    n_matches = pt["result"]["n_matches"]
    good = {"decision": "escalate", "typology": "gather_scatter", "confidence": "medium",
            "facts": [{"step": fan["step"], "path": "result.out_cp", "value": out_cp},
                      {"step": pt["step"], "path": "n_matches", "value": n_matches}],
            "rationale": f"The payer sent to {out_cp} counterparties in 72 hours and {n_matches} incoming payments "
                         f"were passed on (step {pt['step']}).",
            "not_checked": ""}
    bad = json.loads(json.dumps(good))
    bad["facts"][0]["value"] = out_cp + 1                       # 记错的数字
    bad["facts"].append({"step": 99, "path": "n_matches", "value": 0})   # 不存在的步骤
    bad["rationale"] += " It moved 987,654 USD."                # 编出来的数字
    script[:] = [
        [{"id": "c1", "name": "fan_profile", "arguments": json.dumps({"account": "payee", "hours": 24})}],
        [{"id": "c2", "name": "submit_triage", "arguments": json.dumps(bad)}],
        [{"id": "c3", "name": "submit_triage", "arguments": json.dumps(good)}],
    ]
    rec = triage(alert_id, log=False)
    checks = []

    def check(name, ok, detail=""):
        checks.append(ok)
        print(f"  {'✓' if ok else '✗'} {name}" + (f"   {detail}" if detail else ""))
    first = rec.get("first_pass", {})
    check("第一次提交：记错的数字、不存在的步骤被识别", first.get("verified") == 1 and first.get("facts") == 3,
          f"{first}")
    check("第一次提交：理由里编出来的数字被识别", first.get("ungrounded") == 1)
    check("改正后第二次提交通过", rec.get("status") == "done" and rec.get("verified") is True and rec.get("retried"),
          f"status={rec.get('status')} verified={rec.get('verified')}")
    check("模型补查的工具结果也登记了（第 12 步左右）", any(t["tool"] == "fan_profile" for t in rec["trace"]))
    check("理由里的 'step N' 不被当成数字", not rec.get("ungrounded_numbers"))
    print(f"  结论 {rec.get('decision')} / {rec.get('typology')}；轮数 {rec.get('turns')}；首轮 {rec['seed']['calls']} 次查询")

    # 第二个剧本：第一次调用模型时输出被截断 → 追加提醒 → 第二次直接提交（D75）
    calls = {"n": 0}

    def fake_len(messages, tools, purpose, max_tokens=8000, use_cache=True):
        calls["n"] += 1
        if calls["n"] == 1:
            e = llm.LLMError("输出被截断（finish_reason=length）")
            e.usage = {"prompt": 10, "completion": 16000, "hit": 0, "reasoning": 16000, "cached": False}
            raise e
        last = messages[-1]
        ok = last["role"] == "user" and "length limit" in last["content"]
        return {"content": "", "reasoning_content": "", "tool_calls": [
            {"id": "z1", "name": "submit_triage", "arguments": json.dumps(good)}] if ok else []}, \
            {"prompt": 0, "completion": 0, "hit": 0, "reasoning": 0, "cached": True}
    llm.chat_tools = fake_len
    rec2 = triage(alert_id, log=False)
    check("输出截断后提醒一次、随后正常提交", rec2.get("status") == "done" and rec2.get("length_nudged") is True,
          f"status={rec2.get('status')} turns={rec2.get('turns')}")
    check("截断那次调用的用量照常记账（失败同样计费）", rec2["usage"]["completion"] == 16000)
    if not all(checks):
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# 命令行：只跑 dev 告警（或已排除的告警）
# ---------------------------------------------------------------------------
def _dev() -> tuple[list[int], dict[int, str], set[int]]:
    sp = json.loads(SPLIT.read_text(encoding="utf-8"))
    dev = sorted(x["alert_id"] for x in sp["dev"])
    return dev, {x["alert_id"]: x["stratum"] for x in sp["dev"]}, set(dev) | set(sp["excluded_alerts"])


def show(rec: dict, label: str | None = None) -> None:
    print(f"告警 #{rec['alert_id']}  {rec['status']}  首轮焦点 {rec['seed']['focus']}，查实 {rec['seed']['lit']}")
    for t in rec["trace"]:
        if t["tool"] == "submit_triage":
            print(f"    #{t['step']} submit_triage → {t.get('status') or t.get('error')}"
                  + (f"（事实 {t['failed']} 条未通过，理由里 {t['ungrounded']} 个数字无出处）"
                     if t.get("failed") or t.get("ungrounded") else ""))
        else:
            print(f"    #{t['step']} {t['tool']} {t['args'][:120]}" + (f"  → {t['error']}" if t["error"] else ""))
    if rec["status"] == "done":
        print(f"  结论：{rec['decision']}  手法 {rec['typology']}  把握 {rec['confidence']}"
              f"  核验{'全部通过' if rec['verified'] else '有未通过项'}"
              + (f"   [标签：{label}]" if label else ""))
        print(f"  理由：{rec['rationale']}")
        for f in rec["facts"]:
            print(f"    {'✓' if f['verified'] else '✗'} step {f.get('step')} {f.get('path')} = {f.get('value')}"
                  + (f"  ({f['problem']}: {f['detail']})" if not f["verified"] else ""))
        if rec["ungrounded_numbers"]:
            print(f"  理由里无出处的数字：{rec['ungrounded_numbers']}")
        if rec["not_checked"]:
            print(f"  没能查：{rec['not_checked']}")
    else:
        print(f"  {rec.get('error')}")
    print(f"  轮数 {rec.get('turns')}，模型补查 {rec['tool_calls']} 次，费用 ${rec['cost_usd']:.4f}，"
          f"用时 {rec['ms']['total'] / 1000:.1f}s")


TEST_RUNS = PROJECT_ROOT / "reports" / "r9c_test_runs.json"
TEST_PARTIAL = PROJECT_ROOT / "reports" / "r9c_test_runs.partial.json"
DEV_EVAL = PROJECT_ROOT / "reports" / "r9c_eval_dev.json"


def run_test() -> None:
    """R9c：在冻结的 test 上整批运行一次（D75、D76）。

    - 前提：dev 已经评测并选定方案（r9c_eval_dev.json），而且 dev 评测用的就是当前这个版本的 Agent。
    - 只跑一次：最终文件已存在就拒绝。
    - 中途断了（网络、断电）可以再执行同一条命令接着跑：每条告警完成后立即写进 partial 文件，重跑时跳过已完成的，
      包括出错的——出错是这次运行的结果，按"升级"计（D75），不重新抽一次。
    - 运行时不读取、不打印 test 的标签；标签只在 r9c_eval.py 里汇总时使用。
    """
    if TEST_RUNS.exists():
        raise SystemExit(f"{TEST_RUNS.relative_to(PROJECT_ROOT)} 已存在：test 只跑一次")
    if not DEV_EVAL.exists():
        raise SystemExit("还没有 dev 的评测与方案选择（reports/r9c_eval_dev.json）：先运行 r9c_eval.py --split dev")
    dev_eval = json.loads(DEV_EVAL.read_text(encoding="utf-8"))
    if dev_eval.get("selected") not in ("A", "B"):
        raise SystemExit("r9c_eval_dev.json 里没有方案选择")
    if set(dev_eval.get("versions", {})) != {AGENT_VERSION}:
        raise SystemExit(f"dev 评测的版本 {dev_eval.get('versions')} 与当前 Agent 版本 {AGENT_VERSION} 不一致："
                         "test 必须运行 dev 上评测过的同一个版本")
    sp = json.loads(SPLIT.read_text(encoding="utf-8"))
    ids = sorted(x["alert_id"] for x in sp["test"])
    done = json.loads(TEST_PARTIAL.read_text(encoding="utf-8")) if TEST_PARTIAL.exists() else {}
    if done:
        print(f"接着上次中断的运行：已完成 {len(done)} 条，跳过")
    print(f"test {len(ids)} 条，Agent {AGENT_VERSION}，dev 选定方案 {dev_eval['selected']}")
    for n, i in enumerate(ids, 1):
        if str(i) in done:
            continue
        rec = triage(i)
        done[str(i)] = rec
        TEST_PARTIAL.write_text(json.dumps(done, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        print(f"  [{n:>2}/{len(ids)}] #{i:<5} {rec['status']:<6} {rec.get('decision') or rec.get('error', '')[:60]:<10}"
              f"  补查 {rec['tool_calls']}  ${rec['cost_usd']:.4f}  {rec['ms']['total'] / 1000:.0f}s")
    TEST_PARTIAL.replace(TEST_RUNS)
    recs = list(done.values())
    print(f"\n完成：升级 {sum(r.get('decision') == 'escalate' for r in recs)}、关闭 "
          f"{sum(r.get('decision') == 'close' for r in recs)}、出错 {sum(r['status'] != 'done' for r in recs)}；"
          f"费用 ${sum(r['cost_usd'] for r in recs):.3f}；记录写到 {TEST_RUNS.relative_to(PROJECT_ROOT)}")
    print("下一步：python scripts/r9c_eval.py --split test")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="告警分诊 Agent")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--selftest", action="store_true")
    g.add_argument("--alert", type=int)
    g.add_argument("--dev", type=int, metavar="N")
    g.add_argument("--test", action="store_true", help="R9c：在冻结的 test 上整批运行一次")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.test:
        if a.no_cache:
            raise SystemExit("--test 不接受 --no-cache：中断后接着跑要靠缓存与 partial 文件")
        run_test()
        return
    dev, strata, ok = _dev()
    if a.alert is not None:
        ids = [a.alert]
    else:
        # 四层轮流取（每层内按编号）：只按编号取前几条，会全部落在 T1 洗钱层（编号小 = 分数高）
        by = {s: [i for i in dev if strata[i] == s] for s in ("T2/L", "T2/N", "T1/L", "T1/N")}
        order = [x for grp in zip(*by.values()) for x in grp] + \
            [i for i in dev if i not in {x for grp in zip(*by.values()) for x in grp}]
        ids = order[:a.dev]
    if any(i not in ok for i in ids):
        raise SystemExit("只能跑 dev 告警或已排除的告警：test 冻结")
    with connect() as conn, conn.cursor() as cur:            # dev 的标签，只用于打印对照
        cur.execute("SELECT alert_id, is_laundering, pattern_type FROM alert_labels WHERE alert_id = ANY(%s)", (ids,))
        label = {i: ("洗钱" + (f"（{p}）" if p else "（无模式标签）") if l else "非洗钱") for i, l, p in cur.fetchall()}
    recs, cost = [], 0.0
    for i in ids:
        print("=" * 100)
        rec = triage(i, use_cache=not a.no_cache)
        show(rec, f"{strata.get(i, '已排除')} {label.get(i)}")
        recs.append(rec)
        cost += rec["cost_usd"]
    # 汇总（dev，只看样本本身，不加权；加权与基线比较在 R9c）
    print("=" * 100)
    print(f"  {'层':<6}{'条数':>5}{'升级':>6}{'关闭':>6}{'出错':>6}   首次提交即通过核验")
    for st in ("T1/L", "T1/N", "T2/L", "T2/N", "已排除"):
        rs = [r for r in recs if strata.get(r["alert_id"], "已排除") == st]
        if not rs:
            continue
        done = [r for r in rs if r["status"] == "done"]
        fp = [r for r in done if r.get("first_pass") and r["first_pass"]["verified"] == r["first_pass"]["facts"]
              and not r["first_pass"]["ungrounded"]]
        print(f"  {st:<6}{len(rs):>5}{sum(r.get('decision') == 'escalate' for r in rs):>6}"
              f"{sum(r.get('decision') == 'close' for r in rs):>6}{len(rs) - len(done):>6}   {len(fp)}/{len(done)}")
    out = PROJECT_ROOT / "reports" / "r9b_dev_runs.json"
    old = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    old.update({str(r["alert_id"]): r for r in recs})
    out.write_text(json.dumps(old, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("=" * 100)
    print(f"{len(recs)} 条，合计 ${cost:.4f}；记录写到 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()