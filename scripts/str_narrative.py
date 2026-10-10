"""
SentryAML - R10b-2：STR 的"理由"（Reason(s) for Suspicion）由模型起草、程序核验
==============================================================================
复核员确认可疑、草稿的结构化部分已经组装好（R10b-1）之后，起草表格里唯一的自由文本：为什么可疑。

为什么是一次调用而不是 Agent（D80）：证据在分诊时已经查完、核验完，复核员也已经确认了可疑；这一步只是把证据写成
报告文字。如果让模型在这里再调用工具，就会在人做出决定之后，往 STR 里带进没有人复核过的新材料。

给模型的材料：草稿的结构化部分（账户、交易）、核验过的事实所在步骤的工具结果（与分诊时模型看到的完全一致）、
复核员确认可疑时写的理由、选定的手法。**不给** Agent 的分诊理由：它是为"升级还是关闭"写的，充满对弱证据的权衡，
和 STR 要讲的"为什么可疑"不是一回事；给了，模型容易照着它的结构和措辞改写（D80，用户确认）。

程序检查（把 STRO 填表指南的要求变成规则）：
  1. facts：每条 {step, path, value} 必须与登记的工具结果一致（复用分诊 Agent 的 check_facts，D74）
  2. 理由里的每个数字都能在工具结果或草稿的交易里找到（复用 ungrounded_numbers）
  3. 不出现报告机构的名称（指南：Do NOT mention the name of your company）
  4. 不照抄已列入的可疑类别原文（指南：已勾选的可疑类别不要在理由里重复）——任意连续 8 个词相同即算照抄
  5. 长度 200–10,000 字符（表格上限 10,000；太短说明没写清楚）
不通过 → 把问题清单交回模型改正一次；仍不通过 → **不写入草稿**（D80：草稿版本是"有人采纳过的内容"的记录，
一段有无出处数字的理由不应进入这份记录），把问题告诉分析员，他可以重试或用 str_draft.py revise 手写。

通过 → 作为草稿的新版本写入，作者是发起起草的分析员（他采纳了这段文字），generator 记录模型和提示词版本；
之后仍然要另一位 MLRO 批准（R10a/R10b-1 的规则不变）。

用法：
    python scripts/str_narrative.py write 26 --as alice
    python scripts/str_narrative.py selftest          # 假模型、不调用 API；在一个事务里完成，结束时回滚
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import psycopg

import llm
from db import connect, connect_review
from review import act
from str_draft import SGT, explain_str, fixture_case, insert_draft, latest, load_case, revise_body, tool_results
from triage_agent import FactArg, check_facts, number_pool, ungrounded_numbers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = PROJECT_ROOT / "reports" / "r10_narrative_runs.json"
PROMPT_VERSION = "str-narrative-v3.1"
# v3.1（D83）：提示词不变；生成失败（输出截断、JSON 解析失败）时不再直接放弃，而是提醒一次再试——
#   dev 上 #2063 第一次调用就生成失败，v3 没有给第二次机会。成功的调用请求不变，缓存照常命中
GEN_RETRY = ("Your previous reply could not be used ({error}). Reply now with the JSON object only: a narrative "
             "of at most 350 words and at most 12 facts.")
# v2（D80，dev 案件 26 的第一份真实理由之后）：
#   1. 材料里去掉模型内部的统计量：告警的分数分位和档位（第 0 步），以及金额画像（amount_profile）——
#      "训练期洗钱金额区间"是我们模型训练数据的统计，不是可以写进 STR 的"洗钱常见特征"（v1 的理由就这样写了）
#   2. 新增规则 8：只陈述证据显示的事实，不推出证据之外的结论（v1 写了"账户里没有留下任何资金""这组关系是专门
#      为这笔转账建立的"，两句都没有证据）
# v3（D82，dev 成批评测之后）：
#   1. 证据改为逐行 "路径 = 值" 的形式交给模型（之前是整段 JSON）。dev 上 14 份理由有 11 份第一次被退回，
#      全部是事实路径写错（例如漏了 "result."）；逐行列出后，路径可以原样照抄
#   2. 材料里去掉主体名下的其他账户（other_accounts）：它只列前 10 个，模型据此写出"其中 10 个在他行"
INTERNAL_TOOLS = {"amount_profile"}
HIDDEN_KEYS = {"other_accounts"}
INTERNAL_ALERT_FIELDS = {"score_percentile", "tier"}
MAX_ATTEMPTS = 2
COPY_WORDS = 8

SYSTEM = """You write the "Reason(s) for Suspicion" section of a Suspicious Transaction Report (STR) to be filed \
with Singapore's Suspicious Transaction Reporting Office (STRO). A human analyst has already investigated the \
alert and confirmed that it is suspicious; your text explains why, using only the evidence provided.

Rules:
1. Use only the facts in the EVIDENCE section. Do not add facts, guesses about who the people are, or the \
predicate crime. If something was not checked, do not mention it.
2. Every number you write must appear in the evidence (amounts, counts, hours, percentages). Write dates as \
YYYY-MM-DD and times as HH:MM. Refer to accounts by their identifiers exactly as given (bank:account) and to \
transactions by their transaction id.
3. Explain the flow of funds in order (where the money came from, where it went, how quickly, in what amounts), \
then why this pattern is suspicious and how it relates to the selected typology.
4. Do not name the reporting institution and do not refer to yourself.
5. The Categories of Suspicion listed below are already selected in the form. Do not copy or list their wording; \
explain the facts instead.
6. The transaction monitoring score is not evidence; do not cite it.
7. Write in plain English, 120 to 350 words, in 2 to 5 short paragraphs.
8. State only what the evidence shows, then why it is unusual. Do not extend an observation beyond what was measured: a pass-through check covers the matched pair in its window, not the whole account balance; "no prior transactions" means none before the alert time, not that a relationship was created for this payment. Avoid words such as "purpose-built", "clearly", "no funds were retained" unless a fact states exactly that.

Return JSON only:
{"narrative": "<the text>", "facts": [{"step": <int>, "path": "<field path>", "value": <value copied exactly>}]}
"facts" lists 2 to 12 facts the narrative relies on. Each EVIDENCE line has the form \
"step N | path = value": copy N, the path and the value exactly as shown, e.g. {"step": 7, "path": \
"matches[0].lag_hours", "value": 5.5} for the line "step 7 | matches[0].lag_hours = 5.5"."""


# ---------------------------------------------------------------------------
# 材料
# ---------------------------------------------------------------------------
def alert_result(case: dict, cur) -> dict:
    """第 0 步：告警本身。字段与分诊时完全相同（triage_agent.triage），Agent 的事实里 step 0 的路径才对得上。"""
    cur.execute("""SELECT alert_id, txn_id, txn_ts, from_bank, from_account, to_bank, to_account, amount_usd, tier,
                          score_pct FROM alerts WHERE alert_id = %s""", (case["alert_id"],))
    aid, txn, ts, fb, fa, tb, ta, amt, tier, pct = cur.fetchone()
    return {"step": 0, "tool": "alert", "as_of": ts.strftime("%Y-%m-%d %H:%M"), "result": {
        "alert_id": aid, "txn_id": txn, "time": ts.strftime("%Y-%m-%d %H:%M"), "payer": f"{fb}:{fa}",
        "payee": f"{tb}:{ta}", "amount_usd": round(float(amt), 2), "tier": tier, "score_percentile": round(pct, 3)}}


def materials(cur, case: dict, draft: dict) -> tuple[dict[int, dict], str]:
    """返回 (可引用的结果 {step: result}, 给模型的材料文字)。只给核验过的事实所在的那些步骤。"""
    calls = tool_results(cur, case["agent_run_id"])
    steps = sorted({f["step"] for f in draft["body"].get("evidence_facts", []) if f.get("step")})
    # 内部统计量不交给模型（v2）：去掉以后，模型想引用它们也找不到出处，事实核验和数字核验会直接拦下
    a0 = alert_result(case, cur)
    a0["result"] = {k: v for k, v in a0["result"].items() if k not in INTERNAL_ALERT_FIELDS}
    results = {0: a0} | {s: calls[s][1] for s in steps if s in calls and calls[s][0] not in INTERNAL_TOOLS}
    b = draft["body"]
    r = b["reasons"]
    parts = ["CASE",
             f"Alerted transaction: {json.dumps(results[0]['result'], ensure_ascii=False)}",
             f"Typology selected by the analyst: {r['typology']}",
             f"Analyst's note when confirming suspicion: {r.get('reviewer_reason') or '(none)'}",
             "", "REPORTED ACCOUNTS (already in the structured section of the form)"]
    for x in b["accounts"]:
        parts.append(f"- {x['account']} ({'held with the reporting institution' if x['held_with_reporting_institution'] else 'held elsewhere'})")
    parts += ["", "SUSPICIOUS TRANSACTIONS (already in the structured section of the form)"]
    for t in b["suspicious_transactions"]:
        parts.append(f"- txn {t['txn_id']}: {t['time']} {t['from']} -> {t['to']} {t['amount_usd']:.2f} USD")
    parts += ["", "CATEGORIES OF SUSPICION ALREADY SELECTED (do not copy this wording)"]
    parts += [f"- {c['quote']}" for c in r["categories"]]
    parts += ["", "EVIDENCE (tool results from the investigation, one field per line; cite them in \"facts\")"]
    for s, res in results.items():
        parts.append(f"-- step {s}: {res.get('tool', '')}" + (f" on {res['account']}" if res.get("account") else ""))
        parts += [f"step {s} | {path} = {json.dumps(v, ensure_ascii=False, default=str)}" for path, v in flatten(res)]
    return results, "\n".join(parts)


def flatten(obj, prefix: str = "") -> list[tuple[str, object]]:
    """把一个工具结果展开成 (路径, 值)，路径的写法与核验用的 resolve_path 完全一致：
    字典用 "."，列表用 "[i]"。step、tool 是外壳字段，不展开；主体名下的其他账户不交给模型（v3）。"""
    out: list[tuple[str, object]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not prefix and k in ("step", "tool"):
                continue
            if k in HIDDEN_KEYS:
                continue
            out += flatten(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list) and obj and all(isinstance(x, (dict, list)) for x in obj):
        for i, v in enumerate(obj):
            out += flatten(v, f"{prefix}[{i}]")
    else:
        out.append((prefix, obj))
    return out


def number_sources(text: str, results: dict[int, dict], limit: int = 3, cited=()) -> list[dict]:
    """理由里每个数字出自哪里：给人工复核和评分用（R10d 的评分表、以后 R11 的逐句对照界面）。
    匹配规则与 ungrounded_numbers 相同：按写出的精度四舍五入相等。

    同一个数可能出现在很多字段里（"5" 到处都是）。理由自己引用的事实（cited：{(step, path)}）排在最前面，
    其余的最多列 limit 个，再注明还有几处——v3 的评分表只列前 3 个，#3444 的"5 个账户"真正的出处
    （第 13 步 n_accounts）被挤掉了，列出来的反而是不相干的日交易笔数（D83）。"""
    from triage_agent import _NUM, _strip
    cited = {(int(s), p) for s, p in cited}
    leaves = [(s, p, v) for s, res in results.items() for p, v in flatten(res)
              if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out = []
    for tok in _NUM.findall(_strip(text)):
        n = float(tok.replace(",", ""))
        dec = len(tok.split(".")[1]) if "." in tok else 0
        tol = 0.5 * 10 ** (-dec) + 1e-9
        hits = [(s, p) for s, p, v in leaves if abs(n - float(v)) <= tol]
        first = [h for h in hits if h in cited]
        rest = [h for h in hits if h not in cited]
        src = [f"step {s} {p} (cited)" for s, p in first] + [f"step {s} {p}" for s, p in rest[:limit]]
        if len(rest) > limit:
            src.append(f"+{len(rest) - limit} more")
        out.append({"number": tok, "sources": src or ["(structured draft or not found)"]})
    return out


# ---------------------------------------------------------------------------
# 核验
# ---------------------------------------------------------------------------
def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def copied_categories(narrative: str, categories: list[dict], n: int = COPY_WORDS) -> list[str]:
    """理由里照抄了哪些可疑类别原文：任意连续 n 个词与原文相同。"""
    w = _words(narrative)
    grams = {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
    out = []
    for c in categories:
        q = _words(c["quote"])
        k = min(n, len(q))                     # 不足 n 个词的短原文：整句出现才算
        if k < 5:
            continue
        gk = grams if k == n else {tuple(w[i:i + k]) for i in range(len(w) - k + 1)}
        if any(tuple(q[i:i + k]) in gk for i in range(len(q) - k + 1)):
            out.append(c["indicator_id"])
    return out


def verify(out: dict, results: dict[int, dict], draft: dict) -> dict:
    """返回 {"ok": bool, "problems": [...], "facts": [...]}。规则见文件开头。"""
    problems = []
    narrative = (out.get("narrative") or "").strip()
    raw_facts = out.get("facts") or []
    facts: list[dict] = []
    try:
        parsed = [FactArg.model_validate(f) for f in raw_facts]
        facts = check_facts(SimpleNamespace(results=results), parsed)
    except Exception as e:                                      # 格式不对：当作一个问题交回模型
        problems.append({"problem": "bad_facts_format", "detail": str(e)[:300]})
    if not 2 <= len(raw_facts) <= 12:
        problems.append({"problem": "fact_count", "detail": f"{len(raw_facts)} facts; 2 to 12 are required"})
    problems += [{"problem": f["problem"], "detail": f["detail"]} for f in facts if not f["verified"]]

    body = draft["body"]
    pool = number_pool(SimpleNamespace(results=results))
    for t in body["suspicious_transactions"]:
        pool += [float(t["amount_usd"]), float(t["txn_id"])]
    bad = ungrounded_numbers(narrative, pool)
    if bad:
        problems.append({"problem": "ungrounded_numbers", "detail": f"numbers not found in the evidence: {bad}"})
    inst = body["reporting_institution"]["name"]
    names = {inst, inst.replace(" (synthetic)", "")}
    if any(nm.lower() in narrative.lower() for nm in names):
        problems.append({"problem": "institution_named",
                         "detail": "the reporting institution must not be named in this section"})
    copied = copied_categories(narrative, body["reasons"]["categories"])
    if copied:
        problems.append({"problem": "category_copied",
                         "detail": f"wording copied from the selected Categories of Suspicion: {copied}"})
    if not 200 <= len(narrative) <= 10000:
        problems.append({"problem": "length", "detail": f"{len(narrative)} characters; 200 to 10,000 are required"})
    return {"ok": not problems, "problems": problems, "facts": facts}


# ---------------------------------------------------------------------------
# 起草
# ---------------------------------------------------------------------------
def draft_narrative(results: dict, material: str, draft: dict, use_cache: bool = True) -> dict:
    """调用模型，核验，不通过时改正一次。返回运行记录（不写库）。"""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": material}]
    rec = {"prompt_version": PROMPT_VERSION, "model": llm.MODEL, "attempts": [], "usage": {}}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            out, usage = llm.chat_json(messages, purpose="str_narrative", max_tokens=6000, use_cache=use_cache)
        except llm.LLMError as e:
            usage = e.usage or {}
            out = None
            rec["attempts"].append({"attempt": attempt, "error": str(e)})
        for k in ("prompt", "hit", "completion", "reasoning"):
            rec["usage"][k] = rec["usage"].get(k, 0) + (usage.get(k) or 0)
        if out is None:
            # 生成失败：没有可以放回对话的完整回复，只追加一句提醒（v3.1）；与"核验不通过"共用两次机会
            messages = messages + [{"role": "user", "content": GEN_RETRY.format(error=rec["attempts"][-1]["error"])}]
            continue
        v = verify(out, results, draft)
        rec["attempts"].append({"attempt": attempt, "ok": v["ok"], "problems": v["problems"],
                                "chars": len((out.get("narrative") or "").strip())})
        if v["ok"]:
            rec.update(status="verified", narrative=out["narrative"].strip(), facts=v["facts"])
            break
        messages += [{"role": "assistant", "content": usage.get("raw") or json.dumps(out, ensure_ascii=False)},
                     {"role": "user", "content": "The text did not pass the checks:\n" +
                      "\n".join(f"- {p['problem']}: {p['detail']}" for p in v["problems"]) +
                      "\nFix every problem (correct or remove unsupported numbers, do not name the institution, "
                      "do not copy category wording) and return the full JSON again."}]
    rec.setdefault("status", "failed")
    rec["cost_usd"] = round(llm.cost_usd(rec["usage"]), 5)
    return rec


def write(case_id: int, actor: str, use_cache: bool = True, cur=None) -> dict:
    """起草并（通过时）写入新版本。cur 给定时在调用方的事务里执行（自检用），否则自己开复核员连接。"""
    t0 = time.perf_counter()
    own = cur is None
    conn = connect_review() if own else None
    cur = conn.cursor() if own else cur
    try:
        case = load_case(cur, case_id)
        if case["state"] != "str_drafted":
            raise SystemExit(f"案件 {case_id} 的状态是 {case['state']}：先用 str_draft.py build 组装草稿")
        d = latest(cur, case_id)
        results, material = materials(cur, case, d)
        rec = draft_narrative(results, material, d, use_cache=use_cache)
        rec.update(case_id=case_id, base_version=d["version"], actor=actor,
                   at=datetime.now(SGT).strftime("%Y-%m-%d %H:%M"))
        if rec["status"] == "verified":
            body = revise_body(d["body"], rec["narrative"])
            body["reasons"]["narrative_facts"] = rec["facts"]          # 理由所依据的事实，随草稿一起保存
            gen = {"kind": "model", "model": rec["model"], "prompt_version": PROMPT_VERSION,
                   "base_version": d["version"], "attempts": len(rec["attempts"])}
            # 在调用方的事务里执行时（自检、R10d 评测），用保存点隔开：写入被数据库拒绝时只撤销这一步，
            # 不让整个事务进入"已出错"状态、连累后面的案件
            if not own:
                cur.execute("SAVEPOINT narrative_insert")
            try:
                v, _ = insert_draft(cur, case, actor, body, d["csv"], gen)
                rec["new_version"] = v
                if own:
                    conn.commit()
            except psycopg.Error as e:
                if own:
                    conn.rollback()
                else:
                    cur.execute("ROLLBACK TO SAVEPOINT narrative_insert")
                rec.update(status="rejected_by_db", db_error=explain_str(e))
        rec["ms"] = round((time.perf_counter() - t0) * 1000)
        return rec
    finally:
        if own:
            conn.close()


def save_run(rec: dict) -> None:
    runs = json.loads(RUNS.read_text(encoding="utf-8")) if RUNS.exists() else []
    runs.append(rec)
    RUNS.parent.mkdir(exist_ok=True)
    RUNS.write_text(json.dumps(runs, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def cmd_write(a) -> None:
    rec = write(a.case_id, a.actor, use_cache=not a.no_cache)
    save_run(rec)
    for at in rec["attempts"]:
        if at.get("error"):
            print(f"  第 {at['attempt']} 次：生成失败 {at['error']}")
        else:
            print(f"  第 {at['attempt']} 次：{'通过' if at['ok'] else '未通过'}，{at['chars']} 字符"
                  + "".join(f"\n      ✗ {p['problem']}: {p['detail']}" for p in at["problems"]))
    if rec["status"] == "verified" and rec.get("new_version"):
        print(f"案件 {a.case_id}：理由已写入第 {rec['new_version']} 版（{a.actor}，模型 {rec['model']}，"
              f"{PROMPT_VERSION}）；费用 ${rec['cost_usd']:.4f}，{rec['ms'] / 1000:.0f}s")
        print(f"\n{rec['narrative']}\n")
        print("依据的事实（已核对）：" + "；".join(f"第 {f['step']} 步 {f['path']} = {f['value']}" for f in rec["facts"]))
        print(f"下一步：python scripts/str_draft.py show {a.case_id}，"
              f"然后由另一位 MLRO 批准：python scripts/review.py act {a.case_id} approve_str --as carol "
              f"--draft-version {rec['new_version']}")
    else:
        print(f"案件 {a.case_id}：没有写入草稿（{rec['status']}{'：' + rec['db_error'] if rec.get('db_error') else ''}）。"
              f"可以重试，或用 str_draft.py revise 手写理由；费用 ${rec['cost_usd']:.4f}")


# ---------------------------------------------------------------------------
# 自检：假模型
# ---------------------------------------------------------------------------
def cmd_selftest(a) -> None:
    """不调用 API：把 llm.chat_json 换成按脚本返回的假模型。在一个事务里搭测试案件（与 str_draft 自检相同），
    确认可疑、组装草稿后检验三种情形，结束时回滚。"""
    results: list[bool] = []

    def check(ok: bool, what: str, detail: str = "") -> None:
        results.append(ok)
        print(f"  {'✓' if ok else '✗'} {what}" + (f"   {detail}" if detail else ""))

    script: list[dict] = []
    seen_msgs: list[list] = []

    def fake_chat_json(messages, purpose, max_tokens=4000, use_cache=True):
        seen_msgs.append(messages)
        out = script.pop(0)
        if isinstance(out, Exception):
            out.usage = {"prompt": 1000, "hit": 0, "completion": 6000, "reasoning": 5900}
            raise out
        return out, {"prompt": 1000, "hit": 0, "completion": 300, "reasoning": 100, "cached": False,
                     "raw": json.dumps(out)}

    real = llm.chat_json
    llm.chat_json = fake_chat_json
    try:
        with connect() as conn, conn.cursor() as cur:
            fx = fixture_case(cur, "str_narrative selftest")
            cid, m = fx["cid"], fx["match"]
            cur.execute("SET LOCAL ROLE sentry_review")
            act(cur, cid, "confirm_suspicious", "alice", payload={"typology": "gather_scatter"})
            from str_draft import BUILDER, build_body
            case = load_case(cur, cid)
            body, csv_text = build_body(cur, case)
            insert_draft(cur, case, "alice", body, csv_text, {"kind": "builder", "builder": BUILDER})
            cat = body["reasons"]["categories"][0]["quote"]
            inst = body["reporting_institution"]["name"].replace(" (synthetic)", "")
            good = (f"Account {fx['payer']} received {fx['in_amt']:,.2f} USD from {fx['in_from']} "
                    f"(txn {fx['in_txn']}) and paid {fx['amt']:,.2f} USD to the payee in the alerted transaction "
                    f"(txn {fx['txn']}) {m['lag_hours']} hours later. The near-identical amount and the short delay "
                    "indicate that the account was used to pass funds through rather than for its own business, "
                    "which is consistent with the layering stage of a gather-scatter scheme.")
            good_facts = [{"step": 7, "path": "matches[0].in_amt", "value": fx["in_amt"]},
                          {"step": 7, "path": "matches[0].lag_hours", "value": m["lag_hours"]},
                          {"step": 0, "path": "result.amount_usd", "value": fx["amt"]}]

            print("[1] 第一次有四处问题，改正后通过 → 写入新版本")
            bad = (f"{inst} observed that {good} The amount was 987,654.32 USD in total. {cat}.")
            script[:] = [{"narrative": bad, "facts": good_facts[:1] + [{"step": 7, "path": "matches[0].in_amt",
                                                                       "value": 1.0}]},
                         {"narrative": good, "facts": good_facts}]
            rec = write(cid, "alice", cur=cur)
            p1 = {p["problem"] for p in rec["attempts"][0]["problems"]}
            check({"institution_named", "ungrounded_numbers", "category_copied", "value_mismatch"} <= p1,
                  "第一次：机构名、无出处的数字、照抄类别原文、事实值不符 都被识别", str(sorted(p1)))
            check("987,654.32" in rec["attempts"][0]["problems"][[p["problem"] for p in rec["attempts"][0][
                "problems"]].index("ungrounded_numbers")]["detail"], "编造的金额被点名")
            check(rec["status"] == "verified" and rec.get("new_version") == 2, "改正后通过，写入第 2 版",
                  f"status={rec['status']} version={rec.get('new_version')}")
            check(len(seen_msgs[-1]) == 4 and "did not pass" in seen_msgs[-1][-1]["content"],
                  "问题清单作为下一条消息交回模型")
            check("TRIAGE-RATIONALE-MARKER" not in json.dumps(seen_msgs[0], ensure_ascii=False),
                  "材料里没有 Agent 的分诊理由（案件快照里有，但没有交给模型）")
            check("score_percentile" not in seen_msgs[0][1]["content"] and "tier =" not in seen_msgs[0][1]["content"],
                  "材料里没有模型分数分位和档位（v2：内部统计量不是证据）")
            check(f"step 7 | matches[0].in_amt = {fx['in_amt']}" in seen_msgs[0][1]["content"],
                  "证据逐行列出路径和值，路径可以原样照抄（v3）")
            ns = number_sources(good, {7: {"matches": [m]}})
            check(any("matches[0].in_amt" in src for x in ns for src in x["sources"]),
                  "number_sources 能指出理由里的数字出自哪一步哪个字段")
            cur.execute("SELECT created_by, generator, body FROM str_drafts WHERE case_id = %s AND version = 2", (cid,))
            who, gen, b2 = cur.fetchone()
            check(who == "alice" and gen["kind"] == "model" and gen["prompt_version"] == PROMPT_VERSION,
                  "作者是发起起草的分析员，generator 记录模型与提示词版本", f"{who} {gen}")
            check(len(b2["reasons"]["narrative_facts"]) == 3 and all(f["verified"] for f in b2["reasons"][
                "narrative_facts"]), "理由依据的事实随草稿保存，且都已核对")
            cur.execute("SELECT action, payload FROM review_events WHERE case_id = %s ORDER BY seq DESC LIMIT 1",
                        (cid,))
            check(cur.fetchone() == ("edit_str", {"draft_version": 2}), "同一事务里记录了 edit_str 事件")

            print("[2] 两次都不通过 → 不写入草稿")
            script[:] = [{"narrative": bad, "facts": good_facts}, {"narrative": "Too short.", "facts": good_facts}]
            rec = write(cid, "alice", cur=cur)
            cur.execute("SELECT max(version) FROM str_drafts WHERE case_id = %s", (cid,))
            check(rec["status"] == "failed" and cur.fetchone()[0] == 2, "状态 failed，草稿仍是第 2 版",
                  str([sorted(p["problem"] for p in at["problems"]) for at in rec["attempts"]]))

            print("[2b] 生成失败（输出截断）→ 提醒一次后正常提交（v3.1）")
            script[:] = [llm.LLMError("输出被截断（finish_reason=length）"), {"narrative": good, "facts": good_facts}]
            rec = write(cid, "alice", cur=cur)
            check(rec["status"] == "verified" and rec.get("new_version") == 3 and "could not be used" in
                  seen_msgs[-1][-1]["content"], "截断后追加提醒、第二次通过，写入第 3 版",
                  f"status={rec['status']} attempts={[a.get('error') or a.get('ok') for a in rec['attempts']]}")
            check(rec["usage"]["completion"] >= 6000, "截断那次的用量照常记账")
            ns = number_sources(good, {7: {"matches": [m]}, 0: {"result": {"amount_usd": fx["amt"]}}},
                                cited={(0, "result.amount_usd")})
            check(any(src.endswith("(cited)") for x in ns for src in x["sources"]),
                  "number_sources 把理由自己引用的出处排在最前面并标注 (cited)")

            print("[3] 双人复核照常：起草人不能批准，另一位 MLRO 可以")
            from review import act as review_act
            cur.execute("SAVEPOINT s")
            try:
                review_act(cur, cid, "approve_str", "alice", payload={"draft_version": 3})
                check(False, "分析员 alice 批准 → 应被拒绝")
            except psycopg.Error as e:
                cur.execute("ROLLBACK TO SAVEPOINT s")
                check("review_role" in e.diag.message_primary, "分析员 alice 批准 → 拒绝", e.diag.message_primary[:80])
            seq, st = review_act(cur, cid, "approve_str", "carol", payload={"draft_version": 3})
            check(st == "str_approved", "Carol 批准第 3 版（模型起草、alice 采纳、carol 批准）")
            conn.rollback()
    finally:
        llm.chat_json = real
    print(f"\n{sum(results)}/{len(results)} 项通过；测试数据已全部回滚，没有调用 API")
    if not all(results):
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description="STR 理由：模型起草、程序核验（R10b-2）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("write")
    p.add_argument("case_id", type=int)
    p.add_argument("--as", dest="actor", required=True)
    p.add_argument("--no-cache", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"write": cmd_write, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()