"""
SentryAML - R10b-1：STR 草稿（不用模型的部分）
=============================================
复核员确认可疑之后，按 STRO 表格的结构组装草稿（D79）：

    报告机构        付款方账户所在的银行（付款行监控自己客户转出的钱）
    账户信息        告警两端 + 核验过的证据里出现的账户
    主体信息        每个账户的持有主体；合成数据没有姓名、证件、地址，填 Unknown
    可疑交易        告警这一笔 + 证据引用到的交易；所列账户的全部交易放进 CSV 附件
    理由            犯罪类型、手法（复核员确认时选定）、可疑类别（知识图谱原文，按审查日期核对）、
                    发现日期（确认可疑那条复核事件的日期）；自由文本的理由留给 R10b-2

这里没有一样东西由模型生成，也没有一条"草稿合不合格"的规则——规则全部在数据库的 str_draft_problems 里，
写入草稿时由触发器执行。这个脚本只负责把数据组装好、把被拒绝的原因翻译给人看。

用法：
    python scripts/str_draft.py build 26 --as alice          # 组装并写入新版本（首次为 draft_str，之后为 edit_str）
    python scripts/str_draft.py show 26 [--version 1]
    python scripts/str_draft.py check 26 [--version 1]       # 重新执行数据库检查 + 核对附件
    python scripts/str_draft.py revise 26 --as bob --narrative-file reason.txt   # 人工写入理由，生成新版本
    python scripts/str_draft.py selftest                     # 全部在一个事务里，结束时回滚
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg

from db import connect, connect_review
from review import act, explain

BUILDER = "str-builder-v1.2"   # v1.1：同一步的多条事实合并成一条来源说明；v1.2：不再把主体名下的其他账户当作证据账户（D82）
SGT = ZoneInfo("Asia/Singapore")
MAX_ROWS = 40                                  # 表格里账户、交易各最多 40 个（D79）
ACCT_RE = re.compile(r"^\d+:[0-9A-Za-z]+$")
TXN_KEYS = {"in_txn", "out_txn", "txn_id"}
_PATH_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)|\[(\d+)\]")
TYPE_OF_CRIME = "Money laundering (predicate offence not established)"
LIMIT_SYNTHETIC = ("Synthetic data (IBM AMLworld, HI-Small): the dataset has no names, identification numbers or "
                   "addresses, so these entity fields are marked Unknown.")


# ---------------------------------------------------------------------------
# 读取案件与证据
# ---------------------------------------------------------------------------
def load_case(cur, case_id: int) -> dict:
    cur.execute("""SELECT c.case_id, c.alert_id, c.route, c.agent_decision, c.agent_result, c.agent_run_id, s.state,
                          a.txn_id, a.txn_ts, a.from_bank, a.from_account, a.to_bank, a.to_account, a.tier
                   FROM review_cases c JOIN review_case_state s ON s.case_id = c.case_id
                   JOIN alerts a ON a.alert_id = c.alert_id WHERE c.case_id = %s""", (case_id,))
    row = cur.fetchone()
    if row is None:
        raise SystemExit(f"没有案件 {case_id}")
    keys = ("case_id", "alert_id", "route", "agent_decision", "agent_result", "agent_run_id", "state", "txn_id",
            "txn_ts", "from_bank", "from_account", "to_bank", "to_account", "tier")
    case = dict(zip(keys, row))
    case["payer"] = f"{case['from_bank']}:{case['from_account']}"
    case["payee"] = f"{case['to_bank']}:{case['to_account']}"
    cur.execute("""SELECT actor, created_at, payload, reason FROM review_events
                   WHERE case_id = %s AND action = 'confirm_suspicious' ORDER BY seq DESC LIMIT 1""", (case_id,))
    r = cur.fetchone()
    case["confirm"] = dict(zip(("actor", "created_at", "payload", "reason"), r)) if r else None
    return case


def tool_results(cur, run_id) -> dict[int, tuple[str, dict]]:
    """这次分诊每一步的工具结果，与模型当时看到的完全一致（agent_tool_calls，D61）。"""
    if run_id is None:
        return {}
    cur.execute("SELECT step, tool, result FROM agent_tool_calls WHERE run_id = %s", (run_id,))
    return {step: (tool.removeprefix("seed:"), result) for step, tool, result in cur.fetchall()}


def _container(result: dict, path: str):
    """事实指向的那一条记录：'matches[2].in_amt' → matches[2] 这一条过桥匹配；没有下标时就是整个结果。

    为什么取"那一条"而不是整个结果：Agent 引用的是第 2 条匹配，草稿里列出的就应该是这一条的两笔交易，
    而不是这次查询返回的全部匹配。"""
    cur, container = result, result
    for name, idx in _PATH_RE.findall(path.replace(".", " ")):
        try:
            cur = cur[name] if name else cur[int(idx)]
        except (KeyError, IndexError, TypeError):
            return container
        if idx and isinstance(cur, dict):
            container = cur
    return container


# 主体名下的其他账户（inv_entity_profile.other_accounts，最多列 10 个）说明的是"谁持有哪些账户"，不是资金流向；
# 而且只是按账号排序的前 10 个，挑哪几个列入报告完全是任意的。v1.1 把它们当作证据账户列入，dev 上一个案件
# 列了 12 个账户，模型还据此写出"其中 10 个在他行"这种错误推断（D82）
HOLDING_KEYS = {"other_accounts"}


def _scan(obj, txns: list[int], accts: list[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in HOLDING_KEYS:
                continue
            if k in TXN_KEYS and isinstance(v, int):
                txns.append(v)
            elif k == "txn_path" and isinstance(v, list):
                txns.extend(x for x in v if isinstance(x, int))
            else:
                _scan(v, txns, accts)
    elif isinstance(obj, list):
        for v in obj:
            _scan(v, txns, accts)
    elif isinstance(obj, str) and ACCT_RE.match(obj):
        accts.append(obj)


def evidence(case: dict, results: dict) -> tuple[dict[int, list[str]], dict[str, list[str]]]:
    """从核验通过的事实里取出交易和账户，并记下每一项为什么被列入。

    同一步的多条事实合并成一句："evidence: step 7 pass_through (n_matches, matches[0].in_amt, ...)"，
    否则同一笔交易后面会重复出现五六遍同样的来源（v1.1，D79）。"""
    txns: dict[int, dict] = {case["txn_id"]: {"alerted transaction": None}}
    accts: dict[str, dict] = {case["payer"]: {"payer of the alerted transaction": None},
                              case["payee"]: {"payee of the alerted transaction": None}}
    facts = (case["agent_result"] or {}).get("facts") or []
    for f in facts:
        if not f.get("verified") or f.get("step") in (None, 0) or f["step"] not in results:
            continue
        tool, res = results[f["step"]]
        key = f"evidence: step {f['step']} {tool}"
        t: list[int] = []
        a: list[str] = []
        _scan(_container(res, f["path"]), t, a)
        if res.get("account"):
            a.insert(0, res["account"])
        for store, items in ((txns, t), (accts, a)):
            for x in items:
                paths = store.setdefault(x, {}).setdefault(key, [])
                if f["path"] not in paths:
                    paths.append(f["path"])

    def fmt(d: dict) -> list[str]:
        return [k if p is None else f"{k} ({', '.join(p)})" for k, p in d.items()]
    return {k: fmt(v) for k, v in txns.items()}, {k: fmt(v) for k, v in accts.items()}


# ---------------------------------------------------------------------------
# 组装
# ---------------------------------------------------------------------------
def build_body(cur, case: dict) -> tuple[dict, str | None]:
    now_sgt = datetime.now(SGT)
    results = tool_results(cur, case["agent_run_id"])
    txn_why, acct_why = evidence(case, results)
    notes: list[str] = []

    # 账户：告警两端在前，证据账户按出现顺序；超过 40 个只保留前 40 个（其余的交易仍在附件里）
    acct_list = list(acct_why)
    if len(acct_list) > MAX_ROWS:
        notes.append(f"{len(acct_list) - MAX_ROWS} evidence accounts beyond the 40-account limit are omitted.")
        acct_list = acct_list[:MAX_ROWS]
    bank = case["from_bank"]
    accounts, holders, clearing = [], {}, []
    for a in acct_list:
        b, n = a.split(":", 1)
        cur.execute("SELECT entity_id FROM accounts WHERE bank_id = %s AND account_number = %s", (int(b), n))
        r = cur.fetchone()
        cur.execute("SELECT inv_is_clearing(%s, %s)", (int(b), n))
        is_clearing = cur.fetchone()[0]
        if is_clearing:
            clearing.append(a)
        entity = r[0] if r else None
        accounts.append({"account": a, "bank_id": int(b), "account_number": n,
                         "held_with_reporting_institution": int(b) == bank,
                         "entity_id": entity, "clearing_layer": is_clearing, "why": acct_why[a]})
        if entity is not None and entity not in holders:
            holders[entity] = a

    entities = []
    for entity, a in holders.items():
        b, n = a.split(":", 1)
        cur.execute("SELECT entity_type, n_accounts, n_banks FROM inv_entity_profile(%s, %s)", (int(b), n))
        etype, n_acc, n_bank = cur.fetchone()
        entities.append({"entity_id": entity, "entity_type": etype, "accounts_in_dataset": n_acc,
                         "banks_in_dataset": n_bank, "name": "Unknown", "identification": "Unknown",
                         "address": "Unknown", "relationship": "account holder",
                         "reported_accounts": [x["account"] for x in accounts if x["entity_id"] == entity]})

    # 交易：告警这一笔 + 证据交易，按时间排序；超过 40 笔时告警这一笔一定保留
    ids = [case["txn_id"]] + [t for t in txn_why if t != case["txn_id"]]
    if len(ids) > MAX_ROWS:
        notes.append(f"{len(ids) - MAX_ROWS} evidence transactions beyond the 40-transaction limit are listed only "
                     "in the attachment.")
        ids = ids[:MAX_ROWS]
    cur.execute("""SELECT txn_id, to_char(txn_ts, 'YYYY-MM-DD HH24:MI'), from_bank || ':' || from_account,
                          to_bank || ':' || to_account, round(amount_usd, 2), is_cross_currency
                   FROM inv_txn WHERE txn_id = ANY(%s) ORDER BY txn_ts, txn_id""", (ids,))
    txns = [{"txn_id": t, "time": ts, "from": f, "to": to, "amount_usd": float(amt), "cross_currency": cc,
             "why": txn_why[t]} for t, ts, f, to, amt, cc in cur.fetchall()]

    # 附件：所列账户（清算层除外）在数据里的全部交易——审查期（指南要求至少三个月）
    att_accounts = [x["account"] for x in accounts if not x["clearing_layer"]]
    cur.execute("SELECT * FROM str_attachment_rows(%s)", (att_accounts,))
    rows = cur.fetchall()
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["txn_id", "txn_time", "from_bank", "from_account", "to_bank", "to_account", "amount_usd",
                "cross_currency"])
    for r in rows:
        w.writerow([r[0], r[1].strftime("%Y-%m-%d %H:%M"), r[2], r[3], r[4], r[5], f"{r[6]:.2f}", r[7]])
    csv_text = buf.getvalue()
    span = (rows[0][1].strftime("%Y-%m-%d"), rows[-1][1].strftime("%Y-%m-%d")) if rows else (None, None)
    limitations = [LIMIT_SYNTHETIC,
                   f"Transaction data for the listed accounts covers {span[0]} to {span[1]}; the STR form guide asks "
                   "for at least three months covering the suspicious transactions, which this dataset cannot provide."]
    if clearing:
        limitations.append(f"Clearing-layer accounts {clearing} are listed but their full activity is not attached "
                           "(they process thousands of unrelated payments).")
    more = [e for e in entities if e["accounts_in_dataset"] > len(e["reported_accounts"])]
    if more:
        limitations.append("Other accounts held by the same entities are not listed because they do not appear in the "
                           "evidence: " + "; ".join(f"entity {e['entity_id']} holds {e['accounts_in_dataset']} accounts "
                                                    f"at {e['banks_in_dataset']} banks" for e in more) + ".")

    # 理由：手法由复核员确认时选定（D79）；可疑类别 = 这种手法的知识图谱原文中，审查日期仍然有效的那些
    confirm = case["confirm"] or {}
    typology = (confirm.get("payload") or {}).get("typology")
    source = "reviewer"
    if typology is None:                       # R10b 之前确认的案件没有选手法：退回 Agent 的结论，并注明
        typology = (case["agent_result"] or {}).get("typology")
        source = "agent (no typology recorded at confirmation)"
    review_date = now_sgt.date()
    cats, dropped = [], 0
    if typology:
        cur.execute("""SELECT indicator_id, doc_id, unit_id, item, quote, quote_in_force
                       FROM kb_typology_citations(%s, %s)""", (typology, review_date))
        for ind, doc, unit, item, quote, ok in cur.fetchall():
            if ok:
                cats.append({"indicator_id": ind, "doc_id": doc, "unit_id": unit, "item": item, "quote": quote})
            else:
                dropped += 1
    detection = confirm["created_at"].astimezone(SGT).strftime("%Y-%m-%d") if confirm else None

    body = {
        "schema": "str-draft-v1",
        "case_id": case["case_id"], "alert_id": case["alert_id"],
        "review_date": review_date.isoformat(),
        "reporting_institution": {"bank_id": bank, "name": f"Bank {bank} (synthetic)",
                                  "basis": "bank of the payer account of the alerted transaction",
                                  "notice_reference_no": "N/A"},
        "accounts": accounts,
        "entities": entities,
        "suspicious_transactions": txns,
        "attachment": {"filename": f"case{case['case_id']}_transactions.csv", "accounts": att_accounts,
                       "rows": len(rows), "from": span[0], "to": span[1]},
        "reasons": {"type_of_crime": TYPE_OF_CRIME, "typology": typology, "typology_source": source,
                    "categories": cats, "categories_not_in_force": dropped,
                    "detection_date": detection,
                    "reviewer_reason": confirm.get("reason"),
                    "narrative": None},
        "evidence_facts": [f for f in ((case["agent_result"] or {}).get("facts") or []) if f.get("verified")],
        "limitations": limitations + notes,
    }
    return body, csv_text


def insert_draft(cur, case: dict, actor: str, body: dict, csv_text: str | None, generator: dict) -> tuple[int, str]:
    """写入草稿 + 记录复核事件，在同一个事务里：要么都成功，要么都不发生。"""
    cur.execute("""INSERT INTO str_drafts (case_id, version, created_by, generator, body, attachment_csv)
                   VALUES (%s, 0, %s, %s, %s, %s) RETURNING version, attachment_sha256""",
                (case["case_id"], actor, json.dumps(generator), json.dumps(body, ensure_ascii=False), csv_text))
    version, sha = cur.fetchone()
    action = "draft_str" if case["state"] == "suspicious_confirmed" else "edit_str"
    act(cur, case["case_id"], action, actor, payload={"draft_version": version})
    return version, sha


def latest(cur, case_id: int, version: int | None = None):
    cur.execute("""SELECT version, created_by, generator, body, attachment_csv, attachment_sha256, created_at
                   FROM str_drafts WHERE case_id = %s AND (%s::int IS NULL OR version = %s)
                   ORDER BY version DESC LIMIT 1""", (case_id, version, version))
    r = cur.fetchone()
    if r is None:
        raise SystemExit(f"案件 {case_id} 没有草稿" + (f"第 {version} 版" if version else ""))
    return dict(zip(("version", "created_by", "generator", "body", "csv", "sha", "created_at"), r))


# ---------------------------------------------------------------------------
# 命令
# ---------------------------------------------------------------------------
def cmd_build(a) -> None:
    with connect_review() as conn, conn.cursor() as cur:
        case = load_case(cur, a.case_id)
        try:
            body, csv_text = build_body(cur, case)
            v, sha = insert_draft(cur, case, a.actor, body, csv_text, {"kind": "builder", "builder": BUILDER})
            conn.commit()
        except psycopg.Error as e:
            conn.rollback()
            raise SystemExit(f"被拒绝：{explain_str(e)}")
    print(f"案件 {a.case_id}：草稿第 {v} 版已写入（{a.actor}）；账户 {len(body['accounts'])}、主体 {len(body['entities'])}、"
          f"可疑交易 {len(body['suspicious_transactions'])}、附件 {body['attachment']['rows']} 笔（sha256 {sha[:12]}…）、"
          f"可疑类别 {len(body['reasons']['categories'])}")
    print("  理由（自由文本）尚未撰写：R10b-2 由模型起草并核验；在那之前可以用 revise 人工写入")


def cmd_show(a) -> None:
    with connect_review() as conn, conn.cursor() as cur:
        d = latest(cur, a.case_id, a.version)
    print(render(d))


def render(d: dict) -> str:
    b = d["body"]
    r = b["reasons"]
    out = [f"STR 草稿  案件 {b['case_id']}（告警 #{b['alert_id']}）第 {d['version']} 版  "
           f"{d['created_by']} {d['created_at'].astimezone(SGT):%Y-%m-%d %H:%M} SGT  [{d['generator'].get('kind')}]",
           f"审查日期 {b['review_date']}",
           f"\n[报告机构] {b['reporting_institution']['name']}（{b['reporting_institution']['basis']}）  "
           f"NRN: {b['reporting_institution']['notice_reference_no']}",
           f"\n[账户信息] {len(b['accounts'])} 个"]
    for x in b["accounts"]:
        out.append(f"  {x['account']:<22} {'本行' if x['held_with_reporting_institution'] else '他行'}  "
                   f"主体 {x['entity_id']}{'  清算层' if x['clearing_layer'] else ''}  ← {'; '.join(x['why'])}")
    out.append(f"\n[主体信息] {len(b['entities'])} 个（姓名、证件、地址：Unknown，合成数据没有）")
    for e in b["entities"]:
        out.append(f"  {e['entity_id']:<12} {e['entity_type']:<22} 数据中名下 {e['accounts_in_dataset']} 个账户、"
                   f"{e['banks_in_dataset']} 家银行；本报告涉及 {', '.join(e['reported_accounts'])}")
    out.append(f"\n[可疑交易] {len(b['suspicious_transactions'])} 笔")
    for t in b["suspicious_transactions"]:
        out.append(f"  {t['time']}  {t['from']:>20} → {t['to']:<20} {t['amount_usd']:>16,.2f} USD  "
                   f"#{t['txn_id']}  ← {'; '.join(t['why'])}")
    at = b["attachment"]
    out.append(f"  附件 {at['filename']}：{at['rows']} 笔（{at['from']} 至 {at['to']}），sha256 {d['sha'][:16]}…")
    out.append(f"\n[可疑理由]\n  犯罪类型：{r['type_of_crime']}\n  手法：{r['typology']}（来源：{r['typology_source']}）")
    out.append(f"  可疑类别（知识图谱原文，{b['review_date']} 仍然有效；另有 {r['categories_not_in_force']} 条已失效未列入）：")
    for c in r["categories"]:
        out.append(f"    - {c['doc_id']} {c['unit_id']}{' ' + c['item'] if c['item'] else ''}: \"{c['quote']}\"")
    out.append(f"  发现日期：{r['detection_date']}（确认可疑那条复核事件的日期）")
    if r.get("reviewer_reason"):
        out.append(f"  复核员确认可疑时的理由：{r['reviewer_reason']}")
    out.append(f"  理由（自由文本）：{r['narrative'] or '（尚未撰写）'}")
    out.append("\n[局限]")
    out.extend(f"  - {x}" for x in b["limitations"])
    return "\n".join(out)


def cmd_check(a) -> None:
    with connect_review() as conn, conn.cursor() as cur:
        d = latest(cur, a.case_id, a.version)
        cur.execute("SELECT problem, detail FROM str_draft_problems(%s, %s, %s)",
                    (a.case_id, json.dumps(d["body"]), d["created_at"]))
        problems = cur.fetchall()
        cur.execute("SELECT count(*) FROM str_attachment_rows(%s)", (d["body"]["attachment"]["accounts"],))
        n_now = cur.fetchone()[0]
    sha_ok = d["csv"] is not None and hashlib.sha256(d["csv"].encode()).hexdigest() == d["sha"]
    print(f"案件 {a.case_id} 第 {d['version']} 版")
    print(f"  数据库检查（str_draft_problems）：{'全部通过' if not problems else f'{len(problems)} 个问题'}")
    for p, det in problems:
        print(f"    ✗ {p}: {det}")
    print(f"  附件内容与写入时的 sha256 一致：{'是' if sha_ok else '否'}；附件 {d['body']['attachment']['rows']} 笔，"
          f"按所列账户现在重新计算 {n_now} 笔")
    print(f"  理由（自由文本）：{'已撰写' if (d['body']['reasons'].get('narrative') or '').strip() else '尚未撰写（不能批准）'}")


def cmd_revise(a) -> None:
    text = Path(a.narrative_file).read_text(encoding="utf-8").strip()
    with connect_review() as conn, conn.cursor() as cur:
        case = load_case(cur, a.case_id)
        d = latest(cur, a.case_id)
        body = revise_body(d["body"], text)
        try:
            v, _ = insert_draft(cur, case, a.actor, body, d["csv"],
                                {"kind": "human", "base_version": d["version"]})
            conn.commit()
        except psycopg.Error as e:
            conn.rollback()
            raise SystemExit(f"被拒绝：{explain_str(e)}")
    print(f"案件 {a.case_id}：第 {v} 版（{a.actor} 在第 {d['version']} 版基础上写入理由，{len(text)} 字符）")


def revise_body(body: dict, narrative: str) -> dict:
    """人工修改：只改理由（自由文本），审查日期更新为今天；结构化部分不允许手改——它们来自数据，要改就重新组装。"""
    b = copy.deepcopy(body)
    b["reasons"]["narrative"] = narrative
    b["review_date"] = datetime.now(SGT).date().isoformat()
    return b


def explain_str(e: psycopg.Error) -> str:
    msg = (e.diag.message_primary or str(e)) if getattr(e, "diag", None) else str(e)
    hints = {"str_invalid": "草稿没有通过数据库检查", "str_state": "案件还没有确认可疑",
             "str_version": "草稿版本不对", "str_author": "这个版本不是你写的",
             "str_no_draft": "还没有草稿", "str_narrative": "理由（自由文本）还没写，不能批准",
             "str_typology": "确认可疑时必须选定手法"}
    code = msg.split(":", 1)[0]
    return f"{hints[code]}（{msg}）" if code in hints else explain(e)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def fixture_case(cur, label: str) -> dict:
    """测试用案件（以库的所有者身份、在调用方的事务里创建）：一条还没有案件的真实告警，配一次假的分诊运行——
    第 7 步过桥查询找到 1 个匹配：付款方先收到一笔（库里真实的交易），再付出告警这一笔。
    str_draft 和 str_narrative 的自检共用它；调用方负责回滚。"""
    cur.execute("""SELECT a.alert_id, a.txn_id, a.txn_ts, a.from_bank, a.from_account, a.amount_usd,
                          (SELECT i.txn_id FROM inv_txn i WHERE i.to_bank = a.from_bank
                             AND i.to_account = a.from_account AND i.txn_ts < a.txn_ts
                           ORDER BY i.txn_ts DESC LIMIT 1)
                   FROM alerts a
                   WHERE NOT EXISTS (SELECT 1 FROM review_cases c WHERE c.alert_id = a.alert_id)
                     AND NOT inv_is_clearing(a.from_bank, a.from_account)
                   ORDER BY a.alert_id DESC LIMIT 200""")
    pick = next((r for r in cur.fetchall() if r[6] is not None), None)
    if pick is None:
        raise SystemExit("找不到付款方此前收过钱的告警，无法搭建测试案件")
    aid, txn, ts, fb, fa, amt, in_txn = pick
    cur.execute("""SELECT txn_ts, from_bank || ':' || from_account, round(amount_usd, 2)
                   FROM inv_txn WHERE txn_id = %s""", (in_txn,))
    in_ts, in_from, in_amt = cur.fetchone()
    payer = f"{fb}:{fa}"
    run_id = str(uuid.uuid4())
    cur.execute("""INSERT INTO agent_runs (run_id, kind, question, as_of, alert_id, status)
                   VALUES (%s, 'triage', %s, %s, %s, 'done')""", (run_id, label, ts, aid))
    match = {"in_txn": in_txn, "in_ts": in_ts.strftime("%Y-%m-%d %H:%M"), "in_from": in_from,
             "in_amt": float(in_amt), "out_txn": txn, "out_ts": ts.strftime("%Y-%m-%d %H:%M"),
             "out_amt": round(float(amt), 2), "lag_hours": round((ts - in_ts).total_seconds() / 3600, 1)}
    result = {"step": 7, "tool": "pass_through", "account": payer, "n_matches": 1, "matches": [match]}
    cur.execute("""INSERT INTO agent_tool_calls (run_id, step, tool, args, ok, result, ms)
                   VALUES (%s, 7, 'seed:pass_through', '{}', true, %s, 0)""", (run_id, json.dumps(result)))
    snap = {"status": "done", "decision": "escalate", "typology": "gather_scatter",
            "rationale": "TRIAGE-RATIONALE-MARKER: weak signals only on the payee side.", "facts": [
        {"fact": 1, "verified": True, "step": 0, "path": "result.amount_usd", "value": round(float(amt), 2)},
        {"fact": 2, "verified": True, "step": 7, "path": "matches[0].in_amt", "value": float(in_amt)},
        {"fact": 3, "verified": False, "step": 9, "path": "x", "value": 1}]}
    cur.execute("""INSERT INTO review_cases (alert_id, route, agent_run_id, agent_decision, agent_result)
                   VALUES (%s, 'agent_escalate', %s, 'escalate', %s) RETURNING case_id""",
                (aid, run_id, json.dumps(snap)))
    cid = cur.fetchone()[0]
    act(cur, cid, "open", "system")
    return {"aid": aid, "cid": cid, "txn": txn, "in_txn": in_txn, "in_from": in_from, "payer": payer,
            "amt": round(float(amt), 2), "in_amt": float(in_amt), "run_id": run_id, "match": match}


def cmd_selftest(a) -> None:
    """在一个事务里搭一个完整的案件：一条真实告警 + 一次假的分诊运行（过桥：先收一笔、再付出告警这一笔，
    两笔都是库里真实的交易），然后以复核员身份走完 确认 → 组装 → 修改 → 批准，并逐项篡改草稿，确认数据库能拦下。
    结束时回滚，什么都不留下。"""
    results: list[bool] = []

    def check(ok: bool, what: str, detail: str = "") -> None:
        results.append(ok)
        print(f"  {'✓' if ok else '✗'} {what}" + (f"   {detail}" if detail else ""))

    def expect(cur, code: str, fn, what: str) -> None:
        cur.execute("SAVEPOINT s")
        try:
            fn()
        except psycopg.Error as e:
            cur.execute("ROLLBACK TO SAVEPOINT s")
            msg = e.diag.message_primary or str(e)
            check(code in msg, what, msg[:120])
            return
        cur.execute("ROLLBACK TO SAVEPOINT s")
        check(False, what, "没有被拒绝")

    with connect() as conn, conn.cursor() as cur:
        print("[1] 搭建测试案件（事务内）")
        fx = fixture_case(cur, "str_draft selftest")
        aid, cid, txn, in_txn, in_from, payer = (fx[k] for k in ("aid", "cid", "txn", "in_txn", "in_from", "payer"))
        print(f"  告警 #{aid}，案件 {cid}；证据：过桥 {in_from} → {payer}（#{in_txn}）→ 告警这一笔（#{txn}）")

        cur.execute("SET LOCAL ROLE sentry_review")
        print("[2] 确认可疑必须选定手法；确认之前不能写草稿")
        case = load_case(cur, cid)
        expect(cur, "str_state", lambda: insert_draft(cur, case, "alice", {}, None, {"kind": "test"}),
               "确认可疑之前写草稿 → 拒绝")
        expect(cur, "str_typology", lambda: act(cur, cid, "confirm_suspicious", "alice"), "确认时不选手法 → 拒绝")
        expect(cur, "str_typology",
               lambda: act(cur, cid, "confirm_suspicious", "alice", payload={"typology": "money_mule"}),
               "选了知识图谱里没有的手法 → 拒绝")
        act(cur, cid, "confirm_suspicious", "alice", payload={"typology": "gather_scatter"})

        print("[3] 组装草稿")
        case = load_case(cur, cid)
        body, csv_text = build_body(cur, case)
        v, sha = insert_draft(cur, case, "alice", body, csv_text, {"kind": "builder", "builder": BUILDER})
        tx = {t["txn_id"] for t in body["suspicious_transactions"]}
        ac = {x["account"] for x in body["accounts"]}
        check(v == 1, "写入第 1 版，并记录 draft_str 事件（同一事务）")
        check({txn, in_txn} <= tx, "可疑交易包含告警这一笔和证据里的那一笔", f"{sorted(tx)}")
        check(in_from in ac and payer in ac, "账户包含证据里的上家", in_from)
        check(len(tx) == 2, "未通过核验的事实（第 9 步）没有带入任何内容")
        check(len(body["reasons"]["categories"]) > 0, "可疑类别来自知识图谱",
              f"{len(body['reasons']['categories'])} 条，{body['reasons']['categories_not_in_force']} 条已失效未列入")
        check(sha == hashlib.sha256(csv_text.encode()).hexdigest(), "附件 sha256 由数据库计算，与内容一致")
        cur.execute("SELECT count(*) FROM str_draft_problems(%s, %s, now())", (cid, json.dumps(body)))
        check(cur.fetchone()[0] == 0, "数据库检查：0 个问题")

        print("[4] 篡改草稿：每一种都必须被数据库拦下")

        def tampered(fn):
            b = copy.deepcopy(body)
            fn(b)
            return lambda: cur.execute(
                "INSERT INTO str_drafts (case_id, version, created_by, generator, body) VALUES (%s, 0, 'alice', '{}', %s)",
                (cid, json.dumps(b)))
        tests = [
            ("txn_mismatch", lambda b: b["suspicious_transactions"][0].update(
                amount_usd=b["suspicious_transactions"][0]["amount_usd"] + 1), "交易金额改动 1 美元"),
            ("no_alert_txn", lambda b: b.update(suspicious_transactions=[
                t for t in b["suspicious_transactions"] if t["txn_id"] != txn]), "删掉告警这一笔"),
            ("unknown_account", lambda b: b["accounts"].append({"account": "999999:NOTREAL"}), "加一个不存在的账户"),
            ("holder_missing", lambda b: b.update(entities=[]), "删掉主体信息"),
            ("detection_date", lambda b: b["reasons"].update(detection_date="2026-01-01"), "发现日期改成别的日子"),
            ("review_date", lambda b: b.update(review_date="2026-01-01"), "审查日期倒签"),
            ("category_invalid", lambda b: b["reasons"]["categories"][0].update(
                quote=b["reasons"]["categories"][0]["quote"] + " and more"), "可疑类别原文多加几个词"),
            ("typology_mismatch", lambda b: b["reasons"].update(typology="fan_in"), "手法改成复核员没选的"),
            ("no_categories", lambda b: b["reasons"].update(categories=[]), "清空可疑类别"),
            ("attachment_count", lambda b: b["attachment"].update(rows=b["attachment"]["rows"] + 1), "附件笔数对不上"),
            ("narrative_too_long", lambda b: b["reasons"].update(narrative="x" * 10001), "理由超过 10,000 字符"),
        ]
        for code, fn, what in tests:
            expect(cur, code, tampered(fn), f"{what} → {code}")

        print("[5] 事件与版本绑定、批准")
        expect(cur, "str_version", lambda: act(cur, cid, "edit_str", "alice", payload={"draft_version": 1}),
               "同一个版本记录第二次 → 拒绝")
        expect(cur, "str_narrative", lambda: act(cur, cid, "approve_str", "carol", payload={"draft_version": 1}),
               "还没写理由就批准 → 拒绝")
        case = load_case(cur, cid)
        v2, _ = insert_draft(cur, case, "dave", revise_body(body, "Funds received from " + in_from +
                             " were forwarded as the alerted payment."), csv_text, {"kind": "human", "base_version": 1})
        check(v2 == 2, "Dave（MLRO）写入理由，第 2 版 + edit_str 事件")
        expect(cur, "str_author", lambda: act(cur, cid, "edit_str", "carol", payload={"draft_version": 2}),
               "把别人写的版本记在自己名下 → 拒绝")
        expect(cur, "str_version", lambda: act(cur, cid, "approve_str", "carol", payload={"draft_version": 1}),
               "批准的不是最新版本（第 1 版）→ 拒绝")
        expect(cur, "review_four_eyes", lambda: act(cur, cid, "approve_str", "dave", payload={"draft_version": 2}),
               "改过草稿的 MLRO 不能批准自己的版本（双人复核）")
        seq, st = act(cur, cid, "approve_str", "carol", payload={"draft_version": 2})
        check(st == "str_approved", "另一位 MLRO（Carol）批准第 2 版")

        print("[6] 草稿只追加")
        cur.execute("RESET ROLE")
        expect(cur, "review_immutable",
               lambda: cur.execute("UPDATE str_drafts SET body = '{}' WHERE case_id = %s", (cid,)),
               "所有者也不能修改草稿")
        conn.rollback()

    print(f"\n{sum(results)}/{len(results)} 项通过；测试数据已全部回滚")
    if not all(results):
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description="STR 草稿（R10b-1）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("case_id", type=int)
    p.add_argument("--as", dest="actor", required=True)
    for name in ("show", "check"):
        p = sub.add_parser(name)
        p.add_argument("case_id", type=int)
        p.add_argument("--version", type=int)
    p = sub.add_parser("revise")
    p.add_argument("case_id", type=int)
    p.add_argument("--as", dest="actor", required=True)
    p.add_argument("--narrative-file", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"build": cmd_build, "show": cmd_show, "check": cmd_check, "revise": cmd_revise,
     "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()