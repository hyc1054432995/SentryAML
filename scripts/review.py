"""
SentryAML - R10a：人工复核（命令行）
====================================
Agent 判完之后，人怎么接手。规则全部在数据库里（sql/025_review.sql）：这个脚本只负责把动作写进去、
把被拒绝的原因翻译给人看。脚本里没有任何"能不能做"的判断——如果有，就会出现两份规则，
而且绕过脚本直接写库的路径不受约束（D78）。

用法：
    python scripts/review.py open --runs reports/r9b_dev_runs.json      # 流水线：按方案 B 把告警送进复核
    python scripts/review.py queue [--state pending_review]
    python scripts/review.py show 3
    python scripts/review.py act 3 confirm_suspicious --as alice --reason "..."
    python scripts/review.py smoke                                      # 冒烟测试：全部在一个事务里，结束时回滚

两种连接：
    open、smoke  主连接（流水线 / 测试），建案件是流水线的事
    queue、show、act  复核员连接（sentry_review）：只能读、只能追加事件，读不到标签

身份：命令行里复核员是谁由 --as 声明，只用于开发和演示。R11a 起，多人使用走网页接口（review_api.py），
身份来自登录会话，请求里根本不接受 actor 字段（D85）。别人认领的案件，命令行同样会被数据库拒绝（027）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import psycopg

from db import connect, connect_review

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QA_SEED = "r10-qa-v1"
QA_RATE = 0.10


# ---------------------------------------------------------------------------
# 流水线：送进复核
# ---------------------------------------------------------------------------
def qa_sampled(alert_id: int, rate: float, seed: str = QA_SEED) -> bool:
    """被 Agent 关闭的告警是否被质量抽查抽中。

    用哈希而不是 random：结果只取决于告警编号和种子，与处理顺序、这一批里还有哪些告警无关——
    同一条告警今天抽中，明天重跑也抽中，审计时可以复现"为什么是这些被抽查"。
    """
    h = int.from_bytes(hashlib.sha256(f"{seed}:{alert_id}".encode()).digest()[:8], "big")
    return h / 2 ** 64 < rate


def snapshot(rec: dict) -> dict:
    """复核员要看到的 Agent 结论。打开案件时存下来，之后不再变化。"""
    keys = ("status", "error", "decision", "typology", "confidence", "rationale", "facts", "not_checked",
            "ungrounded_numbers", "verified", "prompt_version", "model", "run_id")
    return {k: rec.get(k) for k in keys if k in rec}


def route_of(tier: str, rec: dict, qa_rate: float) -> str | None:
    """方案 B（D76）：T1 全部人工复核；T2 被升级（或 Agent 出错，D75）的进复核；T2 被关闭的按比例抽查。"""
    if tier == "T1":
        return "tier1"
    if rec.get("status") != "done" or rec.get("decision") == "escalate":
        return "agent_escalate"
    return "qa_sample" if qa_sampled(rec["alert_id"], qa_rate) else None


def cmd_open(a) -> None:
    runs = json.loads(Path(a.runs).read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT alert_id, tier FROM alerts WHERE alert_id = ANY(%s)", ([int(k) for k in runs],))
        tier = dict(cur.fetchall())
        cur.execute("SELECT run_id::text FROM agent_runs WHERE run_id::text = ANY(%s)",
                    ([r.get("run_id") for r in runs.values() if r.get("run_id")],))
        known_runs = {r[0] for r in cur.fetchall()}
        for key in sorted(runs, key=int):
            rec = runs[key]
            aid = int(key)
            route = route_of(tier[aid], rec, a.qa_rate)
            if route is None:
                counts["agent_closed_not_sampled"] = counts.get("agent_closed_not_sampled", 0) + 1
                continue
            if route == "tier1":
                # 方案 B 下 T1 不由 Agent 决定：不给复核员看 Agent 的结论，免得它变成事实上的决定
                decision, result, run_id = None, {"note": "tier1: routed to human review without agent decision (plan B)"}, None
            else:
                decision = rec.get("decision") if rec.get("status") == "done" else None
                result = snapshot(rec)
                run_id = rec.get("run_id") if rec.get("run_id") in known_runs else None
            cur.execute("""INSERT INTO review_cases (alert_id, route, agent_run_id, agent_decision, agent_result)
                           VALUES (%s, %s, %s, %s, %s) ON CONFLICT (alert_id) DO NOTHING RETURNING case_id""",
                        (aid, route, run_id, decision, json.dumps(result, ensure_ascii=False, default=str)))
            row = cur.fetchone()
            if row is None:
                counts["already_open"] = counts.get("already_open", 0) + 1
                continue
            cur.execute("""INSERT INTO review_events (case_id, actor, action, seen_state, from_state, to_state, payload)
                           VALUES (%s, 'system', 'open', 'new', '', '', %s)""",
                        (row[0], json.dumps({"route": route, "qa_rate": a.qa_rate if route == "qa_sample" else None})))
            counts[route] = counts.get(route, 0) + 1
        conn.commit()
    print(f"{a.runs}：{len(runs)} 条运行记录")
    for k, v in sorted(counts.items()):
        print(f"  {k:<26}{v:>4}")


# ---------------------------------------------------------------------------
# 复核员
# ---------------------------------------------------------------------------
def cmd_queue(a) -> None:
    with connect_review() as conn, conn.cursor() as cur:
        cur.execute("""SELECT s.case_id, s.alert_id, s.route, s.agent_decision, s.state, s.last_actor, s.last_at, a.tier
                       FROM review_case_state s JOIN alerts a USING (alert_id)
                       WHERE (%s::text IS NULL OR s.state = %s) ORDER BY s.case_id""", (a.state, a.state))
        rows = cur.fetchall()
    print(f"  {'案件':>4} {'告警':>6} {'档':<3}{'来源':<16}{'Agent':<10}{'状态':<22}最后操作")
    for cid, aid, route, dec, state, actor, at, tier in rows:
        print(f"  {cid:>4} {aid:>6} {tier:<3}{route:<16}{dec or '-':<10}{state:<22}"
              f"{actor or ''} {at.strftime('%m-%d %H:%M') if at else ''}")
    print(f"  共 {len(rows)} 个案件")


def cmd_show(a) -> None:
    with connect_review() as conn, conn.cursor() as cur:
        cur.execute("""SELECT c.case_id, c.alert_id, c.route, c.agent_decision, c.agent_result, c.agent_run_id,
                              s.state, a.txn_ts, a.from_bank, a.from_account, a.to_bank, a.to_account,
                              a.amount_usd, a.tier, a.score_pct
                       FROM review_cases c JOIN review_case_state s ON s.case_id = c.case_id
                       JOIN alerts a ON a.alert_id = c.alert_id
                       WHERE c.case_id = %s""", (a.case_id,))
        row = cur.fetchone()
        if row is None:
            raise SystemExit(f"没有案件 {a.case_id}")
        (cid, aid, route, dec, result, run_id, state, ts, fb, fa, tb, ta, amt, tier, pct) = row
        tools = {}
        if run_id:
            cur.execute("SELECT step, tool FROM agent_tool_calls WHERE run_id = %s", (run_id,))
            tools = dict(cur.fetchall())
        cur.execute("""SELECT seq, actor, action, from_state, to_state, overrides_agent, reason, created_at, payload
                       FROM review_events WHERE case_id = %s ORDER BY seq""", (cid,))
        events = cur.fetchall()
        cur.execute("SELECT to_regclass('str_drafts') IS NOT NULL")
        drafts = []
        if cur.fetchone()[0]:
            cur.execute("""SELECT version, created_by, generator ->> 'kind', created_at FROM str_drafts
                           WHERE case_id = %s ORDER BY version""", (cid,))
            drafts = cur.fetchall()
    print(f"案件 {cid}  告警 #{aid}  当前状态：{state}  来源：{route}")
    print(f"  {ts:%Y-%m-%d %H:%M}  {fb}:{fa} → {tb}:{ta}  {float(amt):,.2f} USD  {tier}（分位 {pct:.3f}）")
    # 先给证据，再给结论：复核员先看到查到了什么，再看到 Agent 怎么判（自动化偏差，D78）
    if result.get("facts"):
        print("  Agent 查到的证据（每条都已由程序核对过工具结果）：")
        for f in result["facts"]:
            step = f.get("step")
            tool = "alert" if step == 0 else tools.get(step, "?")
            print(f"    {'✓' if f.get('verified') else '✗'} 第 {step} 步 {tool:<20} {f.get('path')} = {f.get('value')}")
    if result.get("not_checked"):
        print(f"  Agent 没能查的：{result['not_checked']}")
    if route == "tier1":
        print("  T1：直接进入人工复核，没有 Agent 结论（方案 B）")
    elif result.get("status") != "done":
        print(f"  Agent 出错，没有结论：{result.get('error')}")
    else:
        print(f"  Agent 结论：{result['decision']}  手法 {result.get('typology')}  把握 {result.get('confidence')}")
        print(f"  理由：{result.get('rationale')}")
    print("  复核记录：")
    for seq, actor, action, fs, ts2, ov, reason, at, payload in events:
        extra = {k: v for k, v in (payload or {}).items() if k in ("typology", "draft_version")}
        print(f"    {seq}. {at:%m-%d %H:%M:%S} {actor:<7}{action:<20}{fs} → {ts2}"
              + ("  [推翻 Agent]" if ov else "") + (f"  {extra}" if extra else "")
              + (f"\n         理由：{reason}" if reason else ""))
    if drafts:
        print("  STR 草稿：" + "；".join(f"第 {v} 版 {who} [{kind}] {at:%m-%d %H:%M}" for v, who, kind, at in drafts))


def act(cur, case_id: int, action: str, actor: str, reason: str | None = None, seen: str | None = None,
        payload: dict | None = None) -> tuple[int, str]:
    """写入一个复核动作。seen 不给时取当前状态——命令行是"看完马上执行"，界面（R11）必须传入页面加载时的状态。"""
    if seen is None:
        cur.execute("SELECT state FROM review_case_state WHERE case_id = %s", (case_id,))
        r = cur.fetchone()
        if r is None:
            raise SystemExit(f"没有案件 {case_id}")
        seen = r[0]
    # from_state、to_state 随便填什么都会被触发器覆盖；这里留空，表明它们不由客户端决定
    cur.execute("""INSERT INTO review_events (case_id, actor, action, seen_state, from_state, to_state, reason, payload)
                   VALUES (%s, %s, %s, %s, '', '', %s, %s) RETURNING seq, to_state""",
                (case_id, actor, action, seen, reason, json.dumps(payload or {}, ensure_ascii=False)))
    return cur.fetchone()


def explain(e: psycopg.Error) -> str:
    msg = (e.diag.message_primary or str(e)) if getattr(e, "diag", None) else str(e)
    hints = {"review_stale": "别人已经改变了这个案件，请重新查看后再决定",
             "review_illegal": "当前状态下没有这个动作",
             "review_role": "你的角色不能做这个动作",
             "review_reason": "这个动作必须写理由",
             "review_four_eyes": "双人复核：确认可疑或写过草稿的人不能批准",
             "review_actor": "没有这个复核员",
             "review_immutable": "复核记录只能追加，不能修改或删除",
             "str_typology": "确认可疑时必须选定手法（知识图谱里的手法；命令行用 --typology）",
             "str_no_draft": "还没有草稿：先运行 str_draft.py build",
             "str_version": "草稿版本不对：批准或记录的必须是最新版本",
             "str_author": "这个草稿版本不是你写的",
             "str_narrative": "草稿的理由（自由文本）还没写，不能批准",
             "str_invalid": "草稿没有通过数据库检查",
             "str_state": "只有确认可疑之后才能写草稿",
             "review_claimed": "这个案件正被别人认领处理，等对方释放或租约到期（MLRO 可以写明理由接管）",
             "claim_state": "已关闭或已批准的案件不需要认领",
             "claim_none": "这个案件现在没有人认领",
             "claim_role": "只有分析员和 MLRO 能认领案件",
             "session_role": "这个账号不能登录",
             "reg_role": "只有分析员和 MLRO 能问法规",
             "reg_answer": "只能记录刚生成的、还没挂到别的案件上的回答"}
    code = msg.split(":", 1)[0]
    return f"{hints.get(code, '数据库拒绝')}（{msg}）"


def cmd_act(a) -> None:
    payload = json.loads(a.payload) if a.payload else {}
    with connect_review() as conn, conn.cursor() as cur:
        if a.action == "confirm_suspicious":
            # STR 的可疑类别由手法决定，手法由复核员选定（D79）。不指定时用 Agent 给出的手法，并明确告诉复核员
            typ = a.typology
            if typ is None:
                cur.execute("SELECT agent_result ->> 'typology' FROM review_cases WHERE case_id = %s", (a.case_id,))
                r = cur.fetchone()
                typ = r[0] if r and r[0] not in (None, "none", "unclear") else None
                if typ:
                    print(f"未指定 --typology，使用 Agent 给出的手法：{typ}")
            if typ:
                payload["typology"] = typ
        if a.draft_version is not None:
            payload["draft_version"] = a.draft_version
        try:
            seq, to_state = act(cur, a.case_id, a.action, a.actor, a.reason, a.seen, payload or None)
            conn.commit()
        except psycopg.Error as e:
            conn.rollback()
            raise SystemExit(f"被拒绝：{explain(e)}")
    print(f"案件 {a.case_id}：第 {seq} 条记录，{a.actor} {a.action} → {to_state}")


# ---------------------------------------------------------------------------
# 冒烟测试
# ---------------------------------------------------------------------------
def cmd_smoke(a) -> None:
    """全部在一个事务里做，结束时回滚：复核表只追加、不能删，测试数据一旦提交就永远留在审计记录里。

    人工动作用 SET LOCAL ROLE sentry_review 执行，检查的是复核员角色实际拥有的权限，而不是主账号的。
    """
    results: list[tuple[bool, str]] = []

    def check(ok: bool, what: str, detail: str = "") -> None:
        results.append((ok, what))
        print(f"  {'✓' if ok else '✗'} {what}" + (f"   {detail}" if detail else ""))

    def expect_error(cur, code: str, fn, what: str) -> None:
        cur.execute("SAVEPOINT s")
        try:
            fn()
        except psycopg.Error as e:
            cur.execute("ROLLBACK TO SAVEPOINT s")
            msg = e.diag.message_primary or str(e)
            check(msg.startswith(code) or code in msg, what, msg[:110])
            return
        cur.execute("ROLLBACK TO SAVEPOINT s")
        check(False, what, "没有被拒绝")

    with connect() as conn, conn.cursor() as cur:
        print("[1] 静态检查")
        cur.execute("SELECT from_state, action, to_state FROM review_transitions ORDER BY 1, 2")
        trans = {(f, act_): t for f, act_, t in cur.fetchall()}
        expected = {("new", "open"): "pending_review", ("pending_review", "close"): "closed",
                    ("pending_review", "confirm_suspicious"): "suspicious_confirmed",
                    ("pending_review", "request_info"): "pending_review",
                    ("suspicious_confirmed", "draft_str"): "str_drafted", ("str_drafted", "edit_str"): "str_drafted",
                    ("str_drafted", "return_draft"): "str_drafted", ("str_drafted", "approve_str"): "str_approved",
                    ("closed", "reopen"): "pending_review"}
        check(trans == expected, f"状态机与设计一致（{len(trans)} 条转移）")
        cur.execute("""SELECT tgname FROM pg_trigger WHERE NOT tgisinternal
                       AND tgrelid IN ('review_events'::regclass, 'review_cases'::regclass)""")
        trg = {r[0] for r in cur.fetchall()}
        need = {"review_event_guard", "review_events_append_only", "review_events_no_truncate",
                "review_cases_append_only", "review_cases_no_truncate"}
        check(need <= trg, "五个触发器都在", str(sorted(need - trg)) if need - trg else "")

        # 测试用案件：用库里任意 3 条告警（没有案件的），事务结束时全部回滚
        cur.execute("""SELECT a.alert_id FROM alerts a WHERE NOT EXISTS
                       (SELECT 1 FROM review_cases c WHERE c.alert_id = a.alert_id) ORDER BY a.alert_id LIMIT 3""")
        ids = [r[0] for r in cur.fetchall()]
        if len(ids) < 3:
            raise SystemExit("需要至少 3 条还没有案件的告警")

        def new_case(aid: int, route: str, decision: str | None) -> int:
            cur.execute("""INSERT INTO review_cases (alert_id, route, agent_decision, agent_result)
                           VALUES (%s, %s, %s, '{}') RETURNING case_id""", (aid, route, decision))
            cid = cur.fetchone()[0]
            act(cur, cid, "open", "system")
            return cid

        # 026（R10b）在复核事件上加了第二个触发器，要求草稿真实存在、确认时选定手法。这里测的是 025 的通用规则，
        # 在本事务里暂时停用它（ALTER TABLE 也随回滚撤销）；026 的规则由 str_draft.py selftest 单独测试。
        # 代价：本事务持有 review_events 的排他锁，测试的这一两秒里其他复核操作会等待
        cur.execute("SELECT 1 FROM pg_trigger WHERE tgname = 'review_event_guard_str'")
        if cur.fetchone():
            cur.execute("ALTER TABLE review_events DISABLE TRIGGER review_event_guard_str")
            print("  （本事务内停用 026 的触发器 review_event_guard_str，回滚时恢复）")
        c_esc = new_case(ids[0], "agent_escalate", "escalate")
        c_qa = new_case(ids[1], "qa_sample", "close")
        c_t1 = new_case(ids[2], "tier1", None)

        print("[2] 权限（以 sentry_review 身份）")
        cur.execute("SET LOCAL ROLE sentry_review")
        expect_error(cur, "permission denied", lambda: cur.execute("SELECT count(*) FROM alert_labels"),
                     "读不到 alert_labels（复核员没有标准答案）")
        expect_error(cur, "permission denied",
                     lambda: cur.execute("INSERT INTO review_cases (alert_id, route) VALUES (%s, 'tier1')", (ids[0],)),
                     "不能自己建案件（那是流水线的事）")
        expect_error(cur, "permission denied",
                     lambda: cur.execute("UPDATE review_events SET reason = 'x' WHERE case_id = %s", (c_esc,)),
                     "不能修改事件（没有 UPDATE 权限）")
        expect_error(cur, "permission denied",
                     lambda: cur.execute("DELETE FROM review_events WHERE case_id = %s", (c_esc,)),
                     "不能删除事件（没有 DELETE 权限）")

        print("[3] 理由规则")
        expect_error(cur, "review_reason", lambda: act(cur, c_esc, "close", "alice"),
                     "推翻 Agent 的升级、关闭时不写理由 → 拒绝")
        expect_error(cur, "review_reason", lambda: act(cur, c_qa, "confirm_suspicious", "alice"),
                     "推翻 Agent 的关闭（抽查案件）、确认可疑时不写理由 → 拒绝")
        expect_error(cur, "review_reason", lambda: act(cur, c_t1, "close", "alice"),
                     "T1 没有 Agent 结论可依据、关闭时不写理由 → 拒绝")
        cur.execute("SAVEPOINT t1")
        seq, st = act(cur, c_t1, "close", "alice", "regular payroll counterparty, see payee history")
        cur.execute("SELECT overrides_agent FROM review_events WHERE case_id = %s AND seq = %s", (c_t1, seq))
        check(st == "closed" and cur.fetchone()[0] is False,
              "T1 写明理由后可以关闭；没有 Agent 结论时\"推翻 Agent\"记为 false（不是 NULL）")
        cur.execute("ROLLBACK TO SAVEPOINT t1")
        expect_error(cur, "review_reason", lambda: act(cur, c_esc, "request_info", "alice", "  "),
                     "要求补查只写空白 → 拒绝")
        seq, st = act(cur, c_esc, "request_info", "alice", "check the payee's other accounts")
        check(st == "pending_review" and seq == 2, "要求补查：留在待复核，记录序号 2", f"seq={seq} state={st}")

        print("[4] 完整路径与双人复核")
        act(cur, c_esc, "confirm_suspicious", "alice")          # 与 Agent 一致，可以不写理由
        act(cur, c_esc, "draft_str", "alice", payload={"draft_version": 1})
        act(cur, c_esc, "edit_str", "bob", payload={"draft_version": 2})
        expect_error(cur, "review_role", lambda: act(cur, c_esc, "approve_str", "bob"),
                     "分析员不能批准 STR")
        cur.execute("RESET ROLE")
        cur.execute("INSERT INTO reviewers (reviewer_id, name, role) VALUES ('mallory', 'test', 'mlro')")
        cur.execute("SET LOCAL ROLE sentry_review")
        # 一个 MLRO 自己改了草稿，再去批准
        act(cur, c_esc, "return_draft", "carol", "add the onward transfers")
        act(cur, c_esc, "edit_str", "mallory", payload={"draft_version": 3})
        expect_error(cur, "review_four_eyes", lambda: act(cur, c_esc, "approve_str", "mallory"),
                     "改过草稿的 MLRO 不能批准自己的版本（双人复核）")
        seq, st = act(cur, c_esc, "approve_str", "carol")
        check(st == "str_approved", "另一位 MLRO 批准", f"seq={seq}")
        expect_error(cur, "review_illegal", lambda: act(cur, c_esc, "close", "carol", "x"),
                     "已批准的案件不能再关闭（状态机里没有这条转移）")
        cur.execute("SELECT array_agg(seq ORDER BY seq), array_agg(to_state ORDER BY seq) FROM review_events "
                    "WHERE case_id = %s", (c_esc,))
        seqs, states = cur.fetchone()
        check(seqs == list(range(1, len(seqs) + 1)), "事件序号连续", str(seqs))
        check(states[-1] == "str_approved" and states[1] == "pending_review", "状态由数据库计算",
              " → ".join(states))

        print("[5] 防过期操作与伪造字段")
        act(cur, c_qa, "close", "bob")                          # 与 Agent 一致（关闭）：不需要理由
        expect_error(cur, "review_stale",
                     lambda: act(cur, c_qa, "confirm_suspicious", "alice", "looks like layering",
                                 seen="pending_review"),
                     "Alice 看到的是待复核，但 Bob 已经关闭了 → 拒绝，要求重新查看")
        expect_error(cur, "review_role", lambda: act(cur, c_qa, "reopen", "alice", "new information"),
                     "分析员不能重开已关闭的案件")
        seq, st = act(cur, c_qa, "reopen", "carol", "QA review disagrees with closure")
        check(st == "pending_review", "MLRO 写明理由后重开")
        cur.execute("""INSERT INTO review_events (case_id, actor, action, seen_state, from_state, to_state, seq,
                                                  overrides_agent, reason)
                       VALUES (%s, 'alice', 'confirm_suspicious', 'pending_review', 'whatever', 'str_approved', 99,
                               false, 'fan-in into a fresh multi-bank entity') RETURNING seq, from_state, to_state,
                               overrides_agent""", (c_qa,))
        seq, fs, ts, ov = cur.fetchone()
        check((seq, fs, ts, ov) == (4, "pending_review", "suspicious_confirmed", True),
              "客户端填的序号、状态、是否推翻 Agent 全部被数据库重新计算", f"seq={seq} {fs}→{ts} 推翻={ov}")

        print("[6] 只追加（以库的所有者身份）")
        cur.execute("RESET ROLE")
        expect_error(cur, "review_immutable",
                     lambda: cur.execute("UPDATE review_events SET reason = 'edited' WHERE case_id = %s", (c_esc,)),
                     "所有者也不能修改事件")
        expect_error(cur, "review_immutable",
                     lambda: cur.execute("DELETE FROM review_events WHERE case_id = %s", (c_esc,)),
                     "所有者也不能删除事件")
        expect_error(cur, "review_immutable", lambda: cur.execute("TRUNCATE review_events CASCADE"),
                     "所有者也不能清空事件表")
        expect_error(cur, "review_immutable",
                     lambda: cur.execute("UPDATE review_cases SET agent_decision = 'close' WHERE case_id = %s",
                                         (c_esc,)),
                     "所有者也不能改案件里的 Agent 结论快照")
        conn.rollback()

    n_ok = sum(ok for ok, _ in results)
    print(f"\n{n_ok}/{len(results)} 项通过；测试数据已全部回滚")
    if n_ok != len(results):
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description="人工复核（R10a）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("open", help="流水线：按方案 B 把告警送进复核")
    p.add_argument("--runs", required=True)
    p.add_argument("--qa-rate", type=float, default=QA_RATE)
    p = sub.add_parser("queue")
    p.add_argument("--state")
    p = sub.add_parser("show")
    p.add_argument("case_id", type=int)
    p = sub.add_parser("act")
    p.add_argument("case_id", type=int)
    p.add_argument("action")
    p.add_argument("--as", dest="actor", required=True)
    p.add_argument("--reason")
    p.add_argument("--seen", help="你看到的状态（默认取当前状态）")
    p.add_argument("--typology", help="confirm_suspicious 时选定的手法（R10b，D79）；不给时使用 Agent 的手法")
    p.add_argument("--draft-version", type=int,
                   help="approve_str：你审阅并批准的草稿版本（必须是最新版本，D79）")
    p.add_argument("--payload")
    sub.add_parser("smoke")
    a = ap.parse_args()
    {"open": cmd_open, "queue": cmd_queue, "show": cmd_show, "act": cmd_act, "smoke": cmd_smoke}[a.cmd](a)


if __name__ == "__main__":
    main()