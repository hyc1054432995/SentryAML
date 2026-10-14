"""
SentryAML - R11b：复核界面的测试
================================
四组：页面与安全（登录跳转、内容安全策略、转义）、先证据后结论、草稿流程（组装 → 起草理由 → 修改 → 批准，
经过 R11b 的三个草稿接口）、解读提示的规则（三类提示各自该触发 / 不该触发的句子）。

与 review_api_test.py 相同的做法：全部在一个"强制回滚"的事务里，接口的数据库依赖换成这个连接；
接口执行时 SET LOCAL ROLE sentry_review。模型调用换成假的（不调用 API、不花钱）：这里测的是接口、数据库规则
和页面，不是模型写得好不好（那是 R10d 的评测）。

用法：python scripts/review_web_test.py
"""

from __future__ import annotations

import os
import re
import secrets

import psycopg

os.environ["SESSION_SECRET"] = secrets.token_urlsafe(48)

from fastapi.testclient import TestClient  # noqa: E402

import review_api  # noqa: E402
import review_auth as auth  # noqa: E402
import review_hints as rh  # noqa: E402
import str_draft as sd  # noqa: E402
import str_narrative as narr  # noqa: E402
from db import get_dsn  # noqa: E402
from review import act  # noqa: E402

results: list[tuple[bool, str]] = []


def check(ok, what: str, detail: str = "") -> None:
    results.append((bool(ok), what))
    print(f"  {'✓' if ok else '✗'} {what}" + (f"   {detail}" if detail else ""))


def code_of(r) -> str:
    try:
        d = r.json().get("detail")
    except Exception:
        return ""
    return d.get("code", "") if isinstance(d, dict) else ("validation" if isinstance(d, list) else "")


SCRIPT = "<script>alert('x')</script>"


def fake_narrative(text: str, facts: list[dict], status: str = "verified"):
    """替换 str_narrative.draft_narrative：返回一份运行记录，形状与真实的一样。"""
    def f(results, material, draft, use_cache=True):
        rec = {"prompt_version": narr.PROMPT_VERSION, "model": "fake", "usage": {}, "cost_usd": 0.0,
               "attempts": [{"attempt": 1, "ok": status == "verified",
                             "problems": [] if status == "verified" else [{"problem": "ungrounded_numbers",
                                                                          "detail": "numbers not found: ['999']"}]}],
               "status": status}
        if status == "verified":
            rec.update(narrative=text, facts=facts)
        return rec
    return f


def main() -> None:
    saved = []
    narr.save_run = lambda rec: saved.append(rec)              # 不写 reports/ 下的运行记录
    conn = psycopg.connect(get_dsn(), autocommit=True)
    with conn.transaction(force_rollback=True), conn.cursor() as cur:
        # ---------------- 准备（库的所有者身份） ----------------
        fx = sd.fixture_case(cur, "R11b web test")          # 真实告警 + 假的分诊运行（第 7 步过桥 1 个匹配）
        cid = fx["cid"]
        cur.execute("""SELECT a.alert_id FROM alerts a WHERE NOT EXISTS
                       (SELECT 1 FROM review_cases c WHERE c.alert_id = a.alert_id) ORDER BY a.alert_id LIMIT 2""")
        free = [r[0] for r in cur.fetchall()]
        cur.execute("""INSERT INTO review_cases (alert_id, route, agent_decision, agent_result)
                       VALUES (%s, 'agent_escalate', 'escalate', %s) RETURNING case_id""",
                    (free[0], psycopg.types.json.Jsonb({"status": "done", "decision": "escalate", "typology": "fan_in",
                                                        "confidence": "low", "rationale": "RATIONALE " + SCRIPT,
                                                        "facts": []})))
        c_xss = cur.fetchone()[0]
        act(cur, c_xss, "open", "system")
        cur.execute("""INSERT INTO review_cases (alert_id, route, agent_result) VALUES (%s, 'tier1', %s)
                       RETURNING case_id""", (free[1], psycopg.types.json.Jsonb({"note": "tier1"})))
        c_t1 = cur.fetchone()[0]
        act(cur, c_t1, "open", "system")
        pw = {r: secrets.token_urlsafe(18) for r in ("alice", "bob", "carol", "dave")}
        for r, p in pw.items():
            auth.set_password(cur, r, p)
        # 清算层账户从库里取一个真实的（沙盒和真实库各自有自己的清算层账户；第一版写死了沙盒里造的账号，
        # 在真实库上 inv_is_clearing 为假，三项检查失败——D86）
        cur.execute("SELECT bank_id || ':' || account_number FROM account_degrees WHERE total_cp_count > 200 "
                    "ORDER BY 1 LIMIT 1")
        row = cur.fetchone()
        if row is None:
            raise SystemExit("库里没有清算层账户（account_degrees.total_cp_count > 200），无法测试清算层提示")
        clr = row[0]
        cur.execute("SELECT count(*) FROM kb_typology_citations('gather_scatter', (now() AT TIME ZONE 'Asia/Singapore')::date) "
                    "WHERE quote_in_force")
        n_cats = cur.fetchone()[0]
        cur.execute("SET LOCAL ROLE sentry_review")

        def shared_conn():
            yield conn
        review_api.app.dependency_overrides[review_api.get_db] = shared_conn

        def client():
            return TestClient(review_api.app, base_url="https://testserver")

        def login(c, who):
            r = c.post("/api/login", json={"reviewer_id": who, "password": pw[who]})
            c.headers[auth.CSRF_HEADER] = r.json()["csrf"]
            return r

        def state(case_id):
            cur.execute("SELECT state FROM review_case_state WHERE case_id = %s", (case_id,))
            return cur.fetchone()[0]

        def versions(case_id):
            cur.execute("SELECT count(*) FROM str_drafts WHERE case_id = %s", (case_id,))
            return cur.fetchone()[0]

        alice, bob, carol, anon = client(), client(), client(), client()

        # ---------------- [1] 页面与安全 ----------------
        print("[1] 页面与安全")
        r = anon.get("/queue", follow_redirects=False)
        check(r.status_code == 303 and r.headers["location"] == "/login", "没登录打开队列 → 跳到登录页")
        r = anon.get("/login")
        csp = r.headers.get("content-security-policy", "")
        check(r.status_code == 200 and "login-form" in r.text and "script-src 'self'" in csp
              and "frame-ancestors 'none'" in csp, "登录页带内容安全策略：只允许本站脚本", csp[:40] + "…")
        inline = [m for m in re.findall(r"<script[^>]*>", r.text) if "src=" not in m]
        check(not inline and " style=" not in r.text, "页面里没有内联脚本和内联样式（否则会被策略拦下）")
        check("content-security-policy" not in anon.get("/api/docs").headers,
              "接口文档页不套用这条策略（Swagger 要从 CDN 加载）")
        login(alice, "alice")
        login(bob, "bob")
        login(carol, "carol")
        r = alice.get("/queue")
        check(r.status_code == 200 and f"/cases/{cid}" in r.text and "escalate" not in r.text,
              "登录后看到队列；队列里不显示 Agent 的结论")
        meta = re.search(r'<meta name="csrf" content="([0-9a-f]+)">', alice.get(f"/cases/{cid}").text)
        check(meta and meta.group(1) == alice.get("/api/me").json()["csrf"],
              "页面里的 CSRF 令牌与会话对应（按钮靠它调用接口）")
        r = alice.get(f"/cases/{c_xss}")
        check("&lt;script&gt;" in r.text and SCRIPT not in r.text, "Agent 理由里的 <script> 被转义成文字，不会执行")

        # ---------------- [2] 先证据，后结论 ----------------
        print("[2] 先证据，后结论")
        html = alice.get(f"/cases/{cid}").text
        i_ev, i_ag = html.find("2 证据"), html.find("3 Agent 的结论")
        check(0 < i_ev < i_ag, "案件页：证据在前，Agent 结论在后")
        check("<details>" in html and "<details open" not in html and html.find("TRIAGE-RATIONALE-MARKER") > i_ag,
              "Agent 结论默认折叠，点开才看到")
        html = alice.get(f"/cases/{c_t1}").text
        check("没有 Agent 调查的证据" in html and "TRIAGE-RATIONALE" not in html, "T1 案件：没有 Agent 证据和结论")
        check('<option value="">请选择</option>' in html and "selected" not in html,
              "确认可疑的手法下拉框不预选 Agent 的手法（由复核员选定，D79）")

        # ---------------- [3] 草稿流程 ----------------
        print("[3] 草稿流程（经过 R11b 的草稿接口）")
        alice.post(f"/api/cases/{cid}/claim")
        r = alice.post(f"/api/cases/{cid}/actions", json={"action": "confirm_suspicious", "seen_state": "pending_review",
                                                          "typology": "gather_scatter"})
        check(r.status_code == 200, "alice 确认可疑（手法 gather_scatter）")
        r = alice.post(f"/api/cases/{cid}/drafts", json={"seen_state": "pending_review"})
        check(code_of(r) == "review_stale" and versions(cid) == 0,
              "按旧页面组装草稿 → 409 review_stale，草稿没有写入（防过期在数据库里，同一个事务）")
        r = alice.post(f"/api/cases/{cid}/drafts", json={"seen_state": "suspicious_confirmed"})
        check(r.status_code == 200 and r.json()["version"] == 1 and state(cid) == "str_drafted",
              "组装草稿 → 第 1 版，状态变为 STR 草稿中", str(r.json())[:80])
        r = bob.post(f"/api/cases/{cid}/drafts", json={"seen_state": "str_drafted"})
        check(code_of(r) == "review_claimed" and versions(cid) == 1, "alice 认领期间 bob 重新组装 → 409（草稿表上也有租约检查）")
        a = alice.get(f"/api/cases/{cid}/aids").json()
        check(a["draft"]["version"] == 1 and a["draft"]["problems"] == [] and not a["draft"]["has_narrative"]
              and len(a["draft"]["body"]["reasons"]["categories"]) == n_cats >= 2,
              f"aids：第 1 版通过数据库检查，还没有理由，可疑类别 = 知识图谱里仍有效的 {n_cats} 条")
        r = carol.post(f"/api/cases/{cid}/actions", json={"action": "approve_str", "seen_state": "str_drafted",
                                                          "draft_version": 1})
        check(code_of(r) == "str_narrative", "没有理由就批准 → 拒绝 str_narrative")

        in_amt = fx["in_amt"]
        text = (f"On {fx['match']['in_ts'][:10]} the payer {fx['payer']} received {in_amt:,.2f} USD from {fx['in_from']}. "
                f"It paid out {fx['amt']:,.2f} USD to the payee {fx['match']['lag_hours']} hours later. "
                f"Note for testing: {SCRIPT} must stay text.")
        facts = [{"step": 7, "path": "matches[0].in_amt", "value": in_amt, "verified": True}]
        narr.draft_narrative = fake_narrative(text, facts)
        r = alice.post(f"/api/cases/{cid}/narrative", json={"draft_version": 2})
        check(code_of(r) == "draft_stale" and not saved, "按不是最新的版本起草 → 409 draft_stale，没有调用模型")
        r = alice.post(f"/api/cases/{cid}/narrative", json={"draft_version": 1})
        cur.execute("SELECT created_by, generator ->> 'kind' FROM str_drafts WHERE case_id = %s AND version = 2", (cid,))
        check(r.status_code == 200 and r.json()["version"] == 2 and cur.fetchone() == ("alice", "model") and len(saved) == 1,
              "起草理由（假模型）→ 第 2 版，作者是点按钮的 alice；运行记录照常保存")
        narr.draft_narrative = fake_narrative("", [], status="failed")
        r = alice.post(f"/api/cases/{cid}/narrative", json={"draft_version": 2})
        check(code_of(r) == "narrative_failed" and versions(cid) == 2 and "ungrounded_numbers" in r.text,
              "起草没通过核验 → 422，不写入新版本，告诉复核员问题在哪")
        a = alice.get(f"/api/cases/{cid}/aids").json()
        sents = a["draft"]["sentences"]
        cited = [x for s in sents for x in s["numbers"] if any("(cited)" in y for y in x["sources"])]
        check(len(sents) == 3 and cited and cited[0]["number"] == f"{in_amt:,.2f}",
              "逐句对照：3 句，引用的事实标为 cited", f"{cited[0]['number']} ← {cited[0]['sources'][0]}" if cited else "")
        html = alice.get(f"/cases/{cid}").text
        check("&lt;script&gt;" in html and SCRIPT not in html, "模型写的理由里的 <script> 在页面上被转义")
        cats = [c_["indicator_id"] for c_ in a["draft"]["body"]["reasons"]["categories"]]
        r = alice.post(f"/api/cases/{cid}/revise", json={"draft_version": 2, "drop_categories": ["nope"]})
        check(code_of(r) == "category_unknown", "删除不存在的类别 → 422")
        r = alice.post(f"/api/cases/{cid}/revise", json={"draft_version": 2, "drop_categories": cats})
        check(code_of(r) == "str_invalid" and versions(cid) == 2,
              "删掉全部类别 → 数据库拒绝（至少要留一条，no_categories）")
        n_box = html.count('name="drop"')
        check(n_box == (len(cats) if len(cats) > 1 else 0),
              "页面上可勾选去掉的类别：多于 1 条才显示；只剩 1 条时不给勾选（去掉就是 0 条，一定被拒绝）", f"{len(cats)} 条 → {n_box} 个勾选框")
        r = alice.post(f"/api/cases/{cid}/revise", json={"draft_version": 2})
        check(r.status_code == 422 and code_of(r) == "validation" and "没有要保存的修改" in r.text, "什么都没改 → 422，中文说明")
        new_text = text.replace("Note for testing", "Reviewer note")
        r = alice.post(f"/api/cases/{cid}/revise", json={"draft_version": 2, "drop_categories": [cats[0]],
                                                         "narrative": new_text})
        cur.execute("SELECT body, generator FROM str_drafts WHERE case_id = %s AND version = 3", (cid,))
        row = cur.fetchone()
        check(r.status_code == 200 and row and [c_["indicator_id"] for c_ in row[0]["reasons"]["categories"]] == cats[1:]
              and "narrative_facts" not in row[0]["reasons"] and row[1]["narrative_edited"] is True
              and row[0]["reasons"]["categories_dropped_by_reviewer"] == [cats[0]],
              "删一条类别并改理由 → 第 3 版；改过的理由不再带旧的依据事实；记下删了哪条")
        html3 = alice.get(f"/cases/{cid}").text
        left = len(cats) - 1
        check(html3.count('name="drop"') == (left if left > 1 else 0) and (left > 1 or "不能去掉" in html3),
              "第 3 版页面：勾选框按剩下的类别数重新判断", f"剩 {left} 条")
        r = alice.post(f"/api/cases/{cid}/revise", json={"draft_version": 2, "narrative": "x" * 300})
        check(code_of(r) == "draft_stale", "在第 2 版上再改（最新已是第 3 版）→ 409 draft_stale")
        r = carol.post(f"/api/cases/{cid}/actions", json={"action": "approve_str", "seen_state": "str_drafted",
                                                          "draft_version": 2})
        check(code_of(r) == "str_version", "MLRO 批准旧版本 → 拒绝 str_version")
        r = alice.post(f"/api/cases/{cid}/actions", json={"action": "approve_str", "seen_state": "str_drafted",
                                                          "draft_version": 3})
        check(code_of(r) == "review_role", "分析员批准 → 403")
        r = carol.post(f"/api/cases/{cid}/actions", json={"action": "approve_str", "seen_state": "str_drafted",
                                                          "draft_version": 3})
        check(r.status_code == 200 and state(cid) == "str_approved", "另一位 MLRO 批准第 3 版 → STR 已批准")

        # ---------------- [4] 解读提示的规则 ----------------
        print("[4] 解读提示的规则（review_hints）")

        def kinds(res):
            return [h["kind"] for h in rh.case_hints(cur, res)]

        def flagged(res, sentence):
            hs = rh.case_hints(cur, res)
            return [hs[k]["kind"] for k in rh.sentence_view(sentence, res, hs)[0]["hints"]]

        a0 = {"step": 0, "tool": "alert", "result": {"payer": clr, "payee": "12:B1", "amount_usd": 1000.0}}
        fan = {"step": 3, "tool": "fan_profile", "account": clr,
               "result": {"window_hours": 72, "out_cnt": 3177, "out_cp": 2950, "clearing_layer": True}}
        res = {0: a0, 3: fan}
        check(kinds(res) == ["clearing"], "付款方是清算层账户 → 案件级提示 1 条")
        check(flagged(res, "The payer made 3,177 outbound transfers, dispersing funds widely.") == ["clearing"],
              "用\u201c扩散\u201d解读清算层账户的交易量 → 标出")
        check(flagged(res, "That is a heavy one-directional dispersal of funds.") == ["clearing"],
              "没有数字、没有账户号，只用代词指上一句，但在解读成扩散 → 标出（第一版漏掉的就是这种）")
        check(flagged(res, f"In the 72 hours to the alert, {clr} shows 3,177 outbound transfers to 2,950 counterparties.")
              == [], "如实写出清算层账户的笔数和对手方数量 → 不标（第一版在真实数据上误标的就是这种）")
        check(flagged(res, f"On 2022-09-10 {clr} paid 1,000.00 USD to 12:B1.") == [],
              "只是写出清算层账户号 → 不标")
        check(kinds({0: dict(a0, result=dict(a0["result"], payer="70:A1"))}) == [],
              "普通账户 → 没有清算层提示（清算层只有一个定义：inv_is_clearing）")
        pt = {"step": 7, "tool": "pass_through", "account": "12:B1", "n_matches": 3, "matches": [
            {"in_txn": 801, "in_amt": 11376.97, "out_txn": 901, "out_amt": 10937.55},
            {"in_txn": 802, "in_amt": 11376.97, "out_txn": 901, "out_amt": 10937.55},
            {"in_txn": 803, "in_amt": 500.0, "out_txn": 902, "out_amt": 480.0}]}
        res = {0: dict(a0, result=dict(a0["result"], payer="70:A1")), 7: pt}
        check(kinds(res) == ["shared_out"], "两笔转入匹配到同一笔转出 → 提示 1 条（另一对一对一的匹配不提示）")
        check(flagged(res, "Both 11,376.97 USD credits were moved out together as a single payment of 10,937.55 USD.")
              == ["shared_out"], "句子写共用转出的金额 → 标出")
        check(flagged(res, "A further 500.00 USD was passed on as 480.00 USD.") == [],
              "句子只写一对一的那对匹配 → 不标")
        check(flagged(res, "The payee received two credits of 11,376.97 USD each.") == [],
              "句子只如实写收到的两笔转入 → 不标（错的是把共用的转出说成一起转出）")
        pt1 = dict(pt, matches=pt["matches"][2:])
        check(kinds({0: res[0], 7: pt1}) == [], "没有共用转出 → 没有提示")
        hist = {"step": 4, "tool": "counterparty_history", "a_to_b": {"n": 3, "amt": 3000.0}, "b_to_a": {"n": 0, "amt": 0}}
        res = {0: res[0], 4: hist}
        check(kinds(res) == ["regular_repeat"], "此前已付过 3 笔、平均金额与本笔相同 → 规律性付款提示")
        check(flagged(res, "This repeat payment of an identical amount is a reason for suspicion.") == []
              and flagged(res, "The same pair had moved the same amount once before.") == [],
              "规律性付款只做案件级提示，不逐句标注（\u201c当作理由\u201d和\u201c如实陈述\u201d用的是同样的词，规则分辨不了）")
        check(kinds({0: res[0], 4: dict(hist, a_to_b={"n": 3, "amt": 4500.0})}) == [],
              "此前付款的平均金额与本笔不同 → 不提示")
        hs1 = rh.case_hints(cur, {0: res[0], 4: dict(hist, a_to_b={"n": 1, "amt": 1000.0})})
        check([h["kind"] for h in hs1] == ["regular_repeat"] and "不足以证明规律性" in hs1[0]["text"],
              "此前 1 笔、金额相同 → 提示，并如实说明 1 笔不足以证明规律性（第一版要求 2 笔，在真实数据上从不触发）")
        check(kinds({0: res[0], 4: dict(hist, a_to_b={"n": 1, "amt": 640.0})}) == [], "此前 1 笔、金额不同 → 不提示")
        check(flagged(res, "The alerted payment of 1,000.00 USD went to a payee with 1 inbound counterparty.") == [],
              "句子写到告警金额和数字 1 → 不标")
        review_api.app.dependency_overrides.clear()

    conn.close()
    n_ok = sum(ok for ok, _ in results)
    print(f"\n{n_ok}/{len(results)} 项通过；测试数据已全部回滚，没有调用模型")
    if n_ok != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()