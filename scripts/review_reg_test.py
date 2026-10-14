"""
SentryAML - R11c：复核时"问法规"的测试
======================================
三组：
  [1] 提问与记录：记录的复核员来自会话；问法规不改变案件；回答（含没通过核验的论断、拒答）在页面上正确显示并被转义；
      不要求认领；没登录、没带 CSRF 令牌被拒
  [2] 拦在调用模型之前：问题里有账户号、次数用完、并发名额用完、案件不存在——这些情况都不能花钱，所以检查
      "假 Agent 有没有被调用"；Agent 出错时名额要还回去、不留记录
  [3] 数据库规则（绕过接口直接写库）：一条回答只能挂一个案件、旧回答不能挂、system 不能记录、只追加、不能写问答日志

与 review_api_test.py、review_web_test.py 相同的做法：全部在一个"强制回滚"的事务里，接口的数据库依赖换成这个连接，
接口执行时 SET LOCAL ROLE sentry_review。法规 Agent 换成假的：不调用模型、不花钱；它像真的 Agent 一样把回答写进
kb_answer_log（真实的 Agent 用主账号写，这里临时切回库的所有者身份写，写完再切回复核角色）。
这里测的是接口、数据库规则和页面，不是 Agent 答得好不好（那是 R8 的评测）。

用法：python scripts/review_reg_test.py
"""

from __future__ import annotations

import json
import os
import secrets
from datetime import datetime

import psycopg

os.environ["SESSION_SECRET"] = secrets.token_urlsafe(48)

from fastapi.testclient import TestClient  # noqa: E402

import kb_agent  # noqa: E402
import review_api  # noqa: E402
import review_auth as auth  # noqa: E402
import str_draft as sd  # noqa: E402
from db import get_dsn  # noqa: E402

results: list[tuple[bool, str]] = []
SCRIPT = "<script>alert('x')</script>"


def check(ok, what: str, detail: str = "") -> None:
    results.append((bool(ok), what))
    print(f"  {'✓' if ok else '✗'} {what}" + (f"   {detail}" if detail else ""))


def code_of(r) -> str:
    try:
        d = r.json().get("detail")
    except Exception:
        return ""
    return d.get("code", "") if isinstance(d, dict) else ("validation" if isinstance(d, list) else "")


def main() -> None:
    conn = psycopg.connect(get_dsn(), autocommit=True)
    calls: list[dict] = []                 # 假 Agent 每次被调用的参数：用来证明"被拒的请求没有花钱"
    mode = {"kind": "answered"}

    def fake_research(question, as_of=None, **kw):
        calls.append({"question": question, "as_of": as_of})
        if mode["kind"] == "error":
            raise RuntimeError("upstream exploded")
        if mode["kind"] == "abstained":
            rec = {"status": "abstained", "abstain_reason": "out_of_scope", "answer": "", "claims": [],
                   "missing": "超出知识库范围：问题涉及 US；知识库只包含新加坡（MAS、STRO）与国际（FATF）材料。"}
        else:
            rec = {"status": "partial", "abstain_reason": None,
                   "answer": "File an STR promptly. " + SCRIPT,
                   "claims": [{"statement": "An STR must be filed.", "source": "mas_n626:14.1", "unit": "14.1",
                               "title": "MAS Notice 626", "quote": "a bank shall promptly submit a report",
                               "status": "ok", "verified": True},
                              {"statement": "Made-up claim.", "source": "mas_n626:99.9", "unit": "99.9",
                               "title": "MAS Notice 626", "quote": "words that are not in any clause",
                               "status": "quote_not_found", "verified": False}],
                   "missing": ""}
        rec |= {"question": question, "as_of": str(as_of), "cost_usd": 0.0042, "ms": {"total": 12300}}
        with conn.cursor() as c:                         # 真实 Agent 用主账号写日志；这里临时切回所有者身份
            c.execute("RESET ROLE")
            c.execute("""INSERT INTO kb_answer_log (question, as_of, status, abstain_reason, prompt_version, model,
                                                    record) VALUES (%s, %s, %s, %s, 'fake', 'fake', %s)
                         RETURNING answer_id""",
                      (question, as_of, rec["status"], rec["abstain_reason"], json.dumps(rec, ensure_ascii=False)))
            rec["answer_id"] = c.fetchone()[0]
            c.execute("SET LOCAL ROLE sentry_review")
        return rec

    kb_agent.research = fake_research

    with conn.transaction(force_rollback=True), conn.cursor() as cur:
        # ---------------- 准备（库的所有者身份） ----------------
        fx = sd.fixture_case(cur, "R11c regulation test")
        cid = fx["cid"]
        cur.execute("""SELECT a.from_bank || ':' || a.from_account FROM review_cases c JOIN alerts a
                       ON a.alert_id = c.alert_id WHERE c.case_id = %s""", (cid,))
        own_acct = cur.fetchone()[0]
        pw = {r: secrets.token_urlsafe(18) for r in ("alice", "bob", "carol")}
        for r, p in pw.items():
            auth.set_password(cur, r, p)
        # 上限临时改小，便于测"用完"。库里可能已有 alice 最近一小时真实问过的记录（比如刚在浏览器里试过），
        # 所以上限 = 已用 + 6，检查只比较变化量
        cur.execute("""UPDATE review_config SET value = 6 + (SELECT count(*) FROM review_reg_queries WHERE actor = 'alice'
                       AND created_at > clock_timestamp() - interval '1 hour') WHERE key = 'reg_queries_per_hour'""")
        cur.execute("SELECT review_reg_quota_left('alice')")
        left0 = cur.fetchone()[0]
        cur.execute("""INSERT INTO kb_answer_log (question, as_of, status, prompt_version, model, record, created_at)
                       VALUES ('old question', current_date, 'answered', 'fake', 'fake', '{}', now() - interval '1 hour')
                       RETURNING answer_id""")
        old_answer = cur.fetchone()[0]
        cur.execute("SET LOCAL ROLE sentry_review")

        def shared_conn():
            yield conn
        review_api.app.dependency_overrides[review_api.get_db] = shared_conn

        def client():
            return TestClient(review_api.app, base_url="https://testserver")

        def login(c, who):
            r = c.post("/api/login", json={"reviewer_id": who, "password": pw[who]})
            c.headers[auth.CSRF_HEADER] = r.json()["csrf"]

        def count(sql, *args):
            cur.execute(sql, args)
            return cur.fetchone()[0]

        def n_queries():
            return count("SELECT count(*) FROM review_reg_queries WHERE case_id = %s", cid)

        def ask(c, q, case=None):
            return c.post(f"/api/cases/{case or cid}/ask", json={"question": q})

        alice, bob, anon = client(), client(), client()
        login(alice, "alice")
        login(bob, "bob")
        url = f"/api/cases/{cid}/ask"

        # ---------------- [1] 提问与记录 ----------------
        print("[1] 提问与记录")
        state0 = count("SELECT state FROM review_case_state WHERE case_id = %s", cid)
        events0 = count("SELECT count(*) FROM review_events WHERE case_id = %s", cid)
        r = ask(alice, "  What must a bank do   when funds are moved out within hours?  ")
        cur.execute("SELECT actor, answer_id FROM review_reg_queries WHERE case_id = %s", (cid,))
        row = cur.fetchone()
        check(r.status_code == 200 and row and row[0] == "alice" and row[1] == r.json()["answer_id"],
              "alice 提问 → 记录一行，复核员来自会话、回答编号来自 Agent 写的日志", f"{r.status_code} {row}")
        sgt_today = datetime.now(sd.SGT).date()
        check(calls and calls[-1]["as_of"] == sgt_today
              and calls[-1]["question"] == "What must a bank do when funds are moved out within hours?",
              "调查日期 = 今天（新加坡时间）；问题只合并空白，不改写", str(calls[-1]["as_of"]) if calls else "")
        check(count("SELECT state FROM review_case_state WHERE case_id = %s", cid) == state0
              and count("SELECT count(*) FROM review_events WHERE case_id = %s", cid) == events0,
              "问法规不改变案件：状态和复核记录都不变")
        check(review_api.REG_SLOTS._value == review_api.REG_CONCURRENCY, "调用结束后并发名额还回去了")

        html = alice.get(f"/cases/{cid}").text
        check("What must a bank do when funds are moved out within hours?" in html and "mas_n626:14.1" in html
              and "逐字核验通过" in html and "未通过逐字核验" in html,
              "页面显示问题、出处、核验结果；没通过核验的论断标明不要引用")
        check("&lt;script&gt;" in html and SCRIPT not in html, "Agent 回答里的 <script> 在页面上被转义")
        check(f"还能问 {left0 - 1} 次" in html, "页面显示这一小时还能问几次（问一次少一次）", f"{left0} → {left0 - 1}")

        r = bob.post(f"/api/cases/{cid}/claim")
        mode["kind"] = "abstained"
        r = ask(alice, "What is the FinCEN SAR deadline in the US?")
        check(r.status_code == 200 and r.json()["status"] == "abstained",
              "bob 认领着案件，alice 仍然可以提问（问法规不改变案件，不要求认领）", r.text[:80])
        mode["kind"] = "answered"
        a = alice.get(f"/api/cases/{cid}/regulation").json()
        check([q["status"] for q in a["queries"]] == ["abstained", "partial"] and a["quota_left"] == left0 - 2,
              "问答历史新的在前；还能问的次数在数据库里数", f"{[q['status'] for q in a['queries']]} 剩 {a['quota_left']}")
        html = alice.get(f"/cases/{cid}").text
        check("超出知识库范围" in html and "没有回答" in html, "拒答的回答显示原因")

        n_calls = len(calls)
        r = anon.post(url, json={"question": "What is an STR?"})
        check(r.status_code == 401, "没登录 → 401")
        bare = client()
        bare.post("/api/login", json={"reviewer_id": "carol", "password": pw["carol"]})
        r = bare.post(url, json={"question": "What is an STR?"})
        check(code_of(r) == "csrf" and len(calls) == n_calls, "没带 CSRF 令牌 → 403，没有调用 Agent")

        # ---------------- [2] 拦在调用模型之前 ----------------
        print("[2] 拦在调用模型之前（被拒的请求一分钱都不花）")
        n_calls, n0 = len(calls), n_queries()
        r = ask(alice, "Is 222:811B6EEF0 required to be reported?")
        check(code_of(r) == "reg_case_data" and len(calls) == n_calls and n_queries() == n0,
              "问题里有账户号 → 422，没有调用 Agent", r.text[:80])
        r = ask(alice, f"Should {own_acct} be reported?")
        check(code_of(r) == "reg_case_data" and len(calls) == n_calls,
              "问题里有本案的账户（无论格式）→ 422", own_acct)
        r = ask(alice, "Under mas_n626:14.1, is a report due by 10:30 the next day?")
        check(r.status_code == 200 and len(calls) == n_calls + 1,
              "条款编号、时刻不会被当成账户号")
        r = ask(alice, "STR")
        check(code_of(r) == "validation", "问题太短 → 422")
        n_calls = len(calls)
        r = ask(alice, "What is an STR?", case=999999999)
        check(r.status_code == 404 and len(calls) == n_calls, "不存在的案件 → 404，没有调用 Agent")

        left = alice.get(f"/api/cases/{cid}/regulation").json()["quota_left"]
        for _ in range(left):
            ask(alice, "What records must be kept for an STR?")
        n_calls = len(calls)
        r = ask(alice, "What records must be kept for an STR?")
        check(r.status_code == 429 and code_of(r) == "reg_quota" and r.headers.get("retry-after")
              and len(calls) == n_calls, "这一小时次数用完 → 429，带 Retry-After，没有调用 Agent",
              f"用完 {left} 次后")
        r = ask(bob, "What records must be kept for an STR?")
        check(r.status_code == 200, "次数按人计：bob 不受 alice 的影响")

        held = [review_api.REG_SLOTS.acquire(blocking=False) for _ in range(review_api.REG_CONCURRENCY)]
        n_calls = len(calls)
        r = ask(bob, "What is tipping-off?")
        check(all(held) and r.status_code == 503 and code_of(r) == "reg_busy" and r.headers.get("retry-after")
              and len(calls) == n_calls, f"已有 {review_api.REG_CONCURRENCY} 个在跑 → 503 繁忙，没有调用 Agent")
        for _ in held:
            review_api.REG_SLOTS.release()
        r = bob.get("/queue")
        check(r.status_code == 200, "名额用满时，普通页面照常可用（名额只限制法规查询）")

        mode["kind"] = "error"
        n0 = n_queries()
        review_api.log.disabled = True                     # 接口会把异常写进服务器日志；测试里不打印这段预期内的报错
        r = ask(bob, "What is tipping-off?")
        review_api.log.disabled = False
        mode["kind"] = "answered"
        check(r.status_code == 502 and code_of(r) == "reg_failed" and n_queries() == n0
              and review_api.REG_SLOTS._value == review_api.REG_CONCURRENCY,
              "Agent 出错 → 502，名额还回去，不留记录")

        # ---------------- [3] 数据库规则 ----------------
        print("[3] 数据库规则（绕过接口直接写库，复核角色）")
        cur.execute("SELECT answer_id FROM review_reg_queries WHERE case_id = %s ORDER BY query_id LIMIT 1", (cid,))
        used = cur.fetchone()[0]

        def rejected(sql, args=()):
            try:
                with conn.transaction():
                    cur.execute(sql, args)
                return ""
            except psycopg.Error as e:
                return (e.diag.message_primary or str(e)) if getattr(e, "diag", None) else str(e)

        m = rejected("INSERT INTO review_reg_queries (case_id, actor, answer_id) VALUES (%s, 'bob', %s)", (cid, used))
        check("unique" in m, "同一条回答再挂一次 → 拒绝（不能把别人的回答挂到自己名下）", m[:60])
        m = rejected("INSERT INTO review_reg_queries (case_id, actor, answer_id) VALUES (%s, 'bob', %s)",
                     (cid, old_answer))
        check(m.startswith("reg_answer"), "挂一小时前的旧回答 → 拒绝", m[:70])
        m = rejected("INSERT INTO review_reg_queries (case_id, actor, answer_id) VALUES (%s, 'system', %s)",
                     (cid, old_answer))
        check(m.startswith("reg_role"), "流水线账号 system 不能记录提问", m[:60])
        m = rejected("UPDATE review_reg_queries SET actor = 'bob' WHERE case_id = %s", (cid,))
        m2 = rejected("DELETE FROM review_reg_queries WHERE case_id = %s", (cid,))
        check("permission denied" in m and "permission denied" in m2, "复核角色不能修改、删除提问记录")
        m = rejected("UPDATE kb_answer_log SET question = 'x' WHERE answer_id = %s", (used,))
        check("permission denied" in m, "复核角色不能改问答日志（日志由 Agent 写）")

        review_api.app.dependency_overrides.clear()

    conn.close()
    n_ok = sum(ok for ok, _ in results)
    print(f"\n{n_ok}/{len(results)} 项通过；测试数据已全部回滚，没有调用模型")
    if n_ok != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()