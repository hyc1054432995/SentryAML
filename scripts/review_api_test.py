"""
SentryAML - R11a：复核接口的端到端测试
=====================================
用 FastAPI 自带的测试客户端，像浏览器一样发请求（带 cookie、带请求头），检查从 HTTP 到数据库触发器的整条路径。

为什么全部放在一个事务里、结束时回滚：复核事件、认领、会话都只能追加，测试数据一旦提交就永远留在审计记录里。
做法：测试持有一个连接，开一个"强制回滚"的外层事务；接口的数据库依赖（get_db）被替换成这个连接，
接口里每个请求的 conn.transaction() 于是变成保存点——被拒绝时回到保存点，成功时并入外层事务，最后整体撤销。
接口执行时用 SET LOCAL ROLE sentry_review，检查的是复核员角色实际拥有的权限。

租约、会话的到期用 clock_timestamp() 判断（027），所以在同一个事务里把时长调成 1～2 秒、等一下，就能测到期。

用法：python scripts/review_api_test.py
"""

from __future__ import annotations

import os
import secrets
import time
from datetime import datetime, timedelta, timezone

import psycopg

# 测试用自己的随机密钥，不依赖 .env 里的值（db.py 加载 .env 时不会覆盖已有的环境变量）
os.environ["SESSION_SECRET"] = secrets.token_urlsafe(48)

from fastapi.testclient import TestClient  # noqa: E402

import review_api  # noqa: E402
import review_auth as auth  # noqa: E402
from db import get_dsn  # noqa: E402
from review import act  # noqa: E402

results: list[tuple[bool, str]] = []


def check(ok: bool, what: str, detail: str = "") -> None:
    results.append((bool(ok), what))
    print(f"  {'✓' if ok else '✗'} {what}" + (f"   {detail}" if detail else ""))


def code_of(r) -> str:
    d = r.json().get("detail")
    return d.get("code", "") if isinstance(d, dict) else ("validation" if isinstance(d, list) else "")


def detail_of(r, key: str) -> str:
    """被拒绝时的 message / hint；请求意外成功时返回空串，让 check 报 ✗ 而不是程序崩溃。"""
    d = r.json().get("detail")
    return str(d.get(key, "")) if isinstance(d, dict) else ""


def main() -> None:
    conn = psycopg.connect(get_dsn(), autocommit=True)
    with conn.transaction(force_rollback=True), conn.cursor() as cur:
        # ---------------- 准备（库的所有者身份） ----------------
        cur.execute("""SELECT a.alert_id FROM alerts a WHERE NOT EXISTS
                       (SELECT 1 FROM review_cases c WHERE c.alert_id = a.alert_id) ORDER BY a.alert_id LIMIT 4""")
        ids = [r[0] for r in cur.fetchall()]
        if len(ids) < 4:
            raise SystemExit("需要至少 4 条还没有案件的告警")
        cur.execute("SELECT typology_id FROM kb_typologies ORDER BY typology_id LIMIT 1")
        typ = cur.fetchone()[0]
        snap = {"status": "done", "decision": "escalate", "typology": typ, "confidence": "medium",
                "rationale": "TEST-RATIONALE", "facts": [{"step": 0, "path": "amount_usd", "value": 1,
                                                           "verified": True}]}

        def new_case(aid, route, decision, result):
            cur.execute("""INSERT INTO review_cases (alert_id, route, agent_decision, agent_result)
                           VALUES (%s, %s, %s, %s) RETURNING case_id""",
                        (aid, route, decision, psycopg.types.json.Jsonb(result)))
            cid = cur.fetchone()[0]
            act(cur, cid, "open", "system")
            return cid

        c_esc = new_case(ids[0], "agent_escalate", "escalate", snap)
        c_qa = new_case(ids[1], "qa_sample", "close", {**snap, "decision": "close"})
        c_lease = new_case(ids[2], "agent_escalate", "escalate", snap)
        c_exp = new_case(ids[3], "agent_escalate", "escalate", snap)
        pw = {r: secrets.token_urlsafe(18) for r in ("alice", "bob", "carol", "dave")}
        for r, p in pw.items():
            auth.set_password(cur, r, p)
        cur.execute("SET LOCAL ROLE sentry_review")

        def as_owner(sql, params=()):
            cur.execute("RESET ROLE")
            cur.execute(sql, params)
            cur.execute("SET LOCAL ROLE sentry_review")

        def shared_conn():
            yield conn
        review_api.app.dependency_overrides[review_api.get_db] = shared_conn

        def client():
            return TestClient(review_api.app, base_url="https://testserver")

        def login(c, who, password=None):
            r = c.post("/api/login", json={"reviewer_id": who, "password": password or pw[who]})
            if r.status_code == 200:
                c.headers[auth.CSRF_HEADER] = r.json()["csrf"]
            return r

        def detail(c, cid):
            return c.get(f"/api/cases/{cid}").json()

        def n_events(cid):
            cur.execute("SELECT count(*) FROM review_events WHERE case_id = %s", (cid,))
            return cur.fetchone()[0]

        def expect_db_error(code, fn, what):
            cur.execute("SAVEPOINT x")
            try:
                fn()
            except psycopg.Error as e:
                cur.execute("ROLLBACK TO SAVEPOINT x")
                msg = e.diag.message_primary or str(e)
                check(code in msg, what, msg[:100])
                return
            cur.execute("ROLLBACK TO SAVEPOINT x")
            check(False, what, "没有被拒绝")

        alice, bob, carol, dave, anon = client(), client(), client(), client(), client()

        # ---------------- [1] 登录与会话 ----------------
        print("[1] 登录与会话")
        r = anon.get("/api/cases")
        check(r.status_code == 401, "没登录：读队列 → 401")
        r1 = anon.post("/api/login", json={"reviewer_id": "alice", "password": "wrong-password-123"})
        r2 = anon.post("/api/login", json={"reviewer_id": "nobody", "password": "wrong-password-123"})
        check(r1.status_code == r2.status_code == 401 and r1.json() == r2.json(),
              "密码错与账号不存在：同样的 401、同样的提示（不能用来试探账号）")
        r = login(alice, "alice")
        sc = r.headers.get("set-cookie", "")
        check(r.status_code == 200 and "HttpOnly" in sc and "SameSite=strict" in sc and "Secure" in sc,
              "alice 登录成功；cookie 带 HttpOnly、SameSite=Strict、Secure", sc.split(";")[0][:20] + "…")
        token = alice.cookies.get(auth.COOKIE)
        cur.execute("SELECT token_sha256 FROM review_sessions WHERE reviewer_id = 'alice'")
        stored = [x[0] for x in cur.fetchall()]
        check(auth.token_hash(token) in stored and token not in stored,
              "数据库只存令牌的 sha256，不存令牌本身")
        r = alice.get("/api/me")
        check(r.status_code == 200 and r.json()["role"] == "analyst", "/api/me：身份来自会话", r.json().get("role"))
        r = alice.post(f"/api/cases/{c_lease}/claim", headers={auth.CSRF_HEADER: ""})
        r2 = alice.post(f"/api/cases/{c_lease}/claim", headers={auth.CSRF_HEADER: "0" * 64})
        check(r.status_code == r2.status_code == 403 and code_of(r) == "csrf",
              "写操作没带或带错 CSRF 令牌 → 403（只有 cookie 不够）")
        for _ in range(auth.MAX_FAILS):
            dave.post("/api/login", json={"reviewer_id": "dave", "password": "not-the-password"})
        r = login(dave, "dave")
        check(r.status_code == 429 and r.headers.get("retry-after"),
              f"连续输错 {auth.MAX_FAILS} 次后，正确密码也暂停登录 → 429", f"Retry-After={r.headers.get('retry-after')}")
        r = anon.post("/api/login", json={"reviewer_id": "system", "password": "anything-at-all"})
        check(r.status_code == 401, "流水线账号 system 不能登录")
        expect_db_error("session_role", lambda: cur.execute(
            "INSERT INTO review_sessions (token_sha256, reviewer_id, expires_at) VALUES (%s, 'system', now())",
            ("a" * 64,)), "绕过接口直接给 system 建会话 → 数据库拒绝")
        saved = os.environ.pop("SESSION_SECRET")
        try:
            auth.session_secret()
            check(False, "没有 SESSION_SECRET 时拒绝启动")
        except RuntimeError:
            check(True, "没有 SESSION_SECRET 时拒绝启动（不用默认密钥）")
        os.environ["SESSION_SECRET"] = saved

        # ---------------- [2] 身份不能伪造 ----------------
        print("[2] 身份不能伪造")
        login(bob, "bob")
        login(carol, "carol")
        before = n_events(c_esc)
        r = alice.post(f"/api/cases/{c_esc}/actions", json={"action": "request_info", "seen_state": "pending_review",
                                                            "reason": "check payee", "actor": "carol"})
        check(r.status_code == 422 and n_events(c_esc) == before,
              "请求里写 actor=carol → 422，没有写入任何记录", code_of(r))
        r = alice.post(f"/api/cases/{c_esc}/actions",
                       json={"action": "close", "seen_state": "pending_review", "reason": "x", "typology": typ})
        check(r.status_code == 422, "字段与动作不配套（关闭时给手法）→ 422")
        d = detail(alice, c_esc)
        keys = list(d)
        check(keys.index("evidence") < keys.index("agent") and d["evidence"][0]["tool"] == "alert",
              "案件详情：证据排在 Agent 结论之前", " → ".join(keys[:5]))

        # ---------------- [3] 数据库规则经由接口 ----------------
        print("[3] 数据库规则经由接口")
        r = alice.post(f"/api/cases/{c_esc}/actions", json={"action": "close", "seen_state": "pending_review"})
        check(r.status_code == 422 and code_of(r) == "review_reason", "推翻 Agent 的升级、不写理由 → 422 review_reason")
        r = alice.post(f"/api/cases/{c_esc}/actions",
                       json={"action": "confirm_suspicious", "seen_state": "pending_review"})
        check(r.status_code == 422 and code_of(r) == "str_typology", "确认可疑不选手法 → 422 str_typology")
        r = alice.post(f"/api/cases/{c_esc}/actions",
                       json={"action": "confirm_suspicious", "seen_state": "pending_review", "typology": typ})
        cur.execute("SELECT actor, payload ->> 'typology' FROM review_events WHERE case_id = %s ORDER BY seq DESC "
                    "LIMIT 1", (c_esc,))
        check(r.status_code == 200 and r.json()["state"] == "suspicious_confirmed" and cur.fetchone() == ("alice", typ),
              "alice 确认可疑：事件里的 actor 是会话里的 alice")
        r = carol.post(f"/api/cases/{c_esc}/actions",
                       json={"action": "approve_str", "seen_state": "suspicious_confirmed", "draft_version": 1})
        check(r.status_code == 409 and code_of(r) == "review_illegal", "还没有草稿就批准 → 409 review_illegal")
        page_state = detail(alice, c_qa)["case"]["state"]          # alice 打开页面
        r = bob.post(f"/api/cases/{c_qa}/actions", json={"action": "close", "seen_state": "pending_review"})
        check(r.status_code == 200, "bob 关闭抽查案件（与 Agent 一致，不需要理由）")
        r = alice.post(f"/api/cases/{c_qa}/actions", json={"action": "confirm_suspicious", "seen_state": page_state,
                                                           "typology": typ, "reason": "fan-in from new payers"})
        check(r.status_code == 409 and code_of(r) == "review_stale" and detail_of(r, "hint"),
              "alice 按旧页面提交 → 409 review_stale，提示刷新", detail_of(r, "hint")[:30])
        r = bob.post(f"/api/cases/{c_qa}/actions", json={"action": "reopen", "seen_state": "closed", "reason": "x"})
        check(r.status_code == 403 and code_of(r) == "review_role", "分析员重开 → 403 review_role")
        r = alice.get("/api/cases/999999999")
        check(r.status_code == 404, "不存在的案件 → 404")
        r = alice.get("/api/cases?state=1")
        check(r.status_code == 422, "队列的状态写错（state=1）→ 422，而不是返回空队列")

        # ---------------- [4] 认领与租约 ----------------
        print("[4] 认领与租约")
        r = alice.post(f"/api/cases/{c_lease}/claim")
        until = datetime.fromisoformat(r.json()["expires_at"])
        lease = until - datetime.now(timezone.utc)
        check(r.status_code == 200 and timedelta(minutes=29) < lease <= timedelta(minutes=30),
              "alice 认领：到期时间由数据库计算（30 分钟）", f"剩 {lease.total_seconds() / 60:.1f} 分钟")
        r = alice.post(f"/api/cases/{c_lease}/claim", json={"expires_at": "2099-01-01T00:00:00Z"})
        check(r.status_code == 422, "客户端想自己定到期时间 → 422")
        r = bob.post(f"/api/cases/{c_lease}/actions",
                     json={"action": "request_info", "seen_state": "pending_review", "reason": "x"})
        check(r.status_code == 409 and code_of(r) == "review_claimed", "bob 操作 alice 认领的案件 → 409 review_claimed",
              detail_of(r, "message")[:60])
        check(detail(bob, c_lease)["allowed_actions"] == [] and detail(alice, c_lease)["allowed_actions"],
              "bob 的页面上没有可用按钮，alice 的有")
        r = bob.post(f"/api/cases/{c_lease}/claim")
        r2 = bob.post(f"/api/cases/{c_lease}/release")
        check(code_of(r) == code_of(r2) == "review_claimed", "bob 不能抢认领、不能替 alice 释放")
        q_bob = {c["case_id"]: c for c in bob.get("/api/cases").json()["cases"]}
        q_mine = [c["case_id"] for c in alice.get("/api/cases?mine=true").json()["cases"]]
        check(q_bob[c_lease]["holder"] == "alice" and c_lease in q_mine and c_esc not in q_mine,
              "队列显示认领人；alice 的\"我的\"只有她认领的")
        check("agent_decision" not in q_bob[c_lease], "队列里不显示 Agent 结论（先看证据）")
        expect_db_error("review_claimed", lambda: act(cur, c_lease, "request_info", "bob", "via CLI"),
                        "绕过接口（命令行 / 直接写库）同样被数据库拒绝")
        r = carol.post(f"/api/cases/{c_lease}/actions",
                       json={"action": "request_info", "seen_state": "pending_review", "reason": "need KYC file"})
        check(r.status_code == 200, "MLRO 不受认领限制（批准、退回要能进行）")
        r = carol.post(f"/api/cases/{c_lease}/claim")
        check(code_of(r) == "review_reason", "MLRO 接管认领不写理由 → 422")
        r = carol.post(f"/api/cases/{c_lease}/claim", json={"reason": "alice is out today"})
        check(r.status_code == 200 and r.json()["took_over"] == "alice", "MLRO 写明理由接管，记下接管了 alice")
        r = alice.post(f"/api/cases/{c_lease}/actions",
                       json={"action": "request_info", "seen_state": "pending_review", "reason": "x"})
        check(code_of(r) == "review_claimed", "接管之后 alice 也被挡住")
        carol.post(f"/api/cases/{c_lease}/release")
        r1 = alice.post(f"/api/cases/{c_lease}/claim")
        time.sleep(0.05)
        r2 = alice.post(f"/api/cases/{c_lease}/claim")
        check(r1.status_code == r2.status_code == 200 and r2.json()["expires_at"] > r1.json()["expires_at"],
              "释放后 alice 重新认领；再认领一次 = 续期")
        r1 = alice.post(f"/api/cases/{c_lease}/release")
        r2 = alice.post(f"/api/cases/{c_lease}/release")
        check(r1.status_code == 200 and code_of(r2) == "claim_none", "释放；没有认领时再释放 → 409 claim_none")
        r = alice.post(f"/api/cases/{c_qa}/claim")
        check(code_of(r) == "claim_state", "已关闭的案件不能认领")
        expect_db_error("permission denied", lambda: cur.execute("UPDATE review_claims SET expires_at = now()"),
                        "复核角色不能修改认领记录")
        cur.execute("""INSERT INTO review_claims (case_id, actor, kind, created_at, expires_at)
                       VALUES (%s, 'bob', 'claim', '2000-01-01', '2099-01-01') RETURNING created_at, expires_at""",
                    (c_lease,))
        at, exp = cur.fetchone()
        check(exp.year != 2099 and (exp - at) == timedelta(seconds=1800), "直接写库伪造时间：被数据库重新计算")
        cur.execute("INSERT INTO review_claims (case_id, actor, kind) VALUES (%s, 'bob', 'release')", (c_lease,))
        expect_db_error("permission denied",
                        lambda: cur.execute("UPDATE review_sessions SET reviewer_id = 'carol' WHERE reviewer_id = 'alice'"),
                        "复核角色不能把会话改成别人的（只能改 revoked_at）")
        expect_db_error("permission denied", lambda: cur.execute("UPDATE reviewer_credentials SET password_hash = 'x'"),
                        "复核角色不能改密码哈希")

        # ---------------- [5] 到期、注销、停用 ----------------
        print("[5] 到期、注销、停用")
        as_owner("UPDATE review_config SET value = 2 WHERE key = 'lease_seconds'")
        as_owner("UPDATE review_config SET value = 2 WHERE key = 'session_seconds'")
        alice.post(f"/api/cases/{c_exp}/claim")
        short = client()
        login(short, "alice")
        short_token = short.cookies.get(auth.COOKIE)
        r = bob.post(f"/api/cases/{c_exp}/actions",
                     json={"action": "request_info", "seen_state": "pending_review", "reason": "x"})
        check(code_of(r) == "review_claimed", "租约内 bob 被挡住")
        time.sleep(2.2)
        r = bob.post(f"/api/cases/{c_exp}/actions",
                     json={"action": "request_info", "seen_state": "pending_review", "reason": "x"})
        check(r.status_code == 200, "租约到期后 bob 可以操作（不需要任何人解锁）")
        r = short.get("/api/me", headers={"Cookie": f"{auth.COOKIE}={short_token}"})
        check(r.status_code == 401, "会话到期 → 401（服务器端判断，不依赖浏览器删 cookie）")
        r = alice.post("/api/logout")
        r2 = anon.get("/api/me", headers={"Cookie": f"{auth.COOKIE}={token}"})
        check(r.status_code == 200 and r2.status_code == 401, "注销后旧令牌立刻失效")
        as_owner("UPDATE reviewers SET active = false WHERE reviewer_id = 'bob'")
        r = bob.get("/api/me")
        check(r.status_code == 401, "停用复核员 → 他已登录的会话立刻失效")
        review_api.app.dependency_overrides.clear()

    conn.close()
    n_ok = sum(ok for ok, _ in results)
    print(f"\n{n_ok}/{len(results)} 项通过；测试数据已全部回滚")
    if n_ok != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()