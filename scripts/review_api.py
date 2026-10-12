"""
SentryAML - R11a：复核接口（FastAPI）
=====================================
把命令行的复核流程变成多人使用的网页接口。这一层不重新实现任何规则——状态机、角色、理由、双人复核、防过期、
租约全部由数据库强制（025、026、027）。接口只做三件事：
  1. 认出请求是谁发的：身份只来自登录会话（review_auth.py），请求内容里不接受 actor 字段
  2. 调用现有的函数（review.act 等），每个请求一个数据库事务：成功提交，被拒绝整体回滚
  3. 把数据库的拒绝翻译成 HTTP 状态码和人能看懂的提示

接口：
  POST /api/login                     用户名 + 密码 → 设置会话 cookie，返回 CSRF 令牌
  POST /api/logout
  GET  /api/me
  GET  /api/cases?state=&mine=        队列（含认领人）
  GET  /api/cases/{id}                案件详情：先证据、后 Agent 结论；复核记录、草稿版本、认领状态
  POST /api/cases/{id}/claim          认领 / 续期（MLRO 写理由可接管）
  POST /api/cases/{id}/release        释放
  POST /api/cases/{id}/actions        复核动作：必须带 seen_state（页面加载时看到的状态）
  GET  /api/typologies                确认可疑时可选的手法（知识图谱）
  GET  /api/docs                      自动生成的接口文档

启动（只监听本机）：
    python scripts/review_api.py            # http://127.0.0.1:8000/api/docs
测试：
    python scripts/review_api_test.py       # 端到端测试，全部在一个事务里，结束时回滚
"""

from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from typing import Iterator, Literal

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

import review_auth as auth
from db import connect_review
from review import act, explain

log = logging.getLogger("sentry.review_api")

SECRET = auth.session_secret()                                   # 没有密钥就拒绝启动
COOKIE_SECURE = os.getenv("REVIEW_COOKIE_SECURE", "1") != "0"    # 浏览器把 http://localhost 也当作安全来源，本机开发不用关
THROTTLE = auth.Throttle()

# 数据库拒绝的原因 → HTTP 状态码
#   409 Conflict：请求本身没错，但和案件当前的状态冲突（别人刚改过、别人认领了、这个状态下没有这个动作）→ 刷新后重新决定
#   403 Forbidden：你这个人不能做（角色、双人复核、不是你写的版本）
#   422 Unprocessable：内容不完整或不合格（缺理由、缺手法、草稿版本不对、草稿没通过检查）
STATUS = {"review_stale": 409, "review_claimed": 409, "review_illegal": 409, "claim_state": 409, "claim_none": 409,
          "review_role": 403, "review_four_eyes": 403, "review_actor": 403, "str_author": 403, "claim_role": 403,
          "session_role": 403,
          "review_reason": 422, "str_typology": 422, "str_version": 422, "str_no_draft": 422, "str_narrative": 422,
          "str_invalid": 422, "str_state": 422}

app = FastAPI(title="SentryAML review API", version="r11a", docs_url="/api/docs", openapi_url="/api/openapi.json",
              redoc_url=None)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")    # 浏览器不要猜文件类型
    resp.headers.setdefault("X-Frame-Options", "DENY")              # 不能被别的网站嵌进框架里诱导点击
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.url.path.startswith("/api/") and not request.url.path.startswith("/api/docs"):
        resp.headers.setdefault("Cache-Control", "no-store")        # 案件数据不留在浏览器缓存里
    return resp


# ---------------------------------------------------------------------------
# 数据库连接与错误翻译
# ---------------------------------------------------------------------------
def get_db() -> Iterator[psycopg.Connection]:
    """每个请求一个连接（复核员的最小权限角色）。autocommit + conn.transaction()：事务边界由下面的 db_call 明确划定。
    测试时这个依赖被替换成一个共享连接，conn.transaction() 自动变成保存点，整个测试结束时回滚。"""
    conn = connect_review(autocommit=True, options="-c statement_timeout=15000")
    try:
        yield conn
    finally:
        conn.close()


def db_call(conn: psycopg.Connection, fn):
    """在一个事务里执行 fn(cur)。被数据库规则拒绝时整体回滚，并翻译成 HTTP 错误。"""
    try:
        with conn.transaction(), conn.cursor() as cur:
            return fn(cur)
    except psycopg.errors.QueryCanceled:
        raise HTTPException(503, {"code": "timeout", "message": "the query took too long"})
    except psycopg.Error as e:
        msg = (e.diag.message_primary or str(e)) if getattr(e, "diag", None) else str(e)
        code = msg.split(":", 1)[0]
        if code in STATUS:
            raise HTTPException(STATUS[code], {"code": code, "message": msg, "hint": explain(e)})
        # 其他数据库错误（程序错误、权限配置错误）：细节只写日志，不返回给浏览器
        log.exception("database error")
        raise HTTPException(500, {"code": "internal", "message": "database error; see server log"})


def err(status: int, code: str, message: str) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message})


# ---------------------------------------------------------------------------
# 身份
# ---------------------------------------------------------------------------
def me(request: Request, conn: psycopg.Connection = Depends(get_db)) -> auth.Reviewer:
    token = request.cookies.get(auth.COOKIE)
    if not token or len(token) > 128:
        raise err(401, "login_required", "not logged in")
    who = db_call(conn, lambda cur: auth.lookup_session(cur, token))
    if who is None:
        raise err(401, "login_required", "session expired, revoked or unknown; log in again")
    return who


def writer(who: auth.Reviewer = Depends(me),
           csrf: str | None = Header(None, alias=auth.CSRF_HEADER, max_length=128,
                                     description="登录或 /api/me 返回的 csrf 值")) -> auth.Reviewer:
    """会改变数据的请求：除了会话 cookie，还要带 CSRF 令牌（review_auth.py 顶部的说明）。
    声明成请求头参数，接口文档页（/api/docs）上就有一个输入框，可以直接在浏览器里试。"""
    if not auth.csrf_ok(csrf, who.token, SECRET):
        raise err(403, "csrf", f"missing or wrong {auth.CSRF_HEADER} header")
    return who


class LoginIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reviewer_id: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=1024)


@app.post("/api/login")
def login(body: LoginIn, request: Request, response: Response, conn: psycopg.Connection = Depends(get_db)):
    key = body.reviewer_id.strip().lower()
    wait = THROTTLE.blocked_for(key)
    if wait:
        raise HTTPException(429, {"code": "throttled", "message": f"too many failed logins; retry in {wait}s"},
                            headers={"Retry-After": str(wait)})
    who = db_call(conn, lambda cur: auth.verify_login(cur, key, body.password))
    if who is None:
        THROTTLE.fail(key)
        # 不区分"没有这个账号"和"密码错"，免得被用来试探哪些账号存在
        raise err(401, "login_failed", "wrong reviewer id or password")
    THROTTLE.reset(key)
    old = request.cookies.get(auth.COOKIE)

    def start(cur):
        if old and len(old) <= 128:
            auth.revoke_session(cur, old)          # 同一浏览器重新登录：旧会话作废，不留两个有效令牌
        return auth.create_session(cur, who.reviewer_id)

    token, expires_at = db_call(conn, start)
    response.set_cookie(auth.COOKIE, token, httponly=True, secure=COOKIE_SECURE, samesite="strict", path="/",
                        max_age=max(0, int((expires_at - datetime.now(timezone.utc)).total_seconds())))
    return {"reviewer_id": who.reviewer_id, "name": who.name, "role": who.role, "expires_at": expires_at,
            "csrf": auth.csrf_for(token, SECRET)}


@app.post("/api/logout")
def logout(response: Response, who: auth.Reviewer = Depends(writer), conn: psycopg.Connection = Depends(get_db)):
    db_call(conn, lambda cur: auth.revoke_session(cur, who.token))
    response.delete_cookie(auth.COOKIE, path="/", secure=COOKIE_SECURE, httponly=True, samesite="strict")
    return {"ok": True}


@app.get("/api/me")
def whoami(who: auth.Reviewer = Depends(me)):
    return {"reviewer_id": who.reviewer_id, "name": who.name, "role": who.role, "expires_at": who.expires_at,
            "csrf": auth.csrf_for(who.token, SECRET)}


@app.get("/healthz")
def healthz():
    return {"ok": True}


# ---------------------------------------------------------------------------
# 案件
# ---------------------------------------------------------------------------
def _require_case(cur, case_id: int) -> None:
    cur.execute("SELECT 1 FROM review_cases WHERE case_id = %s", (case_id,))
    if cur.fetchone() is None:
        raise err(404, "no_case", f"no case {case_id}")


CaseState = Literal["pending_review", "suspicious_confirmed", "str_drafted", "str_approved", "closed"]


@app.get("/api/cases")
def queue(state: CaseState | None = Query(None, description="只看某个状态的案件；不填 = 全部"),
          mine: bool = Query(False, description="true = 只看我认领中的案件"),
          who: auth.Reviewer = Depends(me), conn: psycopg.Connection = Depends(get_db)):
    # state 写成固定的几个取值：写错（例如 state=1）直接 422，而不是悄悄返回一个空队列，让人以为没有案件
    def q(cur):
        # 队列里不列 Agent 的结论：先看证据再看结论（自动化偏差，D78）；来源（route）本身是分流规则，不是结论
        cur.execute("""SELECT s.case_id, s.alert_id, a.tier, s.route, s.state, s.last_seq, s.last_actor, s.last_at,
                              cl.holder, cl.expires_at
                       FROM review_case_state s JOIN alerts a USING (alert_id)
                       LEFT JOIN review_claim_state cl ON cl.case_id = s.case_id
                       WHERE (%(state)s::text IS NULL OR s.state = %(state)s)
                         AND (NOT %(mine)s OR cl.holder = %(me)s)
                       ORDER BY s.case_id""", {"state": state, "mine": mine, "me": who.reviewer_id})
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    return {"cases": db_call(conn, q)}


def case_detail(cur, case_id: int, who: auth.Reviewer) -> dict:
    cur.execute("""SELECT c.case_id, c.alert_id, c.route, c.agent_result, c.agent_run_id, s.state, s.last_seq,
                          a.txn_id, a.txn_ts, a.from_bank, a.from_account, a.to_bank, a.to_account, a.amount_usd,
                          a.tier, a.score_pct, cl.holder, cl.expires_at
                   FROM review_cases c JOIN review_case_state s ON s.case_id = c.case_id
                   JOIN alerts a ON a.alert_id = c.alert_id
                   LEFT JOIN review_claim_state cl ON cl.case_id = c.case_id
                   WHERE c.case_id = %s""", (case_id,))
    row = cur.fetchone()
    if row is None:
        raise err(404, "no_case", f"no case {case_id}")
    (cid, aid, route, result, run_id, state, last_seq, txn_id, ts, fb, fa, tb, ta, amt, tier, pct,
     holder, until) = row
    result = result or {}
    tools = {}
    if run_id:
        cur.execute("SELECT step, tool FROM agent_tool_calls WHERE run_id = %s", (run_id,))
        tools = dict(cur.fetchall())
    evidence = [{"step": f.get("step"), "tool": "alert" if f.get("step") == 0 else tools.get(f.get("step")),
                 "path": f.get("path"), "value": f.get("value"), "verified": f.get("verified")}
                for f in result.get("facts") or []]
    if route == "tier1":
        agent = None
        agent_note = "tier1: routed to human review without an agent decision (plan B)"
    elif result.get("status") != "done":
        agent, agent_note = None, f"agent error, no decision: {result.get('error')}"
    else:
        agent = {k: result.get(k) for k in ("decision", "typology", "confidence", "rationale", "prompt_version",
                                            "model")}
        agent_note = None
    cur.execute("""SELECT seq, actor, action, from_state, to_state, overrides_agent, reason, payload, created_at
                   FROM review_events WHERE case_id = %s ORDER BY seq""", (cid,))
    cols = [d.name for d in cur.description]
    events = [dict(zip(cols, r)) for r in cur.fetchall()]
    cur.execute("""SELECT version, created_by, generator ->> 'kind' AS kind, created_at FROM str_drafts
                   WHERE case_id = %s ORDER BY version""", (cid,))
    cols = [d.name for d in cur.description]
    drafts = [dict(zip(cols, r)) for r in cur.fetchall()]
    # 能做哪些动作：只是给界面的提示（哪些按钮亮），真正的判断仍在数据库。
    # draft_str / edit_str 和草稿写入绑定在同一个事务里，走草稿接口（R11b），不在这里
    cur.execute("""SELECT action FROM review_transitions
                   WHERE from_state = %s AND %s = ANY (roles) AND action NOT IN ('open', 'draft_str', 'edit_str')
                   ORDER BY action""", (state, who.role))
    allowed = [r[0] for r in cur.fetchall()]
    blocked_by_claim = holder is not None and holder != who.reviewer_id and who.role != "mlro"
    # 字段顺序就是阅读顺序：告警 → 证据 → Agent 没查的 → Agent 的结论（D78：先判断证据，再看结论）
    return {
        "case": {"case_id": cid, "alert_id": aid, "route": route, "state": state, "last_seq": last_seq},
        "alert": {"txn_id": txn_id, "time": ts, "from": f"{fb}:{fa}", "to": f"{tb}:{ta}", "amount_usd": amt,
                  "tier": tier, "score_pct": pct},
        "evidence": evidence,
        "not_checked": result.get("not_checked"),
        "agent": agent,
        "agent_note": agent_note,
        "events": events,
        "drafts": drafts,
        "claim": {"holder": holder, "expires_at": until, "mine": holder == who.reviewer_id},
        "allowed_actions": [] if blocked_by_claim else allowed,
    }


@app.get("/api/cases/{case_id}")
def get_case(case_id: int, who: auth.Reviewer = Depends(me), conn: psycopg.Connection = Depends(get_db)):
    return db_call(conn, lambda cur: case_detail(cur, case_id, who))


class ClaimIn(BaseModel):
    model_config = ConfigDict(extra="forbid")          # 到期时间、认领人都不由客户端决定
    reason: str | None = Field(None, max_length=2000)


def _claim(conn, case_id: int, who: auth.Reviewer, kind: str, reason: str | None) -> dict:
    def go(cur):
        _require_case(cur, case_id)
        cur.execute("""INSERT INTO review_claims (case_id, actor, kind, reason) VALUES (%s, %s, %s, %s)
                       RETURNING kind, created_at, expires_at, took_over""", (case_id, who.reviewer_id, kind, reason))
        k, at, until, took = cur.fetchone()
        return {"case_id": case_id, "kind": k, "at": at, "expires_at": until, "took_over": took}
    return db_call(conn, go)


@app.post("/api/cases/{case_id}/claim")
def claim(case_id: int, body: ClaimIn | None = None, who: auth.Reviewer = Depends(writer),
          conn: psycopg.Connection = Depends(get_db)):
    return _claim(conn, case_id, who, "claim", body.reason if body else None)


@app.post("/api/cases/{case_id}/release")
def release(case_id: int, body: ClaimIn | None = None, who: auth.Reviewer = Depends(writer),
            conn: psycopg.Connection = Depends(get_db)):
    return _claim(conn, case_id, who, "release", body.reason if body else None)


class ActionIn(BaseModel):
    """没有 actor 字段，而且 extra="forbid"：请求里多写一个 actor（或任何别的字段）直接 422，不会被悄悄忽略。"""
    model_config = ConfigDict(extra="forbid")
    action: Literal["close", "confirm_suspicious", "request_info", "return_draft", "approve_str", "reopen"]
    seen_state: str = Field(min_length=1, max_length=40)     # 页面加载时看到的状态（防过期，D78）
    reason: str | None = Field(None, max_length=4000)
    typology: str | None = Field(None, max_length=80)        # 只用于 confirm_suspicious（D79）
    draft_version: int | None = Field(None, ge=1)            # 只用于 approve_str / return_draft：你审阅的那一版

    @model_validator(mode="after")
    def _fields_fit_action(self):
        if self.typology is not None and self.action != "confirm_suspicious":
            raise ValueError("typology is only used with confirm_suspicious")
        if self.draft_version is not None and self.action not in ("approve_str", "return_draft"):
            raise ValueError("draft_version is only used with approve_str or return_draft")
        return self


@app.post("/api/cases/{case_id}/actions")
def do_action(case_id: int, body: ActionIn, who: auth.Reviewer = Depends(writer),
              conn: psycopg.Connection = Depends(get_db)):
    payload = {k: v for k, v in (("typology", body.typology), ("draft_version", body.draft_version))
               if v is not None}

    def go(cur):
        _require_case(cur, case_id)
        seq, to_state = act(cur, case_id, body.action, who.reviewer_id, body.reason, body.seen_state,
                            payload or None)
        return {"case_id": case_id, "seq": seq, "state": to_state, "actor": who.reviewer_id}
    return db_call(conn, go)


@app.get("/api/typologies")
def typologies(who: auth.Reviewer = Depends(me), conn: psycopg.Connection = Depends(get_db)):
    def q(cur):
        cur.execute("SELECT typology_id, name, stage FROM kb_typologies ORDER BY name")
        return [{"typology_id": t, "name": n, "stage": s} for t, n, s in cur.fetchall()]
    return {"typologies": db_call(conn, q)}


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description="复核接口（R11a）")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    # 只监听 127.0.0.1：演示环境没有 HTTPS，不对局域网开放
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()