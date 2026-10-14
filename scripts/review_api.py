"""
SentryAML - R11a–R11c：复核接口与复核界面（FastAPI）
===================================================
把命令行的复核流程变成多人使用的网页接口。这一层不重新实现任何规则——状态机、角色、理由、双人复核、防过期、
租约全部由数据库强制（025、026、027、028）。接口只做三件事：
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
  GET  /api/cases/{id}/aids           R11b：解读提示 + 最新草稿 + 数据库检查 + 理由逐句对照（review_hints.py）
  POST /api/cases/{id}/drafts         R11b：组装 STR 草稿的结构化部分（str_draft.build_body），写入新版本
  POST /api/cases/{id}/narrative      R11b：让模型起草理由并核验（str_narrative.write），通过才写入新版本
  POST /api/cases/{id}/revise         R11b：人工修改理由、删减可疑类别，写入新版本
  POST /api/cases/{id}/ask            R11c：问法规（R8 的法规研究 Agent），记录谁在哪个案件上问的；不改变案件
  GET  /api/cases/{id}/regulation     R11c：这个案件上问过的法规问题和回答
  GET  /api/docs                      自动生成的接口文档
页面（R11b，review_pages.py）：/login、/queue、/cases/{id}——服务端渲染，按钮调用上面这些接口

启动（只监听本机）：
    python scripts/review_api.py            # 复核界面 http://127.0.0.1:8000/   接口文档 /api/docs
测试：
    python scripts/review_api_test.py       # R11a 接口的端到端测试，全部在一个事务里，结束时回滚
    python scripts/review_web_test.py       # R11b 草稿接口、页面、解读提示
    python scripts/review_reg_test.py       # R11c 问法规（假 Agent，不花钱）
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import threading
from datetime import datetime, timezone
from typing import Iterator, Literal

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

import review_auth as auth
from db import connect_review
from review import act, explain
from str_draft import SGT

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
          "session_role": 403, "reg_role": 403,
          "review_reason": 422, "str_typology": 422, "str_version": 422, "str_no_draft": 422, "str_narrative": 422,
          "str_invalid": 422, "str_state": 422, "reg_answer": 422}

# 复核页面的内容安全策略：只允许运行、加载本站的脚本和样式（不允许页面里内联的脚本、不从外部 CDN 加载）。
# 即使模型写的理由里混进了 <script>，模板转义之外还有这一道：浏览器拒绝执行不是来自本站文件的脚本。
# /api/docs 的接口文档页需要从 CDN 加载 Swagger，不套用这条策略
PAGE_CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
            "form-action 'self'; frame-ancestors 'none'; base-uri 'none'; object-src 'none'")

app = FastAPI(title="SentryAML review API", version="r11a", docs_url="/api/docs", openapi_url="/api/openapi.json",
              redoc_url=None)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")    # 浏览器不要猜文件类型
    resp.headers.setdefault("X-Frame-Options", "DENY")              # 不能被别的网站嵌进框架里诱导点击
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    path = request.url.path
    if not path.startswith("/api/docs") and not path.startswith("/api/openapi") and not path.startswith("/static/"):
        resp.headers.setdefault("Cache-Control", "no-store")        # 案件数据不留在浏览器缓存里
    if not path.startswith("/api/"):
        resp.headers.setdefault("Content-Security-Policy", PAGE_CSP)
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
    except SystemExit as e:
        # str_draft / str_narrative 是命令行脚本，遇到"没有草稿""状态不对"之类会 SystemExit；在接口里绝不能让它
        # 结束服务器进程，翻译成 422（事务已随异常回滚）
        raise HTTPException(422, {"code": "rejected", "message": str(e)})
    except psycopg.Error as e:
        msg = (e.diag.message_primary or str(e)) if getattr(e, "diag", None) else str(e)
        code = msg.split(":", 1)[0]
        if code in STATUS:
            raise HTTPException(STATUS[code], {"code": code, "message": msg, "hint": explain(e)})
        # 其他数据库错误（程序错误、权限配置错误）：细节只写日志，不返回给浏览器
        log.exception("database error")
        raise HTTPException(500, {"code": "internal", "message": "database error; see server log"})


def err(status: int, code: str, message: str, hint: str | None = None, headers: dict | None = None) -> HTTPException:
    """hint：给页面显示的中文说明（review.js 优先显示它）；message 是给调用接口的程序看的英文。"""
    body = {"code": code, "message": message} | ({"hint": hint} if hint else {})
    return HTTPException(status, body, headers=headers)


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
    return {"cases": db_call(conn, lambda cur: queue_rows(cur, state, mine, who.reviewer_id))}


def queue_rows(cur, state: str | None, mine: bool, reviewer_id: str) -> list[dict]:
    """队列：接口和页面共用。不列 Agent 的结论：先看证据再看结论（自动化偏差，D78）；来源（route）是分流规则，不是结论。"""
    cur.execute("""SELECT s.case_id, s.alert_id, a.tier, s.route, s.state, s.last_seq, s.last_actor, s.last_at,
                          cl.holder, cl.expires_at
                   FROM review_case_state s JOIN alerts a USING (alert_id)
                   LEFT JOIN review_claim_state cl ON cl.case_id = s.case_id
                   WHERE (%(state)s::text IS NULL OR s.state = %(state)s)
                     AND (NOT %(mine)s OR cl.holder = %(me)s)
                   ORDER BY s.case_id""", {"state": state, "mine": mine, "me": reviewer_id})
    cols = [d.name for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


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


# ---------------------------------------------------------------------------
# R11b：草稿与理由
# ---------------------------------------------------------------------------
# draft_str / edit_str 两个复核事件不在通用动作接口里：它们必须和"写入草稿"在同一个事务里（D79），
# 所以每个草稿接口都调用 str_draft.insert_draft——写草稿、记事件、数据库检查，要么都成功，要么都不发生
def _latest_version(cur, case_id: int) -> int | None:
    cur.execute("SELECT max(version) FROM str_drafts WHERE case_id = %s", (case_id,))
    return cur.fetchone()[0]


def _require_version(cur, case_id: int, seen: int) -> None:
    """页面上看到的是第 seen 版，最新的却不是它：拒绝，要求刷新——在旧版本上修改，会悄悄覆盖别人的改动。"""
    v = _latest_version(cur, case_id)
    if v != seen:
        raise err(409, "draft_stale", f"you are looking at version {seen}, but the latest draft is version {v}; "
                                      "reload and review it again")


def _claim_precheck(cur, case_id: int, who: auth.Reviewer) -> None:
    """只是提前告知（调用模型要十几秒、要花钱）；真正的规则仍在数据库里（027 的租约触发器）。"""
    cur.execute("SELECT holder FROM review_claim_state WHERE case_id = %s", (case_id,))
    r = cur.fetchone()
    if r and r[0] and r[0] != who.reviewer_id and who.role != "mlro":
        raise err(409, "review_claimed", f"case {case_id} is claimed by {r[0]}")


@app.get("/api/cases/{case_id}/aids")
def aids(case_id: int, who: auth.Reviewer = Depends(me), conn: psycopg.Connection = Depends(get_db)):
    import review_hints

    def go(cur):
        _require_case(cur, case_id)
        return review_hints.case_aids(cur, case_id)
    return db_call(conn, go)


class DraftIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seen_state: str = Field(min_length=1, max_length=40)


@app.post("/api/cases/{case_id}/drafts")
def build_draft(case_id: int, body: DraftIn, who: auth.Reviewer = Depends(writer),
                conn: psycopg.Connection = Depends(get_db)):
    """组装结构化部分（不调用模型，D79），写入新版本：首次为 draft_str，之后为 edit_str（理由会清空，需要重新起草）。"""
    import str_draft as sd

    def go(cur):
        _require_case(cur, case_id)
        case = sd.load_case(cur, case_id)
        b, csv_text = sd.build_body(cur, case)
        v, sha = sd.insert_draft(cur, case, who.reviewer_id, b, csv_text,
                                 {"kind": "builder", "builder": sd.BUILDER}, seen=body.seen_state)
        return {"case_id": case_id, "version": v, "accounts": len(b["accounts"]),
                "transactions": len(b["suspicious_transactions"]), "attachment_rows": b["attachment"]["rows"],
                "categories": len(b["reasons"]["categories"]), "attachment_sha256": sha}
    return db_call(conn, go)


class NarrativeIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    draft_version: int = Field(ge=1)


@app.post("/api/cases/{case_id}/narrative")
def write_narrative(case_id: int, body: NarrativeIn, who: auth.Reviewer = Depends(writer),
                    conn: psycopg.Connection = Depends(get_db)):
    """让模型起草理由（一次调用，不是 Agent，D80），程序核验；通过才写入新版本，作者是点按钮的人（他采纳了这段文字）。"""
    import str_narrative as narr

    def go(cur):
        _require_case(cur, case_id)
        cur.execute("SELECT state FROM review_case_state WHERE case_id = %s", (case_id,))
        if cur.fetchone()[0] != "str_drafted":
            raise err(409, "str_state", "assemble the draft first")
        _require_version(cur, case_id, body.draft_version)
        _claim_precheck(cur, case_id, who)
        rec = narr.write(case_id, who.reviewer_id, cur=cur)
        narr.save_run(rec)                                  # 每次运行（含失败）都留记录，与命令行一致
        if rec["status"] != "verified" or "new_version" not in rec:
            last = (rec["attempts"] or [{}])[-1]
            detail = rec.get("db_error") or "; ".join(f"{p['problem']}: {p['detail']}" for p in last.get("problems", [])) \
                or last.get("error") or rec["status"]
            raise err(422, "narrative_failed", f"the drafted text did not pass the checks and was not saved ({detail})")
        return {"case_id": case_id, "version": rec["new_version"], "attempts": len(rec["attempts"]),
                "cost_usd": rec["cost_usd"], "prompt_version": rec["prompt_version"]}
    return db_call(conn, go)


class ReviseIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    draft_version: int = Field(ge=1)
    narrative: str | None = Field(None, max_length=10000)
    drop_categories: list[str] = Field(default_factory=list, max_length=50)

    @model_validator(mode="after")
    def _something(self):
        if self.narrative is None and not self.drop_categories:
            raise ValueError("没有要保存的修改：先改理由文字，或勾选要去掉的可疑类别")
        return self


@app.post("/api/cases/{case_id}/revise")
def revise(case_id: int, body: ReviseIn, who: auth.Reviewer = Depends(writer),
           conn: psycopg.Connection = Depends(get_db)):
    """人工修改：只能改理由、删可疑类别（D79：结构化部分来自数据，要改就重新组装）。不能新增类别——
    类别必须是知识图谱原文，由组装器按手法和审查日期取出；删减由数据库允许，但至少要留一条（no_categories）。"""
    import str_draft as sd

    def go(cur):
        _require_case(cur, case_id)
        case = sd.load_case(cur, case_id)
        _require_version(cur, case_id, body.draft_version)
        d = sd.latest(cur, case_id)
        old = d["body"]["reasons"]
        have = {c["indicator_id"] for c in old["categories"]}
        unknown = sorted(set(body.drop_categories) - have)
        if unknown:
            raise err(422, "category_unknown", f"not in version {d['version']}: {unknown}")
        text = body.narrative.strip() if body.narrative is not None else (old.get("narrative") or "")
        edited = body.narrative is not None and text != (old.get("narrative") or "").strip()
        b = sd.revise_body(d["body"], text or None)
        if edited:
            # 理由改过了，原来随理由保存的"依据的事实"对应的是旧文字，不再成立
            b["reasons"].pop("narrative_facts", None)
        drop = set(body.drop_categories)
        b["reasons"]["categories"] = [c for c in old["categories"] if c["indicator_id"] not in drop]
        b["reasons"]["categories_dropped_by_reviewer"] = sorted(
            set(old.get("categories_dropped_by_reviewer") or []) | drop)
        gen = {"kind": "human", "base_version": d["version"], "narrative_edited": edited,
               "dropped_categories": sorted(drop)}
        v, _ = sd.insert_draft(cur, case, who.reviewer_id, b, d["csv"], gen, seen="str_drafted")
        return {"case_id": case_id, "version": v, "narrative_edited": edited, "dropped_categories": sorted(drop)}
    return db_call(conn, go)


# ---------------------------------------------------------------------------
# R11c：问法规
# ---------------------------------------------------------------------------
# 同时最多几个法规 Agent 在跑。普通 def 接口共用一个线程池（默认 40 个线程）：法规 Agent 每次要等模型 5–30 秒，
# 不设上限的话，几十个人同时提问就会占满线程，连队列页面都打不开。上限也让我们对模型服务的并发不超过 3。
# 这是每个进程一份（进程之间不共享内存）：开 N 个进程，全系统最多 3N 个；每人每小时的次数在数据库里数，不受影响
REG_CONCURRENCY = 3
REG_SLOTS = threading.BoundedSemaphore(REG_CONCURRENCY)
# 问题里像账户号的写法（银行号:账户号，账户号至少 8 位十六进制）。"10:30"、条款编号 "mas_n626:14.1" 不会被当成账户
ACCT_IN_TEXT = re.compile(r"(?<![\w.])\d{1,6}:[0-9A-Fa-f]{8,}\b")
REG_STATUS = {"answered", "partial", "abstained"}


class AskIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=5, max_length=1000)


def _case_accounts(cur, case_id: int) -> list[str]:
    cur.execute("""SELECT a.from_bank || ':' || a.from_account, a.to_bank || ':' || a.to_account
                   FROM review_cases c JOIN alerts a ON a.alert_id = c.alert_id WHERE c.case_id = %s""", (case_id,))
    return [x for x in cur.fetchone() if x]


@app.post("/api/cases/{case_id}/ask")
def ask_regulation(case_id: int, body: AskIn, who: auth.Reviewer = Depends(writer),
                   conn: psycopg.Connection = Depends(get_db)):
    """向法规研究 Agent（R8 选定的 agent-v3）提问。回答只给复核员参考：不改变案件状态，不写进 STR，不能当复核理由。
    顺序：检查问题 → 检查次数 → 取并发名额 → 调用 Agent（这期间不开数据库事务）→ 一个很短的事务写记录。"""
    import kb_agent

    question = " ".join(body.question.split())

    def pre(cur):
        _require_case(cur, case_id)
        accounts = _case_accounts(cur, case_id)
        cur.execute("SELECT review_reg_quota_left(%s)", (who.reviewer_id,))
        return accounts, cur.fetchone()[0]
    accounts, left = db_call(conn, pre)

    # 数据最小化：问答日志是全库共用的，任何复核员都能读；问题里写了账户号，案件信息就流到了不该看的人那里。
    # 而且法规检索靠意思，账户号检索不到任何条款。不悄悄删掉再发：那样日志里记的就不是复核员写的话
    if ACCT_IN_TEXT.search(question) or any(a in question for a in accounts):
        raise err(422, "reg_case_data", "the question contains an account number; ask about the situation in general "
                                        "terms (e.g. 'funds received and moved out within hours'), not about this case",
                  "问题里有账户号。请按一般情形提问（例如\"收到转账后数小时内全部转出\"），不要写本案的账户——"
                  "问答日志所有复核员都能看到，而且法规检索用不上账户号")
    if left <= 0:
        raise err(429, "reg_quota", "hourly limit for regulation questions reached",
                  "这一小时问法规的次数已经用完，请稍后再问", headers={"Retry-After": "600"})
    if not REG_SLOTS.acquire(blocking=False):
        raise err(503, "reg_busy", f"{REG_CONCURRENCY} regulation questions are already running; try again shortly",
                  f"已有 {REG_CONCURRENCY} 个法规查询在进行，请半分钟后再试", headers={"Retry-After": "30"})
    try:
        # 调查日期 = 今天（新加坡时间）：STR 现在提交，适用现在有效的规则版本
        rec = kb_agent.research(question, as_of=datetime.now(SGT).date())
    except Exception:
        log.exception("regulation agent failed")
        raise err(502, "reg_failed", "the regulation agent failed; nothing was recorded",
                  "法规查询出错，没有留下记录，可以再试一次")
    finally:
        REG_SLOTS.release()
    if rec.get("answer_id") is None or rec.get("status") not in REG_STATUS:
        log.error("regulation agent returned no logged answer: %s", rec.get("status"))
        raise err(502, "reg_failed", "the regulation agent did not log its answer; nothing was recorded",
                  "法规查询没有写入日志，没有留下记录，可以再试一次")

    def link(cur):
        cur.execute("""INSERT INTO review_reg_queries (case_id, actor, answer_id) VALUES (%s, %s, %s)
                       RETURNING query_id""", (case_id, who.reviewer_id, rec["answer_id"]))
        return cur.fetchone()[0]
    query_id = db_call(conn, link)
    return {"case_id": case_id, "query_id": query_id, "answer_id": rec["answer_id"], "as_of": rec["as_of"],
            "status": rec["status"], "abstain_reason": rec.get("abstain_reason"), "cost_usd": rec.get("cost_usd")}


def reg_history(cur, case_id: int) -> list[dict]:
    """这个案件上问过的法规问题，新的在前。回答全文来自 kb_answer_log（Agent 写的），这里只挑页面要的字段。"""
    cur.execute("""SELECT q.query_id, q.actor, q.created_at, l.question, l.as_of, l.status, l.abstain_reason, l.record
                   FROM review_reg_queries q JOIN kb_answer_log l ON l.answer_id = q.answer_id
                   WHERE q.case_id = %s ORDER BY q.query_id DESC""", (case_id,))
    out = []
    for qid, actor, at, question, as_of, status, reason, rec in cur.fetchall():
        rec = rec or {}
        claims = [{"statement": c.get("statement") or "", "ref": c.get("source") or "", "title": c.get("title") or "",
                   "quote": c.get("quote") or "", "verified": bool(c.get("verified"))}
                  for c in rec.get("claims") or []]
        out.append({"query_id": qid, "actor": actor, "asked_at": at, "question": question, "as_of": as_of,
                    "status": status, "abstain_reason": reason, "answer": rec.get("answer") or "",
                    "missing": rec.get("missing") or "", "claims": claims, "cost_usd": rec.get("cost_usd"),
                    "seconds": round((rec.get("ms") or {}).get("total", 0) / 1000, 1)})
    return out


@app.get("/api/cases/{case_id}/regulation")
def regulation(case_id: int, who: auth.Reviewer = Depends(me), conn: psycopg.Connection = Depends(get_db)):
    def go(cur):
        _require_case(cur, case_id)
        cur.execute("SELECT review_reg_quota_left(%s)", (who.reviewer_id,))
        left = cur.fetchone()[0]
        return {"queries": reg_history(cur, case_id), "quota_left": left}
    return db_call(conn, go)


@app.get("/api/typologies")
def typologies(who: auth.Reviewer = Depends(me), conn: psycopg.Connection = Depends(get_db)):
    def q(cur):
        cur.execute("SELECT typology_id, name, stage FROM kb_typologies ORDER BY name")
        return [{"typology_id": t, "name": n, "stage": s} for t, n, s in cur.fetchall()]
    return {"typologies": db_call(conn, q)}


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description="复核接口与复核界面（R11）")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    # 只监听 127.0.0.1：演示环境没有 HTTPS，不对局域网开放
    uvicorn.run(app, host="127.0.0.1", port=a.port)


# 复核页面（R11b）：页面模块不反过来 import 本模块，而是由这里把本模块传给它——直接运行本文件时模块名是 __main__，
# 页面模块若 import review_api 会得到第二份副本，路由会注册到一个没有在运行的 app 上
import review_pages  # noqa: E402
review_pages.register(app, sys.modules[__name__])

if __name__ == "__main__":
    main()