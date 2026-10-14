"""
SentryAML - R11b：复核界面（服务端渲染）
========================================
三个页面：/login、/queue、/cases/{id}。页面只负责"显示"：数据来自和接口完全相同的函数（case_detail、queue_rows、
review_hints.case_aids），所以 R11a 接口的测试同时覆盖了页面的数据；页面上的每个按钮都调用已经测好的 JSON 接口
（static/review.js），不另设表单专用的写入路径——写入路径只有一条，规则只有一份（D78）。

为什么服务端渲染（Jinja2）而不是 React 单页应用（D85）：一个程序、一套代码，不需要前端构建工具；所有判断留在后端。

安全：
  - 模板自动转义：模型写的理由、复核员写的理由都是"不可信输入"，里面的 <script> 只会被当作文字显示
  - 内容安全策略（review_api.PAGE_CSP）：只允许本站的脚本和样式文件，页面里不写任何内联脚本或内联样式
  - 页面上没有任何"能不能做"的判断：按钮亮不亮只是提示（allowed_actions），点了以后由数据库决定

本模块不 import review_api（直接运行 review_api.py 时它的模块名是 __main__，再 import 会得到第二份副本），
而是由 review_api 在文件末尾调用 register(app, 自己)。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from fastapi import Depends, Request
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import review_auth as auth
from str_draft import SGT

HERE = Path(__file__).resolve().parent

STATE_NAMES = {"new": "新建", "pending_review": "待复核", "suspicious_confirmed": "已确认可疑",
               "str_drafted": "STR 草稿中", "str_approved": "STR 已批准", "closed": "已关闭"}
ACTION_NAMES = {"open": "送入复核", "close": "关闭", "confirm_suspicious": "确认可疑", "request_info": "要求补查",
                "draft_str": "生成草稿", "edit_str": "修改草稿", "return_draft": "退回草稿", "approve_str": "批准 STR",
                "reopen": "重开"}
ROUTE_NAMES = {"tier1": "T1 直接复核", "agent_escalate": "Agent 升级", "qa_sample": "关闭抽查"}
ROLE_NAMES = {"analyst": "分析员", "mlro": "MLRO", "system": "系统"}
HINT_NAMES = {"clearing": "清算层", "shared_out": "过桥重复匹配", "regular_repeat": "规律性付款"}
# 需要写理由的动作（与 025 的 reason_rule 一致；只用来在页面上提示，真正的检查在数据库）
REASON_ALWAYS = {"request_info", "return_draft", "reopen"}


def sgt(dt) -> str:
    """数据库统一存 UTC（TIMESTAMPTZ），页面统一显示新加坡时间。"""
    if dt is None:
        return ""
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt)
    return dt.astimezone(SGT).strftime("%Y-%m-%d %H:%M") + " SGT"


def register(app, api) -> None:
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.filters["sgt"] = sgt
    templates.env.globals.update(STATE_NAMES=STATE_NAMES, ACTION_NAMES=ACTION_NAMES, ROUTE_NAMES=ROUTE_NAMES,
                                 ROLE_NAMES=ROLE_NAMES, HINT_NAMES=HINT_NAMES, REASON_ALWAYS=REASON_ALWAYS)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    def viewer(request: Request, conn=Depends(api.get_db)):
        """页面用的身份：没登录不报 401，而是返回 None，由页面跳到登录页。"""
        token = request.cookies.get(auth.COOKIE)
        if not token or len(token) > 128:
            return None
        return api.db_call(conn, lambda cur: auth.lookup_session(cur, token))

    def render(request: Request, name: str, who, **ctx):
        csrf = auth.csrf_for(who.token, api.SECRET) if who else ""
        return templates.TemplateResponse(request, name, {"who": who, "csrf": csrf, **ctx})

    def to_login():
        return RedirectResponse("/login", status_code=303)

    @app.get("/", include_in_schema=False)
    def home(who=Depends(viewer)):
        return RedirectResponse("/queue" if who else "/login", status_code=303)

    @app.get("/login", include_in_schema=False)
    def login_page(request: Request, who=Depends(viewer)):
        if who:
            return RedirectResponse("/queue", status_code=303)
        return render(request, "login.html", None)

    @app.get("/queue", include_in_schema=False)
    def queue_page(request: Request, state: str | None = None, mine: bool = False, who=Depends(viewer),
                   conn=Depends(api.get_db)):
        if who is None:
            return to_login()
        if state not in STATE_NAMES:
            state = None
        rows = api.db_call(conn, lambda cur: api.queue_rows(cur, state, mine, who.reviewer_id))
        return render(request, "queue.html", who, rows=rows, state=state, mine=mine)

    @app.get("/cases/{case_id}", include_in_schema=False)
    def case_page(request: Request, case_id: int, who=Depends(viewer), conn=Depends(api.get_db)):
        if who is None:
            return to_login()
        import review_hints

        def go(cur):
            detail = api.case_detail(cur, case_id, who)          # 不存在时抛 404
            aids = review_hints.case_aids(cur, case_id)
            cur.execute("SELECT typology_id, name FROM kb_typologies ORDER BY name")
            return detail, aids, cur.fetchall()
        detail, aids, typologies = api.db_call(conn, go)
        claim = detail["claim"]
        blocked = bool(claim["holder"]) and not claim["mine"] and who.role != "mlro"
        return render(request, "case.html", who, d=detail, aids=aids, typologies=typologies, blocked=blocked)