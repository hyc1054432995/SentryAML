"""
SentryAML - R8a：知识库工具层（给 Agent 调用的法规检索工具）
==============================================================
R7 是一条固定的流水线：检索一次 → 回答一次。D57 的测试集诊断显示，5 道漏题全部是"答案条款不在来源里"，
而且模型在"未覆盖"里已经准确说出了缺什么（q011、q014），或直接点名了需要的条款（q030："paragraph 6.25"）。
R8 让模型自己决定"再查一次"或"按编号去取"。这一层只定义工具本身，不含调用大模型的循环（R8b）。

两个工具（模型能看到的只有名字、说明和参数格式）：
  search_regulations(query)                按意思检索，返回最多 5 段；本次调查中已经返回过的段落会跳过，
                                           所以换个说法再查，带回来的是新材料
  get_clause(document, clause_ids)         按编号取条款原文；附带它引用的条款、前后相邻条款的编号；
                                           "6.19-6.26" 这样的区间自动展开；当天没生效的条款会说明在哪个版本里

设计要点（每一条都对应一个具体问题）：
  1. 调查日期由系统绑定，不是模型的参数。所有工具共用 ToolContext.as_of。
     如果让模型填日期，它可能填"今天"或记错，查到错误的版本，而版本错误正是合规里最不能出的错。
  2. 参数用 pydantic 定义：同一份定义既生成给模型看的 JSON Schema，又负责校验模型给出的参数。
     pydantic 已经随 qdrant-client 安装，不增加依赖；手写两份（schema 一份、校验代码一份）迟早会不一致。
  3. 可以预期的错误作为"观察结果"返回给模型（参数不合法、条款不存在、当天未生效、预算用完、重复调用），
     模型可以据此改正；不可预期的错误（数据库连不上）直接抛出，让这次运行失败，而不是让模型绕过一个坏掉的系统。
  4. 每次调用写入 agent_tool_calls（019），记录模型看到的原样结果，用于调试、审计和评测。
  5. 本次调查看过的每一条条款原文都收进 ToolContext.evidence。最终回答只能引用这里面的条款，
     核验也对照这里的全文（R8b 复用 R7 的逐字核验）。
  R9b-1 起，第 2–4 条的通用实现（Tool、运行上下文、统一调用入口、留痕）移到 agent_core.py，与分诊 Agent 共用；
  这里只保留法规特有的部分：两个工具、文件清单、ToolContext 里的检索块与证据池。

用法：
    python scripts/kb_tools.py --schemas      # 打印模型看到的工具定义
    python scripts/kb_tools.py --offline      # 只测 get_clause（不调用任何 API）
    python scripts/kb_tools.py                # 完整冒烟测试：get_clause + 用 R7 的"未覆盖"做第二轮检索
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

import agent_core
from agent_core import RunContext, Tool
from agent_core import log_call as _log           # 旧名字保留：kb_agent 和下面的 seed_search 用的是 _log
from db import connect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_K = 5                 # 每次检索返回的段落数（比 R7 的 8 少：模型可以再查，不必一次给足）
MAX_UNITS_PER_CALL = 10      # get_clause 一次最多返回的条款数（区间展开后）
MAX_UNIT_CHARS = 3000        # 单条条款展示给模型的最大长度；evidence 里保存全文
DEFAULT_MAX_CALLS = 8        # 每次运行的工具调用预算

# 文件族：模型用这些名字指定文件。说明里写清楚各自的条款编号格式，模型才知道 "6-10-13" 该去 Guidelines 找
# 示例编号都刻意避开了评测集用到的条款（见 TOOLS 上方的说明）
DOCUMENTS = {
    "mas_n626": "MAS Notice 626 (binding AML/CFT requirements for banks). Ids like 9.2, 6.17A, 11A.3, fn5.",
    "mas_n626_guidelines": "Guidelines to MAS Notice 626 (how to comply). Ids like 10-3, 11-9-1, "
                           "appendix red-flag lists B-7, B-8, footnotes fn20.",
    "mas_tm_guidance": "MAS Guidance for Effective AML/CFT Transaction Monitoring Controls (2018). "
                       "Ids like 3.4, 4.3, case-D (Case Study D), fn6.",
    "fatf_cef": "FATF report: Illicit Financial Flows from Cyber-Enabled Fraud (2023). Ids like p43.1, box-30.",
    "fatf_pml": "FATF report: Professional Money Laundering (2018). Ids like p31.2, box-10.",
    "spf_str_form_guide": "Singapore Police Force: Guide to fill up the STR form. Ids like p12, p20.",
    "spf_stro_page": "STRO web page on suspicious transaction reporting. Ids like p4.2.",
    "ibm_aml_paper": "IBM paper on the AMLworld synthetic AML datasets. Ids like p3.4, p8.2.",
}
DocumentName = Literal["mas_n626", "mas_n626_guidelines", "mas_tm_guidance", "fatf_cef", "fatf_pml",
                       "spf_str_form_guide", "spf_stro_page", "ibm_aml_paper"]
assert set(DocumentName.__args__) == set(DOCUMENTS)      # 两处列表必须一致


# ---------------------------------------------------------------------------
# 运行上下文：一次调查共享的状态（通用部分在 agent_core.RunContext，R9b-1 抽出）
# ---------------------------------------------------------------------------
@dataclass
class ToolContext(RunContext):
    kind: str = "research"
    max_calls: int = DEFAULT_MAX_CALLS
    seen_chunks: set[str] = field(default_factory=set)        # 检索已返回过的块：再次检索时跳过
    evidence: dict[str, dict] = field(default_factory=dict)   # ref → 条款全文与出处；最终回答只能引用这些
    budget_note: str = "Answer now with the evidence you have, and state what is missing."
    duplicate_hint: str = "Change the query or fetch a different clause."

    def summary_extra(self) -> dict:
        return {"evidence": len(self.evidence)}


def _ref(family: str, unit_id: str) -> str:
    return f"{family}:{unit_id}"


def _add_evidence(ctx: ToolContext, row: dict) -> None:
    ctx.evidence.setdefault(_ref(row["family"], row["unit_id"]), {
        k: row[k] for k in ("family", "doc_id", "unit_id", "title", "version", "section", "text")})


def _show(text: str) -> str:
    if len(text) <= MAX_UNIT_CHARS:
        return text
    return text[:MAX_UNIT_CHARS] + f" […截断，还有 {len(text) - MAX_UNIT_CHARS} 字；用 get_clause 取单条全文]"


def _unit_rows(cur, doc_id: str, unit_ids: list[str]) -> list[dict]:
    cur.execute("""SELECT d.family, d.doc_id, d.title, d.version_label, u.unit_id, u.heading_path, u.text, u.ord
                   FROM kb_units u JOIN kb_documents d USING (doc_id)
                   WHERE u.doc_id = %s AND u.unit_id = ANY(%s) ORDER BY u.ord""", (doc_id, unit_ids))
    return [{"family": f, "doc_id": d, "title": t, "version": v, "unit_id": u, "section": " > ".join(h or []),
             "text": x, "ord": o} for f, d, t, v, u, h, x, o in cur.fetchall()]


# ---------------------------------------------------------------------------
# 工具 1：按意思检索
# ---------------------------------------------------------------------------
class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=3, max_length=400,
                       description="What to look for, phrased the way a regulation or guidance would state it "
                                   "(English works best), e.g. 'conditions before a bank may rely on a third "
                                   "party to perform customer due diligence'.")


def search_regulations(ctx: ToolContext, a: SearchArgs) -> dict:
    return _search(ctx, a.query, SEARCH_K)


def _search(ctx: ToolContext, query: str, k: int) -> dict:
    from kb_search import retrieve                       # 延迟导入：只用 get_clause 时不需要连 Qdrant
    # 多取一些再去掉已看过的，保证"换个说法再查"能带回新材料（测验 R7 收尾第 1 题：同样的查询得到同样的结果）
    hits = retrieve(query, ctx.as_of, k=min(30, k + len(ctx.seen_chunks)))
    fresh = [h for h in hits if h["chunk_id"] not in ctx.seen_chunks][:k]
    skipped = sum(h["chunk_id"] in ctx.seen_chunks for h in hits)
    results = []
    with connect() as conn, conn.cursor() as cur:
        for i, h in enumerate(fresh, 1):
            rows = _unit_rows(cur, h["doc_id"], h["unit_ids"])
            for r in rows:
                _add_evidence(ctx, r)
            ctx.seen_chunks.add(h["chunk_id"])
            first = rows[0] if rows else {"title": "", "version": "", "section": "", "family": ""}
            results.append({
                "rank": i,
                "refs": [_ref(r["family"], r["unit_id"]) for r in rows],
                "document": first["title"], "version": first["version"], "section": first["section"],
                # 重排分帮助模型判断"有没有真正相关的"。说明里不给固定阈值：测试集 q070（中文问题）的 gold
                # 只有 0.081，按阈值判断会把它当成不相关；而且从测试集推出阈值本身就违反纪律
                "relevance": None if h["score"] is None else round(h["score"], 3),
                "text": "\n".join(f"⟦{r['unit_id']}⟧ {_show(r['text'])}" for r in rows),
            })
    return {"query": query, "as_of": str(ctx.as_of), "results": results,
            "skipped_already_seen": skipped,
            "note": "" if results else "No new passages. Try different wording, or get_clause if you know the id."}


# ---------------------------------------------------------------------------
# 工具 2：按编号取条款
# ---------------------------------------------------------------------------
class GetClauseArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document: DocumentName = Field(description="Which document. " + " ".join(
        f"{k}: {v}" for k, v in DOCUMENTS.items()))
    clause_ids: list[str] = Field(min_length=1, max_length=5,
                                  description="Clause ids exactly as the document numbers them, e.g. ['9.2'], "
                                              "['10-3', 'fn20'], ['case-D']. A range such as '11.3-11.6' is "
                                              f"expanded (at most {MAX_UNITS_PER_CALL} clauses in total).")


RANGE_RE = re.compile(r"^\s*(\d{1,2}[A-Z]?\.\d{1,2}[A-Z]?)\s*(?:-|–|to)\s*(\d{1,2}[A-Z]?\.\d{1,2}[A-Z]?)\s*$")


def _norm_id(document: str, cid: str) -> str:
    """把常见写法统一成库里的编号：'Case Study B' → case-B，'footnote 3' → fn3，'paragraph 6.25' → 6.25。"""
    c = cid.strip()
    c = re.sub(r"^(?:paragraphs?|para\.?|section|clause)\s+", "", c, flags=re.I)
    m = re.fullmatch(r"(?:case(?:\s+study)?)[\s-]*([A-N])", c, re.I)
    if m:
        return f"case-{m.group(1).upper()}"
    m = re.fullmatch(r"(?:footnote|fn)\s*(\d{1,2})", c, re.I)
    if m:
        return f"fn{m.group(1)}"
    m = re.fullmatch(r"box[\s-]*(\d{1,2})", c, re.I)
    if m:
        return f"box-{m.group(1)}"
    return c


def get_clause(ctx: ToolContext, a: GetClauseArgs) -> dict:
    out = []
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, title, version_label FROM kb_documents_as_of(%s) WHERE family = %s",
                    (ctx.as_of, a.document))
        doc = cur.fetchone()
        if doc is None:
            return {"error": f"{a.document} has no version in force on {ctx.as_of}."}
        doc_id = doc[0]

        # 1. 展开区间，统一写法
        wanted: list[str] = []
        for cid in a.clause_ids:
            m = RANGE_RE.match(cid)
            if m:
                cur.execute("""SELECT unit_id FROM kb_units WHERE doc_id = %s AND kind <> 'heading'
                                 AND ord BETWEEN (SELECT ord FROM kb_units WHERE doc_id = %s AND unit_id = %s)
                                             AND (SELECT ord FROM kb_units WHERE doc_id = %s AND unit_id = %s)
                               ORDER BY ord""", (doc_id, doc_id, m.group(1), doc_id, m.group(2)))
                ids = [r[0] for r in cur.fetchall()]
                wanted += ids if ids else [cid]
            else:
                wanted.append(_norm_id(a.document, cid))
        wanted = list(dict.fromkeys(wanted))
        truncated = len(wanted) > MAX_UNITS_PER_CALL
        wanted = wanted[:MAX_UNITS_PER_CALL]

        rows = {r["unit_id"]: r for r in _unit_rows(cur, doc_id, wanted)}
        for uid in wanted:
            r = rows.get(uid)
            if r is None:
                out.append(_not_found(cur, a.document, uid, ctx.as_of, doc_id))
                continue
            _add_evidence(ctx, r)
            # 2. 它引用了哪些条款（只列在知识库里的；知识库之外的只给原文写法，告诉模型不必再追）
            cur.execute("""SELECT DISTINCT target_family, target_unit_id, raw, resolved FROM kb_xrefs
                           WHERE src_doc_id = %s AND src_unit_id = %s ORDER BY 1, 2""", (doc_id, uid))
            refs, outside = [], []
            for fam, tgt, raw, ok in cur.fetchall():
                if ok and fam in DOCUMENTS:
                    refs.append(_ref(fam, tgt))
                else:
                    outside.append(raw)
            # 3. 前后相邻的条款（q011：检索拿到了 6.21、6.22，答案在紧接着的 6.23）
            cur.execute("""(SELECT unit_id, 'previous' FROM kb_units WHERE doc_id = %s AND ord < %s
                              AND kind NOT IN ('heading', 'front') ORDER BY ord DESC LIMIT 1)
                           UNION ALL
                           (SELECT unit_id, 'next' FROM kb_units WHERE doc_id = %s AND ord > %s
                              AND kind NOT IN ('heading', 'front') ORDER BY ord LIMIT 1)""",
                        (doc_id, r["ord"], doc_id, r["ord"]))
            nb = {k: _ref(a.document, u) for u, k in cur.fetchall()}
            out.append({"ref": _ref(a.document, uid), "status": "found", "document": r["title"],
                        "version": r["version"], "section": r["section"], "text": _show(r["text"]),
                        "references": refs, "references_outside_knowledge_base": sorted(set(outside)),
                        "previous": nb.get("previous"), "next": nb.get("next")})
    res = {"document": a.document, "as_of": str(ctx.as_of), "results": out}
    if truncated:
        res["note"] = f"Only the first {MAX_UNITS_PER_CALL} clauses were returned; request the rest separately."
    return res


def _not_found(cur, family: str, uid: str, as_of: date, doc_id: str) -> dict:
    # 是不是别的版本里才有（版本陷阱：2025-03-01 去取 6.14C）
    cur.execute("""SELECT d.version_label, d.effective_from, d.effective_to FROM kb_units u
                   JOIN kb_documents d USING (doc_id) WHERE d.family = %s AND u.unit_id = %s
                   ORDER BY d.effective_from""", (family, uid))
    other = cur.fetchall()
    if other:
        spans = "; ".join(f"version '{v}' in force {f} to {t or 'now'}" for v, f, t in other)
        return {"ref": _ref(family, uid), "status": "not_in_force",
                "message": f"Clause {uid} is not in the version in force on {as_of}. It exists in: {spans}. "
                           f"Do not apply it to this investigation date."}
    # 编号写错：给几个相近的编号
    stem = re.sub(r"[A-Z]$", "", uid)
    cur.execute("""SELECT unit_id FROM kb_units WHERE doc_id = %s AND kind <> 'heading'
                   AND (unit_id LIKE %s OR unit_id LIKE %s) ORDER BY ord LIMIT 8""",
                (doc_id, stem + "%", re.split(r"[.\-]", uid)[0] + "%"))
    return {"ref": _ref(family, uid), "status": "not_found",
            "message": f"No clause '{uid}' in {family}.",
            "similar_ids": [r[0] for r in cur.fetchall()]}


# ---------------------------------------------------------------------------
# 工具登记表与统一调用入口
# ---------------------------------------------------------------------------
# 说明是写给模型的：它完全依靠这几句话决定"什么时候用哪个工具"，所以要写清楚适用和不适用的情形。
# 示例刻意选了评测集里没有的主题（依赖第三方做 CDD）：示例写成某道评测题的答案，就等于把答案写进了提示词
TOOLS = {t.name: t for t in [
    Tool("search_regulations",
         "Search the AML/CFT knowledge base (MAS Notice 626 and its Guidelines, MAS transaction monitoring "
         "guidance, STRO/SPF STR guidance, FATF typology reports, the IBM AMLworld dataset paper) for passages "
         f"relevant to a query. Returns up to {SEARCH_K} passages in force on the investigation date, most relevant "
         "first, each with a relevance score (0-1, higher is more relevant). Passages already "
         "returned earlier in this investigation are skipped, so a reworded search brings new material. "
         "Use it when you do not know which clause answers the question. Phrase the query the way the regulation "
         "would state the requirement, not as a copy of the user's question. "
         "Do not use it to fetch a clause whose id you already know: use get_clause.",
         SearchArgs, search_regulations),
    Tool("get_clause",
         "Fetch the exact text of specific clauses by id, in the version in force on the investigation date. "
         "Use it when a passage or the question names a clause (e.g. 'paragraph 9.2 of the Notice', "
         "'Guidelines 10-3', 'Case Study D', 'footnote 6'), or to read the clause just before or after one you "
         "already have. Each result lists the clauses it cross-references and its previous/next clause ids. "
         "If a clause is not in force on the investigation date, the result says which version contains it.",
         GetClauseArgs, get_clause),
]}


def tool_schemas() -> list[dict]:
    return [t.schema() for t in TOOLS.values()]


def call_tool(ctx: ToolContext, name: str, raw_args: str | dict) -> dict:
    """执行一次模型发起的工具调用并记录（校验、预算、重复调用、留痕的逻辑在 agent_core.call_tool）。"""
    return agent_core.call_tool(ctx, TOOLS, name, raw_args)


def seed_search(ctx: ToolContext, question: str, k: int = 8) -> dict:
    """
    系统发起的首轮检索（R8b）：用用户的问题检索 k 段，与 R7 交给模型的 8 个来源完全相同。
    这样 Agent 是在 R7 的基础上"继续查"，比较 R7 与 Agent 时，差别只来自后续的工具调用。
    不计入模型的调用预算；照常记录在追踪表里（工具名 seed_search）。
    """
    ctx.step += 1
    t0 = time.perf_counter()
    result = _search(ctx, question, k)
    _log(ctx, "seed_search", {"query": question, "k": k}, result, None, t0)
    return result


# ---------------------------------------------------------------------------
# 冒烟测试（只用开发集题目；测试集冻结，不拿来设计工具）
# ---------------------------------------------------------------------------
def _brief(res: dict) -> str:
    if "error" in res:
        return f"✗ {res['error']}" + (f" {res['details']}" if res.get("details") else "")
    lines = []
    for r in res.get("results", []):
        if "rank" in r:
            lines.append(f"    {r['rank']}. {', '.join(r['refs'])}  相关度 {r['relevance']}")
        elif r["status"] == "found":
            lines.append(f"    ✓ {r['ref']}（{len(r['text'])} 字）引用 {r['references'] or '无'}；"
                         f"前 {r['previous']}，后 {r['next']}"
                         + (f"；知识库外 {r['references_outside_knowledge_base']}"
                            if r["references_outside_knowledge_base"] else ""))
        else:
            lines.append(f"    · {r['ref']} {r['status']}：{r['message'][:150]}"
                         + (f" 相近编号 {r['similar_ids']}" if r.get("similar_ids") else ""))
    if res.get("skipped_already_seen"):
        lines.append(f"    （跳过已看过的 {res['skipped_already_seen']} 段）")
    return "\n".join(lines)


def smoke(offline: bool) -> None:
    ctx = ToolContext("kb_tools 冒烟测试", date(2026, 9, 30), kind="smoke").start()
    old = ToolContext("kb_tools 冒烟测试（旧版本）", date(2025, 3, 1), kind="smoke").start()

    print("1. 版本：同一个编号，两个调查日期")
    print("  2025-03-01:\n" + _brief(call_tool(old, "get_clause", {"document": "mas_n626", "clause_ids": ["6.14C"]})))
    print("  2026-09-30:\n" + _brief(call_tool(ctx, "get_clause", {"document": "mas_n626", "clause_ids": ["6.14C"]})))

    print("\n2. 区间展开 + 引用：TM 指引脚注 3 指向 Notice 6.19-6.26（开发集 q032 的桥梁）")
    print(_brief(call_tool(ctx, "get_clause", {"document": "mas_tm_guidance", "clause_ids": ["footnote 3"]})))
    print(_brief(call_tool(ctx, "get_clause", {"document": "mas_n626", "clause_ids": ["6.19-6.26"]})))

    print("\n3. 知识库之外的引用不再列为可追的条款（D59 修正前，FSM-N02 的 4.1 被当成 Notice 626 的 4.1）")
    print(_brief(call_tool(ctx, "get_clause", {"document": "mas_n626_guidelines", "clause_ids": ["6-10-13"]})))

    print("\n4. 可以预期的错误，作为观察结果返回")
    print("  编号不存在：\n" + _brief(call_tool(ctx, "get_clause", {"document": "mas_n626", "clause_ids": ["6.99"]})))
    print("  文件名不合法：" + _brief(call_tool(ctx, "get_clause", {"document": "notice_626", "clause_ids": ["6.1"]})))
    print("  多余的参数：" + _brief(call_tool(ctx, "get_clause", {"document": "mas_n626", "clause_ids": ["6.1"],
                                                                  "as_of": "2020-01-01"})))
    print("  重复调用：" + _brief(call_tool(ctx, "get_clause", {"document": "mas_n626", "clause_ids": ["6.19-6.26"]})))
    print("  参数不是 JSON：" + _brief(call_tool(ctx, "get_clause", '{"document": "mas_n626", clause_ids: 6.1}')))

    if not offline:
        print("\n5. 第二轮检索：用 R7 开发集记录里模型自己写的'未覆盖'作为查询（开发集 q017、q032）")
        qs = {q["id"]: q for q in json.loads((PROJECT_ROOT / "reports" / "rag_golden_resolved.json")
                                             .read_text(encoding="utf-8"))}
        recs = json.loads((PROJECT_ROOT / "reports" / "r7_answer_dev.json").read_text(encoding="utf-8"))
        for qid in ("q017", "q032"):
            q, rec = qs[qid], recs["records"]["full"][qid]
            gold = {_ref(g["family"], g["unit_id"]) for g in q["gold"]}
            c = ToolContext(q["question"], date.fromisoformat(q["as_of"]), kind="smoke").start()
            r1 = call_tool(c, "search_regulations", {"query": q["question"]})
            hit1 = gold & {x for r in r1["results"] for x in r["refs"]}
            print(f"\n  {qid} 第 1 轮（用户原问题）：命中 gold {sorted(hit1) or '无'}\n" + _brief(r1))
            missing = (rec.get("missing") or "").strip()
            if not missing:
                print("  （R7 记录里'未覆盖'为空，跳过第 2 轮）")
                continue
            r2 = call_tool(c, "search_regulations", {"query": missing[:400]})
            hit2 = gold & {x for r in r2["results"] for x in r["refs"]}
            print(f"  {qid} 第 2 轮（查询 = R7 的'未覆盖'：{missing[:100]}…）：命中 gold {sorted(hit2) or '无'}\n"
                  + _brief(r2))
            c.finish("done", {"gold_round1": sorted(hit1), "gold_round2": sorted(hit2)})

    ctx.finish("done")
    old.finish("done")
    print("\n6. 追踪记录（agent_tool_calls）：")
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT tool, count(*), count(*) FILTER (WHERE ok), string_agg(DISTINCT error, ', '),
                              round(avg(ms)) FROM agent_tool_calls
                       WHERE run_id = ANY(%s::uuid[]) GROUP BY tool""", ([ctx.run_id, old.run_id],))
        for tool, n, ok, errs, ms in cur.fetchall():
            print(f"    {tool:<20} {n} 次，成功 {ok}，错误类型 {errs or '无'}，平均 {ms} ms")
    print(f"  本次调查收集的条款（evidence）：{len(ctx.evidence)} 条，例如 {list(ctx.evidence)[:4]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="知识库工具层")
    ap.add_argument("--schemas", action="store_true", help="打印模型看到的工具定义")
    ap.add_argument("--offline", action="store_true", help="只测 get_clause，不调用检索 API")
    a = ap.parse_args()
    if a.schemas:
        print(json.dumps(tool_schemas(), ensure_ascii=False, indent=2))
    else:
        smoke(a.offline)