"""
SentryAML - R8b：法规研究 Agent（手写的工具调用循环）
=====================================================
research(question, as_of) 回答一个法规问题，和 R7 的 answer() 返回同样格式的记录，所以 R7 的评测脚本可以直接比较两者。

与 R7 的区别只有一处：R7 检索一次就回答；这里模型读完首轮检索的结果后，可以自己决定
  再检索一次（search_regulations）、按编号取条款（get_clause），或者提交回答（submit_answer）。

一次运行的流程：
  1. 范围检查与首轮检索并行（同 R7）。首轮检索由系统发起，与 R7 交给模型的 8 个来源完全相同（seed_search）。
     超出范围 → 直接拒答，不进入循环。
  2. 循环：把消息历史和工具清单发给模型 → 模型返回工具调用 → 执行 → 把结果作为 tool 消息追加 → 再发给模型。
     消息历史只追加、从不修改，所以每一轮的前缀都与上一轮相同，DeepSeek 的前缀缓存会命中前面全部内容（R5a 的同一机制）。
  3. 模型调用 submit_answer 时，用 R7 的逐字核验检查每条论断（只能引用本次调查看过的条款，即证据池）。
     有论断没通过 → 把核验结果作为这次工具调用的返回值交给模型，允许改正一次（可以再去查）。
  4. 结束条件（任一满足即停）：提交并通过核验 / 第二次提交 / 模型调用次数达到 MAX_TURNS / 工具预算用完后仍不提交。
  R9b-1 起，循环、工具调用入口、用量记账移到 agent_core.py，与分诊 Agent 共用；这里只保留法规特有的部分
  （提示词、提交格式与逐字核验、状态判定）。抽取前后发给模型的内容逐字节相同，由 agent_regress.py 用缓存回放证明。

为什么"提交回答"也做成一个工具：
  回答的格式（answerable、answer、claims、missing）由工具的参数格式规定并由 pydantic 校验，和其他工具走同一条路径；
  模型"调用 submit_answer"本身就是明确的结束信号，不需要从自由文本里猜它是不是答完了。

用法：
    python scripts/kb_agent.py "What must a bank do before relying on a third party for CDD?"
    python scripts/kb_agent.py --qids q017 q032     # 用开发集题目（拒绝测试集题目）
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import llm
from agent_core import add_usage, inline_schema, new_usage, run_loop
from db import connect
from kb_answer import (MAX_QUOTE_WORDS, MIN_QUOTE_WORDS, _PAIRS, quoted_spans, render, scope_check,
                       unverified_quotes, verify)
from kb_tools import TOOLS, ToolContext, _log, call_tool, seed_search, tool_schemas

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AGENT_VERSION = "agent-v3"   # v2（D64）：拒答可附说明缺口的论断、引文不得写进 missing、可关闭首轮检索；v3（D65）：正文引号内的文字也要逐字核验
MAX_TOOL_CALLS = 6          # 模型可以发起的检索 / 取条款次数（不含首轮检索和提交）
MAX_TURNS = 12              # 调用模型的最多轮数：工具预算之外的第二道保险，防止任何情况下的无限循环
MAX_SUBMITS = 2             # 第一次提交未通过核验时，允许改正一次
MAX_NUDGES = 2              # 模型既不调用工具也不提交时，提醒的次数


# ---------------------------------------------------------------------------
# 提交回答的工具
# ---------------------------------------------------------------------------
class ClaimArg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statement: str = Field(description="One statement made in the answer.")
    ref: str = Field(description="The clause the quote is copied from, exactly as listed in a result's 'refs' "
                                 "or 'ref', e.g. 'mas_n626:9.2'.")
    quote: str = Field(description=f"Words copied character for character from that clause: one contiguous span "
                                   f"of {MIN_QUOTE_WORDS} to {MAX_QUOTE_WORDS} words, no ellipsis, no paraphrase.")


class SubmitArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answerable: bool = Field(description="false if the knowledge base does not answer the question.")
    answer: str = Field(description="The answer, in the language of the question. Every sentence must be supported "
                                    "by at least one claim. Empty string if not answerable.")
    claims: list[ClaimArg] = Field(description="The evidence for the answer. Empty if not answerable.")
    missing: str = Field(description="What the knowledge base does not cover; empty string if nothing is missing. "
                                     "Required whenever answerable is false.")


SUBMIT_TOOL = {"type": "function", "function": {
    "name": "submit_answer",
    "description": "Submit the final answer and finish the investigation. Each claim is checked automatically: "
                   "its quote must appear word for word in the cited clause, and only clauses returned by the tools "
                   "in this investigation can be cited. If some claims fail, the result lists them and you may "
                   "correct and resubmit once.",
    "parameters": inline_schema(SubmitArgs.model_json_schema())}}


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------
AGENT_SYSTEM = """You are a compliance research assistant for the AML team of a bank in Singapore.
You research a question with tools over a knowledge base of Singapore (MAS, STRO) rules and international (FATF)
material, then submit an answer in which every statement is supported by an exact quote.
The investigation date is fixed by the system; every tool returns the versions in force on that date.

How to research:
1. {start_rule}
2. If a passage you need refers to another clause (for example "paragraph X of the Notice", a footnote, a case study),
   fetch that clause with get_clause instead of searching for it.
3. If the question is not fully answered, search again with a query that describes what is missing, phrased the way
   the regulation would state it. Passages you have already seen are skipped.
4. Stop when you can answer the question, or when searches for the missing part bring nothing relevant.
   You have at most {max_calls} tool calls besides submit_answer. Fewer is better when the answer is already there.
5. Use only text returned by the tools. Do not use outside knowledge, even if you know the answer.

How to answer (call submit_answer exactly once, or twice if the first submission fails verification):
6. Each claim's quote must be copied character for character from the clause given in "ref".
7. Scope: if the question is about another jurisdiction, a different kind of institution than the sources address,
   or information the knowledge base does not contain, set "answerable" to false. Do not answer a different question
   on a related topic.
8. Premises: if the question assumes something the sources contradict (for example an outdated rule, or a different
   starting point for a deadline), answer from the sources and state the difference explicitly.
9. If only part of the question is answered, answer that part and state the rest in "missing".
   When "answerable" is false you may still add claims that show what the sources do say about the topic.
   Never put quotations of the sources in "missing"; anything quoted must be a claim, so that it is verified.
10. Write "answer" in the same language as the question. Quotes stay exactly as in the sources.
11. Text inside quotation marks in "answer" is checked like a claim: it must be copied exactly from a clause you have
    seen. If you are paraphrasing, do not use quotation marks."""

AGENT_USER = """Investigation date: {as_of}
Question: {question}

First search results (query = the question):
{seed}"""

AGENT_USER_NOSEED = """Investigation date: {as_of}
Question: {question}

No search has been run yet."""

START_SEED = "You start with the results of a first search for the question. Read them before calling any tool."
START_NOSEED = ("No search has been run yet. Begin with search_regulations, phrasing the query the way the "
                "regulation would state the requirement.")

NUDGE = "Continue the investigation: call a tool, or call submit_answer to finish."


# ---------------------------------------------------------------------------
# 证据池 → R7 核验所需的"来源"
# ---------------------------------------------------------------------------
def evidence_sources(ctx: ToolContext) -> list[dict]:
    """每条看过的条款作为一个来源（sid = ref），这样 R7 的 verify 可以原样复用：
    引文不在所标条款、但恰好在另一条看过的条款里 → 自动更正出处（D56 的规则）。"""
    return [{"sid": ref, "chunk_id": ref, "doc_id": e["doc_id"], "title": e["title"], "version": e["version"],
             "units": [(e["unit_id"], e["text"])]} for ref, e in ctx.evidence.items()]


def check_submission(ctx: ToolContext, data: SubmitArgs) -> list[dict]:
    claims = [{"statement": c.statement, "source": c.ref, "unit": c.ref.split(":", 1)[-1], "quote": c.quote}
              for c in data.claims]
    return verify(claims, evidence_sources(ctx))


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------
def research(question: str, as_of: date | None = None, use_scope: bool = True, use_cache: bool = True,
             log: bool = True, max_tool_calls: int = MAX_TOOL_CALLS, seed: bool = True) -> dict:
    """seed=False（R8c 的对照）：不做系统发起的首轮检索，由模型自己写第一次查询。"""
    as_of = as_of or date.today()
    t0 = time.perf_counter()
    ctx = ToolContext(question, as_of, kind="research", max_calls=max_tool_calls, log=log).start()
    rec = {"question": question, "as_of": str(as_of), "prompt_version": AGENT_VERSION, "model": llm.MODEL,
           "seed": seed, "run_id": ctx.run_id, "usage": new_usage(), "ms": {}, "trace": []}

    def add(u):
        add_usage(rec["usage"], u)

    def finish(status, reason=None):
        rec.update(status=status, abstain_reason=reason)
        rec["sources"] = [{"sid": s["sid"], "chunk_id": s["chunk_id"], "doc_id": s["doc_id"], "title": s["title"],
                           "version": s["version"], "route": "agent", "rerank": None,
                           "units": [u for u, _ in s["units"]]} for s in evidence_sources(ctx)]
        rec["tool_calls"] = {t: sum(x["tool"] == t for x in rec["trace"])
                             for t in ("search_regulations", "get_clause", "submit_answer")}
        rec["ms"]["total"] = round((time.perf_counter() - t0) * 1000)
        rec["cost_usd"] = round(llm.cost_usd(rec["usage"]), 5)
        ctx.finish(status, {"reason": reason, "tool_calls": rec["tool_calls"], "turns": rec.get("turns"),
                            "cost_usd": rec["cost_usd"]})
        if log:
            with connect() as conn, conn.cursor() as cur:
                cur.execute("""INSERT INTO kb_answer_log (question, as_of, status, abstain_reason, prompt_version,
                                                          model, record) VALUES (%s, %s, %s, %s, %s, %s, %s)
                               RETURNING answer_id""",
                            (question, as_of, status, reason, AGENT_VERSION, llm.MODEL,
                             json.dumps(rec, ensure_ascii=False, default=str)))
                # R11c：复核界面要把这条回答挂到案件上（review_reg_queries），所以返回日志编号。
                # 只在写完日志之后加到返回值里，日志里保存的记录和发给模型的内容都不变
                rec["answer_id"] = cur.fetchone()[0]
                conn.commit()
        return rec

    # 1. 范围检查与首轮检索并行
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_seed = pool.submit(seed_search, ctx, question) if seed else None
        f_scope = pool.submit(scope_check, question, use_cache) if use_scope else None
        if f_scope is not None:
            try:
                scope, u = f_scope.result()
                add(u)
                rec["scope"] = scope
            except llm.LLMError as e:
                if e.usage:
                    add(e.usage)
                rec["scope"], scope = {"error": str(e), "in_scope": True}, {"in_scope": True}   # 失败时放行（D54）
            if not scope["in_scope"]:
                if f_seed is not None:
                    f_seed.result()
                rec |= {"answer": "", "claims": [], "turns": 0,
                        "missing": (f"超出知识库范围：问题涉及 {', '.join(scope['outside'])}；"
                                    "知识库只包含新加坡（MAS、STRO）与国际（FATF）材料。")}
                return finish("abstained", "out_of_scope")
        seed_res = f_seed.result() if f_seed is not None else None

    # 2. 循环
    tools = tool_schemas() + [SUBMIT_TOOL]
    system = AGENT_SYSTEM.format(max_calls=max_tool_calls, start_rule=START_SEED if seed else START_NOSEED)
    user = (AGENT_USER.format(as_of=as_of, question=question, seed=json.dumps(seed_res, ensure_ascii=False, indent=1))
            if seed else AGENT_USER_NOSEED.format(as_of=as_of, question=question))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def run_tool(name: str, raw_args: str) -> dict:
        result = call_tool(ctx, name, raw_args)
        rec["trace"].append({"step": ctx.step, "tool": name, "args": raw_args, "error": result.get("error"),
                             "refs": [x for r in result.get("results", [])
                                      for x in r.get("refs", [r.get("ref")]) if x]})
        return result

    # 循环本身在 agent_core.run_loop（R9b-1 抽出，与分诊 Agent 共用）
    try:
        out = run_loop(messages, tools, run_tool=run_tool, submit_name="submit_answer",
                       on_submit=lambda raw, n: _submit(ctx, rec, raw, n), usage=rec["usage"], use_cache=use_cache,
                       max_turns=MAX_TURNS, max_nudges=MAX_NUDGES, nudge=NUDGE)
    except llm.LLMError as e:                       # 失败调用的用量已在 run_loop 里记入
        if e.raw:
            rec["raw_failed"] = e.raw[:8000]
        rec |= {"answer": "", "claims": [], "missing": f"生成失败：{e}", "turns": getattr(e, "turn", 0)}
        return finish("abstained", "generation_error")

    rec["turns"] = out.turns
    if out.final is None:
        rec |= {"answer": "", "claims": [], "missing": f"在 {out.turns} 轮内没有提交回答（提醒 {out.nudges} 次）"}
        return finish("abstained", "loop_limit")        # 兜底拒答（D58 的分类），单独命名便于统计

    # 3. 判定状态（与 R7 相同的规则）
    data, checked = out.final
    rec["answer"], rec["missing"], rec["claims"] = data.answer, data.missing, checked
    if not data.answerable:
        # D64 修改 D56 的判定：拒答时附带说明缺口的论断是允许的（v1 q079 那样的诚实拒答）。
        # 只有"回答栏有内容"或"有论断却没写缺什么"才算自相矛盾（v1 q024、q072 的特征是 missing 为空）
        if data.answer.strip() or (any(c["verified"] for c in checked) and not data.missing.strip()):
            rec["answerable_conflict"] = True          # 标记供复核，仍按拒答处理
        return finish("abstained", "insufficient")
    verified = [c for c in checked if c["verified"]]
    if not verified:
        return finish("abstained", "unverifiable")
    return finish("partial" if len(verified) < len(checked) else "answered")


def _unquote(text: str, bad: list[str]) -> str:
    pieces = [b.split(": ", 1)[1] for b in bad]

    def fix(m):
        inner = m.group(1)
        return inner if any(p[:60] in inner for p in pieces) else m.group(0)
    for rx in _PAIRS:
        text = rx.sub(fix, text)
    return text


def _submit(ctx: ToolContext, rec: dict, raw_args: str, submits: int):
    """处理一次提交。返回 (交给模型的结果, 最终结果或 None)。"""
    ctx.step += 1
    t0 = time.perf_counter()
    try:
        data = SubmitArgs.model_validate(json.loads(raw_args))
    except (json.JSONDecodeError, ValidationError) as e:
        msg = e.msg if isinstance(e, json.JSONDecodeError) else "; ".join(
            f"{'.'.join(map(str, d['loc']))}: {d['msg']}" for d in e.errors())
        result = {"status": "invalid", "error": f"submission does not follow the required format: {msg}"}
        _log(ctx, "submit_answer", {"_raw": raw_args[:2000]}, result, "invalid_args", t0)
        rec["trace"].append({"step": ctx.step, "tool": "submit_answer", "error": "invalid_args"})
        # 格式错误：第二次也不对时，以空回答结束，记为生成失败由上层判定
        if submits + 1 >= MAX_SUBMITS:
            raise llm.LLMError(f"提交格式两次不正确：{msg}")
        return result, None
    checked = check_submission(ctx, data)
    failed = [(i, c) for i, c in enumerate(checked, 1) if not c["verified"]]
    # D65：正文和"未覆盖"里加了引号的原文也要能逐字找到（dev2 的 v225：把转述放进了引号）
    bad_quotes = unverified_quotes({"answer": data.answer, "missing": data.missing},
                                   [e["text"] for e in ctx.evidence.values()])
    if submits == 0:
        rec["first_pass"] = {"claims": len(checked), "verified": len(checked) - len(failed),
                             "bad_quotes": len(bad_quotes)}
    report = {"status": "accepted" if not (failed or bad_quotes) else "rejected",
              "verified": len(checked) - len(failed), "failed": [
                  {"claim": i, "ref": c["source"], "problem": c["status"]} for i, c in failed] + [
                  {"text": b, "problem": "quoted_text_not_verbatim"} for b in bad_quotes]}
    done = not ((data.answerable and failed or bad_quotes) and submits + 1 < MAX_SUBMITS)
    if done and bad_quotes:
        # 最后一次提交仍有对不上的引号：去掉这些引号，使它们不再自称原文，并记录
        data.answer, data.missing = _unquote(data.answer, bad_quotes), _unquote(data.missing, bad_quotes)
        rec["unquoted"] = bad_quotes
        if not failed:
            report["status"] = "accepted_unquoted"
    if not done:
        report["instruction"] = ("quote_not_found means the quote is not an exact substring of the cited clause; "
                                 "quoted_text_not_verbatim means text inside quotation marks in the answer or missing "
                                 "is not an exact copy of any clause you have seen: copy it exactly or remove the "
                                 "quotation marks; "
                                 "bad_source means the ref was not returned by any tool in this investigation. "
                                 "Fix the failed claims (copy the words exactly, or fetch the clause first) or "
                                 "remove them together with the sentences they support, then call submit_answer again.")
    _log(ctx, "submit_answer", data.model_dump(), report, None if not failed else "claims_failed", t0)
    rec["trace"].append({"step": ctx.step, "tool": "submit_answer", "status": report["status"],
                         "failed": len(failed)})
    if submits > 0:
        rec["retried"] = True
    return report, ((data, checked) if done else None)


# ---------------------------------------------------------------------------
# 对照：固定两轮（R8c）
# ---------------------------------------------------------------------------
EXTRA_SOURCES = 4      # 第二轮最多补入的新段落


def two_round(question: str, as_of: date | None = None, use_cache: bool = True, log: bool = True) -> dict:
    """
    规则写死的两轮检索，用来回答"Agent 的好处是否只来自'再查一次'"：
      第 1 轮 = R7 原样（answer()）。如果"未覆盖"不为空（且不是范围外拒答），
      就用"未覆盖"作为查询再检索一次，补入最多 4 段没看过的，与第 1 轮的 8 段一起重新生成（R7 的提示词不变）。
    与 Agent 的区别：查什么、查几次、要不要按编号取条款，全部由规则决定，模型只负责写回答。
    """
    from kb_answer import answer
    from kb_search import retrieve
    as_of = as_of or date.today()
    rec1 = answer(question, as_of, use_cache=use_cache, log=False)
    missing = (rec1.get("missing") or "").strip()
    if rec1.get("abstain_reason") == "out_of_scope" or not missing:
        rec1["rounds"] = 1
        return rec1
    hits1 = retrieve(question, as_of, 8)                 # 与第 1 轮完全相同（命中缓存）
    seen = {h["chunk_id"] for h in hits1}
    extra = [h for h in retrieve(question, as_of, k=8 + len(seen), query=missing[:400])
             if h["chunk_id"] not in seen][:EXTRA_SOURCES]
    if not extra:
        rec1["rounds"] = 1
        return rec1
    rec2 = answer(question, as_of, use_cache=use_cache, log=log, hits=hits1 + extra)
    for k in rec2["usage"]:
        rec2["usage"][k] += rec1["usage"].get(k, 0)
    rec2["cost_usd"] = round(rec1["cost_usd"] + rec2["cost_usd"], 5)
    rec2["ms"]["total"] = rec1["ms"]["total"] + rec2["ms"]["total"]
    rec2["rounds"] = 2
    rec2["first_round"] = {"status": rec1["status"], "abstain_reason": rec1.get("abstain_reason"),
                           "missing": missing[:400], "extra": [h["chunk_id"] for h in extra]}
    return rec2


# ---------------------------------------------------------------------------
# 命令行
# ---------------------------------------------------------------------------
def print_trace(rec: dict) -> None:
    print(f"  轮数 {rec.get('turns')}，工具调用 {rec.get('tool_calls')}，证据池 {len(rec.get('sources', []))} 条")
    for t in rec["trace"]:
        if t["tool"] == "submit_answer":
            print(f"    #{t['step']} submit_answer → {t.get('status') or t.get('error')}"
                  + (f"（{t['failed']} 条未通过）" if t.get("failed") else ""))
        else:
            args = t["args"] if len(t["args"]) < 160 else t["args"][:160] + "…"
            print(f"    #{t['step']} {t['tool']} {args}")
            print(f"        → {t['error'] or ', '.join(t['refs'][:8]) or '（无新结果）'}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="法规研究 Agent")
    ap.add_argument("question", nargs="*")
    ap.add_argument("--as-of", type=date.fromisoformat, default=None)
    ap.add_argument("--qids", nargs="+", help="开发集题号（测试集冻结，不允许）")
    ap.add_argument("--no-cache", action="store_true")
    a = ap.parse_args()
    if a.qids:
        qs = {q["id"]: q for q in json.loads((PROJECT_ROOT / "reports" / "rag_golden_resolved.json")
                                             .read_text(encoding="utf-8"))}
        for qid in a.qids:
            q = qs[qid]
            if q["split"] != "dev":
                raise SystemExit(f"{qid} 属于测试集：测试集冻结，不用来调试")
            print("=" * 100)
            print(f"{qid} [{q['type']}/{q['lang']}] 调查日期 {q['as_of']}\n问：{q['question']}")
            print(f"gold：{'；'.join(g['family'] + ':' + g['unit_id'] for g in q['gold']) or '（无答案题）'}")
            rec = research(q["question"], date.fromisoformat(q["as_of"]), use_cache=not a.no_cache)
            print_trace(rec)
            gold = {g["family"] + ":" + g["unit_id"] for g in q["gold"]}
            cited = {c["source"] for c in rec["claims"] if c.get("verified")}
            print(f"  gold 被引用：{sorted(gold & cited) or '无'}；看过但没引用：{sorted(gold - cited & {s['sid'] for s in rec['sources']}) or '无'}")
            print(render(rec))
    elif a.question:
        rec = research(" ".join(a.question), a.as_of, use_cache=not a.no_cache)
        print_trace(rec)
        print(render(rec))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()