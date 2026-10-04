"""
SentryAML - RAG 层 R7：带引用的回答、引用核验与拒答
====================================================
answer(question, as_of) 的四步：

  1. 范围检查   大模型从问题里抽取限定条件（法域、机构类型、时间），由规则判断是否在知识库范围内。
                知识库只覆盖新加坡（MAS、STRO）和国际（FATF 等）材料；问题指向其他法域时直接拒答，
                不再检索。R6 的数据表明不能靠相关度分数拒答：马来西亚那道题的重排分数是 0.955。
                只按法域拦截：机构类型的覆盖是混合的（Notice 626 只管银行，STR 指南适用于所有报送机构），
                规则判断不了，交给第 4 步。
  2. 带引用生成 检索最终配置（kb_search.retrieve）取 8 个来源，按条款逐段标注 ⟦条款编号⟧ 交给大模型；
                要求 JSON 输出：回答 + 每条论断的（来源、条款、逐字引文）。
  3. 引用核验   确定性检查，不调用大模型：引文必须逐字出现在所标条款的原文里（只归一化引号、空白、
                脚注标记）。条款标错、但引文确实在同一来源的另一条里 → 自动更正并记录；
                完全找不到 → 告诉模型哪条不对，重写一次；仍不通过 → 丢弃该论断。
  4. 拒答       超出范围 / 模型判断资料不足 / 没有一条论断通过核验，都拒答，并给出原因。

为什么要逐字引文：只标 [1][2] 的引用无法检查"这句话是否真的由这个来源支撑"；
有了逐字引文，程序就能核对。与 R3 核对手法图谱引文、R4 核对答案要点是同一思路。

用法：
    python scripts/kb_answer.py "银行确立怀疑后几天内必须提交 STR？"
    python scripts/kb_answer.py "Within how many days must a bank file an STR?" --as-of 2025-03-01
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import llm
from db import connect
from kb_search import retrieve

PROMPT_VERSION = "ans-v2"        # v2：前提纠正规则、拒答必须说明原因（D55）
K_SOURCES = 8
IN_SCOPE = {"singapore", "international", "global", "worldwide", "fatf"}
MIN_QUOTE_WORDS, MAX_QUOTE_WORDS = 5, 80

# ---------------------------------------------------------------------------
# 第 1 步：范围检查
# ---------------------------------------------------------------------------
SCOPE_SYSTEM = "You extract the scope constraints of anti-money-laundering compliance questions. Return JSON only."
SCOPE_USER = """Question: {question}

Return JSON:
{{"jurisdictions": [...], "institution_type": "... or null", "time_reference": "... or null"}}

- "jurisdictions": the countries or regions whose rules or facts the question is about, as English country names.
  Map regulators and laws to their country (MAS, STRO, CDSA -> "Singapore"; Bank Negara -> "Malaysia";
  HKMA -> "Hong Kong"; FinCEN -> "United States"). Use "international" for FATF or general typology questions.
  Return an empty list if the question does not name or clearly imply any jurisdiction.
  Do not assume Singapore when it is not stated.
- "institution_type": the kind of institution the question is about (e.g. "bank", "casino operator"), or null.
- "time_reference": any specific date, year or period the question refers to, or null."""


def scope_check(question: str, use_cache: bool = True) -> tuple[dict, dict]:
    data, usage = llm.chat_json([{"role": "system", "content": SCOPE_SYSTEM},
                                 {"role": "user", "content": SCOPE_USER.format(question=question)}],
                                purpose="scope", max_tokens=3000, use_cache=use_cache)
    juris = [str(j).strip() for j in data.get("jurisdictions") or [] if str(j).strip()]
    outside = [j for j in juris if j.lower() not in IN_SCOPE]
    return {"jurisdictions": juris, "institution_type": data.get("institution_type"),
            "time_reference": data.get("time_reference"), "in_scope": not outside, "outside": outside}, usage


# ---------------------------------------------------------------------------
# 第 2 步：来源与生成
# ---------------------------------------------------------------------------
ANSWER_SYSTEM = ("You are a compliance research assistant for the AML team of a bank in Singapore. "
                 "You answer strictly from the sources provided and cite every statement with an exact quote. "
                 "Return JSON only.")

ANSWER_USER = """Investigation date: {as_of}
Sources (the versions in force on the investigation date; each unit is marked ⟦unit id⟧):

{sources}

Question: {question}

Return JSON:
{{"answerable": true or false,
  "answer": "the answer, with [S#] markers after each sentence; empty string if not answerable",
  "claims": [{{"statement": "...", "source": "S#", "unit": "unit id", "quote": "exact words from that unit"}}],
  "missing": "what the sources do not cover, or empty string"}}

Rules:
1. Use only the sources. Do not use outside knowledge, even if you know the answer.
2. Every sentence of "answer" must be supported by at least one claim. Each claim's "quote" must be copied
   character for character from the text of the cited unit: one contiguous span of {min_w} to {max_w} words,
   no ellipsis, no paraphrase, no added words.
3. Scope: the sources cover Singapore (MAS, STRO) rules and international (FATF) material only. If the question is
   about another jurisdiction, a different kind of institution than the sources address, or information the sources
   do not contain, set "answerable" to false. Do not answer a different question on a related topic.
4. Premises: if the question assumes something the sources do not support or contradict (for example an outdated
   rule, or a different starting point for a deadline), answer from the sources and state the difference explicitly.
   Do not refuse only because the wording of the question does not match the wording of the sources.
5. If the sources answer only part of the question, answer that part and state the rest in "missing".
   Whenever "answerable" is false, "missing" must say what is missing.
6. Write "answer" in the same language as the question. Quotes stay exactly as in the sources."""


def build_sources(hits: list[dict]) -> list[dict]:
    """把检索结果整理成来源：按条款展开原文（长条款被切成多块时，用完整条款），去掉重复。"""
    doc_ids = list({h["doc_id"] for h in hits})
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, title, version_label FROM kb_documents WHERE doc_id = ANY(%s)", (doc_ids,))
        docs = {d: (t, v) for d, t, v in cur.fetchall()}
        cur.execute("SELECT doc_id, unit_id, text FROM kb_units WHERE doc_id = ANY(%s)", (doc_ids,))
        texts = {(d, u): t for d, u, t in cur.fetchall()}
    sources, seen = [], set()
    for h in hits:
        key = (h["doc_id"], tuple(h["unit_ids"]))
        if key in seen:
            continue
        seen.add(key)
        title, version = docs[h["doc_id"]]
        units = [(u, texts[(h["doc_id"], u)]) for u in h["unit_ids"] if (h["doc_id"], u) in texts]
        sources.append({"sid": f"S{len(sources) + 1}", "chunk_id": h["chunk_id"], "doc_id": h["doc_id"],
                        "title": title, "version": version, "heading": " > ".join(h["heading_path"]),
                        "units": units, "route": h["route"], "rerank": h["score"]})
    return sources


def format_sources(sources: list[dict]) -> str:
    out = []
    for s in sources:
        body = "\n".join(f"⟦{u}⟧ {t}" for u, t in s["units"])
        out.append(f"[{s['sid']}] {s['title']} ({s['version']})\nSection: {s['heading']}\n{body}")
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# 第 3 步：引用核验（确定性）
# ---------------------------------------------------------------------------
def norm(s: str) -> str:
    s = re.sub(r"\[\^\d+\]", "", s)
    s = s.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "―": "-", "–": "-", "—": "-"}))
    return re.sub(r"\s+", " ", s).strip().lower()


def verify(claims: list[dict], sources: list[dict]) -> list[dict]:
    by_sid = {s["sid"]: s for s in sources}
    out = []
    for c in claims:
        c = {k: str(c.get(k, "") or "").strip() for k in ("statement", "source", "unit", "quote")}
        src = by_sid.get(c["source"].strip("[]"))
        q = norm(c["quote"])
        n_words = len(q.split())
        status = "ok"
        if src is None:
            status = "bad_source"
        elif "..." in c["quote"] or "…" in c["quote"]:
            status = "ellipsis"
        elif n_words < MIN_QUOTE_WORDS:
            status = "too_short"
        else:
            units = dict(src["units"])
            found = [u for u, t in src["units"] if q in norm(t)]
            if len(found) > 1:
                c["also_in"] = [u for u in found if u != c["unit"]]     # 同一句话出现在多条里，记录下来供人工复核
            if c["unit"] in units and q in norm(units[c["unit"]]):
                status = "ok"
            elif found:
                c["unit_given"], c["unit"], status = c["unit"], found[0], "ok_corrected_unit"
            else:
                # 来源编号标错（D56）：引文逐字存在于另一个来源里。只有恰好一个来源包含它时才更正，
                # 多个来源都有时无法判断模型指的是哪一条，按"找不到"处理
                others = [(s2, u) for s2 in sources if s2 is not src
                          for u, t in s2["units"] if q in norm(t)]
                if len({s2["sid"] for s2, _ in others}) == 1:
                    c["source_given"], c["unit_given"] = c["source"], c["unit"]
                    src, c["source"], c["unit"] = others[0][0], others[0][0]["sid"], others[0][1]
                    status = "ok_corrected_source"
                else:
                    status = "quote_not_found"
        if src is not None:
            c |= {"chunk_id": src["chunk_id"], "doc_id": src["doc_id"], "title": src["title"],
                  "version": src["version"]}
        out.append(c | {"status": status, "verified": status.startswith("ok")})
    return out


def schema_problem(data: dict) -> str | None:
    """检查生成结果的格式；能安全纠正的（"true" 字符串）就地纠正，否则返回问题描述。"""
    a = data.get("answerable")
    if isinstance(a, str) and a.strip().lower() in ("true", "false"):
        data["answerable"] = a.strip().lower() == "true"
    elif not isinstance(a, bool):
        return '"answerable" must be true or false'
    if not isinstance(data.get("claims", []), list):
        return '"claims" must be a list'
    if data["answerable"] and not data.get("claims"):
        return 'an answerable response must include claims with exact quotes'
    return None


def retry_message(checked: list[dict]) -> str:
    lines = [f"- claim {i + 1} ({c['source']} {c['unit']}): {c['status']}" for i, c in enumerate(checked)
             if not c["verified"]]
    return ("Some claims failed verification:\n" + "\n".join(lines) +
            "\nquote_not_found means the quote is not an exact substring of that unit's text. "
            "Return the complete JSON again. Copy quotes character for character from the cited unit; "
            "remove any claim (and the sentence it supports) that you cannot support with an exact quote.")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def answer(question: str, as_of: date | None = None, use_scope: bool = True, use_cache: bool = True,
           log: bool = True) -> dict:
    as_of = as_of or date.today()
    t0 = time.perf_counter()
    rec = {"question": question, "as_of": str(as_of), "prompt_version": PROMPT_VERSION, "model": llm.MODEL,
           "usage": {"calls": 0, "cached_calls": 0, "prompt": 0, "hit": 0, "completion": 0, "reasoning": 0},
           "ms": {}}

    def add(u):
        rec["usage"]["calls"] += 1
        rec["usage"]["cached_calls"] += int(u.get("cached", False))
        for k in ("prompt", "hit", "completion", "reasoning"):
            rec["usage"][k] += u.get(k, 0)

    def finish(status, reason=None):
        rec.update(status=status, abstain_reason=reason)
        rec["ms"]["total"] = round((time.perf_counter() - t0) * 1000)
        rec["cost_usd"] = round(llm.cost_usd(rec["usage"]), 5)
        if log:
            with connect() as conn, conn.cursor() as cur:
                cur.execute("""INSERT INTO kb_answer_log (question, as_of, status, abstain_reason, prompt_version,
                                                          model, record) VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                            (question, as_of, status, reason, PROMPT_VERSION, llm.MODEL,
                             json.dumps(rec, ensure_ascii=False, default=str)))
                conn.commit()
        return rec

    # 1. 范围检查与检索同时进行（D54）：检索不依赖范围检查的结果，并行可省下约 3 秒；
    #    超出范围时检索结果直接丢弃，浪费的只是一次检索
    def timed(fn, *a):
        t = time.perf_counter()
        r = fn(*a)
        return r, round((time.perf_counter() - t) * 1000)

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_hits = pool.submit(timed, retrieve, question, as_of, K_SOURCES)
        f_scope = pool.submit(timed, scope_check, question, use_cache) if use_scope else None
        if f_scope is not None:
            try:
                (scope, u), rec["ms"]["scope"] = f_scope.result()
                add(u)
                rec["scope"] = scope
            except llm.LLMError as e:
                # 范围检查失败时放行（fail-open）：生成提示词的第 3 条规则仍会拒绝范围外的问题；
                # 反过来"失败即拒答"会误拒本来答得了的问题
                rec["scope"], scope = {"error": str(e), "in_scope": True}, {"in_scope": True}
            if not scope["in_scope"]:
                rec["answer"] = ""
                rec["missing"] = (f"超出知识库范围：问题涉及 {', '.join(scope['outside'])}；"
                                  "知识库只包含新加坡（MAS、STRO）与国际（FATF）材料。")
                rec["claims"], rec["sources"] = [], []
                return finish("abstained", "out_of_scope")
        hits, rec["ms"]["retrieval"] = f_hits.result()

    # 2. 生成
    sources = build_sources(hits)
    rec["sources"] = [{k: s[k] for k in ("sid", "chunk_id", "doc_id", "title", "version", "route", "rerank")}
                      | {"units": [u for u, _ in s["units"]]} for s in sources]
    messages = [{"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": ANSWER_USER.format(as_of=as_of, sources=format_sources(sources),
                                                               question=question, min_w=MIN_QUOTE_WORDS,
                                                               max_w=MAX_QUOTE_WORDS)}]
    t = time.perf_counter()
    try:
        data, u = llm.chat_json(messages, purpose="answer", max_tokens=8000, use_cache=use_cache)
        add(u)
        rec["raw"] = u["raw"][:4000]
        problem = schema_problem(data)
        if problem:
            # 格式不对（例如没有 answerable 字段）时不能当作"拒答"：v1 有两道题因此被静默拒答（D55）
            rec["schema_retry"] = problem
            messages = messages + [{"role": "assistant", "content": u["raw"]},
                                   {"role": "user", "content": f"Your JSON does not follow the required format: "
                                                               f"{problem}. Return the complete JSON again in exactly "
                                                               "the specified format."}]
            data, u = llm.chat_json(messages, purpose="answer_schema_retry", max_tokens=8000, use_cache=use_cache)
            add(u)
            if schema_problem(data):
                raise llm.LLMError(f"格式仍不正确：{schema_problem(data)}")
    except llm.LLMError as e:
        rec |= {"answer": "", "claims": [], "missing": f"生成失败：{e}"}
        return finish("abstained", "generation_error")

    # 3. 核验；不通过则重写一次
    checked = verify(data.get("claims") or [], sources)
    rec["first_pass"] = {"claims": len(checked), "verified": sum(c["verified"] for c in checked)}
    if data.get("answerable") and any(not c["verified"] for c in checked):
        retry = messages + [{"role": "assistant", "content": u["raw"]},
                            {"role": "user", "content": retry_message(checked)}]
        try:
            data2, u2 = llm.chat_json(retry, purpose="answer_retry", max_tokens=8000, use_cache=use_cache)
            add(u2)
            checked2 = verify(data2.get("claims") or [], sources)
            if not schema_problem(data2) and sum(c["verified"] for c in checked2) >= sum(c["verified"] for c in checked):
                data, checked = data2, checked2
            rec["retried"] = True
        except llm.LLMError:
            rec["retried"] = "failed"
    rec["ms"]["generation"] = round((time.perf_counter() - t) * 1000)

    # 4. 拒答判断
    rec["answer"], rec["missing"] = data.get("answer") or "", data.get("missing") or ""
    rec["claims"] = checked
    if not data.get("answerable"):
        if rec["answer"].strip() or any(c["verified"] for c in checked):
            # 模型说"不能回答"，却给出了回答和通过核验的引文（v1 的 q024、q072）。按拒答处理并标记供人工复核：
            # 合规场景里误拒的代价小于答错；而且范围外的问题，引用新加坡条款的引文同样能通过核验，
            # 不能因为引文真实就推翻模型的拒答判断（D56）
            rec["answerable_conflict"] = True
        return finish("abstained", "insufficient")
    verified = [c for c in checked if c["verified"]]
    if not verified:
        return finish("abstained", "unverifiable")
    return finish("partial" if len(verified) < len(checked) else "answered")


def render(rec: dict) -> str:
    lines = []
    if rec["status"] == "abstained":
        lines.append(f"【拒答：{rec['abstain_reason']}】{rec.get('missing', '')}")
    else:
        lines.append(rec["answer"])
        if rec.get("missing"):
            lines.append(f"\n（未覆盖：{rec['missing']}）")
        lines.append("\n依据：")
        for c in rec["claims"]:
            mark = "✓" if c["verified"] else "✗"
            fix = f"（模型标为 {c['unit_given']}，已更正）" if c.get("unit_given") else ""
            fix += f"（同一句也出现在 {', '.join(c['also_in'])}）" if c.get("also_in") else ""
            fix += f"（模型标为来源 {c['source_given']}，已更正）" if c.get("source_given") else ""
            lines.append(f"  {mark} [{c['source']}] {c.get('title', '?')} ({c.get('version', '')}) "
                         f"{c['unit']}{fix}：“{c['quote']}”" + ("" if c["verified"] else f"  ← {c['status']}"))
    u = rec["usage"]
    lines.append(f"\n用时 {rec['ms']}，调用 {u['calls']} 次（缓存 {u['cached_calls']}），约 ${rec['cost_usd']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="带引用的回答")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--as-of", type=date.fromisoformat, default=None)
    a = ap.parse_args()
    print(render(answer(" ".join(a.question), a.as_of)))