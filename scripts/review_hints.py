"""
SentryAML - R11b：复核时的解读提示（确定性规则）
================================================
R10d 的 test 上，程序检查全部通过的理由里，第二位评分者仍读出 5 句"数字全对、意思错"的句子（D84），集中在三类。
这里用代码检查证据，在复核界面上把这三类情况标出来，提醒复核员这些地方要特别读：

  clearing        证据里出现的账户是清算层（inv_is_clearing，与工具拒绝、草稿标注用的是同一个定义，D69）。
                  清算层账户的交易量反映的是支付处理，不能写成"资金扩散""分散转出"。
                  句子触发条件：用了"扩散、分散、流回经济体系"这类解读性的词——不看数字和账户号（v2，D86：
                  第一版按"引用了清算层账户的数字"触发，在 R10d test 上把 6 句如实陈述交易量的句子标了，
                  却漏掉了"That is a heavy one-directional dispersal of funds."——错在解读的词，不在数字）
  shared_out      过桥结果里，同一笔转出被两笔以上的转入分别匹配（inv_pass_through 按转入逐笔找金额相近的转出）。
                  这几对共用同一笔转出，金额不能相加，也不能说成"一起转出"
  regular_repeat  付款方在告警之前已经向收款方付过同样金额的款（至少 1 笔，平均金额与本笔相同 ±1%）。
                  重复的同额付款指向既有的往来关系（D74 的分诊规则），不能当作可疑理由；只有 1 笔时如实说明
                  "不足以证明规律性"。v3（D86）：第一版要求至少 2 笔，在 R10d test 上一次都没触发——
                  评分者说的"repeat identical payment"在数据里是"此前 1 笔、金额完全相同"（--dump 880 3502）

为什么用规则而不是再让模型判断一遍：证据是确定的，能用代码判断就不交给模型；而且出错的恰恰是模型对这些字段的理解，
让同类模型再判一遍，很可能错在同一个地方。规则可以写测试，同样的输入永远给出同样的提示。

这些提示**只在复核时显示**，不改动起草理由的提示词（R10d 的 test 结论已冻结，D81）。

两个层面：
  case_hints(cur, results)          整个案件有哪些需要特别解读的情况（证据区和草稿区的顶部显示）
  sentence_view(text, results, ...)  理由逐句：每句的数字出处 + 这句话触发了哪些提示
句子"触发"一条提示（页面上给这句话加底色和"提示 N"标签）的条件，每类不同：
  clearing        用了解读性的词（扩散、分散、流回经济体系……）
  shared_out      引用了共用那笔转出的金额（按 number_hits 的同一套匹配规则），或写出它的交易编号
  regular_repeat  不逐句标注，只做案件级提示（v4，D86）。按数字触发不行（既往金额就等于告警金额、笔数常常是 1）；
                  按词触发也不行：v3 在 test 上误标 3 句、在 dev 上新增误标 2 句——"The same pair had moved the
                  same amount once before"是如实陈述，"repeated small credits"说的是别的账户；真正说过头的句子
                  是把重复付款"当作可疑的理由"。区别在这句话拿这个事实来干什么，规则分辨不了，只能交给读的人
"""

from __future__ import annotations

import re

from str_draft import ACCT_RE
from str_narrative import flatten, number_hits, number_sources

# 清算层：把交易量"解读"成资金去向的词。如实写出笔数、对手方数量不算问题，用这些词解读才算（v2）
DISPERSAL_WORDS = re.compile(r"(?i)\b(dispers\w*|spread\w*|scatter\w*|distribut\w*|circulat\w*|economy|"
                             r"fann?(?:ed|ing)? out|placing funds|layer(?:ed|ing)? (?:out|away)|high-volume outbound)\b")
# 过桥匹配里触发提示的字段：只看共用的那笔转出的金额。错误是把共用的转出说成"一起转出"；
# 只写"收到两笔 11,376.97"是如实描述收款，不该标（沙盒里的页面截图暴露了这一点，D86）。
# lag_hours、diff_pct 这类数字太常见，也不作为触发依据
SHARED_FIELDS = ("out_amt",)


def _accounts_in(results: dict[int, dict]) -> set[str]:
    out: set[str] = set()
    for res in results.values():
        if isinstance(res.get("account"), str) and ACCT_RE.match(res["account"]):
            out.add(res["account"])
        for _, v in flatten(res):
            if isinstance(v, str) and ACCT_RE.match(v):
                out.add(v)
    return out


def _clearing(cur, accounts: set[str]) -> set[str]:
    if not accounts:
        return set()
    cur.execute("""SELECT a FROM unnest(%s::text[]) a
                   WHERE inv_is_clearing(split_part(a, ':', 1)::int, split_part(a, ':', 2))""", (sorted(accounts),))
    return {r[0] for r in cur.fetchall()}


def _fmt(x) -> str:
    return f"{x:,.2f}" if isinstance(x, float) else f"{x:,}" if isinstance(x, int) else str(x)


def case_hints(cur, results: dict[int, dict]) -> list[dict]:
    """返回提示列表。每条：kind、text（给复核员看的中文说明）、refs（涉及的 (step, path)）、
    accounts、txns（句子里直接写出就算触发）、words（配合 accounts 使用的正则名）。"""
    hints: list[dict] = []

    # 1. 清算层：句子靠解读性的词触发，不看数字（refs 为空）
    for acct in sorted(_clearing(cur, _accounts_in(results))):
        hints.append({"kind": "clearing", "accounts": [acct], "txns": [], "refs": set(), "words": "dispersal",
                      "text": f"{acct} 是清算层账户（历史对手方超过 200 个，D17）。它的交易笔数和对手方数量反映的是"
                              "支付处理，不能写成“资金扩散”“分散转出”或“放回经济体系”。"})

    # 2. 过桥：同一笔转出被多笔转入匹配
    for s, res in results.items():
        matches = res.get("matches")
        if not isinstance(matches, list):
            continue
        groups: dict[object, list[int]] = {}
        for i, m in enumerate(matches):
            if isinstance(m, dict) and m.get("out_txn") is not None:
                groups.setdefault(m["out_txn"], []).append(i)
        for out_txn, idx in groups.items():
            if len(idx) < 2:
                continue
            ins = [matches[i] for i in idx]
            refs = {(s, f"matches[{i}].{k}") for i in idx for k in SHARED_FIELDS}
            hints.append({
                "kind": "shared_out", "accounts": [], "words": None, "refs": refs,
                "txns": [str(out_txn)],
                "text": f"第 {s} 步过桥结果里，转出 {out_txn}（{_fmt(ins[0].get('out_amt'))} USD）同时匹配了 {len(idx)} 笔"
                        f"转入（{'、'.join(_fmt(m.get('in_amt')) for m in ins)} USD）。过桥检查按转入逐笔找金额相近的转出，"
                        "同一笔转出可以被重复匹配：这几对的金额不能相加，也不能说成“一起转出”。"})

    # 3. 规律性的同额重复付款
    alert = (results.get(0) or {}).get("result") or {}
    amt0 = alert.get("amount_usd")
    if isinstance(amt0, (int, float)) and amt0 > 0:
        for s, res in results.items():
            leaves = dict(flatten(res))
            for p, n in leaves.items():
                if not (p == "a_to_b.n" or p.endswith(".a_to_b.n")):
                    continue
                base = p[:-2]
                total = leaves.get(f"{base}.amt")
                if not isinstance(n, int) or n < 1 or not isinstance(total, (int, float)):
                    continue
                avg = total / n
                if abs(avg - amt0) > 0.01 * amt0:
                    continue
                if n == 1:
                    text = (f"第 {s} 步：告警之前付款方已向收款方付过 1 笔 {_fmt(total)} USD，与本笔金额相同。重复的同额付款"
                            "指向既有的往来关系（D74 的分诊规则），不能当作可疑理由；只有 1 笔，也不足以证明规律性。")
                else:
                    text = (f"第 {s} 步：告警之前付款方已向收款方付过 {n} 笔、合计 {_fmt(total)} USD，平均 {avg:,.2f} USD，"
                            f"与本笔 {_fmt(amt0)} USD 相同。重复的同额付款是规律性往来的证据（D74 的分诊规则），"
                            "不能当作可疑理由。")
                hints.append({"kind": "regular_repeat", "accounts": [], "txns": [], "words": "repeat",
                              "refs": set(), "text": text})
    return hints


def triggered(sentence: str, hits: list[tuple[int, str]], hint: dict) -> bool:
    """这句话是否触发这条提示。hits：这句话里所有数字可能的出处 (step, path)。"""
    if hint["kind"] == "clearing":
        return bool(DISPERSAL_WORDS.search(sentence))
    if hint["kind"] == "regular_repeat":                 # 只做案件级提示（见文件开头）
        return False
    if hint["refs"] & set(hits):
        return True
    if any(t and t in sentence for t in hint["txns"]):
        return True
    return False


def sentences(text: str) -> list[str]:
    """与 R10d 评测切句的规则相同（r10d_eval.sentences）：句号、问号、感叹号后面跟空白才切；金额里的小数点不切。"""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]


def sentence_view(text: str, results: dict[int, dict], hints: list[dict], cited=()) -> list[dict]:
    """理由逐句：n、text、numbers（每个数字的出处，引用的事实排在前面）、hints（触发的提示下标）。"""
    out = []
    for i, s in enumerate(sentences(text), 1):
        hits = [h for _, hs in number_hits(s, results) for h in hs]
        out.append({"n": i, "text": s, "numbers": number_sources(s, results, cited=cited),
                    "hints": [k for k, h in enumerate(hints) if triggered(s, hits, h)]})
    return out


def public(h: dict) -> dict:
    """给页面和接口的形式（refs 是内部用的集合，不输出）。"""
    return {"kind": h["kind"], "text": h["text"], "accounts": h["accounts"]}


def case_aids(cur, case_id: int) -> dict:
    """复核界面要的全部辅助信息：案件级提示 +（有草稿时）最新草稿、数据库检查结果、理由逐句对照。
    证据范围与模型起草理由时看到的完全一致（str_narrative.evidence_results）：有草稿时用草稿里保存的事实，
    没有草稿时用 Agent 核验通过的事实；T1 没有 Agent 运行，只有告警本身（第 0 步）。"""
    from str_draft import latest, load_case
    from str_narrative import evidence_results
    case = load_case(cur, case_id)
    cur.execute("SELECT max(version) FROM str_drafts WHERE case_id = %s", (case_id,))
    has_draft = cur.fetchone()[0] is not None
    d = latest(cur, case_id) if has_draft else None
    facts = d["body"].get("evidence_facts", []) if d else ((case["agent_result"] or {}).get("facts") or [])
    results = evidence_results(cur, case, facts)
    hints = case_hints(cur, results)
    out = {"hints": [public(h) for h in hints], "draft": None}
    if d:
        b = d["body"]
        r = b["reasons"]
        cur.execute("SELECT problem, detail FROM str_draft_problems(%s, %s, %s)",
                    (case_id, psycopg_json(b), d["created_at"]))
        problems = [{"problem": p, "detail": det} for p, det in cur.fetchall()]
        cited = {(f["step"], f["path"]) for f in (r.get("narrative_facts") or [])}
        out["draft"] = {"version": d["version"], "created_by": d["created_by"], "generator": d["generator"],
                        "created_at": d["created_at"], "sha256": d["sha"], "body": b, "problems": problems,
                        "has_narrative": bool((r.get("narrative") or "").strip()),
                        "sentences": sentence_view(r.get("narrative") or "", results, hints, cited)}
    return out


def psycopg_json(obj):
    from psycopg.types.json import Jsonb
    return Jsonb(obj)