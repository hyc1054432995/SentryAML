"""
SentryAML - R8c 第 1 阶段：HyDE 的检索层消融（只在 dev2 上跑）
================================================================
问题：首轮检索用用户原话，还是先让模型写一段"假想的条文"（HyDE）再用它去检索？

三种首轮查询，检索配置完全相同（最终配置 + 按调查日期过滤），只改"拿什么去检索"：
  question   用户原问题（R7 的做法）
  hyde       模型写的假想段落（60–120 词，英文，模仿 MAS 条文的写法）
  both       原问题 + 假想段落拼在一起
三种做法的重排、条款编号路由都对照用户原问题（kb_search 的 rerank_query）：
  "是否回答了问题"要按用户问的判断；假想段落里模型自己编的条款号不能被当成用户点名的编号。

指标：前 8 名（回答实际使用的来源数，D57 的教训）里 gold 条款是否全部出现（Recall@8，严格），
      以及至少一条 gold 出现（Hit@8）。只看检索、不生成回答，所以便宜。

采用规则（D64，运行前写定）：hyde 或 both 相对 question，Recall@8 全中的题至少多 2 道、且变差的不超过 1 道，
  才把它作为端到端比较里的一种做法；否则首轮检索保持用户原话。

用法：
    python scripts/rag_hyde.py            # dev2；测试集不做检索消融
"""

from __future__ import annotations

import json
import time
from datetime import date
from pathlib import Path

import llm
from kb_search import retrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOLVED = PROJECT_ROOT / "reports" / "rag_golden_v2_resolved.json"
K = 8

HYDE_SYSTEM = "You write hypothetical regulatory text used only as a search query. Return JSON only."
HYDE_USER = """Write one passage of 60 to 120 words, in the style of a paragraph from a Singapore MAS AML/CFT Notice or
its Guidelines, that would directly answer the question below. Write in English even if the question is in another
language. Do not cite paragraph numbers.

Question: {question}

Return JSON: {{"passage": "..."}}"""


def hyde_passage(question: str) -> tuple[str, dict]:
    data, u = llm.chat_json([{"role": "system", "content": HYDE_SYSTEM},
                             {"role": "user", "content": HYDE_USER.format(question=question)}],
                            purpose="hyde", max_tokens=3000)
    return str(data.get("passage", "")).strip(), u


def gold_found(q: dict, hits: list[dict]) -> tuple[bool, bool]:
    """每个 gold 条款有一组可能所在的检索块（gold_chunks）；命中其中任一块就算找到这一条。"""
    ids = {h["chunk_id"] for h in hits[:K]}
    found = [bool(ids & set(cs)) for cs in q["gold_chunks"]]
    return all(found), any(found)


def main():
    qs = [q for q in json.loads(RESOLVED.read_text(encoding="utf-8")) if q["split"] == "dev" and q["answerable"]]
    print(f"dev2 有答案的题 {len(qs)} 道；取前 {K} 名\n")
    rows, cost, t0 = [], 0.0, time.perf_counter()
    for q in qs:
        as_of = date.fromisoformat(q["as_of"])
        passage, u = hyde_passage(q["question"])
        cost += llm.cost_usd(u)
        res = {}
        for name, query in (("question", None), ("hyde", passage), ("both", q["question"] + "\n" + passage)):
            hits = retrieve(q["question"], as_of, k=K, query=query)
            res[name] = gold_found(q, hits)
        rows.append({"id": q["id"], "pattern": q["pattern"], "passage": passage, **{k: v[0] for k, v in res.items()},
                     **{k + "_hit": v[1] for k, v in res.items()}})
        print(f"  {q['id']} [{q['pattern']:<10}] 全中：原问题 {'✓' if res['question'][0] else '·'}  "
              f"HyDE {'✓' if res['hyde'][0] else '·'}  合并 {'✓' if res['both'][0] else '·'}   {q['question'][:60]}")

    n = len(rows)
    print(f"\n  {'首轮查询':<10}{'Recall@8 全中':>14}{'Hit@8':>8}")
    for name in ("question", "hyde", "both"):
        print(f"  {name:<10}{sum(r[name] for r in rows):>10}/{n}{sum(r[name + '_hit'] for r in rows):>6}/{n}")
    print("\n  与原问题配对比较（D64：至少多 2 道、变差不超过 1 道才采用）：")
    for name in ("hyde", "both"):
        better = [r["id"] for r in rows if r[name] and not r["question"]]
        worse = [r["id"] for r in rows if r["question"] and not r[name]]
        verdict = "采用" if len(better) >= 2 and len(worse) <= 1 else "不采用"
        print(f"    {name:<6} 变好 {len(better)} {better}  变差 {len(worse)} {worse}  → {verdict}")
    print(f"\n  生成假想段落的费用 ${cost:.4f}；用时 {time.perf_counter() - t0:.0f}s")
    out = PROJECT_ROOT / "reports" / "r8_hyde_dev.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  逐题结果（含假想段落）已写到 {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()