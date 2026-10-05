"""
SentryAML - 查看某几道题的完整回答记录（R7 错误分析用）
=========================================================
从 reports/r7_answer_<split>.json 里取出指定题目的记录，打印：范围检查、8 个来源（检索块、重排分、来源路由）、
模型原始输出的开头、回答、未覆盖说明、每条论断的核验结果。用来判断一道题答错/拒答的原因出在哪一环：
  检索没找到（来源里没有 gold 条款）→ R6 的问题
  来源里有，但模型说资料不足 → 提示词或模型判断的问题
  模型答了，但引文没通过核验 → 引用习惯的问题

用法：
    python scripts/rag_inspect.py q024 q072
    python scripts/rag_inspect.py q020 --config no_scope
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--config", default="full")
    a = ap.parse_args()
    rep = json.loads((PROJECT_ROOT / "reports" / f"r7_answer_{a.split}.json").read_text(encoding="utf-8"))
    gold = {q["id"]: q for q in json.loads((PROJECT_ROOT / "reports" / "rag_golden_resolved.json")
                                          .read_text(encoding="utf-8"))}
    for qid in a.ids:
        r = rep["records"][a.config].get(qid)
        g = gold[qid]
        print("=" * 100)
        print(f"{qid} [{g['type']}/{g['lang']}] 调查日期 {g['as_of']}\n问：{g['question']}")
        print(f"gold：{'；'.join(x['doc_id'] + ':' + x['unit_id'] for x in g['gold']) or '（无答案题）'}")
        if r is None:
            print("（没有记录：这道题在该配置下出错）")
            continue
        print(f"状态：{r['status']}  原因：{r.get('abstain_reason')}  范围检查：{r.get('scope')}")
        gold_units = {(x["doc_id"], x["unit_id"]) for x in g["gold"]}
        print("来源：")
        for s in r.get("sources", []):
            hit = "★" if any((s["doc_id"], u) in gold_units for u in s["units"]) else " "
            sc = "—" if s.get("rerank") is None else f"{s['rerank']:.3f}"
            print(f"  {hit} {s['sid']} {s['chunk_id']:<45} 重排 {sc}  {s['route']:<14} 条款 {', '.join(s['units'])}")
        if r.get("schema_retry"):
            print(f"格式重写：{r['schema_retry']}")
        if r.get("raw"):
            print(f"模型原始输出（前 600 字）：{r['raw'][:600]}")
        if r.get("raw_failed"):
            print(f"解析失败的原始输出（完整）：\n{r['raw_failed']}")
        print(f"回答：{r.get('answer')}\n未覆盖：{r.get('missing')}")
        for c in r.get("claims", []):
            print(f"  {'✓' if c['verified'] else '✗'} {c['source']} {c['unit']} {c['status']}：{c['quote'][:100]}")
    print("\n★ 表示该来源包含 gold 条款。来源里没有 ★ 的题，问题出在检索（R6）；有 ★ 仍拒答的，问题出在生成。")


if __name__ == "__main__":
    main()