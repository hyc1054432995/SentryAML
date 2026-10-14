"""
SentryAML - R11b：解读提示在 R10d 评分表上的检验
================================================
R10d 的评分表（reports/r10d_rubric_<split>.yaml）里有 5 份理由、逐句的人工判断；第二位评分者的判断在
reports/r10d_rater2_<split>.yaml。这里对每一句计算解读提示（review_hints），然后数：
  - 被判为"不被支撑"的句子，有几句被提示标出（标出率）
  - 没有被判为问题的句子，有几句也被标出（误标：提示太多，复核员会开始忽略它们）
"被判为不被支撑" = 第二位评分者文件里的（有这个文件时）∪ 评分表里本身填了 supported: false 的。

**这是样本内的检验，必须如实说明**：三条规则就是看着 test 上这几句写出来的（D84 → D86），标出它们是预期之中，
不能当作"规则有效"的证据。这里真正有信息量的是误标数：规则在其余句子上会不会乱标。规则是否真的有用，
要等新的理由（新的告警、新的复核）出来以后再看。

证据范围与模型起草时看到的完全一致：告警本身 + Agent 核验通过的事实所在的步骤（str_narrative.evidence_results），
工具结果取自 agent_tool_calls（与模型当时看到的逐字节相同）。只读，不写任何东西。

用法：
    python scripts/r11b_hints_check.py --split test
    python scripts/r11b_hints_check.py --split dev                  # dev：规则不是看着这些句子写的，误标数更有参考价值
    python scripts/r11b_hints_check.py --split test --dump 880 3502 # 打印这些告警的全部证据字段（路径 = 值），查规则为什么没触发
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

import review_hints as rh
from db import connect_review
from str_narrative import evidence_results, number_hits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS = {"dev": PROJECT_ROOT / "reports" / "r9b_dev_runs.json", "test": PROJECT_ROOT / "reports" / "r9c_test_runs.json"}


def main() -> None:
    ap = argparse.ArgumentParser(description="R11b：解读提示在 R10d 评分表上的检验（只读）")
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--dump", type=int, nargs="*", help="只打印这些告警的证据字段，不做汇总")
    a = ap.parse_args()
    if a.dump:
        dump(a.split, a.dump)
        return
    rubric = yaml.safe_load((PROJECT_ROOT / "reports" / f"r10d_rubric_{a.split}.yaml").read_text(encoding="utf-8"))
    r2_path = PROJECT_ROOT / "reports" / f"r10d_rater2_{a.split}.yaml"
    r2 = yaml.safe_load(r2_path.read_text(encoding="utf-8")) if r2_path.exists() else {"sentences": []}
    bad2 = {(x["alert_id"], x["n"]): x for x in r2.get("sentences", []) if not x.get("supported", True)}
    for it in rubric["items"]:                       # 评分表本身判为不被支撑的句子也算
        for s in it["sentences"]:
            if s.get("supported") is False and (it["alert_id"], s["n"]) not in bad2:
                bad2[(it["alert_id"], s["n"])] = {"note": s.get("note", ""), "borderline": False}
    runs = json.loads(RUNS[a.split].read_text(encoding="utf-8"))

    n_sent = 0
    flagged_all: set[tuple[int, int]] = set()
    with connect_review() as conn, conn.cursor() as cur:
        for it in rubric["items"]:
            aid = it["alert_id"]
            rec = runs[str(aid)]
            case = {"alert_id": aid, "agent_run_id": rec.get("run_id")}
            facts = [f for f in rec.get("facts") or [] if f.get("verified")]
            results = evidence_results(cur, case, facts)
            hints = rh.case_hints(cur, results)
            print(f"\n#{aid}（{it.get('typology')}）：{len(hints)} 条案件级提示")
            for k, h in enumerate(hints, 1):
                print(f"  提示 {k} [{h['kind']}] {h['text']}")
            for s in it["sentences"]:
                n_sent += 1
                hits = [h for _, hs in number_hits(s["text"], results) for h in hs]
                kinds = [h["kind"] for h in hints if rh.triggered(s["text"], hits, h)]
                key = (aid, s["n"])
                mark = "评分者：不被支撑" + ("（临界）" if bad2.get(key, {}).get("borderline") else "") \
                    if key in bad2 else ""
                if kinds:
                    flagged_all.add(key)
                if kinds or mark:
                    print(f"  第 {s['n']} 句 {'⚑ ' + ','.join(kinds) if kinds else '（未标出）'}  {mark}\n"
                          f"      {s['text'][:150]}")
        conn.rollback()

    clear = {k for k, x in bad2.items() if not x.get("borderline")}
    border = {k for k, x in bad2.items() if x.get("borderline")}
    extra = flagged_all - set(bad2)
    print(f"\n[汇总] {a.split}：{len(rubric['items'])} 份理由、{n_sent} 句；被提示标出 {len(flagged_all)} 句")
    print(f"  评分者明确判为不被支撑的 {len(clear)} 句：标出 {len(clear & flagged_all)} 句；"
          f"临界的 {len(border)} 句：标出 {len(border & flagged_all)} 句")
    print(f"  误标（评分者没有判为问题、却被标出）：{len(extra)} 句" +
          (f"：{sorted(extra)}" if extra else ""))
    for k in sorted(clear - flagged_all):
        print(f"  未标出：#{k[0]} 第 {k[1]} 句——{bad2[k].get('note', '')[:120]}")
    print("  注意：规则是看着 test 上这几句写出来的，标出率是样本内的，不能当作规则有效的证据；有信息量的是误标数。")


def dump(split: str, alert_ids: list[int]) -> None:
    """打印证据字段：规则为什么没触发，先看数据长什么样，不凭猜去改规则。"""
    from str_narrative import flatten
    runs = json.loads(RUNS[split].read_text(encoding="utf-8"))
    with connect_review() as conn, conn.cursor() as cur:
        for aid in alert_ids:
            rec = runs.get(str(aid))
            if rec is None:
                print(f"#{aid}：{split} 的运行记录里没有")
                continue
            facts = [f for f in rec.get("facts") or [] if f.get("verified")]
            results = evidence_results(cur, {"alert_id": aid, "agent_run_id": rec.get("run_id")}, facts)
            print(f"\n#{aid}：引用的事实 " + "; ".join(f"step {f['step']} {f['path']} = {f['value']}" for f in facts))
            for s, res in results.items():
                print(f"  -- 第 {s} 步 {res.get('tool', '')}" + (f" on {res['account']}" if res.get("account") else ""))
                for p, v in flatten(res):
                    print(f"     {p} = {json.dumps(v, ensure_ascii=False, default=str)[:120]}")
        conn.rollback()


if __name__ == "__main__":
    main()