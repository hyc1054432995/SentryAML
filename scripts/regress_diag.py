"""
SentryAML - R9b-1：回放失败的诊断（agent v229）
================================================
agent_regress.py 里 agent v229 回放失败，但现象很特别：6 次模型调用全部命中缓存，记录却和保存的不同。

全部命中缓存说明：回放走的每一轮请求，都和"以前某一次运行"的请求逐字节相同——
所以这条路径本身是以前真实出现过的，只是不是 reports/r8_answer_dev.json 里保存的那一次。
模型的回复由缓存决定（同样的请求 → 同样的回复），那么两次运行要分岔，只能是某一步的工具结果不同。
对 agent 配置，第一个工具结果是系统发起的首轮检索（seed_search），它直接写进第 1 轮的提示词。
agent_noseed 没有首轮检索，18 题全部回放成功，这一点和"首轮检索不稳定"的猜测一致，但还需要直接证据。

这个脚本收集三样证据（不调用大模型；检索的嵌入和重排走各自的缓存）：
  1. 数据库里这道题以前所有的运行（agent_runs / agent_tool_calls 都有记录）：每次首轮检索返回了哪些条款、什么顺序，
     以及整条调查路径（调用了哪些工具）。看保存的那次和回放走的那次，首轮检索是否不同。
  2. 现在连续跑 5 次首轮检索，每次一个新进程（Python 的字符串哈希每个进程随机，集合的遍历顺序会变），
     看结果是否每次都一样。
  3. 如果不一样：把每次的前 12 名连同重排分数、来源路线（混合检索 / 引用扩展 / 编号路由）列出来，看是在哪里分岔的。

用法：
    python scripts/regress_diag.py              # 默认 v229
    python scripts/regress_diag.py --qid v229 --runs 5
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

from db import connect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOLVED = PROJECT_ROOT / "reports" / "rag_golden_v2_resolved.json"
SAVED = PROJECT_ROOT / "reports" / "r8_answer_dev.json"


def history(question: str, as_of: str, saved_run: str | None):
    print("[1] 数据库里这道题以前的全部运行（按时间）")
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""SELECT r.run_id, r.started_at, r.status, r.summary
                       FROM agent_runs r WHERE r.question = %s AND r.as_of = %s AND r.kind = 'research'
                       ORDER BY r.started_at""",
                    (question, as_of))   # 同一个问题在 v2 里有两个调查日期（版本题），必须按日期区分
        runs = cur.fetchall()
        seeds = {}
        for run_id, started, status, summary in runs:
            cur.execute("""SELECT step, tool, args, error, result FROM agent_tool_calls
                           WHERE run_id = %s ORDER BY step""", (run_id,))
            calls = cur.fetchall()
            seed = next((c[4] for c in calls if c[1] == "seed_search"), None)
            seed_refs = tuple(r["refs"][0] if r.get("refs") else "?" for r in (seed or {}).get("results", []))
            seeds.setdefault(seed_refs, []).append(str(run_id)[:8])
            path = " → ".join(c[1].replace("search_regulations", "search").replace("get_clause", "clause")
                              .replace("submit_answer", "submit").replace("seed_search", "seed")
                              + ("✗" if c[3] else "") for c in calls)
            print(f"  {started:%m-%d %H:%M}  run {str(run_id)[:8]}  {status:<10}"
                  f" 首轮检索 {'（无）' if seed is None else '组合 ' + str(list(seeds).index(seed_refs) + 1)}  路径：{path}"
                  + ("   ← reports 里保存的那次" if str(run_id) == saved_run else ""))
        print(f"\n  首轮检索一共出现过 {len([s for s in seeds if s])} 种不同的结果：")
        for i, (refs, ids) in enumerate(seeds.items(), 1):
            if refs:
                print(f"    组合 {i}（运行 {', '.join(ids)}）：{', '.join(refs)}")
    return seeds


CHILD = r"""
import json, sys
from datetime import date
sys.path.insert(0, sys.argv[1])
from kb_search import retrieve
hits = retrieve(sys.argv[2], date.fromisoformat(sys.argv[3]), 12)
print(json.dumps([[h["chunk_id"], None if h["score"] is None else round(h["score"], 6), h["route"],
                   h.get("first_stage_rank")] for h in hits]))
"""


def repeat(question: str, as_of: str, runs: int):
    print(f"\n[2] 现在连续跑 {runs} 次首轮检索，每次一个新进程")
    outs = []
    for i in range(runs):
        p = subprocess.run([sys.executable, "-c", CHILD, str(Path(__file__).parent), question, as_of],
                           capture_output=True, text=True, encoding="utf-8")
        if p.returncode != 0:
            print(p.stderr[-2000:])
            raise SystemExit("检索子进程出错")
        outs.append(json.loads(p.stdout.strip().splitlines()[-1]))
        print(f"  第 {i + 1} 次 前 8 名：{', '.join(c for c, *_ in outs[-1][:8])}")
    top8 = {tuple(c for c, *_ in o[:8]) for o in outs}
    if len(top8) == 1:
        print("  → 5 次结果完全相同：现在的检索是确定的")
    else:
        print(f"  → 出现了 {len(top8)} 种不同的前 8 名：检索不确定")
    print("\n[3] 每次的前 12 名（块 / 重排分 / 来源路线 / 第一阶段名次）")
    for i, o in enumerate(outs, 1):
        print(f"  第 {i} 次")
        for rank, (cid, score, route, fsr) in enumerate(o, 1):
            ties = [x[0] for x in o if x[1] == score and x[0] != cid and score is not None]
            print(f"    {rank:>2}. {cid:<40} {score!s:<10} {route:<18} 第一阶段 {fsr}"
                  + (f"   与 {ties} 同分" if ties else ""))
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qid", default="v229")
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args()
    q = next(x for x in json.loads(RESOLVED.read_text(encoding="utf-8")) if x["id"] == a.qid)
    if q["split"] != "dev":
        raise SystemExit("只诊断开发集题目")
    print(f"{a.qid}（调查日期 {q['as_of']}）：{q['question']}\n")
    saved = json.loads(SAVED.read_text(encoding="utf-8"))["records"]["agent"][a.qid]
    history(q["question"], q["as_of"], str(saved.get("run_id")))
    print(f"\n  保存的记录：run {str(saved.get('run_id'))[:8]}，状态 {saved['status']}，轮数 {saved.get('turns')}，"
          f"工具 {saved.get('tool_calls')}")
    print("  证据池前 8 条（条款级；首轮检索带回的排在最前）：" + ", ".join(s["sid"] for s in saved["sources"][:8]))
    repeat(q["question"], q["as_of"], a.runs)


if __name__ == "__main__":
    main()