"""
SentryAML - R9b-1：通用层的离线回归测试（脚本化的假模型 + 黄金记录）
=====================================================================
agent_regress.py 用真实缓存回放 R8c 的运行，但只覆盖那些运行实际走过的路径。
这个脚本补上没走过的分支：用一个"按脚本出牌"的假模型和返回固定结果的假工具驱动法规 Agent，
不连数据库、不调 API、几秒跑完。

四个脚本：
  1. 并行调用、没调用工具被提醒、重复调用、文件名不合法、参数不是 JSON、未知工具、提交被退回后改正
  2. 一直不调用工具，提醒次数用完 → loop_limit
  3. 预算用完后继续调用，直到轮数上限 → loop_limit
  4. 提交格式连续两次不合法 → generation_error

每个脚本记录"每一轮发给模型的完整请求"的哈希和最终记录，与 tests/agent_core_golden.json 比较。
黄金记录要用 R9b-1 抽取之前的旧代码、在本机生成（D70）：先 --write，再换上新代码比较。
所以这个测试回答的是：现在的代码与 R8 的行为是否逐字节一致。
（不能用别的机器生成的黄金记录：请求里有 pydantic 生成的参数格式，pydantic 版本不同可能有细微差别。）
以后修改 agent_core.py（为分诊加功能）时每次都跑一遍。

用法：
    python scripts/agent_core_golden.py              # 比较
    python scripts/agent_core_golden.py --write      # 有意改变行为后重新生成黄金记录（要在 decisions.md 里说明原因）
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import llm
import kb_agent
import kb_tools

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOLDEN = PROJECT_ROOT / "tests" / "agent_core_golden.json"
TEXT = ("The bank shall perform customer due diligence measures when establishing business relations "
        "with any customer.")


def fake_search(ctx, a):
    return {"query": a.query, "results": [{"rank": 1, "refs": ["mas_n626:6.1"], "text": "x"}]}


def fake_clause(ctx, a):
    ctx.evidence["mas_n626:6.1"] = {"family": "mas_n626", "doc_id": "d1", "unit_id": "6.1", "title": "Notice",
                                    "version": "v1", "section": "", "text": TEXT}
    return {"document": a.document, "results": [{"ref": "mas_n626:6.1", "status": "found", "text": TEXT}]}


def tc(i, name, args):
    return {"id": f"c{i}", "name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}


GOOD = {"answerable": True, "answer": "Banks must perform CDD.", "missing": "",
        "claims": [{"statement": "CDD", "ref": "mas_n626:6.1",
                    "quote": "perform customer due diligence measures when establishing business relations"}]}
BAD = copy.deepcopy(GOOD)
BAD["claims"][0]["quote"] = "words that are not there at all in the clause"
SCRIPTS = {
    1: [[tc(1, "search_regulations", {"query": "cdd timing"}),
         tc(2, "get_clause", {"document": "mas_n626", "clause_ids": ["6.1"]})],
        [],
        [tc(3, "search_regulations", {"query": "cdd timing"}), tc(4, "submit_answer", BAD),
         tc(5, "get_clause", {"document": "nope", "clause_ids": ["1"]})],
        [tc(6, "get_clause", "{bad json"), tc(7, "unknown_tool", {})],
        [tc(8, "submit_answer", GOOD)]],
    2: [[], [], [], []],
    3: [[tc(10 + i, "search_regulations", {"query": f"q{i}"})] for i in range(20)],
    4: [[tc(1, "submit_answer", {"answerable": "x"})], [tc(2, "submit_answer", {"nope": 1})]],
}


def run_script(steps) -> dict:
    requests = []

    def fake_chat_tools(messages, tools, purpose, max_tokens=8000, use_cache=True):
        requests.append(hashlib.sha256(json.dumps([messages, tools, max_tokens], sort_keys=True,
                                                  ensure_ascii=False).encode()).hexdigest())
        n = sum(m["role"] == "assistant" for m in messages)
        u = {"prompt": 100 + n, "completion": 10, "hit": 50, "reasoning": 5, "cached": False}
        return {"content": "", "reasoning_content": f"think {n}",
                "tool_calls": copy.deepcopy(steps[n]) if n < len(steps) else []}, u

    llm.chat_tools = fake_chat_tools
    rec = kb_agent.research("When must CDD be performed?", date(2026, 9, 30), use_scope=False, use_cache=False,
                            log=False, seed=False, max_tool_calls=6)
    for k in ("run_id", "ms", "cost_usd"):
        rec.pop(k, None)
    return {"requests": requests, "rec": json.loads(json.dumps(rec, sort_keys=True, ensure_ascii=False, default=str))}


def main():
    for name, fn in (("search_regulations", fake_search), ("get_clause", fake_clause)):
        t = kb_tools.TOOLS[name]
        kb_tools.TOOLS[name] = type(t)(t.name, t.description, t.args_model, fn)
    got = {str(k): run_script(v) for k, v in SCRIPTS.items()}
    if "--write" in sys.argv:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(got, ensure_ascii=False, indent=1, sort_keys=True), encoding="utf-8")
        print(f"已写入 {GOLDEN.relative_to(PROJECT_ROOT)}")
        return
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    fails = 0
    for k, g in golden.items():
        r = got[k]
        same_req = r["requests"] == g["requests"]
        diff = sorted(f for f in set(g["rec"]) | set(r["rec"]) if g["rec"].get(f) != r["rec"].get(f))
        ok = same_req and not diff
        fails += not ok
        first = next((i for i, (a, b) in enumerate(zip(r["requests"], g["requests"])) if a != b),
                     None if len(r["requests"]) == len(g["requests"]) else min(len(r["requests"]), len(g["requests"])))
        print(f"  {'✓' if ok else '✗'} 脚本 {k}：{len(r['requests'])} 轮请求"
              + ("逐字节相同" if same_req else f"，第 {first + 1} 轮起不同")
              + (f"；记录状态 {r['rec']['status']}/{r['rec'].get('abstain_reason')}" if not diff
                 else f"；记录中不同的字段 {diff}"))
    if fails:
        raise SystemExit(f"{fails} 个脚本与黄金记录不一致：通用层的改动改变了法规 Agent 的行为")
    print("通用层与 R8 的行为逐字节一致")


if __name__ == "__main__":
    main()