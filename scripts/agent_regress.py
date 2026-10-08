"""
SentryAML - R9b-1：重构回归测试（缓存回放）
============================================
R9b-1 把法规研究 Agent 的循环、工具调用入口、用量记账抽到了 agent_core.py。这是一次"重构"：
只改代码的组织方式，不改行为。这个脚本证明行为确实没变。

原理：llm.chat_tools 的缓存键 = sha256(模型 | 完整消息历史 | 工具定义 | 输出上限)。
  - 重构后的代码如果发给模型的内容和 R8c 那次运行逐字节相同，每一轮都会命中缓存，拿到当时的回复，
    整条调查路径原样重现，最终记录也完全相同。
  - 只要有一处变了（提示词多一个空格、工具说明改一个词、工具结果的 JSON 字段顺序变了、消息的追加顺序变了），
    从那一轮起缓存键就对不上。
这里把真正调用 API 的两个函数换成"一被调用就报错"，所以缓存未命中不会花钱，而是直接暴露为回归失败。

用 dev2 上两种配置（agent：系统首轮检索；agent_noseed：模型自己检索）的全部题目回放，
与 reports/r8_answer_dev.json 里保存的记录逐字段比较。只用开发集：回放不产生新的评测结论，但测试集的纪律不破例。

它证明不了什么：只覆盖这些运行实际走过的路径。脚本最后会列出这些运行触发过哪些分支（重新提交、引号被去掉、
预算用完、重复调用……），没出现过的分支由离线的脚本化测试覆盖（见 D70）。

用法：
    python scripts/agent_regress.py
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import llm
from kb_agent import research

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVED = PROJECT_ROOT / "reports" / "r8_answer_dev.json"
RESOLVED = PROJECT_ROOT / "reports" / "rag_golden_v2_resolved.json"
CONFIGS = {"agent": True, "agent_noseed": False}          # 配置名 → seed
FIELDS = ["status", "abstain_reason", "answer", "missing", "claims", "turns", "tool_calls", "trace", "scope",
          "sources", "first_pass", "retried", "unquoted", "answerable_conflict", "prompt_version", "seed"]


class CacheMiss(Exception):
    pass


def _forbid(*args, **kwargs):
    raise CacheMiss("请求没有命中缓存：发给模型的内容与 R8c 运行时不同")


def main():
    llm._call = _forbid              # chat_json（范围检查）真正调用 API 的函数
    llm._call_tools = _forbid        # chat_tools（循环每一轮）真正调用 API 的函数

    saved = json.loads(SAVED.read_text(encoding="utf-8"))["records"]
    qs = [q for q in json.loads(RESOLVED.read_text(encoding="utf-8")) if q["split"] == "dev"]
    print(f"回放 dev2 {len(qs)} 题 × {len(CONFIGS)} 种配置；任何一次请求未命中缓存都算失败（不会调用 API）\n")

    fails, branches, n = [], Counter(), 0
    for config, seed in CONFIGS.items():
        for q in qs:
            old = saved.get(config, {}).get(q["id"])
            if old is None:
                print(f"  {config:<13}{q['id']}  （保存的记录里没有这道题，跳过）")
                continue
            n += 1
            try:
                new = research(q["question"], date.fromisoformat(q["as_of"]), use_cache=True, log=False, seed=seed)
            except CacheMiss as e:
                fails.append((config, q["id"], str(e)))
                print(f"  ✗ {config:<13}{q['id']}  {e}")
                continue
            new = json.loads(json.dumps(new, ensure_ascii=False, default=str))   # 与保存时同样经过一次 JSON
            diff = [k for k in FIELDS if old.get(k) != new.get(k)]
            all_cached = new["usage"]["cached_calls"] == new["usage"]["calls"]
            ok = not diff and all_cached
            if not ok:
                fails.append((config, q["id"], f"字段不同 {diff}" if diff else "有调用未走缓存"))
            print(f"  {'✓' if ok else '✗'} {config:<13}{q['id']}  {new['status']:<10}"
                  f"模型调用 {new['usage']['calls']}（缓存 {new['usage']['cached_calls']}）  "
                  f"工具 {sum(v for k, v in new['tool_calls'].items() if k != 'submit_answer')}"
                  + (f"  不同的字段 {diff}" if diff else ""))
            # 这些运行走过哪些分支
            branches[f"状态 {new['status']}" + (f"/{new['abstain_reason']}" if new.get("abstain_reason") else "")] += 1
            branches["第一次提交被退回、重新提交"] += int(bool(new.get("retried")))
            branches["引号被去掉（accepted_unquoted）"] += int(bool(new.get("unquoted")))
            branches["拒答但标记自相矛盾"] += int(bool(new.get("answerable_conflict")))
            for t in new["trace"]:
                if t.get("error"):
                    branches[f"工具错误：{str(t['error'])[:40]}"] += 1

    print(f"\n回放 {n} 次，失败 {len(fails)} 次")
    for f in fails:
        print(f"  ✗ {f[0]} {f[1]}：{f[2]}")
    print("\n这些运行覆盖到的分支：")
    for k, v in sorted(branches.items()):
        print(f"  {k:<40}{v:>4}")
    if fails:
        raise SystemExit(1)
    print("\n重构前后行为一致：发给模型的每一轮请求逐字节相同，记录逐字段相同，费用 0")


if __name__ == "__main__":
    main()