"""
SentryAML - R9b-1：Agent 的通用部分（从 R8 的法规研究 Agent 中抽出）
====================================================================
R8 的法规研究 Agent（kb_agent.py + kb_tools.py）里，有一半代码和"法规"无关：
  工具的登记与参数校验、预算与重复调用、调用留痕、和模型来回对话的循环、用量记账。
R9 的告警分诊 Agent 需要完全相同的这一半，只是工具不同（查交易而不是查条款）、提交的东西不同（分诊结论而不是回答）。
把这一半抽到这里，两个 Agent 共用；各自只写自己的工具、提示词和"提交时怎么核验"。

为什么现在抽、而不是写 R8 时就抽：
  只有一个使用者时，"通用"的边界是猜的——很容易把法规特有的东西（比如"已看过的检索块"）当成通用，
  或者把通用的东西写死成法规的样子。现在有了第二个具体的使用者（分诊），哪些真的通用、哪些不是，可以逐项对照着定。

这一层提供四样东西：
  1. Tool / RunContext / call_tool   工具登记、一次运行的共享状态、统一的调用入口（校验、预算、重复、留痕）
  2. run_loop                        与模型来回对话的循环：发消息 → 执行工具 → 追加结果 → 直到提交或触发停止条件
  3. new_usage / add_usage           用量记账（含缓存命中、失败调用的用量）
  4. strip_titles / inline_schema    把 pydantic 模型变成模型看得懂的工具参数格式

循环不认识任何具体工具：执行工具、处理提交都由调用方传进来的函数完成（依赖注入）。
所以循环本身不需要因为换了 Agent 而修改；改动法规 Agent 的工具，也不可能影响分诊 Agent 的循环。

重构纪律：抽取前后，法规研究 Agent 发给模型的每一个字节都不变。scripts/agent_regress.py 用缓存回放证明这一点
（缓存键 = 完整消息历史 + 工具定义的哈希；任何一处文字变化都会导致缓存未命中，回放脚本会直接报错）。
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

import llm
from db import connect


# ---------------------------------------------------------------------------
# 1. 用量记账
# ---------------------------------------------------------------------------
def new_usage() -> dict:
    return {"calls": 0, "cached_calls": 0, "prompt": 0, "hit": 0, "completion": 0, "reasoning": 0}


def add_usage(total: dict, u: dict) -> None:
    """u 是 llm.chat_json / chat_tools 返回的单次用量；失败的调用（LLMError.usage）同样要记，因为同样计费。"""
    total["calls"] += 1
    total["cached_calls"] += int(u.get("cached", False))
    for k in ("prompt", "hit", "completion", "reasoning"):
        total[k] += u.get(k, 0)


# ---------------------------------------------------------------------------
# 2. 工具参数格式
# ---------------------------------------------------------------------------
def strip_titles(s):
    """pydantic 生成的 schema 里每个字段都带 title，对模型没有信息量，只占 token。"""
    if isinstance(s, dict):
        return {k: strip_titles(v) for k, v in s.items() if k != "title"}
    if isinstance(s, list):
        return [strip_titles(v) for v in s]
    return s


def inline_schema(schema: dict) -> dict:
    """把 pydantic 生成的 $defs / $ref 展开成内联结构，并去掉 title：不依赖 API 对 $ref 的支持，模型读起来也更直接。
    用于嵌套结构的参数（提交工具：一个回答里有多条论断 / 一个分诊结论里有多条事实）。"""
    defs = schema.pop("$defs", {})

    def walk(x):
        if isinstance(x, dict):
            if "$ref" in x:
                return walk(dict(defs[x["$ref"].split("/")[-1]]))
            return {k: walk(v) for k, v in x.items() if k != "title"}
        if isinstance(x, list):
            return [walk(v) for v in x]
        return x
    return walk(schema)


# ---------------------------------------------------------------------------
# 3. 工具、运行上下文、统一调用入口
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Tool:
    name: str
    description: str                                   # 写给模型的：它完全依靠这几句话决定什么时候用哪个工具
    args_model: type[BaseModel]                         # 同一份定义既生成 schema 给模型看，又校验模型给出的参数
    fn: Callable[["RunContext", BaseModel], dict]

    def schema(self) -> dict:
        """OpenAI 兼容的 function calling 格式（DeepSeek 使用同一格式）。"""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": strip_titles(self.args_model.model_json_schema())}}


@dataclass
class RunContext:
    """
    一次 Agent 运行共享的状态。各个 Agent 继承它，加上自己的状态
    （法规研究：已看过的检索块、证据池；分诊：告警、已出现的账户、事实登记表）。

    as_of 由系统绑定、所有工具共用，模型不能修改（D60）：法规研究是调查日期，分诊是告警时刻。
    """
    question: str
    as_of: date | datetime
    kind: str = "agent"
    max_calls: int = 6
    log: bool = True
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step: int = 0                                             # 已记录的步数（含系统发起的首轮调用）
    model_calls: int = 0                                      # 模型发起的工具调用次数：预算按它计算
    calls: dict[str, int] = field(default_factory=dict)       # 调用指纹 → 第几步：识别重复调用
    # 预算用完、重复调用时附在错误信息后面的提示：不同 Agent 的"下一步该怎么做"不同
    budget_note: str = "Submit now with the evidence you have."
    duplicate_hint: str = "Change the arguments."

    def summary_extra(self) -> dict:
        """结束时写进 agent_runs.summary 的额外字段；子类覆盖。"""
        return {}

    def start(self) -> "RunContext":
        if self.log:
            with connect() as conn, conn.cursor() as cur:
                cur.execute("INSERT INTO agent_runs (run_id, kind, question, as_of) VALUES (%s, %s, %s, %s)",
                            (self.run_id, self.kind, self.question, self.as_of))
                conn.commit()
        return self

    def finish(self, status: str, summary: dict | None = None) -> None:
        if self.log:
            with connect() as conn, conn.cursor() as cur:
                cur.execute("""UPDATE agent_runs SET status = %s, finished_at = now(), summary = %s
                               WHERE run_id = %s""",
                            (status, json.dumps({"tool_calls": self.step, **self.summary_extra(), **(summary or {})},
                                                ensure_ascii=False, default=str),
                             self.run_id))
                conn.commit()


def log_call(ctx: RunContext, name: str, args: dict, result: dict, error: str | None, t0: float) -> None:
    """写入 agent_tool_calls（019）：记录模型看到的原样结果，用于调试、审计和评测。"""
    if not ctx.log:
        return
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO agent_tool_calls (run_id, step, tool, args, ok, error, result, ms)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                    (ctx.run_id, ctx.step, name, json.dumps(args, ensure_ascii=False), error is None, error,
                     json.dumps(result, ensure_ascii=False, default=str),
                     round((time.perf_counter() - t0) * 1000)))
        conn.commit()


def call_tool(ctx: RunContext, registry: dict[str, Tool], name: str, raw_args: str | dict) -> dict:
    """
    执行一次模型发起的工具调用并记录。raw_args 是模型给出的参数（function calling 返回的是 JSON 字符串）。
    返回给模型的内容永远是一个 dict。
      可以预期的错误（参数不是 JSON / 工具不存在 / 预算用完 / 重复调用 / 参数不合法 / 工具自己报告的问题）
        放在 "error" 里返回，由模型决定怎么改正；
      不可预期的错误（数据库连不上、代码错误）先记录再抛出，让这次运行失败，而不是让模型绕过一个坏掉的系统。
    """
    ctx.step += 1
    ctx.model_calls += 1
    t0 = time.perf_counter()
    result: dict
    error = None
    try:
        args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    except json.JSONDecodeError as e:
        args, result, error = {"_raw": str(raw_args)[:500]}, {"error": f"arguments are not valid JSON: {e.msg}"}, \
            "bad_json"
    else:
        fp = json.dumps([name, args], sort_keys=True, ensure_ascii=False)
        tool = registry.get(name)
        if tool is None:
            result, error = {"error": f"unknown tool '{name}'. Available: {', '.join(registry)}"}, "unknown_tool"
        elif ctx.model_calls > ctx.max_calls:
            result, error = {"error": f"tool budget of {ctx.max_calls} calls is used up. {ctx.budget_note}"}, "budget"
        elif fp in ctx.calls:
            # 完全相同的调用：不再执行，提醒模型（防止原地打转）
            result, error = {"error": f"duplicate of call #{ctx.calls[fp]}; the result has not changed. "
                                      f"{ctx.duplicate_hint}"}, "duplicate"
        else:
            try:
                parsed = tool.args_model.model_validate(args)
            except ValidationError as e:
                result = {"error": "invalid arguments",
                          "details": [f"{'.'.join(map(str, d['loc'])) or '(root)'}: {d['msg']}" for d in e.errors()]}
                error = "invalid_args"
            else:
                ctx.calls[fp] = ctx.step
                try:
                    result = tool.fn(ctx, parsed)
                except Exception as e:                 # 不可预期的错误：先记录，再抛出
                    log_call(ctx, name, args, {"error": f"{type(e).__name__}: {e}"}, "exception", t0)
                    raise
                if "error" in result:
                    error = "tool_error"
    log_call(ctx, name, args, result, error, t0)
    return result


# ---------------------------------------------------------------------------
# 4. 与模型来回对话的循环
# ---------------------------------------------------------------------------
@dataclass
class LoopOutcome:
    final: Any = None        # on_submit 返回的最终结果；None = 没有完成提交（触发了停止条件）
    turns: int = 0           # 调用模型的轮数
    nudges: int = 0          # 模型既没调用工具也没提交、被提醒的次数


def run_loop(messages: list[dict], tools: list[dict], *,
             run_tool: Callable[[str, str], dict],
             submit_name: str,
             on_submit: Callable[[str, int], tuple[dict, Any]],
             usage: dict,
             use_cache: bool = True,
             max_turns: int = 12,
             max_nudges: int = 2,
             nudge: str = "Continue: call a tool, or submit to finish.",
             purpose: str = "agent",
             max_tokens: int = 8000) -> LoopOutcome:
    """
    循环（消息历史只追加、从不修改，所以每一轮的前缀都与上一轮相同，服务端前缀缓存命中前面全部内容）：
      1. 把消息历史和工具清单发给模型
      2. 模型没有调用任何工具 → 提醒一次（最多 max_nudges 次，超过就停）
      3. 模型调用了工具 → 逐个执行，结果作为 tool 消息追加；同一轮里的提交放在最后执行
         （模型可能"查一下、顺便提交"：先让它的查询生效）
      4. on_submit 返回 (交给模型的结果, 最终结果或 None)：None 表示核验没通过、允许改正，循环继续
      5. 任一停止条件满足即停：完成提交 / 达到 max_turns / 提醒次数用完
    工具预算（max_calls）在 call_tool 里执行，超出后工具返回错误、请模型提交；max_turns 是预算之外的第二道保险。

    模型调用失败（llm.LLMError）时：把这次失败调用的用量记入 usage（失败同样计费），
    在异常上附带 .turn（失败发生在第几轮），然后原样抛出，由调用方决定如何记录。
    """
    submits, nudges, final, turn = 0, 0, None, 0
    try:
        for turn in range(1, max_turns + 1):
            msg, u = llm.chat_tools(messages, tools, purpose=purpose, max_tokens=max_tokens, use_cache=use_cache)
            add_usage(usage, u)
            messages.append(llm.assistant_message(msg))
            if not msg["tool_calls"]:
                nudges += 1
                if nudges > max_nudges:
                    break
                messages.append({"role": "user", "content": nudge})
                continue
            calls = sorted(msg["tool_calls"], key=lambda tc: tc["name"] == submit_name)
            for tc in calls:
                if tc["name"] == submit_name:
                    result, final = on_submit(tc["arguments"], submits)
                    submits += 1
                else:
                    result = run_tool(tc["name"], tc["arguments"])
                messages.append({"role": "tool", "tool_call_id": tc["id"],
                                 "content": json.dumps(result, ensure_ascii=False)})
            if final is not None:
                break
    except llm.LLMError as e:
        if e.usage:
            add_usage(usage, e.usage)
        e.turn = turn
        raise
    return LoopOutcome(final=final, turns=turn, nudges=nudges)