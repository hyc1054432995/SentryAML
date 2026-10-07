"""
SentryAML - 大模型调用的统一入口（R7 起使用）
=============================================
所有"要求返回 JSON"的大模型调用都走 chat_json()，统一处理四件事：

  1. 缓存（018 的 llm_cache）：键 = sha256(模型 | 消息 | 参数)。同样的输入重跑零成本、结果完全相同
  2. 输出上限留足：DeepSeek 会先"思考"再作答，思考也算在输出里（R5a 的教训：上限 300 时三分之一被截断）
  3. 宽容解析：去掉 ```json 代码块、前后多余文字；失败时给出具体原因（截断 / 空内容 / 解析失败）
  4. 记账：每次调用的输入、命中缓存、输出、思考 token，用于成本统计

R5a 的 kb_context.py 先于这个模块写成，逻辑相同；以后新代码统一用这里。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from db import connect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
BASE_URL = "https://api.deepseek.com"
PRICE = {"hit": 0.006, "miss": 0.30, "out": 1.20}       # 美元 / 百万 token，按官方高峰价保守估算
_client = None


class LLMError(Exception):
    """调用或解析失败。由 chat_json 抛出时带有 .raw（模型原始输出）和 .usage（这次调用的用量）。"""
    raw: str | None = None
    usage: dict | None = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise SystemExit("没有找到 DEEPSEEK_API_KEY（.env）")
        _client = OpenAI(api_key=key, base_url=BASE_URL, timeout=180, max_retries=3)
    return _client


def _call(messages: list[dict], max_tokens: int) -> tuple[str, dict]:
    """真正调用 API。单独成函数，测试时可以替换。"""
    resp = _get_client().chat.completions.create(
        model=MODEL, messages=messages, temperature=0.0, max_tokens=max_tokens,
        response_format={"type": "json_object"})
    u = resp.usage
    details = getattr(u, "completion_tokens_details", None)
    return resp.choices[0].message.content or "", {
        "prompt": u.prompt_tokens, "completion": u.completion_tokens,
        "hit": getattr(u, "prompt_cache_hit_tokens", 0) or 0,
        "reasoning": (getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
        "finish": resp.choices[0].finish_reason}


def parse_json(raw: str, finish: str | None) -> dict:
    if finish == "length":
        raise LLMError("输出被截断（finish_reason=length）")
    if not raw.strip():
        raise LLMError("返回内容为空")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0) if m else text)
    except json.JSONDecodeError as e:
        # 带上解析器给出的具体原因和位置（R7 测试集 q027：只记了前 80 字，看不出坏在哪里）
        raise LLMError(f"JSON 解析失败（{e.msg}，第 {e.pos} 个字符）：{raw[:80]!r}") from e


def chat_json(messages: list[dict], purpose: str, max_tokens: int = 4000, use_cache: bool = True
              ) -> tuple[dict, dict]:
    """返回 (解析后的 JSON, 用量)。用量里 cached=True 表示来自本地缓存、没有调用 API。"""
    key = hashlib.sha256(json.dumps([MODEL, messages, max_tokens], ensure_ascii=False,
                                    sort_keys=True).encode("utf-8")).hexdigest()
    if use_cache:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT response FROM llm_cache WHERE cache_key = %s", (key,))
            row = cur.fetchone()
        if row:
            return parse_json(row[0], "stop"), {"prompt": 0, "completion": 0, "hit": 0, "reasoning": 0,
                                                "cached": True, "raw": row[0]}
    raw, usage = _call(messages, max_tokens)
    if usage["finish"] == "length":
        # 思考过程偶尔很长，把输出上限吃光（R7 开发集第一次运行时，范围检查在 1,500 的上限处被截断）。
        # 加倍上限重试一次；未用到的额度不计费
        raw, u2 = _call(messages, min(max_tokens * 2, 16000))
        usage = {k: usage[k] + u2[k] for k in ("prompt", "completion", "hit", "reasoning")} | {"finish": u2["finish"],
                                                                                            "length_retry": True}
    try:
        data = parse_json(raw, usage["finish"])        # 解析失败直接抛出，不写缓存
    except LLMError as e:
        # 失败的调用也要留下原始输出和用量：原始输出用于事后诊断，用量用于如实记账（失败的调用同样计费）。
        # R7 测试集 q027 生成失败时，这两样都丢了
        e.raw, e.usage = raw, {**usage, "cached": False}
        raise
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO llm_cache (cache_key, model, purpose, response, prompt_tokens, cache_hit_tokens,
                                              completion_tokens, reasoning_tokens)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (cache_key) DO NOTHING""",
                    (key, MODEL, purpose, raw, usage["prompt"], usage["hit"], usage["completion"], usage["reasoning"]))
        conn.commit()
    return data, {**usage, "cached": False, "raw": raw}


def cost_usd(usage: dict) -> float:
    miss = usage.get("prompt", 0) - usage.get("hit", 0)
    return (usage.get("hit", 0) * PRICE["hit"] + miss * PRICE["miss"] + usage.get("completion", 0) * PRICE["out"]) / 1e6


# ---------------------------------------------------------------------------
# 带工具的调用（R8b）
# ---------------------------------------------------------------------------
def _call_tools(messages: list[dict], tools: list[dict], max_tokens: int) -> tuple[dict, dict]:
    resp = _get_client().chat.completions.create(
        model=MODEL, messages=messages, tools=tools, temperature=0.0, max_tokens=max_tokens)
    m = resp.choices[0].message
    u = resp.usage
    details = getattr(u, "completion_tokens_details", None)
    # 思考模式下的思考内容。openai SDK 不认识这个字段，它放在 model_extra 里
    reasoning = getattr(m, "reasoning_content", None) or (getattr(m, "model_extra", None) or {}).get(
        "reasoning_content")
    msg = {"content": m.content or "", "reasoning_content": reasoning or "",
           "tool_calls": [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                          for tc in (m.tool_calls or [])]}
    return msg, {"prompt": u.prompt_tokens, "completion": u.completion_tokens,
                 "hit": getattr(u, "prompt_cache_hit_tokens", 0) or 0,
                 "reasoning": (getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
                 "finish": resp.choices[0].finish_reason}


def chat_tools(messages: list[dict], tools: list[dict], purpose: str, max_tokens: int = 8000,
               use_cache: bool = True) -> tuple[dict, dict]:
    """
    带工具的一轮调用。返回 (消息, 用量)；消息 = {content, reasoning_content, tool_calls: [{id, name, arguments}]}。
    缓存键包含完整的消息历史和工具定义：同一条调查路径重跑时每一轮都命中缓存，整条路径完全复现、零成本；
    任何一步的工具结果变了（例如知识库更新），从那一步起自然失效。
    """
    key = hashlib.sha256(json.dumps([MODEL, messages, tools, max_tokens], ensure_ascii=False,
                                    sort_keys=True).encode("utf-8")).hexdigest()
    if use_cache:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT response FROM llm_cache WHERE cache_key = %s", (key,))
            row = cur.fetchone()
        if row:
            return json.loads(row[0]), {"prompt": 0, "completion": 0, "hit": 0, "reasoning": 0, "cached": True}
    msg, usage = _call_tools(messages, tools, max_tokens)
    if usage["finish"] == "length":
        msg, u2 = _call_tools(messages, tools, min(max_tokens * 2, 16000))
        usage = {k: usage[k] + u2[k] for k in ("prompt", "completion", "hit", "reasoning")} | {"finish": u2["finish"]}
    if usage["finish"] == "length":
        e = LLMError("输出被截断（finish_reason=length）")
        e.raw, e.usage = json.dumps(msg, ensure_ascii=False), {**usage, "cached": False}
        raise e
    raw = json.dumps(msg, ensure_ascii=False)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""INSERT INTO llm_cache (cache_key, model, purpose, response, prompt_tokens, cache_hit_tokens,
                                              completion_tokens, reasoning_tokens)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (cache_key) DO NOTHING""",
                    (key, MODEL, purpose, raw, usage["prompt"], usage["hit"], usage["completion"], usage["reasoning"]))
        conn.commit()
    return msg, {**usage, "cached": False}


def assistant_message(msg: dict) -> dict:
    """
    把模型的一轮输出转成可以放回消息历史的 assistant 消息。
    DeepSeek 思考模式的硬性要求：带 tools 的多轮请求里，每一轮的 reasoning_content 都必须原样传回，
    否则返回 400 错误（D60）。
    """
    out = {"role": "assistant", "content": msg["content"] or None, "reasoning_content": msg["reasoning_content"]}
    if msg["tool_calls"]:
        out["tool_calls"] = [{"id": tc["id"], "type": "function",
                              "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                             for tc in msg["tool_calls"]]
    return out