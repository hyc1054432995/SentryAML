"""
SentryAML - RAG 层 R6b：重排（cross-encoder）
=============================================
第一阶段（稠密 + BM25 + 编号路由）的特点是快：问题和块各自先变成向量，比较时只算一次相似度。
代价是"问题"和"块"从来没有被放在一起读过——向量是分别算的，模型看不到两者之间具体哪句话对应哪句话。

重排模型（cross-encoder）把"问题 + 一个块"拼在一起输入，逐字交叉比对后给出一个相关度分数。
它准得多，但每个块都要单独跑一次模型，没法对 1,295 个块都做。于是两阶段：
  第一阶段从 1,295 块里快速取出约 30 个候选（追求"别漏"，看 Recall@30）
  第二阶段只对这 30 个精细打分重新排序（追求"排准"，看 MRR、Hit@1）

模型：BAAI/bge-reranker-v2-m3，与 bge-m3 同系列、多语言（中文问题对英文条款）。
运行方式与向量一样可切换（RERANK_BACKEND）：
  siliconflow（默认）  调硅基流动的 /rerank 接口
  local                本机 CPU 用 sentence-transformers 的 CrossEncoder；30 个候选约十几秒，只适合复现

分数缓存在 kb_rerank_cache（017）：键 = sha256(模型 | 问题 | 文档文字)，评测反复跑时不重复付费。
"""

from __future__ import annotations

import hashlib
import math
import os

from db import connect
from kb_embed import PROJECT_ROOT  # noqa: F401  （确保 .env 已加载）

BACKEND = os.environ.get("RERANK_BACKEND", "siliconflow")
SF_MODEL = os.environ.get("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
SF_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
_local_model = None


def _siliconflow(query: str, docs: list[str]) -> list[float]:
    import httpx
    key = os.environ.get("SILICONFLOW_API_KEY")
    if not key:
        raise SystemExit("没有找到 SILICONFLOW_API_KEY（或设 RERANK_BACKEND=local）")
    for attempt in range(4):
        try:
            r = httpx.post(f"{SF_BASE_URL}/rerank", timeout=60,
                           headers={"Authorization": f"Bearer {key}"},
                           json={"model": SF_MODEL, "query": query, "documents": docs,
                                 "top_n": len(docs), "return_documents": False})
            r.raise_for_status()
            break
        except (httpx.HTTPStatusError, httpx.TransportError):
            if attempt == 3:
                raise
            import time
            time.sleep(2 ** attempt)
    scores = [0.0] * len(docs)
    for item in r.json()["results"]:
        scores[item["index"]] = float(item["relevance_score"])
    return scores


def _local(query: str, docs: list[str]) -> list[float]:
    global _local_model
    if _local_model is None:
        from sentence_transformers import CrossEncoder
        _local_model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu", max_length=1024)
    logits = _local_model.predict([(query, d) for d in docs])
    return [1 / (1 + math.exp(-float(x))) for x in logits]       # 与 API 一样映射到 0–1


def model_name() -> str:
    return SF_MODEL if BACKEND == "siliconflow" else "local/BAAI/bge-reranker-v2-m3"


def rerank(query: str, docs: list[str], use_cache: bool = True) -> list[float]:
    """返回与 docs 同顺序的相关度分数（0–1，越大越相关）。use_cache=False 时不读缓存（测真实延迟用），仍会写入。"""
    if not docs:
        return []
    m = model_name()
    keys = [hashlib.sha256(f"{m}|{query}|{d}".encode("utf-8")).hexdigest() for d in docs]
    found = {}
    if use_cache:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT cache_key, score FROM kb_rerank_cache WHERE cache_key = ANY(%s)", (keys,))
            found = dict(cur.fetchall())
    todo = list({k: i for i, k in reversed(list(enumerate(keys))) if k not in found}.values())   # 相同文字只算一次
    if todo:
        fn = _siliconflow if BACKEND == "siliconflow" else _local
        new = fn(query, [docs[i] for i in todo])
        with connect() as conn, conn.cursor() as cur:
            cur.executemany("INSERT INTO kb_rerank_cache (cache_key, model, score) VALUES (%s, %s, %s) "
                            "ON CONFLICT (cache_key) DO NOTHING", [(keys[i], m, s) for i, s in zip(todo, new)])
            conn.commit()
        found.update({keys[i]: s for i, s in zip(todo, new)})
    return [found[k] for k in keys]


if __name__ == "__main__":
    q = "How long must a bank keep transaction records?"
    docs = ["A bank shall retain records of transactions for a period of at least 5 years following completion.",
            "银行应保存交易记录至少五年。",
            "A scatter-gather pattern connects two vertices through intermediate vertices."]
    for d, s in zip(docs, rerank(q, docs)):
        print(f"  {s:.3f}  {d}")