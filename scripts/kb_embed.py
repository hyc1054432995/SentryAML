"""
SentryAML - RAG 层 R5b：稠密向量（bge-m3），两个可切换的后端 + Postgres 缓存
=============================================================================
同一个模型 BAAI/bge-m3，两种运行方式，由 .env 里的 EMBED_BACKEND 选择：

  siliconflow（默认）  调用硅基流动的 API，本机不需要算力。需要 SILICONFLOW_API_KEY。
  local                在本机 CPU 上用 sentence-transformers 运行，首次会下载约 2.2GB 的模型。
                       给没有 key 的人复现项目用；建索引慢（十几到几十分钟），单条查询零点几秒。

为什么做成可切换：项目要开源，不能要求每个复现者都注册某家平台；
同时两个后端是同一个模型，评测结果可以互相对照。

为什么 bge-m3：多语言（R4 有中文提问、英文原文），最长 8192 token（块 + 说明放得下），开源可本地运行。
bge-m3 不需要像 bge-en-v1.5 那样给查询加 "Represent this sentence…" 的指令前缀，问题和文档直接编码。

缓存：每段文字先查 kb_embeddings（016），没有的才计算，算完写回。
缓存键包含模型名：换模型（例如 R6 对比 Qwen3-Embedding）时不会误用旧向量。
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

from db import connect

PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

BACKEND = os.environ.get("EMBED_BACKEND", "siliconflow")
SF_MODEL = os.environ.get("SILICONFLOW_EMBED_MODEL", "Pro/BAAI/bge-m3")
SF_BASE_URL = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
LOCAL_MODEL = "BAAI/bge-m3"
BATCH = 16                       # 每次 API 请求的文字条数；块最长约 2,500 字符，16 条远低于单次请求的限制


class Embedder:
    """统一接口：embed(texts) → 向量列表。缓存逻辑在外层，后端只管计算。"""
    model_name: str
    dim: int = 1024

    def _compute(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed(self, texts: list[str], use_cache: bool = True, progress: bool = False) -> list[list[float]]:
        keys = [hashlib.sha256(f"{self.model_name}|{t}".encode("utf-8")).hexdigest() for t in texts]
        found: dict[str, list[float]] = {}
        if use_cache:
            with connect() as conn, conn.cursor() as cur:
                cur.execute("SELECT text_hash, vector FROM kb_embeddings WHERE text_hash = ANY(%s)",
                            (list(set(keys)),))
                found = {k: list(v) for k, v in cur.fetchall()}
        todo = list({k: t for k, t in zip(keys, texts) if k not in found}.items())
        if progress:
            print(f"    向量缓存命中 {len(set(keys)) - len(todo)}，需要计算 {len(todo)}（{self.model_name}）")
        t0 = time.perf_counter()
        for i in range(0, len(todo), BATCH):
            batch = todo[i:i + BATCH]
            vecs = self._compute([t for _, t in batch])
            if use_cache:
                with connect() as conn, conn.cursor() as cur:
                    cur.executemany(
                        "INSERT INTO kb_embeddings (text_hash, model, dim, vector, n_chars) VALUES (%s, %s, %s, %s, %s) "
                        "ON CONFLICT (text_hash) DO NOTHING",
                        [(k, self.model_name, len(v), v, len(t)) for (k, t), v in zip(batch, vecs)])
                    conn.commit()
            found.update({k: v for (k, _), v in zip(batch, vecs)})
            if progress and (i // BATCH) % 10 == 9:
                print(f"    已计算 {min(i + BATCH, len(todo))}/{len(todo)}，{time.perf_counter() - t0:.0f}s")
        return [found[k] for k in keys]


class SiliconFlowEmbedder(Embedder):
    def __init__(self):
        from openai import OpenAI                       # 硅基流动的向量接口与 OpenAI 格式兼容
        key = os.environ.get("SILICONFLOW_API_KEY")
        if not key:
            raise SystemExit("没有找到 SILICONFLOW_API_KEY：请在 .env 里设置，或设 EMBED_BACKEND=local 改用本机 CPU")
        self.client = OpenAI(api_key=key, base_url=SF_BASE_URL, timeout=60, max_retries=5)
        self.model_name = SF_MODEL

    def _compute(self, texts):
        resp = self.client.embeddings.create(model=self.model_name, input=texts, encoding_format="float")
        return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]


class LocalEmbedder(Embedder):
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(LOCAL_MODEL, device="cpu")
        self.model_name = f"local/{LOCAL_MODEL}"

    def _compute(self, texts):
        return self.model.encode(texts, batch_size=8, normalize_embeddings=True).tolist()


def get_embedder() -> Embedder:
    if BACKEND == "local":
        return LocalEmbedder()
    if BACKEND == "siliconflow":
        return SiliconFlowEmbedder()
    raise SystemExit(f"未知的 EMBED_BACKEND={BACKEND}（可选 siliconflow / local）")


if __name__ == "__main__":
    # 冒烟测试：中英文同义句应比无关句更相似
    import math
    e = get_embedder()
    a, b, c = e.embed(["How long must a bank keep transaction records?",
                       "银行的交易记录要保存多久？",
                       "What is a scatter-gather laundering pattern?"], use_cache=False)
    cos = lambda x, y: sum(p * q for p, q in zip(x, y)) / math.sqrt(sum(p * p for p in x) * sum(q * q for q in y))
    print(f"模型 {e.model_name}，维度 {len(a)}")
    print(f"  英文问题 vs 中文同义问题：{cos(a, b):.3f}")
    print(f"  英文问题 vs 无关问题：    {cos(a, c):.3f}")