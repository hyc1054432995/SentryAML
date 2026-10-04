"""
SentryAML - RAG 层 R5b：把检索块写入 Qdrant（稠密 + BM25 两路索引，三个消融版本）
==================================================================================
每个块在 Qdrant 里是一个"点"（point）：
  id       uuid5(chunk_id)：同一个块永远得到同一个 id，重复写入只会覆盖（幂等）
  vector   "dense"：bge-m3 的 1024 维向量；"bm25"：BM25 稀疏向量（IDF 由 Qdrant 统计）
  payload  chunk_id、doc_id、文件族、unit_ids、生效区间、原文……检索结果直接带回，不必再查 Postgres

三个版本，各建一个集合，用于 R6 的消融：
  raw  只有块原文
  hdr  结构化抬头 + 块原文
  ctx  结构化抬头 + 上下文说明 + 块原文（没有说明的块退化为 hdr）
三个版本的文字不同，BM25 的词频和 IDF 也不同，所以分成独立的集合，而不是在一个集合里存三组向量。

版本切换：每次重建都建新集合（名字带时间戳），写完、核对数量后，
再把别名 kb_raw / kb_hdr / kb_ctx 一步切到新集合。查询方只认别名：
切换是原子的，不会读到一半新一半旧；新集合有问题时把别名切回去即可回滚。

用法：
    python scripts/kb_index.py                       # 三个版本全部重建并切换别名
    python scripts/kb_index.py --variants ctx        # 只重建 ctx
    python scripts/kb_index.py --prune               # 重建后删除不再被别名指向的旧集合
    python scripts/kb_index.py --set-current hdr     # 把 kb_current 指向 hdr 版本（D51 选定的最终配置）
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from datetime import datetime

from qdrant_client import QdrantClient, models

import kb_bm25
from db import connect
from kb_context import structural_header
from kb_embed import get_embedder

VARIANTS = ("raw", "hdr", "ctx")
OPEN_END = "9999-12-31T00:00:00Z"        # 仍然有效的版本：用一个很远的日期代替"没有截止日"，过滤条件只需一个区间


def qdrant() -> QdrantClient:
    if os.environ.get("QDRANT_PATH"):                    # 本地模式：测试用，不需要启动服务
        return QdrantClient(path=os.environ["QDRANT_PATH"])
    return QdrantClient(url=os.environ.get("QDRANT_URL", "http://localhost:6333"), timeout=60)


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"sentryaml/{chunk_id}"))


def load_chunks() -> list[dict]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT c.chunk_id, c.doc_id, d.family, d.title, d.version_label, d.effective_from, d.effective_to,
                   c.unit_ids, c.kind, c.heading_path, c.text, c.context, c.page_start, c.page_end
            FROM kb_chunks c JOIN kb_documents d USING (doc_id)
            ORDER BY c.doc_id, c.chunk_id""")
        cols = [d.name for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def variant_text(c: dict, variant: str) -> str:
    if variant == "raw":
        return c["text"]
    header = structural_header({"title": c["title"], "version_label": c["version_label"]}, c)
    if variant == "ctx" and c["context"]:
        return f"{header}\n\n{c['context']}\n\n{c['text']}"
    return f"{header}\n\n{c['text']}"


def iso(d) -> str:
    return f"{d.isoformat()}T00:00:00Z" if d else OPEN_END


def build(client: QdrantClient, embedder, chunks: list[dict], variant: str, tag: str) -> str:
    name = f"kb_{variant}_{tag}"
    texts = [variant_text(c, variant) for c in chunks]
    print(f"\n[{variant}] 集合 {name}：{len(texts)} 块，平均 {sum(map(len, texts)) / len(texts):,.0f} 字符")

    dense = embedder.embed(texts, progress=True)
    avgdl = kb_bm25.avg_len(texts)
    sparse = [kb_bm25.doc_vector(t, avgdl) for t in texts]
    print(f"    BM25：平均每块 {avgdl:.0f} 个词，平均 {sum(len(i) for i, _ in sparse) / len(sparse):.0f} 个不同词")

    client.create_collection(
        name,
        vectors_config={"dense": models.VectorParams(size=len(dense[0]), distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    # 过滤字段建索引：按日期、文件族过滤是每次查询都会用的条件
    for field, schema in (("family", models.PayloadSchemaType.KEYWORD),
                          ("doc_id", models.PayloadSchemaType.KEYWORD),
                          ("unit_ids", models.PayloadSchemaType.KEYWORD),
                          ("effective_from", models.PayloadSchemaType.DATETIME),
                          ("effective_to", models.PayloadSchemaType.DATETIME)):
        client.create_payload_index(name, field_name=field, field_schema=schema)

    points = [
        models.PointStruct(
            id=point_id(c["chunk_id"]),
            vector={"dense": dv, "bm25": models.SparseVector(indices=si, values=sv)},
            payload={"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "family": c["family"],
                     "unit_ids": c["unit_ids"], "kind": c["kind"], "heading_path": c["heading_path"],
                     "effective_from": iso(c["effective_from"]), "effective_to": iso(c["effective_to"]),
                     "page_start": c["page_start"], "page_end": c["page_end"],
                     "text": c["text"], "has_context": bool(c["context"]), "variant": variant})
        for c, dv, (si, sv) in zip(chunks, dense, sparse)]
    for i in range(0, len(points), 128):
        client.upsert(name, points=points[i:i + 128], wait=True)

    n = client.count(name, exact=True).count
    if n != len(chunks):
        raise SystemExit(f"    ✗ {name} 写入 {n} 个点，应为 {len(chunks)}，不切换别名")
    print(f"    ✓ 写入 {n} 个点")
    return name


def swap_alias(client: QdrantClient, alias: str, collection: str) -> str | None:
    old = next((a.collection_name for a in client.get_aliases().aliases if a.alias_name == alias), None)
    ops = []
    if old:
        ops.append(models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=alias)))
    ops.append(models.CreateAliasOperation(create_alias=models.CreateAlias(collection_name=collection, alias_name=alias)))
    client.update_collection_aliases(change_aliases_operations=ops)       # 一次请求里删旧建新：原子切换
    print(f"    别名 {alias}：{old or '（无）'} → {collection}")
    return old


def main():
    ap = argparse.ArgumentParser(description="建立 Qdrant 检索索引")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M"))
    ap.add_argument("--prune", action="store_true", help="删除不再被别名指向的旧集合")
    ap.add_argument("--set-current", choices=VARIANTS,
                    help="不重建，只把别名 kb_current 指向该版本当前所在的集合（Agent 检索的入口）")
    args = ap.parse_args()

    if args.set_current:
        client = qdrant()
        target = next((a.collection_name for a in client.get_aliases().aliases
                       if a.alias_name == f"kb_{args.set_current}"), None)
        if target is None:
            raise SystemExit(f"别名 kb_{args.set_current} 不存在，先建索引")
        swap_alias(client, "kb_current", target)
        client.close()
        return

    chunks = load_chunks()
    n_ctx = sum(bool(c["context"]) for c in chunks)
    print(f"检索块 {len(chunks)}，其中有上下文说明 {n_ctx}")
    client, embedder = qdrant(), get_embedder()
    t0 = time.perf_counter()
    for v in args.variants:
        name = build(client, embedder, chunks, v, args.tag)
        swap_alias(client, f"kb_{v}", name)

    if args.prune:
        live = {a.collection_name for a in client.get_aliases().aliases}
        for col in client.get_collections().collections:
            if col.name.startswith("kb_") and col.name not in live:      # kb_current 指向的集合也算"仍在使用"
                client.delete_collection(col.name)
                print(f"  删除旧集合 {col.name}")
    print(f"\n完成，用时 {time.perf_counter() - t0:.0f}s")
    client.close()


if __name__ == "__main__":
    main()