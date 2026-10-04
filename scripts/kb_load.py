"""
SentryAML - RAG 层 R2d：解析 → 切块 → 交叉引用 → 写入 Postgres（一条命令）
======================================================================
对 kb_documents 里登记的每一份文件：
  1. kb_parser 解析成结构单元（同时写 data/corpus/parsed/*.jsonl，方便人工查看）
  2. 质量门禁：编号断档、重复编号、脚注对不上 → 任何一份不通过，整批不写库
  3. kb_chunk 生成检索块
  4. kb_xrefs 抽取全部文件之间的交叉引用
  5. 在一个事务里：删除这些文件的旧单元/块/引用，写入新的，状态改为 parsed

为什么"先删后写"而不是逐条更新：
  解析规则一改，单元编号可能整体变化（例如修好 11A 之后多出 21 段），逐条比对既复杂又容易漏。
  文件是整体的单位——整份替换，放在一个事务里，要么全部成功，要么什么都不变。

为什么要质量门禁：
  解析出错的数据一旦写进去，后面的向量索引、评测都会建立在错误之上，而且很难察觉。
  宁可这一步报错停下。

用法：
    python scripts/kb_load.py
    python scripts/kb_load.py --only mas_n626_2025 mas_n626_guidelines_2025
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from dataclasses import asdict

import kb_chunk
import kb_parser
import kb_xrefs
from db import connect

KNOWN_EXCEPTIONS = {
    # 已人工确认、可以放行的质量问题
    ("ibm_aml_paper", "footnote_refs_without_text"),   # 作者单位的上标 1/2/3 被当成脚注引用
    ("ibm_aml_paper", "footnotes_unreferenced"),
}


def content_hash(text: str) -> str:
    t = text.replace("“", '"').replace("”", '"').replace("’", "'")
    t = re.sub(r"\[\^\d+\]", "", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return hashlib.sha256(t.encode("utf-8")).hexdigest()


def gate(doc_id: str, rep: dict) -> list[str]:
    problems = []
    for key in ("numbering_gaps", "duplicate_ids", "footnote_refs_without_text", "footnotes_unreferenced"):
        if rep.get(key) and (doc_id, key) not in KNOWN_EXCEPTIONS:
            problems.append(f"{key} = {rep[key]}")
    return problems


def main():
    ap = argparse.ArgumentParser(description="知识库：解析、切块、交叉引用、入库")
    ap.add_argument("--only", nargs="+", help="只处理这几份文件")
    args = ap.parse_args()

    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT doc_id, family FROM kb_documents ORDER BY doc_id")
        registry = dict(cur.fetchall())
    family_map = dict(registry)
    targets = [d for d in registry if not args.only or d in args.only]
    if not targets:
        raise SystemExit("kb_documents 里没有要处理的文件（先运行 register_corpus.py）")

    # ---------------- 1–3. 解析、门禁、切块 ----------------
    t0 = time.perf_counter()
    units_by_doc, chunks_by_doc, failed = {}, {}, {}
    for doc_id in targets:
        units, rep = kb_parser.run(doc_id)
        problems = gate(doc_id, rep)
        if problems:
            failed[doc_id] = problems
            continue
        udicts = [asdict(u) for u in units]
        units_by_doc[doc_id] = udicts
        chunks_by_doc[doc_id] = kb_chunk.build_chunks(doc_id, udicts)
        print(f"  {doc_id:<26} 单元 {len(udicts):>4}  块 {len(chunks_by_doc[doc_id]):>4}")
    if failed:
        print("\n质量门禁未通过，本次不写数据库：")
        for d, ps in failed.items():
            print(f"  ✗ {d}: {'; '.join(ps)}")
        raise SystemExit(1)

    # ---------------- 4. 交叉引用（需要全部文件，才能解析跨文件引用） ----------------
    all_units = dict(units_by_doc)
    for d in registry:                        # --only 时，其他文件用磁盘上的解析结果做引用目标
        if d not in all_units:
            p = kb_parser.OUT_DIR / f"{d}.jsonl"
            if p.exists():
                import json
                with open(p, encoding="utf-8") as f:
                    all_units[d] = [json.loads(l) for l in f]
    edges = [e for e in kb_xrefs.extract(all_units, family_map) if e["src_doc_id"] in units_by_doc]
    print(f"\n解析与切块用时 {time.perf_counter() - t0:.0f}s；交叉引用 {len(edges)} 条")
    kb_xrefs.report(edges)

    # ---------------- 5. 入库（一个事务） ----------------
    t1 = time.perf_counter()
    docs = list(units_by_doc)
    with connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM kb_units  WHERE doc_id = ANY(%s)", (docs,))     # 级联删除引用
        cur.execute("DELETE FROM kb_chunks WHERE doc_id = ANY(%s)", (docs,))
        cur.executemany(
            """INSERT INTO kb_units (doc_id, unit_id, kind, heading_path, text, page_start, page_end,
                                     parent_id, footnote_refs, amended_by, ord, content_hash)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            [(u["doc_id"], u["unit_id"], u["kind"], u["heading_path"], u["text"], u["page_start"],
              u["page_end"], u["parent_id"], u["footnote_refs"], u["amended_by"], u["order"],
              content_hash(u["text"])) for d in docs for u in units_by_doc[d]])
        cur.executemany(
            """INSERT INTO kb_chunks (chunk_id, doc_id, unit_ids, kind, heading_path, text,
                                      page_start, page_end, amended_by, n_chars)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            [(c["chunk_id"], c["doc_id"], c["unit_ids"], c["kind"], c["heading_path"], c["text"],
              c["page_start"], c["page_end"], c["amended_by"], c["n_chars"])
             for d in docs for c in chunks_by_doc[d]])
        cur.executemany(
            """INSERT INTO kb_xrefs (src_doc_id, src_unit_id, kind, target_family, target_unit_id, raw, resolved)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            [(e["src_doc_id"], e["src_unit_id"], e["kind"], e["target_family"], e["target_unit_id"],
              e["raw"], e["resolved"]) for e in edges])
        cur.execute("UPDATE kb_documents SET status = 'parsed', updated_at = now() WHERE doc_id = ANY(%s)",
                    (docs,))
        cur.execute("SELECT (SELECT COUNT(*) FROM kb_units), (SELECT COUNT(*) FROM kb_chunks), "
                    "(SELECT COUNT(*) FROM kb_xrefs)")
        n_u, n_c, n_x = cur.fetchone()
        conn.commit()
    print(f"\n已写入（{time.perf_counter() - t1:.1f}s）：kb_units {n_u}，kb_chunks {n_c}，kb_xrefs {n_x}；"
          f"{len(docs)} 份文件状态改为 parsed")


if __name__ == "__main__":
    main()