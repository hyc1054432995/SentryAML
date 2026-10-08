-- ============================================================================
-- SentryAML - 022：引用扩展的确定性（R9b-1，D70）
-- ============================================================================
-- 起因：R9b-1 的回归测试发现同一个问题的检索结果不可复现（D70），根因是 RRF 同分在候选池截断线上的先后
-- 由 Qdrant 内部决定（修正在 kb_search.py）。排查同一类问题时，在这里发现了另一处：
--
--   kb_expand_chunk 用 DISTINCT ON (chunk_id) 给每个相邻块只留一行，ORDER BY 只有 (chunk_id, direction)。
--   一个块通过多条条款与当前块相连时（同方向多行），留下哪一行没有规定，由 Postgres 排序时的输入顺序决定；
--   数据的物理顺序变了（例如重建交叉引用表、VACUUM 之后），留下的那一行就可能变。
--   留下的 link_unit 决定"聚焦重排"给哪一条条款单独打分（D50），所以会影响重排名次。
--
-- 这一处没有直接观察到不稳定（retrieval_ties.py 的扫描只查第一阶段，不经过引用扩展；同一个库、同一个执行计划下结果通常恒定），
-- 但它与 RRF 同分是同一类缺陷："先后没有规定，交给系统内部决定"。一并修掉：
-- ORDER BY 加上 link_unit、via_unit、raw，同方向多行时总是留下编号最小的那条连接。
--
-- 函数其余部分与 017 完全相同。用法：psql ... -f sql/022_kb_expand_deterministic.sql
-- ============================================================================

DROP FUNCTION IF EXISTS kb_expand_chunk(TEXT, DATE, INT);
CREATE OR REPLACE FUNCTION kb_expand_chunk(p_chunk TEXT, p_as_of DATE, p_max_fanin INT DEFAULT 5)
RETURNS TABLE (direction TEXT, via_unit TEXT, raw TEXT, chunk_id TEXT, link_unit TEXT)
LANGUAGE sql STABLE AS $$
    WITH c AS (
        SELECT k.chunk_id, k.doc_id, k.unit_ids, k.heading_path, d.family
        FROM kb_chunks k JOIN kb_documents d USING (doc_id)
        WHERE k.chunk_id = p_chunk
    ),
    live AS (SELECT doc_id, family FROM kb_documents_as_of(p_as_of)),
    -- 这个块自己的条款 + 它所在的各级标题
    targets AS (
        SELECT u AS unit_id FROM c, unnest(c.unit_ids) u
        UNION
        SELECT h.unit_id FROM c JOIN kb_units h
          ON h.doc_id = c.doc_id AND h.kind = 'heading' AND h.text = ANY(c.heading_path)
    ),
    -- 每个目标在当时有效的文件里被引用了多少次（衡量"宽泛"程度）
    fanin AS (
        SELECT x.target_unit_id, COUNT(*) AS n
        FROM kb_xrefs x JOIN c ON x.target_family = c.family
        JOIN live l ON l.doc_id = x.src_doc_id
        WHERE x.target_unit_id IN (SELECT unit_id FROM targets)
        GROUP BY x.target_unit_id
    ),
    outgoing AS (
        SELECT 'out'::TEXT, x.target_unit_id, x.raw, t.chunk_id, x.target_unit_id
        FROM c
        JOIN kb_xrefs x ON x.src_doc_id = c.doc_id AND x.src_unit_id = ANY(c.unit_ids)
                       AND x.kind <> 'external' AND x.resolved
        JOIN live l ON l.family = x.target_family
        JOIN kb_chunks t ON t.doc_id = l.doc_id AND x.target_unit_id = ANY(t.unit_ids)
    ),
    incoming AS (
        SELECT 'in'::TEXT, x.target_unit_id, x.raw, s.chunk_id, x.src_unit_id
        FROM c
        JOIN kb_xrefs x ON x.target_family = c.family
        JOIN targets tg ON tg.unit_id = x.target_unit_id
        JOIN fanin f ON f.target_unit_id = x.target_unit_id AND f.n <= p_max_fanin
        JOIN live l ON l.doc_id = x.src_doc_id
        JOIN kb_chunks s ON s.doc_id = x.src_doc_id AND x.src_unit_id = ANY(s.unit_ids)
    )
    SELECT DISTINCT ON (r.chunk_id) r.*
    FROM (SELECT * FROM outgoing UNION ALL SELECT * FROM incoming) r(direction, via_unit, raw, chunk_id, link_unit), c
    WHERE r.chunk_id <> c.chunk_id
    ORDER BY r.chunk_id, r.direction, r.link_unit, r.via_unit, r.raw   -- 022：同一块多条连接时，选哪一条由这里决定
$$;

COMMENT ON FUNCTION kb_expand_chunk IS 'R6b：沿交叉引用找相邻的块，按调查日期选版本，跳过被宽泛引用的目标；022 起同分连接按编号确定';