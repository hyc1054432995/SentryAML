-- ============================================================================
-- SentryAML - 017：沿交叉引用扩展候选块 + 重排分数缓存（RAG 层 R6b）
-- ============================================================================
-- 1. kb_expand_chunk(块, 调查日期)：给定一个已检索到的块，沿 R2 抽取的交叉引用找"相邻"的块。
--
--    两个方向：
--      out  这个块引用了谁。例：Guidelines 6-10-13 写"comply with paragraph 6.25 of the Notice"
--           → 把 Notice 6.25 所在的块带进来
--      in   谁引用了这个块。例：检索到 Notice 6.22，而 TM 指引脚注 3 写"See Paragraphs 6.19-6.26 of
--           MAS Notice 626" → 把 TM 指引中带有脚注 3 的块带进来（R6a 失败题 q032 正是缺了这一块）
--
--    "被引用"不只看块本身的条款，还看它所在的标题：Guidelines 14-3 写"set out in Appendix B"，
--    引用目标是附录 B 这个标题，而不是 B-2 这一段。检索到 B-2 时，也应该能找回 14-3（失败题 q029）。
--
--    但宽泛的标题不算：Notice 第 6 章被 58 处引用（"paragraph 6 of the Notice"），
--    检索到第 6 章里的任何一段都把这 58 处带进来，等于噪声。所以只沿"被引用次数 ≤ p_max_fanin"
--    的目标扩展。思路和 R3 的特异度一样：被很多地方指向的目标，指向关系本身几乎不携带信息。
--
--    link_unit：返回的块里"承载这条引用"的那一条（例如 TM 指引 2.10 这一块里的脚注 3）。
--    重排时会单独拿这一条去打分（D50）：脚注附在一个讲别的事的长块末尾时，整块读起来并不相关。
--
--    版本：引用目标按文件族记录（D43），这里按调查日期选出当时有效的版本；
--    反向引用的来源也必须是当时有效的文件。
--
-- 2. kb_rerank_cache：重排分数缓存，键 = sha256(模型 | 问题 | 文档文字)。评测会反复跑同样的问题，
--    重排要调 API，结果不变就不重复付费。
--
-- 用法：psql ... -f sql/017_kb_expand.sql   （可重复执行）
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
    ORDER BY r.chunk_id, r.direction
$$;

CREATE TABLE IF NOT EXISTS kb_rerank_cache (
    cache_key   CHAR(64) PRIMARY KEY,
    model       TEXT NOT NULL,
    score       REAL NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON FUNCTION kb_expand_chunk IS 'R6b：沿交叉引用（引用与被引用）找相邻的块，按调查日期选版本，跳过被宽泛引用的目标';
COMMENT ON TABLE kb_rerank_cache IS 'R6b：重排分数缓存，键为 sha256(模型 | 问题 | 文档文字)';