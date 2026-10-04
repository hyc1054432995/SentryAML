-- ============================================================================
-- SentryAML - 013：知识库的结构单元、检索块、交叉引用（RAG 层 R2）
-- ============================================================================
-- 三张表的分工：
--   kb_units   结构单元：引用的最小单位（14.2、2.1:beneficial owner、box-9、p65.4、fn14）
--   kb_chunks  检索块：向量化和召回的单位；一个块包含一个或几个单元（unit_ids）
--   kb_xrefs   交叉引用：条款 → 条款的边；目标只记"文件族 + 编号"，查询时按日期选版本
--
-- 全部挂在 kb_documents 上，ON DELETE CASCADE：删掉一份文件的登记，它的单元、块、引用一起消失。
-- 用法：psql ... -f sql/013_kb_units.sql   （可重复执行）
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb_units (
    doc_id         TEXT NOT NULL REFERENCES kb_documents(doc_id) ON DELETE CASCADE,
    unit_id        TEXT NOT NULL,
    kind           TEXT NOT NULL CHECK (kind IN
                     ('front', 'heading', 'paragraph', 'definition', 'block', 'box', 'footnote', 'endnote')),
    heading_path   TEXT[] NOT NULL,
    text           TEXT NOT NULL,
    page_start     INT NOT NULL,
    page_end       INT NOT NULL,
    parent_id      TEXT,
    footnote_refs  INT[] NOT NULL DEFAULT '{}',
    amended_by     TEXT[] NOT NULL DEFAULT '{}',
    ord            INT NOT NULL,                  -- 在文件中的先后顺序
    content_hash   CHAR(64) NOT NULL,             -- 归一化文本的哈希：跨版本、跨排版识别"同一段内容"
    PRIMARY KEY (doc_id, unit_id),
    CHECK (page_end >= page_start)
);
CREATE INDEX IF NOT EXISTS kb_units_hash ON kb_units (content_hash);

CREATE TABLE IF NOT EXISTS kb_chunks (
    chunk_id       TEXT PRIMARY KEY,              -- <doc_id>::<首个 unit_id>[#序号]
    doc_id         TEXT NOT NULL REFERENCES kb_documents(doc_id) ON DELETE CASCADE,
    unit_ids       TEXT[] NOT NULL,
    kind           TEXT NOT NULL,
    heading_path   TEXT[] NOT NULL,
    text           TEXT NOT NULL,
    page_start     INT NOT NULL,
    page_end       INT NOT NULL,
    amended_by     TEXT[] NOT NULL DEFAULT '{}',
    n_chars        INT NOT NULL,
    context        TEXT,                          -- R5：上下文增强生成的说明
    indexed_at     TIMESTAMPTZ                    -- R5：写入向量库的时间
);
CREATE INDEX IF NOT EXISTS kb_chunks_doc ON kb_chunks (doc_id);
CREATE INDEX IF NOT EXISTS kb_chunks_units ON kb_chunks USING gin (unit_ids);   -- "哪个块包含 14.2"

CREATE TABLE IF NOT EXISTS kb_xrefs (
    xref_id         BIGSERIAL PRIMARY KEY,
    src_doc_id      TEXT NOT NULL,
    src_unit_id     TEXT NOT NULL,
    kind            TEXT NOT NULL CHECK (kind IN ('internal', 'cross_doc', 'external')),
    target_family   TEXT NOT NULL,
    target_unit_id  TEXT NOT NULL,
    raw             TEXT NOT NULL,                -- 原文里的引用写法
    resolved        BOOLEAN NOT NULL,
    FOREIGN KEY (src_doc_id, src_unit_id) REFERENCES kb_units (doc_id, unit_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS kb_xrefs_src ON kb_xrefs (src_doc_id, src_unit_id);
CREATE INDEX IF NOT EXISTS kb_xrefs_tgt ON kb_xrefs (target_family, target_unit_id);

COMMENT ON TABLE kb_units  IS 'RAG 结构单元：可精确引用的最小单位';
COMMENT ON TABLE kb_chunks IS 'RAG 检索块：一个或多个结构单元，向量化的单位';
COMMENT ON TABLE kb_xrefs  IS 'RAG 交叉引用：条款到条款的边，目标按文件族记录，查询时按日期选版本';

-- ----------------------------------------------------------------------------
-- 沿引用走一跳：给定一个单元和调查日期，返回它引用的、在那一天有效的条款原文
-- 例：SELECT * FROM kb_follow_refs('mas_tm_guidance', 'fn3', DATE '2025-03-01');
--     → MAS Notice 626 第 6.19–6.26 段，取 2024 版（2025-03-01 那天有效的版本）
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION kb_follow_refs(p_doc TEXT, p_unit TEXT, p_as_of DATE)
RETURNS TABLE (kind TEXT, raw TEXT, target_doc_id TEXT, target_unit_id TEXT, target_text TEXT)
LANGUAGE sql STABLE AS $$
    SELECT x.kind, x.raw, u.doc_id, u.unit_id, u.text
    FROM kb_xrefs x
    JOIN kb_documents_as_of(p_as_of) d ON d.family = x.target_family
    JOIN kb_units u ON u.doc_id = d.doc_id AND u.unit_id = x.target_unit_id
    WHERE x.src_doc_id = p_doc AND x.src_unit_id = p_unit
    ORDER BY u.ord
$$;