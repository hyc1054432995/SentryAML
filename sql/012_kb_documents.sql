-- ============================================================================
-- SentryAML - 012：知识库文档登记表（RAG 层 R1）
-- ============================================================================
-- 这张表是知识库的"唯一真相来源"（system of record）：
--   每一份进入知识库的文件是哪个版本、何时生效、内容哈希是多少，都记在这里。
--   Qdrant 里的向量索引只是由它派生出来的，随时可以删掉重建。
--
-- 三个设计点：
--   1. 版本用"生效区间"表示：[effective_from, effective_to)，effective_to 为空表示仍然有效。
--      Agent 调查某个日期的交易时，只能引用当时有效的版本（as-of 查询）。
--   2. 同一份文件（family）的不同版本，生效区间不允许重叠——由数据库用 EXCLUDE 约束强制，
--      而不是靠写代码的人自觉。
--   3. sha256 是文件内容的指纹：文件名可以不变而内容被替换，哈希不会骗人。
--      哈希变了 → 状态改成 changed → 后续步骤只重建这一份的索引（增量更新）。
--
-- 用法：psql ... -f sql/012_kb_documents.sql   （可重复执行）
-- ============================================================================

-- btree_gist：让 EXCLUDE 约束能同时比较普通列（=）和区间（&&）
CREATE EXTENSION IF NOT EXISTS btree_gist;

CREATE TABLE IF NOT EXISTS kb_documents (
    doc_id          TEXT PRIMARY KEY,               -- 例：mas_n626_2025
    family          TEXT NOT NULL,                  -- 同一文件的不同版本共用，例：mas_n626
    title           TEXT NOT NULL,
    issuer          TEXT NOT NULL,                  -- MAS / FATF / SPF / IBM Research
    doc_type        TEXT NOT NULL CHECK (doc_type IN
                      ('notice', 'guidelines', 'guidance', 'typology', 'str_guidance', 'dataset_paper')),
    jurisdiction    TEXT NOT NULL CHECK (jurisdiction IN ('SG', 'INTL')),
    version_label   TEXT NOT NULL,                  -- 文件自己写的版本说明，例："Last revised on 30 June 2025"
    issued_date     DATE,
    effective_from  DATE NOT NULL,
    effective_to    DATE,                           -- NULL = 仍然有效
    source_url      TEXT NOT NULL,
    file_path       TEXT NOT NULL,                  -- 相对项目根目录
    sha256          CHAR(64) NOT NULL,
    n_pages         INT  NOT NULL CHECK (n_pages > 0),
    n_bytes         BIGINT NOT NULL CHECK (n_bytes > 0),
    status          TEXT NOT NULL DEFAULT 'registered' CHECK (status IN
                      ('registered', 'changed', 'parsed', 'indexed')),
    registered_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT kb_doc_valid_range CHECK (effective_to IS NULL OR effective_to > effective_from),
    -- 同一 family 的生效区间不能重叠：任意一天最多只有一个版本有效
    CONSTRAINT kb_doc_no_overlap EXCLUDE USING gist (
        family WITH =,
        daterange(effective_from, effective_to, '[)') WITH &&
    )
);

COMMENT ON TABLE  kb_documents IS 'RAG 知识库文档登记：版本、生效区间、内容哈希。向量索引由它派生';
COMMENT ON COLUMN kb_documents.effective_to IS '生效区间右端（不含）。NULL 表示仍然有效';
COMMENT ON COLUMN kb_documents.sha256 IS '文件内容哈希，用于变更检测和审计';

-- 当前有效的版本
CREATE OR REPLACE VIEW kb_documents_current AS
SELECT * FROM kb_documents WHERE effective_to IS NULL;

-- as-of 查询：某一天有效的全部文件
-- 用法：SELECT * FROM kb_documents_as_of(DATE '2025-03-01');
CREATE OR REPLACE FUNCTION kb_documents_as_of(d DATE)
RETURNS SETOF kb_documents
LANGUAGE sql STABLE AS $$
    SELECT * FROM kb_documents
    WHERE daterange(effective_from, effective_to, '[)') @> d
$$;