-- ============================================================================
-- SentryAML - 016：稠密向量缓存（RAG 层 R5b）
-- ============================================================================
-- 和 015 的上下文说明缓存是同一个思路：按"内容"而不是按块编号索引。
--
--   text_hash = sha256(向量模型名 | 被向量化的完整文字)
--
-- 为什么需要它：
--   Qdrant 里的索引是派生数据，每次重建（换切块规则、加新版本文件、做消融）都会整份重建。
--   向量要调 API 计算，文字没变就不应该重新付费、重新等待。
--   2025 版文件里与 2024 版相同的块、三个消融版本里相同的文字，都只算一次。
--
-- 为什么存在 Postgres 而不是 Qdrant：Qdrant 的集合会被整份删除重建，缓存必须活得比索引久。
-- 用法：psql ... -f sql/016_kb_embeddings.sql   （可重复执行）
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb_embeddings (
    text_hash   CHAR(64) PRIMARY KEY,
    model       TEXT NOT NULL,
    dim         INT NOT NULL,
    vector      REAL[] NOT NULL,
    n_chars     INT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (array_length(vector, 1) = dim)
);

COMMENT ON TABLE kb_embeddings IS 'R5b 稠密向量缓存：按 (模型, 文字) 的哈希索引，重建 Qdrant 索引时不重复调用向量 API';