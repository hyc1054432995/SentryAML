-- ============================================================================
-- SentryAML - 015：检索块的上下文说明缓存（RAG 层 R5a）
-- ============================================================================
-- 上下文说明（Contextual Retrieval）由大模型为每个检索块生成，写回 kb_chunks.context。
-- 但 kb_chunks 每次 kb_load 都会整份删掉重写（D43），context 列会跟着丢失。
-- 所以生成结果单独存在这张缓存表里，按"内容"而不是按 chunk_id 索引：
--
--   cache_key = sha256(提示词版本 | 模型 | 文件族 | 归一化的块文本)
--
-- 带来三个好处：
--   1. 重新入库后，只需从缓存回填，不再调用大模型
--   2. 新版本文件里内容没变的块（2025 版 Notice 有 193 段与 2024 版相同）直接复用，只为改动的块付费
--   3. 改了提示词或换了模型，键自动变化，旧结果不会被误用
--
-- 用法：psql ... -f sql/015_kb_context.sql   （可重复执行）
-- ============================================================================

CREATE TABLE IF NOT EXISTS kb_chunk_context (
    cache_key          CHAR(64) PRIMARY KEY,
    prompt_version     TEXT NOT NULL,
    model              TEXT NOT NULL,
    family             TEXT NOT NULL,
    sample_chunk_id    TEXT NOT NULL,              -- 第一次生成时对应的块，便于人工查看
    context            TEXT,                       -- 通过校验的说明；被拒绝时为 NULL
    status             TEXT NOT NULL CHECK (status IN ('ok', 'rejected')),
    reject_reason      TEXT,
    attempts           INT NOT NULL DEFAULT 1,
    prompt_tokens      INT,
    cache_hit_tokens   INT,                        -- 命中服务端前缀缓存的输入 token
    completion_tokens  INT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- 第二轮增加的诊断字段（ctx-v1 有约三分之一的输出无法解析，需要看到结束原因和思考 token 才能定位）
ALTER TABLE kb_chunk_context ADD COLUMN IF NOT EXISTS reasoning_tokens INT;
ALTER TABLE kb_chunk_context ADD COLUMN IF NOT EXISTS finish_reason TEXT;
CREATE INDEX IF NOT EXISTS kb_chunk_context_status ON kb_chunk_context (status);

COMMENT ON TABLE kb_chunk_context IS 'R5a 检索块上下文说明的缓存：按内容哈希索引，跨重新入库、跨文件版本复用';