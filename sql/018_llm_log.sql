-- ============================================================================
-- SentryAML - 018：大模型调用缓存 + 回答日志（RAG 层 R7）
-- ============================================================================
-- 1. llm_cache：大模型调用的结果缓存
--      键 = sha256(模型 | 完整的消息列表 | 参数)。消息里包含提示词版本、检索到的原文、问题，
--      任何一处变化都会产生新键，所以不会误用旧结果。
--    为什么需要：评测会反复跑同一批题（改评测代码、加指标、做对比），提示词和检索结果没变时，
--      重跑应当零成本、并且得到完全相同的回答——这让"改评测代码后重跑"不会引入新的随机性。
--    只缓存正常结束（finish_reason = 'stop'）的调用；被截断的结果不缓存，下次重新调用。
--
-- 2. kb_answer_log：每次回答的完整记录（问题、调查日期、范围检查、来源、每条论断及核验结果、耗时、token）
--    为什么需要：合规场景要求可追溯——事后要能回答"当时系统依据哪几条原文、怎么得出这个结论"。
--      R8 的 Agent 也会把每次检索问答写进这里，作为调查过程的审计记录。
--
-- 用法：psql ... -f sql/018_llm_log.sql   （可重复执行）
-- ============================================================================

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key          CHAR(64) PRIMARY KEY,
    model              TEXT NOT NULL,
    purpose            TEXT NOT NULL,                 -- scope / answer / answer_retry ……
    response           TEXT NOT NULL,
    prompt_tokens      INT,
    cache_hit_tokens   INT,
    completion_tokens  INT,
    reasoning_tokens   INT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS kb_answer_log (
    answer_id      BIGSERIAL PRIMARY KEY,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    question       TEXT NOT NULL,
    as_of          DATE NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('answered', 'partial', 'abstained')),
    abstain_reason TEXT,
    prompt_version TEXT NOT NULL,
    model          TEXT NOT NULL,
    record         JSONB NOT NULL                     -- 完整的回答对象（来源、论断、核验、耗时、token）
);
CREATE INDEX IF NOT EXISTS kb_answer_log_time ON kb_answer_log (created_at);

COMMENT ON TABLE llm_cache IS 'R7：大模型调用缓存，键为 sha256(模型 | 消息 | 参数)，只缓存正常结束的调用';
COMMENT ON TABLE kb_answer_log IS 'R7：每次带引用回答的完整审计记录';