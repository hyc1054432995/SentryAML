-- ============================================================================
-- SentryAML - 019：Agent 运行与工具调用的追踪表（R8）
-- ============================================================================
-- 两张表：
--   agent_runs        一次 Agent 运行（一次法规研究、之后 R9 的一条告警分诊）：问题、调查日期、状态、结果摘要
--   agent_tool_calls  这次运行里的每一次工具调用：第几步、哪个工具、参数、返回结果、成功与否、耗时
--
-- 为什么需要：
--   1. 调试。Agent 由模型决定下一步做什么，同样的问题两次运行可能走不同的路径。
--      答错时要能还原"它查了什么、看到了什么、为什么停下"，否则无从判断是工具的问题还是模型的判断问题。
--   2. 审计。合规场景要求事后能回答"系统依据什么得出这个结论"。R7 的 kb_answer_log 记录了回答本身，
--      这里记录得出回答的过程。
--   3. 评测。R8 要比较"检索一次"与"多轮检索"，需要统计每道题调用了几次工具、哪类工具、花了多久。
--
-- 为什么放在 Postgres 而不是 LangSmith、Langfuse 这类追踪服务：
--   数据和知识库、回答日志在同一个库里，可以直接 JOIN（例如"这次运行看过的条款，哪些后来被引用了"）；
--   不把监管调查过程发到第三方服务；不增加一个需要部署的组件。
--   字段按"一次运行 → 多个步骤"组织，与 OpenTelemetry 的 trace / span 结构对应，以后需要时可以导出。
--
-- 用法：psql ... -f sql/019_agent_trace.sql   （可重复执行）
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_runs (
    run_id       UUID PRIMARY KEY,
    kind         TEXT NOT NULL,                     -- research（法规研究）/ smoke（冒烟测试）/ 之后的 triage
    question     TEXT NOT NULL,
    as_of        DATE NOT NULL,                     -- 调查日期：由系统绑定，所有工具共用，模型不能修改
    status       TEXT NOT NULL DEFAULT 'running'
                   CHECK (status IN ('running', 'answered', 'partial', 'abstained', 'error', 'done')),
    started_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at  TIMESTAMPTZ,
    summary      JSONB                              -- 结束时的摘要：调用次数、用量、最终结果的要点
);
CREATE INDEX IF NOT EXISTS agent_runs_time ON agent_runs (started_at);

CREATE TABLE IF NOT EXISTS agent_tool_calls (
    run_id      UUID NOT NULL REFERENCES agent_runs(run_id) ON DELETE CASCADE,
    step        INT  NOT NULL,                      -- 本次运行中的第几次工具调用，从 1 开始
    tool        TEXT NOT NULL,
    args        JSONB NOT NULL,                     -- 模型给出的参数（校验前的原样）
    ok          BOOLEAN NOT NULL,                   -- 参数校验失败、找不到条款、超出预算等都记为 false
    error       TEXT,
    result      JSONB NOT NULL,                     -- 返回给模型的内容（与模型看到的完全一致）
    ms          INT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, step)
);
CREATE INDEX IF NOT EXISTS agent_tool_calls_tool ON agent_tool_calls (tool);

COMMENT ON TABLE agent_runs IS 'R8：Agent 运行记录（一次研究或一次分诊）';
COMMENT ON TABLE agent_tool_calls IS 'R8：每次工具调用的参数、返回结果、耗时；返回结果与模型看到的完全一致';