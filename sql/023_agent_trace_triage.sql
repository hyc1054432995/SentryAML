-- ============================================================================
-- SentryAML - 023：追踪表支持告警分诊（R9b-2）
-- ============================================================================
-- 019 的 agent_runs 是为法规研究设计的，有两处不适合分诊：
--
-- 1. as_of 是 DATE。法规研究的"调查日期"精确到天就够了（文件版本按天生效）；
--    分诊的 as_of 是告警时刻，精确到分钟：9/9 10:00 的告警和 9/9 16:00 的告警能看到的交易不同（R9a 的回流例子）。
--    存成 DATE 会把时刻截掉，事后审计时就无法还原"这次调查当时能看到什么"。
--    改为 TIMESTAMP：已有的法规研究记录会变成当天 00:00，含义不变；按日期查询（as_of = '2025-03-01'）照样能匹配。
--
-- 2. 没有告警编号。分诊运行要能按告警查（"告警 #1906 被调查过几次、每次结论是什么"），
--    放在 question 文字里只能做字符串匹配。加一列 alert_id，法规研究为 NULL。
--
-- 状态（status）不加新值：分诊结论（升级 / 关闭）是业务结果，放在 summary 里；
-- status 只表示这次运行本身怎么结束（正常完成 done / 出错 error），两种 Agent 含义一致。
--
-- 用法：psql ... -f sql/023_agent_trace_triage.sql   （可重复执行）
-- ============================================================================

DO $$
BEGIN
    IF (SELECT data_type FROM information_schema.columns
        WHERE table_name = 'agent_runs' AND column_name = 'as_of') = 'date' THEN
        ALTER TABLE agent_runs ALTER COLUMN as_of TYPE TIMESTAMP USING as_of::timestamp;
    END IF;
END $$;

ALTER TABLE agent_runs ADD COLUMN IF NOT EXISTS alert_id INT;
CREATE INDEX IF NOT EXISTS agent_runs_alert ON agent_runs (alert_id) WHERE alert_id IS NOT NULL;

COMMENT ON COLUMN agent_runs.as_of IS '调查时点：法规研究为调查日期（00:00），分诊为告警时刻；由系统绑定，模型不能修改';
COMMENT ON COLUMN agent_runs.alert_id IS 'R9b：分诊运行对应的告警；法规研究为 NULL';