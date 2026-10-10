-- ============================================================================
-- SentryAML - 024：分诊 Agent 角色可以读手法知识图谱（R9b-3）
-- ============================================================================
-- 确定性首轮（scripts/triage_seed.py）要用手法知识图谱（014）做两件事：
--   1. 把模型的 SHAP 特征对应到风险信号（kb_signals.features）
--   2. 由已经查实的信号找可能的手法（kb_typologies_for_signals，特异度加权，D44）
-- 这些查询也走 sentry_agent 角色（connect_agent）：所有进入分诊 Agent 上下文的数据，都经过同一个最小权限角色，
-- "Agent 读不到标签"这条保证就只需要在一个地方检查（D68）。
--
-- 只开放知识库的表：手法图谱 7 张表 + 视图，以及 R9b-4 "手法卡片"要用的文件与条款表（法规原文，不含任何交易标签）。
-- kb_typologies_for_signals、kb_typology_citations、kb_documents_as_of 都是 SQL 函数，以调用者权限执行，
-- 所以需要的是这些表的 SELECT 权限；函数的 EXECUTE 默认对所有角色开放。
--
-- 用法：psql ... -f sql/024_agent_kb_grants.sql   （可重复执行）
-- ============================================================================

GRANT SELECT ON kb_typologies, kb_indicators, kb_signals, kb_tools,
                kb_typology_indicators, kb_typology_signals, kb_signal_tools, kb_signal_tool_plan,
                kb_documents, kb_units
TO sentry_agent;