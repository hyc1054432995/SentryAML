-- ============================================================
-- 时间索引方案对比实验
-- ============================================================
\timing on
\pset pager off

\echo ''
\echo '=== 实验 1: COUNT(*)，当前计划（预期走 idx_txn_from 索引全扫）==='
EXPLAIN ANALYZE
SELECT COUNT(*) FROM transactions
WHERE txn_ts >= '2022-09-05' AND txn_ts < '2022-09-06';

\echo ''
\echo '=== 实验 2: 同样的 COUNT(*)，禁用索引扫描，逼它用 BRIN ==='
SET enable_indexscan = off;
SET enable_indexonlyscan = off;
EXPLAIN ANALYZE
SELECT COUNT(*) FROM transactions
WHERE txn_ts >= '2022-09-05' AND txn_ts < '2022-09-06';
RESET enable_indexscan;
RESET enable_indexonlyscan;

\echo ''
\echo '=== 实验 3: 真实场景——需要 amount_paid，无法走覆盖索引 ==='
EXPLAIN ANALYZE
SELECT COUNT(*), SUM(amount_paid) FROM transactions
WHERE txn_ts >= '2022-09-05' AND txn_ts < '2022-09-06';

\timing off