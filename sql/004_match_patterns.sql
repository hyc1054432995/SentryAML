-- ============================================================================
-- SentryAML - 004: 把模式明细回连到交易表
-- ============================================================================
-- 执行:
--   docker compose exec -T postgres psql -U sentry -d sentryaml -f /sql/004_match_patterns.sql
--
-- 目标：给 pattern_transactions 的 3,209 行填上 transactions.txn_id。
--
-- patterns 文件里没有 txn_id，只能靠 8 个业务字段值匹配：
--   txn_ts, from_bank, from_account, to_bank, to_account,
--   amount_paid, currency_paid, payment_format
--
-- 匹配不保证一对一，因为：
--   - 时间只精确到分钟
--   - 交易表本身有 9 组完全重复行
-- 所以三种结果都要记录：matched / ambiguous / not_found
--
-- 【not_found 必须是 0】——这些交易百分之百存在于交易表里，
-- 找不到只可能是解析或类型转换有 bug。匹配率就是解析正确性的独立验证。
-- ============================================================================

\timing on
\pset pager off


-- ============================================================================
-- 第 1 步：统计每行明细能匹配到几条交易
-- ============================================================================
-- 临时表（TEMP TABLE）只在当前数据库会话内存在，断开连接自动消失。
-- 适合存放这种中间结果，不用手动清理、也不会污染正式表。
-- ============================================================================
DROP TABLE IF EXISTS tmp_match;

CREATE TEMP TABLE tmp_match AS
SELECT
    d.pattern_id,
    d.seq_no,
    -- COUNT(t.txn_id) 只统计非 NULL 的值。
    -- LEFT JOIN 下匹配不上时 t.txn_id 是 NULL，所以这里会得到 0。
    -- （注意 COUNT(*) 会把那一行也算成 1，是常见的坑）
    COUNT(t.txn_id)                                    AS n_hits,
    -- MIN 在只有一条命中时就是那一条的 id；多条命中时这个值不可信，
    -- 所以下面只在 n_hits = 1 的情况下使用它。
    MIN(t.txn_id)                                      AS matched_txn_id
FROM pattern_transactions d
-- LEFT JOIN：保留 d 的全部 3,209 行。
-- 用 INNER JOIN 的话，匹配不上的行会直接消失，失败就不可见了。
LEFT JOIN transactions t
       ON t.txn_ts            = d.txn_ts
      AND t.from_bank         = d.from_bank
      AND t.from_account      = d.from_account
      AND t.to_bank           = d.to_bank
      AND t.to_account        = d.to_account
      AND t.amount_paid       = d.amount_paid
      AND t.currency_paid     = d.currency_paid
      AND t.payment_format    = d.payment_format
GROUP BY d.pattern_id, d.seq_no;

CREATE INDEX ON tmp_match (pattern_id, seq_no);


-- ============================================================================
-- 第 2 步：把结果写回 pattern_transactions
-- ============================================================================
-- UPDATE ... FROM 是 Postgres 的写法：
--   "用另一张表的数据来更新这张表"，WHERE 里写关联条件。
-- 标准 SQL 里要写成带子查询的形式，Postgres 这个语法更直观也更快。
--
-- CASE WHEN ... THEN ... ELSE ... END 是 SQL 的条件表达式，
-- 相当于其他语言里的 if/elif/else。
-- ============================================================================
UPDATE pattern_transactions d
SET txn_id = CASE WHEN m.n_hits = 1 THEN m.matched_txn_id ELSE NULL END,
    match_status = CASE
        WHEN m.n_hits = 1 THEN 'matched'
        WHEN m.n_hits = 0 THEN 'not_found'
        ELSE 'ambiguous'
    END
FROM tmp_match m
WHERE d.pattern_id = m.pattern_id
  AND d.seq_no     = m.seq_no;


-- ============================================================================
-- 第 3 步：结果统计
-- ============================================================================
\echo ''
\echo '=== 匹配结果 ==='

SELECT
    match_status                                              AS "状态",
    COUNT(*)                                                  AS "行数",
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) || '%' AS "占比"
FROM pattern_transactions
GROUP BY match_status
ORDER BY COUNT(*) DESC;
-- SUM(COUNT(*)) OVER () 是【窗口函数】：
-- OVER () 表示"在全部结果行上"求和，也就是总数。
-- 这样一条查询就能同时算出分组计数和它占总数的比例，
-- 不需要先查一次总数再查一次分组。


\echo ''
\echo '=== 按模式类型看匹配率 ==='

SELECT
    p.pattern_type                                                    AS "类型",
    COUNT(*)                                                          AS "明细数",
    COUNT(*) FILTER (WHERE d.match_status = 'matched')                AS "已匹配",
    COUNT(*) FILTER (WHERE d.match_status = 'ambiguous')              AS "歧义",
    COUNT(*) FILTER (WHERE d.match_status = 'not_found')              AS "未找到"
FROM pattern_transactions d
JOIN laundering_patterns p ON p.pattern_id = d.pattern_id
GROUP BY p.pattern_type
ORDER BY COUNT(*) FILTER (WHERE d.match_status <> 'matched') DESC;
-- COUNT(*) FILTER (WHERE ...) 是"只统计满足条件的行"，
-- 比写一堆 SUM(CASE WHEN ... THEN 1 ELSE 0 END) 清爽得多。
-- 这是 Postgres 支持的标准 SQL 语法。


\echo ''
\echo '=== 未匹配的样例（如果有，说明解析有 bug）==='

SELECT d.pattern_id, d.seq_no, d.txn_ts,
       d.from_bank, d.from_account, d.to_bank, d.to_account,
       d.amount_paid, d.currency_paid, d.payment_format
FROM pattern_transactions d
WHERE d.match_status = 'not_found'
LIMIT 5;


\echo ''
\echo '=== 歧义样例（同一组字段命中多条交易）==='

SELECT d.pattern_id, d.seq_no, d.txn_ts,
       d.from_account, d.to_account, d.amount_paid,
       m.n_hits AS "命中数"
FROM pattern_transactions d
JOIN tmp_match m ON m.pattern_id = d.pattern_id AND m.seq_no = d.seq_no
WHERE d.match_status = 'ambiguous'
ORDER BY m.n_hits DESC
LIMIT 5;


-- ============================================================================
-- 第 4 步：交叉验证
-- ============================================================================
\echo ''
\echo '=== 交叉验证：匹配到的交易是否都被标为洗钱 ==='
-- 逻辑上，模式内的每一笔交易在交易表里都应该 is_laundering = 1。
-- 如果出现 0，说明匹配到了错误的行 —— 这是一个独立于匹配率的正确性检查。

SELECT
    t.is_laundering       AS "is_laundering",
    COUNT(*)              AS "行数"
FROM pattern_transactions d
JOIN transactions t ON t.txn_id = d.txn_id
WHERE d.txn_id IS NOT NULL
GROUP BY t.is_laundering;


\echo ''
\echo '=== 覆盖度：5,177 笔洗钱交易里有多少被模式覆盖 ==='
-- 已知有约 1,968 笔洗钱交易不属于任何标注模式，
-- 这批数据在做"模式分类"任务时要排除，在做"是否洗钱"任务时仍然可用。

SELECT
    (SELECT COUNT(*) FROM transactions WHERE is_laundering = 1)          AS "洗钱交易总数",
    (SELECT COUNT(DISTINCT txn_id) FROM pattern_transactions
      WHERE txn_id IS NOT NULL)                                          AS "被模式覆盖",
    (SELECT COUNT(*) FROM transactions WHERE is_laundering = 1)
      - (SELECT COUNT(DISTINCT txn_id) FROM pattern_transactions
          WHERE txn_id IS NOT NULL)                                      AS "无模式标签";

\timing off