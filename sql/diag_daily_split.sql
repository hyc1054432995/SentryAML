-- ============================================================================
-- SentryAML - 诊断：按天分布，决定训练/验证/测试的时间切分点（M1）
-- ============================================================================
-- 执行:
--   docker compose exec -T postgres psql -U sentry -d sentryaml -f /sql/diag_daily_split.sql
--
-- 回答三个问题：
--   1. 前几天是不是生成器的"预热期"，要不要整段排除？
--   2. 每一段各有多少洗钱样本，够不够训练、够不够可靠地评测？
--   3. 在不同切分日下，有多少条洗钱链会被切成两半？
--
-- 只读查询，不改变数据库。
-- ============================================================================

\timing on
\pset pager off


-- ============================================================================
-- 1. 按天分布
-- ============================================================================
\echo ''
\echo '============================================================'
\echo '1. 每天的交易量、自环占比、洗钱量'
\echo '============================================================'

WITH daily AS (
    SELECT
        txn_ts::date                                              AS d,
        COUNT(*)                                                  AS n_all,
        COUNT(*) FILTER (WHERE is_self_loop)                      AS n_selfloop,
        COUNT(*) FILTER (WHERE NOT is_self_loop)                  AS n_edges,
        COUNT(*) FILTER (WHERE is_laundering = 1 AND NOT is_self_loop) AS n_laund,
        MIN(txn_ts)                                               AS first_ts,
        MAX(txn_ts)                                               AS last_ts
    FROM transactions
    GROUP BY 1
),
labeled AS (
    -- 有模式标签的洗钱交易，按天
    SELECT t.txn_ts::date AS d, COUNT(*) AS n_pattern
    FROM pattern_transactions p
    JOIN transactions t ON t.txn_id = p.txn_id
    WHERE NOT t.is_self_loop
    GROUP BY 1
)
SELECT
    d.d                                                        AS "日期",
    d.n_all                                                    AS "全部交易",
    ROUND(100.0 * d.n_selfloop / d.n_all, 1)                   AS "自环%",
    d.n_edges                                                  AS "非自环",
    d.n_laund                                                  AS "洗钱",
    COALESCE(l.n_pattern, 0)                                   AS "其中有模式",
    -- 每万笔非自环交易里有几笔洗钱。用"每万笔"是因为百分比太小不好读
    ROUND(10000.0 * d.n_laund / NULLIF(d.n_edges, 0), 2)       AS "洗钱/万笔",
    -- 累计占比：窗口函数 SUM() OVER (ORDER BY ...) 会从第一行累加到当前行，
    -- 用来看"截至这一天，已经覆盖了多少比例的数据"
    ROUND(100.0 * SUM(d.n_edges) OVER (ORDER BY d.d)
          / SUM(d.n_edges) OVER (), 1)                         AS "累计交易%",
    ROUND(100.0 * SUM(d.n_laund) OVER (ORDER BY d.d)
          / SUM(d.n_laund) OVER (), 1)                         AS "累计洗钱%",
    to_char(d.last_ts, 'HH24:MI')                              AS "当天最后一笔"
FROM daily d
LEFT JOIN labeled l ON l.d = d.d
ORDER BY d.d;

\echo ''
\echo '（"当天最后一笔"用来发现不完整的日期：最后一天如果停在下午，就不是完整的一天）'


-- ============================================================================
-- 2. 洗钱模式的时间分布
-- ============================================================================
\echo ''
\echo '============================================================'
\echo '2. 洗钱模式在哪几天开始、在哪几天结束'
\echo '============================================================'

WITH spans AS (
    SELECT p.pattern_id,
           MIN(t.txn_ts)::date AS start_d,
           MAX(t.txn_ts)::date AS end_d
    FROM pattern_transactions p
    JOIN transactions t ON t.txn_id = p.txn_id
    GROUP BY p.pattern_id
)
SELECT
    d::date                                             AS "日期",
    COUNT(*) FILTER (WHERE start_d = d)                 AS "当天开始的模式",
    COUNT(*) FILTER (WHERE end_d = d)                   AS "当天结束的模式",
    -- 当天仍在进行中的模式：已开始、未结束
    COUNT(*) FILTER (WHERE start_d <= d AND end_d >= d) AS "进行中的模式"
FROM spans
CROSS JOIN generate_series(
    (SELECT MIN(start_d) FROM spans),
    (SELECT MAX(end_d)   FROM spans),
    INTERVAL '1 day'
) AS g(d)
-- generate_series 生成一个连续的日期序列，
-- 保证即使某天没有任何模式开始或结束，这一天也会出现在结果里。
GROUP BY d
ORDER BY d;


-- ============================================================================
-- 3. 候选切分日：会切断多少条洗钱链
-- ============================================================================
-- 一条洗钱链 "开始于切分日之前、结束于切分日当天或之后"，就是被切成两半：
-- 前半段进了前一个数据集，后半段进了后一个。这是轻微的泄漏。
-- ============================================================================
\echo ''
\echo '============================================================'
\echo '3. 候选切分日：会切断多少条洗钱链'
\echo '============================================================'
\echo '（切分日 = 后一段的第一天。例如 09-11 表示 "09-10 及以前" 与 "09-11 及以后" 分开）'
\echo ''

WITH spans AS (
    SELECT p.pattern_id,
           MIN(t.txn_ts) AS start_ts,
           MAX(t.txn_ts) AS end_ts
    FROM pattern_transactions p
    JOIN transactions t ON t.txn_id = p.txn_id
    GROUP BY p.pattern_id
),
laund AS (
    SELECT txn_ts FROM transactions WHERE is_laundering = 1 AND NOT is_self_loop
)
SELECT
    c.cut::date                                                   AS "切分日",
    (SELECT COUNT(*) FROM laund WHERE txn_ts <  c.cut)            AS "之前的洗钱",
    (SELECT COUNT(*) FROM laund WHERE txn_ts >= c.cut)            AS "之后的洗钱",
    COUNT(*) FILTER (WHERE s.start_ts < c.cut AND s.end_ts >= c.cut) AS "被切断的模式",
    ROUND(100.0 * COUNT(*) FILTER (WHERE s.start_ts < c.cut AND s.end_ts >= c.cut)
          / COUNT(*), 1)                                          AS "占全部模式%"
FROM spans s
CROSS JOIN (VALUES
    ('2022-09-05'::timestamp), ('2022-09-07'::timestamp), ('2022-09-08'::timestamp),
    ('2022-09-09'::timestamp), ('2022-09-10'::timestamp), ('2022-09-11'::timestamp),
    ('2022-09-12'::timestamp), ('2022-09-13'::timestamp), ('2022-09-14'::timestamp),
    ('2022-09-15'::timestamp)
) AS c(cut)
GROUP BY c.cut
ORDER BY c.cut;

\timing off