-- ============================================================
-- 洗钱模式的时间与连接性诊断
-- ============================================================
-- 目的：确定图遍历时能否用"时间严格递增"作为约束。
--
-- 背景：在一个 7 跳 CYCLE 样例中观察到时间倒流
-- （第2跳 09-05 16:25 -> 第3跳 09-04 07:52），
-- 物理上不可能。需要量化这个现象的普遍程度。
-- ============================================================
\pset pager off
\timing on

\echo ''
\echo '=== 1. 时间单调性（按 seq_no 排序后，时间是否递增）==='

WITH stepped AS (
    SELECT
        d.pattern_id,
        p.pattern_type,
        d.seq_no,
        d.txn_ts,
        -- LAG 取"上一行"的值（按 PARTITION 内的 ORDER 定义）。
        -- 这是窗口函数家族里最常用的两个之一（另一个是 LEAD 取下一行）。
        LAG(d.txn_ts) OVER (
            PARTITION BY d.pattern_id ORDER BY d.seq_no
        ) AS prev_ts
    FROM pattern_transactions d
    JOIN laundering_patterns p ON p.pattern_id = d.pattern_id
),
per_pattern AS (
    SELECT
        pattern_id,
        pattern_type,
        COUNT(*) FILTER (WHERE prev_ts IS NOT NULL)                AS n_steps,
        COUNT(*) FILTER (WHERE txn_ts < prev_ts)                   AS n_backwards
    FROM stepped
    GROUP BY pattern_id, pattern_type
)
SELECT
    pattern_type                                                   AS "模式类型",
    COUNT(*)                                                       AS "模式数",
    COUNT(*) FILTER (WHERE n_backwards = 0)                        AS "时间完全递增",
    ROUND(100.0 * COUNT(*) FILTER (WHERE n_backwards = 0)
          / COUNT(*), 1)                                           AS "单调占比%",
    SUM(n_backwards)                                               AS "倒流步数",
    SUM(n_steps)                                                   AS "总步数",
    ROUND(100.0 * SUM(n_backwards) / NULLIF(SUM(n_steps), 0), 1)   AS "倒流步占比%"
FROM per_pattern
GROUP BY pattern_type
ORDER BY 4 DESC;


\echo ''
\echo '=== 2. 链式连接性（前一笔的收款方 = 后一笔的付款方）==='
\echo '（检验"文件顺序 = 资金流顺序"这个假设本身是否成立）'

WITH stepped AS (
    SELECT
        d.pattern_id,
        p.pattern_type,
        d.seq_no,
        d.from_bank, d.from_account,
        LAG(d.to_bank)    OVER w AS prev_to_bank,
        LAG(d.to_account) OVER w AS prev_to_account
    FROM pattern_transactions d
    JOIN laundering_patterns p ON p.pattern_id = d.pattern_id
    WINDOW w AS (PARTITION BY d.pattern_id ORDER BY d.seq_no)
)
SELECT
    pattern_type                                          AS "模式类型",
    COUNT(*) FILTER (WHERE prev_to_bank IS NOT NULL)      AS "相邻步数",
    COUNT(*) FILTER (
        WHERE prev_to_bank = from_bank
          AND prev_to_account = from_account
    )                                                     AS "首尾相接",
    ROUND(100.0 * COUNT(*) FILTER (
        WHERE prev_to_bank = from_bank
          AND prev_to_account = from_account
    ) / NULLIF(COUNT(*) FILTER (WHERE prev_to_bank IS NOT NULL), 0), 1)
                                                          AS "相接占比%"
FROM stepped
GROUP BY pattern_type
ORDER BY 4 DESC;


\echo ''
\echo '=== 3. 时间跨度（决定遍历时间窗要开多大）==='

WITH span AS (
    SELECT
        p.pattern_type,
        p.pattern_id,
        MAX(d.txn_ts) - MIN(d.txn_ts) AS duration
    FROM pattern_transactions d
    JOIN laundering_patterns p ON p.pattern_id = d.pattern_id
    GROUP BY p.pattern_type, p.pattern_id
)
SELECT
    pattern_type                                                        AS "模式类型",
    COUNT(*)                                                            AS "模式数",
    ROUND(EXTRACT(EPOCH FROM MIN(duration)) / 3600, 1)                  AS "最短(小时)",
    ROUND(EXTRACT(EPOCH FROM PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY duration
    )) / 3600, 1)                                                       AS "中位数(小时)",
    ROUND(EXTRACT(EPOCH FROM MAX(duration)) / 3600, 1)                  AS "最长(小时)"
FROM span
GROUP BY pattern_type
ORDER BY 4 DESC;
-- EXTRACT(EPOCH FROM interval) 把时间间隔转成总秒数，
-- 除以 3600 得到小时。interval 类型不能直接参与数值运算，
-- 必须先转成秒。


\echo ''
\echo '=== 4. 如果加"时间严格递增"约束，会损失多少模式 ==='

WITH stepped AS (
    SELECT d.pattern_id,
           d.txn_ts,
           LAG(d.txn_ts) OVER (PARTITION BY d.pattern_id ORDER BY d.seq_no) AS prev_ts
    FROM pattern_transactions d
),
flagged AS (
    SELECT pattern_id,
           BOOL_OR(txn_ts < prev_ts) AS has_backward
    FROM stepped
    GROUP BY pattern_id
)
SELECT
    COUNT(*)                                            AS "模式总数",
    COUNT(*) FILTER (WHERE NOT has_backward)            AS "时间单调",
    COUNT(*) FILTER (WHERE has_backward)                AS "含倒流",
    ROUND(100.0 * COUNT(*) FILTER (WHERE has_backward) / COUNT(*), 1) AS "含倒流占比%"
FROM flagged;
-- BOOL_OR 是布尔聚合：组内只要有一个 true，结果就是 true。
-- 配套的还有 BOOL_AND（全部为 true 才是 true）。

\timing off