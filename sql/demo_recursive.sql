-- ============================================================
-- 递归 CTE 最小示例
-- ============================================================
\pset pager off
\timing on

\echo ''
\echo '=== 例子 1：纯粹看递归机制（数 1 到 5）==='

WITH RECURSIVE counter AS (
    SELECT 1 AS n
    UNION ALL
    SELECT n + 1 FROM counter WHERE n < 5
)
SELECT n FROM counter;


\echo ''
\echo '=== 例子 2：从一个账户往下走 3 跳 ==='
\echo '起点 (11974, 8011D4F20) 是一个 7 跳 CYCLE 的起点'
\echo ''

WITH RECURSIVE trace AS (
    -- 锚定成员：起点账户的所有直接转出（第 1 跳）
    SELECT
        1                       AS hop,        -- 当前是第几跳
        e.to_bank,
        e.to_account,
        e.amount_usd,
        e.txn_ts
    FROM txn_norm e
    WHERE e.from_bank = 11974
      AND e.from_account = '8011D4F20'

    UNION ALL

    -- 递归成员：从上一跳的终点，继续往下走一跳
    SELECT
        t.hop + 1,
        e.to_bank,
        e.to_account,
        e.amount_usd,
        e.txn_ts
    FROM trace t
    JOIN txn_norm e
      ON e.from_bank    = t.to_bank        -- 上一跳的终点 = 这一跳的起点
     AND e.from_account = t.to_account
    WHERE t.hop < 3                         -- 终止条件：最多走 3 跳
)
SELECT hop AS "第几跳",
       to_bank AS "银行",
       to_account AS "账户",
       ROUND(amount_usd, 2) AS "金额USD",
       txn_ts AS "时间"
FROM trace
ORDER BY hop, txn_ts
LIMIT 40;


\echo ''
\echo '=== 每一跳分别扩散出多少节点 ==='

WITH RECURSIVE trace AS (
    SELECT 1 AS hop, e.to_bank, e.to_account
    FROM txn_norm e
    WHERE e.from_bank = 11974 AND e.from_account = '8011D4F20'

    UNION ALL

    SELECT t.hop + 1, e.to_bank, e.to_account
    FROM trace t
    JOIN txn_norm e
      ON e.from_bank = t.to_bank AND e.from_account = t.to_account
    WHERE t.hop < 4
)
SELECT hop                                          AS "第几跳",
       COUNT(*)                                     AS "路径数",
       COUNT(DISTINCT ROW(to_bank, to_account))     AS "不同账户数"
FROM trace
GROUP BY hop
ORDER BY hop;

\timing off