-- ============================================================================
-- SentryAML - 009: 多跳资金链路遍历（闭环追踪）
-- ============================================================================
-- 执行:
--   docker compose exec -T postgres psql -U sentry -d sentryaml -f /sql/009_cycle_trace.sql
--
-- 目标：只给起点账户，让遍历自己找出资金闭环，
--       再和 54 个标注好的 CYCLE 对照，看能追出几个、追得对不对。
--
-- 【重要】遍历只使用起点账户，不使用标注里的路径或时间，
--         否则就是拿答案找答案，验证没有意义。
--
-- 【五个约束，缺一不可】
--   1. 环检测     —— 数组记录已访问节点；但【允许回到起点】，那正是要找的闭环
--   2. 时间非递减 —— 下一跳 >= 上一跳。用 >= 是因为时间只精确到分钟
--   3. 7 天窗口   —— CYCLE 跨度中位数 71.9h、最长 95.6h，7 天留足余量
--   4. 清算层剪枝 —— 排除 total_cp_count > 200 的 15 个清算账户（银行 70）
--                   这里直接写阈值，不依赖 is_hub 列，避免 008 里的旧定义
--                   （P99.9=38）把洗钱浓度最高的 11-38 档剪掉
--   5. 最多 12 跳  —— 最初按一个 CYCLE 标题 "Max 10 hops" 设成了 10，
--                   但标注里最长的 CYCLE 有 12 笔（标题里也出现过 "Max 11/12 hops"）。
--                   diag_cycle_control 查出：没追出的 4 个 + 对不上的 1 个，
--                   原因全是"超过 10 跳"。上限改为 12，以数据为准。
--
-- 【第 6 个优化：平行边去重（最早到达）】
--   上一次的小例子里，路径数 1 -> 2 -> 4 翻倍，但不同账户数一直是 1：
--   两个账户之间存在多笔交易（多重图），每多一条平行边，路径数就乘一次。
--   走 10 跳就是 2^10 = 1024 倍的无效膨胀。
--
--   解法：从当前节点出发，对每个【不同的下一跳账户】只保留
--   满足时间约束的【最早】那一笔。
--   为什么不会漏掉闭环：在"时间只能往后走"的约束下，
--   越早到达某个节点，后续可选的交易就越多（后到能走的路，早到一定也能走）。
--   这叫"最早到达（earliest arrival）"，是时序图搜索的标准剪枝。
--   代价：追出的具体交易可能不是标注的那一笔（同一对账户间更早的另一笔），
--         所以第 4 步的精确率检查会把这一点量化出来。
-- ============================================================================

\timing on
\pset pager off

-- 防止意外爆炸把连接挂死。超时会报错退出，不留脏数据。
SET statement_timeout = '600s';


-- ============================================================================
-- 第 1 步：遍历，只保留成功闭环的路径
-- ============================================================================
DROP TABLE IF EXISTS cycle_trace_result CASCADE;

CREATE TABLE cycle_trace_result AS
WITH RECURSIVE

-- 起点：每个 CYCLE 模式第 1 笔交易的付款方
-- （诊断已确认 CYCLE 100% 首尾相接，所以 seq_no=1 的付款方一定在环上）
starts AS (
    SELECT
        p.pattern_id,
        d.from_bank    AS s_bank,
        d.from_account AS s_acct
    FROM laundering_patterns p
    JOIN pattern_transactions d
      ON d.pattern_id = p.pattern_id AND d.seq_no = 1
    WHERE p.pattern_type = 'CYCLE'
),

walk AS (
    -- ---------- 锚定成员：起点账户的每一笔转出都是一个出发点 ----------
    -- 这里【不做】最早到达去重：出发时间不同，7 天窗口的截止时间也不同，
    -- 它们是真正不同的搜索，不能合并。
    -- 但【同一分钟、同一收款方】的平行边是完全相同的搜索，只保留一条，
    -- 否则每条平行边都会把整棵搜索树重复跑一遍。
    -- 整个锚定成员用括号包起来：UNION 前半段带 ORDER BY 时必须加括号，
    -- 否则 SQL 会把 ORDER BY 当成对整个 UNION 结果的排序而报语法错误。
    (SELECT DISTINCT ON (s.pattern_id, e.to_bank, e.to_account, e.txn_ts)
        s.pattern_id,
        s.s_bank,
        s.s_acct,
        e.to_bank              AS cur_bank,
        e.to_account           AS cur_acct,
        1                      AS hop,
        e.txn_ts               AS start_ts,
        e.txn_ts               AS cur_ts,
        -- 节点编码成 "银行:账户" 文本，数组操作比复合类型简单得多
        ARRAY[s.s_bank::text || ':' || s.s_acct,
              e.to_bank::text || ':' || e.to_account] AS path,
        ARRAY[e.txn_id]        AS txn_path,
        FALSE                  AS closed     -- txn_norm 已排除自环，第 1 跳不可能闭环
    FROM starts s
    JOIN txn_norm e
      ON e.from_bank = s.s_bank AND e.from_account = s.s_acct
    LEFT JOIN account_degrees dg
      ON dg.bank_id = e.to_bank AND dg.account_number = e.to_account
    WHERE COALESCE(dg.total_cp_count, 0) <= 200                  -- 约束 4
    ORDER BY s.pattern_id, e.to_bank, e.to_account, e.txn_ts, e.txn_id)

    UNION ALL

    -- ---------- 递归成员：从当前节点再走一跳 ----------
    SELECT
        w.pattern_id,
        w.s_bank,
        w.s_acct,
        nx.to_bank,
        nx.to_account,
        w.hop + 1,
        w.start_ts,
        nx.txn_ts,
        w.path     || (nx.to_bank::text || ':' || nx.to_account),
        w.txn_path || nx.txn_id,
        (nx.to_bank = w.s_bank AND nx.to_account = w.s_acct)
    FROM walk w
    -- LATERAL：右边的子查询可以引用左边当前行（w）的列。
    -- 相当于"对 walk 里的每一行，各跑一次这个子查询"。
    -- 递归成员里不允许直接用 DISTINCT ON / ORDER BY / LIMIT，
    -- 但放进 LATERAL 子查询就可以——因为子查询本身没有引用 walk 这个名字。
    CROSS JOIN LATERAL (
        SELECT DISTINCT ON (e.to_bank, e.to_account)            -- 约束 6：每个下一跳只留一笔
               e.to_bank, e.to_account, e.txn_ts, e.txn_id
        FROM txn_norm e
        LEFT JOIN account_degrees dg
          ON dg.bank_id = e.to_bank AND dg.account_number = e.to_account
        WHERE e.from_bank    = w.cur_bank
          AND e.from_account = w.cur_acct
          AND e.txn_ts >= w.cur_ts                               -- 约束 2
          AND e.txn_ts <= w.start_ts + INTERVAL '7 days'         -- 约束 3
          AND COALESCE(dg.total_cp_count, 0) <= 200              -- 约束 4
        ORDER BY e.to_bank, e.to_account, e.txn_ts, e.txn_id     -- 取最早那一笔
    ) nx
    WHERE w.hop < 12                                             -- 约束 5
      AND NOT w.closed                                           -- 已闭环的路径不再扩展
      AND (
            -- 约束 1：回到起点就是我们要找的闭环，放行
            (nx.to_bank = w.s_bank AND nx.to_account = w.s_acct)
            -- 其他节点不得重复访问（否则会在环里无限绕圈）
            OR NOT ((nx.to_bank::text || ':' || nx.to_account) = ANY (w.path))
          )
)
SELECT
    pattern_id,
    s_bank,
    s_acct,
    hop                                                          AS found_hops,
    path,
    txn_path,
    start_ts,
    cur_ts                                                       AS end_ts,
    ROUND(EXTRACT(EPOCH FROM (cur_ts - start_ts)) / 3600.0, 1)   AS span_hours
FROM walk
WHERE closed;

CREATE INDEX ON cycle_trace_result (pattern_id);
ANALYZE cycle_trace_result;


-- ============================================================================
-- 第 2 步：召回率 —— 54 个 CYCLE 追出了几个
-- ============================================================================
-- 按"标注时间是否单调"分组统计：
-- 诊断已发现 5 个 CYCLE 的标注存在跨天时间倒流（生成器只建模了拓扑、
-- 没建模时间因果），这些在"时间非递减"约束下本来就追不出来。
-- 不分组的话，会把数据集缺陷算成算法失败。
-- ============================================================================
\echo ''
\echo '============================================================'
\echo '1. 闭环追踪召回率'
\echo '============================================================'

WITH ordered AS (
    SELECT pattern_id, txn_ts,
           LAG(txn_ts) OVER (PARTITION BY pattern_id ORDER BY seq_no) AS prev_ts
    FROM pattern_transactions
),
monotonic AS (
    SELECT o.pattern_id,
           NOT COALESCE(BOOL_OR(o.txn_ts < o.prev_ts), FALSE) AS is_monotonic
    FROM ordered o
    JOIN laundering_patterns p ON p.pattern_id = o.pattern_id
    WHERE p.pattern_type = 'CYCLE'
    GROUP BY o.pattern_id
),
found AS (
    SELECT DISTINCT pattern_id FROM cycle_trace_result
)
SELECT
    CASE WHEN m.is_monotonic THEN 'A. 时间单调（理论可追）'
         ELSE                     'B. 含时间倒流（数据缺陷）' END   AS "模式分组",
    COUNT(*)                                                         AS "模式数",
    COUNT(*) FILTER (WHERE f.pattern_id IS NOT NULL)                 AS "追出",
    ROUND(100.0 * COUNT(*) FILTER (WHERE f.pattern_id IS NOT NULL)
          / COUNT(*), 1)                                             AS "召回率%"
FROM monotonic m
LEFT JOIN found f ON f.pattern_id = m.pattern_id
GROUP BY 1
ORDER BY 1;

\echo ''
SELECT
    COUNT(DISTINCT pattern_id)            AS "追出的模式数",
    COUNT(*)                              AS "闭环路径总数",
    ROUND(COUNT(*)::numeric / NULLIF(COUNT(DISTINCT pattern_id), 0), 1)
                                          AS "每模式平均闭环数",
    ROUND(AVG(found_hops), 2)             AS "平均跳数",
    MIN(found_hops)                       AS "最短",
    MAX(found_hops)                       AS "最长",
    ROUND(AVG(span_hours), 1)             AS "平均跨度h"
FROM cycle_trace_result;
-- "每模式平均闭环数" 如果远大于 1，说明从同一个起点能绕出很多不同的环，
-- 其中大部分可能是巧合形成的——这就是第 4 步要检查的"假阳性闭环"。


-- ============================================================================
-- 第 3 步：追出的环是不是【那个】环 —— 节点集合对比
-- ============================================================================
-- 只比跳数不够：跳数相同的环可能经过完全不同的账户。
-- 这里把标注环的节点集合和追出环的节点集合直接比较。
-- ============================================================================
\echo ''
\echo '============================================================'
\echo '2. 追出的环与标注环是否一致（节点集合对比）'
\echo '============================================================'

WITH labeled_nodes AS (
    -- 标注环的节点 = 模式内所有交易的付款方（CYCLE 首尾相接，付款方集合即全部节点）
    SELECT d.pattern_id,
           ARRAY_AGG(DISTINCT d.from_bank::text || ':' || d.from_account
                     ORDER BY d.from_bank::text || ':' || d.from_account) AS nodes
    FROM pattern_transactions d
    JOIN laundering_patterns p ON p.pattern_id = d.pattern_id
    WHERE p.pattern_type = 'CYCLE'
    GROUP BY d.pattern_id
),
found_nodes AS (
    -- 追出的环：path 首尾都是起点，去重后排序，便于和标注集合直接比较
    SELECT r.pattern_id, r.found_hops,
           ARRAY(SELECT DISTINCT x FROM unnest(r.path) AS x ORDER BY x) AS nodes
    FROM cycle_trace_result r
),
best AS (
    -- 每个模式里，只要有一条追出的环与标注完全一致，就算"精确追出"
    SELECT l.pattern_id,
           BOOL_OR(f.nodes = l.nodes)      AS exact_match,
           -- @> 是数组"包含"运算符：左边是否包含右边的所有元素
           BOOL_OR(f.nodes @> l.nodes)     AS superset_match
    FROM labeled_nodes l
    JOIN found_nodes f ON f.pattern_id = l.pattern_id
    GROUP BY l.pattern_id
)
SELECT
    COUNT(*)                                       AS "追出的模式数",
    COUNT(*) FILTER (WHERE exact_match)            AS "节点完全一致",
    COUNT(*) FILTER (WHERE NOT exact_match AND superset_match)
                                                   AS "包含标注环(环更大)",
    COUNT(*) FILTER (WHERE NOT superset_match)     AS "只找到别的环"
FROM best;


\echo ''
\echo '=== 样例：3 个追出的环（每个模式取最短的一条）==='

WITH shortest AS (
    -- DISTINCT ON (col) 是 Postgres 特有语法：每个 col 值只保留一行，
    -- 保留哪行由 ORDER BY 决定（这里取跳数最少的那条）。
    -- 标准 SQL 要用 ROW_NUMBER() 窗口函数绕一圈才能做到。
    SELECT DISTINCT ON (pattern_id)
           pattern_id, found_hops, span_hours, path
    FROM cycle_trace_result
    ORDER BY pattern_id, found_hops, span_hours
)
SELECT
    s.pattern_id                          AS "模式ID",
    p.txn_count                           AS "标注笔数",
    s.found_hops                          AS "追出跳数",
    s.span_hours                          AS "跨度h",
    array_to_string(s.path, ' -> ')       AS "路径"
FROM shortest s
JOIN laundering_patterns p ON p.pattern_id = s.pattern_id
ORDER BY s.found_hops
LIMIT 3;


-- ============================================================================
-- 第 4 步：精确率 —— 追出的环里，有多少是巧合形成的
-- ============================================================================
-- 召回率回答"能不能找到"，精确率回答"找到的是不是真的"。
-- 一个起点可能绕出很多环；如果大量环上的交易 is_laundering = 0，
-- 说明遍历在正常资金网络里也能绕出闭环——那么 Agent 就不能
-- "看到闭环就判洗钱"，必须结合其他证据（金额、时间、度数）。
-- ============================================================================
\echo ''
\echo '============================================================'
\echo '3. 追出的环上，洗钱交易占比'
\echo '============================================================'

WITH ring_stats AS (
    -- 按 txn_path 分组，而不是按 path（节点序列）：
    -- 同一串账户之间可能有多笔平行交易，节点相同但交易不同，是不同的环。
    SELECT r.pattern_id, r.txn_path,
           COUNT(*)                            AS n_txn,
           SUM(n.is_laundering)                AS n_laundering
    FROM cycle_trace_result r
    CROSS JOIN LATERAL unnest(r.txn_path) AS t(txn_id)
    -- unnest 把数组展开成多行，是数组操作里最常用的函数
    JOIN txn_norm n ON n.txn_id = t.txn_id
    GROUP BY r.pattern_id, r.txn_path
)
SELECT
    CASE
        WHEN n_laundering = n_txn THEN 'A. 全部是洗钱交易'
        WHEN n_laundering > 0     THEN 'B. 部分是洗钱交易'
        ELSE                           'C. 一笔洗钱都没有（巧合闭环）'
    END                                   AS "环的构成",
    COUNT(*)                              AS "环数",
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS "占比%"
FROM ring_stats
GROUP BY 1
ORDER BY 1;


-- ============================================================================
-- 第 5 步：单次遍历成本 —— Agent 处理一条告警时跑得起吗
-- ============================================================================
-- 用前面小例子里那个 7 跳 CYCLE 的起点 (11974, 8011D4F20)。
-- ============================================================================
\echo ''
\echo '============================================================'
\echo '4. 单个起点的遍历成本'
\echo '============================================================'

EXPLAIN (ANALYZE, BUFFERS, COSTS OFF, SUMMARY ON)
WITH RECURSIVE walk AS (
    SELECT 11974 AS s_bank, '8011D4F20'::varchar AS s_acct,
           e.to_bank AS cur_bank, e.to_account AS cur_acct,
           1 AS hop, e.txn_ts AS start_ts, e.txn_ts AS cur_ts,
           ARRAY['11974:8011D4F20', e.to_bank::text || ':' || e.to_account] AS path,
           FALSE AS closed
    FROM txn_norm e
    WHERE e.from_bank = 11974 AND e.from_account = '8011D4F20'

    UNION ALL

    SELECT w.s_bank, w.s_acct, nx.to_bank, nx.to_account, w.hop + 1,
           w.start_ts, nx.txn_ts,
           w.path || (nx.to_bank::text || ':' || nx.to_account),
           (nx.to_bank = w.s_bank AND nx.to_account = w.s_acct)
    FROM walk w
    CROSS JOIN LATERAL (
        SELECT DISTINCT ON (e.to_bank, e.to_account)
               e.to_bank, e.to_account, e.txn_ts
        FROM txn_norm e
        LEFT JOIN account_degrees dg
          ON dg.bank_id = e.to_bank AND dg.account_number = e.to_account
        WHERE e.from_bank = w.cur_bank AND e.from_account = w.cur_acct
          AND e.txn_ts >= w.cur_ts
          AND e.txn_ts <= w.start_ts + INTERVAL '7 days'
          AND COALESCE(dg.total_cp_count, 0) <= 200
        ORDER BY e.to_bank, e.to_account, e.txn_ts
    ) nx
    WHERE w.hop < 12
      AND NOT w.closed
      AND ((nx.to_bank = w.s_bank AND nx.to_account = w.s_acct)
           OR NOT ((nx.to_bank::text || ':' || nx.to_account) = ANY (w.path)))
)
SELECT COUNT(*) AS walks, COUNT(*) FILTER (WHERE closed) AS cycles FROM walk;

RESET statement_timeout;
\timing off