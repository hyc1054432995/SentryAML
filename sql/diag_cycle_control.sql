-- ============================================================================
-- SentryAML - 诊断：闭环追踪的失败原因 + 对照组实验
-- ============================================================================
-- 执行（依赖 009 生成的 cycle_trace_result）:
--   docker compose exec -T postgres psql -U sentry -d sentryaml -f /sql/diag_cycle_control.sql
--
-- 两件事：
--   第一部分：查清 009 里没对上的 5 个模式
--             （4 个时间正常却没追出 + 1 个追出了别的环）
--   第二部分：对照组实验 —— 普通账户会不会也经常绕出闭环
--
-- 本文件只用临时表（TEMP TABLE），psql 退出后自动消失，
-- 不改变数据库状态，符合 diag_ 前缀"只读诊断"的约定。
-- ============================================================================

\timing on
\pset pager off
SET statement_timeout = '1200s';


-- ############################################################################
-- 第一部分：没对上的 5 个模式，各自卡在哪
-- ############################################################################

\echo ''
\echo '============================================================'
\echo '1-A. 时间正常却没追出的 CYCLE：逐个排查原因'
\echo '============================================================'
\echo '（每一列对应一个可能的原因，最后一列给出第一个命中的原因）'
\echo ''

WITH ordered AS (
    SELECT pattern_id, txn_ts,
           LAG(txn_ts) OVER (PARTITION BY pattern_id ORDER BY seq_no) AS prev_ts
    FROM pattern_transactions
),
monotonic AS (
    SELECT o.pattern_id
    FROM ordered o
    JOIN laundering_patterns p ON p.pattern_id = o.pattern_id
    WHERE p.pattern_type = 'CYCLE'
    GROUP BY o.pattern_id
    HAVING NOT COALESCE(BOOL_OR(o.txn_ts < o.prev_ts), FALSE)
),
missed AS (
    -- 时间正常 且 009 一条环都没追出
    SELECT m.pattern_id
    FROM monotonic m
    WHERE NOT EXISTS (
        SELECT 1 FROM cycle_trace_result r WHERE r.pattern_id = m.pattern_id
    )
    -- NOT EXISTS 比 NOT IN 更安全：子查询结果里只要有一个 NULL，
    -- NOT IN 就会整体返回"未知"，导致一行都查不出来。这是 SQL 的经典坑。
),
per_pattern AS (
    SELECT
        d.pattern_id,
        COUNT(*)                                                     AS n_txn,
        COUNT(DISTINCT d.from_bank::text || ':' || d.from_account)   AS n_nodes,
        ROUND(EXTRACT(EPOCH FROM (MAX(d.txn_ts) - MIN(d.txn_ts))) / 3600.0, 1)
                                                                     AS span_h,
        -- 自己转给自己：txn_norm 已剔除自环，这一步在图里根本不存在
        COUNT(*) FILTER (WHERE d.from_bank = d.to_bank
                           AND d.from_account = d.to_account)        AS n_selfloop,
        -- 模式里有没有交易不在 txn_norm 里（自环以外的原因被过滤掉）
        COUNT(*) FILTER (WHERE n.txn_id IS NULL)                     AS n_missing_edge,
        -- 路径上经过的账户里，对手方数最多的那个
        MAX(GREATEST(COALESCE(df.total_cp_count, 0),
                     COALESCE(dt.total_cp_count, 0)))                AS max_cp
    FROM missed ms
    JOIN pattern_transactions d ON d.pattern_id = ms.pattern_id
    LEFT JOIN txn_norm n         ON n.txn_id = d.txn_id
    LEFT JOIN account_degrees df ON df.bank_id = d.from_bank AND df.account_number = d.from_account
    LEFT JOIN account_degrees dt ON dt.bank_id = d.to_bank   AND dt.account_number = d.to_account
    GROUP BY d.pattern_id
)
SELECT
    pattern_id                       AS "模式ID",
    n_txn                            AS "笔数",
    n_nodes                          AS "不同账户",
    span_h                           AS "跨度h",
    n_selfloop                       AS "自环步",
    n_missing_edge                   AS "缺失边",
    max_cp                           AS "最大对手方数",
    CASE
        WHEN n_selfloop > 0      THEN '含自环（txn_norm 已剔除）'
        WHEN n_missing_edge > 0  THEN '有交易不在 txn_norm'
        WHEN max_cp > 200        THEN '经过清算账户（被剪枝）'
        WHEN n_txn > 12          THEN '超过 12 跳'
        WHEN span_h > 168        THEN '超过 7 天窗口'
        WHEN n_nodes < n_txn     THEN '中途重复经过同一账户（8 字形）'
        ELSE                          '未知 —— 需要逐笔看'
    END                              AS "判定"
FROM per_pattern
ORDER BY pattern_id;


\echo ''
\echo '============================================================'
\echo '1-B. 追出了环、但节点和标注对不上的模式'
\echo '============================================================'

WITH labeled AS (
    SELECT d.pattern_id,
           ARRAY_AGG(DISTINCT d.from_bank::text || ':' || d.from_account
                     ORDER BY d.from_bank::text || ':' || d.from_account) AS nodes
    FROM pattern_transactions d
    JOIN laundering_patterns p ON p.pattern_id = d.pattern_id
    WHERE p.pattern_type = 'CYCLE'
    GROUP BY d.pattern_id
),
found AS (
    SELECT r.pattern_id, r.found_hops, r.path,
           ARRAY(SELECT DISTINCT x FROM unnest(r.path) AS x ORDER BY x) AS nodes
    FROM cycle_trace_result r
),
mismatch AS (
    SELECT l.pattern_id
    FROM labeled l
    JOIN found f ON f.pattern_id = l.pattern_id
    GROUP BY l.pattern_id
    HAVING NOT BOOL_OR(f.nodes = l.nodes)
)
SELECT
    m.pattern_id                                         AS "模式ID",
    p.txn_count                                          AS "标注笔数",
    array_to_string(l.nodes, ', ')                       AS "标注节点",
    (SELECT array_to_string(f.path, ' -> ')
     FROM found f WHERE f.pattern_id = m.pattern_id
     ORDER BY f.found_hops DESC LIMIT 1)                 AS "追出的最长环"
FROM mismatch m
JOIN labeled l             ON l.pattern_id = m.pattern_id
JOIN laundering_patterns p ON p.pattern_id = m.pattern_id;


-- ############################################################################
-- 第二部分：对照组实验
-- ############################################################################
-- 问题：009 的起点全是已知洗钱账户，追出的环 100% 带洗钱交易，
--       但这存在【选择偏差】——没回答"普通账户会不会也经常绕出闭环"。
--
-- 设计：三组起点，用【完全相同】的遍历规则跑，比较"能形成闭环的比例"。
--   T1  CYCLE 起点账户         —— 009 的实验组
--   T2  其他涉洗钱账户（500 个）—— 看闭环信号对 CYCLE 以外的洗钱有没有用
--   C   正常账户（2000 个）     —— 对照组，按 T1 的度数分布【分层抽样】
--
-- 为什么要按度数分层抽样：
--   正常账户里一半以上只有 1 个对手方，天生绕不出环。
--   随便抽的话，对照组闭环率会被这些账户压得很低，
--   让"洗钱账户更容易成环"这个结论被夸大——比较就不公平了。
--   让对照组的度数分布和 T1 一致，才是在"条件相同"下比较。
--
-- 为什么用 md5 排序而不是 random() 抽样：
--   md5(账户号) 是确定性的"伪随机"，每次跑抽到的都是同一批账户，
--   结果可以复现。random() 每次都不同，而 setseed 在并行查询下也不保证可复现。
-- ############################################################################

\echo ''
\echo '============================================================'
\echo '2-A. 构建三组起点'
\echo '============================================================'

DROP TABLE IF EXISTS tmp_starts;
CREATE TEMP TABLE tmp_starts (
    grp     TEXT,
    s_bank  INTEGER,
    s_acct  VARCHAR(16),
    deg_bin TEXT
);

-- 度数分档，和 diag_hub_laundering 里用的一致
CREATE OR REPLACE FUNCTION pg_temp.deg_bin(cp BIGINT) RETURNS TEXT AS $$
    SELECT CASE
        WHEN cp = 1    THEN 'A. 1'
        WHEN cp <= 3   THEN 'B. 2-3'
        WHEN cp <= 10  THEN 'C. 4-10'
        WHEN cp <= 38  THEN 'D. 11-38'
        ELSE                'E. 39-200'
    END
$$ LANGUAGE sql IMMUTABLE;
-- pg_temp. 前缀 = 临时函数，会话结束自动删除，和临时表一样不留痕迹。

-- T1：CYCLE 起点账户（去重）
INSERT INTO tmp_starts
SELECT DISTINCT 'T1 CYCLE起点', d.from_bank, d.from_account,
       pg_temp.deg_bin(dg.total_cp_count)
FROM laundering_patterns p
JOIN pattern_transactions d ON d.pattern_id = p.pattern_id AND d.seq_no = 1
JOIN account_degrees dg     ON dg.bank_id = d.from_bank AND dg.account_number = d.from_account
WHERE p.pattern_type = 'CYCLE';

-- T2：其他涉洗钱账户，排除已在 T1 的
INSERT INTO tmp_starts
SELECT 'T2 其他涉洗钱', bank_id, account_number, pg_temp.deg_bin(total_cp_count)
FROM account_degrees dg
WHERE touched_laundering = 1
  AND out_txn_count > 0                      -- 起点必须有转出，否则没法出发
  AND total_cp_count <= 200                  -- 与遍历的剪枝规则一致
  AND NOT EXISTS (SELECT 1 FROM tmp_starts t
                  WHERE t.s_bank = dg.bank_id AND t.s_acct = dg.account_number)
ORDER BY md5(bank_id::text || ':' || account_number)
LIMIT 500;

-- C：正常账户，按 T1 的度数分布分层抽 2000 个
WITH t1_share AS (
    SELECT deg_bin, COUNT(*)::numeric / SUM(COUNT(*)) OVER () AS share
    FROM tmp_starts WHERE grp = 'T1 CYCLE起点'
    GROUP BY deg_bin
),
candidates AS (
    SELECT bank_id, account_number,
           pg_temp.deg_bin(total_cp_count) AS deg_bin,
           ROW_NUMBER() OVER (
               PARTITION BY pg_temp.deg_bin(total_cp_count)
               ORDER BY md5(bank_id::text || ':' || account_number)
           ) AS rn
    FROM account_degrees
    WHERE touched_laundering = 0
      AND out_txn_count > 0
      AND total_cp_count <= 200
)
INSERT INTO tmp_starts
SELECT 'C 正常账户', c.bank_id, c.account_number, c.deg_bin
FROM candidates c
JOIN t1_share s ON s.deg_bin = c.deg_bin
WHERE c.rn <= CEIL(2000 * s.share);

\echo ''
\echo '--- 各组度数分布（C 应与 T1 接近）---'
SELECT grp                                             AS "组",
       deg_bin                                         AS "度数档",
       COUNT(*)                                        AS "起点数",
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY grp), 1) AS "组内占比%"
FROM tmp_starts
GROUP BY grp, deg_bin
ORDER BY grp, deg_bin;


\echo ''
\echo '============================================================'
\echo '2-B. 用和 009 完全相同的规则遍历（约 3000 个起点，可能需要几分钟）'
\echo '============================================================'

DROP TABLE IF EXISTS tmp_rings;
CREATE TEMP TABLE tmp_rings AS
WITH RECURSIVE walk AS (
    (SELECT DISTINCT ON (s.grp, s.s_bank, s.s_acct, e.to_bank, e.to_account, e.txn_ts)
        s.grp, s.s_bank, s.s_acct,
        e.to_bank AS cur_bank, e.to_account AS cur_acct,
        1 AS hop, e.txn_ts AS start_ts, e.txn_ts AS cur_ts,
        ARRAY[s.s_bank::text || ':' || s.s_acct,
              e.to_bank::text || ':' || e.to_account] AS path,
        ARRAY[e.txn_id] AS txn_path,
        FALSE AS closed
    FROM tmp_starts s
    JOIN txn_norm e
      ON e.from_bank = s.s_bank AND e.from_account = s.s_acct
    LEFT JOIN account_degrees dg
      ON dg.bank_id = e.to_bank AND dg.account_number = e.to_account
    WHERE COALESCE(dg.total_cp_count, 0) <= 200
    ORDER BY s.grp, s.s_bank, s.s_acct, e.to_bank, e.to_account, e.txn_ts, e.txn_id)

    UNION ALL

    SELECT w.grp, w.s_bank, w.s_acct,
           nx.to_bank, nx.to_account, w.hop + 1, w.start_ts, nx.txn_ts,
           w.path     || (nx.to_bank::text || ':' || nx.to_account),
           w.txn_path || nx.txn_id,
           (nx.to_bank = w.s_bank AND nx.to_account = w.s_acct)
    FROM walk w
    CROSS JOIN LATERAL (
        SELECT DISTINCT ON (e.to_bank, e.to_account)
               e.to_bank, e.to_account, e.txn_ts, e.txn_id
        FROM txn_norm e
        LEFT JOIN account_degrees dg
          ON dg.bank_id = e.to_bank AND dg.account_number = e.to_account
        WHERE e.from_bank = w.cur_bank AND e.from_account = w.cur_acct
          AND e.txn_ts >= w.cur_ts
          AND e.txn_ts <= w.start_ts + INTERVAL '7 days'
          AND COALESCE(dg.total_cp_count, 0) <= 200
        ORDER BY e.to_bank, e.to_account, e.txn_ts, e.txn_id
    ) nx
    WHERE w.hop < 12
      AND NOT w.closed
      AND ((nx.to_bank = w.s_bank AND nx.to_account = w.s_acct)
           OR NOT ((nx.to_bank::text || ':' || nx.to_account) = ANY (w.path)))
)
SELECT grp, s_bank, s_acct, hop AS found_hops, txn_path
FROM walk
WHERE closed;

-- 每个起点汇总成一行：有没有环、最短的环几跳
DROP TABLE IF EXISTS tmp_per_start;
CREATE TEMP TABLE tmp_per_start AS
SELECT s.grp, s.s_bank, s.s_acct, s.deg_bin,
       COUNT(r.found_hops)                              AS n_rings,
       MIN(r.found_hops)                                AS min_hops,
       COUNT(r.found_hops) FILTER (WHERE r.found_hops >= 3) AS n_rings_3plus
FROM tmp_starts s
LEFT JOIN tmp_rings r
  ON r.grp = s.grp AND r.s_bank = s.s_bank AND r.s_acct = s.s_acct
GROUP BY s.grp, s.s_bank, s.s_acct, s.deg_bin;


\echo ''
\echo '============================================================'
\echo '2-C. 结果：各组能形成闭环的比例'
\echo '============================================================'
\echo '（"3跳及以上"单独列出：两跳的 A->B->A 在正常商业里很常见，'
\echo '  比如退款、押金退还，不能直接当可疑信号）'
\echo ''

SELECT
    grp                                                        AS "组",
    COUNT(*)                                                   AS "起点数",
    COUNT(*) FILTER (WHERE n_rings > 0)                        AS "有闭环",
    ROUND(100.0 * COUNT(*) FILTER (WHERE n_rings > 0) / COUNT(*), 2)
                                                               AS "闭环率%",
    -- 只有两跳回流、没有更长的环
    ROUND(100.0 * COUNT(*) FILTER (WHERE n_rings > 0 AND n_rings_3plus = 0) / COUNT(*), 2)
                                                               AS "仅两跳回流%",
    -- 至少有一个 >= 3 跳的环，才算真正的多跳闭环
    ROUND(100.0 * COUNT(*) FILTER (WHERE n_rings_3plus > 0) / COUNT(*), 2)
                                                               AS "有3跳以上环%"
FROM tmp_per_start p
GROUP BY grp
ORDER BY grp;


\echo ''
\echo '--- 分度数档比较（排除度数差异的影响，这是最公平的比较）---'

SELECT
    deg_bin                                                    AS "度数档",
    grp                                                        AS "组",
    COUNT(*)                                                   AS "起点数",
    ROUND(100.0 * COUNT(*) FILTER (WHERE n_rings > 0) / COUNT(*), 1)
                                                               AS "闭环率%",
    ROUND(100.0 * COUNT(*) FILTER (WHERE n_rings_3plus > 0) / COUNT(*), 1)
                                                               AS "3跳以上%"
FROM tmp_per_start
GROUP BY deg_bin, grp
ORDER BY deg_bin, grp;


\echo ''
\echo '--- 似然比：有闭环时，是洗钱账户的证据有多强 ---'
\echo '（似然比 = P(有闭环|洗钱) / P(有闭环|正常)。'
\echo '  1 表示没区分度；越大，"看到闭环"越能提高洗钱的可能性）'

WITH r AS (
    SELECT grp,
           AVG((n_rings > 0)::int)::numeric        AS p_ring,
           AVG((n_rings_3plus > 0)::int)::numeric  AS p_ring3
    FROM tmp_per_start
    GROUP BY grp
)
SELECT
    t.grp                                                     AS "实验组",
    ROUND(100 * t.p_ring, 2)                                  AS "实验组闭环率%",
    ROUND(100 * c.p_ring, 2)                                  AS "对照组闭环率%",
    ROUND(t.p_ring / NULLIF(c.p_ring, 0), 1)                  AS "似然比(任意环)",
    ROUND(t.p_ring3 / NULLIF(c.p_ring3, 0), 1)                AS "似然比(3跳以上)"
FROM r t
CROSS JOIN r c
WHERE t.grp LIKE 'T%' AND c.grp = 'C 正常账户'
ORDER BY t.grp;

RESET statement_timeout;
\timing off