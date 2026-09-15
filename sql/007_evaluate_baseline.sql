-- ============================================================================
-- SentryAML - 007: 规则引擎评测
-- ============================================================================
-- 执行:
--   docker compose exec -T postgres psql -U sentry -d sentryaml -f /sql/007_evaluate_baseline.sql
--
-- 四个部分：
--   A. 逐规则指标（精确率/召回率/F1/误报率）
--   B. 合并指标 + 边际贡献（哪些规则是多余的）
--   C. 按洗钱模式类型看召回率  <- 最有说服力的一张表
--   D. 阈值敏感性扫描          <- 论证"这是范式缺陷，不是调参问题"
--
-- 【指标口径】
--   本文件的"误报率" = FP / (TP + FP) = 1 - 精确率，
--   即"告警中有多少是假的"。这是反洗钱行业口径，
--   也是"TMS 误报率超过 95%"这类说法的定义。
--   注意它不同于统计学的 FPR = FP / (FP + TN)：
--   后者在 1:980 的极端不平衡下永远接近 0，毫无信息量。
--
--   分母基准：txn_norm 共 4,487,133 笔，其中洗钱 5,166 笔
--   （全表 5,177 笔，有 11 笔是自环，已在 txn_norm 中排除）。
-- ============================================================================

\timing on
\pset pager off


-- ============================================================================
-- 预备：把昂贵的聚合【只算一次】，存成中间表
-- ============================================================================
-- 阈值扫描要对每条规则试 6-8 个阈值。如果每个阈值都重跑一遍窗口函数，
-- 就是 40 多次全表扫描，几分钟起步。
--
-- 正确做法：把"每笔交易的 24 小时计数""每个账户-日的对手方数"这类
-- 昂贵的中间结果算一次存下来，之后换阈值只是在这张小表上加个 WHERE。
-- 这个"算一次、扫多次"的思路在做任何参数扫描时都适用。
-- ============================================================================

DROP TABLE IF EXISTS eval_velocity CASCADE;
DROP TABLE IF EXISTS eval_fanout   CASCADE;
DROP TABLE IF EXISTS eval_fanin    CASCADE;
DROP TABLE IF EXISTS eval_passthru CASCADE;

-- 逐笔：24 小时内的转出笔数 + 落在可疑金额带内的笔数
-- 两个窗口函数共享同一个 PARTITION/ORDER，Postgres 只排序一次
CREATE TABLE eval_velocity AS
SELECT
    txn_id,
    is_laundering,
    amount_usd,
    COUNT(*) OVER w                                                     AS n_txn_24h,
    SUM(CASE WHEN amount_usd >= 3000 AND amount_usd < 10000
             THEN 1 ELSE 0 END) OVER w                                  AS n_in_band
FROM txn_norm
WINDOW w AS (
    PARTITION BY from_bank, from_account
    ORDER BY txn_ts
    RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
);
-- WINDOW w AS (...) 是【具名窗口】：定义一次，多个函数共用，
-- 避免把同样的 OVER 子句抄两遍，也让优化器确定只需排序一次。

CREATE INDEX ON eval_velocity (n_txn_24h);
CREATE INDEX ON eval_velocity (n_in_band);

-- 账户-日：扇出对手方数，以及该账户-日的交易数与洗钱交易数
CREATE TABLE eval_fanout AS
SELECT
    from_bank, from_account, txn_date,
    COUNT(DISTINCT ROW(to_bank, to_account))  AS n_counterparties,
    COUNT(*)                                  AS n_txns,
    SUM(is_laundering)                        AS n_laundering
FROM txn_norm
GROUP BY 1, 2, 3;

CREATE INDEX ON eval_fanout (n_counterparties);

-- 账户-日：扇入
CREATE TABLE eval_fanin AS
SELECT
    to_bank, to_account, txn_date,
    COUNT(DISTINCT ROW(from_bank, from_account)) AS n_senders,
    COUNT(*)                                     AS n_txns,
    SUM(is_laundering)                           AS n_laundering
FROM txn_norm
GROUP BY 1, 2, 3;

CREATE INDEX ON eval_fanin (n_senders);

ANALYZE eval_velocity;
ANALYZE eval_fanout;
ANALYZE eval_fanin;


-- ============================================================================
-- A. 逐规则指标
-- ============================================================================
\echo ''
\echo '============================================================'
\echo 'A. 逐规则指标'
\echo '============================================================'

WITH base AS (
    SELECT COUNT(*) AS n_total,
           SUM(is_laundering) AS n_laundering
    FROM txn_norm
),
per_rule AS (
    SELECT
        a.rule_code,
        COUNT(*)                                          AS n_alerts,
        COUNT(*) FILTER (WHERE n.is_laundering = 1)       AS tp,
        COUNT(*) FILTER (WHERE n.is_laundering = 0)       AS fp
    FROM rule_alerts a
    JOIN txn_norm n ON n.txn_id = a.txn_id
    GROUP BY a.rule_code
)
SELECT
    r.rule_code                                              AS "规则",
    d.rule_name                                              AS "名称",
    d.rule_tier                                              AS "层",
    r.n_alerts                                               AS "告警数",
    r.tp                                                     AS "命中",
    ROUND(100.0 * r.tp / r.n_alerts, 3)                      AS "精确率%",
    ROUND(100.0 * r.tp / b.n_laundering, 2)                  AS "召回率%",
    ROUND(100.0 * r.fp / r.n_alerts, 3)                      AS "误报率%",
    -- F1 = 2PR/(P+R)，精确率和召回率的调和平均。
    -- 用调和平均而非算术平均，是因为它对"一高一低"惩罚更重：
    -- P=100% R=1% 的算术平均是 50.5%，调和平均只有 1.98%。
    ROUND(200.0 * r.tp / NULLIF(r.n_alerts + b.n_laundering, 0), 3) AS "F1%",
    -- 提升度 = 精确率 / 基础发生率。=1 表示和随机猜没区别。
    ROUND((1.0 * r.tp / r.n_alerts) / (1.0 * b.n_laundering / b.n_total), 2) AS "提升度"
FROM per_rule r
JOIN rule_definitions d ON d.rule_code = r.rule_code
CROSS JOIN base b
ORDER BY r.rule_code;

\echo ''
\echo '（基础发生率 = 5,166 / 4,487,133 = 0.1151%。提升度 1.0 等于随机猜测）'


-- ============================================================================
-- B. 合并指标与边际贡献
-- ============================================================================
\echo ''
\echo '============================================================'
\echo 'B. 七条规则合并后的整体表现'
\echo '============================================================'

WITH base AS (
    SELECT COUNT(*) AS n_total, SUM(is_laundering) AS n_laundering FROM txn_norm
),
union_alerts AS (
    SELECT DISTINCT txn_id FROM rule_alerts
),
agg AS (
    SELECT COUNT(*)                                     AS n_alerts,
           COUNT(*) FILTER (WHERE n.is_laundering = 1)  AS tp
    FROM union_alerts u
    JOIN txn_norm n ON n.txn_id = u.txn_id
)
SELECT
    a.n_alerts                                          AS "告警总数",
    ROUND(100.0 * a.n_alerts / b.n_total, 2)            AS "占全部交易%",
    a.tp                                                AS "命中洗钱",
    b.n_laundering                                      AS "洗钱总数",
    ROUND(100.0 * a.tp / a.n_alerts, 4)                 AS "精确率%",
    ROUND(100.0 * a.tp / b.n_laundering, 2)             AS "召回率%",
    ROUND(100.0 * (a.n_alerts - a.tp) / a.n_alerts, 3)  AS "误报率%",
    -- 每抓到 1 笔真洗钱，分析师要白看多少笔
    ROUND(1.0 * (a.n_alerts - a.tp) / NULLIF(a.tp, 0), 0) AS "每命中1笔需审查"
FROM agg a CROSS JOIN base b;


\echo ''
\echo '--- 边际贡献：去掉某条规则会损失多少独有命中 ---'
\echo '（独有命中 = 只有这条规则抓到、其他规则都没抓到的洗钱交易）'

WITH laundering_alerts AS (
    SELECT a.rule_code, a.txn_id
    FROM rule_alerts a
    JOIN txn_norm n ON n.txn_id = a.txn_id
    WHERE n.is_laundering = 1
),
hit_counts AS (
    SELECT txn_id, COUNT(DISTINCT rule_code) AS n_rules
    FROM laundering_alerts
    GROUP BY txn_id
)
SELECT
    la.rule_code                                            AS "规则",
    COUNT(DISTINCT la.txn_id)                               AS "命中洗钱",
    COUNT(DISTINCT la.txn_id) FILTER (WHERE hc.n_rules = 1) AS "独有命中",
    (SELECT COUNT(*) FROM rule_alerts x WHERE x.rule_code = la.rule_code) AS "告警数",
    -- 独有命中 / 告警数：这条规则"不可替代的价值密度"
    ROUND(1000000.0 * COUNT(DISTINCT la.txn_id) FILTER (WHERE hc.n_rules = 1)
          / (SELECT COUNT(*) FROM rule_alerts x WHERE x.rule_code = la.rule_code), 2)
                                                            AS "独有命中/百万告警"
FROM laundering_alerts la
JOIN hit_counts hc ON hc.txn_id = la.txn_id
GROUP BY la.rule_code
ORDER BY "独有命中" DESC;


-- ============================================================================
-- C. 按洗钱模式类型看召回率  <- 核心论证
-- ============================================================================
\echo ''
\echo '============================================================'
\echo 'C. 按洗钱模式类型的召回率'
\echo '============================================================'
\echo '（规则只能抓它被设计来抓的形态。R6/R7 直接对应 FAN-OUT/FAN-IN，'
\echo '  而 CYCLE/STACK 需要多跳遍历才能发现，规则引擎无法表达）'
\echo ''

WITH pattern_txns AS (
    -- 只取能回连到 txn_norm 的模式交易（排除自环）
    SELECT p.pattern_type, d.txn_id
    FROM pattern_transactions d
    JOIN laundering_patterns p ON p.pattern_id = d.pattern_id
    JOIN txn_norm n            ON n.txn_id     = d.txn_id
    WHERE d.txn_id IS NOT NULL
),
caught AS (
    SELECT DISTINCT txn_id FROM rule_alerts
)
SELECT
    pt.pattern_type                                                  AS "模式类型",
    COUNT(DISTINCT pt.txn_id)                                        AS "交易数",
    COUNT(DISTINCT pt.txn_id) FILTER (WHERE c.txn_id IS NOT NULL)    AS "被抓到",
    ROUND(100.0 * COUNT(DISTINCT pt.txn_id) FILTER (WHERE c.txn_id IS NOT NULL)
          / COUNT(DISTINCT pt.txn_id), 1)                            AS "召回率%"
FROM pattern_txns pt
LEFT JOIN caught c ON c.txn_id = pt.txn_id
GROUP BY pt.pattern_type
ORDER BY 4 DESC;


\echo ''
\echo '--- 规则 x 模式 召回矩阵（每格 = 该规则在该模式上的召回率%）---'

WITH pattern_txns AS (
    SELECT p.pattern_type, d.txn_id
    FROM pattern_transactions d
    JOIN laundering_patterns p ON p.pattern_id = d.pattern_id
    JOIN txn_norm n            ON n.txn_id     = d.txn_id
),
totals AS (
    SELECT pattern_type, COUNT(DISTINCT txn_id) AS n FROM pattern_txns GROUP BY 1
)
SELECT
    t.pattern_type                                                          AS "模式",
    t.n                                                                     AS "总数",
    ROUND(100.0*COUNT(DISTINCT pt.txn_id) FILTER (WHERE a.rule_code='R1')/t.n,1) AS "R1大额",
    ROUND(100.0*COUNT(DISTINCT pt.txn_id) FILTER (WHERE a.rule_code='R2')/t.n,1) AS "R2渠道",
    ROUND(100.0*COUNT(DISTINCT pt.txn_id) FILTER (WHERE a.rule_code='R3')/t.n,1) AS "R3拆分",
    ROUND(100.0*COUNT(DISTINCT pt.txn_id) FILTER (WHERE a.rule_code='R4')/t.n,1) AS "R4高频",
    ROUND(100.0*COUNT(DISTINCT pt.txn_id) FILTER (WHERE a.rule_code='R5')/t.n,1) AS "R5过账",
    ROUND(100.0*COUNT(DISTINCT pt.txn_id) FILTER (WHERE a.rule_code='R6')/t.n,1) AS "R6扇出",
    ROUND(100.0*COUNT(DISTINCT pt.txn_id) FILTER (WHERE a.rule_code='R7')/t.n,1) AS "R7扇入"
FROM totals t
JOIN pattern_txns pt ON pt.pattern_type = t.pattern_type
LEFT JOIN rule_alerts a ON a.txn_id = pt.txn_id
GROUP BY t.pattern_type, t.n
ORDER BY t.pattern_type;


-- ============================================================================
-- D. 阈值敏感性扫描  <- 论证这是范式缺陷而非调参问题
-- ============================================================================
\echo ''
\echo '============================================================'
\echo 'D. 阈值敏感性扫描'
\echo '============================================================'

\echo ''
\echo '--- R1 大额单笔：阈值 (USD) ---'
SELECT
    th                                                          AS "阈值USD",
    COUNT(*) FILTER (WHERE n.amount_usd >= th)                  AS "告警数",
    COUNT(*) FILTER (WHERE n.amount_usd >= th AND n.is_laundering = 1) AS "命中",
    ROUND(100.0 * COUNT(*) FILTER (WHERE n.amount_usd >= th AND n.is_laundering = 1)
          / NULLIF(COUNT(*) FILTER (WHERE n.amount_usd >= th), 0), 3) AS "精确率%",
    ROUND(100.0 * COUNT(*) FILTER (WHERE n.amount_usd >= th AND n.is_laundering = 1)
          / 5166.0, 2)                                          AS "召回率%"
FROM txn_norm n
CROSS JOIN (VALUES (1000),(3000),(5000),(10000),(20000),(50000),(100000)) AS v(th)
GROUP BY th ORDER BY th;
-- CROSS JOIN (VALUES ...) 是个实用技巧：
-- 把一组待测参数当成一张临时表，和数据做笛卡尔积，
-- 一条查询就能同时算出所有阈值的结果，不用写 7 遍。

\echo ''
\echo '--- R4 高频转出：阈值 (24h 内笔数) ---'
SELECT
    th                                                          AS "阈值笔数",
    COUNT(*) FILTER (WHERE v.n_txn_24h >= th)                   AS "告警数",
    COUNT(*) FILTER (WHERE v.n_txn_24h >= th AND v.is_laundering = 1) AS "命中",
    ROUND(100.0 * COUNT(*) FILTER (WHERE v.n_txn_24h >= th AND v.is_laundering = 1)
          / NULLIF(COUNT(*) FILTER (WHERE v.n_txn_24h >= th), 0), 3) AS "精确率%",
    ROUND(100.0 * COUNT(*) FILTER (WHERE v.n_txn_24h >= th AND v.is_laundering = 1)
          / 5166.0, 2)                                          AS "召回率%"
FROM eval_velocity v
CROSS JOIN (VALUES (5),(10),(20),(30),(50),(100),(200)) AS t(th)
GROUP BY th ORDER BY th;

\echo ''
\echo '--- R6 扇出：阈值 (单日不同对手方数) ---'
SELECT
    th                                                          AS "阈值对手方",
    SUM(f.n_txns) FILTER (WHERE f.n_counterparties >= th)       AS "告警数",
    SUM(f.n_laundering) FILTER (WHERE f.n_counterparties >= th) AS "命中",
    ROUND(100.0 * SUM(f.n_laundering) FILTER (WHERE f.n_counterparties >= th)
          / NULLIF(SUM(f.n_txns) FILTER (WHERE f.n_counterparties >= th), 0), 3) AS "精确率%",
    ROUND(100.0 * SUM(f.n_laundering) FILTER (WHERE f.n_counterparties >= th)
          / 5166.0, 2)                                          AS "召回率%"
FROM eval_fanout f
CROSS JOIN (VALUES (4),(6),(8),(12),(16),(24),(32)) AS t(th)
GROUP BY th ORDER BY th;

\echo ''
\echo '--- R7 扇入：阈值 (单日不同付款方数) ---'
SELECT
    th                                                          AS "阈值付款方",
    SUM(f.n_txns) FILTER (WHERE f.n_senders >= th)              AS "告警数",
    SUM(f.n_laundering) FILTER (WHERE f.n_senders >= th)        AS "命中",
    ROUND(100.0 * SUM(f.n_laundering) FILTER (WHERE f.n_senders >= th)
          / NULLIF(SUM(f.n_txns) FILTER (WHERE f.n_senders >= th), 0), 3) AS "精确率%",
    ROUND(100.0 * SUM(f.n_laundering) FILTER (WHERE f.n_senders >= th)
          / 5166.0, 2)                                          AS "召回率%"
FROM eval_fanin f
CROSS JOIN (VALUES (4),(6),(8),(12),(16),(24),(32)) AS t(th)
GROUP BY th ORDER BY th;

\timing off

\echo ''
\echo '============================================================'
\echo '判读要点：'
\echo '  1. D 部分找一找：有没有任何阈值能同时做到 召回>=80% 且 精确率>=5%？'
\echo '  2. C 部分看：CYCLE / STACK 的召回率是不是明显低于 FAN-OUT / FAN-IN？'
\echo '  3. B 部分看：R4 告警 114 万，它的"独有命中"有几个？'
\echo '============================================================'