-- ============================================================================
-- SentryAML - 008: 节点度数表（图特征的基础）
-- ============================================================================
-- 执行:
--   docker compose exec -T postgres psql -U sentry -d sentryaml -f /sql/008_node_degrees.sql
--
-- 产出两张表，一表三吃：
--   1. 超级节点识别 -> 递归遍历的剪枝依据
--   2. ML 核心特征  -> 直接进模型
--   3. Agent 取证材料 -> 数字最终要能写成一句人话
--      （"该账户 17 天内仅 2 笔转出但单笔均超 3,000 USD，符合骡子账户特征"）
--
-- 【度数的两种数法，必须分开算】
--   交易笔数   : 转出了几笔
--   对手方数   : 转给了几个【不同】的账户
--   "10 笔 / 1 个对手方" = 固定伙伴往来（正常）
--   "10 笔 / 10 个对手方" = 资金撒向四面八方（FAN-OUT 特征）
--
-- 【重要前提】基于 txn_norm（已排除自环、已归一化为美元）。
-- 自环是记账不是转账，计入度数会让所有图特征失真。
-- ============================================================================

\timing on
\pset pager off


-- ============================================================================
-- 第 1 步：账户级度数
-- ============================================================================
-- 实现技巧：把每笔交易【拆成两行】——付款方视角一行、收款方视角一行，
-- 合并成一张 897 万行的"参与表"，然后一次聚合同时算出出度和入度。
--
-- 替代方案是分别算出度和入度再 FULL JOIN，但那样要扫两遍表、
-- 而且"活跃天数"这种跨方向的指标还得再单独算一次。
-- 拆行合并只扫一遍，代价是中间结果翻倍。
-- ============================================================================
DROP TABLE IF EXISTS account_degrees CASCADE;

CREATE TABLE account_degrees AS
WITH participation AS (
    -- 付款方视角
    SELECT from_bank AS bank_id, from_account AS account_number,
           'out'     AS direction,
           to_bank   AS cp_bank, to_account AS cp_account,
           amount_usd, txn_ts, txn_date, is_laundering
    FROM txn_norm

    UNION ALL

    -- 收款方视角
    SELECT to_bank   AS bank_id, to_account AS account_number,
           'in'      AS direction,
           from_bank AS cp_bank, from_account AS cp_account,
           amount_usd, txn_ts, txn_date, is_laundering
    FROM txn_norm
)
SELECT
    bank_id,
    account_number,

    -- ---------- 出度 ----------
    COUNT(*) FILTER (WHERE direction = 'out')                       AS out_txn_count,
    COUNT(DISTINCT ROW(cp_bank, cp_account))
        FILTER (WHERE direction = 'out')                            AS out_cp_count,
    COALESCE(SUM(amount_usd) FILTER (WHERE direction = 'out'), 0)   AS out_amount_sum,
    -- FILTER 子句同样适用于 PERCENTILE_CONT 这类有序集聚合函数
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount_usd)
        FILTER (WHERE direction = 'out')                            AS out_amount_median,
    MAX(amount_usd) FILTER (WHERE direction = 'out')                AS out_amount_max,

    -- ---------- 入度 ----------
    COUNT(*) FILTER (WHERE direction = 'in')                        AS in_txn_count,
    COUNT(DISTINCT ROW(cp_bank, cp_account))
        FILTER (WHERE direction = 'in')                             AS in_cp_count,
    COALESCE(SUM(amount_usd) FILTER (WHERE direction = 'in'), 0)    AS in_amount_sum,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount_usd)
        FILTER (WHERE direction = 'in')                             AS in_amount_median,
    MAX(amount_usd) FILTER (WHERE direction = 'in')                 AS in_amount_max,

    -- ---------- 合计 ----------
    COUNT(*)                                                        AS total_txn_count,
    COUNT(DISTINCT ROW(cp_bank, cp_account))                        AS total_cp_count,

    -- ---------- 时间行为 ----------
    COUNT(DISTINCT txn_date)                                        AS active_days,
    MIN(txn_ts)                                                     AS first_ts,
    MAX(txn_ts)                                                     AS last_ts,
    -- 活跃时间跨度（小时）。跨度小而笔数多 = 短期爆发
    ROUND(EXTRACT(EPOCH FROM (MAX(txn_ts) - MIN(txn_ts))) / 3600.0, 2)
                                                                    AS span_hours,

    -- ---------- 标签（供分析用，建模时要小心不能当特征）----------
    MAX(is_laundering)                                              AS touched_laundering,
    SUM(is_laundering)                                              AS laundering_txn_count

FROM participation
GROUP BY bank_id, account_number;


-- 派生指标单独用 ALTER + UPDATE 加，而不是写在上面的聚合里，
-- 是因为它们依赖已聚合的结果（比如 out_amount_sum / in_amount_sum），
-- 在同一个 SELECT 里没法引用刚算出来的别名。
ALTER TABLE account_degrees
    -- 比值型特征在重尾分布下会产生极端值（流入极小时比值可炸到 10^10 以上），
    -- 所以精度要给足，并在下面的 UPDATE 里做封顶。
    ADD COLUMN flow_ratio        NUMERIC(24,6),   -- 流出/流入，接近 1 = 过账
    ADD COLUMN net_flow          NUMERIC(24,2),   -- 净流入（正=沉淀，负=流出）
    ADD COLUMN txn_per_cp        NUMERIC(12,4),   -- 笔数/对手方数，衡量集中度
    ADD COLUMN txn_per_day       NUMERIC(12,4),   -- 日均笔数
    ADD COLUMN is_hub            BOOLEAN DEFAULT FALSE;

UPDATE account_degrees
SET flow_ratio  = LEAST(
                      ROUND(out_amount_sum / NULLIF(in_amount_sum, 0), 6),
                      999999.0
                  ),
    net_flow    = in_amount_sum - out_amount_sum,
    -- txn_per_cp 接近 1 = 每个对手方只来往一次（撒网型）
    -- 数值很大   = 和少数对手方反复交易（固定伙伴）
    txn_per_cp  = ROUND(total_txn_count::numeric / NULLIF(total_cp_count, 0), 4),
    txn_per_day = ROUND(total_txn_count::numeric / NULLIF(active_days, 0), 4);

ALTER TABLE account_degrees
    ADD CONSTRAINT pk_account_degrees PRIMARY KEY (bank_id, account_number);

CREATE INDEX idx_degrees_out_cp   ON account_degrees (out_cp_count DESC);
CREATE INDEX idx_degrees_in_cp    ON account_degrees (in_cp_count DESC);
CREATE INDEX idx_degrees_total_cp ON account_degrees (total_cp_count DESC);
CREATE INDEX idx_degrees_hub      ON account_degrees (is_hub) WHERE is_hub;

ANALYZE account_degrees;


-- ============================================================================
-- 第 2 步：用数据定枢纽阈值（不拍脑袋）
-- ============================================================================
\echo ''
\echo '=== 对手方数的分位数分布 ==='
\echo '（重尾分布下平均数没有意义，看分位数才能定阈值）'

SELECT
    'out_cp_count' AS "指标",
    ROUND(AVG(out_cp_count), 2)                                              AS "平均",
    PERCENTILE_DISC(0.50) WITHIN GROUP (ORDER BY out_cp_count)               AS "P50",
    PERCENTILE_DISC(0.90) WITHIN GROUP (ORDER BY out_cp_count)               AS "P90",
    PERCENTILE_DISC(0.99) WITHIN GROUP (ORDER BY out_cp_count)               AS "P99",
    PERCENTILE_DISC(0.999) WITHIN GROUP (ORDER BY out_cp_count)              AS "P99.9",
    PERCENTILE_DISC(0.9999) WITHIN GROUP (ORDER BY out_cp_count)             AS "P99.99",
    MAX(out_cp_count)                                                        AS "最大"
FROM account_degrees
UNION ALL
SELECT
    'in_cp_count',
    ROUND(AVG(in_cp_count), 2),
    PERCENTILE_DISC(0.50) WITHIN GROUP (ORDER BY in_cp_count),
    PERCENTILE_DISC(0.90) WITHIN GROUP (ORDER BY in_cp_count),
    PERCENTILE_DISC(0.99) WITHIN GROUP (ORDER BY in_cp_count),
    PERCENTILE_DISC(0.999) WITHIN GROUP (ORDER BY in_cp_count),
    PERCENTILE_DISC(0.9999) WITHIN GROUP (ORDER BY in_cp_count),
    MAX(in_cp_count)
FROM account_degrees
UNION ALL
SELECT
    'total_cp_count',
    ROUND(AVG(total_cp_count), 2),
    PERCENTILE_DISC(0.50) WITHIN GROUP (ORDER BY total_cp_count),
    PERCENTILE_DISC(0.90) WITHIN GROUP (ORDER BY total_cp_count),
    PERCENTILE_DISC(0.99) WITHIN GROUP (ORDER BY total_cp_count),
    PERCENTILE_DISC(0.999) WITHIN GROUP (ORDER BY total_cp_count),
    PERCENTILE_DISC(0.9999) WITHIN GROUP (ORDER BY total_cp_count),
    MAX(total_cp_count)
FROM account_degrees;
-- PERCENTILE_DISC 返回数据中【实际存在】的某个值（离散），
-- PERCENTILE_CONT 会在两个值之间【插值】（连续）。
-- 计数类指标用 DISC 更合适——"3.7 个对手方"没有意义。


-- 枢纽判定：清算层识别，而非"度数高就是枢纽"。
-- 依据：洗钱浓度按度数分档呈【非单调】——峰值在 11-38 档(0.3761%)，
-- 而 >200 档回落至 0.1398%（与全网 0.1151% 持平）。
-- 说明 >200 的节点是基础设施(银行 70 Oasis Thrift 的 15 个清算账户，
-- 承载 5% 全网交易量)，不是风险节点。
-- 用 P99.9(阈值 38) 剪枝会剪掉洗钱浓度最高的区间，是错的。
UPDATE account_degrees
SET is_hub = TRUE
WHERE total_cp_count > 200;

ANALYZE account_degrees;

\echo ''
\echo '=== 枢纽节点统计 ==='

SELECT
    COUNT(*) FILTER (WHERE is_hub)                                    AS "枢纽节点数",
    COUNT(*)                                                          AS "账户总数",
    ROUND(100.0 * COUNT(*) FILTER (WHERE is_hub) / COUNT(*), 4)       AS "占比%",
    MIN(total_cp_count) FILTER (WHERE is_hub)                         AS "枢纽阈值",
    -- 枢纽节点承载了多少比例的交易量 —— 这个数字决定剪枝的影响面
    ROUND(100.0 * SUM(total_txn_count) FILTER (WHERE is_hub)
          / SUM(total_txn_count), 2)                                  AS "承载交易占比%"
FROM account_degrees;


-- ============================================================================
-- 第 3 步：实体级度数
-- ============================================================================
-- 一个实体可持有多个账户、跨多家银行（数据里 63,362 个实体是跨行的）。
-- 把钱拆散到自己名下不同银行的账户来规避监控，就是 structuring ——
-- 这在账户级图上完全看不见，只有实体级图能暴露。
-- ============================================================================
DROP TABLE IF EXISTS entity_degrees CASCADE;

CREATE TABLE entity_degrees AS
WITH participation AS (
    SELECT a.entity_id,
           'out' AS direction,
           ca.entity_id AS cp_entity_id,
           n.amount_usd, n.txn_date, n.is_laundering
    FROM txn_norm n
    JOIN accounts a  ON a.bank_id  = n.from_bank AND a.account_number  = n.from_account
    JOIN accounts ca ON ca.bank_id = n.to_bank   AND ca.account_number = n.to_account

    UNION ALL

    SELECT a.entity_id,
           'in',
           ca.entity_id,
           n.amount_usd, n.txn_date, n.is_laundering
    FROM txn_norm n
    JOIN accounts a  ON a.bank_id  = n.to_bank   AND a.account_number  = n.to_account
    JOIN accounts ca ON ca.bank_id = n.from_bank AND ca.account_number = n.from_account
)
SELECT
    p.entity_id,
    e.entity_type,

    COUNT(*) FILTER (WHERE direction = 'out')                         AS out_txn_count,
    COUNT(DISTINCT cp_entity_id) FILTER (WHERE direction = 'out')     AS out_cp_entity_count,
    COALESCE(SUM(amount_usd) FILTER (WHERE direction = 'out'), 0)     AS out_amount_sum,

    COUNT(*) FILTER (WHERE direction = 'in')                          AS in_txn_count,
    COUNT(DISTINCT cp_entity_id) FILTER (WHERE direction = 'in')      AS in_cp_entity_count,
    COALESCE(SUM(amount_usd) FILTER (WHERE direction = 'in'), 0)      AS in_amount_sum,

    COUNT(*)                                                          AS total_txn_count,
    COUNT(DISTINCT cp_entity_id)                                      AS total_cp_entity_count,
    COUNT(DISTINCT txn_date)                                          AS active_days,
    SUM(is_laundering)                                                AS laundering_txn_count

FROM participation p
JOIN entities e ON e.entity_id = p.entity_id
GROUP BY p.entity_id, e.entity_type;

-- 补上"名下账户数"和"跨行数"——这两个来自 accounts 表，不是交易算出来的
ALTER TABLE entity_degrees
    ADD COLUMN n_accounts INTEGER,
    ADD COLUMN n_banks    INTEGER;

UPDATE entity_degrees ed
SET n_accounts = s.n_accounts,
    n_banks    = s.n_banks
FROM (
    SELECT entity_id,
           COUNT(*)                  AS n_accounts,
           COUNT(DISTINCT bank_id)   AS n_banks
    FROM accounts
    GROUP BY entity_id
) s
WHERE s.entity_id = ed.entity_id;

ALTER TABLE entity_degrees
    ADD CONSTRAINT pk_entity_degrees PRIMARY KEY (entity_id);

CREATE INDEX idx_entity_deg_cp    ON entity_degrees (total_cp_entity_count DESC);
CREATE INDEX idx_entity_deg_banks ON entity_degrees (n_banks DESC);

ANALYZE entity_degrees;


-- ============================================================================
-- 第 4 步：完整性检查
-- ============================================================================
\echo ''
\echo '=== 完整性检查 ==='

SELECT
    (SELECT COUNT(*) FROM account_degrees)                      AS "账户度数表行数",
    (SELECT COUNT(DISTINCT ROW(from_bank, from_account))
     FROM txn_norm)                                             AS "出现过的付款账户",
    (SELECT COUNT(*) FROM entity_degrees)                       AS "实体度数表行数",
    (SELECT COUNT(*) FROM entities)                             AS "实体总数";

\echo ''
\echo '（账户度数表行数应 <= 515,088，因为只包含在 txn_norm 中出现过的账户；'
\echo '  实体度数表同理会略少于 166,207）'


-- ============================================================================
-- 第 5 步：特征有效性预览 —— 洗钱 vs 正常账户的度数差异
-- ============================================================================
-- 这一步直接预告"度数特征到底有没有用"。
-- 如果两组的分布几乎重合，说明这些特征区分度很弱，要重新设计。
--
-- 注意：这里用 touched_laundering 只是【探索性分析】。
-- 建模时绝不能把它当特征——它由标签派生，属于典型的数据泄漏。
-- ============================================================================
\echo ''
\echo '=== 涉及洗钱的账户 vs 正常账户（中位数对比）==='

SELECT
    CASE WHEN touched_laundering = 1 THEN '涉洗钱' ELSE '正常' END   AS "账户类型",
    COUNT(*)                                                          AS "账户数",
    PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY out_txn_count)        AS "出度笔数",
    PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY out_cp_count)         AS "出度对手方",
    PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY in_txn_count)         AS "入度笔数",
    PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY in_cp_count)          AS "入度对手方",
    PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY active_days)          AS "活跃天数",
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY out_amount_median)::numeric, 2)
                                                                      AS "转出中位额",
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY txn_per_cp)::numeric, 2)
                                                                      AS "笔数/对手方"
FROM account_degrees
GROUP BY touched_laundering
ORDER BY touched_laundering DESC;


\echo ''
\echo '=== 度数最高的 10 个账户（看看枢纽长什么样）==='

SELECT
    d.bank_id                    AS "银行",
    d.account_number             AS "账户",
    e.entity_type                AS "实体类型",
    d.out_cp_count               AS "出对手方",
    d.in_cp_count                AS "入对手方",
    d.total_txn_count            AS "总笔数",
    d.active_days                AS "活跃天数",
    d.touched_laundering         AS "涉洗钱"
FROM account_degrees d
JOIN accounts a  ON a.bank_id = d.bank_id AND a.account_number = d.account_number
JOIN entities e  ON e.entity_id = a.entity_id
ORDER BY d.total_cp_count DESC
LIMIT 10;

\timing off