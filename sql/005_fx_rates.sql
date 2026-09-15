-- ============================================================================
-- SentryAML - 005: 从数据反推汇率，建立美元归一化视图
-- ============================================================================
-- 执行:
--   docker compose exec -T postgres psql -U sentry -d sentryaml -f /sql/005_fx_rates.sql
--
-- 背景：数据集有 15 种货币但没给汇率表，而所有金额阈值规则都需要统一币种。
--
-- 解法：从数据本身反推。
--   已验证"同币种时收付金额必然相等，跨币种时差异纯来自汇率"
--   （跨币种 72,170 笔 vs 金额不等 72,158 笔，两数几乎相同）
--   所以对每笔跨币种交易：  汇率 = amount_received / amount_paid
--
-- 用【中位数】而非平均数聚合：中位数对极端值不敏感，
-- 万一有异常数据，平均值会被拉偏，中位数几乎不受影响。
--
-- 自带验证：如果同一币种对的汇率在几千笔交易里高度一致（离散度极小），
-- 说明推导成立；离散度大则说明假设有问题，需要换方案。
-- ============================================================================

\timing on
\pset pager off


-- ============================================================================
-- 第 1 步：诊断 —— 每个币种对的隐含汇率有多稳定
-- ============================================================================
\echo ''
\echo '=== 币种对的隐含汇率离散度（只看与美元直接配对的）==='

SELECT
    currency_paid                                             AS "付款币种",
    currency_received                                         AS "收款币种",
    COUNT(*)                                                  AS "样本数",
    ROUND(MIN(amount_received / amount_paid), 8)              AS "最小汇率",
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY amount_received / amount_paid
    )::numeric, 8)                                            AS "中位数汇率",
    ROUND(MAX(amount_received / amount_paid), 8)              AS "最大汇率",
    -- 变异系数 = 标准差 / 均值。它是【无量纲】的，
    -- 所以不同数量级的汇率可以直接横向比较离散程度。
    -- 接近 0 表示高度一致；超过 0.01 就要怀疑假设了。
    ROUND(
        STDDEV(amount_received / amount_paid)
        / NULLIF(AVG(amount_received / amount_paid), 0), 8
    )                                                         AS "变异系数"
FROM transactions
WHERE is_cross_currency
  AND amount_paid > 0
  AND (currency_paid = 'US Dollar' OR currency_received = 'US Dollar')
GROUP BY currency_paid, currency_received
ORDER BY COUNT(*) DESC;
-- NULLIF(x, 0) 在 x 等于 0 时返回 NULL，避免除零错误。
-- 除以 NULL 的结果是 NULL，不会让整条查询崩掉。


-- ============================================================================
-- 第 2 步：建汇率表
-- ============================================================================
DROP TABLE IF EXISTS fx_rates CASCADE;

CREATE TABLE fx_rates (
    currency     VARCHAR(24)   NOT NULL,
    -- 1 单位该货币 = 多少美元
    usd_rate     NUMERIC(24,10) NOT NULL,
    -- 推导来源，便于审计
    sample_count INTEGER       NOT NULL,
    cv           NUMERIC(12,8),   -- 变异系数，质量指标

    CONSTRAINT pk_fx_rates PRIMARY KEY (currency),
    CONSTRAINT ck_fx_rate_positive CHECK (usd_rate > 0)
);

COMMENT ON TABLE fx_rates IS '从跨币种交易反推的汇率；非真实市场汇率，仅用于本数据集内部归一化';

-- 把两个方向的交易统一成 "1 单位外币 = 多少美元" 的形式：
--   外币 -> 美元:  rate = amount_received / amount_paid
--   美元 -> 外币:  rate = amount_paid / amount_received   （取倒数）
-- UNION ALL 把两个结果集拼起来（ALL 表示不去重，比 UNION 快）
WITH implied AS (
    SELECT currency_paid AS currency,
           amount_received / amount_paid AS rate
    FROM transactions
    WHERE is_cross_currency
      AND currency_received = 'US Dollar'
      AND amount_paid > 0

    UNION ALL

    SELECT currency_received AS currency,
           amount_paid / amount_received AS rate
    FROM transactions
    WHERE is_cross_currency
      AND currency_paid = 'US Dollar'
      AND amount_received > 0
)
INSERT INTO fx_rates (currency, usd_rate, sample_count, cv)
SELECT
    currency,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY rate)::numeric,
    COUNT(*),
    STDDEV(rate) / NULLIF(AVG(rate), 0)
FROM implied
GROUP BY currency;

-- 美元自己对自己是 1，单独插入
INSERT INTO fx_rates (currency, usd_rate, sample_count, cv)
VALUES ('US Dollar', 1.0, 0, 0);


-- ============================================================================
-- 第 3 步：检查覆盖度 —— 有没有货币缺汇率
-- ============================================================================
\echo ''
\echo '=== 汇率表 ==='

SELECT currency      AS "货币",
       usd_rate      AS "1单位=?美元",
       sample_count  AS "样本数",
       ROUND(cv, 6)  AS "变异系数"
FROM fx_rates
ORDER BY usd_rate DESC;

\echo ''
\echo '=== 缺汇率的货币（必须为空，否则归一化会漏数据）==='

SELECT DISTINCT t.currency_paid AS "缺汇率的货币"
FROM transactions t
LEFT JOIN fx_rates f ON f.currency = t.currency_paid
WHERE f.currency IS NULL;


-- ============================================================================
-- 第 4 步：美元归一化视图
-- ============================================================================
-- 用视图而不是给 transactions 加一列并 UPDATE：
--   UPDATE 508 万行会导致 Postgres 重写整张表（MVCC 机制下每次更新都是
--   "写新版本 + 标记旧版本失效"），代价是几百 MB 的额外写入和一次 VACUUM。
--   而 fx_rates 只有 15 行，JOIN 它的代价极低（哈希连接，一次性装进内存）。
-- 这也符合"先用简单方案、测出瓶颈再优化"的原则。
-- ============================================================================
DROP VIEW IF EXISTS txn_usd CASCADE;

CREATE VIEW txn_usd AS
SELECT
    t.txn_id,
    t.txn_ts,
    t.from_bank,
    t.from_account,
    t.to_bank,
    t.to_account,
    t.amount_paid,
    t.currency_paid,
    -- 归一化后的美元金额
    (t.amount_paid * f.usd_rate)::numeric(24,2)  AS amount_usd,
    t.payment_format,
    t.is_laundering,
    t.is_self_loop,
    t.is_cross_currency,
    t.is_intra_branch
FROM transactions t
JOIN fx_rates f ON f.currency = t.currency_paid;

COMMENT ON VIEW txn_usd IS '交易表 + 美元归一化金额；amount_usd 基于 fx_rates 反推汇率';


-- ============================================================================
-- 第 5 步：归一化结果的合理性检查
-- ============================================================================
\echo ''
\echo '=== 归一化后的金额分布（按币种）==='
-- 如果汇率推导正确，各币种归一化后的【中位数金额】应该在同一数量级。
-- 某个币种的中位数比别的高/低几个数量级，说明它的汇率推错了。

SELECT
    currency_paid                                    AS "币种",
    COUNT(*)                                         AS "笔数",
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (
        ORDER BY amount_usd
    )::numeric, 2)                                   AS "中位数(USD)",
    ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (
        ORDER BY amount_usd
    )::numeric, 2)                                   AS "P99(USD)",
    ROUND(MAX(amount_usd), 2)                        AS "最大(USD)"
FROM txn_usd
WHERE NOT is_self_loop
GROUP BY currency_paid
ORDER BY PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount_usd) DESC;

\echo ''
\echo '=== 洗钱交易 vs 正常交易的金额分布 ==='

SELECT
    is_laundering                                    AS "是否洗钱",
    COUNT(*)                                         AS "笔数",
    ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount_usd)::numeric, 2) AS "P25",
    ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY amount_usd)::numeric, 2) AS "中位数",
    ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount_usd)::numeric, 2) AS "P75",
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount_usd)::numeric, 2) AS "P95"
FROM txn_usd
WHERE NOT is_self_loop
GROUP BY is_laundering;

\timing off