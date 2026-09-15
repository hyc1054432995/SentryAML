-- ============================================================================
-- SentryAML - 006: 规则引擎 Baseline
-- ============================================================================
-- 执行:
--   docker compose exec -T postgres psql -U sentry -d sentryaml -f /sql/006_rule_baseline.sql
--
-- 目的：复刻传统交易监控系统(TMS)的规则集，量化它的误报率。
-- 这个数字是整个项目的参照系，也是选题成立的实证依据。
--
-- 【设计前提】告警粒度 = 逐笔交易。
--   真实 TMS 通常按"账户-周期"产生告警，这里简化为逐笔，
--   理由是与逐笔标签 1:1 可比，且与后续 ML 打分粒度一致。
--   这是一个需要在面试中主动说明的简化。
--
-- 【关键发现驱动的设计】
--   洗钱交易 P25-P95 = 2,154 ~ 24,075 USD，
--   正常交易 P25-P95 =   176 ~ 137,366 USD。
--   洗钱金额被压缩在狭窄区间内、贴着监管申报阈值(CTR 10,000 USD)走 ——
--   这是典型的 structuring。因此单笔金额规则注定失效，
--   规则越靠近"多笔之间的关系"，效果应该越好。
--   R1/R2 -> R3/R4/R5 -> R6/R7 正是按这个梯度排列的。
-- ============================================================================

\timing on
\pset pager off


-- ============================================================================
-- 第 0 步：物化归一化交易表
-- ============================================================================
-- 普通视图每次查询都要重新 JOIN fx_rates 并扫全表（实测约 7 秒），
-- 而且视图上不能建索引，窗口函数每次都要重新排序 449 万行。
--
-- 物化视图(MATERIALIZED VIEW)会把结果真正存下来、可以建索引，
-- 代价是数据变了要手动 REFRESH。我们的底层数据是静态的，完美契合。
-- 用一次性存储换掉七条规则的重复计算。
-- ============================================================================
DROP MATERIALIZED VIEW IF EXISTS txn_norm CASCADE;

CREATE MATERIALIZED VIEW txn_norm AS
SELECT
    txn_id,
    txn_ts,
    txn_ts::date        AS txn_date,   -- 自然日，给 tumbling 窗口用
    from_bank,
    from_account,
    to_bank,
    to_account,
    amount_usd,
    payment_format,
    is_laundering
FROM txn_usd
WHERE NOT is_self_loop;               -- 自环是记账，不参与规则判定

-- 出边方向：R3/R4/R6 按付款账户聚合
CREATE INDEX idx_norm_from ON txn_norm (from_bank, from_account, txn_ts);
-- 入边方向：R7 按收款账户聚合
CREATE INDEX idx_norm_to   ON txn_norm (to_bank, to_account, txn_ts);
-- 主键，供 alerts 表回连
CREATE UNIQUE INDEX idx_norm_txn ON txn_norm (txn_id);

ANALYZE txn_norm;


-- ============================================================================
-- 第 1 步：规则定义表
-- ============================================================================
-- 把规则的元信息存进数据库而不是散落在代码注释里，
-- 好处是评测报告可以直接 JOIN 出规则名称和参数，
-- 而且以后做阈值敏感性分析时，参数是可查询的数据而不是硬编码。
-- ============================================================================
DROP TABLE IF EXISTS rule_alerts CASCADE;
DROP TABLE IF EXISTS rule_definitions CASCADE;

CREATE TABLE rule_definitions (
    rule_code    VARCHAR(8)   NOT NULL,
    rule_name    VARCHAR(64)  NOT NULL,
    rule_tier    SMALLINT     NOT NULL,   -- 1=单笔属性 2=账户时间窗 3=一跳关系
    window_type  VARCHAR(16)  NOT NULL,   -- none / sliding_24h / daily
    description  TEXT         NOT NULL,

    CONSTRAINT pk_rule_definitions PRIMARY KEY (rule_code),
    CONSTRAINT ck_rule_tier CHECK (rule_tier IN (1, 2, 3))
);

INSERT INTO rule_definitions VALUES
('R1', '大额单笔交易',       1, 'none',
 '单笔 >= 10,000 USD。对应 FinCEN CTR 申报门槛，是最经典也最基础的规则。'),
('R2', '高风险渠道大额',     1, 'none',
 '支付方式为 Bitcoin/Cash/Wire 且单笔 >= 5,000 USD。现金与虚拟资产可追溯性低。'),
('R3', '结构化拆分',         2, 'sliding_24h',
 '账户 24 小时内有 >= 3 笔金额落在 3,000-10,000 USD 的转出，疑似规避申报阈值。'),
('R4', '异常高频转出',       2, 'sliding_24h',
 '账户 24 小时内转出 >= 10 笔。'),
('R5', '快进快出',           2, 'daily',
 '账户单日流入与流出金额比值在 0.9-1.1 之间且双向均 >= 10,000 USD，疑似过账。'),
('R6', '资金扇出',           3, 'daily',
 '账户单日转给 >= 8 个不同对手方。直接对应 FAN-OUT 洗钱模式。'),
('R7', '资金扇入',           3, 'daily',
 '账户单日从 >= 8 个不同对手方收款。直接对应 FAN-IN 洗钱模式。');


-- ============================================================================
-- 第 2 步：告警表
-- ============================================================================
CREATE TABLE rule_alerts (
    alert_id      BIGSERIAL     NOT NULL,
    rule_code     VARCHAR(8)    NOT NULL,
    txn_id        BIGINT        NOT NULL,
    -- 触发告警的主体账户（单笔规则填付款方）
    subject_bank  INTEGER       NOT NULL,
    subject_acct  VARCHAR(16)   NOT NULL,
    -- 触发时的度量值（金额/笔数/对手方数），便于事后解释与调阈值
    trigger_value NUMERIC(24,4),

    CONSTRAINT pk_rule_alerts PRIMARY KEY (alert_id),
    -- 同一条规则对同一笔交易只告警一次
    CONSTRAINT uq_rule_alerts UNIQUE (rule_code, txn_id)
);


-- ============================================================================
-- R1 - 大额单笔交易（Tier 1）
-- ============================================================================
INSERT INTO rule_alerts (rule_code, txn_id, subject_bank, subject_acct, trigger_value)
SELECT 'R1', txn_id, from_bank, from_account, amount_usd
FROM txn_norm
WHERE amount_usd >= 10000;


-- ============================================================================
-- R2 - 高风险渠道大额（Tier 1）
-- ============================================================================
INSERT INTO rule_alerts (rule_code, txn_id, subject_bank, subject_acct, trigger_value)
SELECT 'R2', txn_id, from_bank, from_account, amount_usd
FROM txn_norm
WHERE payment_format IN ('Bitcoin', 'Cash', 'Wire')
  AND amount_usd >= 5000;


-- ============================================================================
-- R3 - 结构化拆分（Tier 2，滑动 24h 窗口）
-- ============================================================================
-- 窗口函数语法拆解：
--   OVER (PARTITION BY ... ORDER BY ... RANGE BETWEEN ... )
--   PARTITION BY = 按账户分组，每组内部独立计算
--   ORDER BY txn_ts = 组内按时间排序
--   RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
--       = 窗口范围是"当前行往前 24 小时"，这是【滑动窗口】
--
-- 注意用 SUM(CASE WHEN ...) 而不是 COUNT(*) FILTER (...)：
-- 窗口函数不支持 FILTER 子句，只能用 CASE 变通。
--
-- 窗口函数不能直接写在 WHERE 里（执行顺序上 WHERE 先于窗口计算），
-- 所以必须先在子查询/CTE 里算出来，再在外层过滤。
-- ============================================================================
WITH banded AS (
    SELECT
        txn_id, from_bank, from_account, amount_usd,
        SUM(CASE WHEN amount_usd >= 3000 AND amount_usd < 10000 THEN 1 ELSE 0 END)
            OVER (
                PARTITION BY from_bank, from_account
                ORDER BY txn_ts
                RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
            ) AS n_in_band
    FROM txn_norm
)
INSERT INTO rule_alerts (rule_code, txn_id, subject_bank, subject_acct, trigger_value)
SELECT 'R3', txn_id, from_bank, from_account, n_in_band
FROM banded
WHERE n_in_band >= 3
  -- 被告警的交易自己也必须落在可疑金额带内
  AND amount_usd >= 3000 AND amount_usd < 10000;


-- ============================================================================
-- R4 - 异常高频转出（Tier 2，滑动 24h 窗口）
-- ============================================================================
WITH velocity AS (
    SELECT
        txn_id, from_bank, from_account,
        COUNT(*) OVER (
            PARTITION BY from_bank, from_account
            ORDER BY txn_ts
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
        ) AS n_txn_24h
    FROM txn_norm
)
INSERT INTO rule_alerts (rule_code, txn_id, subject_bank, subject_acct, trigger_value)
SELECT 'R4', txn_id, from_bank, from_account, n_txn_24h
FROM velocity
WHERE n_txn_24h >= 10;


-- ============================================================================
-- R5 - 快进快出 / 过账（Tier 2，自然日窗口）
-- ============================================================================
-- 需要同时看一个账户的流入和流出，窗口函数做不到（它只能在单一排序上滑动），
-- 所以用自然日聚合 + 两个方向分别汇总后关联。
--
-- FULL JOIN：只有流入没有流出（或反之）的账户-日也要保留，
-- 用 COALESCE 把 NULL 当成 0 处理。
-- ============================================================================
WITH daily_out AS (
    SELECT from_bank AS bank, from_account AS acct, txn_date,
           SUM(amount_usd) AS total_out
    FROM txn_norm
    GROUP BY 1, 2, 3
),
daily_in AS (
    SELECT to_bank AS bank, to_account AS acct, txn_date,
           SUM(amount_usd) AS total_in
    FROM txn_norm
    GROUP BY 1, 2, 3
),
pass_through AS (
    SELECT
        COALESCE(o.bank, i.bank)           AS bank,
        COALESCE(o.acct, i.acct)           AS acct,
        COALESCE(o.txn_date, i.txn_date)   AS txn_date,
        COALESCE(o.total_out, 0)           AS total_out,
        COALESCE(i.total_in, 0)            AS total_in
    FROM daily_out o
    FULL JOIN daily_in i
           ON o.bank = i.bank AND o.acct = i.acct AND o.txn_date = i.txn_date
)
INSERT INTO rule_alerts (rule_code, txn_id, subject_bank, subject_acct, trigger_value)
SELECT 'R5', t.txn_id, t.from_bank, t.from_account,
       ROUND(p.total_out / NULLIF(p.total_in, 0), 4)
FROM pass_through p
JOIN txn_norm t
  ON t.from_bank = p.bank AND t.from_account = p.acct AND t.txn_date = p.txn_date
WHERE p.total_in  >= 10000
  AND p.total_out >= 10000
  AND p.total_out / NULLIF(p.total_in, 0) BETWEEN 0.9 AND 1.1;


-- ============================================================================
-- R6 - 资金扇出（Tier 3，自然日窗口）
-- ============================================================================
-- COUNT(DISTINCT ...) 在窗口函数里【不被支持】，
-- 这是这条规则用自然日聚合而非滑动窗口的唯一原因。
-- 代价：跨午夜的扇出会被切成两段，可能双双不触发。
-- 缓解方案(未实现)：再跑一遍偏移 12 小时的分桶取并集。
--
-- ROW(a, b) 把两列打包成一个复合值，
-- 这样 COUNT(DISTINCT) 才能对"银行+账户"这个组合去重。
-- ============================================================================
WITH fan_out AS (
    SELECT from_bank, from_account, txn_date,
           COUNT(DISTINCT ROW(to_bank, to_account)) AS n_counterparties
    FROM txn_norm
    GROUP BY 1, 2, 3
    HAVING COUNT(DISTINCT ROW(to_bank, to_account)) >= 8
)
INSERT INTO rule_alerts (rule_code, txn_id, subject_bank, subject_acct, trigger_value)
SELECT 'R6', t.txn_id, t.from_bank, t.from_account, f.n_counterparties
FROM fan_out f
JOIN txn_norm t
  ON t.from_bank = f.from_bank
 AND t.from_account = f.from_account
 AND t.txn_date = f.txn_date;


-- ============================================================================
-- R7 - 资金扇入（Tier 3，自然日窗口）
-- ============================================================================
-- 主体是【收款账户】，所以告警挂在入边上。
-- ============================================================================
WITH fan_in AS (
    SELECT to_bank, to_account, txn_date,
           COUNT(DISTINCT ROW(from_bank, from_account)) AS n_senders
    FROM txn_norm
    GROUP BY 1, 2, 3
    HAVING COUNT(DISTINCT ROW(from_bank, from_account)) >= 8
)
INSERT INTO rule_alerts (rule_code, txn_id, subject_bank, subject_acct, trigger_value)
SELECT 'R7', t.txn_id, t.to_bank, t.to_account, f.n_senders
FROM fan_in f
JOIN txn_norm t
  ON t.to_bank = f.to_bank
 AND t.to_account = f.to_account
 AND t.txn_date = f.txn_date;


-- ============================================================================
-- 索引与统计
-- ============================================================================
CREATE INDEX idx_alerts_txn  ON rule_alerts (txn_id);
CREATE INDEX idx_alerts_rule ON rule_alerts (rule_code);
ANALYZE rule_alerts;


-- ============================================================================
-- 快速概览（详细评测在 007）
-- ============================================================================
\echo ''
\echo '=== 各规则告警量 ==='

SELECT
    d.rule_code                           AS "规则",
    d.rule_name                           AS "名称",
    d.rule_tier                           AS "层级",
    d.window_type                         AS "窗口",
    COUNT(a.alert_id)                     AS "告警数"
FROM rule_definitions d
LEFT JOIN rule_alerts a ON a.rule_code = d.rule_code
GROUP BY d.rule_code, d.rule_name, d.rule_tier, d.window_type
ORDER BY d.rule_code;

\echo ''
\echo '=== 去重后的告警交易总数 ==='

SELECT COUNT(DISTINCT txn_id) AS "被告警的交易数" FROM rule_alerts;

\timing off