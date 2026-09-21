-- ============================================================================
-- SentryAML - 010: 离线特征快照（M2 第一步）
-- ============================================================================
-- 执行:
--   docker compose exec -T postgres psql -U sentry -d sentryaml -f /sql/010_feature_snapshot.sql
--
-- 为每个账户、每个日期（9/3 ~ 9/10）计算一行，
-- 内容是【截至当天 00:00】的过去 1 天 / 3 天行为统计。
--
-- 【为什么这样能保证时点正确】
--   9/5 当天的所有交易，都只使用 as_of_date = 9/5 的快照，
--   而这份快照只包含 9/5 00:00 之前的交易。
--   不可能混入任何 9/5 及以后的信息 —— 正确性由结构保证，不靠逐笔检查。
--
-- 【为什么拆成"离线快照 + 当日实时"两部分】
--   1. COUNT(DISTINCT) 不能用在窗口函数里，但可以用在 GROUP BY 里；
--   2. 逐笔回查会在清算账户上爆炸（单账户 3 天内十几万笔）；
--   3. 真实银行的特征系统本来就是这样设计的：
--      夜间批量算 T-1 快照（offline store），白天用内存计数器补当天的部分（online store）。
--   当天 00:00 到交易发生前的部分，在 011 里用窗口函数计算。
--
-- 【时间范围】
--   样本期 9/3 ~ 9/10（见 M1 的切分方案）；
--   最晚的快照是 9/10 00:00，所以只需要读 9/10 之前的交易。
--   9/11 以后只剩洗钱尾巴，完全不参与。
-- ============================================================================

\timing on
\pset pager off

-- 这一步要对约 2,000 万行做分组和去重，给足内存可以减少落盘排序
SET work_mem = '256MB';


-- ============================================================================
-- 第 1 步：计算快照
-- ============================================================================
DROP TABLE IF EXISTS feat_acct_snapshot CASCADE;

CREATE TABLE feat_acct_snapshot AS
WITH
-- 快照时点：9/3 00:00 ~ 9/10 00:00，每天一个
as_of AS (
    SELECT d::timestamp AS as_of
    FROM generate_series('2022-09-03'::date, '2022-09-10'::date, INTERVAL '1 day') AS g(d)
),
-- 把每笔交易拆成"付款方视角"和"收款方视角"两行（和 008 同样的做法）
part AS (
    SELECT from_bank  AS bank_id, from_account AS account_number, 'out'::text AS dir,
           to_bank    AS cp_bank, to_account   AS cp_acct,
           amount_usd, txn_ts
    FROM txn_norm
    WHERE txn_ts < '2022-09-10'

    UNION ALL

    SELECT to_bank, to_account, 'in',
           from_bank, from_account,
           amount_usd, txn_ts
    FROM txn_norm
    WHERE txn_ts < '2022-09-10'
)
SELECT
    p.bank_id,
    p.account_number,
    a.as_of::date                                                         AS as_of_date,

    -- ---------------- 过去 1 天（as_of 前 24 小时）----------------
    -- 1 天窗口是 3 天窗口的子集，用 FILTER 把更早的两天排除，一次扫描算两个窗口
    COUNT(*)            FILTER (WHERE dir = 'out' AND txn_ts >= a.as_of - INTERVAL '1 day') AS out_cnt_1d,
    COUNT(DISTINCT ROW(cp_bank, cp_acct))
                        FILTER (WHERE dir = 'out' AND txn_ts >= a.as_of - INTERVAL '1 day') AS out_cp_1d,
    COALESCE(SUM(amount_usd)
                        FILTER (WHERE dir = 'out' AND txn_ts >= a.as_of - INTERVAL '1 day'), 0) AS out_amt_1d,
    COUNT(*)            FILTER (WHERE dir = 'in'  AND txn_ts >= a.as_of - INTERVAL '1 day') AS in_cnt_1d,
    COUNT(DISTINCT ROW(cp_bank, cp_acct))
                        FILTER (WHERE dir = 'in'  AND txn_ts >= a.as_of - INTERVAL '1 day') AS in_cp_1d,
    COALESCE(SUM(amount_usd)
                        FILTER (WHERE dir = 'in'  AND txn_ts >= a.as_of - INTERVAL '1 day'), 0) AS in_amt_1d,

    -- ---------------- 过去 3 天（as_of 前 72 小时）----------------
    COUNT(*)            FILTER (WHERE dir = 'out')                        AS out_cnt_3d,
    COUNT(DISTINCT ROW(cp_bank, cp_acct)) FILTER (WHERE dir = 'out')      AS out_cp_3d,
    COALESCE(SUM(amount_usd) FILTER (WHERE dir = 'out'), 0)               AS out_amt_3d,
    MAX(amount_usd)     FILTER (WHERE dir = 'out')                        AS out_amt_max_3d,
    COUNT(*)            FILTER (WHERE dir = 'in')                         AS in_cnt_3d,
    COUNT(DISTINCT ROW(cp_bank, cp_acct)) FILTER (WHERE dir = 'in')       AS in_cp_3d,
    COALESCE(SUM(amount_usd) FILTER (WHERE dir = 'in'), 0)                AS in_amt_3d,
    MAX(amount_usd)     FILTER (WHERE dir = 'in')                         AS in_amt_max_3d,

    -- 3 天里有交易的天数：连续多天活跃 vs 集中在某一天爆发
    COUNT(DISTINCT txn_ts::date)                                          AS active_days_3d,

    -- 可用历史天数：9/3 往前只有 9/1、9/2 两天，其余都是完整的 3 天。
    -- 这个值本身也作为特征，让模型知道"这条样本的历史不完整"（Q4 追问里的做法）
    LEAST(3, (a.as_of::date - DATE '2022-09-01'))                        AS hist_days
FROM part p
-- 每笔交易复制到它所属的每个快照：as_of 前 3 天内的交易都算进去
JOIN as_of a
  ON p.txn_ts >= a.as_of - INTERVAL '3 days'
 AND p.txn_ts <  a.as_of                  -- 严格小于：快照不包含 as_of 当天的任何交易
GROUP BY p.bank_id, p.account_number, a.as_of;

ALTER TABLE feat_acct_snapshot
    ADD CONSTRAINT pk_feat_acct_snapshot PRIMARY KEY (bank_id, account_number, as_of_date);

ANALYZE feat_acct_snapshot;

COMMENT ON TABLE feat_acct_snapshot IS
    '离线特征快照：每账户每日一行，统计截至当天 00:00 的过去 1 天 / 3 天行为；时点正确由 as_of 严格小于保证';


-- ============================================================================
-- 第 2 步：概览
-- ============================================================================
\echo ''
\echo '=== 每个快照日期覆盖的账户数与典型值 ==='

SELECT
    as_of_date                                                                  AS "快照日",
    hist_days                                                                   AS "可用历史天数",
    COUNT(*)                                                                    AS "账户数",
    PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY out_cp_3d)                      AS "3天转出对手方P50",
    PERCENTILE_DISC(0.99) WITHIN GROUP (ORDER BY out_cp_3d)                     AS "P99",
    MAX(out_cp_3d)                                                              AS "最大"
FROM feat_acct_snapshot
GROUP BY as_of_date, hist_days
ORDER BY as_of_date;

\echo ''
\echo '（9/3 只有 2 天历史，账户数和数值应该明显偏小 —— 这就是预热期效应）'


-- ============================================================================
-- 第 3 步：自检 —— 用最朴素的写法重算 5 行，和快照逐项对比
-- ============================================================================
-- 快照用了"复制到多个日期 + FILTER"的写法，比较绕，容易写错边界。
-- 这里对确定性抽出的 5 行，用直白的子查询重新计算，两边必须完全一致。
-- 这相当于给 SQL 写单元测试：复杂写法的结果，用简单写法来验证。
-- ============================================================================
\echo ''
\echo '=== 自检：抽 5 行用朴素写法重算（"一致" 列必须全是 t）==='

WITH sample AS (
    SELECT *
    FROM feat_acct_snapshot
    WHERE out_cnt_3d BETWEEN 3 AND 50              -- 挑有一定交易量的账户，太稀疏的测不出问题
    ORDER BY md5(bank_id::text || account_number || as_of_date::text)
    LIMIT 5
)
SELECT
    s.bank_id                                                       AS "银行",
    s.account_number                                                AS "账户",
    s.as_of_date                                                    AS "快照日",
    s.out_cnt_3d                                                    AS "快照_3天笔数",
    chk.cnt3                                                        AS "重算_3天笔数",
    s.out_cp_3d                                                     AS "快照_3天对手方",
    chk.cp3                                                         AS "重算_3天对手方",
    s.out_cnt_1d                                                    AS "快照_1天笔数",
    chk.cnt1                                                        AS "重算_1天笔数",
    (s.out_cnt_3d = chk.cnt3 AND s.out_cp_3d = chk.cp3
     AND s.out_cnt_1d = chk.cnt1)                                   AS "一致"
FROM sample s
CROSS JOIN LATERAL (
    SELECT
        COUNT(*)                                                    AS cnt3,
        COUNT(DISTINCT ROW(e.to_bank, e.to_account))                AS cp3,
        COUNT(*) FILTER (WHERE e.txn_ts >= s.as_of_date::timestamp - INTERVAL '1 day') AS cnt1
    FROM txn_norm e
    WHERE e.from_bank    = s.bank_id
      AND e.from_account = s.account_number
      AND e.txn_ts >= s.as_of_date::timestamp - INTERVAL '3 days'
      AND e.txn_ts <  s.as_of_date::timestamp
) chk;

RESET work_mem;
\timing off