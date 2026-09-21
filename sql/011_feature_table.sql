-- ============================================================================
-- SentryAML - 011: 拼装样本特征表（M2 第二步）
-- ============================================================================
-- 执行（依赖 010 生成的 feat_acct_snapshot）:
--   docker compose exec -T postgres psql -U sentry -d sentryaml -f /sql/011_feature_table.sql
--
-- 每个样本 = 9/3 ~ 9/10 的一笔非自环交易（约 314 万行），标签 = is_laundering。
--
-- 特征分五类：
--   ① 交易本身      金额、是否跨币种、是否同分行、支付方式（泄漏审计用）
--   ② 付款方行为    离线快照（截至当天 00:00 的 1 天 / 3 天）+ 当日实时（今天在这笔之前）
--   ③ 收款方行为    同上
--   ④ 账户对关系    这对账户之前来往过几次；资金是否曾经反方向流过（两跳回流）
--   ⑤ 实体资料      名下账户数、跨几家银行、实体类型、收付双方是否同一实体
--
-- 【时点正确的三道保证】
--   1. 离线快照：as_of = 交易当天 00:00，快照只含 as_of 之前的交易（010 已自检）
--   2. 当日实时：窗口右端是 INTERVAL '1 second' PRECEDING，
--      时间戳精确到分钟，所以同一分钟的交易（包括自己）一律不算
--   3. 账户对关系：同样只数严格早于当前分钟的交易
--   文件末尾对 5 行做朴素重算自检。
-- ============================================================================

\timing on
\pset pager off
SET work_mem = '256MB';


DROP TABLE IF EXISTS feat_txn CASCADE;

CREATE TABLE feat_txn AS
WITH
-- ----------------------------------------------------------------------------
-- base：9/11 之前的全部非自环交易
-- 9/1 ~ 9/2 的交易不做样本，但要参与"账户对关系"的历史计数
-- ----------------------------------------------------------------------------
base AS (
    SELECT
        n.txn_id,
        n.txn_ts,
        n.txn_ts::date                         AS d,
        n.from_bank, n.from_account,
        n.to_bank,   n.to_account,
        n.amount_usd::float8                   AS amount_usd,
        n.payment_format,
        n.is_laundering,
        t.is_cross_currency,
        t.is_intra_branch,
        -- 账户对的"规范方向"：把 (A,B) 和 (B,A) 归到同一组，用来数反方向的历史。
        -- ROW(...) < ROW(...) 是按列依次比较的"字典序"比较
        (ROW(n.from_bank, n.from_account) < ROW(n.to_bank, n.to_account)) AS fwd
    FROM txn_norm n
    JOIN transactions t ON t.txn_id = n.txn_id
    WHERE n.txn_ts < '2022-09-11'
),

-- ----------------------------------------------------------------------------
-- flags："今天第一次转给这个人" / "今天第一次收到这个人的钱" 标记
-- ----------------------------------------------------------------------------
-- 这是"当日去重计数"的技巧：
--   窗口函数不支持 COUNT(DISTINCT)，但"今天"的窗口起点固定在 00:00，
--   所以可以先给每行打标记：这对账户今天第一次出现 = 1，否则 = 0，
--   再对标记求累计和 —— 得到的就是"截至目前，今天出现过的不同对手方数"。
--   （"过去 3 天"是滑动窗口，起点随每笔交易移动，这个技巧就不成立了）
-- ----------------------------------------------------------------------------
flags AS (
    SELECT
        b.*,
        (ROW_NUMBER() OVER (
            PARTITION BY from_bank, from_account, d, to_bank, to_account
            ORDER BY txn_ts, txn_id) = 1)::int                     AS first_out_today,
        (ROW_NUMBER() OVER (
            PARTITION BY to_bank, to_account, d, from_bank, from_account
            ORDER BY txn_ts, txn_id) = 1)::int                     AS first_in_today,
        -- 规范化后的账户对（较小的一方放前面）
        CASE WHEN fwd THEN from_bank    ELSE to_bank    END        AS ka_bank,
        CASE WHEN fwd THEN from_account ELSE to_account END        AS ka_acct,
        CASE WHEN fwd THEN to_bank      ELSE from_bank  END        AS kb_bank,
        CASE WHEN fwd THEN to_account   ELSE from_account END      AS kb_acct
    FROM base b
),

-- ----------------------------------------------------------------------------
-- win：当日实时特征 + 账户对历史
-- ----------------------------------------------------------------------------
-- 窗口右端 INTERVAL '1 second' PRECEDING：
--   时间戳只精确到分钟，同一分钟的交易先后无法区分，
--   所以一律排除（包括当前这笔自己），只统计严格更早的分钟。
-- 窗口为空时 COUNT 返回 0，SUM 返回 NULL，用 COALESCE 补 0。
-- ----------------------------------------------------------------------------
win AS (
    SELECT
        f.*,
        -- 付款方：今天在这笔之前的转出
        COUNT(*)                          OVER w_out_today                  AS p_today_out_cnt,
        COALESCE(SUM(amount_usd)          OVER w_out_today, 0)              AS p_today_out_amt,
        COALESCE(SUM(first_out_today)     OVER w_out_today, 0)              AS p_today_out_cp,
        -- 收款方：今天在这笔之前的收入
        COUNT(*)                          OVER w_in_today                   AS r_today_in_cnt,
        COALESCE(SUM(amount_usd)          OVER w_in_today, 0)               AS r_today_in_amt,
        COALESCE(SUM(first_in_today)      OVER w_in_today, 0)               AS r_today_in_cp,
        -- 同方向：付款方之前给这个收款方转过几次（自 9/1 起）
        COUNT(*)                          OVER w_pair                       AS pair_prev_cnt,
        -- 规范账户对上，两个方向各自的历史笔数
        COALESCE(SUM(CASE WHEN fwd THEN 1 ELSE 0 END) OVER w_canon, 0)      AS canon_fwd_prev,
        COALESCE(SUM(CASE WHEN fwd THEN 0 ELSE 1 END) OVER w_canon, 0)      AS canon_bwd_prev
    FROM flags f
    WINDOW
        w_out_today AS (PARTITION BY from_bank, from_account, d
                        ORDER BY txn_ts
                        RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '1 second' PRECEDING),
        w_in_today  AS (PARTITION BY to_bank, to_account, d
                        ORDER BY txn_ts
                        RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '1 second' PRECEDING),
        w_pair      AS (PARTITION BY from_bank, from_account, to_bank, to_account
                        ORDER BY txn_ts
                        RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '1 second' PRECEDING),
        w_canon     AS (PARTITION BY ka_bank, ka_acct, kb_bank, kb_acct
                        ORDER BY txn_ts
                        RANGE BETWEEN UNBOUNDED PRECEDING AND INTERVAL '1 second' PRECEDING)
),

-- ----------------------------------------------------------------------------
-- ent：实体资料（来自开户登记，不随交易变化，不存在时点问题）
-- ----------------------------------------------------------------------------
ent AS (
    SELECT a.bank_id, a.account_number, a.entity_id, e.entity_type,
           s.n_accounts, s.n_banks
    FROM accounts a
    JOIN entities e ON e.entity_id = a.entity_id
    JOIN (SELECT entity_id,
                 COUNT(*)                AS n_accounts,
                 COUNT(DISTINCT bank_id) AS n_banks
          FROM accounts GROUP BY entity_id) s
      ON s.entity_id = a.entity_id
),

-- 模式类型：只用于事后分析，【不作为特征】
ptype AS (
    SELECT pt.txn_id, MIN(lp.pattern_type) AS pattern_type
    FROM pattern_transactions pt
    JOIN laundering_patterns lp ON lp.pattern_id = pt.pattern_id
    WHERE pt.txn_id IS NOT NULL
    GROUP BY pt.txn_id
)

SELECT
    -- ======================== 元数据（不是特征）========================
    w.txn_id,
    w.txn_ts,
    CASE
        WHEN w.d BETWEEN DATE '2022-09-03' AND DATE '2022-09-07' THEN 'train'
        WHEN w.d =       DATE '2022-09-08'                        THEN 'val'
        ELSE                                                           'test'
    END                                                   AS split,
    w.from_bank, w.from_account, w.to_bank, w.to_account,  -- 评测时按账户分组用
    pt.pattern_type,                                        -- 事后分析用
    w.is_laundering                                       AS label,

    -- ======================== ① 交易本身 ========================
    LN(1 + w.amount_usd)                                  AS f_log_amount,
    w.is_cross_currency::int                              AS f_cross_currency,
    w.is_intra_branch::int                                AS f_intra_branch,
    w.payment_format                                      AS f_payment_format,  -- 已知泄漏，M6 审计

    -- ======================== ② 付款方：离线快照 ========================
    COALESCE(sp.out_cnt_1d, 0)      AS p_out_cnt_1d,
    COALESCE(sp.out_cp_1d, 0)       AS p_out_cp_1d,
    COALESCE(sp.out_amt_1d, 0)::float8 AS p_out_amt_1d,
    COALESCE(sp.in_cnt_1d, 0)       AS p_in_cnt_1d,
    COALESCE(sp.in_cp_1d, 0)        AS p_in_cp_1d,
    COALESCE(sp.in_amt_1d, 0)::float8  AS p_in_amt_1d,
    COALESCE(sp.out_cnt_3d, 0)      AS p_out_cnt_3d,
    COALESCE(sp.out_cp_3d, 0)       AS p_out_cp_3d,
    COALESCE(sp.out_amt_3d, 0)::float8 AS p_out_amt_3d,
    COALESCE(sp.out_amt_max_3d, 0)::float8 AS p_out_amt_max_3d,
    COALESCE(sp.in_cnt_3d, 0)       AS p_in_cnt_3d,
    COALESCE(sp.in_cp_3d, 0)        AS p_in_cp_3d,
    COALESCE(sp.in_amt_3d, 0)::float8  AS p_in_amt_3d,
    COALESCE(sp.in_amt_max_3d, 0)::float8 AS p_in_amt_max_3d,
    COALESCE(sp.active_days_3d, 0)  AS p_active_days_3d,

    -- ======================== ② 付款方：当日实时 ========================
    w.p_today_out_cnt,
    w.p_today_out_amt,
    w.p_today_out_cp,

    -- ======================== ③ 收款方：离线快照 ========================
    COALESCE(sr.out_cnt_1d, 0)      AS r_out_cnt_1d,
    COALESCE(sr.out_cp_1d, 0)       AS r_out_cp_1d,
    COALESCE(sr.out_amt_1d, 0)::float8 AS r_out_amt_1d,
    COALESCE(sr.in_cnt_1d, 0)       AS r_in_cnt_1d,
    COALESCE(sr.in_cp_1d, 0)        AS r_in_cp_1d,
    COALESCE(sr.in_amt_1d, 0)::float8  AS r_in_amt_1d,
    COALESCE(sr.out_cnt_3d, 0)      AS r_out_cnt_3d,
    COALESCE(sr.out_cp_3d, 0)       AS r_out_cp_3d,
    COALESCE(sr.out_amt_3d, 0)::float8 AS r_out_amt_3d,
    COALESCE(sr.out_amt_max_3d, 0)::float8 AS r_out_amt_max_3d,
    COALESCE(sr.in_cnt_3d, 0)       AS r_in_cnt_3d,
    COALESCE(sr.in_cp_3d, 0)        AS r_in_cp_3d,
    COALESCE(sr.in_amt_3d, 0)::float8  AS r_in_amt_3d,
    COALESCE(sr.in_amt_max_3d, 0)::float8 AS r_in_amt_max_3d,
    COALESCE(sr.active_days_3d, 0)  AS r_active_days_3d,

    -- ======================== ③ 收款方：当日实时 ========================
    w.r_today_in_cnt,
    w.r_today_in_amt,
    w.r_today_in_cp,

    -- 可用历史天数（9/3 为 2，其余为 3）
    LEAST(3, w.d - DATE '2022-09-01')                     AS f_hist_days,

    -- ======================== ④ 账户对关系 ========================
    w.pair_prev_cnt                                       AS f_pair_prev_cnt,
    -- 反方向：收款方之前给付款方转过几次（资金是否曾经回流）
    CASE WHEN w.fwd THEN w.canon_bwd_prev ELSE w.canon_fwd_prev END
                                                          AS f_pair_rev_prev_cnt,

    -- ======================== ⑤ 实体资料 ========================
    pe.entity_type                                        AS p_entity_type,
    pe.n_accounts                                         AS p_ent_n_accounts,
    pe.n_banks                                            AS p_ent_n_banks,
    re.entity_type                                        AS r_entity_type,
    re.n_accounts                                         AS r_ent_n_accounts,
    re.n_banks                                            AS r_ent_n_banks,
    -- 收付双方属于同一个实体：在自己名下不同账户之间倒腾资金
    (pe.entity_id = re.entity_id)::int                    AS f_same_entity

FROM win w
LEFT JOIN feat_acct_snapshot sp
       ON sp.bank_id = w.from_bank AND sp.account_number = w.from_account AND sp.as_of_date = w.d
LEFT JOIN feat_acct_snapshot sr
       ON sr.bank_id = w.to_bank   AND sr.account_number = w.to_account   AND sr.as_of_date = w.d
LEFT JOIN ent pe ON pe.bank_id = w.from_bank AND pe.account_number = w.from_account
LEFT JOIN ent re ON re.bank_id = w.to_bank   AND re.account_number = w.to_account
LEFT JOIN ptype pt ON pt.txn_id = w.txn_id
WHERE w.d BETWEEN DATE '2022-09-03' AND DATE '2022-09-10';   -- 只保留样本期

ALTER TABLE feat_txn ADD CONSTRAINT pk_feat_txn PRIMARY KEY (txn_id);
CREATE INDEX idx_feat_txn_split ON feat_txn (split);
ANALYZE feat_txn;

COMMENT ON TABLE feat_txn IS
    '样本特征表：9/3~9/10 每笔非自环交易一行；f_/p_/r_ 前缀为特征，其余为元数据';


-- ============================================================================
-- 检查 1：各切分的样本量
-- ============================================================================
\echo ''
\echo '=== 各切分的样本量（预期：共约 314 万行，洗钱 3,785 笔）==='

SELECT
    split                                                        AS "切分",
    COUNT(*)                                                     AS "样本数",
    SUM(label)                                                   AS "洗钱",
    ROUND(10000.0 * SUM(label) / COUNT(*), 2)                    AS "洗钱/万笔",
    MIN(txn_ts)::date                                            AS "起",
    MAX(txn_ts)::date                                            AS "止"
FROM feat_txn
GROUP BY split
ORDER BY MIN(txn_ts);


-- ============================================================================
-- 检查 2：时点正确性自检 —— 抽 5 行，用最直白的子查询重算
-- ============================================================================
\echo ''
\echo '=== 自检：当日实时特征与账户对特征（"一致" 列必须全是 t）==='

WITH sample AS (
    SELECT *
    FROM feat_txn
    WHERE p_today_out_cnt BETWEEN 2 AND 50       -- 挑有当日历史的样本
    ORDER BY md5(txn_id::text)
    LIMIT 5
)
SELECT
    s.txn_id                                              AS "txn_id",
    s.p_today_out_cnt  || ' / ' || c.today_cnt            AS "今日转出笔数 表/重算",
    s.p_today_out_cp   || ' / ' || c.today_cp             AS "今日对手方 表/重算",
    s.r_today_in_cp    || ' / ' || c.today_in_cp          AS "收款方今日来源 表/重算",
    s.f_pair_prev_cnt  || ' / ' || c.pair_cnt             AS "同向历史 表/重算",
    s.f_pair_rev_prev_cnt || ' / ' || c.rev_cnt           AS "反向历史 表/重算",
    (s.p_today_out_cnt = c.today_cnt AND s.p_today_out_cp = c.today_cp
     AND s.r_today_in_cp = c.today_in_cp
     AND s.f_pair_prev_cnt = c.pair_cnt AND s.f_pair_rev_prev_cnt = c.rev_cnt) AS "一致"
FROM sample s
CROSS JOIN LATERAL (
    SELECT
        -- 今天 00:00 到这笔所在分钟之前，付款方转出几笔、几个不同对手方
        (SELECT COUNT(*) FROM txn_norm e
          WHERE e.from_bank = s.from_bank AND e.from_account = s.from_account
            AND e.txn_ts >= s.txn_ts::date AND e.txn_ts < s.txn_ts)          AS today_cnt,
        (SELECT COUNT(DISTINCT ROW(e.to_bank, e.to_account)) FROM txn_norm e
          WHERE e.from_bank = s.from_bank AND e.from_account = s.from_account
            AND e.txn_ts >= s.txn_ts::date AND e.txn_ts < s.txn_ts)          AS today_cp,
        -- 收款方今天之前收到过几个不同来源
        (SELECT COUNT(DISTINCT ROW(e.from_bank, e.from_account)) FROM txn_norm e
          WHERE e.to_bank = s.to_bank AND e.to_account = s.to_account
            AND e.txn_ts >= s.txn_ts::date AND e.txn_ts < s.txn_ts)          AS today_in_cp,
        -- 同方向历史（自数据开始）
        (SELECT COUNT(*) FROM txn_norm e
          WHERE e.from_bank = s.from_bank AND e.from_account = s.from_account
            AND e.to_bank = s.to_bank AND e.to_account = s.to_account
            AND e.txn_ts < s.txn_ts)                                         AS pair_cnt,
        -- 反方向历史
        (SELECT COUNT(*) FROM txn_norm e
          WHERE e.from_bank = s.to_bank AND e.from_account = s.to_account
            AND e.to_bank = s.from_bank AND e.to_account = s.from_account
            AND e.txn_ts < s.txn_ts)                                         AS rev_cnt
) c;


-- ============================================================================
-- 检查 3：特征预览 —— 训练集里洗钱与正常样本的对比
-- ============================================================================
-- 只在训练集上看（不偷看验证集和测试集）。
-- 目的：在建模之前，粗略判断哪些特征有区分度。
-- ============================================================================
\echo ''
\echo '=== 训练集特征预览：洗钱 vs 正常 ==='

SELECT
    CASE WHEN label = 1 THEN '洗钱' ELSE '正常' END                            AS "类别",
    COUNT(*)                                                                     AS "样本",
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY f_log_amount)::numeric, 2) AS "log金额P50",
    PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY r_in_cp_3d)                      AS "收款方3天来源数P50",
    PERCENTILE_DISC(0.9) WITHIN GROUP (ORDER BY r_in_cp_3d)                      AS "P90",
    ROUND(100.0 * AVG((f_pair_prev_cnt = 0)::int), 1)                            AS "首次来往%",
    ROUND(100.0 * AVG((f_pair_rev_prev_cnt > 0)::int), 2)                        AS "曾反向流动%",
    ROUND(100.0 * AVG(f_same_entity), 2)                                         AS "同一实体%",
    ROUND(100.0 * AVG(f_cross_currency), 2)                                      AS "跨币种%"
FROM feat_txn
WHERE split = 'train'
GROUP BY label
ORDER BY label DESC;

\echo ''
\echo '=== 表体积 ==='
SELECT pg_size_pretty(pg_total_relation_size('feat_txn')) AS "feat_txn 总体积";

RESET work_mem;
\timing off