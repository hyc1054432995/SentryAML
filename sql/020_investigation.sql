-- ============================================================================
-- SentryAML - 020：告警表与时点正确的调查工具（R9a）
-- ============================================================================
-- 分诊 Agent 调查一条告警时调用的数据工具，全部做成 SQL 函数。三条硬性规则：
--
-- 1. 时点正确：每个函数都带 p_as_of（告警时刻），只看这一刻之前（含）发生的交易。
--    这个参数由 Python 包装层按告警绑定，模型不能填写（与 R8 的调查日期同理，D60）。
--    例外：账户对的"既往来往"只数严格早于告警时刻的交易，因为要回答的是"这笔之前他们来往过吗"。
--
-- 2. 看不到答案：工具只读 inv_txn 这个视图，视图里没有 is_laundering、没有支付方式
--    （payment_format 是数据生成器的痕迹，D07；模型同样不用它）。
--    标签放在 alert_labels，只有评测脚本读。
--    再加一道数据库层面的保证：工具用 sentry_agent 角色连接，这个角色没有读标签表的权限。
--    这样即使函数写错、或者以后有人加了一个工具直接查 txn_norm，也会在数据库层面被拒绝，而不是悄悄泄漏。
--
-- 3. 已确认案件按时间生效：inv_cases 是"银行在测试期开始时已知的洗钱案件"，
--    即训练期和验证期（9/1–9/8）被标注为洗钱的交易所涉及的账户，确认时间统一记为 2022-09-09 00:00。
--    这是 D37 盲区（已知洗钱账户的后续交易，模型召回只有 4%）要用的工具。
--    简化假设：这一时期的洗钱全部已被调查确认。真实银行只知道被调查过的那部分，所以这是"已知案件"的上限。
--
-- 用法：psql ... -f sql/020_investigation.sql   （可重复执行；告警数据由 scripts/alert_queue.py 写入）
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 0. 索引：工具按"账户 + 时间"查询，两个方向各一个
-- ----------------------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'txn_norm'
                   AND indexdef ILIKE '%(from_bank, from_account, txn_ts)%') THEN
        CREATE INDEX txn_norm_from_ts ON txn_norm (from_bank, from_account, txn_ts);
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE tablename = 'txn_norm'
                   AND indexdef ILIKE '%(to_bank, to_account, txn_ts)%') THEN
        CREATE INDEX txn_norm_to_ts ON txn_norm (to_bank, to_account, txn_ts);
    END IF;
END $$;

-- ----------------------------------------------------------------------------
-- 1. 不含标签的交易视图
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW inv_txn AS
SELECT n.txn_id, n.txn_ts, n.from_bank, n.from_account, n.to_bank, n.to_account,
       n.amount_usd, t.is_cross_currency
FROM txn_norm n
JOIN transactions t ON t.txn_id = n.txn_id;
COMMENT ON VIEW inv_txn IS 'R9a：调查工具唯一可读的交易视图，不含 is_laundering 和 payment_format';

-- ----------------------------------------------------------------------------
-- 2. 告警与标签（分开存放）
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    alert_id      INT PRIMARY KEY,                -- 按分数从高到低编号
    txn_id        BIGINT NOT NULL UNIQUE,
    txn_ts        TIMESTAMP NOT NULL,             -- 告警时刻 = 工具的 p_as_of
    from_bank     INT NOT NULL,
    from_account  TEXT NOT NULL,
    to_bank       INT NOT NULL,
    to_account    TEXT NOT NULL,
    amount_usd    NUMERIC NOT NULL,
    score         DOUBLE PRECISION NOT NULL,      -- 模型原始分数（下采样训练，未校准，不对外展示为概率，D31）
    score_pct     DOUBLE PRECISION NOT NULL,      -- 在测试期全部交易中的分位（越接近 100 越可疑）
    tier          TEXT NOT NULL,                  -- T1：前 0.1%；T2：0.1%–0.5%
    shap          JSONB NOT NULL,                 -- 推高 / 拉低分数最多的特征（model_explain.explain_rows）
    model         TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS alert_labels (        -- 只给评测用；sentry_agent 无权读取
    alert_id      INT PRIMARY KEY REFERENCES alerts(alert_id) ON DELETE CASCADE,
    is_laundering SMALLINT NOT NULL,
    pattern_type  TEXT
);
COMMENT ON TABLE alerts IS 'R9a：告警队列（M4b 在测试期分数最高的 0.5%）';
COMMENT ON TABLE alert_labels IS 'R9a：告警的真实标签，仅评测使用';

-- ----------------------------------------------------------------------------
-- 3. 已确认案件（银行在测试期开始时已知）
-- ----------------------------------------------------------------------------
DROP TABLE IF EXISTS inv_cases CASCADE;
CREATE TABLE inv_cases AS
WITH lt AS (
    SELECT n.txn_id, n.txn_ts, n.from_bank, n.from_account, n.to_bank, n.to_account,
           (SELECT MIN(lp.pattern_type) FROM pattern_transactions pt
            JOIN laundering_patterns lp ON lp.pattern_id = pt.pattern_id
            WHERE pt.txn_id = n.txn_id) AS pattern_type
    FROM txn_norm n
    WHERE n.is_laundering = 1 AND n.txn_ts < TIMESTAMP '2022-09-09 00:00'
),
roles AS (
    SELECT from_bank AS bank_id, from_account AS account_number, 'payer' AS role, txn_ts, pattern_type FROM lt
    UNION ALL
    SELECT to_bank, to_account, 'payee', txn_ts, pattern_type FROM lt
)
SELECT bank_id, account_number::text AS account_number, role,
       COUNT(*)::int AS n_txns, MIN(txn_ts) AS first_ts, MAX(txn_ts) AS last_ts,
       ARRAY_REMOVE(ARRAY_AGG(DISTINCT pattern_type), NULL) AS pattern_types,
       TIMESTAMP '2022-09-09 00:00' AS confirmed_at
FROM roles
GROUP BY bank_id, account_number, role;
CREATE INDEX inv_cases_acct ON inv_cases (bank_id, account_number);
COMMENT ON TABLE inv_cases IS 'R9a：测试期开始时已确认的洗钱案件涉及的账户（9/1–9/8 的标注），confirmed_at 之后才可见';

-- ----------------------------------------------------------------------------
-- 4. 调查工具
-- ----------------------------------------------------------------------------

-- 4.1 账户逐日活动：识别"休眠后突然启用"（信号 s_dormant_reactivation）
DROP FUNCTION IF EXISTS inv_account_activity(INT, TEXT, TIMESTAMP, INT);
CREATE FUNCTION inv_account_activity(p_bank INT, p_acct TEXT, p_as_of TIMESTAMP, p_days INT DEFAULT 7)
RETURNS TABLE (day DATE, in_cnt INT, in_cp INT, in_amt NUMERIC, out_cnt INT, out_cp INT, out_amt NUMERIC)
LANGUAGE sql STABLE AS $$
    WITH d AS (
        SELECT generate_series(p_as_of::date - (p_days - 1), p_as_of::date, INTERVAL '1 day')::date AS day
    ),
    i AS (
        SELECT txn_ts::date AS day, COUNT(*)::int AS c, COUNT(DISTINCT (from_bank, from_account))::int AS cp,
               SUM(amount_usd) AS a
        FROM inv_txn
        WHERE to_bank = p_bank AND to_account = p_acct
          AND txn_ts >= p_as_of::date - (p_days - 1) AND txn_ts <= p_as_of
        GROUP BY 1
    ),
    o AS (
        SELECT txn_ts::date AS day, COUNT(*)::int AS c, COUNT(DISTINCT (to_bank, to_account))::int AS cp,
               SUM(amount_usd) AS a
        FROM inv_txn
        WHERE from_bank = p_bank AND from_account = p_acct
          AND txn_ts >= p_as_of::date - (p_days - 1) AND txn_ts <= p_as_of
        GROUP BY 1
    )
    SELECT d.day, COALESCE(i.c, 0), COALESCE(i.cp, 0), ROUND(COALESCE(i.a, 0), 2),
           COALESCE(o.c, 0), COALESCE(o.cp, 0), ROUND(COALESCE(o.a, 0), 2)
    FROM d LEFT JOIN i USING (day) LEFT JOIN o USING (day)
    ORDER BY d.day
$$;

-- 4.2 收付对手方画像：近 N 小时的扇入扇出，并标记清算层账户（D17：历史对手方 > 200）
DROP FUNCTION IF EXISTS inv_fan_profile(INT, TEXT, TIMESTAMP, INT);
CREATE FUNCTION inv_fan_profile(p_bank INT, p_acct TEXT, p_as_of TIMESTAMP, p_hours INT DEFAULT 72)
RETURNS TABLE (window_hours INT, in_cnt INT, in_cp INT, in_amt NUMERIC, out_cnt INT, out_cp INT, out_amt NUMERIC,
               cp_total_before INT, clearing_layer BOOLEAN)
LANGUAGE sql STABLE AS $$
    WITH w AS (
        SELECT * FROM inv_txn
        WHERE ((to_bank = p_bank AND to_account = p_acct) OR (from_bank = p_bank AND from_account = p_acct))
          AND txn_ts > p_as_of - make_interval(hours => p_hours) AND txn_ts <= p_as_of
    ),
    hist AS (                       -- 告警时刻之前全部历史里的不同对手方（两个方向合并）
        SELECT COUNT(*)::int AS n FROM (
            SELECT from_bank, from_account FROM inv_txn
            WHERE to_bank = p_bank AND to_account = p_acct AND txn_ts <= p_as_of
            UNION
            SELECT to_bank, to_account FROM inv_txn
            WHERE from_bank = p_bank AND from_account = p_acct AND txn_ts <= p_as_of
        ) x
    )
    SELECT p_hours,
           COUNT(*) FILTER (WHERE to_bank = p_bank AND to_account = p_acct)::int,
           COUNT(DISTINCT (from_bank, from_account)) FILTER (WHERE to_bank = p_bank AND to_account = p_acct)::int,
           ROUND(COALESCE(SUM(amount_usd) FILTER (WHERE to_bank = p_bank AND to_account = p_acct), 0), 2),
           COUNT(*) FILTER (WHERE from_bank = p_bank AND from_account = p_acct)::int,
           COUNT(DISTINCT (to_bank, to_account)) FILTER (WHERE from_bank = p_bank AND from_account = p_acct)::int,
           ROUND(COALESCE(SUM(amount_usd) FILTER (WHERE from_bank = p_bank AND from_account = p_acct), 0), 2),
           (SELECT n FROM hist), (SELECT n FROM hist) > 200
    FROM w
$$;

-- 4.3 两个账户之间的既往来往（严格早于告警时刻），两个方向各一行
DROP FUNCTION IF EXISTS inv_counterparty_history(INT, TEXT, INT, TEXT, TIMESTAMP);
CREATE FUNCTION inv_counterparty_history(a_bank INT, a_acct TEXT, b_bank INT, b_acct TEXT, p_as_of TIMESTAMP)
RETURNS TABLE (direction TEXT, n INT, amt NUMERIC, first_ts TIMESTAMP, last_ts TIMESTAMP)
LANGUAGE sql STABLE AS $$
    SELECT 'a_to_b', COUNT(*)::int, ROUND(COALESCE(SUM(amount_usd), 0), 2), MIN(txn_ts), MAX(txn_ts)
    FROM inv_txn WHERE from_bank = a_bank AND from_account = a_acct AND to_bank = b_bank AND to_account = b_acct
                   AND txn_ts < p_as_of
    UNION ALL
    SELECT 'b_to_a', COUNT(*)::int, ROUND(COALESCE(SUM(amount_usd), 0), 2), MIN(txn_ts), MAX(txn_ts)
    FROM inv_txn WHERE from_bank = b_bank AND from_account = b_acct AND to_bank = a_bank AND to_account = a_acct
                   AND txn_ts < p_as_of
$$;

-- 4.4 过桥检查：收到一笔钱后 N 小时内，是否转出了金额相近（±p_tol）的一笔。逐笔匹配，不是累计金额
--     （MAS TM 指引 Case B：按累计金额的规则会漏掉 1:1 过桥）
DROP FUNCTION IF EXISTS inv_pass_through(INT, TEXT, TIMESTAMP, INT, NUMERIC);
CREATE FUNCTION inv_pass_through(p_bank INT, p_acct TEXT, p_as_of TIMESTAMP, p_hours INT DEFAULT 72,
                                 p_tol NUMERIC DEFAULT 0.10)
RETURNS TABLE (in_txn BIGINT, in_ts TIMESTAMP, in_from TEXT, in_amt NUMERIC,
               out_txn BIGINT, out_ts TIMESTAMP, out_to TEXT, out_amt NUMERIC, lag_hours NUMERIC, diff_pct NUMERIC)
LANGUAGE sql STABLE AS $$
    SELECT DISTINCT ON (i.txn_id)
           i.txn_id, i.txn_ts, i.from_bank || ':' || i.from_account, ROUND(i.amount_usd, 2),
           o.txn_id, o.txn_ts, o.to_bank || ':' || o.to_account, ROUND(o.amount_usd, 2),
           ROUND(EXTRACT(EPOCH FROM (o.txn_ts - i.txn_ts)) / 3600.0, 1),
           ROUND(100 * ABS(o.amount_usd - i.amount_usd) / NULLIF(i.amount_usd, 0), 1)
    FROM inv_txn i
    JOIN inv_txn o
      ON o.from_bank = p_bank AND o.from_account = p_acct
     AND o.txn_ts >= i.txn_ts AND o.txn_ts <= i.txn_ts + make_interval(hours => p_hours)
     AND o.txn_ts <= p_as_of
     AND ABS(o.amount_usd - i.amount_usd) <= p_tol * i.amount_usd
     AND NOT (o.to_bank = i.from_bank AND o.to_account = i.from_account)     -- 原路退回不算过桥
    WHERE i.to_bank = p_bank AND i.to_account = p_acct
      AND i.txn_ts > p_as_of - make_interval(hours => p_hours) AND i.txn_ts <= p_as_of
    ORDER BY i.txn_id, ABS(o.amount_usd - i.amount_usd), o.txn_ts
    LIMIT 20
$$;

-- 4.5 资金回流追踪：从账户出发、时间递增、7 天窗口、最多 12 跳回到自己（009 的方法，D19）
--     只用告警时刻之前的交易；清算层账户（account_degrees.total_cp_count > 200）剪枝。
--     注意：account_degrees 是全期统计，清算层是稳定的结构属性（D17 的 15 个账户都在银行 70），这里接受这个近似
DROP FUNCTION IF EXISTS inv_trace_cycle(INT, TEXT, TIMESTAMP, INT, INT, INT);
CREATE FUNCTION inv_trace_cycle(p_bank INT, p_acct TEXT, p_as_of TIMESTAMP, p_max_hops INT DEFAULT 12,
                                p_window_days INT DEFAULT 7, p_limit INT DEFAULT 5)
RETURNS TABLE (hops INT, path TEXT[], txn_path BIGINT[], start_ts TIMESTAMP, end_ts TIMESTAMP, span_hours NUMERIC)
LANGUAGE sql STABLE AS $$
    WITH RECURSIVE walk AS (
        (SELECT DISTINCT ON (e.to_bank, e.to_account, e.txn_ts)
                e.to_bank AS cur_bank, e.to_account AS cur_acct, 1 AS hop,
                e.txn_ts AS st, e.txn_ts AS cur_ts,
                ARRAY[p_bank::text || ':' || p_acct, e.to_bank::text || ':' || e.to_account] AS path,
                ARRAY[e.txn_id] AS txn_path, FALSE AS closed
         FROM inv_txn e
         LEFT JOIN account_degrees dg ON dg.bank_id = e.to_bank AND dg.account_number = e.to_account
         WHERE e.from_bank = p_bank AND e.from_account = p_acct
           AND e.txn_ts > p_as_of - make_interval(days => p_window_days) AND e.txn_ts <= p_as_of
           AND COALESCE(dg.total_cp_count, 0) <= 200
         ORDER BY e.to_bank, e.to_account, e.txn_ts, e.txn_id)
        UNION ALL
        SELECT nx.to_bank, nx.to_account, w.hop + 1, w.st, nx.txn_ts,
               w.path || (nx.to_bank::text || ':' || nx.to_account), w.txn_path || nx.txn_id,
               (nx.to_bank = p_bank AND nx.to_account = p_acct)
        FROM walk w
        CROSS JOIN LATERAL (
            SELECT DISTINCT ON (e.to_bank, e.to_account) e.to_bank, e.to_account, e.txn_ts, e.txn_id
            FROM inv_txn e
            LEFT JOIN account_degrees dg ON dg.bank_id = e.to_bank AND dg.account_number = e.to_account
            WHERE e.from_bank = w.cur_bank AND e.from_account = w.cur_acct
              AND e.txn_ts >= w.cur_ts
              AND e.txn_ts <= w.st + make_interval(days => p_window_days)
              AND e.txn_ts <= p_as_of
              AND COALESCE(dg.total_cp_count, 0) <= 200
            ORDER BY e.to_bank, e.to_account, e.txn_ts, e.txn_id
        ) nx
        WHERE w.hop < p_max_hops AND NOT w.closed
          AND ((nx.to_bank = p_bank AND nx.to_account = p_acct)
               OR NOT ((nx.to_bank::text || ':' || nx.to_account) = ANY (w.path)))
    )
    SELECT hop, path, txn_path, st, cur_ts, ROUND(EXTRACT(EPOCH FROM (cur_ts - st)) / 3600.0, 1)
    FROM walk WHERE closed
    ORDER BY cur_ts DESC, hop
    LIMIT p_limit
$$;

-- 4.6 两跳结构：
--   gather_out  从本账户分散出去的钱，经不同中间账户后汇到同一个终点（scatter-gather 的后半段）
--   gather_in   同一个源头的钱，经不同中间账户后汇入本账户
--   只列至少经过 2 个不同中间账户的终点 / 源头；中间账户排除清算层
DROP FUNCTION IF EXISTS inv_two_hop(INT, TEXT, TIMESTAMP, INT);
CREATE FUNCTION inv_two_hop(p_bank INT, p_acct TEXT, p_as_of TIMESTAMP, p_hours INT DEFAULT 72)
RETURNS TABLE (direction TEXT, endpoint TEXT, n_intermediaries INT, intermediaries TEXT[], amt_second_hop NUMERIC)
LANGUAGE sql STABLE AS $$
    WITH first_out AS (
        SELECT e.to_bank AS x_bank, e.to_account AS x_acct, MIN(e.txn_ts) AS t1
        FROM inv_txn e
        LEFT JOIN account_degrees dg ON dg.bank_id = e.to_bank AND dg.account_number = e.to_account
        WHERE e.from_bank = p_bank AND e.from_account = p_acct
          AND e.txn_ts > p_as_of - make_interval(hours => p_hours) AND e.txn_ts <= p_as_of
          AND COALESCE(dg.total_cp_count, 0) <= 200
        GROUP BY 1, 2
    ),
    gout AS (
        SELECT 'gather_out'::text AS direction, s.to_bank || ':' || s.to_account AS endpoint,
               COUNT(DISTINCT f.x_bank || ':' || f.x_acct)::int AS n_x,
               (ARRAY_AGG(DISTINCT f.x_bank || ':' || f.x_acct))[1:10] AS xs,
               ROUND(SUM(s.amount_usd), 2) AS amt
        FROM first_out f
        JOIN inv_txn s ON s.from_bank = f.x_bank AND s.from_account = f.x_acct
                      AND s.txn_ts >= f.t1 AND s.txn_ts <= p_as_of
                      AND NOT (s.to_bank = p_bank AND s.to_account = p_acct)
        GROUP BY 1, 2
        HAVING COUNT(DISTINCT f.x_bank || ':' || f.x_acct) >= 2
        ORDER BY 3 DESC, 5 DESC LIMIT 5
    ),
    last_in AS (
        SELECT e.from_bank AS x_bank, e.from_account AS x_acct, MAX(e.txn_ts) AS t2
        FROM inv_txn e
        LEFT JOIN account_degrees dg ON dg.bank_id = e.from_bank AND dg.account_number = e.from_account
        WHERE e.to_bank = p_bank AND e.to_account = p_acct
          AND e.txn_ts > p_as_of - make_interval(hours => p_hours) AND e.txn_ts <= p_as_of
          AND COALESCE(dg.total_cp_count, 0) <= 200
        GROUP BY 1, 2
    ),
    gin AS (
        SELECT 'gather_in'::text, s.from_bank || ':' || s.from_account,
               COUNT(DISTINCT l.x_bank || ':' || l.x_acct)::int,
               (ARRAY_AGG(DISTINCT l.x_bank || ':' || l.x_acct))[1:10],
               ROUND(SUM(s.amount_usd), 2)
        FROM last_in l
        JOIN inv_txn s ON s.to_bank = l.x_bank AND s.to_account = l.x_acct
                      AND s.txn_ts <= l.t2 AND s.txn_ts > l.t2 - make_interval(hours => p_hours)
                      AND NOT (s.from_bank = p_bank AND s.from_account = p_acct)
        GROUP BY 1, 2
        HAVING COUNT(DISTINCT l.x_bank || ':' || l.x_acct) >= 2
        ORDER BY 3 DESC, 5 DESC LIMIT 5
    )
    SELECT * FROM gout UNION ALL SELECT * FROM gin
$$;

-- 4.7 实体画像：账户所属实体名下的账户和银行（开户登记信息，不随时间变化）
DROP FUNCTION IF EXISTS inv_entity_profile(INT, TEXT);
CREATE FUNCTION inv_entity_profile(p_bank INT, p_acct TEXT)
RETURNS TABLE (entity_id TEXT, entity_type TEXT, n_accounts INT, n_banks INT, other_accounts TEXT[])
LANGUAGE sql STABLE AS $$
    SELECT a.entity_id::text, e.entity_type::text,
           (SELECT COUNT(*)::int FROM accounts x WHERE x.entity_id = a.entity_id),
           (SELECT COUNT(DISTINCT x.bank_id)::int FROM accounts x WHERE x.entity_id = a.entity_id),
           (SELECT (ARRAY_AGG(x.bank_id || ':' || x.account_number ORDER BY x.bank_id, x.account_number))[1:10]
            FROM accounts x WHERE x.entity_id = a.entity_id
              AND NOT (x.bank_id = p_bank AND x.account_number = p_acct))
    FROM accounts a JOIN entities e ON e.entity_id = a.entity_id
    WHERE a.bank_id = p_bank AND a.account_number = p_acct
$$;

-- 4.8 案件历史：告警时刻之前已确认的案件
DROP FUNCTION IF EXISTS inv_case_history(INT, TEXT, TIMESTAMP);
CREATE FUNCTION inv_case_history(p_bank INT, p_acct TEXT, p_as_of TIMESTAMP)
RETURNS TABLE (role TEXT, n_txns INT, first_ts TIMESTAMP, last_ts TIMESTAMP, pattern_types TEXT[], confirmed_at TIMESTAMP)
LANGUAGE sql STABLE AS $$
    SELECT role, n_txns, first_ts, last_ts, pattern_types, confirmed_at
    FROM inv_cases
    WHERE bank_id = p_bank AND account_number = p_acct AND confirmed_at <= p_as_of
    ORDER BY role
$$;

-- 4.9 金额画像：告警这一笔的金额是否落在洗钱集中区间（D13：训练期洗钱金额 P25–P95 = 2,154–24,075 美元）
DROP FUNCTION IF EXISTS inv_amount_profile(BIGINT);
CREATE FUNCTION inv_amount_profile(p_txn BIGINT)
RETURNS TABLE (amount_usd NUMERIC, in_laundering_band BOOLEAN, band_note TEXT, cross_currency BOOLEAN)
LANGUAGE sql STABLE AS $$
    SELECT ROUND(amount_usd, 2), amount_usd BETWEEN 2154 AND 24075,
           'training-period laundering amounts P25-P95: 2,154-24,075 USD', is_cross_currency
    FROM inv_txn WHERE txn_id = p_txn
$$;

-- ----------------------------------------------------------------------------
-- 5. 最小权限角色：调查工具用它连接，读不到标签
-- ----------------------------------------------------------------------------
-- 密码只是本地开发用的默认值；在 .env 的 AGENT_DATABASE_URL 里使用，生产环境必须更换
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sentry_agent') THEN
        CREATE ROLE sentry_agent LOGIN PASSWORD 'sentry_agent_dev';
    END IF;
END $$;
DO $$ BEGIN EXECUTE format('GRANT CONNECT ON DATABASE %I TO sentry_agent', current_database()); END $$;
GRANT USAGE ON SCHEMA public TO sentry_agent;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM sentry_agent;
-- 只开放这几张：交易视图（视图以所有者权限读取底层表，所以角色本身不需要 txn_norm 的权限）、
-- 账户与实体、清算层统计、已确认案件、告警（不含标签）
GRANT SELECT ON inv_txn, accounts, entities, account_degrees, inv_cases, alerts TO sentry_agent;