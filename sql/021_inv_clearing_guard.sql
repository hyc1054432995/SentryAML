-- ============================================================================
-- SentryAML - 021：路径类调查工具对清算层账户明确拒绝（R9a 修订，D69）
-- ============================================================================
-- 起因：冒烟测试的延迟部分出现超时。诊断（scripts/inv_latency_diag.py）显示，
-- 慢调用和超时全部出在清算层账户（account_degrees.total_cp_count > 200，D17）上；
-- 随机对照的普通账户每个工具 1–9 毫秒。全部 4,226 条告警里有 271 条的付款方是清算层账户。
--
-- 为什么不是"加索引 / 调大超时 / 加 LIMIT"，而是对清算层直接拒绝：
--   这三个工具（过桥、回流、两跳汇聚）在清算层账户上没有业务意义。
--   D17：清算层 15 个账户承载全网 5% 的交易，洗钱浓度与全网持平，是基础设施不是风险节点；
--   一个 72 小时内付出几万笔、收进几百笔的账户，随便就能配出金额相近的"过桥"、找到"回流"——都是噪音。
--   变快了也只是更快地产出噪音，所以修法是不做，而不是做得更快。
--
-- 为什么是报错，而不是返回 0 行：
--   0 行对模型的意思是"查过了，没有发现过桥 / 回流"，这会被当成排除嫌疑的证据——而事实是"没查"。
--   报错（RAISE EXCEPTION，消息以 clearing_layer 开头）由 R9b 的工具包装层转成一条明确的观察：
--   "该账户是清算层账户，不做路径分析（D17）；请改查对手方"。这与 R8a 的约定一致：预期内的情况作为观察返回，
--   但要让模型看得出"没做"和"做了没发现"的区别。
--
-- 改动：
--   1. 对手方画像的 clearing_layer 与拒绝用同一个定义（见 4.2 的注释；在测试库上发现两个定义会给出矛盾的观察）。
--   2. 三个路径函数从 LANGUAGE sql 改成 plpgsql（SQL 函数里不能先判断再报错），查询本身与 020 完全相同。
--   #variable_conflict use_column：plpgsql 会把 RETURNS TABLE 的列名（如 path、direction）变成变量，
--   与查询里同名的列冲突时报"ambiguous"；这条指令让同名时按列解析。
-- 其余工具不变：活动、对手方画像（返回 clearing_layer = true）、既往来往、实体、案件、金额
--   都是描述性事实，在清算层账户上也有意义，而且都在 1.2 秒以内。
--
-- 用法：psql ... -f sql/021_inv_clearing_guard.sql   （可重复执行；需先执行 020）
-- ============================================================================

-- 判断是否清算层：与 020 里回流 / 两跳剪枝用的是同一个定义（全期统计，D68 记录了这个近似）
CREATE OR REPLACE FUNCTION inv_is_clearing(p_bank INT, p_acct TEXT)
RETURNS BOOLEAN
LANGUAGE sql STABLE AS $$
    SELECT COALESCE((SELECT total_cp_count > 200 FROM account_degrees
                     WHERE bank_id = p_bank AND account_number = p_acct), FALSE)
$$;

CREATE OR REPLACE FUNCTION inv_guard_clearing(p_bank INT, p_acct TEXT, p_tool TEXT)
RETURNS VOID
LANGUAGE plpgsql STABLE AS $$
BEGIN
    IF inv_is_clearing(p_bank, p_acct) THEN
        RAISE EXCEPTION 'clearing_layer: % is a clearing-layer account (>200 historical counterparties); % is not run on it', p_bank || ':' || p_acct, p_tool
            USING HINT = 'Path analysis on clearing-layer accounts is noise (D17). Investigate the counterparty instead.';
    END IF;
END $$;

-- 4.2 收付对手方画像：clearing_layer 改用与拒绝、剪枝相同的定义
--   020 里这一列按"告警时刻之前的不同对手方 > 200"现算，而拒绝用的是 account_degrees（全期统计）。
--   两个定义在边界附近的账户上可能不一致：模型会同时看到"不是清算层"和"清算层，拒绝分析"两条矛盾的观察。
--   统一成一个定义（inv_is_clearing）；告警时刻之前的对手方数仍在 cp_total_before 里如实返回。
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
    hist AS (
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
           (SELECT n FROM hist), inv_is_clearing(p_bank, p_acct)
    FROM w
$$;

-- 4.4 过桥检查（查询与 020 相同）
DROP FUNCTION IF EXISTS inv_pass_through(INT, TEXT, TIMESTAMP, INT, NUMERIC);
CREATE FUNCTION inv_pass_through(p_bank INT, p_acct TEXT, p_as_of TIMESTAMP, p_hours INT DEFAULT 72,
                                 p_tol NUMERIC DEFAULT 0.10)
RETURNS TABLE (in_txn BIGINT, in_ts TIMESTAMP, in_from TEXT, in_amt NUMERIC,
               out_txn BIGINT, out_ts TIMESTAMP, out_to TEXT, out_amt NUMERIC, lag_hours NUMERIC, diff_pct NUMERIC)
LANGUAGE plpgsql STABLE AS $$
#variable_conflict use_column
BEGIN
    PERFORM inv_guard_clearing(p_bank, p_acct, 'inv_pass_through');
    RETURN QUERY
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
     AND NOT (o.to_bank = i.from_bank AND o.to_account = i.from_account)
    WHERE i.to_bank = p_bank AND i.to_account = p_acct
      AND i.txn_ts > p_as_of - make_interval(hours => p_hours) AND i.txn_ts <= p_as_of
    ORDER BY i.txn_id, ABS(o.amount_usd - i.amount_usd), o.txn_ts
    LIMIT 20;
END $$;

-- 4.5 资金回流追踪（查询与 020 相同）
DROP FUNCTION IF EXISTS inv_trace_cycle(INT, TEXT, TIMESTAMP, INT, INT, INT);
CREATE FUNCTION inv_trace_cycle(p_bank INT, p_acct TEXT, p_as_of TIMESTAMP, p_max_hops INT DEFAULT 12,
                                p_window_days INT DEFAULT 7, p_limit INT DEFAULT 5)
RETURNS TABLE (hops INT, path TEXT[], txn_path BIGINT[], start_ts TIMESTAMP, end_ts TIMESTAMP, span_hours NUMERIC)
LANGUAGE plpgsql STABLE AS $$
#variable_conflict use_column
BEGIN
    PERFORM inv_guard_clearing(p_bank, p_acct, 'inv_trace_cycle');
    RETURN QUERY
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
    SELECT walk.hop, walk.path, walk.txn_path, walk.st, walk.cur_ts,
           ROUND(EXTRACT(EPOCH FROM (walk.cur_ts - walk.st)) / 3600.0, 1)
    FROM walk WHERE walk.closed
    ORDER BY walk.cur_ts DESC, walk.hop
    LIMIT p_limit;
END $$;

-- 4.6 两跳结构（查询与 020 相同）
DROP FUNCTION IF EXISTS inv_two_hop(INT, TEXT, TIMESTAMP, INT);
CREATE FUNCTION inv_two_hop(p_bank INT, p_acct TEXT, p_as_of TIMESTAMP, p_hours INT DEFAULT 72)
RETURNS TABLE (direction TEXT, endpoint TEXT, n_intermediaries INT, intermediaries TEXT[], amt_second_hop NUMERIC)
LANGUAGE plpgsql STABLE AS $$
#variable_conflict use_column
BEGIN
    PERFORM inv_guard_clearing(p_bank, p_acct, 'inv_two_hop');
    RETURN QUERY
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
        SELECT 'gather_out'::text AS dir, s.to_bank || ':' || s.to_account AS ep,
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
        SELECT 'gather_in'::text AS dir, s.from_bank || ':' || s.from_account AS ep,
               COUNT(DISTINCT l.x_bank || ':' || l.x_acct)::int AS n_x,
               (ARRAY_AGG(DISTINCT l.x_bank || ':' || l.x_acct))[1:10] AS xs,
               ROUND(SUM(s.amount_usd), 2) AS amt
        FROM last_in l
        JOIN inv_txn s ON s.to_bank = l.x_bank AND s.to_account = l.x_acct
                      AND s.txn_ts <= l.t2 AND s.txn_ts > l.t2 - make_interval(hours => p_hours)
                      AND NOT (s.from_bank = p_bank AND s.from_account = p_acct)
        GROUP BY 1, 2
        HAVING COUNT(DISTINCT l.x_bank || ':' || l.x_acct) >= 2
        ORDER BY 3 DESC, 5 DESC LIMIT 5
    )
    SELECT gout.dir, gout.ep, gout.n_x, gout.xs, gout.amt FROM gout
    UNION ALL
    SELECT gin.dir, gin.ep, gin.n_x, gin.xs, gin.amt FROM gin;
END $$;