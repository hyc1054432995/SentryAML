-- ============================================================================
-- SentryAML - 025：人工复核的数据层（R10a）
-- ============================================================================
-- Agent 判完之后，人怎么接手。四张表 + 一个视图 + 一个最小权限角色：
--
--   reviewers           复核人员与角色：system（流水线）、analyst（分析员）、mlro（反洗钱报告负责人）
--   review_transitions  状态机：在哪个状态、做什么动作、进入哪个状态、谁能做、要不要写理由。
--                       状态机写成数据而不是写死在代码里：规则一目了然，冒烟测试可以直接核对
--   review_cases        复核案件：一条告警一个案件，打开时把 Agent 的结论"拍快照"存下来——
--                       事后审计要能还原"复核员当时看到的是什么"，不能去引用一个之后可能变化的东西
--   review_events       复核事件：每个动作一行，只追加，永不修改、永不删除。案件的当前状态 = 最后一条事件的 to_state
--   review_case_state   视图：每个案件的当前状态
--
-- 四条规则由数据库强制（触发器），任何写入路径都绕不过（D78）：
--   1. 只能按状态机转移；不合法的动作拒绝
--   2. 角色：例如只有 mlro 能批准 STR、能重开已关闭的案件
--   3. 理由：推翻 Agent 的结论、或没有 Agent 结论（T1 直接进复核、Agent 出错）时，必须写理由
--   4. 双人复核（maker-checker）：确认可疑、写过或改过 STR 草稿的人，不能批准这份 STR
-- 另外两条保护：
--   5. 防过期操作（乐观并发控制）：动作必须带上"我看到的状态"(seen_state)；
--      如果这期间别人已经改变了案件状态，拒绝，而不是在新状态上执行一个基于旧信息的决定
--   6. 只追加：review_cases、review_events 的 UPDATE / DELETE / TRUNCATE 一律被触发器拒绝，连库的所有者也一样
--
-- 用法：psql ... -f sql/025_review.sql   （可重复执行；不会删除已有的案件和事件）
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. 复核人员
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reviewers (
    reviewer_id  TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    role         TEXT NOT NULL CHECK (role IN ('system', 'analyst', 'mlro')),
    active       BOOLEAN NOT NULL DEFAULT true
);
-- 演示用的人员。真实系统里身份来自登录（R11 的接口层负责认证）；这里只有命令行，身份由 --as 声明
INSERT INTO reviewers (reviewer_id, name, role) VALUES
    ('system', 'Triage pipeline', 'system'),
    ('alice',  'Alice (analyst)', 'analyst'),
    ('bob',    'Bob (analyst)',   'analyst'),
    ('carol',  'Carol (MLRO)',    'mlro'),
    ('dave',   'Dave (MLRO)',     'mlro')
ON CONFLICT (reviewer_id) DO NOTHING;

-- ---------------------------------------------------------------------------
-- 2. 状态机
-- ---------------------------------------------------------------------------
--   new ──open──► pending_review ──close──────────────► closed ──reopen(mlro)──► pending_review
--                     │  └─request_info（留在原状态，写明要补查什么）
--                     └─confirm_suspicious──► suspicious_confirmed ──draft_str──► str_drafted
--                                                str_drafted ──edit_str / return_draft(mlro)──► str_drafted
--                                                str_drafted ──approve_str(mlro，且不是制作人)──► str_approved
--
-- reason_rule：never 不要求；always 必须写；on_override 推翻 Agent 结论或没有 Agent 结论时必须写
CREATE TABLE IF NOT EXISTS review_transitions (
    from_state   TEXT NOT NULL,
    action       TEXT NOT NULL,
    to_state     TEXT NOT NULL,
    roles        TEXT[] NOT NULL,
    reason_rule  TEXT NOT NULL CHECK (reason_rule IN ('never', 'always', 'on_override')),
    PRIMARY KEY (from_state, action)
);
-- 状态机是配置：每次执行本文件都整体重写成下面的定义（它不是审计数据，可以替换）
DELETE FROM review_transitions;
INSERT INTO review_transitions (from_state, action, to_state, roles, reason_rule) VALUES
    ('new',                  'open',               'pending_review',       '{system}',        'never'),
    ('pending_review',       'close',              'closed',               '{analyst,mlro}',  'on_override'),
    ('pending_review',       'confirm_suspicious', 'suspicious_confirmed', '{analyst,mlro}',  'on_override'),
    ('pending_review',       'request_info',       'pending_review',       '{analyst,mlro}',  'always'),
    ('suspicious_confirmed', 'draft_str',          'str_drafted',          '{analyst,mlro}',  'never'),
    ('str_drafted',          'edit_str',           'str_drafted',          '{analyst,mlro}',  'never'),
    ('str_drafted',          'return_draft',       'str_drafted',          '{mlro}',          'always'),
    ('str_drafted',          'approve_str',        'str_approved',         '{mlro}',          'never'),
    ('closed',               'reopen',             'pending_review',       '{mlro}',          'always');

-- ---------------------------------------------------------------------------
-- 3. 案件
-- ---------------------------------------------------------------------------
-- route：为什么进入人工复核（方案 B，D76）
--   tier1           T1 告警全部直接升级，由人复核
--   agent_escalate  T2 告警被 Agent 升级（Agent 出错也算，D75）
--   qa_sample       T2 告警被 Agent 关闭，但被质量抽查抽中——关闭是漏掉洗钱的方向，必须有人抽查
CREATE TABLE IF NOT EXISTS review_cases (
    case_id         BIGSERIAL PRIMARY KEY,
    alert_id        INT NOT NULL UNIQUE REFERENCES alerts(alert_id),
    route           TEXT NOT NULL CHECK (route IN ('tier1', 'agent_escalate', 'qa_sample')),
    agent_run_id    UUID REFERENCES agent_runs(run_id),
    agent_decision  TEXT CHECK (agent_decision IN ('escalate', 'close')),   -- NULL：没有 Agent 结论
    agent_result    JSONB,                                                  -- 打开时的快照：结论、理由、核验过的事实
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- 4. 事件（只追加）
-- ---------------------------------------------------------------------------
-- 客户端只提供：case_id、actor、action、seen_state、reason、payload。
-- seq、from_state、to_state、overrides_agent、created_at 一律由触发器计算并覆盖——不信任客户端给的值
CREATE TABLE IF NOT EXISTS review_events (
    event_id         BIGSERIAL PRIMARY KEY,
    case_id          BIGINT NOT NULL REFERENCES review_cases(case_id),
    seq              INT NOT NULL,
    actor            TEXT NOT NULL REFERENCES reviewers(reviewer_id),
    action           TEXT NOT NULL,
    seen_state       TEXT NOT NULL,
    from_state       TEXT NOT NULL,
    to_state         TEXT NOT NULL,
    overrides_agent  BOOLEAN NOT NULL DEFAULT false,
    reason           TEXT,
    payload          JSONB NOT NULL DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (case_id, seq)
);
CREATE INDEX IF NOT EXISTS review_events_actor ON review_events (actor);

CREATE OR REPLACE FUNCTION review_event_guard() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    cur_state  TEXT;
    last_seq   INT;
    t          review_transitions%ROWTYPE;
    a_role     TEXT;
    a_dec      TEXT;
    override   BOOLEAN;
BEGIN
    -- 同一个案件的事件逐个处理：两个复核员同时操作时，第二个要等第一个的事务结束，
    -- 然后在第一个写入之后的状态上重新检查（事务级咨询锁，事务结束自动释放）
    PERFORM pg_advisory_xact_lock(25, NEW.case_id::int);

    SELECT e.to_state, e.seq INTO cur_state, last_seq
    FROM review_events e WHERE e.case_id = NEW.case_id ORDER BY e.seq DESC LIMIT 1;
    IF NOT FOUND THEN
        cur_state := 'new';
        last_seq := 0;
    END IF;

    -- 5. 防过期操作
    IF NEW.seen_state IS DISTINCT FROM cur_state THEN
        RAISE EXCEPTION 'review_stale: case % is now "%", but the action was taken on "%"; reload and decide again',
            NEW.case_id, cur_state, NEW.seen_state;
    END IF;

    -- 1. 状态机
    SELECT * INTO t FROM review_transitions WHERE from_state = cur_state AND action = NEW.action;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'review_illegal: action "%" is not allowed in state "%"', NEW.action, cur_state;
    END IF;

    -- 2. 角色
    SELECT role INTO a_role FROM reviewers WHERE reviewer_id = NEW.actor AND active;
    IF a_role IS NULL THEN
        RAISE EXCEPTION 'review_actor: unknown or inactive reviewer "%"', NEW.actor;
    END IF;
    IF NOT (a_role = ANY (t.roles)) THEN
        RAISE EXCEPTION 'review_role: "%" (role %) cannot "%"; allowed roles: %', NEW.actor, a_role, NEW.action, t.roles;
    END IF;

    -- 3. 理由：推翻 Agent，或者没有 Agent 结论可依据
    SELECT agent_decision INTO a_dec FROM review_cases WHERE case_id = NEW.case_id;
    -- coalesce 不能省：没有 Agent 结论时 a_dec 是 NULL，"NULL = 'escalate'" 的结果是 NULL 而不是 false
    -- （SQL 的三值逻辑），整个表达式会变成 NULL，写进 NOT NULL 的 overrides_agent 时报错
    override := coalesce((NEW.action = 'close' AND a_dec = 'escalate')
                      OR (NEW.action = 'confirm_suspicious' AND a_dec = 'close'), false);
    IF (t.reason_rule = 'always' OR (t.reason_rule = 'on_override' AND (override OR a_dec IS NULL)))
       AND coalesce(btrim(NEW.reason), '') = '' THEN
        RAISE EXCEPTION 'review_reason: "%" on case % requires a reason (%)', NEW.action, NEW.case_id,
            CASE WHEN t.reason_rule = 'always' THEN 'always required'
                 WHEN override THEN 'it overrides the agent''s decision'
                 ELSE 'there is no agent decision to rely on' END;
    END IF;

    -- 4. 双人复核：确认可疑、写过或改过草稿的人不能批准
    IF NEW.action = 'approve_str' AND EXISTS (
        SELECT 1 FROM review_events e
        WHERE e.case_id = NEW.case_id AND e.actor = NEW.actor
          AND e.action IN ('confirm_suspicious', 'draft_str', 'edit_str')) THEN
        RAISE EXCEPTION 'review_four_eyes: "%" confirmed or drafted this STR and cannot approve it', NEW.actor;
    END IF;

    NEW.seq := last_seq + 1;
    NEW.from_state := cur_state;
    NEW.to_state := t.to_state;
    NEW.overrides_agent := override;
    NEW.created_at := clock_timestamp();
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS review_event_guard ON review_events;
CREATE TRIGGER review_event_guard BEFORE INSERT ON review_events
    FOR EACH ROW EXECUTE FUNCTION review_event_guard();

-- 6. 只追加
CREATE OR REPLACE FUNCTION review_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'review_immutable: % on % is not allowed; review records are append-only', TG_OP, TG_TABLE_NAME;
END $$;

DROP TRIGGER IF EXISTS review_events_append_only ON review_events;
CREATE TRIGGER review_events_append_only BEFORE UPDATE OR DELETE ON review_events
    FOR EACH ROW EXECUTE FUNCTION review_append_only();
DROP TRIGGER IF EXISTS review_events_no_truncate ON review_events;
CREATE TRIGGER review_events_no_truncate BEFORE TRUNCATE ON review_events
    FOR EACH STATEMENT EXECUTE FUNCTION review_append_only();
DROP TRIGGER IF EXISTS review_cases_append_only ON review_cases;
CREATE TRIGGER review_cases_append_only BEFORE UPDATE OR DELETE ON review_cases
    FOR EACH ROW EXECUTE FUNCTION review_append_only();
DROP TRIGGER IF EXISTS review_cases_no_truncate ON review_cases;
CREATE TRIGGER review_cases_no_truncate BEFORE TRUNCATE ON review_cases
    FOR EACH STATEMENT EXECUTE FUNCTION review_append_only();

-- ---------------------------------------------------------------------------
-- 5. 当前状态视图
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW review_case_state AS
SELECT c.case_id, c.alert_id, c.route, c.agent_decision,
       coalesce(e.to_state, 'new') AS state, coalesce(e.seq, 0) AS last_seq,
       e.actor AS last_actor, e.created_at AS last_at
FROM review_cases c
LEFT JOIN LATERAL (
    SELECT to_state, seq, actor, created_at FROM review_events
    WHERE case_id = c.case_id ORDER BY seq DESC LIMIT 1
) e ON true;

-- ---------------------------------------------------------------------------
-- 6. 复核界面用的最小权限角色
-- ---------------------------------------------------------------------------
-- 能读：告警、Agent 运行与工具调用记录（复核员要能看到 Agent 每一步查到了什么）、复核表
-- 能写：只有 review_events 的 INSERT。不能建案件（那是流水线的事）、不能改或删任何记录
-- 读不到：alert_labels（真实的复核员没有"标准答案"）
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sentry_review') THEN
        CREATE ROLE sentry_review LOGIN PASSWORD 'sentry_review_dev';
    END IF;
END $$;
DO $$ BEGIN EXECUTE format('GRANT CONNECT ON DATABASE %I TO sentry_review', current_database()); END $$;
GRANT USAGE ON SCHEMA public TO sentry_review;
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM sentry_review;
GRANT SELECT ON alerts, agent_runs, agent_tool_calls, reviewers, review_transitions, review_cases,
                review_events, review_case_state TO sentry_review;
GRANT INSERT ON review_events TO sentry_review;
GRANT USAGE ON SEQUENCE review_events_event_id_seq TO sentry_review;

COMMENT ON TABLE review_cases IS 'R10a：复核案件，一条告警一个；打开时保存 Agent 结论的快照；只追加';
COMMENT ON TABLE review_events IS 'R10a：复核事件，只追加；状态机、角色、理由、双人复核、防过期操作由触发器强制';
COMMENT ON TABLE review_transitions IS 'R10a：复核状态机（配置）';