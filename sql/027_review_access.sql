-- ============================================================================
-- SentryAML - 027：登录、会话、认领与租约（R11a）
-- ============================================================================
-- R11 把命令行的复核流程变成多人使用的网页系统。命令行里"你是谁"由 --as 声明，谁都能写成 carol（D78 的局限）；
-- 网页系统里身份来自登录。这个文件加四样东西：
--
--   review_config          两个时长：认领的租约多长、登录一次管多久。只有库的所有者能改
--   reviewer_credentials   密码的哈希（argon2id）。CHECK 保证存进来的一定是哈希，不可能是明文
--   review_sessions        登录会话：浏览器的 cookie 里是一串随机令牌，这里只存它的 sha256——
--                          数据库泄露也拿不到能直接用的令牌。只追加；唯一允许的修改是"注销"（填 revoked_at）
--   review_claims          认领与释放，只追加。最后一条是"认领"且没过期 = 这个案件正被这个人处理
--
-- 租约规则由数据库强制（与 D78 一致：规则在库里，任何写入路径都绕不过）：
--   1. 案件被别人认领、且租约没过期时，分析员不能对它写复核事件、不能写草稿；MLRO 例外（批准、退回要能进行）
--   2. 没有人认领的案件谁都可以操作——命令行、评测脚本照旧可用；认领是界面防止重复劳动的手段，不是操作的前提
--   3. 租约到期自动失效，不需要任何人"解锁"；认领人可以续期（再认领一次）或主动释放
--   4. MLRO 可以接管别人的认领，但必须写理由，并记下接管了谁
--   5. 认领时间、到期时间、接管了谁全部由数据库计算，不信任客户端
--
-- 依赖 025、026。用法：psql ... -f sql/027_review_access.sql   （可重复执行）
-- 注意：025 开头会 REVOKE sentry_review 的全部表权限再重新授权，重跑 025 之后要按顺序重跑 026、027
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. 时长配置
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS review_config (
    key    TEXT PRIMARY KEY,
    value  INT NOT NULL CHECK (value > 0),
    note   TEXT
);
INSERT INTO review_config (key, value, note) VALUES
    ('lease_seconds',   1800,  '认领一次的租约：30 分钟'),
    ('session_seconds', 28800, '登录一次的有效期：8 小时，到期重新登录')
ON CONFLICT (key) DO NOTHING;

CREATE OR REPLACE FUNCTION review_setting(p_key TEXT) RETURNS INT
LANGUAGE plpgsql STABLE AS $$
DECLARE v INT;
BEGIN
    SELECT value INTO v FROM review_config WHERE key = p_key;
    IF v IS NULL THEN
        RAISE EXCEPTION 'review_config: setting "%" is missing', p_key;
    END IF;
    RETURN v;
END $$;

-- ---------------------------------------------------------------------------
-- 2. 密码哈希
-- ---------------------------------------------------------------------------
-- 只有库的所有者能写（scripts/review_auth.py set-password）；复核角色只能读，用来核对登录
CREATE TABLE IF NOT EXISTS reviewer_credentials (
    reviewer_id    TEXT PRIMARY KEY REFERENCES reviewers(reviewer_id),
    password_hash  TEXT NOT NULL CHECK (password_hash LIKE '$argon2id$%'),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- 3. 会话
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS review_sessions (
    session_id    BIGSERIAL PRIMARY KEY,
    token_sha256  CHAR(64) NOT NULL UNIQUE CHECK (token_sha256 ~ '^[0-9a-f]{64}$'),
    reviewer_id   TEXT NOT NULL REFERENCES reviewers(reviewer_id),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    expires_at    TIMESTAMPTZ NOT NULL,
    revoked_at    TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS review_sessions_reviewer ON review_sessions (reviewer_id);

CREATE OR REPLACE FUNCTION review_session_guard() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        -- 只有在岗的分析员、MLRO 能登录；流水线账号 system 不能
        IF NOT EXISTS (SELECT 1 FROM reviewers WHERE reviewer_id = NEW.reviewer_id AND active
                       AND role IN ('analyst', 'mlro')) THEN
            RAISE EXCEPTION 'session_role: "%" cannot log in', NEW.reviewer_id;
        END IF;
        NEW.created_at := clock_timestamp();
        NEW.expires_at := NEW.created_at + make_interval(secs => review_setting('session_seconds'));
        NEW.revoked_at := NULL;
        RETURN NEW;
    END IF;
    -- UPDATE：唯一允许的修改是注销一次
    IF OLD.revoked_at IS NOT NULL THEN
        RAISE EXCEPTION 'review_immutable: session % is already revoked', OLD.session_id;
    END IF;
    IF NEW.session_id IS DISTINCT FROM OLD.session_id OR NEW.token_sha256 IS DISTINCT FROM OLD.token_sha256
       OR NEW.reviewer_id IS DISTINCT FROM OLD.reviewer_id OR NEW.created_at IS DISTINCT FROM OLD.created_at
       OR NEW.expires_at IS DISTINCT FROM OLD.expires_at OR NEW.revoked_at IS NULL THEN
        RAISE EXCEPTION 'review_immutable: a session can only be revoked';
    END IF;
    NEW.revoked_at := clock_timestamp();
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS review_session_guard ON review_sessions;
CREATE TRIGGER review_session_guard BEFORE INSERT OR UPDATE ON review_sessions
    FOR EACH ROW EXECUTE FUNCTION review_session_guard();
DROP TRIGGER IF EXISTS review_sessions_no_delete ON review_sessions;
CREATE TRIGGER review_sessions_no_delete BEFORE DELETE ON review_sessions
    FOR EACH ROW EXECUTE FUNCTION review_append_only();
DROP TRIGGER IF EXISTS review_sessions_no_truncate ON review_sessions;
CREATE TRIGGER review_sessions_no_truncate BEFORE TRUNCATE ON review_sessions
    FOR EACH STATEMENT EXECUTE FUNCTION review_append_only();

-- ---------------------------------------------------------------------------
-- 4. 认领与释放（只追加）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS review_claims (
    claim_id    BIGSERIAL PRIMARY KEY,
    case_id     BIGINT NOT NULL REFERENCES review_cases(case_id),
    actor       TEXT NOT NULL REFERENCES reviewers(reviewer_id),
    kind        TEXT NOT NULL CHECK (kind IN ('claim', 'release')),
    took_over   TEXT REFERENCES reviewers(reviewer_id),   -- 由触发器计算：MLRO 接管或释放了谁的认领
    reason      TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    expires_at  TIMESTAMPTZ                                -- 由触发器计算；release 为 NULL
);
CREATE INDEX IF NOT EXISTS review_claims_case ON review_claims (case_id, claim_id DESC);

-- 当前认领人：最后一条是 claim 且没过期。用 clock_timestamp() 而不是 now()：
-- now() 是事务开始的时刻，在一个长事务里不会前进，租约就永远不会"在事务中途"到期
CREATE OR REPLACE FUNCTION review_claim_holder(p_case BIGINT)
RETURNS TABLE (holder TEXT, expires_at TIMESTAMPTZ)
LANGUAGE sql VOLATILE AS $$
    SELECT c.actor, c.expires_at
    FROM (SELECT actor, kind, expires_at FROM review_claims
          WHERE case_id = p_case ORDER BY claim_id DESC LIMIT 1) c
    WHERE c.kind = 'claim' AND c.expires_at > clock_timestamp()
$$;

CREATE OR REPLACE FUNCTION review_claim_guard() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    v_role    TEXT;
    v_state   TEXT;
    v_holder  TEXT;
    v_until   TIMESTAMPTZ;
BEGIN
    PERFORM pg_advisory_xact_lock(25, NEW.case_id::int);        -- 与复核事件、草稿共用同一把锁
    SELECT role INTO v_role FROM reviewers WHERE reviewer_id = NEW.actor AND active;
    IF v_role IS NULL OR v_role NOT IN ('analyst', 'mlro') THEN
        RAISE EXCEPTION 'claim_role: "%" cannot claim cases', NEW.actor;
    END IF;
    SELECT state INTO v_state FROM review_case_state WHERE case_id = NEW.case_id;
    SELECT h.holder, h.expires_at INTO v_holder, v_until FROM review_claim_holder(NEW.case_id) h;
    NEW.created_at := clock_timestamp();
    NEW.took_over := NULL;

    IF NEW.kind = 'claim' THEN
        IF v_state IN ('closed', 'str_approved') THEN
            RAISE EXCEPTION 'claim_state: case % is "%"; there is nothing to work on', NEW.case_id, v_state;
        END IF;
        IF v_holder IS NOT NULL AND v_holder <> NEW.actor THEN
            IF v_role <> 'mlro' THEN
                RAISE EXCEPTION 'review_claimed: case % is claimed by % until % (SGT)', NEW.case_id, v_holder,
                    to_char(v_until AT TIME ZONE 'Asia/Singapore', 'HH24:MI');
            END IF;
            IF coalesce(btrim(NEW.reason), '') = '' THEN
                RAISE EXCEPTION 'review_reason: taking over %''s claim on case % requires a reason', v_holder, NEW.case_id;
            END IF;
            NEW.took_over := v_holder;
        END IF;
        -- 自己再认领一次 = 续期：从现在起重新计算
        NEW.expires_at := NEW.created_at + make_interval(secs => review_setting('lease_seconds'));
    ELSE
        IF v_holder IS NULL THEN
            RAISE EXCEPTION 'claim_none: case % has no active claim', NEW.case_id;
        END IF;
        IF v_holder <> NEW.actor THEN
            IF v_role <> 'mlro' THEN
                RAISE EXCEPTION 'review_claimed: case % is claimed by % until % (SGT)', NEW.case_id, v_holder,
                    to_char(v_until AT TIME ZONE 'Asia/Singapore', 'HH24:MI');
            END IF;
            IF coalesce(btrim(NEW.reason), '') = '' THEN
                RAISE EXCEPTION 'review_reason: releasing %''s claim on case % requires a reason', v_holder, NEW.case_id;
            END IF;
            NEW.took_over := v_holder;
        END IF;
        NEW.expires_at := NULL;
    END IF;
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS review_claim_guard ON review_claims;
CREATE TRIGGER review_claim_guard BEFORE INSERT ON review_claims
    FOR EACH ROW EXECUTE FUNCTION review_claim_guard();
DROP TRIGGER IF EXISTS review_claims_append_only ON review_claims;
CREATE TRIGGER review_claims_append_only BEFORE UPDATE OR DELETE ON review_claims
    FOR EACH ROW EXECUTE FUNCTION review_append_only();
DROP TRIGGER IF EXISTS review_claims_no_truncate ON review_claims;
CREATE TRIGGER review_claims_no_truncate BEFORE TRUNCATE ON review_claims
    FOR EACH STATEMENT EXECUTE FUNCTION review_append_only();

CREATE OR REPLACE VIEW review_claim_state AS
SELECT c.case_id, h.holder, h.expires_at
FROM review_cases c
LEFT JOIN LATERAL review_claim_holder(c.case_id) h ON true;

-- ---------------------------------------------------------------------------
-- 5. 租约检查：复核事件和草稿写入前
-- ---------------------------------------------------------------------------
-- 触发器按名字的字母顺序执行：
--   review_events：review_event_guard（025，取锁、状态机…）→ review_event_guard_lease（本文件）→ review_event_guard_str（026）
--   str_drafts：   str_draft_guard（026，取锁、检查…）     → str_draft_guard_lease（本文件）
-- 锁由前面的触发器先取到，这里读到的认领状态不会在检查过程中被别人改掉
CREATE OR REPLACE FUNCTION review_lease_check(p_case BIGINT, p_actor TEXT) RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
    v_role    TEXT;
    v_holder  TEXT;
    v_until   TIMESTAMPTZ;
BEGIN
    SELECT role INTO v_role FROM reviewers WHERE reviewer_id = p_actor;
    IF v_role IN ('system', 'mlro') THEN
        RETURN;
    END IF;
    SELECT h.holder, h.expires_at INTO v_holder, v_until FROM review_claim_holder(p_case) h;
    IF v_holder IS NOT NULL AND v_holder <> p_actor THEN
        RAISE EXCEPTION 'review_claimed: case % is claimed by % until % (SGT); only they or an MLRO can act on it now',
            p_case, v_holder, to_char(v_until AT TIME ZONE 'Asia/Singapore', 'HH24:MI');
    END IF;
END $$;

CREATE OR REPLACE FUNCTION review_event_guard_lease() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM review_lease_check(NEW.case_id, NEW.actor);
    RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS review_event_guard_lease ON review_events;
CREATE TRIGGER review_event_guard_lease BEFORE INSERT ON review_events
    FOR EACH ROW EXECUTE FUNCTION review_event_guard_lease();

CREATE OR REPLACE FUNCTION str_draft_guard_lease() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM review_lease_check(NEW.case_id, NEW.created_by);
    RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS str_draft_guard_lease ON str_drafts;
CREATE TRIGGER str_draft_guard_lease BEFORE INSERT ON str_drafts
    FOR EACH ROW EXECUTE FUNCTION str_draft_guard_lease();

-- ---------------------------------------------------------------------------
-- 6. 权限
-- ---------------------------------------------------------------------------
-- 复核角色：能读配置和密码哈希（登录时核对）；能建会话、只能改会话的 revoked_at（注销）；能写认领记录。
-- 不能改密码、不能改配置、不能改或删认领记录
GRANT SELECT ON review_config, reviewer_credentials, review_claim_state TO sentry_review;
GRANT SELECT, INSERT ON review_sessions, review_claims TO sentry_review;
GRANT UPDATE (revoked_at) ON review_sessions TO sentry_review;
GRANT USAGE ON SEQUENCE review_sessions_session_id_seq, review_claims_claim_id_seq TO sentry_review;

COMMENT ON TABLE reviewer_credentials IS 'R11a：复核员密码的 argon2id 哈希；只有所有者能写';
COMMENT ON TABLE review_sessions IS 'R11a：登录会话，只存令牌的 sha256；只追加，唯一允许的修改是注销';
COMMENT ON TABLE review_claims IS 'R11a：认领与释放，只追加；租约由触发器计算，别人认领期间分析员不能操作';