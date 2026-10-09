-- ============================================================================
-- SentryAML - 026：STR 草稿（R10b-1）
-- ============================================================================
-- 复核员确认可疑之后，系统按 STRO 表格的结构组装 STR 草稿（scripts/str_draft.py）。这个文件定义草稿怎么存、
-- 什么样的草稿算"合格"，以及草稿和复核事件怎么绑定：
--
--   str_drafts                 草稿的各个版本，只追加。每生成或修改一次就是一个新版本
--   str_attachment_rows(...)   附件里的交易：所列账户在数据里的全部交易（清算层账户除外）。
--                              组装草稿和核对草稿调用的是同一个函数，所以"附件该有哪些交易"只有一个定义
--   str_draft_problems(...)    草稿的全部检查，返回问题清单（空 = 合格）。写入草稿时、批准时都由数据库调用它；
--                              脚本的 check 命令也只是调用它——检查规则只有这一份（D79）
--   两个触发器                 str_drafts 写入前检查；review_events 上的第二个触发器把动作和草稿版本绑定
--
-- 草稿里每一项结构化内容都能在库里找到原记录：交易逐笔核对时间、双方账户、金额；账户必须存在；
-- 每个账户的持有主体都要列出；可疑类别必须是知识图谱里、在审查日期仍然有效的原文；发现日期必须是确认可疑那条
-- 复核事件的日期；审查日期必须是草稿生成的日期（不能倒签）。
--
-- 依赖 025_review.sql。用法：psql ... -f sql/026_str_drafts.sql   （可重复执行；重跑 025 不会影响这里的触发器）
-- ============================================================================

-- ---------------------------------------------------------------------------
-- 1. 复核角色需要读的数据：组装草稿时要查交易、账户、主体和知识图谱
--    仍然读不到 alert_labels、txn_norm（含标签的原表）
-- ---------------------------------------------------------------------------
GRANT SELECT ON inv_txn, accounts, entities, account_degrees, inv_cases,
                kb_typologies, kb_indicators, kb_typology_indicators, kb_documents, kb_units
TO sentry_review;

-- ---------------------------------------------------------------------------
-- 2. 草稿表
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS str_drafts (
    draft_id           BIGSERIAL PRIMARY KEY,
    case_id            BIGINT NOT NULL REFERENCES review_cases(case_id),
    version            INT NOT NULL,                  -- 由触发器分配：这个案件已有的最大版本 + 1
    created_by         TEXT NOT NULL REFERENCES reviewers(reviewer_id),
    generator          JSONB NOT NULL,                -- 谁生成的：程序组装（哪个版本的组装器）/ 人工修改 / 模型撰写
    body               JSONB NOT NULL,                -- 草稿内容，按 STRO 表格的分区组织
    attachment_csv     TEXT,                          -- 附件：所列账户的全部交易（指南：超过 40 笔用 CSV）
    attachment_sha256  CHAR(64),                      -- 由触发器计算，不信任客户端
    created_at         TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
    UNIQUE (case_id, version)
);

-- ---------------------------------------------------------------------------
-- 3. 附件里的交易
-- ---------------------------------------------------------------------------
-- p_accounts：'bank:account' 形式。调用方负责去掉清算层账户（它们的全部交易有几十万笔，且与本案无关，D17）
CREATE OR REPLACE FUNCTION str_attachment_rows(p_accounts TEXT[])
RETURNS TABLE (txn_id BIGINT, txn_ts TIMESTAMP, from_bank INT, from_account TEXT, to_bank INT, to_account TEXT,
               amount_usd NUMERIC, is_cross_currency BOOLEAN)
LANGUAGE sql STABLE AS $$
    WITH acc AS (
        SELECT split_part(a, ':', 1)::int AS bank, split_part(a, ':', 2) AS acct FROM unnest(p_accounts) a
    ), hits AS (
        SELECT t.txn_id FROM inv_txn t JOIN acc ON t.from_bank = acc.bank AND t.from_account = acc.acct
        UNION
        SELECT t.txn_id FROM inv_txn t JOIN acc ON t.to_bank = acc.bank AND t.to_account = acc.acct
    )
    SELECT t.txn_id, t.txn_ts, t.from_bank, t.from_account::text, t.to_bank, t.to_account::text,
           round(t.amount_usd, 2), t.is_cross_currency
    FROM inv_txn t JOIN hits h USING (txn_id)
    ORDER BY t.txn_ts, t.txn_id
$$;

-- ---------------------------------------------------------------------------
-- 4. 草稿检查：返回问题清单，空 = 合格
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION str_draft_problems(p_case BIGINT, p_body JSONB, p_created_at TIMESTAMPTZ)
RETURNS TABLE (problem TEXT, detail TEXT)
LANGUAGE plpgsql STABLE AS $$
DECLARE
    v_alert_txn  BIGINT;
    v_confirm    RECORD;
    v_typology   TEXT;
    v_review     DATE;
    v_n          INT;
    v_att        TEXT[];
BEGIN
    SELECT a.txn_id INTO v_alert_txn FROM review_cases c JOIN alerts a USING (alert_id) WHERE c.case_id = p_case;
    SELECT e.created_at, e.payload INTO v_confirm FROM review_events e
    WHERE e.case_id = p_case AND e.action = 'confirm_suspicious' ORDER BY e.seq DESC LIMIT 1;
    v_typology := p_body #>> '{reasons,typology}';
    v_review := (p_body ->> 'review_date')::date;

    -- 交易：告警这一笔必须在；每一笔逐项与原记录一致；不超过 40 笔（表格上限）
    IF NOT EXISTS (SELECT 1 FROM jsonb_array_elements(p_body -> 'suspicious_transactions') x
                   WHERE (x ->> 'txn_id')::bigint = v_alert_txn) THEN
        problem := 'no_alert_txn'; detail := format('alerted transaction %s is not listed', v_alert_txn); RETURN NEXT;
    END IF;
    FOR problem, detail IN
        SELECT 'txn_mismatch', format('transaction %s does not match the record', x ->> 'txn_id')
        FROM jsonb_array_elements(p_body -> 'suspicious_transactions') x
        WHERE NOT EXISTS (
            SELECT 1 FROM inv_txn t
            WHERE t.txn_id = (x ->> 'txn_id')::bigint
              AND to_char(t.txn_ts, 'YYYY-MM-DD HH24:MI') = x ->> 'time'
              AND t.from_bank || ':' || t.from_account = x ->> 'from'
              AND t.to_bank || ':' || t.to_account = x ->> 'to'
              AND round(t.amount_usd, 2) = (x ->> 'amount_usd')::numeric)
    LOOP RETURN NEXT; END LOOP;
    v_n := jsonb_array_length(p_body -> 'suspicious_transactions');
    IF v_n > 40 THEN problem := 'too_many_txns'; detail := v_n || ' > 40'; RETURN NEXT; END IF;

    -- 账户：必须存在；不超过 40 个；每个账户的持有主体都要列在主体信息里
    v_n := jsonb_array_length(p_body -> 'accounts');
    IF v_n > 40 THEN problem := 'too_many_accounts'; detail := v_n || ' > 40'; RETURN NEXT; END IF;
    FOR problem, detail IN
        SELECT 'unknown_account', x ->> 'account'
        FROM jsonb_array_elements(p_body -> 'accounts') x
        WHERE NOT EXISTS (SELECT 1 FROM accounts a      -- 按 (bank_id, account_number) 拆开比较，才能用上主键索引
                          WHERE a.bank_id = split_part(x ->> 'account', ':', 1)::int
                            AND a.account_number = split_part(x ->> 'account', ':', 2))
    LOOP RETURN NEXT; END LOOP;
    FOR problem, detail IN
        SELECT 'holder_missing', format('%s (entity %s)', x ->> 'account', a.entity_id)
        FROM jsonb_array_elements(p_body -> 'accounts') x
        JOIN accounts a ON a.bank_id = split_part(x ->> 'account', ':', 1)::int
                       AND a.account_number = split_part(x ->> 'account', ':', 2)
        WHERE NOT EXISTS (SELECT 1 FROM jsonb_array_elements(p_body -> 'entities') y
                          WHERE y ->> 'entity_id' = a.entity_id::text)
    LOOP RETURN NEXT; END LOOP;

    -- 日期：审查日期 = 草稿生成当天（新加坡时间，不能倒签）；发现日期 = 确认可疑那条事件的日期
    IF v_review IS DISTINCT FROM (p_created_at AT TIME ZONE 'Asia/Singapore')::date THEN
        problem := 'review_date'; detail := format('review_date %s is not the drafting date %s', v_review,
                                                   (p_created_at AT TIME ZONE 'Asia/Singapore')::date);
        RETURN NEXT;
    END IF;
    IF v_confirm IS NULL THEN
        problem := 'not_confirmed'; detail := 'no confirm_suspicious event on this case'; RETURN NEXT;
    ELSIF (p_body #>> '{reasons,detection_date}') IS DISTINCT FROM
          to_char(v_confirm.created_at AT TIME ZONE 'Asia/Singapore', 'YYYY-MM-DD') THEN
        problem := 'detection_date';
        detail := format('detection_date %s is not the date suspicion was confirmed (%s)',
                         p_body #>> '{reasons,detection_date}',
                         to_char(v_confirm.created_at AT TIME ZONE 'Asia/Singapore', 'YYYY-MM-DD'));
        RETURN NEXT;
    END IF;

    -- 手法与可疑类别：手法必须是复核员确认时选定的那一个；每条类别必须是这种手法在知识图谱里的原文，且审查日期仍然有效
    IF v_confirm IS NOT NULL AND (v_confirm.payload ->> 'typology') IS NOT NULL
       AND v_typology IS DISTINCT FROM v_confirm.payload ->> 'typology' THEN
        problem := 'typology_mismatch';
        detail := format('draft says %s, the reviewer confirmed %s', v_typology, v_confirm.payload ->> 'typology');
        RETURN NEXT;
    END IF;
    IF coalesce(jsonb_array_length(p_body #> '{reasons,categories}'), 0) = 0 THEN
        problem := 'no_categories'; detail := 'at least one category of suspicion is required'; RETURN NEXT;
    END IF;
    FOR problem, detail IN
        SELECT 'category_invalid', x ->> 'indicator_id'
        FROM jsonb_array_elements(coalesce(p_body #> '{reasons,categories}', '[]')) x
        WHERE NOT EXISTS (SELECT 1 FROM kb_typology_citations(v_typology, v_review) c
                          WHERE c.indicator_id = x ->> 'indicator_id' AND c.quote = x ->> 'quote'
                            AND c.quote_in_force)
    LOOP RETURN NEXT; END LOOP;

    -- 附件：笔数必须等于按所列账户重新计算的结果（内容的完整性由 sha256 保证）
    v_att := ARRAY(SELECT jsonb_array_elements_text(coalesce(p_body #> '{attachment,accounts}', '[]')));
    SELECT count(*) INTO v_n FROM str_attachment_rows(v_att);
    IF v_n IS DISTINCT FROM (p_body #>> '{attachment,rows}')::int THEN
        problem := 'attachment_count'; detail := format('attachment says %s rows, the data has %s',
                                                        p_body #>> '{attachment,rows}', v_n);
        RETURN NEXT;
    END IF;

    -- 理由（自由文本）：表格上限 10,000 字符（R10b-2 撰写；为空不算问题，批准时另行要求）
    IF length(coalesce(p_body #>> '{reasons,narrative}', '')) > 10000 THEN
        problem := 'narrative_too_long'; detail := length(p_body #>> '{reasons,narrative}') || ' > 10000';
        RETURN NEXT;
    END IF;
END $$;

-- ---------------------------------------------------------------------------
-- 5. 写入草稿前：案件状态、版本号、附件哈希、全部检查
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION str_draft_guard() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    v_state  TEXT;
    v_list   TEXT;
BEGIN
    PERFORM pg_advisory_xact_lock(25, NEW.case_id::int);          -- 与复核事件共用同一把锁
    SELECT state INTO v_state FROM review_case_state WHERE case_id = NEW.case_id;
    IF v_state IS NULL OR v_state NOT IN ('suspicious_confirmed', 'str_drafted') THEN
        RAISE EXCEPTION 'str_state: case % is "%"; drafts can only be written after suspicion is confirmed',
            NEW.case_id, v_state;
    END IF;
    SELECT coalesce(max(version), 0) + 1 INTO NEW.version FROM str_drafts WHERE case_id = NEW.case_id;
    NEW.created_at := clock_timestamp();
    NEW.attachment_sha256 := CASE WHEN NEW.attachment_csv IS NULL THEN NULL
                                  ELSE encode(sha256(convert_to(NEW.attachment_csv, 'UTF8')), 'hex') END;
    SELECT string_agg(p.problem || ': ' || p.detail, '; ') INTO v_list
    FROM str_draft_problems(NEW.case_id, NEW.body, NEW.created_at) p;
    IF v_list IS NOT NULL THEN
        RAISE EXCEPTION 'str_invalid: %', v_list;
    END IF;
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS str_draft_guard ON str_drafts;
CREATE TRIGGER str_draft_guard BEFORE INSERT ON str_drafts FOR EACH ROW EXECUTE FUNCTION str_draft_guard();

DROP TRIGGER IF EXISTS str_drafts_append_only ON str_drafts;
CREATE TRIGGER str_drafts_append_only BEFORE UPDATE OR DELETE ON str_drafts
    FOR EACH ROW EXECUTE FUNCTION review_append_only();
DROP TRIGGER IF EXISTS str_drafts_no_truncate ON str_drafts;
CREATE TRIGGER str_drafts_no_truncate BEFORE TRUNCATE ON str_drafts
    FOR EACH STATEMENT EXECUTE FUNCTION review_append_only();

-- ---------------------------------------------------------------------------
-- 6. 复核事件上的第二个触发器：动作与草稿绑定
-- ---------------------------------------------------------------------------
-- 触发器按名字的字母顺序执行：review_event_guard（025，状态机等）先跑，review_event_guard_str 后跑。
-- 分成两个触发器，是为了重跑 025 时不会覆盖这里的规则。
--   confirm_suspicious      必须选定手法（payload.typology），且是知识图谱里的手法——STR 的可疑类别由它决定
--   draft_str / edit_str    必须指向这个案件最新的草稿版本，且这个版本是本人写的、还没有被别的事件引用过
--   approve_str             必须写明批准的是哪个版本，且是最新版本；这个版本必须合格，并且已经写了理由（自由文本）
CREATE OR REPLACE FUNCTION review_event_guard_str() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    v_ver     INT;
    v_latest  str_drafts%ROWTYPE;
    v_used    INT;
    v_list    TEXT;
BEGIN
    IF NEW.action = 'confirm_suspicious' THEN
        IF NOT EXISTS (SELECT 1 FROM kb_typologies WHERE typology_id = NEW.payload ->> 'typology') THEN
            RAISE EXCEPTION 'str_typology: confirming suspicion requires a typology from the knowledge graph (got %)',
                coalesce(NEW.payload ->> 'typology', 'none');
        END IF;
        RETURN NEW;
    END IF;
    IF NEW.action NOT IN ('draft_str', 'edit_str', 'approve_str') THEN
        RETURN NEW;
    END IF;

    v_ver := (NEW.payload ->> 'draft_version')::int;
    SELECT * INTO v_latest FROM str_drafts WHERE case_id = NEW.case_id ORDER BY version DESC LIMIT 1;
    IF v_latest.draft_id IS NULL THEN
        RAISE EXCEPTION 'str_no_draft: case % has no STR draft', NEW.case_id;
    END IF;
    IF v_ver IS DISTINCT FROM v_latest.version THEN
        RAISE EXCEPTION 'str_version: "%" refers to version %, but the latest draft is version %',
            NEW.action, coalesce(v_ver::text, 'none'), v_latest.version;
    END IF;

    IF NEW.action IN ('draft_str', 'edit_str') THEN
        IF v_latest.created_by <> NEW.actor THEN
            RAISE EXCEPTION 'str_author: version % was written by %, not %', v_ver, v_latest.created_by, NEW.actor;
        END IF;
        SELECT max((payload ->> 'draft_version')::int) INTO v_used FROM review_events
        WHERE case_id = NEW.case_id AND action IN ('draft_str', 'edit_str');
        IF v_used IS NOT NULL AND v_ver <= v_used THEN
            RAISE EXCEPTION 'str_version: version % has already been recorded', v_ver;
        END IF;
    ELSE   -- approve_str
        SELECT string_agg(p.problem || ': ' || p.detail, '; ') INTO v_list
        FROM str_draft_problems(NEW.case_id, v_latest.body, v_latest.created_at) p;
        IF v_list IS NOT NULL THEN
            RAISE EXCEPTION 'str_invalid: version % cannot be approved: %', v_ver, v_list;
        END IF;
        IF coalesce(btrim(v_latest.body #>> '{reasons,narrative}'), '') = '' THEN
            RAISE EXCEPTION 'str_narrative: version % has no Reason(s) for Suspicion yet', v_ver;
        END IF;
    END IF;
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS review_event_guard_str ON review_events;
CREATE TRIGGER review_event_guard_str BEFORE INSERT ON review_events
    FOR EACH ROW EXECUTE FUNCTION review_event_guard_str();

-- ---------------------------------------------------------------------------
-- 7. 权限
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT ON str_drafts TO sentry_review;
GRANT USAGE ON SEQUENCE str_drafts_draft_id_seq TO sentry_review;

COMMENT ON TABLE str_drafts IS 'R10b：STR 草稿的各个版本，只追加；写入前由 str_draft_problems 检查';
COMMENT ON FUNCTION str_draft_problems(BIGINT, JSONB, TIMESTAMPTZ) IS 'R10b：STR 草稿的全部检查（唯一的一份规则）';