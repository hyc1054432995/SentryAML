-- ============================================================================
-- SentryAML - 028：复核时"问法规"的记录（R11c）
-- ============================================================================
-- 复核员在案件页上向法规研究 Agent（R8 选定的 agent-v3，kb_agent.research）提问。
-- 问答全文由 Agent 自己写进 kb_answer_log（R7 起就有：问题、调查日期、来源、每条论断与逐字核验结果、费用），
-- 但那张表不知道"是谁、为哪个案件问的"。这个文件补上这一环：
--
--   review_reg_queries   一行 = 某个复核员在某个案件上问过一次，指向 kb_answer_log 里的那条回答。只追加
--   review_reg_quota_left(actor)   这个人这一小时还能问几次（每小时上限写在 review_config）
--
-- 为什么不写进 review_events：那张表是状态机，每一行都是一次动作和状态转移，触发器都围着"从哪个状态到哪个状态"写；
--   问法规不改变案件状态，塞进去就要给每个触发器加例外。性质不同的记录分开放。
--
-- 规则（触发器强制，与 025–027 一样，任何写入路径都绕不过）：
--   1. 只有在岗的分析员、MLRO 能记录提问；流水线账号 system 不能
--   2. 一条回答只能挂到一个案件上（answer_id 唯一）——不能把别人的回答挂到自己的案件下，冒充"我查过"
--   3. 只能挂刚生成的回答（15 分钟内）——不能翻出日志里的旧回答来挂
--   4. 记录时间由数据库给出，不信任客户端
--
-- 每小时上限的检查在接口里、调用 Agent 之前做（查询 review_reg_quota_left）：钱是在调用 Agent 时花掉的，
--   等到写记录时再拒绝已经晚了——而且那时拒绝，会让一次真实发生过的提问没有记录，审计反而缺了一行。
--   所以触发器不检查上限。
--
-- 依赖 018（kb_answer_log）、025、027。用法：psql ... -f sql/028_review_reg.sql   （可重复执行）
-- 注意：025 开头会 REVOKE sentry_review 的全部表权限再重新授权，重跑 025 之后要按顺序重跑 026、027、028
-- ============================================================================

INSERT INTO review_config (key, value, note) VALUES
    ('reg_queries_per_hour', 20, '每个复核员每小时最多问法规几次（每次约 0.2–0.8 美分）')
ON CONFLICT (key) DO NOTHING;

CREATE TABLE IF NOT EXISTS review_reg_queries (
    query_id    BIGSERIAL PRIMARY KEY,
    case_id     BIGINT NOT NULL REFERENCES review_cases(case_id),
    actor       TEXT NOT NULL REFERENCES reviewers(reviewer_id),
    answer_id   BIGINT NOT NULL UNIQUE REFERENCES kb_answer_log(answer_id),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS review_reg_queries_case ON review_reg_queries (case_id, query_id DESC);
CREATE INDEX IF NOT EXISTS review_reg_queries_actor ON review_reg_queries (actor, created_at);

CREATE OR REPLACE FUNCTION review_reg_query_guard() RETURNS trigger
LANGUAGE plpgsql AS $$
DECLARE
    v_at TIMESTAMPTZ;
BEGIN
    IF NOT EXISTS (SELECT 1 FROM reviewers WHERE reviewer_id = NEW.actor AND active
                   AND role IN ('analyst', 'mlro')) THEN
        RAISE EXCEPTION 'reg_role: "%" cannot record regulation queries', NEW.actor;
    END IF;
    SELECT created_at INTO v_at FROM kb_answer_log WHERE answer_id = NEW.answer_id;
    IF v_at IS NULL THEN
        RAISE EXCEPTION 'reg_answer: answer % does not exist', NEW.answer_id;
    END IF;
    IF v_at < clock_timestamp() - interval '15 minutes' THEN
        RAISE EXCEPTION 'reg_answer: answer % was produced at % (SGT); only an answer produced just now can be recorded',
            NEW.answer_id, to_char(v_at AT TIME ZONE 'Asia/Singapore', 'YYYY-MM-DD HH24:MI');
    END IF;
    NEW.created_at := clock_timestamp();
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS review_reg_query_guard ON review_reg_queries;
CREATE TRIGGER review_reg_query_guard BEFORE INSERT ON review_reg_queries
    FOR EACH ROW EXECUTE FUNCTION review_reg_query_guard();
DROP TRIGGER IF EXISTS review_reg_queries_append_only ON review_reg_queries;
CREATE TRIGGER review_reg_queries_append_only BEFORE UPDATE OR DELETE ON review_reg_queries
    FOR EACH ROW EXECUTE FUNCTION review_append_only();
DROP TRIGGER IF EXISTS review_reg_queries_no_truncate ON review_reg_queries;
CREATE TRIGGER review_reg_queries_no_truncate BEFORE TRUNCATE ON review_reg_queries
    FOR EACH STATEMENT EXECUTE FUNCTION review_append_only();

-- 这一小时还能问几次。"最近 60 分钟"是滑动窗口：不会在整点一下子清零。
-- 计数放在数据库里，服务开几个进程、重启几次都数的是同一份
CREATE OR REPLACE FUNCTION review_reg_quota_left(p_actor TEXT) RETURNS INT
LANGUAGE sql VOLATILE AS $$
    SELECT review_setting('reg_queries_per_hour') - count(*)::int
    FROM review_reg_queries
    WHERE actor = p_actor AND created_at > clock_timestamp() - interval '1 hour'
$$;

-- 权限：复核角色能读写这张表（只能追加），能读问答日志（页面上显示回答全文）。
-- 问答日志本身由 Agent 用主账号写入（R7、R8 起如此），复核角色不能写、不能改
GRANT SELECT, INSERT ON review_reg_queries TO sentry_review;
GRANT USAGE ON SEQUENCE review_reg_queries_query_id_seq TO sentry_review;
GRANT SELECT ON kb_answer_log TO sentry_review;

COMMENT ON TABLE review_reg_queries IS 'R11c：复核员在案件上问法规的记录，指向 kb_answer_log 的回答；只追加';