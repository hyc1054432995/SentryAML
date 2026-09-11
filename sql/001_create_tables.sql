-- ============================================================================
-- SentryAML - 001: 建表
-- ============================================================================
-- 执行:
--   docker compose exec -T postgres psql -U sentry -d sentryaml -f /sql/001_create_tables.sql
--
-- 本脚本只建【表结构】，不建索引。
-- 索引放在 003，因为"先灌数据再建索引"比反过来快好几倍。
--
-- 所有设计决定都基于 reports/step1b_schema_facts.json 里验证过的事实，
-- 关键依据写在各字段的注释里。
-- ============================================================================

-- 可重复执行：每次先删干净再建。
-- CASCADE 表示连带删除依赖它的对象（比如视图）。
-- 顺序要和依赖关系相反：先删被依赖的，再删依赖它的。
DROP VIEW  IF EXISTS txn_edges            CASCADE;
DROP TABLE IF EXISTS pattern_transactions CASCADE;
DROP TABLE IF EXISTS laundering_patterns  CASCADE;
DROP TABLE IF EXISTS transactions         CASCADE;
DROP TABLE IF EXISTS accounts             CASCADE;
DROP TABLE IF EXISTS entities             CASCADE;
DROP TABLE IF EXISTS banks                CASCADE;


-- ============================================================================
-- 1. banks —— 银行/分行维度表
-- ============================================================================
-- 事实依据：bank_id -> bank_name 一对一（30,470 个 bank_id）
-- 但反向不成立：468 个银行名对应多个 bank_id，最多的一个名字对应 48 个。
-- 说明 bank_name 是"机构"，bank_id 是"网点/分行"。
-- 所以 bank_name 只能是描述字段，绝不能当键。
-- ============================================================================
CREATE TABLE banks (
    bank_id     INTEGER      NOT NULL,
    -- 实测最长 30 字符，留一倍余量
    bank_name   VARCHAR(64)  NOT NULL,

    CONSTRAINT pk_banks PRIMARY KEY (bank_id)
);

COMMENT ON TABLE  banks           IS '银行/分行维度表';
COMMENT ON COLUMN banks.bank_id   IS '分行级标识，全局唯一';
COMMENT ON COLUMN banks.bank_name IS '机构名称，非唯一（一个机构可对应多个 bank_id）';


-- ============================================================================
-- 2. entities —— 实体（客户）维度表
-- ============================================================================
-- 事实依据：entity_id -> entity_name 一对一（166,207 个实体）
-- entity_name 形如 "Corporation #33520"，前半段是类型、后半段是编号。
-- 反洗钱里"实体"才是真正的客户；一个实体可持有多个账户、跨多家银行
-- （你的数据里 63,362 个实体是跨行的），这正是 structuring 的信号。
-- ============================================================================
CREATE TABLE entities (
    -- 实测最长 9 字符，形如 800C998A0
    entity_id    VARCHAR(16)  NOT NULL,
    entity_name  VARCHAR(64)  NOT NULL,
    -- 从 entity_name 剥掉 " #数字" 得到，由 ETL 填充
    -- 已知取值：Partnership / Corporation / Sole Proprietorship /
    --           Country / Individual / Direct
    entity_type  VARCHAR(32)  NOT NULL,

    CONSTRAINT pk_entities PRIMARY KEY (entity_id)
);

COMMENT ON TABLE  entities             IS '实体（客户）维度表';
COMMENT ON COLUMN entities.entity_type IS '客户类型，AML 风险分层的基础特征';


-- ============================================================================
-- 3. accounts —— 账户表
-- ============================================================================
-- 事实依据：(bank_id, account_number) 有 518,581 个唯一组合 = 总行数，完全唯一。
-- 而裸 account_number 有 16 行重复：同一账户号在不同银行属于不同实体
-- （例：80A7FD400 在 Australia Bank #44 属 Partnership #36506，
--       在 Australia Bank #47 属 Corporation #33736）。
-- 只用 account_number 当主键会把无关实体合并成同一个图节点，
-- 凭空造出不存在的资金通路，且错误会沿多跳路径传播。
-- ============================================================================
CREATE TABLE accounts (
    bank_id         INTEGER      NOT NULL,
    -- 实测最长 9 字符，形如 8000EBD30
    account_number  VARCHAR(16)  NOT NULL,
    entity_id       VARCHAR(16)  NOT NULL,

    -- 复合主键：列顺序决定索引能加速哪些查询（最左前缀原则）。
    -- bank_id 在前，可加速"某银行的所有账户"和"某银行的某账户"，
    -- 但加速不了"所有叫某账户号的记录"——后者我们不需要，
    -- 因为业务上单独一个账户号本来就没有意义。
    CONSTRAINT pk_accounts PRIMARY KEY (bank_id, account_number)

    -- 刻意不建外键到 banks / entities：
    -- 校验已确认 100% 可关联（0 条孤儿记录），
    -- 但外键会让每次插入多一次查表，批量导入慢一个数量级。
    -- 改为导入后跑一次一致性检查（见 004_validate.sql）。
);

COMMENT ON TABLE accounts IS '账户表；主键为 (bank_id, account_number) 复合键';


-- ============================================================================
-- 4. transactions —— 交易事实表（5,078,345 行）
-- ============================================================================
-- 这张表同时是两样东西：
--   (a) 交易流水 —— ML 特征工程的数据源
--   (b) 资金图的邻接表 —— 每行就是一条边，起点终点各是一个 (bank, account)
--
-- 事实依据：
--   - 存在 9 行完全重复（时间戳只精确到分钟导致），无自然主键 -> 用代理主键
--   - 全表零空值 -> 所有列都可以 NOT NULL
--   - 最大金额 1,046,302,363,293.48（16 位有效数字）-> 见下方 NUMERIC 说明
-- ============================================================================
CREATE TABLE transactions (
    -- 代理主键（surrogate key）：数据本身没有任何列能唯一标识一行，
    -- 所以引入一个纯粹为了当主键而存在的自增列。
    -- BIGSERIAL = 8 字节整数 + 自动递增，插入时不用管这一列。
    txn_id             BIGSERIAL    NOT NULL,

    -- TIMESTAMP 不带时区。源数据没有时区信息，强行用 TIMESTAMPTZ
    -- 会让 Postgres 按服务器时区做隐式转换，凭空引入偏移错误。
    -- 数据没有的信息，就不要假装它有。
    txn_ts             TIMESTAMP    NOT NULL,

    from_bank          INTEGER      NOT NULL,
    from_account       VARCHAR(16)  NOT NULL,
    to_bank            INTEGER      NOT NULL,
    to_account         VARCHAR(16)  NOT NULL,

    -- 为什么是 NUMERIC 而不是 DOUBLE PRECISION：
    -- DOUBLE PRECISION 是浮点数，只有约 15.9 位有效数字，
    -- 而你的最大金额是 16 位——正好卡在精度边界，可能丢失分位。
    -- NUMERIC 是精确十进制，没有浮点误差（不会出现 0.1+0.2 != 0.3）。
    -- (24,8) = 总共 24 位数字，其中 8 位在小数点后，
    -- 整数部分留 16 位（比最大值还多 4 个数量级），
    -- 8 位小数足够容纳 Bitcoin 的最小单位。
    -- 代价是运算比浮点慢，但金额精确性优先级更高。
    amount_received    NUMERIC(24,8) NOT NULL,
    currency_received  VARCHAR(24)   NOT NULL,
    amount_paid        NUMERIC(24,8) NOT NULL,
    currency_paid      VARCHAR(24)   NOT NULL,

    -- 已知 7 种取值：Cheque / Credit Card / ACH / Cash / Reinvestment / Wire / Bitcoin
    payment_format     VARCHAR(24)   NOT NULL,

    -- 标签列。保持源数据的 0/1 而不是转成 BOOLEAN，
    -- 因为 ML 侧要直接当数值用，来回转换容易出错。
    is_laundering      SMALLINT      NOT NULL,

    -- ------------------------------------------------------------------
    -- 生成列（GENERATED ALWAYS AS ... STORED）
    -- ------------------------------------------------------------------
    -- 值由 Postgres 根据同一行的其他列自动算出并存下来，你不能手动写入。
    -- 好处：定义只有一处，永远不会和源数据不一致；
    --       可以像普通列一样建索引和查询。
    -- 要求：表达式必须是 IMMUTABLE 的（相同输入永远相同输出），
    --       所以只放简单比较，复杂逻辑留给 ETL。
    -- ------------------------------------------------------------------

    -- 自环：收付方完全相同。591,212 笔（11.64%）。
    -- 其中 Reinvestment 481,056 笔是 100% 自环，纯记账、不是真实转账。
    -- 但另有约 11 万笔自环分布在 ACH/Bitcoin 等方式里，
    -- 且洗钱模式内部确实存在自环交易，所以【不删除，只打标】——
    -- 原始层忠实记录事实，过滤放在下面的视图里。
    is_self_loop       BOOLEAN GENERATED ALWAYS AS (
                           from_bank = to_bank AND from_account = to_account
                       ) STORED,

    -- 跨币种：72,170 笔。汇率换算噪音大，
    -- 数据里"同币种则金额必然相等"，差异纯来自汇率、无手续费建模。
    is_cross_currency  BOOLEAN GENERATED ALWAYS AS (
                           currency_received <> currency_paid
                       ) STORED,

    -- 同分行内部划转。因为 bank_id 是分行级，
    -- 行内划转和跨行清算的风险含义完全不同，这是一个现成的特征。
    is_intra_branch    BOOLEAN GENERATED ALWAYS AS (
                           from_bank = to_bank
                       ) STORED,

    CONSTRAINT pk_transactions PRIMARY KEY (txn_id),

    -- CHECK 约束：插入时校验条件，不满足就拒绝。
    -- 作用是把"我以为数据长这样"变成"数据库保证数据长这样"。
    -- 如果 ETL 有 bug 写进了脏数据，这里会当场报错，
    -- 而不是等到三周后模型效果诡异时才发现。
    CONSTRAINT ck_txn_laundering_flag CHECK (is_laundering IN (0, 1)),
    CONSTRAINT ck_txn_amount_paid     CHECK (amount_paid     >= 0),
    CONSTRAINT ck_txn_amount_received CHECK (amount_received >= 0)
);

COMMENT ON TABLE  transactions               IS '交易事实表；同时是资金图的邻接表';
COMMENT ON COLUMN transactions.txn_id        IS '代理主键；源数据存在 9 行完全重复，无自然主键';
COMMENT ON COLUMN transactions.is_laundering IS '标签列：1=洗钱交易（全表 5,177 笔，占 0.1019%）';
COMMENT ON COLUMN transactions.is_self_loop  IS '生成列：收付方相同，建图时应过滤';


-- ============================================================================
-- 5. laundering_patterns —— 洗钱模式（370 个）
-- ============================================================================
-- 来自 HI-Small_Patterns.txt，格式为
--   BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 10 hops
--   ...若干交易明细...
--   END LAUNDERING ATTEMPT
--
-- 这是本项目最有价值的标注：is_laundering 只说"这笔可疑"，
-- 而它说的是"这几笔交易构成一个完整洗钱链，手法是 XXX"。
-- Agent 的最终输出正是这种结论，且可以用它来验证判得对不对。
-- ============================================================================
CREATE TABLE laundering_patterns (
    pattern_id     SERIAL       NOT NULL,

    -- 8 种：CYCLE / GATHER-SCATTER / BIPARTITE / FAN-OUT /
    --       SCATTER-GATHER / STACK / RANDOM / FAN-IN
    pattern_type   VARCHAR(32)  NOT NULL,

    -- 标题行冒号后的补充说明，如 "Max 16-degree Fan-Out"、"Max 10 hops"
    pattern_note   VARCHAR(128),

    -- 该模式包含的交易笔数（1 ~ 32，中位数 6.5）
    txn_count      INTEGER      NOT NULL,

    CONSTRAINT pk_laundering_patterns PRIMARY KEY (pattern_id),
    CONSTRAINT ck_pattern_txn_count   CHECK (txn_count > 0)
);

COMMENT ON TABLE laundering_patterns IS '洗钱模式标注；370 个模式，共 3,209 笔交易';


-- ============================================================================
-- 6. pattern_transactions —— 模式包含哪些交易
-- ============================================================================
-- 一个模式对多笔交易，是典型的一对多关系，所以单独建表。
--
-- 关键设计：txn_id 可以为空。
-- patterns 文件里的交易需要按 (时间, 收付方, 金额, 币种) 回连到 transactions 表，
-- 但可能匹配失败或匹配到多行（交易表本来就有 9 组完全重复行）。
-- 所以保留原始字段 + 一个匹配状态列，让匹配质量可审计，
-- 而不是匹配不上就静默丢弃。
--
-- 另一个坑：patterns 文件里的银行号带前导零（021174），
-- 而交易表里是纯整数（21174）。ETL 必须统一转 int 再匹配，
-- 否则字符串比对会一条都对不上。
-- ============================================================================
CREATE TABLE pattern_transactions (
    pattern_id     INTEGER       NOT NULL,
    -- 该交易在模式内的顺序（从 1 开始）。
    -- 对 CYCLE 这类有方向的模式，顺序本身就是资金流向，必须保留。
    seq_no         INTEGER       NOT NULL,

    -- 来自 patterns 文件的原始字段（前导零已在 ETL 中转换）
    txn_ts         TIMESTAMP     NOT NULL,
    from_bank      INTEGER       NOT NULL,
    from_account   VARCHAR(16)   NOT NULL,
    to_bank        INTEGER       NOT NULL,
    to_account     VARCHAR(16)   NOT NULL,
    amount_paid    NUMERIC(24,8) NOT NULL,
    currency_paid  VARCHAR(24)   NOT NULL,
    payment_format VARCHAR(24)   NOT NULL,

    -- 回连到 transactions.txn_id；匹配不上时为 NULL
    txn_id         BIGINT,
    -- matched / not_found / ambiguous —— 让匹配失败可见、可统计
    match_status   VARCHAR(16)   NOT NULL DEFAULT 'pending',

    CONSTRAINT pk_pattern_transactions PRIMARY KEY (pattern_id, seq_no),
    CONSTRAINT ck_pattern_match_status
        CHECK (match_status IN ('pending', 'matched', 'not_found', 'ambiguous'))
);

COMMENT ON COLUMN pattern_transactions.seq_no       IS '模式内顺序；CYCLE 等有向模式依赖它还原资金流向';
COMMENT ON COLUMN pattern_transactions.match_status IS '回连结果；让匹配失败可见而非静默丢弃';


-- ============================================================================
-- 7. txn_edges —— 资金图的边（视图）
-- ============================================================================
-- 视图（VIEW）是一个"存起来的查询"，不占额外存储。
-- 每次查它，Postgres 就把下面这段 SQL 展开执行一遍。
--
-- 为什么用视图而不是在 ETL 里直接删掉自环：
--   删掉的数据拿不回来，而视图随时能改。
--   现在觉得该滤掉的东西，两周后做特征工程时可能正好需要。
--   原始层忠实记录，派生层负责加工——这条原则会一直用到项目结束。
--
-- 剔除自环后：5,078,345 -> 4,487,133 条边
-- ============================================================================
CREATE VIEW txn_edges AS
SELECT
    txn_id,
    txn_ts,
    from_bank,
    from_account,
    to_bank,
    to_account,
    amount_paid,
    currency_paid,
    payment_format,
    is_laundering,
    is_cross_currency,
    is_intra_branch
FROM transactions
WHERE NOT is_self_loop;

COMMENT ON VIEW txn_edges IS '资金图的边集合；已剔除自环（Reinvestment 等记账类交易）';


-- ============================================================================
-- 完成
-- ============================================================================
\echo '=== 建表完成，当前表清单： ==='
\dt
\echo ''
\echo '=== 视图： ==='
\dv