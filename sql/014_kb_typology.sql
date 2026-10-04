-- ============================================================================
-- SentryAML - 014：洗钱手法知识图谱（RAG 层 R3）
-- ============================================================================
-- 内容来自 config/typology_kg.yaml，由 scripts/kb_typology_load.py 校验后整体写入。
-- YAML 是唯一的真相来源；这里的表只是它的"可查询形态"，不要手工改表。
--
-- 为什么用"按类型分开的节点表 + 按关系分开的边表"，而不是一张通用的 nodes/edges：
--   图的形状是固定的（手法 → 风险信号 / 数据信号 → 工具），节点只有 4 类、关系只有 3 种。
--   分开建表可以用外键和 CHECK 保证"边的两端一定存在、类型一定对"，
--   通用 nodes/edges 表做不到这一点，只能在应用层检查。
--   图很小（几十个节点），查询是一两跳的 JOIN，用 Neo4j 没有收益（D38）。
--
-- 用法：psql ... -f sql/014_kb_typology.sql   （可重复执行）
-- ============================================================================

-- ---------------- 节点 ----------------
CREATE TABLE IF NOT EXISTS kb_typologies (
    typology_id      TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    dataset_pattern  TEXT UNIQUE,                  -- 对应 IBM 数据集的 pattern_type；数据集之外的手法为 NULL
    stage            TEXT NOT NULL CHECK (stage IN ('placement', 'layering', 'integration')),
    description      TEXT NOT NULL,
    def_family       TEXT NOT NULL,                -- 定义的出处：文件族 + 条款 + 原文
    def_unit_id      TEXT NOT NULL,
    def_item         TEXT,
    def_quote        TEXT NOT NULL,
    def_verified_in  TEXT[] NOT NULL               -- 原文逐字核对通过的文件版本
);

CREATE TABLE IF NOT EXISTS kb_indicators (
    indicator_id     TEXT PRIMARY KEY,
    family           TEXT NOT NULL,                -- 只记文件族，不绑版本；查询时按日期选版本（与 kb_xrefs 一致）
    unit_id          TEXT NOT NULL,
    item             TEXT,                         -- 列表项编号，如 "iv)"、"(b)"
    quote            TEXT NOT NULL,
    verified_in      TEXT[] NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_signals (
    signal_id        TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    features         TEXT[] NOT NULL DEFAULT '{}', -- feat_txn 里的列名；空 = 只能靠工具查
    evidence         TEXT NOT NULL,                -- 我们自己实测的证据（决策编号）
    caution          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_tools (
    tool_id          TEXT PRIMARY KEY,
    status           TEXT NOT NULL CHECK (status IN ('planned', 'implemented')),
    backing          TEXT,                         -- 已有的实现基础（SQL 文件等）
    description      TEXT NOT NULL
);

-- ---------------- 边 ----------------
CREATE TABLE IF NOT EXISTS kb_typology_indicators (
    typology_id   TEXT NOT NULL REFERENCES kb_typologies(typology_id) ON DELETE CASCADE,
    indicator_id  TEXT NOT NULL REFERENCES kb_indicators(indicator_id) ON DELETE CASCADE,
    PRIMARY KEY (typology_id, indicator_id)
);
CREATE TABLE IF NOT EXISTS kb_typology_signals (
    typology_id   TEXT NOT NULL REFERENCES kb_typologies(typology_id) ON DELETE CASCADE,
    signal_id     TEXT NOT NULL REFERENCES kb_signals(signal_id) ON DELETE CASCADE,
    PRIMARY KEY (typology_id, signal_id)
);
CREATE TABLE IF NOT EXISTS kb_signal_tools (
    signal_id     TEXT NOT NULL REFERENCES kb_signals(signal_id) ON DELETE CASCADE,
    tool_id       TEXT NOT NULL REFERENCES kb_tools(tool_id) ON DELETE CASCADE,
    PRIMARY KEY (signal_id, tool_id)
);
CREATE INDEX IF NOT EXISTS kb_typology_signals_sig ON kb_typology_signals (signal_id);

COMMENT ON TABLE kb_typologies IS 'R3 手法知识图谱：洗钱手法节点，定义带原文出处';
COMMENT ON TABLE kb_indicators IS 'R3 手法知识图谱：监管/FATF 风险信号，引用原文且逐字核对';
COMMENT ON TABLE kb_signals    IS 'R3 手法知识图谱：可在我们数据中观测的信号';
COMMENT ON TABLE kb_tools      IS 'R3 手法知识图谱：Agent 调查工具';

-- ----------------------------------------------------------------------------
-- 1. 由信号找手法：Agent 看到告警上亮起的信号，问"这像哪种手法"
--    例：SELECT * FROM kb_typologies_for_signals(ARRAY['s_first_contact','s_pass_through','s_fan_in_cp']);
--
--    排序用"特异度加权"，思路同 TF-IDF 里的 IDF：
--      specificity(信号) = ln(手法总数 / 含有该信号的手法数)
--    首次来往被 5 种手法共用，命中它几乎不能说明是哪一种（ln(11/5) ≈ 0.79）；
--    过桥只属于 gather-scatter，命中它指向性很强（ln(11/1) ≈ 2.40）。
--    score    = 命中信号的特异度之和：证据有多"指向"这种手法
--    coverage = 命中特异度 / 该手法全部信号特异度：这种手法的特征还缺多少没验证（missing 列给出下一步该查什么）
--    注意：这是"像哪种手法"的排序，不是洗钱概率——是否可疑由模型分数和工具查证决定。
-- ----------------------------------------------------------------------------
DROP FUNCTION IF EXISTS kb_typologies_for_signals(TEXT[]);   -- 返回列变化时 OR REPLACE 不够用
CREATE OR REPLACE FUNCTION kb_typologies_for_signals(p_signals TEXT[])
RETURNS TABLE (typology_id TEXT, name TEXT, stage TEXT, matched TEXT[], missing TEXT[],
               score NUMERIC, coverage NUMERIC)
LANGUAGE sql STABLE AS $$
    WITH spec AS (
        SELECT signal_id,
               ln((SELECT COUNT(*) FROM kb_typologies)::numeric / COUNT(*)) AS w
        FROM kb_typology_signals GROUP BY signal_id
    )
    SELECT t.typology_id, t.name, t.stage,
           array_agg(ts.signal_id ORDER BY sp.w DESC) FILTER (WHERE ts.signal_id = ANY(p_signals)),
           array_agg(ts.signal_id ORDER BY sp.w DESC) FILTER (WHERE NOT ts.signal_id = ANY(p_signals)),
           round(SUM(sp.w) FILTER (WHERE ts.signal_id = ANY(p_signals)), 2),
           round(SUM(sp.w) FILTER (WHERE ts.signal_id = ANY(p_signals)) / NULLIF(SUM(sp.w), 0), 2)
    FROM kb_typologies t
    JOIN kb_typology_signals ts ON ts.typology_id = t.typology_id
    JOIN spec sp ON sp.signal_id = ts.signal_id
    GROUP BY t.typology_id, t.name, t.stage
    HAVING COUNT(*) FILTER (WHERE ts.signal_id = ANY(p_signals)) > 0
    ORDER BY 6 DESC, 7 DESC, 1
$$;

-- ----------------------------------------------------------------------------
-- 2. 手法卡片：一种手法的全部监管依据，按调查日期取当时有效的版本原文
--    例：SELECT * FROM kb_typology_citations('gather_scatter', DATE '2025-03-01');
--    quote_in_force = false 表示：这句话在那一天有效的版本里已经找不到了（条文被修订或删除），
--    Agent 不能再把它当作当时的监管依据引用。
-- ----------------------------------------------------------------------------
DROP FUNCTION IF EXISTS kb_typology_citations(TEXT, DATE);
CREATE OR REPLACE FUNCTION kb_typology_citations(p_typology TEXT, p_as_of DATE)
RETURNS TABLE (indicator_id TEXT, doc_id TEXT, unit_id TEXT, item TEXT, quote TEXT,
               quote_in_force BOOLEAN, unit_text TEXT)
LANGUAGE sql STABLE AS $$
    SELECT i.indicator_id, d.doc_id, i.unit_id, i.item, i.quote,
           u.unit_id IS NOT NULL AND
             position(lower(regexp_replace(translate(i.quote, '“”‘’', '""'''''), '\s+', ' ', 'g'))
                      IN lower(regexp_replace(translate(regexp_replace(u.text, '\[\^\d+\]', '', 'g'),
                                                        '“”‘’', '""'''''), '\s+', ' ', 'g'))) > 0,
           u.text
    FROM kb_typology_indicators ti
    JOIN kb_indicators i ON i.indicator_id = ti.indicator_id
    LEFT JOIN kb_documents_as_of(p_as_of) d ON d.family = i.family
    LEFT JOIN kb_units u ON u.doc_id = d.doc_id AND u.unit_id = i.unit_id
    WHERE ti.typology_id = p_typology
    ORDER BY i.family, i.unit_id, i.item
$$;

-- ----------------------------------------------------------------------------
-- 3. 信号 → 工具：Agent 决定"要验证这个信号，该调哪个工具"
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW kb_signal_tool_plan AS
SELECT s.signal_id, s.name AS signal_name, s.features, t.tool_id, t.status, t.description AS tool_description
FROM kb_signals s
JOIN kb_signal_tools st ON st.signal_id = s.signal_id
JOIN kb_tools t ON t.tool_id = st.tool_id;