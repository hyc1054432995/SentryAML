\pset pager off

\echo '=== 1. 这批 1004xxx 账户是什么来头 ==='
SELECT
    LEFT(account_number, 4)                        AS "账户号前缀",
    COUNT(*)                                       AS "账户数",
    COUNT(DISTINCT bank_id)                        AS "涉及银行数",
    SUM(total_txn_count)                           AS "总交易笔数",
    SUM(laundering_txn_count)                      AS "其中洗钱笔数",
    ROUND(100.0*SUM(laundering_txn_count)/SUM(total_txn_count), 4) AS "洗钱占比%"
FROM account_degrees
GROUP BY 1
ORDER BY 3 DESC
LIMIT 10;

\echo ''
\echo '=== 2. 枢纽节点里，洗钱浓度是高还是低 ==='
SELECT
    CASE WHEN is_hub THEN '枢纽' ELSE '普通' END   AS "节点类型",
    COUNT(*)                                       AS "账户数",
    SUM(total_txn_count)                           AS "交易笔数",
    SUM(laundering_txn_count)                      AS "洗钱笔数",
    ROUND(100.0*SUM(laundering_txn_count)/SUM(total_txn_count), 4) AS "洗钱浓度%"
FROM account_degrees
GROUP BY is_hub;

\echo ''
\echo '=== 3. 按度数分档看洗钱浓度（关键）==='
SELECT
    CASE
        WHEN total_cp_count = 1              THEN 'A. 1'
        WHEN total_cp_count <= 3             THEN 'B. 2-3'
        WHEN total_cp_count <= 10            THEN 'C. 4-10'
        WHEN total_cp_count <= 38            THEN 'D. 11-38'
        WHEN total_cp_count <= 200           THEN 'E. 39-200'
        ELSE                                      'F. >200'
    END                                            AS "对手方数档位",
    COUNT(*)                                       AS "账户数",
    SUM(total_txn_count)                           AS "交易笔数",
    SUM(laundering_txn_count)                      AS "洗钱笔数",
    ROUND(100.0*SUM(laundering_txn_count)/SUM(total_txn_count), 4) AS "洗钱浓度%"
FROM account_degrees
GROUP BY 1 ORDER BY 1;

\echo ''
\echo '=== 4. 银行 70 是什么 ==='
SELECT b.bank_id, b.bank_name,
       COUNT(a.account_number)                     AS "账户数"
FROM banks b
LEFT JOIN accounts a ON a.bank_id = b.bank_id
WHERE b.bank_id = 70
GROUP BY 1, 2;