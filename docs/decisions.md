## 时间索引方案：BRIN vs B-tree

**背景**：交易表 508 万行，时间窗查询是 Agent 的高频操作
（"该账户过去 7 天的资金流"）。

**初始方案**：txn_ts 上建 BRIN（56 KB），依据是数据按时间顺序导入。

**问题**：测试时优化器不选 BRIN，查询耗时 1,154 ms。

**排查过程**：
1. 首个假设——数据物理顺序不满足 BRIN 前提。
   查 pg_stats：correlation = 0.99937，假设被推翻。
2. 重读执行计划，发现走的是 Index Only Scan + Heap Fetches: 0。
   定位根因：测试用的 COUNT(*) 不需要读任何列值，
   而 txn_ts 恰好在复合索引 (from_bank, from_account, txn_ts) 的尾部，
   触发了零回表的覆盖索引扫描。**测试用例不具代表性。**
3. 用 enable_indexscan=off 强制对比两条路径。

**实测**（2022-09-05 单日，48 万笔）：
| 查询 | 计划 | 耗时 |
|---|---|---|
| COUNT(*)，优化器自选 | Index Only Scan | 2,143 ms |
| COUNT(*)，强制 BRIN | Bitmap + BRIN | 489 ms |
| COUNT(*) + SUM(amount)，自选 | **自动选 BRIN** | **75.6 ms** |

**结论**：
- BRIN 方案正确，保留。56 KB 索引 vs B-tree 约 110 MB，
  真实查询下 75.6 ms。
- 附带发现：在覆盖索引可用的场景下，优化器代价模型会高估
  Bitmap Heap Scan 的随机 IO 成本，选出比最优慢 4.4 倍的计划。
  这类偏差只能靠 EXPLAIN ANALYZE 实测发现。