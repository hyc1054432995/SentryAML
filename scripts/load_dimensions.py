"""
SentryAML - Step 3a: 灌维度表 (banks / entities / accounts)
==========================================================
数据源：HI-Small_accounts.csv（518,581 行，32.96 MB）

一个 CSV 拆成三张表，因为验证过：
  - bank_id   -> bank_name   一对一（30,470 个）
  - entity_id -> entity_name 一对一（166,207 个）
所以把名字重复存在 51.8 万行的 accounts 里是纯浪费。
这个"把重复的属性抽到独立表"的过程叫【规范化 normalization】。

预期结果：
  banks      30,470 行
  entities  166,207 行
  accounts  518,581 行

用法：
    python scripts/load_dimensions.py
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from db import connect, PROJECT_ROOT


# ---------------------------------------------------------------------------
# 读取与拆分
# ---------------------------------------------------------------------------

def load_source(path: Path) -> pd.DataFrame:
    """读账户 CSV。32 MB，可以整表读入，不用分块。"""
    print(f"读取 {path.name} ...")
    df = pd.read_csv(path, dtype=str)          # 全部按字符串读，避免 ID 被当成数字
    df.columns = [c.strip() for c in df.columns]

    # bank_id 验证过是纯数字、最大 356,303，安全转 int
    df["bank_id"] = df["Bank ID"].astype("int64")

    print(f"  {len(df):,} 行")
    return df


def split_banks(df: pd.DataFrame) -> pd.DataFrame:
    """抽取银行维度：(bank_id, bank_name) 去重。"""
    banks = (
        df[["bank_id", "Bank Name"]]
        .drop_duplicates(subset=["bank_id"])
        .rename(columns={"Bank Name": "bank_name"})
        .sort_values("bank_id")
        .reset_index(drop=True)
    )
    print(f"  banks: {len(banks):,} 行")
    return banks


def split_entities(df: pd.DataFrame) -> pd.DataFrame:
    """
    抽取实体维度：(entity_id, entity_name, entity_type) 去重。

    entity_type 从 entity_name 剥掉尾部的 " #数字" 得到：
        "Corporation #33520"        -> "Corporation"
        "Sole Proprietorship #50438"-> "Sole Proprietorship"

    这里用正则而不是数据库生成列，是因为生成列要求表达式 IMMUTABLE，
    复杂逻辑放在 ETL 里更稳、更好调试。
    """
    entities = (
        df[["Entity ID", "Entity Name"]]
        .drop_duplicates(subset=["Entity ID"])
        .rename(columns={"Entity ID": "entity_id", "Entity Name": "entity_name"})
        .reset_index(drop=True)
    )
    entities["entity_type"] = (
        entities["entity_name"].str.replace(r"\s*#\d+\s*$", "", regex=True).str.strip()
    )

    print(f"  entities: {len(entities):,} 行")
    print("  实体类型分布:")
    for t, c in entities["entity_type"].value_counts().items():
        print(f"    {t:<22} {c:>8,}")
    return entities


def split_accounts(df: pd.DataFrame) -> pd.DataFrame:
    """抽取账户表：(bank_id, account_number, entity_id)。"""
    accounts = (
        df[["bank_id", "Account Number", "Entity ID"]]
        .rename(columns={"Account Number": "account_number", "Entity ID": "entity_id"})
        .reset_index(drop=True)
    )
    print(f"  accounts: {len(accounts):,} 行")
    return accounts


# ---------------------------------------------------------------------------
# 写入：用 COPY 而不是 INSERT
# ---------------------------------------------------------------------------

def copy_dataframe(cur, table: str, columns: list[str], df: pd.DataFrame) -> int:
    """
    用 COPY ... FROM STDIN 把 DataFrame 灌进表。

    COPY 是 Postgres 的批量导入通道：一次解析、一次计划，
    数据连续流入，比逐条 INSERT 快 50-100 倍。

    psycopg 的 cur.copy() 返回一个上下文管理器，
    write_row() 逐行写入，但底层是攒够一批才发送，所以不慢。
    """
    col_list = ", ".join(columns)
    sql = f"COPY {table} ({col_list}) FROM STDIN"

    t0 = time.perf_counter()
    with cur.copy(sql) as copy:
        # itertuples 比 iterrows 快一个数量级：
        # iterrows 每行都会构造一个 Series 对象，开销很大
        for row in df[columns].itertuples(index=False, name=None):
            copy.write_row(row)
    elapsed = time.perf_counter() - t0

    print(f"  {table:<10} 写入 {len(df):>8,} 行  ({elapsed:.1f}s)")
    return len(df)


# ---------------------------------------------------------------------------
# 校验
# ---------------------------------------------------------------------------

def verify(cur) -> None:
    """灌完立刻核对行数，错了当场发现，不要等到下游才暴露。"""
    print("\n--- 校验 ---")
    expected = {"banks": 30_470, "entities": 166_207, "accounts": 518_581}

    all_ok = True
    for table, exp in expected.items():
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        actual = cur.fetchone()[0]
        ok = actual == exp
        all_ok &= ok
        mark = "OK " if ok else "!! "
        print(f"  {mark}{table:<10} 实际 {actual:>8,}   预期 {exp:>8,}")

    # 一致性检查：accounts 里的外键指向是否都存在。
    # 我们没建外键约束（为了导入性能），改为事后检查。
    cur.execute("""
        SELECT COUNT(*) FROM accounts a
        LEFT JOIN banks b ON a.bank_id = b.bank_id
        WHERE b.bank_id IS NULL
    """)
    orphan_bank = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM accounts a
        LEFT JOIN entities e ON a.entity_id = e.entity_id
        WHERE e.entity_id IS NULL
    """)
    orphan_entity = cur.fetchone()[0]

    print(f"  {'OK ' if orphan_bank == 0 else '!! '}孤儿账户(银行不存在): {orphan_bank}")
    print(f"  {'OK ' if orphan_entity == 0 else '!! '}孤儿账户(实体不存在): {orphan_entity}")

    if not (all_ok and orphan_bank == 0 and orphan_entity == 0):
        raise SystemExit("\n校验未通过，请检查上面标了 !! 的项。")
    print("\n全部通过。")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="灌维度表")
    parser.add_argument(
        "--accounts-csv",
        default=str(PROJECT_ROOT / "data" / "raw" / "HI-Small_accounts.csv"),
    )
    args = parser.parse_args()

    src = load_source(Path(args.accounts_csv))

    print("\n--- 拆分维度 ---")
    banks = split_banks(src)
    entities = split_entities(src)
    accounts = split_accounts(src)

    print("\n--- 写入数据库 ---")
    # with connect() as conn:  代码块正常结束 -> 自动 COMMIT
    #                          抛出异常       -> 自动 ROLLBACK
    # 这就是事务的原子性：三张表要么全部灌好，要么一行都不留。
    with connect() as conn:
        with conn.cursor() as cur:
            # 可重复执行：先清空。
            # TRUNCATE 比 DELETE 快得多——它直接丢弃数据文件，
            # 而 DELETE 要逐行标记删除并写日志。
            # CASCADE 会连带清空依赖它的表（这里没有，写上是习惯）。
            cur.execute("TRUNCATE banks, entities, accounts CASCADE")

            copy_dataframe(cur, "banks", ["bank_id", "bank_name"], banks)
            copy_dataframe(cur, "entities",
                           ["entity_id", "entity_name", "entity_type"], entities)
            copy_dataframe(cur, "accounts",
                           ["bank_id", "account_number", "entity_id"], accounts)

            verify(cur)


if __name__ == "__main__":
    main()