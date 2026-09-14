"""
SentryAML - Step 3b: 灌交易事实表
=================================
数据源：HI-Small_Trans.csv（5,078,345 行，453 MB）

和维度表不同，这里【不能整表读进 pandas】——453 MB 的 CSV 在内存里
会膨胀到 3-5 GB。改用流式管道：
    读一块 -> 转换 -> COPY 进库 -> 丢掉 -> 读下一块
峰值内存恒定在几百 MB，和文件大小无关。

关键设计：
  1. 金额【全程按字符串处理】，不经过 float，由 Postgres 解析成 NUMERIC，
     精度零损失。
  2. 时间戳只做 "/" -> "-" 的字符串替换，变成无歧义的 ISO 格式。
  3. 生成列(is_self_loop 等)和 txn_id 不出现在 COPY 列表里——
     它们由数据库自己计算/自增，手动写入会直接报错。

预期结果：5,078,345 行，其中洗钱 5,177 笔、自环 591,212 笔

用法：
    python scripts/load_transactions.py
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from db import connect, PROJECT_ROOT

CHUNK_SIZE = 200_000

# 全部按字符串读入：
#   - 账户号本来就是字符串
#   - 金额按字符串读是为了绕开 float，保住精度
#   - bank_id / is_laundering 交给 Postgres 解析成整数，比 pandas 转一遍更省事
SRC_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency",
    "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering",
]

# COPY 的目标列。顺序必须和下面 write_row 的元组顺序严格一致。
# 注意这里【没有】txn_id（BIGSERIAL 自增）和三个生成列。
TARGET_COLUMNS = [
    "txn_ts", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "currency_received",
    "amount_paid", "currency_paid",
    "payment_format", "is_laundering",
]


def transform(chunk: pd.DataFrame) -> pd.DataFrame:
    """把一块原始数据转换成可以直接 COPY 的形态。全部保持字符串。"""
    out = pd.DataFrame(index=chunk.index)

    # "2022/09/01 00:20" -> "2022-09-01 00:20"
    # 斜杠格式在 Postgres 里有歧义(可能被当成 MM/DD 或 DD/MM)，
    # 短横格式是标准 ISO，全世界只有一种解释。
    # 向量化的 str.replace 比 to_datetime + strftime 快好几倍。
    out["txn_ts"] = chunk["Timestamp"].str.replace("/", "-", regex=False)

    out["from_bank"] = chunk["From Bank"]
    out["from_account"] = chunk["Account"]
    out["to_bank"] = chunk["To Bank"]
    out["to_account"] = chunk["Account.1"]      # pandas 对重名列的自动编号

    # 金额原样传递，不转 float。字符串 "3697.34" 进 NUMERIC(24,8) 就是精确的 3697.34
    out["amount_received"] = chunk["Amount Received"]
    out["currency_received"] = chunk["Receiving Currency"]
    out["amount_paid"] = chunk["Amount Paid"]
    out["currency_paid"] = chunk["Payment Currency"]

    out["payment_format"] = chunk["Payment Format"]
    out["is_laundering"] = chunk["Is Laundering"]

    return out


def load(cur, csv_path: Path) -> int:
    col_list = ", ".join(TARGET_COLUMNS)
    sql = f"COPY transactions ({col_list}) FROM STDIN"

    total = 0
    t_start = time.perf_counter()

    reader = pd.read_csv(
        csv_path,
        usecols=SRC_COLUMNS,
        dtype=str,              # 全字符串，避免任何隐式类型转换
        chunksize=CHUNK_SIZE,
        keep_default_na=False,  # 不要把空字符串变成 NaN；已验证全表无空值
    )

    # 整个导入放在【一个 COPY 流】里，比每块开一次 COPY 更快
    with cur.copy(sql) as copy:
        for i, chunk in enumerate(reader, start=1):
            rows = transform(chunk)
            for row in rows.itertuples(index=False, name=None):
                copy.write_row(row)

            total += len(chunk)
            elapsed = time.perf_counter() - t_start
            rate = total / elapsed if elapsed else 0
            print(f"    已写入 {total:>9,} 行  |  {rate:>7,.0f} 行/秒  "
                  f"|  已用 {elapsed:>5.1f}s", end="\r")

    print(f"    已写入 {total:>9,} 行，完成。" + " " * 30)
    return total


def verify(cur) -> None:
    """核对行数和几个关键分布，和 reports/step1_data_profile.json 对照。"""
    print("\n--- 校验 ---")

    checks = [
        ("总行数",        "SELECT COUNT(*) FROM transactions",                        5_078_345),
        ("洗钱交易",      "SELECT COUNT(*) FROM transactions WHERE is_laundering = 1",     5_177),
        ("自环交易",      "SELECT COUNT(*) FROM transactions WHERE is_self_loop",        591_212),
        ("跨币种交易",    "SELECT COUNT(*) FROM transactions WHERE is_cross_currency",    72_170),
        ("图的边(非自环)", "SELECT COUNT(*) FROM txn_edges",                            4_487_133),
    ]

    all_ok = True
    for label, sql, expected in checks:
        cur.execute(sql)
        actual = cur.fetchone()[0]
        ok = actual == expected
        all_ok &= ok
        print(f"  {'OK ' if ok else '!! '}{label:<16} 实际 {actual:>9,}   预期 {expected:>9,}")

    # 时间范围：确认没有因为格式解析错误导致日期错乱
    cur.execute("SELECT MIN(txn_ts), MAX(txn_ts) FROM transactions")
    lo, hi = cur.fetchone()
    print(f"\n  时间范围: {lo}  ~  {hi}")
    print("  (预期: 2022-09-01 00:00:00 ~ 2022-09-18 16:18:00)")

    # 金额精度：看最大值有没有因为浮点转换丢失分位
    cur.execute("SELECT MAX(amount_paid) FROM transactions")
    print(f"  最大金额: {cur.fetchone()[0]:,}")
    print("  (预期: 1046302363293.48000000)")

    # 一致性：交易表的账户是否都在 accounts 表里（我们没建外键，所以事后查）
    cur.execute("""
        SELECT COUNT(*) FROM transactions t
        LEFT JOIN accounts a
          ON t.from_bank = a.bank_id AND t.from_account = a.account_number
        WHERE a.account_number IS NULL
    """)
    orphan = cur.fetchone()[0]
    print(f"  {'OK ' if orphan == 0 else '!! '}孤儿交易(付款账户不存在): {orphan:,}")

    if not all_ok or orphan != 0:
        raise SystemExit("\n校验未通过，请检查标了 !! 的项。")
    print("\n全部通过。")


def main():
    parser = argparse.ArgumentParser(description="灌交易事实表")
    parser.add_argument(
        "--trans-csv",
        default=str(PROJECT_ROOT / "data" / "raw" / "HI-Small_Trans.csv"),
    )
    args = parser.parse_args()

    csv_path = Path(args.trans_csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    size_mb = csv_path.stat().st_size / 1024 / 1024
    print(f"源文件: {csv_path.name}  ({size_mb:.1f} MB)")
    if size_mb < 400:
        raise SystemExit(
            f"文件只有 {size_mb:.1f} MB，预期约 453 MB。"
            "疑似被 Excel 截断过，请重新下载。"
        )

    print("\n--- 导入 ---")
    with connect() as conn:
        with conn.cursor() as cur:
            # synchronous_commit=off: 提交时不等预写日志(WAL)真正落盘。
            # 正常业务里关掉它有丢数据风险，但批量导入可以——
            # 万一崩了整个事务回滚，重跑一遍即可。能省不少 IO 等待。
            # SET LOCAL 表示只对当前事务生效，不影响数据库全局配置。
            cur.execute("SET LOCAL synchronous_commit = off")

            cur.execute("TRUNCATE transactions CASCADE")
            load(cur, csv_path)

            # ANALYZE 更新表的统计信息(行数、各列的值分布)。
            # 查询优化器靠它决定"该走索引还是全表扫描"。
            # 刚灌完数据统计信息还是空的，不 ANALYZE 的话优化器会瞎猜。
            print("\n更新统计信息 (ANALYZE) ...")
            cur.execute("ANALYZE transactions")

            verify(cur)


if __name__ == "__main__":
    main()