"""
SentryAML - Step 3c: 解析并灌入洗钱模式
=======================================
数据源：HI-Small_Patterns.txt（370 个模式，3,209 笔交易）

文件结构：
    BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
    2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1
    ...
    END LAUNDERING ATTEMPT - FAN-OUT

两个必须处理的格式细节：
  1. 136 个标题行【没有冒号和说明】（BIPARTITE / SCATTER-GATHER / STACK），
     正则里冒号部分必须可选，否则会静默跳过 37% 的模式。
  2. 银行号带前导零（021174 vs 交易表里的 21174），
     必须用 int() 转换，否则后续匹配一条都对不上。

数据行字段顺序和 HI-Small_Trans.csv 完全一致（11 列）。

用法：
    python scripts/load_patterns.py
"""

import argparse
import re
from collections import Counter
from pathlib import Path

from db import connect, PROJECT_ROOT

# 冒号和说明部分用 (?: ... )? 包成可选组
PATTERN_BEGIN = re.compile(
    r"^BEGIN LAUNDERING ATTEMPT\s*-\s*([A-Z\-]+?)\s*(?::\s*(.*))?$"
)
PATTERN_END = re.compile(r"^END LAUNDERING ATTEMPT")

# 数据行的 11 个字段
FIELD_NAMES = [
    "txn_ts", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "currency_received",
    "amount_paid", "currency_paid",
    "payment_format", "is_laundering",
]


def parse_data_line(line: str) -> dict:
    """解析一行交易明细。"""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 11:
        raise ValueError(f"期望 11 个字段，实际 {len(parts)}: {line}")

    row = dict(zip(FIELD_NAMES, parts))

    # "2022/09/01 00:06" -> "2022-09-01 00:06"（无歧义的 ISO 格式）
    row["txn_ts"] = row["txn_ts"].replace("/", "-")

    # 前导零：int("021174") -> 21174，Python 自动忽略前导零
    row["from_bank"] = int(row["from_bank"])
    row["to_bank"] = int(row["to_bank"])

    # 金额保持字符串，交给 Postgres 解析成 NUMERIC，不经过 float
    return row


def parse_file(path: Path) -> list[dict]:
    """
    解析整个文件，返回模式列表。
    每个模式是 {"pattern_type", "pattern_note", "transactions": [...]}。
    """
    print(f"解析 {path.name} ...")

    patterns = []
    current = None
    line_no = 0

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line_no += 1
            line = raw.strip()
            if not line:
                continue

            m = PATTERN_BEGIN.match(line)
            if m:
                if current is not None:
                    raise ValueError(f"第 {line_no} 行：上一个模式没有 END 就开始了新的")
                pattern_type, note = m.group(1), m.group(2)
                current = {
                    "pattern_type": pattern_type,
                    "pattern_note": note.strip() if note else None,
                    "transactions": [],
                }
                continue

            if PATTERN_END.match(line):
                if current is None:
                    raise ValueError(f"第 {line_no} 行：出现 END 但没有对应的 BEGIN")
                if not current["transactions"]:
                    raise ValueError(f"第 {line_no} 行：模式内没有任何交易")
                patterns.append(current)
                current = None
                continue

            if current is None:
                raise ValueError(f"第 {line_no} 行：数据行出现在任何模式之外: {line}")

            current["transactions"].append(parse_data_line(line))

    if current is not None:
        raise ValueError("文件结束时仍有未闭合的模式")

    # --- 解析结果自检 ---
    type_counter = Counter(p["pattern_type"] for p in patterns)
    total_txns = sum(len(p["transactions"]) for p in patterns)
    with_note = sum(1 for p in patterns if p["pattern_note"])

    print(f"  模式数: {len(patterns):,}   交易数: {total_txns:,}")
    print(f"  有说明的标题: {with_note}   无说明的标题: {len(patterns) - with_note}")
    print("  类型分布:")
    for t, c in type_counter.most_common():
        print(f"    {t:<16} {c:>4}")

    if len(patterns) != 370 or total_txns != 3209:
        raise SystemExit(
            f"\n解析结果不符预期（应为 370 个模式 / 3,209 笔交易），请检查解析逻辑。"
        )

    return patterns


def load(cur, patterns: list[dict]) -> None:
    """写入 laundering_patterns 和 pattern_transactions。"""
    print("\n--- 写入数据库 ---")

    cur.execute("TRUNCATE laundering_patterns, pattern_transactions CASCADE")
    # TRUNCATE 不会重置 SERIAL 的计数器，重复执行会让 pattern_id 从上次的值继续。
    # RESTART IDENTITY 才会归零。这里显式重置，保证可重复执行结果一致。
    cur.execute("ALTER SEQUENCE laundering_patterns_pattern_id_seq RESTART WITH 1")

    # --- 1. 模式主表。用 RETURNING 拿回数据库生成的自增 id ---
    pattern_ids = []
    for p in patterns:
        cur.execute(
            """
            INSERT INTO laundering_patterns (pattern_type, pattern_note, txn_count)
            VALUES (%s, %s, %s)
            RETURNING pattern_id
            """,
            (p["pattern_type"], p["pattern_note"], len(p["transactions"])),
        )
        pattern_ids.append(cur.fetchone()[0])

    print(f"  laundering_patterns  写入 {len(pattern_ids):>5,} 行")

    # --- 2. 明细表。只有 3,209 行，用 COPY 也行，但逐条 INSERT 更好读 ---
    detail_columns = [
        "pattern_id", "seq_no", "txn_ts",
        "from_bank", "from_account", "to_bank", "to_account",
        "amount_paid", "currency_paid", "payment_format",
    ]
    sql = f"COPY pattern_transactions ({', '.join(detail_columns)}) FROM STDIN"

    n_details = 0
    with cur.copy(sql) as copy:
        for pid, p in zip(pattern_ids, patterns):
            # seq_no 从 1 开始，按文件里的出现顺序。
            # 【必须独立存储，不能靠时间排序还原】——
            # 实测存在同一模式内后一行时间早于前一行的情况。
            for seq, t in enumerate(p["transactions"], start=1):
                copy.write_row((
                    pid, seq, t["txn_ts"],
                    t["from_bank"], t["from_account"],
                    t["to_bank"], t["to_account"],
                    t["amount_paid"], t["currency_paid"], t["payment_format"],
                ))
                n_details += 1

    print(f"  pattern_transactions 写入 {n_details:>5,} 行")


def verify(cur) -> None:
    print("\n--- 校验 ---")

    cur.execute("SELECT COUNT(*) FROM laundering_patterns")
    n_patterns = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM pattern_transactions")
    n_details = cur.fetchone()[0]

    print(f"  {'OK ' if n_patterns == 370 else '!! '}模式数   {n_patterns:>5,}  预期   370")
    print(f"  {'OK ' if n_details == 3209 else '!! '}明细数   {n_details:>5,}  预期 3,209")

    # txn_count 和实际明细行数是否一致（防止解析时漏行）
    cur.execute("""
        SELECT COUNT(*) FROM laundering_patterns p
        WHERE p.txn_count <> (
            SELECT COUNT(*) FROM pattern_transactions d WHERE d.pattern_id = p.pattern_id
        )
    """)
    mismatch = cur.fetchone()[0]
    print(f"  {'OK ' if mismatch == 0 else '!! '}txn_count 与明细数不符的模式: {mismatch}")

    # 各类型的模式数与交易数
    cur.execute("""
        SELECT p.pattern_type,
               COUNT(DISTINCT p.pattern_id) AS n_patterns,
               COUNT(d.*)                   AS n_txns,
               ROUND(AVG(p.txn_count), 1)   AS avg_size
        FROM laundering_patterns p
        JOIN pattern_transactions d ON d.pattern_id = p.pattern_id
        GROUP BY p.pattern_type
        ORDER BY n_patterns DESC
    """)
    print("\n  类型分布:")
    print(f"    {'类型':<18}{'模式数':>8}{'交易数':>8}{'平均长度':>10}")
    for row in cur.fetchall():
        print(f"    {row[0]:<18}{row[1]:>8}{row[2]:>8}{row[3]:>10}")

    if n_patterns != 370 or n_details != 3209 or mismatch != 0:
        raise SystemExit("\n校验未通过。")
    print("\n全部通过。")


def main():
    parser = argparse.ArgumentParser(description="解析并灌入洗钱模式")
    parser.add_argument(
        "--patterns-file",
        default=str(PROJECT_ROOT / "data" / "raw" / "HI-Small_Patterns.txt"),
    )
    args = parser.parse_args()

    path = Path(args.patterns_file)
    if not path.exists():
        raise FileNotFoundError(path)

    patterns = parse_file(path)

    with connect() as conn:
        with conn.cursor() as cur:
            load(cur, patterns)
            verify(cur)


if __name__ == "__main__":
    main()