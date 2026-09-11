"""
SentryAML - Step 1: 数据探查
============================
目的：在设计数据库表结构之前，先把三个原始文件的事实摸清楚。
本脚本【只读不写】，不会修改任何原始数据。

内存策略：
  - 全程分块(chunk)读取交易表，峰值内存约 300-500 MB，与文件大小无关
  - 字符串列声明为 category / str，避免 object dtype 的对象头开销
  - 只累加计数器和集合，不在内存里保留完整 DataFrame

用法：
    python explore_data.py --data-dir "D:/NTU/SentryAML/data/raw"
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

CHUNK_SIZE = 200_000  # 每次读入的行数。内存紧张就调小，比如 100_000

# 显式声明 dtype，这是控制内存的关键。
# 注意：CSV 里有两列都叫 "Account"，pandas 会自动把第二列重命名为 "Account.1"
TRANS_DTYPES = {
    "From Bank": "int32",
    "Account": "str",
    "To Bank": "int32",
    "Account.1": "str",
    "Amount Received": "float64",   # 金额不要用 float32，会丢精度
    "Receiving Currency": "category",
    "Amount Paid": "float64",
    "Payment Currency": "category",
    "Payment Format": "category",
    "Is Laundering": "int8",
}

# 统一的列名映射：一旦重命名，后续代码永远不再出现 "Account.1" 这种无语义列名
TRANS_RENAME = {
    "Timestamp": "timestamp",
    "From Bank": "from_bank",
    "Account": "from_account",
    "To Bank": "to_bank",
    "Account.1": "to_account",
    "Amount Received": "amount_received",
    "Receiving Currency": "currency_received",
    "Amount Paid": "amount_paid",
    "Payment Currency": "currency_paid",
    "Payment Format": "payment_format",
    "Is Laundering": "is_laundering",
}

PATTERN_BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT\s*-\s*([A-Z\-]+)")
PATTERN_END = "END LAUNDERING ATTEMPT"


# ---------------------------------------------------------------------------
# 1. 交易表探查
# ---------------------------------------------------------------------------

def explore_transactions(path: Path) -> dict:
    """分块扫描交易表，累加各类统计量。"""
    print(f"\n[1/3] 扫描交易表: {path.name}")

    total_rows = 0
    laundering_rows = 0
    self_loop_rows = 0                 # 自环：收付方 (bank, account) 完全相同
    self_loop_laundering = 0
    cross_currency_rows = 0            # 收付币种不同
    amount_mismatch_rows = 0           # 收付金额不等

    currency_counter = Counter()
    format_counter = Counter()
    format_laundering_counter = Counter()

    from_accounts = set()              # 用 (bank, account) 复合键
    to_accounts = set()
    bare_account_ids = set()           # 只用账户号，用来检验全局唯一性
    banks = set()

    min_ts, max_ts = None, None
    max_amount = 0.0

    reader = pd.read_csv(
        path,
        dtype=TRANS_DTYPES,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for i, chunk in enumerate(reader, start=1):
        chunk = chunk.rename(columns=TRANS_RENAME)

        # 时间戳格式: 2022/09/01 00:20
        ts = pd.to_datetime(chunk["timestamp"], format="%Y/%m/%d %H:%M", errors="coerce")
        chunk_min, chunk_max = ts.min(), ts.max()
        min_ts = chunk_min if min_ts is None else min(min_ts, chunk_min)
        max_ts = chunk_max if max_ts is None else max(max_ts, chunk_max)

        n = len(chunk)
        total_rows += n

        is_laundering = chunk["is_laundering"] == 1
        laundering_rows += int(is_laundering.sum())

        # --- 自环检测：必须同时比较银行号和账户号 ---
        self_loop = (
            (chunk["from_bank"] == chunk["to_bank"])
            & (chunk["from_account"] == chunk["to_account"])
        )
        self_loop_rows += int(self_loop.sum())
        self_loop_laundering += int((self_loop & is_laundering).sum())

        # --- 跨币种 / 金额不一致 ---
        cross_currency_rows += int(
            (chunk["currency_received"].astype(str) != chunk["currency_paid"].astype(str)).sum()
        )
        amount_mismatch_rows += int(
            (chunk["amount_received"] != chunk["amount_paid"]).sum()
        )

        # --- 类别分布 ---
        currency_counter.update(chunk["currency_paid"].astype(str).value_counts().to_dict())
        format_counter.update(chunk["payment_format"].astype(str).value_counts().to_dict())
        format_laundering_counter.update(
            chunk.loc[is_laundering, "payment_format"].astype(str).value_counts().to_dict()
        )

        # --- 节点集合（复合键 vs 裸账户号）---
        from_accounts.update(zip(chunk["from_bank"].tolist(), chunk["from_account"].tolist()))
        to_accounts.update(zip(chunk["to_bank"].tolist(), chunk["to_account"].tolist()))
        bare_account_ids.update(chunk["from_account"].tolist())
        bare_account_ids.update(chunk["to_account"].tolist())
        banks.update(chunk["from_bank"].tolist())
        banks.update(chunk["to_bank"].tolist())

        max_amount = max(max_amount, float(chunk["amount_paid"].max()))

        print(f"    已处理 {total_rows:,} 行 (第 {i} 块)", end="\r")

    all_nodes = from_accounts | to_accounts

    print(f"    已处理 {total_rows:,} 行，完成。          ")

    return {
        "总交易数": total_rows,
        "洗钱交易数": laundering_rows,
        "洗钱占比": f"{laundering_rows / total_rows * 100:.4f}%",
        "时间范围": f"{min_ts} ~ {max_ts}",
        "时间跨度天数": (max_ts - min_ts).days,
        "自环交易数": self_loop_rows,
        "自环占比": f"{self_loop_rows / total_rows * 100:.2f}%",
        "自环中的洗钱交易数": self_loop_laundering,
        "跨币种交易数": cross_currency_rows,
        "收付金额不等的交易数": amount_mismatch_rows,
        "最大单笔金额": f"{max_amount:,.2f}",
        "唯一银行数": len(banks),
        "唯一账户数(复合键 bank+account)": len(all_nodes),
        "唯一账户号(裸account)": len(bare_account_ids),
        "账户号是否全局唯一": len(all_nodes) == len(bare_account_ids),
        "币种分布": dict(currency_counter.most_common()),
        "支付方式分布": dict(format_counter.most_common()),
        "洗钱交易的支付方式分布": dict(format_laundering_counter.most_common()),
    }


# ---------------------------------------------------------------------------
# 2. 账户表探查
# ---------------------------------------------------------------------------

def explore_accounts(path: Path) -> dict:
    """账户表很小，可以整表读入。重点看实体(Entity)这一层。"""
    print(f"\n[2/3] 扫描账户表: {path.name}")

    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # 每个实体持有几个账户 -> 反洗钱里"一人多账户跨行"是 structuring 的典型信号
    accounts_per_entity = df.groupby("Entity ID")["Account Number"].nunique()
    banks_per_entity = df.groupby("Entity ID")["Bank ID"].nunique()

    # 实体类型：Corporation / Sole Proprietorship / Partnership ...
    entity_type = df["Entity Name"].str.replace(r"\s*#\d+$", "", regex=True)

    return {
        "列名": list(df.columns),
        "账户总数": len(df),
        "唯一账户号数": df["Account Number"].nunique(),
        "账户号是否唯一": bool(df["Account Number"].nunique() == len(df)),
        "唯一实体数": df["Entity ID"].nunique(),
        "唯一银行数": df["Bank ID"].nunique(),
        "每实体平均账户数": round(float(accounts_per_entity.mean()), 2),
        "每实体最多账户数": int(accounts_per_entity.max()),
        "持有多个账户的实体数": int((accounts_per_entity > 1).sum()),
        "跨多家银行开户的实体数": int((banks_per_entity > 1).sum()),
        "跨行实体的最大银行数": int(banks_per_entity.max()),
        "实体类型分布": entity_type.value_counts().to_dict(),
    }


# ---------------------------------------------------------------------------
# 3. 洗钱模式文件探查
# ---------------------------------------------------------------------------

def explore_patterns(path: Path) -> dict:
    """
    解析 BEGIN/END 分隔的模式块。
    注意：这个文件里的银行号带前导零(如 021174)，而交易表里是纯整数(21174)。
    关联时必须统一转 int，否则一条都匹配不上。
    """
    print(f"\n[3/3] 扫描模式文件: {path.name}")

    pattern_counter = Counter()
    txn_per_pattern = []
    total_txns = 0
    current_type = None
    current_count = 0
    leading_zero_examples = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = PATTERN_BEGIN.match(line)
            if m:
                current_type = m.group(1)
                current_count = 0
                continue

            if line.startswith(PATTERN_END):
                if current_type:
                    pattern_counter[current_type] += 1
                    txn_per_pattern.append((current_type, current_count))
                current_type = None
                continue

            if current_type:
                current_count += 1
                total_txns += 1
                parts = line.split(",")
                if len(parts) > 1 and parts[1].startswith("0") and len(leading_zero_examples) < 5:
                    leading_zero_examples.append(parts[1])

    sizes = pd.Series([c for _, c in txn_per_pattern])

    return {
        "模式总数": sum(pattern_counter.values()),
        "模式内交易总数": total_txns,
        "模式类型分布": dict(pattern_counter.most_common()),
        "单个模式交易数_最小": int(sizes.min()),
        "单个模式交易数_最大": int(sizes.max()),
        "单个模式交易数_中位数": float(sizes.median()),
        "带前导零的银行号样例": leading_zero_examples,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SentryAML 数据探查")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="存放三个原始文件的目录，例如 D:/NTU/SentryAML/data/raw",
    )
    parser.add_argument(
        "--out",
        default="reports/step1_data_profile.json",
        help="探查结果输出路径",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    trans_path = data_dir / "HI-Small_Trans.csv"
    accounts_path = data_dir / "HI-Small_accounts.csv"
    patterns_path = data_dir / "HI-Small_Patterns.txt"

    for p in (trans_path, accounts_path, patterns_path):
        if not p.exists():
            raise FileNotFoundError(f"找不到文件: {p}")

    report = {
        "transactions": explore_transactions(trans_path),
        "accounts": explore_accounts(accounts_path),
        "patterns": explore_patterns(patterns_path),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 60)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print("=" * 60)
    print(f"\n完整报告已写入: {out_path.resolve()}")


if __name__ == "__main__":
    main()