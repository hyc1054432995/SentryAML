"""
SentryAML - Step 1b: schema 设计前的事实校验
============================================
目的：把写 DDL 之前必须确认的事实一次查清。本脚本【只读不写】。

每一项检查都对应一个具体的表设计决定，输出里会直接给出建议。

内存策略：
  - 交易表全程分块读取
  - 重复行检测用 64 位哈希 + numpy 数组（5M 行仅占约 40 MB），
    而不是把整行存进 set（那样要好几 GB）

用法：
    python scripts/validate_schema_facts.py --data-dir "D:/NTU/SentryAML/data/raw"
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

CHUNK_SIZE = 200_000

TRANS_DTYPES = {
    "From Bank": "int32",
    "Account": "str",
    "To Bank": "int32",
    "Account.1": "str",
    "Amount Received": "float64",
    "Receiving Currency": "category",
    "Amount Paid": "float64",
    "Payment Currency": "category",
    "Payment Format": "category",
    "Is Laundering": "int8",
}

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


# ---------------------------------------------------------------------------
# A. 账户表：复合主键可行性 + 维度表拆分判断
# ---------------------------------------------------------------------------

def check_accounts(path: Path) -> dict:
    print(f"\n[A] 校验账户表: {path.name}")

    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    n_rows = len(df)

    # --- A1. 复合主键 (bank_id, account_number) 是否唯一 ---
    composite_dupes = df.duplicated(subset=["Bank ID", "Account Number"], keep=False)
    n_composite_unique = df.groupby(["Bank ID", "Account Number"]).ngroups

    dupe_examples = []
    if composite_dupes.any():
        dupe_examples = (
            df[composite_dupes]
            .sort_values(["Bank ID", "Account Number"])
            .head(6)
            .to_dict("records")
        )

    # --- A2. 裸账户号重复的情况长什么样（上一轮说差 8 个，看看具体是什么）---
    bare_dupes = df.duplicated(subset=["Account Number"], keep=False)
    bare_dupe_examples = (
        df[bare_dupes].sort_values("Account Number").head(10).to_dict("records")
        if bare_dupes.any() else []
    )

    # --- A3. entity_id -> entity_name 是否一对一（决定要不要拆 entities 表）---
    names_per_entity = df.groupby("Entity ID")["Entity Name"].nunique()
    entity_one_to_one = bool((names_per_entity == 1).all())

    # --- A4. bank_id -> bank_name 是否一对一（决定要不要拆 banks 表）---
    names_per_bank = df.groupby("Bank ID")["Bank Name"].nunique()
    bank_one_to_one = bool((names_per_bank == 1).all())
    # 反向：同一个银行名对应几个 bank_id（截图里 National Bank of Harrisburg 出现过两个）
    ids_per_bank_name = df.groupby("Bank Name")["Bank ID"].nunique()

    # --- A5. 字段长度（决定 VARCHAR(n) 的 n）---
    lengths = {
        col: int(df[col].astype(str).str.len().max())
        for col in ["Bank ID", "Account Number", "Entity ID", "Entity Name", "Bank Name"]
    }

    # --- A6. 空值 ---
    nulls = {col: int(df[col].isna().sum()) for col in df.columns}

    # --- A7. bank_id 能否安全转成 INTEGER ---
    bank_id_numeric = bool(df["Bank ID"].str.fullmatch(r"\d+").all())
    max_bank_id = int(df["Bank ID"].astype("int64").max()) if bank_id_numeric else None

    return {
        "总行数": n_rows,
        "A1_复合键唯一组合数": n_composite_unique,
        "A1_复合键是否唯一": n_composite_unique == n_rows,
        "A1_复合键重复行数": int(composite_dupes.sum()),
        "A1_重复样例": dupe_examples,
        "A2_裸账户号重复行数": int(bare_dupes.sum()),
        "A2_裸账户号重复样例": bare_dupe_examples,
        "A3_entity_id到name是否一对一": entity_one_to_one,
        "A4_bank_id到name是否一对一": bank_one_to_one,
        "A4_同名银行对应多个ID的数量": int((ids_per_bank_name > 1).sum()),
        "A4_单个银行名最多对应几个ID": int(ids_per_bank_name.max()),
        "A5_各列最大字符长度": lengths,
        "A6_各列空值数": nulls,
        "A7_bank_id是否纯数字": bank_id_numeric,
        "A7_bank_id最大值": max_bank_id,
    }


# ---------------------------------------------------------------------------
# B. 交易表：重复行、孤儿记录、字段边界
# ---------------------------------------------------------------------------

def check_transactions(trans_path: Path, accounts_path: Path) -> dict:
    print(f"\n[B] 校验交易表: {trans_path.name}")

    # 先把账户表的 (bank_id, account_number) 装进 set，用于孤儿检测
    acc = pd.read_csv(accounts_path, dtype=str)
    acc.columns = [c.strip() for c in acc.columns]
    known_accounts = set(zip(acc["Bank ID"].astype("int64").tolist(),
                             acc["Account Number"].tolist()))
    print(f"    账户表已加载 {len(known_accounts):,} 个 (bank, account)")
    del acc

    total_rows = 0
    orphan_from = 0
    orphan_to = 0
    orphan_either = 0

    null_counts = Counter()
    max_acct_len = 0
    max_amount = 0.0
    min_amount = float("inf")
    max_decimals = 0
    negative_amounts = 0
    zero_amounts = 0

    # 自环 × 支付方式 交叉表（上一轮欠的）
    selfloop_by_format = Counter()
    nonselfloop_by_format = Counter()

    row_hashes = []  # 每块一个 uint64 数组，最后拼起来

    reader = pd.read_csv(trans_path, dtype=TRANS_DTYPES, chunksize=CHUNK_SIZE, low_memory=False)

    for i, chunk in enumerate(reader, start=1):
        chunk = chunk.rename(columns=TRANS_RENAME)
        n = len(chunk)
        total_rows += n

        # --- B1. 整行哈希，用于重复检测 ---
        # hash_pandas_object 对每行算一个 uint64；index=False 保证不把行号算进去
        row_hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype=np.uint64))

        # --- B2. 孤儿记录检测 ---
        from_pairs = list(zip(chunk["from_bank"].tolist(), chunk["from_account"].tolist()))
        to_pairs = list(zip(chunk["to_bank"].tolist(), chunk["to_account"].tolist()))
        f_missing = np.fromiter((p not in known_accounts for p in from_pairs), dtype=bool, count=n)
        t_missing = np.fromiter((p not in known_accounts for p in to_pairs), dtype=bool, count=n)
        orphan_from += int(f_missing.sum())
        orphan_to += int(t_missing.sum())
        orphan_either += int((f_missing | t_missing).sum())

        # --- B3. 空值 ---
        for col in chunk.columns:
            null_counts[col] += int(chunk[col].isna().sum())

        # --- B4. 字段边界（决定 VARCHAR / NUMERIC 精度）---
        max_acct_len = max(
            max_acct_len,
            int(chunk["from_account"].astype(str).str.len().max()),
            int(chunk["to_account"].astype(str).str.len().max()),
        )
        max_amount = max(max_amount, float(chunk["amount_paid"].max()),
                         float(chunk["amount_received"].max()))
        min_amount = min(min_amount, float(chunk["amount_paid"].min()),
                         float(chunk["amount_received"].min()))
        negative_amounts += int((chunk["amount_paid"] < 0).sum())
        zero_amounts += int((chunk["amount_paid"] == 0).sum())

        # 小数位数：金额是否真的最多 2 位（决定 NUMERIC(x,2) 够不够）
        as_str = chunk["amount_paid"].astype(str)
        dec = as_str.str.split(".").str[-1].str.len().where(as_str.str.contains("."), 0)
        max_decimals = max(max_decimals, int(dec.max()))

        # --- B5. 自环 × 支付方式 ---
        self_loop = ((chunk["from_bank"] == chunk["to_bank"])
                     & (chunk["from_account"] == chunk["to_account"]))
        selfloop_by_format.update(
            chunk.loc[self_loop, "payment_format"].astype(str).value_counts().to_dict())
        nonselfloop_by_format.update(
            chunk.loc[~self_loop, "payment_format"].astype(str).value_counts().to_dict())

        print(f"    已处理 {total_rows:,} 行 (第 {i} 块)", end="\r")

    print(f"    已处理 {total_rows:,} 行，完成。          ")

    # --- B1 收尾：找出重复的哈希 ---
    all_hashes = np.concatenate(row_hashes)
    del row_hashes
    _, counts = np.unique(all_hashes, return_counts=True)
    n_dupe_groups = int((counts > 1).sum())
    n_dupe_rows = int(counts[counts > 1].sum() - n_dupe_groups)  # 多出来的行数
    del all_hashes, counts

    # 自环交叉表整理
    all_formats = set(selfloop_by_format) | set(nonselfloop_by_format)
    crosstab = {
        fmt: {
            "自环": selfloop_by_format.get(fmt, 0),
            "非自环": nonselfloop_by_format.get(fmt, 0),
        }
        for fmt in sorted(all_formats)
    }

    return {
        "总行数": total_rows,
        "B1_完全重复的行数": n_dupe_rows,
        "B1_重复组数": n_dupe_groups,
        "B1_是否需要代理主键": n_dupe_rows > 0,
        "B2_付款账户不在账户表中的交易数": orphan_from,
        "B2_收款账户不在账户表中的交易数": orphan_to,
        "B2_任一方为孤儿的交易数": orphan_either,
        "B2_孤儿占比": f"{orphan_either / total_rows * 100:.4f}%",
        "B3_各列空值数": dict(null_counts),
        "B4_账户号最大长度": max_acct_len,
        "B4_金额最大值": f"{max_amount:,.2f}",
        "B4_金额最小值": f"{min_amount:,.2f}",
        "B4_金额最大小数位": max_decimals,
        "B4_负数金额笔数": negative_amounts,
        "B4_零金额笔数": zero_amounts,
        "B5_自环与支付方式交叉表": crosstab,
    }


# ---------------------------------------------------------------------------
# 建议生成
# ---------------------------------------------------------------------------

def make_recommendations(acc: dict, txn: dict) -> list:
    rec = []

    if acc["A1_复合键是否唯一"]:
        rec.append("accounts 表：可以用 PRIMARY KEY (bank_id, account_number)")
    else:
        rec.append(
            f"accounts 表：复合键仍有 {acc['A1_复合键重复行数']} 行重复，"
            "不能直接做主键，需要加代理主键或先去重（看重复样例判断是否为同一账户的冗余记录）"
        )

    if txn["B1_是否需要代理主键"]:
        rec.append(
            f"transactions 表：存在 {txn['B1_完全重复的行数']} 行完全重复，"
            "必须用 BIGSERIAL 代理主键"
        )
    else:
        rec.append("transactions 表：无完全重复行，但仍建议用 BIGSERIAL 代理主键（列少、通用、便于引用）")

    if txn["B2_任一方为孤儿的交易数"] > 0:
        rec.append(
            f"外键：有 {txn['B2_任一方为孤儿的交易数']} 笔交易的账户不在 accounts 表中"
            "（{}），确认不建外键约束，改为导入后一致性检查".format(txn["B2_孤儿占比"])
        )
    else:
        rec.append("外键：交易表账户完全被 accounts 覆盖，理论上可建外键，但为导入性能仍建议不建")

    n = int(str(txn["B4_账户号最大长度"]))
    rec.append(f"账户号字段：最大长度 {n}，建议 VARCHAR({max(n + 5, 20)})（留余量）")

    if txn["B4_金额最大小数位"] <= 2:
        rec.append("金额字段：最多 2 位小数，建议 NUMERIC(20,2)（精确十进制，适合金额）")
    else:
        rec.append(
            f"金额字段：出现了 {txn['B4_金额最大小数位']} 位小数，"
            "需确认是否为浮点表示误差，再定 NUMERIC 精度"
        )

    if not acc["A4_bank_id到name是否一对一"]:
        rec.append("banks 维度表：bank_id 对应多个名称，需要清洗后再拆表")
    elif acc["A4_同名银行对应多个ID的数量"] > 0:
        rec.append(
            f"banks 维度表：可以拆（bank_id -> name 一对一），"
            f"但有 {acc['A4_同名银行对应多个ID的数量']} 个银行名对应多个 bank_id，"
            "所以银行名不能作为任何键"
        )

    if acc["A3_entity_id到name是否一对一"]:
        rec.append("entities 维度表：entity_id -> name 一对一，可以拆出独立表")

    return rec


def main():
    parser = argparse.ArgumentParser(description="SentryAML schema 事实校验")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="reports/step1b_schema_facts.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    trans_path = data_dir / "HI-Small_Trans.csv"
    accounts_path = data_dir / "HI-Small_accounts.csv"

    for p in (trans_path, accounts_path):
        if not p.exists():
            raise FileNotFoundError(f"找不到文件: {p}")

    acc = check_accounts(accounts_path)
    txn = check_transactions(trans_path, accounts_path)
    rec = make_recommendations(acc, txn)

    report = {"accounts": acc, "transactions": txn, "建议": rec}

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