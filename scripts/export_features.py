"""
SentryAML - M3 第一步：把特征表导出为 Parquet
=============================================
从 Postgres 的 feat_txn 表分块读取，按 train / val / test 分别写成 Parquet 文件：
    data/features/train.parquet
    data/features/val.parquet
    data/features/test.parquet

三个技术点：
  1. 服务器端游标（命名游标）：数据库保留查询结果，Python 每次只取一块，
     峰值内存与总行数无关。
  2. 降精度：计数 int64 -> int32，金额 float64 -> float32，体积和内存减半。
     这里的金额是【特征】，模型只关心大小关系，7 位有效数字足够；
     与 D06（记账必须精确）不矛盾 —— 同一数据在不同用途下精度要求不同。
  3. 固定 schema：写之前明确每一列的类型，不让 pyarrow 按块推断。
     否则某一块里某列恰好全是空值时，会被推断成"空类型"，与其他块对不上而报错。

用法：
    python scripts/export_features.py
"""

import argparse
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from db import connect, PROJECT_ROOT
from feature_spec import (
    ALL_COLS, CAT_FEATURES, LABEL_COL, META_COLS, NUM_FEATURES, is_amount_col,
)

CHUNK_ROWS = 100_000
SPLITS = ["train", "val", "test"]


# ---------------------------------------------------------------------------
# schema：每一列的类型都写死
# ---------------------------------------------------------------------------
def build_schema() -> pa.Schema:
    fields = []
    for col in ALL_COLS:
        if col == "txn_id":
            t = pa.int64()
        elif col == "txn_ts":
            t = pa.timestamp("s")
        elif col in ("from_bank", "to_bank"):
            t = pa.int32()
        elif col in ("split", "from_account", "to_account", "pattern_type"):
            t = pa.string()
        elif col == LABEL_COL:
            t = pa.int8()
        elif col in CAT_FEATURES:
            # 字典编码：几百万行里只有几种取值，存一张字典 + 小整数下标
            t = pa.dictionary(pa.int8(), pa.string())
        elif col in NUM_FEATURES:
            t = pa.float32() if is_amount_col(col) else pa.int32()
        else:
            raise ValueError(f"未知列：{col}，请在 feature_spec.py 中登记")
        fields.append(pa.field(col, t, nullable=True))
    return pa.schema(fields)


def rows_to_table(rows: list[tuple], schema: pa.Schema) -> pa.Table:
    """把一块元组（行）转成列式的 pyarrow Table。"""
    columns = list(zip(*rows))            # 行转列
    arrays = []
    for field, values in zip(schema, columns):
        if pa.types.is_dictionary(field.type):
            # 先建普通字符串数组，再做字典编码
            arr = pa.array(values, type=pa.string()).dictionary_encode()
            arr = arr.cast(field.type)
        else:
            arr = pa.array(values, type=field.type)
        arrays.append(arr)
    return pa.Table.from_arrays(arrays, schema=schema)


# ---------------------------------------------------------------------------
# 导出一个切分
# ---------------------------------------------------------------------------
def export_split(conn, split: str, out_dir: Path, schema: pa.Schema) -> tuple[int, int]:
    out_path = out_dir / f"{split}.parquet"
    col_sql = ", ".join(ALL_COLS)
    sql = f"SELECT {col_sql} FROM feat_txn WHERE split = %s ORDER BY txn_id"

    n_rows = 0
    n_pos = 0
    t0 = time.perf_counter()

    # 命名游标 = 服务器端游标：结果留在数据库，fetchmany 每次只取一块
    with conn.cursor(name=f"export_{split}") as cur:
        cur.itersize = CHUNK_ROWS
        cur.execute(sql, (split,))

        with pq.ParquetWriter(out_path, schema, compression="zstd") as writer:
            while True:
                rows = cur.fetchmany(CHUNK_ROWS)
                if not rows:
                    break
                table = rows_to_table(rows, schema)
                writer.write_table(table)

                n_rows += table.num_rows
                n_pos += int(pc.sum(table[LABEL_COL]).as_py() or 0)
                elapsed = time.perf_counter() - t0
                print(f"    {split:<5} 已写 {n_rows:>9,} 行  "
                      f"({n_rows / elapsed:>8,.0f} 行/秒)", end="\r")

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"    {split:<5} 完成 {n_rows:>9,} 行，洗钱 {n_pos:>5,}，"
          f"文件 {size_mb:6.1f} MB，用时 {time.perf_counter() - t0:5.1f}s")
    return n_rows, n_pos


# ---------------------------------------------------------------------------
# 校验：读回文件，和数据库逐项对比
# ---------------------------------------------------------------------------
def verify(conn, out_dir: Path) -> None:
    print("\n--- 校验：文件 vs 数据库 ---")
    with conn.cursor() as cur:
        cur.execute("SELECT split, COUNT(*), SUM(label) FROM feat_txn GROUP BY split")
        db = {s: (int(n), int(p)) for s, n, p in cur.fetchall()}

    all_ok = True
    for split in SPLITS:
        # 只读 label 一列 —— 列式存储的好处：不用把其他 60 列读进来
        t = pq.read_table(out_dir / f"{split}.parquet", columns=[LABEL_COL])
        n_file = t.num_rows
        p_file = int(pc.sum(t[LABEL_COL]).as_py() or 0)
        n_db, p_db = db[split]
        ok = (n_file == n_db) and (p_file == p_db)
        all_ok &= ok
        print(f"  {'OK ' if ok else '!! '}{split:<5} 文件 {n_file:>9,} 行 / {p_file:>5,} 洗钱   "
              f"数据库 {n_db:>9,} 行 / {p_db:>5,} 洗钱")

    # 检查 schema：读回来的类型和写进去的一致
    schema = pq.read_schema(out_dir / "train.parquet")
    print(f"\n  列数：{len(schema)}（元数据 {len(META_COLS)} + 标签 1 + "
          f"分类特征 {len(CAT_FEATURES)} + 数值特征 {len(NUM_FEATURES)}）")

    if not all_ok:
        raise SystemExit("\n校验未通过。")
    print("\n全部通过。")


def main():
    parser = argparse.ArgumentParser(description="导出特征表为 Parquet")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "data" / "features"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema = build_schema()

    print(f"输出目录：{out_dir}")
    print(f"每块 {CHUNK_ROWS:,} 行\n")

    with connect() as conn:
        for split in SPLITS:
            export_split(conn, split, out_dir, schema)
        verify(conn, out_dir)


if __name__ == "__main__":
    main()