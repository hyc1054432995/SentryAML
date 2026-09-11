"""
SentryAML - 数据库连接helper
============================
所有需要连数据库的脚本都从这里拿连接，避免每个文件重复写连接逻辑。
"""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

# 项目根目录 = 本文件所在目录的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# encoding="utf-8-sig" 是必须的：
# PowerShell 的 Out-File -Encoding utf8 会在文件开头写入 BOM（字节顺序标记）。
# 用普通 utf-8 读取时，第一行的变量名会变成 "\ufeffPOSTGRES_USER"，
# 导致第一个变量永远读不到——而且不会报错，只会返回 None，非常难排查。
# utf-8-sig 会自动吃掉 BOM。
load_dotenv(PROJECT_ROOT / ".env", encoding="utf-8-sig")


def get_dsn() -> str:
    """返回数据库连接串。DSN = Data Source Name，就是连接地址。"""
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "读不到 DATABASE_URL。检查项目根目录下是否有 .env 文件，"
            "且里面有 DATABASE_URL=postgresql://用户:密码@localhost:端口/库名"
        )
    return dsn


def connect() -> psycopg.Connection:
    """建立数据库连接。用法：with connect() as conn: ..."""
    return psycopg.connect(get_dsn())


def check_connection() -> None:
    """连通性自检，顺便打印服务端版本。"""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
    print(f"连接成功: {version.split(',')[0]}")


if __name__ == "__main__":
    check_connection()