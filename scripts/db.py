"""
SentryAML - 数据库连接helper
============================
所有需要连数据库的脚本都从这里拿连接，避免每个文件重复写连接逻辑。

三种连接：
  connect()         主连接（DATABASE_URL，库的所有者）。建表、写数据、评测脚本读标签都用它。
  connect_agent()   调查工具专用（AGENT_DATABASE_URL，最小权限角色 sentry_agent，020_investigation.sql 创建）。
                    这个角色读不到 txn_norm、alert_labels 等含标签的表，也不能写任何表。
  connect_review()  复核操作专用（REVIEW_DATABASE_URL，最小权限角色 sentry_review，025_review.sql 创建，R10a）。
                    只能读告警、Agent 记录和复核表，只能往 review_events 里追加；读不到标签。
"""

import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

# 项目根目录 = 本文件所在目录的上一级
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# encoding="utf-8-sig" 是必须的：
# PowerShell 的 Out-File -Encoding utf8 会在文件开头写入 BOM（字节顺序标记）。
# 用普通 utf-8 读取时，第一行的变量名会变成 "﻿POSTGRES_USER"，
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


def connect_agent(statement_timeout_ms: int = 15000, autocommit: bool = True) -> psycopg.Connection:
    """调查工具专用连接（最小权限角色）。

    为什么读不到 AGENT_DATABASE_URL 就直接报错、而不是退回用 DATABASE_URL：
      退回去的话，配置漏写时一切照常运行，只是 Agent 悄悄拿到了能读标签的超级权限——
      "数据库层面看不到答案"这道保证就失效了，而且没有任何报错提醒。宁可启动失败。
    同样的理由，连上后再检查一次：如果这个账号是超级用户（比如有人把 URL 写成了主账号），也报错。

    statement_timeout：资金回流追踪是递归查询，遇到交易特别多的账户可能跑很久。
      给每条语句设上限，超时会抛 QueryCanceled，由 Agent 的工具层当作"预期内的错误"返回给模型，
      而不是让整个调查卡死。
    autocommit=True：工具都是只读查询；不开事务的话，一次报错不会让后面的查询都失败
      （psycopg 默认开事务，事务里出错后必须回滚才能继续）。
    """
    dsn = os.getenv("AGENT_DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "读不到 AGENT_DATABASE_URL。调查工具必须用最小权限角色连接，不会退回主账号。"
            "在 .env 里加一行：AGENT_DATABASE_URL=postgresql://sentry_agent:sentry_agent_dev@localhost:端口/库名"
            "（角色由 sql/020_investigation.sql 创建）"
        )
    if dsn == os.getenv("DATABASE_URL"):
        raise RuntimeError("AGENT_DATABASE_URL 和 DATABASE_URL 相同：调查工具不能用主账号连接")
    conn = psycopg.connect(dsn, autocommit=autocommit,
                           options=f"-c statement_timeout={int(statement_timeout_ms)}")
    with conn.cursor() as cur:
        cur.execute("SELECT current_user, rolsuper FROM pg_roles WHERE rolname = current_user")
        user, is_super = cur.fetchone()
    if is_super:
        conn.close()
        raise RuntimeError(f"AGENT_DATABASE_URL 连上的是超级用户 {user}：调查工具必须用 sentry_agent 这类最小权限角色")
    return conn


def _connect_restricted(env_var: str, role_hint: str, sql_file: str, **kwargs) -> psycopg.Connection:
    """最小权限连接的共同检查：变量必须存在、不能等于主连接、连上的不能是超级用户。理由见 connect_agent。"""
    dsn = os.getenv(env_var)
    if not dsn:
        raise RuntimeError(
            f"读不到 {env_var}。这里必须用最小权限角色连接，不会退回主账号。"
            f"在 .env 里加一行：{env_var}=postgresql://{role_hint}:{role_hint}_dev@localhost:端口/库名"
            f"（角色由 sql/{sql_file} 创建）")
    if dsn == os.getenv("DATABASE_URL"):
        raise RuntimeError(f"{env_var} 和 DATABASE_URL 相同：不能用主账号连接")
    conn = psycopg.connect(dsn, **kwargs)
    with conn.cursor() as cur:
        cur.execute("SELECT current_user, rolsuper FROM pg_roles WHERE rolname = current_user")
        user, is_super = cur.fetchone()
    if is_super:
        conn.close()
        raise RuntimeError(f"{env_var} 连上的是超级用户 {user}：必须用 {role_hint} 这类最小权限角色")
    return conn


def connect_review() -> psycopg.Connection:
    """复核操作专用连接（R10a）。每个复核动作是一个独立事务：写入成功就提交，被规则拒绝就整体回滚。"""
    return _connect_restricted("REVIEW_DATABASE_URL", "sentry_review", "025_review.sql")


def check_connection() -> None:
    """连通性自检，顺便打印服务端版本。"""
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
    print(f"连接成功: {version.split(',')[0]}")


if __name__ == "__main__":
    check_connection()