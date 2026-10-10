"""
SentryAML - R11a：登录与会话
============================
网页接口（review_api.py）用到的身份相关的一切都在这里：密码哈希、会话令牌、防跨站请求伪造的令牌、登录失败限流。

流程：
  1. 登录：用户名 + 密码。数据库里只存 argon2id 哈希（reviewer_credentials），用同样的算法核对。
  2. 核对通过：生成一串随机令牌（256 位），放进浏览器的 cookie；数据库只存它的 sha256（review_sessions）。
     之后每个请求浏览器都会自动带上 cookie，服务器算出 sha256 去表里查：没注销、没过期、人还在岗 → 就是这个人。
  3. 注销：把会话标记为已注销。令牌立刻失效（这是"服务器端会话"相对"把身份写进签名 cookie"的好处，D85）。

防跨站请求伪造（CSRF）：会改变数据的请求（POST）除了 cookie，还必须带一个请求头 X-CSRF-Token。
  它由服务器的密钥和会话令牌算出来（HMAC），登录时返回给页面。别的网站能诱导浏览器发请求、浏览器也可能自动带上
  cookie，但别的网站读不到这个值，也就伪造不了请求头。cookie 的 SameSite=Strict 是第一道，这是第二道。

用法（设置密码，只能用主连接，密码从键盘输入、不出现在命令行历史里）：
    python scripts/review_auth.py set-password alice
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import hmac
import os
import secrets
import threading
import time
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from db import connect

PH = PasswordHasher()                       # argon2id，库的默认参数（64 MiB 内存、3 轮），刻意做得慢，抵御暴力破解
_DUMMY_HASH = PH.hash(secrets.token_urlsafe(16))
COOKIE = "sentry_session"
CSRF_HEADER = "X-CSRF-Token"
MIN_PASSWORD = 12
MAX_FAILS = 5                               # 同一账号 15 分钟内输错 5 次，暂停登录 15 分钟
FAIL_WINDOW = 15 * 60


def session_secret() -> bytes:
    """计算 CSRF 令牌用的服务器密钥。没有或太短就拒绝启动——用一个默认值启动，等于所有部署共用同一把钥匙。"""
    s = os.getenv("SESSION_SECRET", "")
    if len(s) < 32:
        raise RuntimeError(
            "SESSION_SECRET 没有设置或太短（至少 32 个字符）。生成一个写进 .env："
            "python -c \"import secrets; print(secrets.token_urlsafe(48))\"")
    return s.encode()


def token_hash(token: str) -> str:
    # 令牌本身是 256 位随机数，猜不出来，所以这里用快速的 sha256 就够了；
    # 密码是人想出来的、可能很弱，才需要 argon2 这种刻意做慢的算法
    return hashlib.sha256(token.encode()).hexdigest()


def csrf_for(token: str, secret: bytes) -> str:
    return hmac.new(secret, b"csrf:" + token.encode(), hashlib.sha256).hexdigest()


def csrf_ok(sent: str | None, token: str, secret: bytes) -> bool:
    # compare_digest：比较时间不随"前几位对了几位"变化，不能靠计时一位一位猜
    return bool(sent) and hmac.compare_digest(sent, csrf_for(token, secret))


# ---------------------------------------------------------------------------
# 登录失败限流（进程内存）
# ---------------------------------------------------------------------------
class Throttle:
    """按账号计数。只在一个进程里有效：部署多个进程时要换成共享的存储（D85 记为局限）。"""

    def __init__(self, max_fails: int = MAX_FAILS, window: int = FAIL_WINDOW):
        self.max_fails, self.window = max_fails, window
        self._fails: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> list[float]:
        xs = [t for t in self._fails.get(key, []) if now - t < self.window]
        self._fails[key] = xs
        return xs

    def blocked_for(self, key: str) -> int:
        """还要等多少秒；0 = 可以尝试。"""
        now = time.time()
        with self._lock:
            xs = self._recent(key, now)
            return int(self.window - (now - xs[0])) + 1 if len(xs) >= self.max_fails else 0

    def fail(self, key: str) -> None:
        with self._lock:
            self._recent(key, time.time()).append(time.time())

    def reset(self, key: str) -> None:
        with self._lock:
            self._fails.pop(key, None)


# ---------------------------------------------------------------------------
# 核对密码、会话
# ---------------------------------------------------------------------------
@dataclass
class Reviewer:
    reviewer_id: str
    name: str
    role: str
    token: str = ""
    expires_at: object = None


def verify_login(cur, reviewer_id: str, password: str) -> Reviewer | None:
    cur.execute("""SELECT r.reviewer_id, r.name, r.role, c.password_hash
                   FROM reviewers r JOIN reviewer_credentials c USING (reviewer_id)
                   WHERE r.reviewer_id = %s AND r.active AND r.role IN ('analyst', 'mlro')""", (reviewer_id,))
    row = cur.fetchone()
    # 账号不存在时也算一遍哈希：否则"账号不存在"立刻返回、"密码错"要等几十毫秒，
    # 从响应时间就能试出哪些账号存在
    try:
        PH.verify(row[3] if row else _DUMMY_HASH, password)
        ok = row is not None
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        ok = False
    return Reviewer(row[0], row[1], row[2]) if ok else None


def create_session(cur, reviewer_id: str) -> tuple[str, object]:
    token = secrets.token_urlsafe(32)
    # 创建时间、到期时间由数据库触发器计算（027）
    cur.execute("INSERT INTO review_sessions (token_sha256, reviewer_id, expires_at) VALUES (%s, %s, now()) "
                "RETURNING expires_at", (token_hash(token), reviewer_id))
    return token, cur.fetchone()[0]


def lookup_session(cur, token: str) -> Reviewer | None:
    """每个请求都查一次：注销、过期、被停用都立刻生效。"""
    cur.execute("""SELECT r.reviewer_id, r.name, r.role, s.expires_at
                   FROM review_sessions s JOIN reviewers r USING (reviewer_id)
                   WHERE s.token_sha256 = %s AND s.revoked_at IS NULL AND s.expires_at > clock_timestamp()
                     AND r.active AND r.role IN ('analyst', 'mlro')""", (token_hash(token),))
    row = cur.fetchone()
    return Reviewer(row[0], row[1], row[2], token, row[3]) if row else None


def revoke_session(cur, token: str) -> None:
    cur.execute("UPDATE review_sessions SET revoked_at = now() WHERE token_sha256 = %s AND revoked_at IS NULL",
                (token_hash(token),))


# ---------------------------------------------------------------------------
# 命令行：设置密码
# ---------------------------------------------------------------------------
def set_password(cur, reviewer_id: str, password: str) -> None:
    if len(password) < MIN_PASSWORD:
        raise ValueError(f"密码至少 {MIN_PASSWORD} 个字符")
    cur.execute("""INSERT INTO reviewer_credentials (reviewer_id, password_hash) VALUES (%s, %s)
                   ON CONFLICT (reviewer_id) DO UPDATE SET password_hash = EXCLUDED.password_hash, updated_at = now()""",
                (reviewer_id, PH.hash(password)))


def cmd_set_password(a) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT role, active FROM reviewers WHERE reviewer_id = %s", (a.reviewer_id,))
        row = cur.fetchone()
        if row is None or row[0] not in ("analyst", "mlro"):
            raise SystemExit(f"没有可以登录的复核员 {a.reviewer_id}（system 账号不能登录）")
        pw = getpass.getpass(f"{a.reviewer_id} 的新密码（至少 {MIN_PASSWORD} 个字符）：")
        if pw != getpass.getpass("再输入一次："):
            raise SystemExit("两次输入不一致")
        try:
            set_password(cur, a.reviewer_id, pw)
        except ValueError as e:
            raise SystemExit(str(e))
        conn.commit()
    print(f"已设置 {a.reviewer_id} 的密码（数据库里只存 argon2id 哈希）")


def main():
    ap = argparse.ArgumentParser(description="复核员登录（R11a）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("set-password")
    p.add_argument("reviewer_id")
    a = ap.parse_args()
    {"set-password": cmd_set_password}[a.cmd](a)


if __name__ == "__main__":
    main()