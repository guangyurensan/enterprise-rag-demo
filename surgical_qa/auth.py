# -*- coding: utf-8 -*-
"""
企业级知识库 - 认证模块（SSO 地基）

设计目标（2026-10-25）：
1. 身份必须由服务端解出，不信任请求体——前端传 user_role 的旧通道作废
2. 审计闭环：每条审计日志带 user_id，能回答"谁"在"何时"查了"什么"
3. SSO 预留：AuthProvider 适配器接口，未来接企业微信/钉钉/LDAP 时
   只需新增一个 Provider 实现，登录签发与校验链路零改动

实现（零第三方依赖）：
- 密码：PBKDF2-HMAC-SHA256（10 万轮 + 随机盐）
- 令牌：JWT HS256（标准头/载荷/签名三段式，8 小时有效期）
- 密钥：环境变量 AUTH_JWT_SECRET 优先，否则自动生成并持久化到
  auth_secret.key（避免重启后已签发令牌全部失效）
- 用户表：auth_users.json（首次运行自动播种 4 个演示账号）

安全边界（诚实声明）：
- auth_users.json 与密钥文件同机存储，适合内网试点；
  公网暴露前应把用户表挪到数据库、密钥进 KMS/环境变量注入
- 令牌可吊销：登出时 jti 进黑名单（本地文件持久化，重启不丢）；
  边界是单副本，多副本部署须换 Redis
"""

import os
import json
import base64
import hashlib
import hmac
import secrets
import time
import logging
import threading
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "auth_users.json")
SECRET_FILE = os.path.join(BASE_DIR, "auth_secret.key")
REVOKED_FILE = os.path.join(BASE_DIR, "auth_revoked.json")

PBKDF2_ROUNDS = 100_000
TOKEN_TTL_SECONDS = 8 * 3600  # 8 小时，覆盖一个工作日


# ============================================================
# 密码散列（PBKDF2，纯标准库）
# ============================================================
def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    """返回 "pbkdf2$rounds$salt_hex$hash_hex" 格式的散列串"""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS
    )
    return f"pbkdf2${PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """常数时间比较，防时序侧信道"""
    try:
        algo, rounds, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"),
            bytes.fromhex(salt_hex), int(rounds),
        )
        return hmac.compare_digest(digest.hex(), hash_hex)
    except (ValueError, AttributeError):
        return False


# ============================================================
# JWT HS256（标准三段式，纯标准库）
# ============================================================
def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _load_secret() -> bytes:
    env = os.environ.get("AUTH_JWT_SECRET")
    if env:
        return env.encode("utf-8")
    if os.path.exists(SECRET_FILE):
        with open(SECRET_FILE, "rb") as f:
            return f.read()
    secret = secrets.token_bytes(32)
    with open(SECRET_FILE, "wb") as f:
        f.write(secret)
    logger.info("[Auth] 已生成并持久化 JWT 密钥: %s", SECRET_FILE)
    return secret


def issue_token(claims: Dict) -> str:
    """签发 JWT。claims 需含 sub；iat/exp 自动补"""
    now = int(time.time())
    payload = {
        **claims,
        "iat": now,
        "exp": now + TOKEN_TTL_SECONDS,
        "jti": secrets.token_hex(8),
    }
    header = _b64url(json.dumps(
        {"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header}.{body}".encode("ascii")
    sig = hmac.new(_load_secret(), signing_input, hashlib.sha256).digest()
    return f"{header}.{body}.{_b64url(sig)}"


def verify_token(token: str) -> Optional[Dict]:
    """校验 JWT，返回 claims；无效/过期/被篡改返回 None（不抛异常）"""
    if not isinstance(token, str) or not token:
        return None
    try:
        header, body, sig = token.split(".")
        signing_input = f"{header}.{body}".encode("ascii")
        expect = hmac.new(_load_secret(), signing_input, hashlib.sha256).digest()
        if not hmac.compare_digest(expect, _b64url_decode(sig)):
            return None
        claims = json.loads(_b64url_decode(body))
        if claims.get("exp", 0) < time.time():
            return None
        return claims
    except (ValueError, json.JSONDecodeError):
        return None


# ============================================================
# 令牌吊销黑名单（登出即失效，弥补 JWT 无状态的缺陷）
#
# 存储：内存 + 文件持久化（重启后黑名单不丢）。
# 条目 = {jti: exp}，过期自动清理，文件不会无限膨胀。
# 边界：单副本进程内有效；多副本部署须换 Redis 共享存储。
# ============================================================
_revoked_store: Optional[Dict[str, float]] = None
_revoked_lock = threading.Lock()


def _load_revoked() -> Dict[str, float]:
    global _revoked_store
    if _revoked_store is None:
        if os.path.exists(REVOKED_FILE):
            try:
                with open(REVOKED_FILE, "r", encoding="utf-8") as f:
                    _revoked_store = json.load(f)
            except (json.JSONDecodeError, OSError):
                logger.warning("[Auth] 吊销名单文件损坏，按空名单处理")
                _revoked_store = {}
        else:
            _revoked_store = {}
    return _revoked_store


def _save_revoked(store: Dict[str, float]):
    _purge_expired(store)
    with open(REVOKED_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f)


def _purge_expired(store: Dict[str, float]):
    """清掉已过期条目——过期令牌本来就不通过校验，留着只占地方"""
    now = time.time()
    for jti in [j for j, exp in store.items() if exp < now]:
        del store[jti]


def revoke_token(token: str) -> bool:
    """吊销令牌（登出时调用）。返回 True=吊销成功，False=令牌本就无效"""
    claims = verify_token(token)
    if not claims:
        return False
    with _revoked_lock:
        store = _load_revoked()
        store[claims.get("jti", "")] = claims["exp"]
        _save_revoked(store)
    logger.info("[Auth] 令牌已吊销: user=%s jti=%s",
                claims.get("username"), claims.get("jti"))
    return True


def is_token_revoked(claims: Dict) -> bool:
    """jti 是否在黑名单（调用前须先通过 verify_token）"""
    with _revoked_lock:
        return claims.get("jti", "") in _load_revoked()


def verify_token_active(token: str) -> Optional[Dict]:
    """业务侧统一入口：签名/有效期校验 + 吊销名单检查"""
    claims = verify_token(token)
    if not claims or is_token_revoked(claims):
        return None
    return claims


# ============================================================
# 身份提供者（SSO 适配器接口）
# ============================================================
class Identity:
    """认证成功后的标准化身份（与 UserContext 字段对齐）"""

    def __init__(self, user_id: str, username: str, display_name: str,
                 role: str, groups: List[str]):
        self.user_id = user_id
        self.username = username
        self.display_name = display_name
        self.role = role
        self.groups = groups

    def to_claims(self) -> Dict:
        return {
            "sub": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "role": self.role,
            "groups": self.groups,
        }

    @classmethod
    def from_claims(cls, claims: Dict) -> "Identity":
        return cls(
            user_id=claims.get("sub", ""),
            username=claims.get("username", ""),
            display_name=claims.get("display_name", claims.get("username", "")),
            role=claims.get("role", "employee"),
            groups=claims.get("groups", ["全员"]),
        )


class AuthProvider:
    """SSO 适配器接口。

    接入企业微信/钉钉/LDAP 时实现本接口：authenticate 收到凭据
    （OAuth 码 / 域账号密码），返回 Identity 或 None。
    登录路由、签发、校验、审计链路零改动。
    """

    name = "abstract"

    def authenticate(self, credentials: Dict) -> Optional[Identity]:
        raise NotImplementedError


class LocalPasswordProvider(AuthProvider):
    """本地账号密码 Provider（当前实现，演示/内网试点用）"""

    name = "local"

    def __init__(self, users_file: str = USERS_FILE):
        self.users_file = users_file

    # ---- 用户表 ----
    def _load_users(self) -> Dict:
        if not os.path.exists(self.users_file):
            self._seed_users()
        with open(self.users_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _seed_users(self):
        """首次运行播种 4 个演示账号（密码均为演示用，上线必须改）"""
        demo = [
            ("u-admin", "admin", "系统管理员", "admin",
             ["全员", "管理层", "人事部", "财务部", "法务部"], "admin123"),
            ("u-manager", "manager", "部门经理", "manager",
             ["全员", "管理层"], "manager123"),
            ("u-employee", "employee", "普通员工", "employee",
             ["全员"], "employee123"),
            ("u-guest", "guest", "访客", "guest", ["全员"], "guest123"),
        ]
        users = {}
        for uid, uname, disp, role, groups, pwd in demo:
            users[uname] = {
                "user_id": uid,
                "username": uname,
                "display_name": disp,
                "role": role,
                "groups": groups,
                "password_hash": hash_password(pwd),
                "disabled": False,
            }
        with open(self.users_file, "w", encoding="utf-8") as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        logger.info("[Auth] 已播种 %d 个演示账号 -> %s", len(users), self.users_file)

    def authenticate(self, credentials: Dict) -> Optional[Identity]:
        username = (credentials or {}).get("username", "")
        password = (credentials or {}).get("password", "")
        if not username or not password:
            return None
        users = self._load_users()
        user = users.get(username)
        # 统一走一次散列比较，避免"用户不存在"的快速返回暴露时序差异
        stored = user["password_hash"] if user else hash_password("dummy")
        ok = verify_password(password, stored)
        if not user or not ok or user.get("disabled"):
            return None
        return Identity(
            user_id=user["user_id"], username=user["username"],
            display_name=user["display_name"], role=user["role"],
            groups=user["groups"],
        )


# 模块级单例：未来切 SSO 时替换此行（如 WeComProvider()）
auth_provider = LocalPasswordProvider()
