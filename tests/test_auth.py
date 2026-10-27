# -*- coding: utf-8 -*-
"""
鉴权模块测试

覆盖:
- 口令散列：PBKDF2 往返、错误口令、损坏散列串
- JWT：签发/校验、篡改载荷/篡改签名、过期、jti 唯一
- 吊销：登出后 verify_token_active 失效，但签名本身仍合法（两层分离）
- 身份提供者：本地账号登录、错误口令、不存在用户、禁用账号、文件持久化
- Identity <-> claims 往返

所有用例都在 tmp_path 下跑，不碰仓库里的 auth_users.json / auth_secret.key。
"""

import hashlib
import hmac
import json
import time

import pytest

import auth


@pytest.fixture
def env(tmp_path, monkeypatch):
    """把密钥、吊销名单、用户表全部指向临时目录"""
    monkeypatch.setattr(auth, "SECRET_FILE", str(tmp_path / "secret.key"))
    monkeypatch.setattr(auth, "REVOKED_FILE", str(tmp_path / "revoked.json"))
    monkeypatch.delenv("AUTH_JWT_SECRET", raising=False)
    auth._revoked_store = None
    yield tmp_path
    auth._revoked_store = None


def _claims(**extra):
    base = {
        "sub": "u-admin",
        "username": "admin",
        "display_name": "管理员",
        "role": "admin",
        "groups": ["全员", "管理层"],
    }
    base.update(extra)
    return base


# ============================================================
# 口令散列
# ============================================================
class TestPasswordHash:
    def test_roundtrip(self):
        stored = auth.hash_password("correct horse")
        assert stored != "correct horse"
        assert auth.verify_password("correct horse", stored) is True
        assert auth.verify_password("wrong horse", stored) is False

    def test_salt_is_random(self):
        a = auth.hash_password("same-password")
        b = auth.hash_password("same-password")
        assert a != b
        assert auth.verify_password("same-password", a)
        assert auth.verify_password("same-password", b)

    def test_explicit_salt_is_reproducible(self):
        salt = b"0123456789abcdef"
        assert auth.hash_password("p", salt) == auth.hash_password("p", salt)

    def test_malformed_stored_value(self):
        for bad in ("", "plain", "pbkdf2$100$salt", "pbkdf2$abc$salt$hash"):
            assert auth.verify_password("anything", bad) is False

    def test_empty_password(self):
        stored = auth.hash_password("")
        assert auth.verify_password("", stored) is True
        assert auth.verify_password("x", stored) is False


# ============================================================
# 令牌签发与校验
# ============================================================
class TestToken:
    def test_roundtrip_keeps_claims(self, env):
        token = auth.issue_token(_claims())
        got = auth.verify_token(token)
        assert got is not None
        assert got["sub"] == "u-admin"
        assert got["role"] == "admin"
        assert got["groups"] == ["全员", "管理层"]
        assert got["exp"] > time.time()
        assert got["iat"] <= time.time()
        assert len(got["jti"]) == 16

    def test_jti_is_unique(self, env):
        a = auth.verify_token(auth.issue_token(_claims()))
        b = auth.verify_token(auth.issue_token(_claims()))
        assert a["jti"] != b["jti"]

    def test_tampered_payload_rejected(self, env):
        header, body, sig = auth.issue_token(_claims(role="employee")).split(".")
        payload = json.loads(auth._b64url_decode(body))
        payload["role"] = "admin"          # 提权尝试
        forged_body = auth._b64url(json.dumps(payload, separators=(",", ":")).encode())
        assert auth.verify_token(f"{header}.{forged_body}.{sig}") is None

    def test_tampered_signature_rejected(self, env):
        header, body, sig = auth.issue_token(_claims()).split(".")
        flipped = ("A" if sig[0] != "A" else "B") + sig[1:]
        assert auth.verify_token(f"{header}.{body}.{flipped}") is None

    def test_alg_none_rejected(self, env):
        """把 header 改成 alg=none 也不能绕过——签名用固定算法重算"""
        header, body, sig = auth.issue_token(_claims()).split(".")
        none_header = auth._b64url(json.dumps({"alg": "none", "typ": "JWT"}).encode())
        assert auth.verify_token(f"{none_header}.{body}.{sig}") is None

    def test_expired_token_rejected(self, env, monkeypatch):
        monkeypatch.setattr(auth, "TOKEN_TTL_SECONDS", -1)
        assert auth.verify_token(auth.issue_token(_claims())) is None

    def test_malformed_token(self, env):
        for bad in ("", "abc", "a.b", "a.b.c.d", "..", None):
            assert auth.verify_token(bad) is None

    def test_secret_rotation_invalidates(self, env, monkeypatch):
        token = auth.issue_token(_claims())
        assert auth.verify_token(token) is not None
        monkeypatch.setenv("AUTH_JWT_SECRET", "another-secret")
        assert auth.verify_token(token) is None

    def test_secret_persisted_to_file(self, env):
        auth.issue_token(_claims())
        secret_path = env / "secret.key"
        assert secret_path.exists()
        assert len(secret_path.read_bytes()) == 32


# ============================================================
# 吊销（登出即失效）
# ============================================================
class TestRevocation:
    def test_revoke_blocks_active_check(self, env):
        token = auth.issue_token(_claims())
        assert auth.verify_token_active(token) is not None

        assert auth.revoke_token(token) is True
        # 签名仍然合法（JWT 本身没变），是黑名单拦住的——两层职责分离
        assert auth.verify_token(token) is not None
        assert auth.verify_token_active(token) is None

    def test_revoke_invalid_token_returns_false(self, env):
        assert auth.revoke_token("not-a-token") is False

    def test_blacklist_persisted(self, env):
        token = auth.issue_token(_claims())
        auth.revoke_token(token)
        assert (env / "revoked.json").exists()

        # 模拟进程重启：清掉内存缓存，从文件恢复
        auth._revoked_store = None
        assert auth.verify_token_active(token) is None

    def test_blacklist_isolated_by_jti(self, env):
        """同一账号多设备：吊销一台不影响另一台"""
        phone = auth.issue_token(_claims())
        laptop = auth.issue_token(_claims())
        auth.revoke_token(phone)
        assert auth.verify_token_active(phone) is None
        assert auth.verify_token_active(laptop) is not None

    def test_expired_entries_purged(self, env):
        token = auth.issue_token(_claims())
        auth.revoke_token(token)
        store = auth._load_revoked()
        assert store
        # 把过期时间改到过去，再写一次就应被清理
        for jti in list(store):
            store[jti] = time.time() - 10
        auth._save_revoked(store)
        assert auth._load_revoked() == {}


# ============================================================
# 身份提供者（本地账号）
# ============================================================
class TestLocalProvider:
    def _provider(self, tmp_path):
        return auth.LocalPasswordProvider(users_file=str(tmp_path / "users.json"))

    def test_first_run_seeds_accounts(self, env):
        p = self._provider(env)
        assert p.authenticate({"username": "admin", "password": "admin123"}) is not None
        users = json.load(open(env / "users.json", encoding="utf-8"))
        assert len(users) == 4
        for u in users.values():
            assert u["password_hash"].startswith("pbkdf2$")
            assert "password" not in u           # 只存散列，不存明文

    def test_login_ok(self, env):
        ident = self._provider(env).authenticate({"username": "admin", "password": "admin123"})
        assert ident is not None
        assert ident.user_id == "u-admin"
        assert ident.role == "admin"
        assert "全员" in ident.groups

    def test_login_wrong_password(self, env):
        assert self._provider(env).authenticate(
            {"username": "admin", "password": "wrong"}) is None

    def test_login_unknown_user(self, env):
        assert self._provider(env).authenticate(
            {"username": "nobody", "password": "admin123"}) is None

    def test_login_empty_credentials(self, env):
        p = self._provider(env)
        assert p.authenticate({}) is None
        assert p.authenticate(None) is None
        assert p.authenticate({"username": "admin"}) is None

    def test_disabled_user_cannot_login(self, env):
        p = self._provider(env)
        assert p.authenticate({"username": "admin", "password": "admin123"}) is not None
        path = env / "users.json"
        users = json.load(open(path, encoding="utf-8"))
        users["admin"]["disabled"] = True
        json.dump(users, open(path, "w", encoding="utf-8"), ensure_ascii=False)
        assert p.authenticate({"username": "admin", "password": "admin123"}) is None

    def test_users_file_not_reseeded(self, env):
        """第二次启动不应覆盖已有用户（否则改过的密码会被打回演示值）"""
        p1 = self._provider(env)
        assert p1.authenticate({"username": "admin", "password": "admin123"}) is not None
        users = json.load(open(env / "users.json", encoding="utf-8"))
        users["admin"]["password_hash"] = auth.hash_password("changed-password")
        json.dump(users, open(env / "users.json", "w", encoding="utf-8"), ensure_ascii=False)

        p2 = self._provider(env)
        assert p2.authenticate({"username": "admin", "password": "changed-password"}) is not None
        assert p2.authenticate({"username": "admin", "password": "admin123"}) is None


# ============================================================
# Identity <-> claims
# ============================================================
class TestIdentityClaims:
    def test_roundtrip(self):
        ident = auth.Identity("u-1", "alice", "爱丽丝", "manager", ["全员", "管理层"])
        claims = ident.to_claims()
        assert claims["sub"] == "u-1"
        back = auth.Identity.from_claims(claims)
        assert (back.user_id, back.username, back.display_name, back.role, back.groups) == \
               ("u-1", "alice", "爱丽丝", "manager", ["全员", "管理层"])

    def test_from_claims_defaults_are_least_privileged(self):
        """缺字段时不能默认成高权限角色"""
        back = auth.Identity.from_claims({})
        assert back.role == "employee"
        assert back.groups == ["全员"]
