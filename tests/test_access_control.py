# -*- coding: utf-8 -*-
"""
ACL 权限控制模块测试

覆盖:
- UserContext 初始化与角色降级
- AccessPolicy.check_access 四分支判定
- AccessPolicy.build_filter_expr Milvus 过滤表达式生成
- clean_groups 组名清洗
- AuditLogger 基本功能
"""

import os
import json
import pytest
import tempfile
import shutil

from access_control import UserContext, AccessPolicy, clean_groups, AuditLogger


# ============================================================
# UserContext
# ============================================================
class TestUserContext:
    def test_valid_roles(self):
        for role in ["admin", "manager", "employee", "guest"]:
            ctx = UserContext(role=role)
            assert ctx.role == role

    def test_invalid_role_downgrades(self):
        ctx = UserContext(role="superuser")
        assert ctx.role == "employee"

    def test_default_groups(self):
        ctx = UserContext()
        assert ctx.groups == ["全员"]

    def test_custom_groups(self):
        ctx = UserContext(role="manager", groups=["人事部", "财务部"])
        assert "人事部" in ctx.groups
        assert "财务部" in ctx.groups


# ============================================================
# AccessPolicy.check_access
# ============================================================
class TestCheckAccess:
    # -- admin 全部可见 --
    def test_admin_sees_all(self, admin_ctx):
        assert AccessPolicy.check_access("public", "all", admin_ctx) is True
        assert AccessPolicy.check_access("internal", "all", admin_ctx) is True
        assert AccessPolicy.check_access("restricted", "all", admin_ctx) is True
        assert AccessPolicy.check_access("confidential", "all", admin_ctx) is True

    # -- guest 仅 public --
    def test_guest_sees_public_only(self, guest_ctx):
        assert AccessPolicy.check_access("public", "all", guest_ctx) is True
        assert AccessPolicy.check_access("internal", "all", guest_ctx) is False
        assert AccessPolicy.check_access("restricted", "all", guest_ctx) is False
        assert AccessPolicy.check_access("confidential", "all", guest_ctx) is False

    # -- employee: public + internal --
    def test_employee_sees_public_and_internal(self, employee_ctx):
        assert AccessPolicy.check_access("public", "all", employee_ctx) is True
        assert AccessPolicy.check_access("internal", "all", employee_ctx) is True
        assert AccessPolicy.check_access("restricted", "all", employee_ctx) is False
        assert AccessPolicy.check_access("confidential", "all", employee_ctx) is False

    # -- manager: public + internal + restricted(组匹配) --
    def test_manager_restricted_group_match(self, manager_ctx):
        assert AccessPolicy.check_access("public", "all", manager_ctx) is True
        assert AccessPolicy.check_access("internal", "all", manager_ctx) is True
        assert AccessPolicy.check_access("restricted", "all", manager_ctx) is True
        assert AccessPolicy.check_access("restricted", "人事部", manager_ctx) is True
        assert AccessPolicy.check_access("restricted", "财务部", manager_ctx) is False
        assert AccessPolicy.check_access("confidential", "all", manager_ctx) is False

    # -- 空值默认处理 --
    def test_none_defaults(self, employee_ctx):
        assert AccessPolicy.check_access(None, None, employee_ctx) is True  # 默认 internal
        assert AccessPolicy.check_access("", "", employee_ctx) is True


# ============================================================
# AccessPolicy.build_filter_expr
# ============================================================
class TestBuildFilterExpr:
    def test_admin_no_acl_filter(self, admin_ctx):
        expr = AccessPolicy.build_filter_expr(admin_ctx)
        assert "access_level" not in expr or "is_deleted" in expr
        assert "is_deleted == false" in expr

    def test_employee_filter(self, employee_ctx):
        expr = AccessPolicy.build_filter_expr(employee_ctx)
        assert "public" in expr
        assert "internal" in expr
        assert "restricted" not in expr
        assert "confidential" not in expr
        assert "is_deleted == false" in expr

    def test_guest_filter(self, guest_ctx):
        expr = AccessPolicy.build_filter_expr(guest_ctx)
        assert "public" in expr
        assert "internal" not in expr

    def test_manager_filter(self, manager_ctx):
        expr = AccessPolicy.build_filter_expr(manager_ctx)
        assert "public" in expr
        assert "internal" in expr
        assert "restricted" in expr
        assert "confidential" not in expr


# ============================================================
# clean_groups
# ============================================================
class TestCleanGroups:
    def test_basic_clean(self):
        assert clean_groups("人事部,财务部") == "人事部,财务部"

    def test_strip_and_dedup(self):
        result = clean_groups(" 人事部 , 财务部 , 人事部 ")
        assert result == "人事部,财务部"

    def test_empty_input(self):
        assert clean_groups("") == "all"
        assert clean_groups("  ") == "all"
        assert clean_groups(None) == "all"

    def test_remove_empty_segments(self):
        result = clean_groups("人事部,,财务部,")
        assert "人事部" in result
        assert "财务部" in result
        assert ",," not in result


# ============================================================
# AuditLogger
# ============================================================
class TestAuditLogger:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.logger = AuditLogger(log_dir=self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_log_search(self, employee_ctx):
        self.logger.log_search(
            user_ctx=employee_ctx,
            filter_expr='access_level == "internal"',
            result_count=3,
            query="请假怎么申请",
            latency_ms=150.5,
            channels=["vector", "bm25"],
        )
        logs = self.logger.query_logs(event_type="search")
        assert len(logs) == 1
        assert logs[0]["result_count"] == 3
        assert logs[0]["event_type"] == "search"

    def test_log_document_operation(self, admin_ctx):
        self.logger.log_document_operation(
            user_ctx=admin_ctx,
            operation="upload",
            doc_id="doc_001",
            success=True,
            detail="file=test.pdf",
        )
        logs = self.logger.query_logs(event_type="document_op")
        assert len(logs) == 1
        assert logs[0]["success"] is True

    def test_query_logs_with_role_filter(self, admin_ctx, employee_ctx):
        self.logger.log_search(admin_ctx, "", 1, query="admin query")
        self.logger.log_search(employee_ctx, "", 2, query="employee query")

        admin_logs = self.logger.query_logs(role="admin")
        assert len(admin_logs) == 1

        employee_logs = self.logger.query_logs(role="employee")
        assert len(employee_logs) == 1

    def test_get_stats(self, employee_ctx):
        self.logger.log_search(employee_ctx, "", 3, query="test", latency_ms=100)
        self.logger.log_search(employee_ctx, "", 5, query="test2", latency_ms=200)

        stats = self.logger.get_stats()
        assert stats["total"] == 2
        assert stats["avg_latency_ms"] == 150.0

    def test_pii_desensitization_in_logs(self, employee_ctx):
        self.logger.log_search(
            employee_ctx, "", 1,
            query="我的手机号是13812345678，请帮我查询",
        )
        logs = self.logger.query_logs()
        assert "13812345678" not in logs[0]["query"]
        assert "***手机号***" in logs[0]["query"]
