# -*- coding: utf-8 -*-
"""
测试公共 fixtures
"""

import sys
import os
import pytest

# 将 surgical_qa 目录加入 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'surgical_qa'))


@pytest.fixture
def sample_records():
    """模拟问答记录"""
    return [
        {
            "question": "公司年假政策是什么？",
            "answer": "根据公司制度，正式员工每年享有5-15天年假，按工龄递增。",
            "department": "人事",
            "source": "csv",
            "doc_id": "",
            "chunk_id": "",
            "chunk_index": -1,
            "page_number": -1,
            "chunk_type": "text",
            "parent_title": "",
            "truncated": False,
            "doc_name": "",
            "access_level": "internal",
            "allowed_groups": "all",
            "is_deleted": False,
            "version": 1,
        },
        {
            "question": "差旅报销需要什么材料？",
            "answer": "差旅报销需要提供发票、行程单、审批单等材料。",
            "department": "财务",
            "source": "csv",
            "doc_id": "",
            "chunk_id": "",
            "chunk_index": -1,
            "page_number": -1,
            "chunk_type": "text",
            "parent_title": "",
            "truncated": False,
            "doc_name": "",
            "access_level": "internal",
            "allowed_groups": "all",
            "is_deleted": False,
            "version": 1,
        },
        {
            "question": "如何申请加班？",
            "answer": "加班需提前提交加班申请，经主管审批后方可执行。",
            "department": "人事",
            "source": "csv",
            "doc_id": "",
            "chunk_id": "",
            "chunk_index": -1,
            "page_number": -1,
            "chunk_type": "text",
            "parent_title": "",
            "truncated": False,
            "doc_name": "",
            "access_level": "internal",
            "allowed_groups": "all",
            "is_deleted": False,
            "version": 1,
        },
        {
            "question": "机密文件如何管理？",
            "answer": "机密文件需存放在保密柜中，仅限授权人员访问。",
            "department": "行政",
            "source": "csv",
            "doc_id": "",
            "chunk_id": "",
            "chunk_index": -1,
            "page_number": -1,
            "chunk_type": "text",
            "parent_title": "",
            "truncated": False,
            "doc_name": "",
            "access_level": "confidential",
            "allowed_groups": "all",
            "is_deleted": False,
            "version": 1,
        },
        {
            "question": "供应商采购流程",
            "answer": "供应商采购需经过申请、审批、比价、签约四个步骤。",
            "department": "采购",
            "source": "doc",
            "doc_id": "doc_001",
            "chunk_id": "chunk_0",
            "chunk_index": 0,
            "page_number": 1,
            "chunk_type": "text",
            "parent_title": "采购管理办法",
            "truncated": False,
            "doc_name": "供应商采购管理办法_V1.1.txt",
            "access_level": "restricted",
            "allowed_groups": "采购部,财务部",
            "is_deleted": False,
            "version": 1,
        },
    ]


@pytest.fixture
def admin_ctx():
    """admin 用户上下文"""
    from access_control import UserContext
    return UserContext(role="admin", groups=["全员"])


@pytest.fixture
def employee_ctx():
    """employee 用户上下文"""
    from access_control import UserContext
    return UserContext(role="employee", groups=["全员", "人事部"])


@pytest.fixture
def manager_ctx():
    """manager 用户上下文"""
    from access_control import UserContext
    return UserContext(role="manager", groups=["全员", "人事部"])


@pytest.fixture
def guest_ctx():
    """guest 用户上下文"""
    from access_control import UserContext
    return UserContext(role="guest", groups=["全员"])
