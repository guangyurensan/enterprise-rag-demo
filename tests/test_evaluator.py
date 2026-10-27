# -*- coding: utf-8 -*-
"""
评测模块测试

覆盖:
- EvalSetManager 评测集管理
- EvalMetrics 指标计算 (Recall@K, MRR, keyword_hit_rate)
"""

import os
import json
import pytest
import tempfile
import shutil

from evaluator import EvalSetManager, EvalMetrics


# ============================================================
# EvalSetManager
# ============================================================
class TestEvalSetManager:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mgr = EvalSetManager(eval_dir=self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        items = [
            {"id": "q001", "query": "请假", "expected_keywords": ["请假"]},
            {"id": "q002", "query": "报销", "expected_keywords": ["报销"]},
        ]
        self.mgr.save_set("test_set", items, metadata={"description": "测试"})
        loaded = self.mgr.load_set("test_set")
        assert len(loaded) == 2
        assert loaded[0]["query"] == "请假"

    def test_list_sets(self):
        self.mgr.save_set("set_a", [])
        self.mgr.save_set("set_b", [])
        sets = self.mgr.list_sets()
        assert "set_a" in sets
        assert "set_b" in sets

    def test_load_nonexistent(self):
        loaded = self.mgr.load_set("nonexistent")
        assert loaded == []

    def test_add_item(self):
        self.mgr.save_set("test", [{"id": "q001", "query": "a"}])
        self.mgr.add_item("test", {"query": "b"})
        loaded = self.mgr.load_set("test")
        assert len(loaded) == 2


# ============================================================
# EvalMetrics
# ============================================================
class TestEvalMetrics:
    def setup_method(self):
        self.metrics = EvalMetrics()

    def test_recall_at_k_full_hit(self):
        results = [
            {"doc_id": "doc_001"},
            {"doc_id": "doc_002"},
            {"doc_id": "doc_003"},
        ]
        recall = self.metrics.recall_at_k(results, ["doc_001", "doc_002"], k=3)
        assert recall == 1.0

    def test_recall_at_k_partial_hit(self):
        results = [
            {"doc_id": "doc_001"},
            {"doc_id": "doc_002"},
        ]
        recall = self.metrics.recall_at_k(results, ["doc_001", "doc_003"], k=2)
        assert recall == 0.5

    def test_recall_at_k_no_hit(self):
        results = [{"doc_id": "doc_001"}]
        recall = self.metrics.recall_at_k(results, ["doc_999"], k=1)
        assert recall == 0.0

    def test_recall_at_k_empty_expected(self):
        results = [{"doc_id": "doc_001"}]
        recall = self.metrics.recall_at_k(results, [], k=1)
        assert recall == 1.0  # 无期望视为完全命中

    def test_mrr_first_position(self):
        results = [{"doc_id": "doc_001"}, {"doc_id": "doc_002"}]
        mrr = self.metrics.mrr(results, ["doc_001"])
        assert mrr == 1.0

    def test_mrr_second_position(self):
        results = [{"doc_id": "doc_001"}, {"doc_id": "doc_002"}]
        mrr = self.metrics.mrr(results, ["doc_002"])
        assert mrr == 0.5

    def test_mrr_no_hit(self):
        results = [{"doc_id": "doc_001"}]
        mrr = self.metrics.mrr(results, ["doc_999"])
        assert mrr == 0.0

    def test_keyword_hit_rate_full(self):
        results = [{"content": "请假审批流程"}]
        rate = self.metrics.keyword_hit_rate(results, ["请假", "审批"])
        assert rate == 1.0

    def test_keyword_hit_rate_partial(self):
        results = [{"content": "请假流程"}]
        rate = self.metrics.keyword_hit_rate(results, ["请假", "审批", "流程"])
        assert abs(rate - 2 / 3) < 0.01

    def test_keyword_hit_rate_empty_keywords(self):
        results = [{"content": "test"}]
        rate = self.metrics.keyword_hit_rate(results, [])
        assert rate == 1.0
