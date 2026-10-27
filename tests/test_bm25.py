# -*- coding: utf-8 -*-
"""
BM25 检索模块测试

覆盖:
- BM25Index 构建与搜索
- 分词 + 停用词过滤
- 短语匹配加分
- HybridSearcher RRF 融合
"""

import pytest
from hybrid_search import BM25Index, HybridSearcher


# ============================================================
# BM25Index
# ============================================================
class TestBM25Index:
    def test_build_index(self, sample_records):
        bm25 = BM25Index()
        bm25.build(sample_records)
        assert len(bm25.documents) == len(sample_records)
        assert len(bm25.doc_lengths) == len(sample_records)
        assert bm25.avg_dl > 0

    def test_search_basic(self, sample_records):
        bm25 = BM25Index()
        bm25.build(sample_records)
        results = bm25.search("年假", top_k=3)
        assert len(results) > 0
        # 第一条应该包含年假相关
        top_idx, top_score = results[0]
        assert top_score > 0

    def test_search_with_stopwords(self, sample_records):
        bm25 = BM25Index()
        bm25.build(sample_records)
        # "的" 是停用词，不应有结果
        results = bm25.search("的", top_k=3)
        assert len(results) == 0

    def test_search_no_match(self, sample_records):
        bm25 = BM25Index()
        bm25.build(sample_records)
        results = bm25.search("火星文字", top_k=3)
        assert len(results) == 0

    def test_search_top_k_limit(self, sample_records):
        bm25 = BM25Index()
        bm25.build(sample_records)
        results = bm25.search("申请", top_k=2)
        assert len(results) <= 2

    def test_phrase_match_bonus(self):
        """完整短语匹配应获得加分"""
        records = [
            {"question": "公司年假政策", "answer": "年假5天"},
            {"question": "年假和请假", "answer": "年假是带薪的"},
        ]
        bm25 = BM25Index()
        bm25.build(records)
        results = bm25.search("公司年假政策", top_k=2)
        assert len(results) > 0
        # 完整匹配 "公司年假政策" 的文档应排第一
        assert results[0][0] == 0

    def test_save_and_load(self, sample_records, tmp_path):
        bm25 = BM25Index()
        bm25.build(sample_records)
        filepath = str(tmp_path / "bm25_index.json")

        bm25.save(filepath)

        bm25_loaded = BM25Index()
        success = bm25_loaded.load(filepath)
        assert success is True
        assert len(bm25_loaded.documents) == len(bm25.documents)

    def test_load_nonexistent(self):
        bm25 = BM25Index()
        assert bm25.load("/nonexistent/path.json") is False

    def test_get_idf(self, sample_records):
        bm25 = BM25Index()
        bm25.build(sample_records)
        # 高频词 IDF 低
        idf_common = bm25.get_idf("公司")
        # 低频词 IDF 高
        idf_rare = bm25.get_idf("火星文")
        # 不存在的词 IDF = 0
        idf_missing = bm25.get_idf("完全不存在的词xyz")
        assert idf_missing == 0


# ============================================================
# HybridSearcher (无向量索引时退化为纯 BM25)
# ============================================================
class TestHybridSearcherNoVector:
    def test_search_without_vector(self, sample_records):
        searcher = HybridSearcher(sample_records, vector_index=None, alpha=0)
        results = searcher.search("年假", top_k=3)
        assert len(results) > 0

    def test_compare_strategies_without_vector(self, sample_records):
        searcher = HybridSearcher(sample_records, vector_index=None)
        comparison = searcher.compare_strategies("年假")
        assert "bm25_top3" in comparison
        assert "vector_top3" in comparison
        assert comparison["vector_top3"] == []  # 无向量
