# -*- coding: utf-8 -*-
"""
企业级知识库 - BM25 混合检索模块
将 BM25 关键词检索与向量语义检索结合，大幅提升检索准确率

原理：
- BM25: 精确匹配关键词（如"年假"必须出现）
- 向量检索: 语义相似（如"请假"≈"休假"）
- 混合: RRF(Reciprocal Rank Fusion) 融合两路结果
"""

import os
import json
import math
from collections import defaultdict
import jieba
from config import STOPWORDS


class BM25Index:
    """BM25 关键词检索索引"""

    def __init__(self, k1=1.5, b=0.75):
        """
        初始化 BM25 参数
        :param k1: 词频饱和参数
        :param b: 文档长度归一化参数
        """
        self.k1 = k1
        self.b = b
        self.documents = []  # 分词后的文档列表
        self.doc_lengths = []  # 文档长度
        self.avg_dl = 0  # 平均文档长度
        self.df = defaultdict(int)  # 词频（包含该词的文档数）
        self.tfidf = []  # 每个词的 IDF 值
        self.records = []  # 原始记录引用

    def build(self, records):
        """
        从问答记录构建 BM25 索引
        :param records: 问答记录列表 [{question, answer, department}, ...]
        """
        print(f"[BM25] 构建索引，共 {len(records)} 条记录...")

        self.records = records
        self.documents = []

        for r in records:
            # 使用 full_text 或 question 进行索引（文档用 full_text，QA 用 question）
            text = r.get('full_text', r['question'])
            words = [w.strip() for w in jieba.cut(text)
                     if w.strip() and w.strip() not in STOPWORDS]
            self.documents.append(words)

        # 计算文档长度
        self.doc_lengths = [len(doc) for doc in self.documents]
        self.avg_dl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 1

        # 计算 DF（文档频率）
        self.df = defaultdict(int)
        for doc in self.documents:
            unique_words = set(doc)
            for word in unique_words:
                self.df[word] += 1

        print(f"[BM25] 索引构建完成，词汇表大小: {len(self.df)}")

    def get_idf(self, term):
        """计算 IDF 值"""
        n = len(self.documents)
        df = self.df.get(term, 0)
        if df == 0:
            return 0
        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def search(self, query, top_k=10):
        """
        BM25 检索（优化版：支持整词匹配和短语加分）
        :param query: 查询文本
        :param top_k: 返回数量
        :return: [(record_index, score), ...]
        """
        # 分词
        query_words = [w.strip() for w in jieba.cut(query)
                       if w.strip() and w.strip() not in STOPWORDS]

        if not query_words:
            return []

        # 计算每个文档的 BM25 分数
        scores = []
        query_lower = query.lower()

        for idx, doc in enumerate(self.documents):
            score = 0
            doc_len = self.doc_lengths[idx]
            doc_text = ' '.join(doc).lower()

            # 统计词频
            term_freq = defaultdict(int)
            for word in doc:
                term_freq[word] += 1

            for qword in query_words:
                tf = term_freq.get(qword, 0)
                if tf == 0:
                    continue

                idf = self.get_idf(qword)
                # BM25 公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
                word_score = idf * numerator / denominator

                # 长词加分（2字以上的词更重要）
                if len(qword) >= 2:
                    word_score *= 1.5

                score += word_score

            # 短语匹配加分：如果文档包含完整查询短语，额外加分
            if query_lower in doc_text:
                score *= 2.0

            if score > 0:
                scores.append((idx, score))

        # 排序取 top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, filepath):
        """保存索引"""
        data = {
            'k1': self.k1,
            'b': self.b,
            'documents': self.documents,
            'doc_lengths': self.doc_lengths,
            'avg_dl': self.avg_dl,
            'df': dict(self.df),
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, filepath):
        """加载索引"""
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.k1 = data['k1']
            self.b = data['b']
            self.documents = data['documents']
            self.doc_lengths = data['doc_lengths']
            self.avg_dl = data['avg_dl']
            self.df = defaultdict(int, data['df'])
            return True
        except Exception:
            return False


class HybridSearcher:
    """混合检索器：BM25 + 向量检索（向量可选）"""

    def __init__(self, records, vector_index=None, alpha=0.5):
        """
        初始化混合检索器
        :param records: 问答记录列表
        :param vector_index: 向量索引 (VectorIndex 实例，可为 None)
        :param alpha: 混合权重 (0=纯BM25, 1=纯向量, 0.5=均衡)
        """
        self.records = records
        self.vector_index = vector_index
        self.alpha = alpha if vector_index else 0  # 无向量时纯用 BM25

        # 初始化 BM25 索引
        self.bm25 = BM25Index()
        self.bm25.build(records)

        mode = "BM25+向量" if vector_index else "纯BM25"
        print(f"[HybridSearch] 初始化完成，模式={mode}，alpha={self.alpha}")

    def search(self, query, top_k=3, bm25_k=5, vector_k=5, department=None):
        """
        混合检索
        :param query: 查询文本
        :param top_k: 最终返回数量
        :param bm25_k: BM25 检索取前几条
        :param vector_k: 向量检索取前几条
        :param department: 限定部门
        :return: 融合后的检索结果列表
        """
        # 1. BM25 检索
        bm25_results = self.bm25.search(query, top_k=bm25_k)
        bm25_rank = {idx: rank for rank, (idx, score) in enumerate(bm25_results)}

        # 2. 向量检索（如果有的话）
        vector_rank = {}
        vector_scores = {}
        if self.vector_index:
            vector_results = self.vector_index.search(query, top_k=vector_k)
            vector_rank = {r['index']: rank for rank, r in enumerate(vector_results)}
            vector_scores = {r['index']: r['score'] for r in vector_results}

        # 3. RRF (Reciprocal Rank Fusion) 融合
        all_indices = set(bm25_rank.keys()) | set(vector_rank.keys())
        rrf_scores = {}

        for idx in all_indices:
            bm25_rrf = 1.0 / (60 + bm25_rank.get(idx, 999))
            vector_rrf = 1.0 / (60 + vector_rank.get(idx, 999))
            rrf_scores[idx] = (1 - self.alpha) * bm25_rrf + self.alpha * vector_rrf

        # 排序
        sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # 构建结果
        results = []
        for idx, score in sorted_indices[:top_k]:
            if idx < len(self.records):
                record = self.records[idx]
                results.append({
                    'record': record,
                    'score': score,
                    'vector_score': vector_scores.get(idx, 0.0),
                    'index': idx,
                    'bm25_rank': bm25_rank.get(idx, -1),
                    'vector_rank': vector_rank.get(idx, -1)
                })

        # 部门过滤
        if department and department != '全部':
            dept_results = [r for r in results if r['record']['department'] == department]
            other_results = [r for r in results if r['record']['department'] != department]
            results = dept_results + other_results

        return results[:top_k]

    def compare_strategies(self, query, top_k=5):
        """
        对比不同检索策略的效果
        :param query: 查询文本
        :param top_k: 返回数量
        :return: 对比结果
        """
        # 纯 BM25
        bm25_results = self.bm25.search(query, top_k=top_k)
        bm25_top = [self.records[idx] for idx, _ in bm25_results[:3]]

        # 纯向量（如果向量索引可用）
        vector_top = []
        if self.vector_index:
            try:
                vector_results = self.vector_index.search(query, top_k=top_k)
                vector_top = [r['record'] for r in vector_results[:3]]
            except Exception:
                vector_top = []
        else:
            vector_top = []

        # 混合
        hybrid_results = self.search(query, top_k=top_k)
        hybrid_top = [r['record'] for r in hybrid_results[:3]]

        return {
            'query': query,
            'bm25_top3': [{'question': r['question'][:50], 'department': r['department']} for r in bm25_top],
            'vector_top3': [{'question': r['question'][:50], 'department': r['department']} for r in vector_top],
            'hybrid_top3': [{'question': r['question'][:50], 'department': r['department']} for r in hybrid_top],
        }



