# -*- coding: utf-8 -*-
"""
企业级知识库 - Rerank 重排模块

设计原则：
- 使用 DashScope gte-rerank API（与 Embedding 同源，无额外依赖）
- 输入：query + 候选 chunk 列表
- 输出：按 rerank 分数降序排列的结果列表
- 主干不做自定义打分公式，裸 rerank
"""

import os
import logging
import requests
from typing import List, Dict

from retrieval_config import RERANK

logger = logging.getLogger(__name__)


class DashScopeReranker:
    """
    DashScope gte-rerank 重排器

    用法：
        reranker = DashScopeReranker()
        ranked = reranker.rerank(query, candidates, top_n=6)
    """

    def __init__(self, model: str = None):
        self.model = model or RERANK["model"]
        self.api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "未设置 DASHSCOPE_API_KEY 环境变量！\n"
                "请设置: $env:DASHSCOPE_API_KEY='your-api-key'"
            )
        # DashScope rerank API 端点
        self.url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
        logger.info(f"[Reranker] 初始化完成，模型: {self.model}")

    @staticmethod
    def _document_text(doc: Dict) -> str:
        """
        候选文本 = 文档名 + 章节标题 + 正文

        必要性：表格/列表切出的碎片（如 "3. 3天以上：部门负责人→人事部审批"）脱离语境后，
        任何含"审批"的问题都会误命中。补上标题让重排模型知道这段在讲什么。
        """
        stem = os.path.splitext(doc.get("doc_name") or "")[0]
        parts = []
        if stem:
            parts.append(f"文档：{stem}")
        if doc.get("parent_title"):
            parts.append(f"章节：{doc['parent_title']}")
        parts.append((doc.get("content") or "")[:2000])
        return "\n".join(parts)

    def rerank(self, query: str, documents: List[Dict],
               top_n: int = None) -> List[Dict]:
        """
        对候选文档执行 rerank

        :param query: 用户查询
        :param documents: 候选文档列表，每项需包含 'content' 字段
            格式: [{"content": str, "index": int, ...}, ...]
        :param top_n: 返回前 N 个，None 则返回全部
        :return: 按 rerank 分数降序排列的列表，
            每项新增 'rerank_score' 字段
        """
        if not documents:
            return []

        if top_n is None:
            top_n = RERANK["top_n"]

        # 提取文本（截断过长内容，rerank 模型一般有 4K token 上限）
        texts = [self._document_text(doc) for doc in documents]

        if not any(texts):
            return []

        try:
            # 调用 DashScope rerank API
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model,
                "input": {
                    "query": query,
                    "documents": texts,
                },
            }

            resp = requests.post(
                self.url,
                headers=headers,
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            result = resp.json()

            # 解析返回结果
            # DashScope rerank 返回格式:
            # {"output": {"results": [{"index": 0, "relevance_score": 0.95}, ...]}}
            output = result.get("output", {})
            rerank_results = output.get("results", [])

            if not rerank_results:
                logger.warning("[Reranker] API 返回空结果")
                return documents[:top_n]

            # 按 relevance_score 降序排列
            rerank_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

            # 组装结果
            ranked = []
            for item in rerank_results[:top_n]:
                orig_idx = item["index"]
                if orig_idx < len(documents):
                    doc = documents[orig_idx].copy()
                    doc["rerank_score"] = item.get("relevance_score", 0.0)
                    ranked.append(doc)

            logger.info(
                f"[Reranker] 重排完成: {len(documents)} 个候选 → "
                f"top {len(ranked)}，最高分: {ranked[0]['rerank_score']:.4f}" if ranked else "无结果"
            )
            return ranked

        except requests.exceptions.Timeout:
            logger.error("[Reranker] API 请求超时（15s）")
            # 降级：按原始顺序返回
            return self._fallback(documents, top_n)
        except requests.exceptions.RequestException as e:
            logger.error(f"[Reranker] API 请求失败: {e}")
            return self._fallback(documents, top_n)
        except Exception as e:
            logger.error(f"[Reranker] 重排异常: {e}")
            return self._fallback(documents, top_n)

    @staticmethod
    def _fallback(documents: List[Dict], top_n: int) -> List[Dict]:
        """
        降级策略：不调 rerank，按原始分数排序返回
        """
        sorted_docs = sorted(
            documents,
            key=lambda x: x.get("score", x.get("rerank_score", 0)),
            reverse=True,
        )
        result = []
        for doc in sorted_docs[:top_n]:
            d = doc.copy()
            d["rerank_score"] = d.get("score", 0.0)
            d["rerank_fallback"] = True
            result.append(d)
        return result



