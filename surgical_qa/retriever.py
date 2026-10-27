# -*- coding: utf-8 -*-
"""
企业级知识库 - 检索与回答生成模块

检索主干流程：
1. Query 预处理（清洗 + 改写）
2. 多路召回：向量召回 + BM25 召回（并行）
3. 合并去重（RRF 融合）
4. Rerank 重排
5. [可选] truncated 相邻回填（代偿分支）
6. Token 预算截断
7. 送入 LLM 生成回答
"""

import os
import re
import time
import logging
from collections import defaultdict
from openai import OpenAI, APITimeoutError
from vector_index import VectorIndex
from image_analyzer import ImageAnalyzer
from query_rewriter import QueryRewriter
from hybrid_search import HybridSearcher, BM25Index
from cache_manager import CacheManager
from document_manager import DocumentManager
from reranker import DashScopeReranker
from access_control import UserContext, AccessPolicy, audit_logger
from security import security_gateway
from resilience import llm_breaker, rerank_breaker, CircuitOpenError
from version_policy import build_display_name
from metrics import metrics
from config import LLM_MODEL, SYSTEM_PROMPT, TOP_K, LLM_TEMPERATURE, LLM_TIMEOUT_SECONDS
from retrieval_config import (
    RERANK, VECTOR_RECALL, BM25_RECALL, RRF,
    TRUNCATED_BACKFILL, TOKEN_BUDGET, DIVERSITY, REWRITE,
)

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """企业级知识库检索器（新检索主干 + 向后兼容）"""

    def __init__(self, records, use_vector_index=True,
                 doc_manager=None):
        self.records = records
        self.use_vector_index = use_vector_index
        self.doc_manager = doc_manager

        # ---- 组件初始化 ----
        print("\n[KnowledgeRetriever] 初始化组件...")

        # 向量索引
        self.vector_index = None
        if use_vector_index:
            self.vector_index = VectorIndex()
            if not self.vector_index.load_index():
                self.vector_index.build_index(records)
        else:
            print("[KnowledgeRetriever] 向量索引已禁用")

        # LLM 客户端 (DeepSeek)
        from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
        api_key = DEEPSEEK_API_KEY
        if not api_key:
            raise ValueError("未设置 DEEPSEEK_API_KEY 环境变量！")
        self.llm_client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL,
                                 timeout=LLM_TIMEOUT_SECONDS)

        # 图片分析器
        self.image_analyzer = ImageAnalyzer()

        # Query 改写器
        self.query_rewriter = QueryRewriter()

        # BM25 索引（始终构建，用于辅助召回）
        self.bm25 = BM25Index()
        self.bm25.build(records)

        # 混合检索器（兼容旧接口）
        self.degraded = False            # 检索是否处于降级状态（向量通道不可用）
        self.degraded_reason = ""
        # 懒加载：HybridSearcher 内部会再建一份完整 BM25，主链路只用 self.bm25，
        # 它仅服务于「策略对比」调试面板。常驻等于白占一份内存，且文档增删后
        # 不会跟着重建（面板会看到旧数据），所以推迟到真正调用时再建。
        self._hybrid_searcher = None

        # Reranker（主干组件）
        self.reranker = None
        if RERANK["enabled"]:
            try:
                self.reranker = DashScopeReranker()
                print("[KnowledgeRetriever] Reranker 已启用")
            except Exception as e:
                print(f"[KnowledgeRetriever] Reranker 初始化失败: {e}，将使用降级策略")

        # 缓存管理器
        self.cache_manager = CacheManager(version=self._corpus_fingerprint())

        # 文档管理器
        if self.doc_manager is None:
            self.doc_manager = DocumentManager()

        # Chunk 顺序索引（用于 truncated 回填）
        self.chunk_index_map = {}
        self._build_chunk_index_map()

        print("[KnowledgeRetriever] 初始化完成！")

    # ============================================================
    # Chunk 顺序索引（truncated 回填用）
    # ============================================================
    def _corpus_fingerprint(self):
        """语料指纹：chunk_id 集合的摘要，用于让缓存随索引变化失效"""
        from hashlib import md5
        ids = "|".join(sorted(c.get('chunk_id', '') for c in self.records))
        return md5(ids.encode('utf-8')).hexdigest()[:12]

    def _build_chunk_index_map(self):
        """
        构建 doc_id + chunk_index → records 中的位置索引
        用于 truncated 回填时快速查找相邻 chunk
        """
        self.chunk_index_map = {}
        for idx, r in enumerate(self.records):
            doc_id = r.get('doc_id')
            chunk_index = r.get('chunk_index')
            if doc_id and chunk_index is not None:
                key = doc_id
                if key not in self.chunk_index_map:
                    self.chunk_index_map[key] = {}
                self.chunk_index_map[key][chunk_index] = idx

    # ============================================================
    # 新检索主干
    # ============================================================
    def _retrieve_pipeline(self, query, department=None, user_ctx=None, eval_mode=False):
        """
        新检索主干（核心方法）

        流程：安全检查 → 向量召回 + BM25 → RRF合并 → Rerank → 回填 → Token预算 → 脱敏

        :param query: 清洗后的查询
        :param department: 部门过滤
        :param user_ctx: 用户上下文（必传，用于 ACL 过滤）
        :return: 最终检索结果列表
        """
        import time
        _start_time = time.time()

        # 默认用户上下文（向后兼容）
        if user_ctx is None:
            user_ctx = UserContext(role="employee", groups=["全员"])

        # ---- Step 0: 安全检查（Prompt 注入检测） ----
        is_safe, clean_query = security_gateway.check_query(query)
        if not is_safe:
            logger.warning(f"[Security] Query rejected (injection): '{query[:60]}'")
            return []  # 返回空结果，由上层处理
        if clean_query != query:
            logger.info(f"[Security] Query sanitized: '{query[:40]}' -> '{clean_query[:40]}'")
        query = clean_query

        # ---- Step 1: 多路召回（带 ACL 过滤） ----
        vector_results = self._vector_recall(query, department=department, user_ctx=user_ctx)
        bm25_results = self._bm25_recall(query, department=department, user_ctx=user_ctx)

        # 记录召回通道
        channels = []
        if vector_results:
            channels.append("vector")
        if bm25_results:
            channels.append("bm25")

        # ---- Step 2: 合并去重（RRF） ----
        merged = self._merge_dedup(vector_results, bm25_results)

        if not merged:
            return []

        # ---- Step 3: Rerank ----
        reranked = self._do_rerank(query, merged)

        # ---- Step 4: Truncated 回填（代偿分支） ----
        if TRUNCATED_BACKFILL["enabled"]:
            reranked = self._backfill_truncated(reranked)

        # ---- Step 5: 单文档配额（防长文档碎片挤占） + Token 预算 ----
        final = self._apply_token_budget(self._diversify(reranked))

        # ---- Step 6: 上下文脱敏（PII 替换） ----
        if not eval_mode:
            final = security_gateway.desensitize_context(final)

        # ---- Step 7: 审计日志（骨架层一期固化：扩展字段） ----
        _latency_ms = (time.time() - _start_time) * 1000
        metrics.incr("search_total")
        metrics.observe("search_latency_ms", _latency_ms)
        if not final:
            metrics.incr("search_empty")
        if channels == ["bm25"]:
            metrics.incr("search_bm25_only")  # 向量通道缺席次数
        acl_expr = AccessPolicy.build_filter_expr(user_ctx)

        # 构建命中文档摘要（前5条）——access_level 必须落审计：
        # 内审要求能回答"谁查过涉密内容"，没有密级就要事后 join registry
        hit_docs = []
        for r in final[:5]:
            hit_docs.append({
                "doc_name": r.get("doc_name", ""),
                "chunk_id": r.get("chunk_id", ""),
                "score": round(r.get("rerank_score", r.get("rrf_score", 0)), 3),
                "access_level": (r.get("record") or {}).get("access_level", "unknown"),
            })

        if not eval_mode:
            audit_logger.log_search(
                user_ctx=user_ctx,
                filter_expr=acl_expr,
                result_count=len(final),
                filtered_count=len(vector_results) + len(bm25_results) - len(merged),
                query=query,
                latency_ms=_latency_ms,
                channels=channels,
                hit_docs=hit_docs,
            )

        return final

    def _vector_recall(self, query, top_k=None, department=None, user_ctx=None):
        """向量召回（利用 Milvus 引擎侧过滤，ACL 不可绕过）"""
        if not self.vector_index or not self.use_vector_index:
            return []

        if top_k is None:
            top_k = VECTOR_RECALL["top_k"]

        if user_ctx is None:
            user_ctx = UserContext(role="employee", groups=["全员"])

        # 构建 Milvus 过滤表达式
        # ACL 条件（强制，不可绕过）
        acl_expr = AccessPolicy.build_filter_expr(user_ctx)

        # 部门条件（可选）
        dept_expr = None
        if department and department != '全部':
            dept_expr = f'department == "{department}"'

        # 合并过滤表达式
        filter_parts = [p for p in [acl_expr, dept_expr] if p]
        filter_expr = ' and '.join(filter_parts) if filter_parts else None

        fetch_k = top_k * 2
        try:
            raw_results = self.vector_index.search(
                query, top_k=fetch_k, filter_expr=filter_expr
            )
        except Exception as e:
            # 向量通道故障（欠费/限流/网络）不应让整条检索链路崩掉：
            # 降级为仅 BM25，并显式标记 degraded，避免质量下滑被静默吞掉。
            self.degraded = True
            self.degraded_reason = f"{type(e).__name__}: {str(e)[:120]}"
            metrics.incr("vector_recall_error")  # 与"真没命中"区分开——通道故障专项计数
            print(f"  [降级] 向量召回不可用，本次仅用 BM25 —— {self.degraded_reason}")
            return []

        results = []
        for r in raw_results:
            meta = r['record']
            score = r['score']

            # 最低分数过滤
            if score < VECTOR_RECALL["min_score"]:
                continue

            results.append({
                'content': meta.get('answer', meta.get('question', '')),
                'record': meta,
                'score': score,
                'recall_source': 'vector',
                'index': r['index'],
                'doc_id': meta.get('doc_id', ''),
                'chunk_index': meta.get('chunk_index', -1),
                'parent_title': meta.get('parent_title', ''),
                'truncated': meta.get('truncated', False),
                'page_number': meta.get('page_number', -1),
                'doc_name': meta.get('doc_name', ''),
            })

        return results[:top_k]

    def _bm25_recall(self, query, top_k=None, department=None, user_ctx=None):
        """BM25 召回（辅助通道，ACL 内存过滤）"""
        if not BM25_RECALL["enabled"]:
            return []

        if top_k is None:
            top_k = BM25_RECALL["top_k"]

        if user_ctx is None:
            user_ctx = UserContext(role="employee", groups=["全员"])

        raw_results = self.bm25.search(query, top_k=top_k * 2 if department else top_k)

        results = []
        for idx, score in raw_results:
            if idx >= len(self.records):
                continue
            record = self.records[idx]

            # 部门过滤
            if department and department != '全部':
                if record.get('department', '') != department:
                    continue

            # ACL 过滤（内存侧，BM25 不支持引擎侧过滤）
            doc_level = record.get('access_level', 'internal')
            doc_groups = record.get('allowed_groups', 'all')
            if not AccessPolicy.check_access(doc_level, doc_groups, user_ctx):
                continue

            content = record.get('answer', record.get('full_text', ''))
            results.append({
                'content': content,
                'record': record,
                'score': score,
                'recall_source': 'bm25',
                'index': idx,
                'doc_id': record.get('doc_id', ''),
                'chunk_index': record.get('chunk_index', -1),
                'parent_title': record.get('parent_title', ''),
                'truncated': record.get('truncated', False),
                'page_number': record.get('page_number', -1),
                'doc_name': record.get('doc_name', ''),
            })

        return results[:top_k]

    def _merge_dedup(self, vector_results, bm25_results):
        """RRF 合并 + 按 chunk_id / index 去重"""
        k = RRF["k"]
        alpha = RRF["alpha"]

        # 用 (doc_id, chunk_index) 或 records index 作为去重键
        rrf_scores = {}
        item_map = {}

        # 向量结果排名
        for rank, item in enumerate(vector_results):
            dedup_key = self._dedup_key(item)
            if dedup_key not in rrf_scores:
                rrf_scores[dedup_key] = 0.0
                item_map[dedup_key] = item
            rrf_score = 1.0 / (k + rank + 1)
            rrf_scores[dedup_key] += (1 - alpha) * rrf_score

        # BM25 结果排名
        for rank, item in enumerate(bm25_results):
            dedup_key = self._dedup_key(item)
            if dedup_key not in rrf_scores:
                rrf_scores[dedup_key] = 0.0
                item_map[dedup_key] = item
            rrf_score = 1.0 / (k + rank + 1)
            rrf_scores[dedup_key] += alpha * rrf_score

        # 按 RRF 分数降序
        sorted_keys = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        merged = []
        for key, score in sorted_keys:
            item = item_map[key].copy()
            item['rrf_score'] = score
            # 补齐展示字段（Milvus payload 只存最小字段集，缺啥从内存 records 反查）
            # doc_name 供追溯、base_title+version_label 供引用展示，缺了会导致
            # 同一次回答里一部分引用显示干净制度名、另一部分显示原始文件名
            pos = self.chunk_index_map.get(item.get('doc_id', ''), {}).get(item.get('chunk_index', -1))
            if pos is not None:
                src = self.records[pos]
                for field in ('doc_name', 'base_title', 'version_label'):
                    if not item.get(field):
                        item[field] = src.get(field, '')
            merged.append(item)

        return merged

    @staticmethod
    def _dedup_key(item):
        """生成去重键"""
        doc_id = item.get('doc_id', '')
        chunk_index = item.get('chunk_index', -1)
        if doc_id and chunk_index >= 0:
            return f"{doc_id}_{chunk_index}"
        # CSV 记录用 records index
        return f"idx_{item.get('index', id(item))}"

    def _do_rerank(self, query, candidates):
        """Rerank 重排（带熔断保护）"""
        if not self.reranker or not RERANK["enabled"]:
            # 降级：按 RRF 分数排序
            candidates.sort(key=lambda x: x.get('rrf_score', 0), reverse=True)
            return candidates[:RERANK["top_n"]]

        top_n = RERANK["top_n"]

        # 熔断保护：如果 Rerank 服务不可用，直接降级
        if rerank_breaker.is_open:
            logger.warning("[Retriever] Rerank circuit breaker OPEN, fallback to RRF")
            candidates.sort(key=lambda x: x.get('rrf_score', 0), reverse=True)
            return candidates[:top_n]

        try:
            return rerank_breaker.call(
                self.reranker.rerank, query, candidates, top_n=top_n
            )
        except CircuitOpenError:
            logger.warning("[Retriever] Rerank breaker tripped, fallback to RRF")
            candidates.sort(key=lambda x: x.get('rrf_score', 0), reverse=True)
            return candidates[:top_n]
        except Exception as e:
            logger.error(f"[Retriever] Rerank failed: {e}, fallback to RRF")
            candidates.sort(key=lambda x: x.get('rrf_score', 0), reverse=True)
            return candidates[:top_n]

    def _backfill_truncated(self, ranked_results):
        """
        Truncated 相邻回填（代偿分支）

        对 truncated=True 的 chunk，查找同文档内相邻 chunk 补充上下文
        约束：前后各 N 条，总共最多追加 M 条
        """
        before_n = TRUNCATED_BACKFILL["before_n"]
        after_n = TRUNCATED_BACKFILL["after_n"]
        max_backfill = before_n + after_n

        backfill_items = []
        seen_keys = {self._dedup_key(item) for item in ranked_results}

        for item in ranked_results:
            if not item.get('truncated'):
                continue

            doc_id = item.get('doc_id', '')
            chunk_index = item.get('chunk_index', -1)

            if not doc_id or chunk_index < 0:
                continue

            doc_chunks = self.chunk_index_map.get(doc_id, {})
            if not doc_chunks:
                continue

            # 向前回填
            for offset in range(1, before_n + 1):
                adj_idx = chunk_index - offset
                if adj_idx in doc_chunks:
                    rec_pos = doc_chunks[adj_idx]
                    adj_item = self._record_to_candidate(rec_pos)
                    adj_key = self._dedup_key(adj_item)
                    if adj_key not in seen_keys:
                        seen_keys.add(adj_key)
                        adj_item['is_backfill'] = True
                        backfill_items.append(adj_item)

            # 向后回填
            for offset in range(1, after_n + 1):
                adj_idx = chunk_index + offset
                if adj_idx in doc_chunks:
                    rec_pos = doc_chunks[adj_idx]
                    adj_item = self._record_to_candidate(rec_pos)
                    adj_key = self._dedup_key(adj_item)
                    if adj_key not in seen_keys:
                        seen_keys.add(adj_key)
                        adj_item['is_backfill'] = True
                        backfill_items.append(adj_item)

        # 回填结果追加到末尾（不参与 rerank，直接进 token 预算）
        ranked_results.extend(backfill_items[:max_backfill])

        if backfill_items:
            logger.info(f"[Retriever] Truncated 回填: 追加 {len(backfill_items)} 个相邻 chunk")

        return ranked_results

    def _diversify(self, results):
        """
        两级单文档配额：区块级（doc × parent_title）+ 文档级总量上限
        区块级：多工作表 Excel 的"按客户统计"类汇总 sheet，与明细 sheet 结构迥异，
        按文档统一配额会让明细页挤掉汇总页（ext-020 误拒的根因），故配额细化到区块
        文档级：防止区块多的文档靠多区块绕过单区块配额
        超出配额的条目按原排序追加到末尾，供 token 预算按需取用
        """
        if not DIVERSITY["enabled"]:
            return results

        cap = DIVERSITY["max_per_doc"]
        doc_cap = DIVERSITY.get("max_per_doc_total", cap * 2)
        kept, overflow = [], []
        sec_counts, doc_counts = defaultdict(int), defaultdict(int)
        for r in results:
            doc_id = r.get('doc_id', '')
            section = (doc_id, r.get('parent_title', ''))
            if sec_counts[section] < cap and doc_counts[doc_id] < doc_cap:
                sec_counts[section] += 1
                doc_counts[doc_id] += 1
                kept.append(r)
            else:
                overflow.append(r)
        return kept + overflow

    def _record_to_candidate(self, record_idx):
        """将 records 中的记录转换为候选 item 格式"""
        record = self.records[record_idx]
        content = record.get('answer', record.get('full_text', ''))
        return {
            'content': content,
            'record': record,
            'score': 0.0,
            'recall_source': 'backfill',
            'index': record_idx,
            'doc_id': record.get('doc_id', ''),
            'chunk_index': record.get('chunk_index', -1),
            'parent_title': record.get('parent_title', ''),
            'truncated': record.get('truncated', False),
            'page_number': record.get('page_number', -1),
            'doc_name': record.get('doc_name', ''),
            'rerank_score': 0.0,
        }

    def _apply_token_budget(self, results):
        """
        Token 预算控制（主干强制）
        按 rerank 顺序依次累加 token，达到上限就停止
        """
        max_token = TOKEN_BUDGET["max_context_token"]
        ratio = TOKEN_BUDGET["token_estimate_ratio"]

        total_token = 0
        final = []

        for item in results:
            content = item.get('content', '')
            est_token = max(1, int(len(content) * ratio))

            if total_token + est_token > max_token:
                # 当前 chunk 放不下了，停止
                break

            final.append(item)
            total_token += est_token

        logger.info(
            f"[Retriever] Token 预算: {len(final)}/{len(results)} 个 chunk, "
            f"≈{total_token} token"
        )
        return final

    # ============================================================
    # 对外接口（兼容 app.py 调用）
    # ============================================================
    def retrieve(self, query, top_k=None, department=None, user_ctx=None):
        """
        基础检索（兼容旧接口，走新主干）
        """
        if top_k is None:
            top_k = TOP_K

        results = self._retrieve_pipeline(query, department=department, user_ctx=user_ctx)
        return results[:top_k]

    def retrieve_hybrid(self, query, top_k=None, department=None, user_ctx=None, eval_mode=False):
        """
        混合检索（兼容旧接口，走新主干）
        :param eval_mode: 评测模式，跳过脱敏和审计日志
        """
        if top_k is None:
            top_k = TOP_K
        if not query or not query.strip():
            # 空查询在向量通道会返回任意文档（实测空白串返回 5 条无关结果），必须显式短路
            return []
        return self._retrieve_pipeline(query, department=department, user_ctx=user_ctx, eval_mode=eval_mode)[:top_k]

    def retrieve_with_rewrite(self, query, conversation_history="", top_k=None,
                              department=None, use_hybrid=True, user_ctx=None):
        """
        带 Query 改写的检索（app.py 主入口）

        :param query: 用户原始查询
        :param conversation_history: 对话历史文本
        :param top_k: 返回数量
        :param department: 限定部门
        :param use_hybrid: 是否使用混合检索（默认 True，始终走新主干）
        :param user_ctx: 用户上下文（必传，用于 ACL 过滤）
        :return: (检索结果列表, 改写信息字典)
        """
        if top_k is None:
            top_k = TOP_K

        # Step 1: Query 改写（无对话历史时跳过，见 retrieval_config.REWRITE 的实测依据）
        has_history = bool((conversation_history or "").strip())
        skip_rewrite = (not REWRITE["enabled"]) or (REWRITE["with_history_only"] and not has_history)

        if skip_rewrite:
            rewrite_result = {
                "original_query": query,
                "rewritten_query": query,
                "sub_queries": [query],
                "rewrite_type": "跳过改写",
            }
        else:
            print(f"  [Query改写] 原始查询: {query}")
            rewrite_result = self.query_rewriter.auto_rewrite(query, conversation_history)
            print(f"  [Query改写] 类型: {rewrite_result['rewrite_type']}")
            print(f"  [Query改写] 改写后: {rewrite_result['rewritten_query']}")

        sub_queries = rewrite_result['sub_queries']

        # 子查询清洗：去掉空串/纯空白（改写模型实测会吐出 ''，空串检索会带回无关文档）
        sub_queries = [s.strip() for s in sub_queries if isinstance(s, str) and s.strip()]

        # 原查询始终参与召回：改写是"补充视角"，不能替换用户原话
        # （实测替换会丢掉原文关键词，p-035 / ext-005 由 1.0 掉到 0）
        merged_queries = [query.strip()] if query.strip() else []
        for sq in sub_queries[:REWRITE["max_sub_queries"]]:
            if sq not in merged_queries:
                merged_queries.append(sq)
        if not merged_queries:
            merged_queries = [query.strip()]
        rewrite_result['sub_queries'] = sub_queries

        # Step 2: 每个查询分别走新检索主干，按 RRF 跨查询融合
        # （不同查询的 rerank 分数不可直接比较，逐查询名次融合才是正确做法）
        fused, order = {}, []
        for q in merged_queries:
            results = self._retrieve_pipeline(q, department=department, user_ctx=user_ctx)
            for rank, r in enumerate(results):
                key = self._dedup_key(r)
                if key not in fused:
                    fused[key] = r
                    order.append(key)
                fused[key]['_fuse_score'] = fused[key].get('_fuse_score', 0.0) + 1.0 / (60 + rank + 1)

        for key, item in fused.items():
            item['fuse_score'] = round(item.pop('_fuse_score', 0.0), 6)

        all_results = [fused[k] for k in order]
        all_results.sort(key=lambda x: x.get('fuse_score', 0.0), reverse=True)
        final_results = self._diversify(all_results)[:top_k]

        print(f"  [Query改写] 子查询数: {len(sub_queries)}, "
              f"参与召回: {len(merged_queries)}, 合并后结果: {len(final_results)} 条")

        return final_results, rewrite_result

    def compare_retrieval_strategies(self, query):
        """对比不同检索策略的效果（调试面板用，首次调用才构建）"""
        if self._hybrid_searcher is None:
            self._hybrid_searcher = HybridSearcher(
                self.records, self.vector_index, alpha=RRF["alpha"]
            )
        return self._hybrid_searcher.compare_strategies(query)

    # ============================================================
    # 上下文格式化 & LLM 回答生成
    # ============================================================
    def format_context(self, results):
        """
        将检索结果格式化为 LLM 上下文（带引文标记）
        """
        context_parts = []
        for i, r in enumerate(results, 1):
            record = r.get('record', {})
            score = r.get('rerank_score', r.get('rrf_score', r.get('score', 0)))
            dept = record.get('department', '')
            content = r.get('content', '')

            # 引文信息（骨架层一期固化）
            doc_name = r.get('doc_name', '') or record.get('doc_name', '')
            page_number = r.get('page_number', -1) or record.get('page_number', -1)
            parent_title = r.get('parent_title', '') or record.get('parent_title', '')

            # 引用名与 API / Gradio 共用一套规则：模型会把这里的名字抄进回答正文，
            # 所以这里脏，回答里的引注就脏（曾出现 `[示例智能…快报_1792565911, 章节：…]`）
            cite_name = build_display_name(
                r.get('base_title', '') or record.get('base_title', ''),
                r.get('version_label', '') or record.get('version_label', ''),
                doc_name,
            )

            # 组装引文标记
            cite_parts = []
            if cite_name:
                cite_parts.append(f"文档：{cite_name}")
            if page_number and page_number >= 0:
                cite_parts.append(f"P{page_number}")
            if parent_title:
                cite_parts.append(f"章节：{parent_title}")
            cite_tag = f" [{', '.join(cite_parts)}]" if cite_parts else ""

            # 优先用 answer 字段（文档 chunk），否则用 question+answer
            answer = record.get('answer', content)
            question = record.get('question', '')

            if question:
                text = f"标题：{question}\n内容：{answer}"
            else:
                text = f"内容：{answer}"

            # 标记回填 chunk
            backfill_tag = " [回填]" if r.get('is_backfill') else ""
            trunc_tag = " [截断]" if r.get('truncated') else ""

            context_parts.append(
                f"【参考资料 {i}】(来源：{dept}, 相关度：{score:.3f}{cite_tag}{backfill_tag}{trunc_tag})\n{text}"
            )

        return "\n\n---\n\n".join(context_parts)

    def generate_answer(self, query, results):
        """
        基于检索结果流式生成回答（生成器，逐 token yield；带重试 + 熔断保护）
        非流式消费方式："".join(self.generate_answer(query, results)).strip()
        """
        # 熔断检查：LLM 服务不可用时直接返回
        if llm_breaker.is_open:
            yield "抱歉，AI 服务暂时不可用，请稍后重试。"
            return

        context = self.format_context(results)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"以下是与用户问题相关的参考资料：\n\n{context}\n\n---\n\n用户问题：{query}"}
        ]

        max_retries = 2
        _t0 = time.time()
        for attempt in range(max_retries + 1):
            try:
                response = self.llm_client.chat.completions.create(
                    model=LLM_MODEL, messages=messages, stream=True,
                    temperature=LLM_TEMPERATURE
                )
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                llm_breaker._on_success()
                metrics.incr("llm_success")
                metrics.observe("llm_latency_ms", (time.time() - _t0) * 1000)
                return
            except APITimeoutError as e:
                # 超时是"挂起"类故障的信号（服务端无响应/chunk 停流），
                # 与一般 API 错误分开计数，便于定位占 worker 的问题
                llm_breaker._on_failure()
                metrics.incr("llm_error")
                metrics.incr("llm_timeout")
                if attempt < max_retries:
                    print(f"  [LLM超时] 第{attempt+1}次: {e}, {2 ** attempt}s后重试...")
                    time.sleep(2 ** attempt)
                else:
                    yield "抱歉，生成回答超时，请稍后重试。"
            except Exception as e:
                llm_breaker._on_failure()
                metrics.incr("llm_error")
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    print(f"  [API重试] 第{attempt+1}次失败: {e}, {wait_time}s后重试...")
                    time.sleep(wait_time)
                else:
                    yield f"抱歉，生成回答时出现错误，请稍后重试。({str(e)[:50]})"

    def ask(self, query, department=None, stream=True):
        """完整问答流程（兼容旧接口）"""
        print(f"\n[问答] 用户问题: {query}")
        results = self.retrieve(query, department=department)
        print(f"  检索到 {len(results)} 条相关参考资料")
        if stream:
            return self.generate_answer(query, results), results
        else:
            answer = "".join(self.generate_answer(query, results)).strip()
            return answer, results

    # ============================================================
    # 企业文档管理
    # ============================================================
    # ============================================================
    # 版本治理（通用规则：多版本制度只保留现行版可召回，换数据无需改动）
    # ============================================================
    def _enforce_version_policy(self):
        """
        在文档增/改/恢复后自动执行版本治理：
        - 文件名解析版本号 → 每个制度标记唯一现行版
        - 被新版本取代的旧版自动软删除（可 restore，审计留痕）
        - 同步清理 Milvus 中旧版向量
        幂等，失败不影响主流程
        """
        try:
            from version_policy import enforce_version_policy, cleanup_milvus_vectors, backup_registry
            report = enforce_version_policy(self.doc_manager.registry.documents)
            # 治理会改写「谁是现行版」和软删除标记。上传触发的这条路径如果解析错了版本号，
            # 旧版就被静默退场；git 未必有这中间状态的快照，所以写盘前留一份本地备份。
            if report['newly_superseded'] or report['restored']:
                backup_registry(self.doc_manager.registry.registry_path)
            self.doc_manager.registry._save()
            superseded = report['newly_superseded'] + report['already_superseded']
            if self.use_vector_index and self.vector_index and superseded:
                cleanup_milvus_vectors(self.vector_index, superseded)
            if report['newly_superseded'] or report['restored']:
                print(f"[VersionPolicy] 现行 {len(report['current_marked'])} | "
                      f"新退场 {len(report['newly_superseded'])} | 恢复 {len(report['restored'])}")
            return report
        except Exception as e:
            logger.warning(f"[VersionPolicy] 执行失败(不影响主流程): {e}")
            return None

    def add_document(self, file_path, parser=None, tags=None,
                     access_level=None, allowed_groups=None):
        """添加企业文档（增量插入，不全量重建）"""
        parsed_doc = self.doc_manager.add_document(
            file_path, parser=parser, tags=tags,
            access_level=access_level, allowed_groups=allowed_groups,
        )
        if parsed_doc.status == "success":
            # 增量插入到 Milvus（不需要全量重建）
            if self.use_vector_index and self.vector_index:
                from data_loader import load_documents_as_records
                new_records = load_documents_as_records(
                    self.doc_manager, doc_ids=[parsed_doc.doc_id]
                )
                if new_records:
                    self.vector_index.insert_records(new_records)
            # BM25 仍需全量重建（BM25 不支持增量）
            self._refresh_document_records()
            self.bm25.build(self.records)
            self._build_chunk_index_map()
            # 版本治理：新文档可能是某制度的更新版本，自动退场旧版
            # 报告写进 metadata，供上传接口把判定结果回给用户——
            # 否则「这份被当成旧版退场了」这件事用户完全看不到
            report = self._enforce_version_policy()
            if report:
                parsed_doc.metadata['version_report'] = {
                    'is_current': parsed_doc.doc_id in report.get('current_marked', []),
                    'superseded_others': report.get('newly_superseded', []),
                    'conflicts': report.get('conflicts', []),
                }
            # 治理可能软删除了旧版，records 需再刷新一次
            self._refresh_document_records()
            self.bm25.build(self.records)
            self._build_chunk_index_map()
        return parsed_doc

    def remove_document(self, doc_id):
        """删除文档（Milvus 按 doc_id 精确删除）"""
        success = self.doc_manager.remove_document(doc_id)
        if success:
            # Milvus 按 doc_id 删除对应 chunk
            if self.use_vector_index and self.vector_index:
                self.vector_index.delete_by_doc_id(doc_id)
            # BM25 仍需全量重建
            self._refresh_document_records()
            self.bm25.build(self.records)
            self._build_chunk_index_map()
        return success

    def update_document(self, doc_id, new_file_path, parser=None):
        """
        更新文档（删旧 chunk → 重解析 → 重入库，version +1）
        保留原 doc_id、权限、标签
        """
        # 1. 先从 Milvus 删除旧 chunk
        if self.use_vector_index and self.vector_index:
            self.vector_index.delete_by_doc_id(doc_id)

        # 2. 重新解析并入库
        parsed_doc = self.doc_manager.update_document(doc_id, new_file_path, parser=parser)

        if parsed_doc.status == "success":
            # 3. 增量插入新 chunk 到 Milvus
            if self.use_vector_index and self.vector_index:
                from data_loader import load_documents_as_records
                new_records = load_documents_as_records(
                    self.doc_manager, doc_ids=[doc_id]
                )
                if new_records:
                    self.vector_index.insert_records(new_records)

            # 4. 刷新 BM25 和 chunk 索引
            self._refresh_document_records()
            self.bm25.build(self.records)
            self._build_chunk_index_map()
            # 版本治理：更新后的文档即现行版，旧版文件若仍在库中则自动退场
            self._enforce_version_policy()
            self._refresh_document_records()
            self.bm25.build(self.records)
            self._build_chunk_index_map()

        return parsed_doc

    def restore_document(self, doc_id):
        """
        恢复软删除的文档（Milvus 中 is_deleted 重新置为 false）
        """
        success = self.doc_manager.restore_document(doc_id)
        if success:
            # 版本治理：恢复的文档若是旧版，会自动顶替现行版并把原现行版退场
            self._enforce_version_policy()
            # 恢复后需重新加载 records 并重建 BM25
            self._refresh_document_records()
            self.bm25.build(self.records)
            self._build_chunk_index_map()

            # Milvus 中该文档的 is_deleted 需要更新
            # 由于 Milvus Lite 不支持 partial update，采用删旧插新策略
            if self.use_vector_index and self.vector_index:
                from data_loader import load_documents_as_records
                # 先删除旧数据（is_deleted=true 的）
                self.vector_index.delete_by_doc_id(doc_id)
                # 重新插入（is_deleted=false）
                records = load_documents_as_records(
                    self.doc_manager, doc_ids=[doc_id]
                )
                if records:
                    self.vector_index.insert_records(records)
        return success

    def _refresh_document_records(self):
        """从文档管理器重新加载文档 chunks 到 records"""
        from data_loader import load_documents_as_records
        csv_records = [r for r in self.records if r.get('source', 'csv') == 'csv']
        doc_records = load_documents_as_records(self.doc_manager)
        self.records = csv_records + doc_records
        # records 换了，懒加载的 HybridSearcher 持有的是旧引用，必须失效
        self._hybrid_searcher = None
        # 缓存同理：语料变了，旧答案不该再命中
        if getattr(self, 'cache_manager', None) is not None:
            self.cache_manager.version = self._corpus_fingerprint()
        print(f"[KnowledgeRetriever] 记录已刷新: {len(self.records)} 条 "
              f"(CSV: {len(csv_records)}, 文档: {len(doc_records)})")

    def get_document_stats(self):
        """获取文档库统计"""
        return self.doc_manager.get_stats()

    def list_documents(self):
        """列出所有已入库文档"""
        return self.doc_manager.list_documents()

    # ============================================================
    # 图片问答
    # ============================================================
    def ask_with_image(self, image_path, query="", department=None, stream=False):
        """带图片的问答流程"""
        print(f"\n[图片问答] 用户问题: {query}")
        print(f"  图片: {image_path}")

        print("  [Step 1] 分析图片内容...")
        image_analysis = self.image_analyzer.analyze_image(image_path, query)
        print(f"  图片分析完成: {image_analysis[:80]}...")

        search_query = self.image_analyzer.get_search_query(image_analysis, query)
        print(f"  [Step 2] 检索查询: {search_query[:80]}...")

        results = self.retrieve(search_query, department=department)
        print(f"  检索到 {len(results)} 条相关参考资料")

        enhanced_query = f"{query}\n\n[图片分析结果]: {image_analysis}"

        if stream:
            return self.generate_answer(enhanced_query, results), results, image_analysis
        else:
            answer = "".join(self.generate_answer(enhanced_query, results)).strip()
            return answer, results, image_analysis



