# 检索模块规格文档

> 版本: 1.0 | 日期: 2026-10-20 | 状态: 基线固化

## 1. 检索主干 (KnowledgeRetriever._retrieve_pipeline)

### 1.1 流程步骤

| 步骤 | 组件 | 输入 | 输出 | 异常处理 |
|------|------|------|------|----------|
| 0. 安全检查 | SecurityGateway | 原始 query | is_safe, clean_query | 注入 -> 返回空结果 |
| 1. 多路召回 | _vector_recall + _bm25_recall | clean_query, user_ctx | vector_results, bm25_results | 各自独立容错 |
| 2. RRF 融合 | _merge_dedup | 两路结果 | merged (去重+RRF分数) | 空则返回 |
| 3. Rerank | DashScopeReranker | merged | reranked (top_n) | 熔断降级 -> RRF排序 |
| 4. Truncated 回填 | _backfill_truncated | reranked | 追加相邻 chunk | 无则跳过 |
| 5. Token 预算 | _apply_token_budget | 全部结果 | final (截断) | 硬截断 |
| 6. 上下文脱敏 | SecurityGateway | final | 脱敏后结果 | eval_mode 跳过 |
| 7. 审计日志 | AuditLogger | 全部元数据 | 持久化 | eval_mode 跳过 |

### 1.2 向量召回 (_vector_recall)

- **引擎**: Milvus，引擎侧 ACL 过滤 (filter_expr)
- **top_k**: 配置 `VECTOR_RECALL["top_k"]` = 20，实际 fetch `top_k * 2`
- **最低分数**: `VECTOR_RECALL["min_score"]` = 0.35
- **ACL**: 通过 `AccessPolicy.build_filter_expr(user_ctx)` 生成 Milvus 过滤表达式，不可绕过
- **部门过滤**: 可选，追加到 filter_expr

### 1.3 BM25 召回 (_bm25_recall)

- **引擎**: 自实现 BM25 (jieba 分词)
- **top_k**: 配置 `BM25_RECALL["top_k"]` = 10
- **ACL**: 内存侧过滤 (BM25 不支持引擎侧过滤)
- **部门过滤**: 可选

### 1.4 RRF 融合 (_merge_dedup)

- **公式**: `RRF_score(d) = (1-alpha) * 1/(k+rank_bm25) + alpha * 1/(k+rank_vector)`
- **参数**: k=60, alpha=0.5 (BM25 权重)
- **去重键**: `(doc_id, chunk_index)` 或 `idx_{index}` (CSV 记录)

### 1.5 Rerank (_do_rerank)

- **模型**: DashScope gte-rerank
- **top_n**: 配置 `RERANK["top_n"]` = 6
- **降级策略**: 熔断器 OPEN 或调用失败时，按 RRF 分数排序取 top_n
- **熔断器**: rerank_breaker (阈值 3 次, 恢复 30s)

### 1.6 Truncated 回填 (_backfill_truncated)

- **开关**: `TRUNCATED_BACKFILL["enabled"]` = False (默认关闭)
- **策略**: 对 truncated=True 的 chunk，向前回填 before_n=2 个、向后回填 after_n=2 个相邻 chunk
- **约束**: 同文档内，不重复已命中的 chunk

### 1.7 Token 预算 (_apply_token_budget)

- **上限**: `TOKEN_BUDGET["max_context_token"]` = 6000
- **估算**: `token ≈ len(content) * 0.55`
- **策略**: 按 rerank 顺序累加，超限即停止 (硬截断)

## 2. Query 改写 (QueryRewriter)

- **输入**: 用户原始 query + 对话历史
- **输出**: 改写类型 + 改写后 query + 子查询列表
- **场景**: 多轮对话上下文补全、指代消解

## 3. 缓存 (CacheManager)

- **存储**: JSON 文件 (`cache/qa_cache.json`)
- **TTL**: 86400s (24小时)
- **键**: (query, department) -> answer
- **命中**: 直接返回缓存回答，跳过检索和生成

## 4. 对外接口

| 方法 | 用途 | 参数 |
|------|------|------|
| `retrieve(query, top_k, department, user_ctx)` | 基础检索 | 走新主干 |
| `retrieve_hybrid(query, top_k, department, user_ctx, eval_mode)` | 混合检索 | eval_mode 跳过脱敏/审计 |
| `retrieve_with_rewrite(query, history, top_k, department, use_hybrid, user_ctx)` | 带 Query 改写的检索 | app.py 主入口 |
| `generate_answer(query, results, stream, max_retries)` | LLM 回答生成 | 流式/非流式 |
| `ask(query, department, stream)` | 完整问答 | 兼容旧接口 |
| `ask_with_image(image_path, query, department, stream)` | 图片问答 | 图片分析+检索+生成 |

## 5. 检索参数配置 (retrieval_config.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| RERANK.enabled | True | Rerank 开关 |
| RERANK.top_n | 6 | Rerank 后截取数 |
| VECTOR_RECALL.top_k | 20 | 向量召回候选数 |
| VECTOR_RECALL.min_score | 0.35 | 向量最低相似度 |
| BM25_RECALL.enabled | True | BM25 开关 |
| BM25_RECALL.top_k | 10 | BM25 召回候选数 |
| RRF.k | 60 | RRF 常数 |
| RRF.alpha | 0.5 | BM25 权重 |
| TRUNCATED_BACKFILL.enabled | False | 回填开关 |
| TOKEN_BUDGET.max_context_token | 6000 | 上下文 token 上限 |
