# 系统架构规格文档

> 版本: 1.0 | 日期: 2026-10-20 | 状态: 基线固化

## 1. 系统概述

企业级知识库问答系统，基于 RAG（检索增强生成）架构，支持多格式企业文档上传、解析、检索与智能问答。

### 1.1 核心能力

| 能力 | 说明 |
|------|------|
| 三管线文档解析 | MinerU / Docling / Unstructured，按文件类型智能路由 |
| 混合检索 | BM25 关键词 + 向量语义，RRF 融合 |
| Rerank 重排 | DashScope gte-rerank 精排 |
| LLM 回答生成 | DeepSeek-V3，流式输出 |
| 图片问答 | Qwen-VL 视觉模型分析 + 检索 |
| ACL 权限控制 | 四级密级 (public/internal/restricted/confidential) |
| 安全防护 | Prompt 注入检测 + PII 脱敏 |
| 容错韧性 | 熔断器 + 限流器 + 死信队列 |
| 评测框架 | Recall@K / MRR / 关键词命中率 + A/B 灰度对比 |

## 2. 模块架构

```
app.py (Gradio UI)
  |
  +-- retriever.py (KnowledgeRetriever) -- 检索主干
  |     |-- vector_index.py (VectorIndex) -- Milvus 向量索引
  |     |-- hybrid_search.py (BM25Index + HybridSearcher) -- BM25 + 混合检索
  |     |-- reranker.py (DashScopeReranker) -- 重排
  |     |-- query_rewriter.py (QueryRewriter) -- Query 改写
  |     |-- image_analyzer.py (ImageAnalyzer) -- 图片分析
  |     |-- cache_manager.py (CacheManager) -- 缓存
  |     |-- security.py (SecurityGateway) -- 安全防护
  |     |-- access_control.py (AccessPolicy + AuditLogger) -- ACL + 审计
  |     |-- resilience.py (CircuitBreaker + RateLimiter + DLQ) -- 容错
  |     +-- retrieval_config.py -- 检索参数配置
  |
  +-- document_manager.py (DocumentManager) -- 文档管理
  |     +-- pipeline.py (DataPipeline) -- 文档入库管线
  |           |-- prescreen.py -- 前置粗筛
  |           |-- router.py -- 解析器路由
  |           |-- document_parser.py -- 解析器适配
  |           |-- block_adapter.py -- 块归一化
  |           |-- cleaner.py -- 差异化清洗
  |           |-- quality_scorer.py -- 质量打分
  |           |-- smart_chunker.py -- 智能分块
  |           |-- dedup.py -- Chunk 去重
  |           +-- pipeline_config.py -- 管线配置
  |
  +-- evaluator.py (EvalSetManager + EvalRunner + GrayComparator) -- 评测
  +-- config.py -- 全局配置
```

## 3. 数据流

### 3.1 问答主流程

```
用户输入
  |
  v
[安全检查] Prompt 注入检测 -> 拒绝 / 清洗
  |
  v
[Query 改写] 上下文理解 -> 子查询拆分
  |
  v
[多路召回] 向量召回(Milvus) + BM25 召回 (并行，带 ACL 过滤)
  |
  v
[RRF 融合] 合并去重
  |
  v
[Rerank] gte-rerank 精排 (熔断降级: 按 RRF 排序)
  |
  v
[Truncated 回填] 相邻 chunk 补全 (可选)
  |
  v
[Token 预算] 截断至 max_context_token
  |
  v
[上下文脱敏] PII 替换
  |
  v
[审计日志] 记录检索事件
  |
  v
[LLM 生成] DeepSeek-V3 流式回答
  |
  v
[缓存] 缓存问答对
  |
  v
输出回答 + 参考资料
```

### 3.2 文档入库流程

```
文件上传
  |
  v
[前置粗筛] 文件大小/格式/重复检查
  |
  v
[路由调度] 按文件类型选择解析器 (MinerU/Docling/Unstructured)
  |
  v
[解析] 三管线之一解析文档
  |
  v
[归一适配] BlockAdapter 统一格式
  |
  v
[差异化清洗] 按文档类型执行清洗规则
  |
  v
[质量打分] 评估文档质量
  |
  v
[智能分块] 按语义边界分块
  |
  v
[Chunk 去重] 内容去重
  |
  v
[写入向量库] Milvus 增量插入 + BM25 重建
```

## 4. 技术栈

| 组件 | 技术选型 |
|------|----------|
| Web UI | Gradio 4.x |
| 向量数据库 | Milvus Lite (可切换 Docker) |
| Embedding | DashScope text-embedding-v4 (1024维) |
| LLM | DeepSeek-V3 |
| 视觉模型 | Qwen-VL-Max |
| Rerank | DashScope gte-rerank |
| 中文分词 | jieba |
| 关键词检索 | BM25 (自实现) |
| 文档解析 | MinerU / Docling / Unstructured |

## 5. 外部依赖

| 服务 | 用途 | 容错机制 |
|------|------|----------|
| DashScope API | Embedding + Rerank | 熔断器 (embedding_breaker / rerank_breaker) |
| DeepSeek API | LLM 回答生成 | 熔断器 (llm_breaker) + 指数退避重试 |
| Milvus Lite | 向量存储 | 本地文件，无网络依赖 |

## 6. 配置管理

所有配置集中在 `config.py` 和 `retrieval_config.py`，支持环境变量覆盖：

- `DEEPSEEK_API_KEY` — DeepSeek API 密钥
- `DASHSCOPE_API_KEY` — DashScope API 密钥
- 其余参数均为代码内常量

## 7. 存储结构

```
surgical_qa/
  document_store/       # 文档存储
    uploads/            # 上传文件
    document_registry.json  # 文档注册表
  parsed_cache/         # 解析结果缓存
  milvus_data.db/       # Milvus Lite 数据
  cache/qa_cache.json   # QA 缓存
  eval_sets/            # 评测集
  eval_reports/         # 评测报告
  audit_logs/           # 审计日志 (JSONL, 按天分割)
  dead_letters/         # 死信队列
  pipeline_logs/        # 管线指标日志
```
