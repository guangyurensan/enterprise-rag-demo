# 文档入库管线规格文档

> 版本: 1.0 | 日期: 2026-10-20 | 状态: 基线固化

## 1. 管线流程 (DataPipeline)

```
文件上传 → 前置粗筛 → 路由调度 → 解析 → 归一适配 → 差异化清洗 → 质量打分 → 智能分块 → Chunk去重 → 写入向量库
```

### 1.1 各阶段职责

| 阶段 | 模块 | 职责 |
|------|------|------|
| 前置粗筛 | prescreen.py | 文件大小/格式校验、SHA256 去重、扫描件检测 |
| 路由调度 | router.py | 按文件类型+业务规则选择解析器 |
| 解析 | document_parser.py | 调用 MinerU/Docling/Unstructured 解析 |
| 归一适配 | block_adapter.py | 统一各解析器输出为 UnifiedBlock 格式 |
| 差异化清洗 | cleaner.py | 按文档类型执行清洗规则 (去噪、格式化) |
| 质量打分 | quality_scorer.py | 评估文档质量 (清晰度、完整度) |
| 智能分块 | smart_chunker.py | 按语义边界分块，控制 chunk 大小 |
| Chunk去重 | dedup.py | 内容哈希去重 |
| 写入 | vector_index.py + BM25 | Milvus 增量插入 + BM25 全量重建 |

## 2. 三管线解析器

### 2.1 解析器能力矩阵

| 解析器 | 支持格式 | 擅长场景 |
|--------|----------|----------|
| MinerU | PDF | 论文、财报、中文密集排版、公式、扫描件 |
| Docling | PDF, DOCX, PPTX | 合同、英文文档、结构化文档 |
| Unstructured | PDF, DOCX, XLSX, HTML, EML, CSV, TXT 等 | Excel、HTML、邮件、通用办公 |

### 2.2 路由规则 (pipeline_config.py)

- **自动选择**: 根据文件扩展名 + 文档类型标签自动匹配最优解析器
- **手动指定**: 用户可在上传时指定解析器
- **降级**: 首选解析器不可用时，自动降级到备选

## 3. 文档管理 (DocumentManager)

### 3.1 文档注册表

`document_store/document_registry.json` 记录所有已入库文档的元数据：

```json
{
  "doc_id": "uuid",
  "filename": "文件名",
  "file_type": ".pdf",
  "parser_used": "mineru",
  "chunk_count": 15,
  "version": 1,
  "parse_status": "success",
  "access_level": "internal",
  "allowed_groups": "all",
  "is_deleted": false,
  "tags": ["人事", "制度"],
  "uploaded_at": "2026-10-20T10:00:00"
}
```

### 3.2 文档操作

| 操作 | 说明 | Milvus 影响 |
|------|------|-------------|
| add_document | 上传并解析，增量插入 | insert_records |
| remove_document | 软删除 (is_deleted=true) | delete_by_doc_id |
| update_document | 删旧 chunk -> 重解析 -> 重入库 | delete + insert |
| restore_document | 恢复软删除 (is_deleted=false) | delete + insert (重新插入) |

### 3.3 版本管理

- 每次 update_document，version +1
- 保留版本号用于追溯

## 4. 旁路机制

### 4.1 指标埋点

- 管线每次执行的指标记录到 `pipeline_logs/pipeline_metrics.jsonl`
- 指标: 文件名、解析器、耗时、chunk 数、质量分

### 4.2 死信队列

- 解析失败的任务写入 `dead_letters/ingestion_failures.json`
- 支持重跑和统计
- 由 `resilience.py` 的 DeadLetterQueue 管理

## 5. 数据存储

| 目录 | 用途 |
|------|------|
| document_store/uploads/ | 原始上传文件 |
| parsed_cache/ | 解析结果 JSON 缓存 (SHA256 命名) |
| milvus_data.db/ | Milvus Lite 向量数据 |
