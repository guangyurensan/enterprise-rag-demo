# -*- coding: utf-8 -*-
"""
企业级知识库 - 向量索引构建模块
使用 DashScope text-embedding-v4 生成向量，Milvus 构建索引
支持 CSV 问答数据 + 企业文档 chunks

Milvus 优势：
- 原生支持元数据过滤（权限、部门、文档ID等）
- 增量插入/删除，无需全量重建
- 本地文件（Milvus Lite）→ Docker 无缝迁移
"""

import os
import logging
import numpy as np
from openai import OpenAI
from pymilvus import MilvusClient, DataType
from resilience import embedding_breaker, CircuitOpenError
from config import (
    EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_BATCH_SIZE,
    MILVUS_URI, MILVUS_COLLECTION, TOP_K
)

logger = logging.getLogger(__name__)


def get_client():
    """获取 DashScope OpenAI 兼容客户端"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError(
            "未设置 DASHSCOPE_API_KEY 环境变量！\n"
            "请设置: $env:DASHSCOPE_API_KEY='your-api-key'\n"
            "获取地址: https://dashscope.console.aliyun.com/"
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )


def get_embeddings(client, texts, batch_size=None):
    """
    批量获取文本的 Embedding 向量（带熔断保护）
    :param client: OpenAI 客户端
    :param texts: 文本列表
    :param batch_size: 每批大小
    :return: 向量列表
    :raises: CircuitOpenError 如果 Embedding 服务熔断
    """
    if batch_size is None:
        batch_size = EMBEDDING_BATCH_SIZE

    # 熔断检查
    if embedding_breaker.is_open:
        raise CircuitOpenError(f"Embedding 服务熔断中，状态: {embedding_breaker.state.name}")

    all_embeddings = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        try:
            response = embedding_breaker.call(
                client.embeddings.create,
                model=EMBEDDING_MODEL,
                input=batch,
                dimensions=EMBEDDING_DIM,
                encoding_format="float"
            )
            for item in response.data:
                all_embeddings.append(item.embedding)
        except CircuitOpenError:
            logger.error(f"[VectorIndex] Embedding 熔断，已处理 {i}/{total}，中止")
            raise
        except Exception as e:
            logger.error(f"[VectorIndex] Embedding 批次 {i//batch_size + 1} 失败: {e}")
            raise

        # 进度显示
        if (i + batch_size) % 60 == 0 or (i + batch_size) >= total:
            print(f"  Embedding 进度: {min(i + batch_size, total)}/{total}")

    return all_embeddings


# ============================================================
# Milvus Collection Schema 定义
# ============================================================
def _build_schema() -> dict:
    """
    定义 Milvus Collection Schema
    所有检索需要的元数据都作为标量字段，支持过滤
    """
    schema = MilvusClient.create_schema(auto_id=True, enable_dynamic_field=False)

    # 主键（自增）
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    # 向量字段
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)

    # ---- 标量字段（元数据） ----
    schema.add_field("department", DataType.VARCHAR, max_length=128, default="")
    schema.add_field("question", DataType.VARCHAR, max_length=2048, default="")
    schema.add_field("answer", DataType.VARCHAR, max_length=65535, default="")
    schema.add_field("source", DataType.VARCHAR, max_length=32, default="csv")

    # 文档 chunk 专用字段
    schema.add_field("doc_id", DataType.VARCHAR, max_length=128, default="")
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=128, default="")
    schema.add_field("chunk_index", DataType.INT64, default=0)
    schema.add_field("page_number", DataType.INT64, default=-1)
    schema.add_field("chunk_type", DataType.VARCHAR, max_length=32, default="text")
    schema.add_field("parent_title", DataType.VARCHAR, max_length=512, default="")
    schema.add_field("truncated", DataType.BOOL, default=False)
    schema.add_field("doc_name", DataType.VARCHAR, max_length=512, default="")

    # 权限与生命周期字段（骨架层一期固化）
    schema.add_field("access_level", DataType.VARCHAR, max_length=32, default="internal")
    schema.add_field("allowed_groups", DataType.VARCHAR, max_length=1024, default="all")
    schema.add_field("is_deleted", DataType.BOOL, default=False)
    schema.add_field("version", DataType.INT64, default=1)

    return schema


def _build_index_params() -> dict:
    """Milvus 索引参数（IVF_FLAT + COSINE）"""
    return {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    }


# ============================================================
# VectorIndex（Milvus 版）
# ============================================================
class VectorIndex:
    """Milvus 向量索引管理器"""

    def __init__(self, uri: str = None, collection_name: str = None):
        """
        初始化 Milvus 连接
        :param uri: Milvus URI（默认从 config 读取，本地文件）
        :param collection_name: Collection 名称
        """
        self.uri = uri or MILVUS_URI
        self.collection_name = collection_name or MILVUS_COLLECTION
        self.client = MilvusClient(uri=self.uri)
        self._embedding_client = get_client()

        print(f"[VectorIndex] Milvus 连接成功: {self.uri}")
        print(f"[VectorIndex] Collection: {self.collection_name}")

    def _ensure_collection(self):
        """确保 Collection 存在，不存在则创建"""
        if self.client.has_collection(self.collection_name):
            return

        schema = _build_schema()
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",
            params={"nlist": 128},
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        print(f"[VectorIndex] Collection '{self.collection_name}' 已创建")

    def build_index(self, records):
        """
        全量重建索引：清空旧数据 → 插入全部记录
        :param records: 记录列表 [{question/full_text, answer, department, source, ...}, ...]
        """
        print(f"\n[VectorIndex] 全量重建索引，共 {len(records)} 条记录...")

        # 删除旧 Collection 并重建
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        self._ensure_collection()

        if not records:
            print("[VectorIndex] 无记录，跳过插入")
            return

        # 提取索引文本
        index_texts = []
        for r in records:
            if r.get('question'):
                index_texts.append(r['question'])
            elif r.get('full_text'):
                index_texts.append(r['full_text'][:200])
            else:
                index_texts.append(r.get('answer', '')[:200])

        # 生成 Embedding
        print("[VectorIndex] 生成 Embedding 向量...")
        embeddings = get_embeddings(self._embedding_client, index_texts)

        # 组装 Milvus 数据（确保所有字段类型正确，None 替换为默认值）
        data = []
        for r, emb in zip(records, embeddings):
            # 安全取值：None -> 默认值
            page_num = r.get('page_number')
            if page_num is None:
                page_num = -1
            chunk_idx = r.get('chunk_index')
            if chunk_idx is None:
                chunk_idx = 0
            is_del = r.get('is_deleted')
            if is_del is None:
                is_del = False
            ver = r.get('version')
            if ver is None:
                ver = 1
            trunc = r.get('truncated')
            if trunc is None:
                trunc = False

            row = {
                "embedding": emb,
                "department": r.get('department') or '',
                "question": r.get('question') or '',
                "answer": r.get('answer') or '',
                "source": r.get('source') or 'csv',
                "doc_id": r.get('doc_id') or '',
                "chunk_id": r.get('chunk_id') or '',
                "chunk_index": int(chunk_idx),
                "page_number": int(page_num),
                "chunk_type": r.get('chunk_type') or 'text',
                "parent_title": r.get('parent_title') or '',
                "truncated": bool(trunc),
                "doc_name": r.get('doc_name') or '',
                "access_level": r.get('access_level') or 'internal',
                "allowed_groups": r.get('allowed_groups') or 'all',
                "is_deleted": bool(is_del),
                "version": int(ver),
            }
            data.append(row)

        # 批量插入（Milvus 单次上限约 1000 条）
        batch_size = 500
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            result = self.client.insert(
                collection_name=self.collection_name,
                data=batch,
            )
            print(f"  插入批次 {i//batch_size + 1}: {result.get('insert_count', 0)} 条")

        # Flush 确保数据可见
        self.client.flush(self.collection_name)
        print("[VectorIndex] 数据已 flush")

        total = self.count()
        print(f"[VectorIndex] 索引构建完成！共 {total} 条向量")

    def load_index(self):
        """
        检查 Collection 是否存在并加载到内存
        :return: True 如果 Collection 存在且有数据
        """
        if not self.client.has_collection(self.collection_name):
            print("[VectorIndex] Collection 不存在，需要重新构建")
            return False

        total = self.count()
        if total == 0:
            print("[VectorIndex] Collection 为空，需要重新构建")
            return False

        # Milvus 要求搜索前必须将 collection 加载到内存
        self.client.load_collection(self.collection_name)
        print(f"[VectorIndex] Collection 已加载到内存，共 {total} 条向量")
        return True

    def search(self, query, top_k=None, filter_expr=None):
        """
        向量检索（支持元数据过滤）
        :param query: 查询文本
        :param top_k: 返回数量
        :param filter_expr: Milvus 过滤表达式，如 'department == "人事"'
            用于权限过滤、部门过滤等
        :return: 检索结果列表 [{record, score, index}, ...]
        """
        if not self.client.has_collection(self.collection_name):
            raise RuntimeError("Collection 未创建，请先调用 build_index()")

        if top_k is None:
            top_k = TOP_K

        # 生成查询向量
        query_response = self._embedding_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query],
            dimensions=EMBEDDING_DIM,
            encoding_format="float"
        )
        query_vec = [query_response.data[0].embedding]

        # 搜索参数
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

        # 输出字段
        output_fields = [
            "department", "question", "answer", "source",
            "doc_id", "chunk_id", "chunk_index", "page_number",
            "chunk_type", "parent_title", "truncated", "doc_name",
            "access_level", "allowed_groups", "is_deleted", "version",
        ]

        # 执行搜索
        kwargs = {
            "collection_name": self.collection_name,
            "data": query_vec,
            "limit": top_k,
            "output_fields": output_fields,
            "search_params": search_params,
        }
        if filter_expr:
            kwargs["filter"] = filter_expr

        results_raw = self.client.search(**kwargs)

        # 转换为兼容格式
        results = []
        for hit in results_raw[0]:
            entity = hit.get("entity", {})
            record = {
                "department": entity.get("department", ""),
                "question": entity.get("question", ""),
                "answer": entity.get("answer", ""),
                "source": entity.get("source", "csv"),
                "doc_id": entity.get("doc_id", ""),
                "chunk_id": entity.get("chunk_id", ""),
                "chunk_index": entity.get("chunk_index", 0),
                "page_number": entity.get("page_number", -1),
                "chunk_type": entity.get("chunk_type", "text"),
                "parent_title": entity.get("parent_title", ""),
                "truncated": entity.get("truncated", False),
                "doc_name": entity.get("doc_name", ""),
                "access_level": entity.get("access_level", "internal"),
                "allowed_groups": entity.get("allowed_groups", "all"),
                "is_deleted": entity.get("is_deleted", False),
                "version": entity.get("version", 1),
            }
            results.append({
                "record": record,
                "score": float(hit.get("distance", 0)),
                "index": int(hit.get("id", 0)),
            })

        return results

    # ============================================================
    # 增量操作（Milvus 独有优势）
    # ============================================================
    def insert_records(self, records):
        """
        增量插入记录（不重建索引）
        :param records: 记录列表
        """
        if not records:
            return

        self._ensure_collection()

        # 生成 Embedding
        index_texts = []
        for r in records:
            if r.get('question'):
                index_texts.append(r['question'])
            elif r.get('full_text'):
                index_texts.append(r['full_text'][:200])
            else:
                index_texts.append(r.get('answer', '')[:200])

        embeddings = get_embeddings(self._embedding_client, index_texts)

        data = []
        for r, emb in zip(records, embeddings):
            row = {
                "embedding": emb,
                "department": r.get('department', ''),
                "question": r.get('question', ''),
                "answer": r.get('answer', ''),
                "source": r.get('source', 'csv'),
                "doc_id": r.get('doc_id', ''),
                "chunk_id": r.get('chunk_id', ''),
                "chunk_index": r.get('chunk_index', 0),
                "page_number": r.get('page_number', -1),
                "chunk_type": r.get('chunk_type', 'text'),
                "parent_title": r.get('parent_title', ''),
                "truncated": r.get('truncated', False),
                "doc_name": r.get('doc_name', ''),
                "access_level": r.get('access_level', 'internal'),
                "allowed_groups": r.get('allowed_groups', 'all'),
                "is_deleted": r.get('is_deleted', False),
                "version": r.get('version', 1),
            }
            data.append(row)

        self.client.insert(
            collection_name=self.collection_name,
            data=data,
        )
        print(f"[VectorIndex] 增量插入 {len(records)} 条记录")

    def delete_by_doc_id(self, doc_id: str):
        """
        按 doc_id 删除所有相关 chunk（软删除的前置步骤）
        :param doc_id: 文档 ID
        :return: 删除的数量
        """
        if not self.client.has_collection(self.collection_name):
            return 0

        filter_expr = f'doc_id == "{doc_id}"'
        # Milvus delete by filter
        self.client.delete(
            collection_name=self.collection_name,
            filter=filter_expr,
        )
        print(f"[VectorIndex] 已删除 doc_id={doc_id} 的全部 chunk")

    def update_doc_access(self, doc_id: str, access_level: str,
                          allowed_groups: str = None) -> int:
        """
        同步文档密级到 Milvus chunk（2026-10-24 新增）

        背景：ACL 引擎侧过滤读的是 Milvus 行上的 access_level，
        只改 registry 不过滤器照旧命中（原 update_access 注释里
        "Milvus 需同步更新"的 TODO 从未实现）。
        实现：Milvus auto_id 主键不支持 update，走 查询→删除→重插，
        重插不带 id 让 auto_id 重新分配。

        :param doc_id: 文档 ID
        :param access_level: 新密级
        :param allowed_groups: 新允许组（None=保持原值）
        :return: 更新的 chunk 数
        """
        if not self.client.has_collection(self.collection_name):
            return 0

        rows = self.client.query(
            collection_name=self.collection_name,
            filter=f'doc_id == "{doc_id}"',
            output_fields=["*"],
        )
        if not rows:
            return 0

        # 先取回全部字段（含向量）再删除，避免窗口期数据丢失
        new_rows, seen_chunk = [], set()
        for r in rows:
            r.pop('id', None)      # auto_id 主键不可重插
            r.pop('distance', None)
            r['access_level'] = access_level
            if allowed_groups is not None:
                r['allowed_groups'] = allowed_groups
            # 按 chunk_id 去重（防御历史重复行被原样继承）
            cid = r.get('chunk_id', '')
            if cid and cid in seen_chunk:
                continue
            seen_chunk.add(cid)
            new_rows.append(r)

        self.client.delete(
            collection_name=self.collection_name,
            filter=f'doc_id == "{doc_id}"',
        )
        self.client.insert(collection_name=self.collection_name, data=new_rows)
        return len(new_rows)

    def count(self) -> int:
        """获取当前向量总数"""
        if not self.client.has_collection(self.collection_name):
            return 0
        stats = self.client.get_collection_stats(self.collection_name)
        return int(stats.get("row_count", 0))
