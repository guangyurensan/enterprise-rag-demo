# -*- coding: utf-8 -*-
"""
企业级知识库 - 文档管理模块
负责文档的上传、解析、存储、索引构建、版本管理等

核心功能:
1. 文档上传与去重（MD5 校验）
2. 双管线自动解析（MinerU / Docling）
3. 解析结果持久化存储
4. 文档生命周期管理（增删改查）
5. 索引增量更新
"""

import os
import json
import shutil
import hashlib
import logging
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from document_parser import (
    MultiPipelineDispatcher, ParsedDocument, ParsedChunk,
    BaseParser, DocumentClassifier
)
from pipeline import DataPipeline, PipelineResult
from metrics import metrics

logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================
# 路径与切块参数统一从 config 取：这里原先自己算了一遍，改 config 不生效
from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENT_STORE_DIR, PARSED_CACHE_DIR,
)
DOCUMENT_REGISTRY_FILE = os.path.join(
    DOCUMENT_STORE_DIR, 'document_registry.json'
)


# ============================================================
# 文档注册表
# ============================================================
class DocumentRegistry:
    """
    文档注册表 - 跟踪所有已入库文档的状态

    记录信息:
    - doc_id, filename, file_path, file_hash
    - parser_used, parse_status, parse_time
    - chunk_count, created_at, updated_at
    - tags (用户标签)
    """

    def __init__(self, registry_path: str = DOCUMENT_REGISTRY_FILE):
        self.registry_path = registry_path
        self.documents: Dict[str, Dict] = {}  # doc_id -> info
        self._load()

    def _load(self):
        """从磁盘加载注册表"""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.documents = data.get('documents', {})
                logger.info(f"文档注册表加载完成，共 {len(self.documents)} 个文档")
            except Exception as e:
                logger.error(f"加载文档注册表失败: {e}")
                self.documents = {}

    def _save(self):
        """保存注册表到磁盘

        先写临时文件再原子替换：直接覆盖写如果中途断电/写满，registry 会被截断，
        而它是知识库唯一的状态源，截断等于整库文档状态丢失且无原文可重建。
        """
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        data = {
            'documents': self.documents,
            'total': len(self.documents),
            'last_updated': datetime.now().isoformat()
        }
        tmp = self.registry_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.registry_path)

    def register(self, parsed_doc: ParsedDocument, tags: List[str] = None,
                 access_level: str = None, allowed_groups: str = None):
        """注册一个新文档"""
        self.documents[parsed_doc.doc_id] = {
            'doc_id': parsed_doc.doc_id,
            'filename': parsed_doc.filename,
            'file_path': parsed_doc.file_path,
            'file_type': parsed_doc.file_type,
            'file_size': parsed_doc.file_size,
            'file_hash': parsed_doc.file_hash,
            'parser_used': parsed_doc.parser_used,
            'parse_status': parsed_doc.status,
            'parse_time': parsed_doc.parse_time,
            'chunk_count': len(parsed_doc.chunks),
            'tags': tags or [],
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'metadata': parsed_doc.metadata,
            'error_msg': parsed_doc.error_msg,
            'is_deleted': False,
            # 权限字段（骨架层一期固化）
            'access_level': access_level or 'internal',
            'allowed_groups': allowed_groups or 'all',
            'version': 1,
        }
        self._save()

    def update(self, doc_id: str, updates: Dict):
        """更新文档信息"""
        if doc_id in self.documents:
            self.documents[doc_id].update(updates)
            self.documents[doc_id]['updated_at'] = datetime.now().isoformat()
            self._save()

    def soft_delete(self, doc_id: str) -> Optional[Dict]:
        """软删除文档（标记 is_deleted，保留审计痕迹）"""
        if doc_id not in self.documents:
            return None
        self.documents[doc_id]['is_deleted'] = True
        self.documents[doc_id]['deleted_at'] = datetime.now().isoformat()
        self.documents[doc_id]['updated_at'] = datetime.now().isoformat()
        self._save()
        return self.documents[doc_id]

    def restore(self, doc_id: str) -> Optional[Dict]:
        """恢复软删除的文档"""
        if doc_id not in self.documents:
            return None
        if not self.documents[doc_id].get('is_deleted'):
            return self.documents[doc_id]  # 本来就没删除
        self.documents[doc_id]['is_deleted'] = False
        self.documents[doc_id].pop('deleted_at', None)
        self.documents[doc_id]['updated_at'] = datetime.now().isoformat()
        self._save()
        return self.documents[doc_id]

    def purge(self, doc_id: str) -> Optional[Dict]:
        """物理移除文档注册信息（不可恢复，慎用）"""
        doc = self.documents.pop(doc_id, None)
        if doc:
            self._save()
        return doc

    def get(self, doc_id: str, include_deleted: bool = False) -> Optional[Dict]:
        """获取文档信息（默认跳过已删除文档）"""
        doc = self.documents.get(doc_id)
        if doc and not include_deleted and doc.get('is_deleted'):
            return None
        return doc

    def get_by_hash(self, file_hash: str) -> Optional[Dict]:
        """根据文件哈希查找文档（去重，跳过已删除文档）"""
        for doc in self.documents.values():
            if doc.get('is_deleted'):
                continue
            if doc.get('file_hash') == file_hash:
                return doc
        return None

    def list_all(self, status_filter: str = None,
                 tag_filter: str = None,
                 include_deleted: bool = False) -> List[Dict]:
        """列出所有文档（默认排除已软删除的文档）"""
        docs = list(self.documents.values())
        if not include_deleted:
            docs = [d for d in docs if not d.get('is_deleted')]
        if status_filter:
            docs = [d for d in docs if d.get('parse_status') == status_filter]
        if tag_filter:
            docs = [d for d in docs if tag_filter in d.get('tags', [])]
        return sorted(docs, key=lambda x: x.get('created_at', ''), reverse=True)

    def get_stats(self) -> Dict:
        """获取文档统计信息（排除已删除文档）"""
        docs = [d for d in self.documents.values() if not d.get('is_deleted')]
        total_size = sum(d.get('file_size', 0) for d in docs)
        total_chunks = sum(d.get('chunk_count', 0) for d in docs)
        parser_dist = defaultdict(int)
        type_dist = defaultdict(int)
        tag_dist = defaultdict(int)

        for d in docs:
            parser_dist[d.get('parser_used', 'unknown')] += 1
            type_dist[d.get('file_type', 'unknown')] += 1
            for tag in d.get('tags', []):
                tag_dist[tag] += 1

        return {
            'total_documents': len(docs),
            'total_chunks': total_chunks,
            'total_size_bytes': total_size,
            'total_size_human': self._human_size(total_size),
            'parser_distribution': dict(parser_dist),
            'file_type_distribution': dict(type_dist),
            'tag_distribution': dict(tag_dist),
            'status_distribution': dict(defaultdict(
                int, {d.get('parse_status', 'unknown'): 1 for d in docs}
            ))
        }

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        """字节数转人类可读格式"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"


# ============================================================
# 文档管理器
# ============================================================
class DocumentManager:
    """
    企业文档管理器

    核心流程:
    1. 上传文档 → 去重检查 → 自动分类 → 双管线解析
    2. 解析结果持久化 → 注册表更新 → 索引构建
    3. 支持文档的增删改查和生命周期管理
    """

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None,
                 use_pipeline: bool = True):
        # 默认值在调用时取 config，函数定义时就绑定会把 CHUNK_SIZE 的改动吞掉
        chunk_size = CHUNK_SIZE if chunk_size is None else chunk_size
        chunk_overlap = CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
        self.dispatcher = MultiPipelineDispatcher(
            default_chunk_size=chunk_size,
            default_chunk_overlap=chunk_overlap
        )
        self.registry = DocumentRegistry()
        self.use_pipeline = use_pipeline

        # 新管线（数据清洗管线）
        if use_pipeline:
            # 构建已知哈希表（从注册表加载）
            known_hashes = {
                d.get('file_hash', ''): d.get('filename', '')
                for d in self.registry.documents.values()
                if d.get('file_hash')
            }
            self.data_pipeline = DataPipeline(known_hashes=known_hashes)
        else:
            self.data_pipeline = None

        # 确保存储目录存在
        os.makedirs(DOCUMENT_STORE_DIR, exist_ok=True)
        os.makedirs(PARSED_CACHE_DIR, exist_ok=True)

        # 上传目录
        self.upload_dir = os.path.join(DOCUMENT_STORE_DIR, 'uploads')
        os.makedirs(self.upload_dir, exist_ok=True)

        logger.info(f"[DocumentManager] 初始化完成 (管线模式: {'新清洗管线' if use_pipeline else '传统模式'})")
        logger.info(f"  文档存储: {DOCUMENT_STORE_DIR}")
        logger.info(f"  已注册文档: {len(self.registry.documents)} 个")

    # ============================================================
    # 文档上传与解析
    # ============================================================
    def add_document(self, file_path: str, parser: str = None,
                     tags: List[str] = None, copy_file: bool = True,
                     access_level: str = None, allowed_groups: str = None) -> ParsedDocument:
        """
        添加并解析一个文档

        :param file_path: 源文件路径
        :param parser: 指定解析器 (None=自动, "mineru", "docling")
        :param tags: 文档标签
        :param copy_file: 是否复制文件到存储目录
        :param access_level: 访问级别 "public"|"internal"|"restricted"|"confidential"
        :param allowed_groups: 允许组（逗号分隔字符串，如 "人事部,财务部"）
        :return: ParsedDocument
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        filename = os.path.basename(file_path)

        # 0. 自动定密（2026-10-24）：未显式指定密级时按文件名规则分类，
        # 不再一律默认 internal（曾导致工资表对全员可见）
        if access_level is None or allowed_groups is None:
            from access_control import classify_document
            auto_level, auto_groups = classify_document(filename)
            access_level = access_level or auto_level
            allowed_groups = allowed_groups if allowed_groups is not None else auto_groups
            if access_level != 'internal':
                print(f"  [自动定密] {filename} -> {access_level}（组:{allowed_groups}）")

        # 1. 去重检查 —— 必须用 SHA256（registry 的 file_hash 由 prescreen 写入，
        # 两个体系不一致时 MD5 永远查不中，幂等检查形同虚设：重复上传会漏到
        # prescreen 才被拦，失败结果又覆盖同 doc_id 的正常记录。2026-10-24 实测）
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for blk in iter(lambda: f.read(1 << 20), b''):
                sha256.update(blk)
        existing = self.registry.get_by_hash(sha256.hexdigest())
        if existing:
            logger.info(f"文档已存在（SHA256 匹配），跳过: {filename}")
            logger.info(f"  已有文档: {existing['filename']} (doc_id: {existing['doc_id']})")
            # 返回已有文档信息
            return self._load_parsed_doc(existing['doc_id'])

        # 2. 复制文件到存储目录
        if copy_file:
            dest_path = os.path.join(self.upload_dir, filename)
            # 避免文件名冲突
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(filename)
                dest_path = os.path.join(
                    self.upload_dir,
                    f"{base}_{int(time.time())}{ext}"
                )
            shutil.copy2(file_path, dest_path)
            file_path = dest_path

        # 3. 解析文档（使用新管线或传统模式）
        print(f"\n[DocumentManager] 开始解析文档: {filename}")

        if self.use_pipeline and self.data_pipeline:
            # 新清洗管线
            pipeline_result = self.data_pipeline.process(
                file_path, parser_override=parser
            )
            parsed_doc = self._pipeline_result_to_parsed_doc(pipeline_result)
        else:
            # 传统模式
            parsed_doc = self.dispatcher.parse(file_path, parser=parser)

        # 4. 保存解析结果
        # 5. 注册文档 —— 只在解析成功时注册。doc_id = sha256 前 16 位，
        # 失败结果一旦注册会覆盖同 doc_id 的正常记录（chunk_count 清零、
        # parser 变 unknown），把好文档变成空壳（2026-10-24 实测）。
        if parsed_doc.status == "success" and parsed_doc.chunks:
            self._save_parsed_doc(parsed_doc)
            metrics.incr("parse_success")
            metrics.observe("parse_latency_ms", parsed_doc.parse_time * 1000)
            self.registry.register(parsed_doc, tags=tags,
                                   access_level=access_level, allowed_groups=allowed_groups)
        else:
            metrics.incr("parse_fail")

        print(f"[DocumentManager] 文档解析完成:")
        print(f"  解析器: {parsed_doc.parser_used}")
        print(f"  块数: {len(parsed_doc.chunks)}")
        print(f"  耗时: {parsed_doc.parse_time:.1f}s")
        print(f"  状态: {parsed_doc.status}")

        return parsed_doc

    def add_documents_batch(self, file_paths: List[str], parser: str = None,
                            tags: List[str] = None,
                            progress_callback=None) -> List[ParsedDocument]:
        """
        批量添加文档

        :param file_paths: 文件路径列表
        :param parser: 指定解析器
        :param tags: 文档标签
        :param progress_callback: 进度回调
        :return: ParsedDocument 列表
        """
        results = []
        total = len(file_paths)

        for i, fp in enumerate(file_paths):
            if progress_callback:
                progress_callback(i + 1, total, os.path.basename(fp))

            try:
                doc = self.add_document(fp, parser=parser, tags=tags)
                results.append(doc)
            except Exception as e:
                logger.error(f"添加文档失败 [{fp}]: {e}")
                results.append(ParsedDocument(
                    doc_id="error",
                    filename=os.path.basename(fp),
                    file_path=fp,
                    file_type=os.path.splitext(fp)[1].lstrip('.'),
                    file_size=0,
                    file_hash="",
                    parser_used=parser or "auto",
                    parse_time=0,
                    status="failed",
                    error_msg=str(e)
                ))

        return results

    def add_from_text(self, text: str, title: str = "untitled",
                      tags: List[str] = None) -> ParsedDocument:
        """
        直接从文本添加文档（无需文件）

        :param text: 文本内容
        :param title: 文档标题
        :param tags: 标签
        :return: ParsedDocument
        """
        doc_id = BaseParser.generate_doc_id(title + str(time.time()))

        # 简单分块
        chunk_size = self.dispatcher.default_chunk_size
        overlap = self.dispatcher.default_chunk_overlap
        chunks = []
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(ParsedChunk(
                chunk_id=BaseParser.generate_chunk_id(doc_id, idx),
                doc_id=doc_id,
                content=chunk_text,
                chunk_index=idx,
                title=title,
                chunk_type="text"
            ))
            idx += 1
            start += chunk_size - overlap

        if not chunks:
            chunks.append(ParsedChunk(
                chunk_id=BaseParser.generate_chunk_id(doc_id, 0),
                doc_id=doc_id,
                content=text,
                chunk_index=0,
                title=title,
                chunk_type="text"
            ))

        parsed_doc = ParsedDocument(
            doc_id=doc_id,
            filename=f"{title}.txt",
            file_path="",
            file_type="text",
            file_size=len(text.encode('utf-8')),
            file_hash=BaseParser.compute_file_hash(title.encode()),
            parser_used="direct_text",
            parse_time=0,
            chunks=chunks,
            status="success"
        )

        self._save_parsed_doc(parsed_doc)
        self.registry.register(parsed_doc, tags=tags)

        return parsed_doc

    # ============================================================
    # 文档查询与管理
    # ============================================================
    def list_documents(self, status_filter: str = None,
                       tag_filter: str = None) -> List[Dict]:
        """列出所有已注册文档"""
        return self.registry.list_all(status_filter, tag_filter)

    def get_document(self, doc_id: str) -> Optional[ParsedDocument]:
        """获取解析后的文档"""
        return self._load_parsed_doc(doc_id)

    def get_document_info(self, doc_id: str) -> Optional[Dict]:
        """获取文档注册信息"""
        return self.registry.get(doc_id)

    def remove_document(self, doc_id: str) -> bool:
        """
        软删除文档（标记 is_deleted，保留解析缓存和源文件，审计留痕）
        如需物理删除，请使用 purge_document()
        """
        doc_info = self.registry.get(doc_id, include_deleted=True)
        if not doc_info:
            logger.warning(f"文档不存在: {doc_id}")
            return False

        if doc_info.get('is_deleted'):
            logger.warning(f"文档已删除: {doc_info.get('filename', doc_id)}")
            return False

        # 软删除：标记 is_deleted，不删除缓存和源文件
        self.registry.soft_delete(doc_id)
        logger.info(f"文档已软删除: {doc_info.get('filename', doc_id)}")
        return True

    def restore_document(self, doc_id: str) -> bool:
        """
        恢复软删除的文档（撤销 is_deleted 标记）
        注意：恢复后需调用方同步 Milvus 中的 is_deleted 字段
        """
        doc_info = self.registry.get(doc_id, include_deleted=True)
        if not doc_info:
            logger.warning(f"文档不存在: {doc_id}")
            return False

        if not doc_info.get('is_deleted'):
            logger.info(f"文档未处于删除状态: {doc_info.get('filename', doc_id)}")
            return True

        self.registry.restore(doc_id)
        logger.info(f"文档已恢复: {doc_info.get('filename', doc_id)}")
        return True

    def update_document(self, doc_id: str, new_file_path: str,
                        parser: str = None) -> ParsedDocument:
        """
        更新文档（重新解析并替换旧 chunk）

        流程：
        1. 保留原 doc_id、权限、标签
        2. 重新解析新文件
        3. 更新注册表（version +1）
        4. 返回新的 ParsedDocument（调用方需同步 Milvus）
        """
        doc_info = self.registry.get(doc_id, include_deleted=True)
        if not doc_info:
            raise ValueError(f"文档不存在: {doc_id}")

        if not os.path.exists(new_file_path):
            raise FileNotFoundError(f"文件不存在: {new_file_path}")

        filename = os.path.basename(new_file_path)
        file_hash = BaseParser.compute_file_hash(new_file_path)

        # 复制新文件到存储目录
        dest_path = os.path.join(self.upload_dir, filename)
        if os.path.exists(dest_path):
            base, ext = os.path.splitext(filename)
            dest_path = os.path.join(
                self.upload_dir,
                f"{base}_{int(time.time())}{ext}"
            )
        shutil.copy2(new_file_path, dest_path)

        # 重新解析
        print(f"\n[DocumentManager] 重新解析文档: {filename}")
        if self.use_pipeline and self.data_pipeline:
            pipeline_result = self.data_pipeline.process(
                dest_path, parser_override=parser
            )
            parsed_doc = self._pipeline_result_to_parsed_doc(pipeline_result)
        else:
            parsed_doc = self.dispatcher.parse(dest_path, parser=parser)

        if parsed_doc.status == "success" and parsed_doc.chunks:
            # 强制使用原 doc_id（保持 chunk 关联不变）
            parsed_doc.doc_id = doc_id
            self._save_parsed_doc(parsed_doc)

        # 更新注册表（保留权限、标签，version +1）
        old_tags = doc_info.get('tags', [])
        old_access = doc_info.get('access_level', 'internal')
        old_groups = doc_info.get('allowed_groups', 'all')
        old_version = doc_info.get('version', 1)

        self.registry.documents[doc_id].update({
            'filename': parsed_doc.filename,
            'file_path': parsed_doc.file_path,
            'file_type': parsed_doc.file_type,
            'file_size': parsed_doc.file_size,
            'file_hash': parsed_doc.file_hash,
            'parser_used': parsed_doc.parser_used,
            'parse_status': parsed_doc.status,
            'parse_time': parsed_doc.parse_time,
            'chunk_count': len(parsed_doc.chunks),
            'updated_at': datetime.now().isoformat(),
            'version': old_version + 1,
            'error_msg': parsed_doc.error_msg,
            # 保留原权限和标签
            'tags': old_tags,
            'access_level': old_access,
            'allowed_groups': old_groups,
        })
        self.registry._save()

        print(f"[DocumentManager] 文档已更新: {filename}")
        print(f"  版本: {old_version} -> {old_version + 1}")
        print(f"  块数: {doc_info.get('chunk_count', 0)} -> {len(parsed_doc.chunks)}")

        return parsed_doc

    def purge_document(self, doc_id: str) -> bool:
        """
        物理删除文档（不可恢复，慎用）
        包括注册信息、解析缓存、源文件
        """
        doc_info = self.registry.get(doc_id, include_deleted=True)
        if not doc_info:
            logger.warning(f"文档不存在: {doc_id}")
            return False

        # 删除解析缓存
        cache_path = os.path.join(PARSED_CACHE_DIR, f"{doc_id}.json")
        if os.path.exists(cache_path):
            os.remove(cache_path)

        # 删除源文件（如果在 upload 目录中）
        # 必须用归一化绝对路径判断：file_path 存的是绝对路径而 upload_dir 是相对路径，
        # 字符串 startswith 永远不命中 → purge 只删 registry 不清实体，
        # uploads 里同 hash 副本越积越多（2026-10-24 曾一次清出 18 个）
        file_path = doc_info.get('file_path', '')
        if file_path and os.path.exists(file_path):
            norm_fp = os.path.normcase(os.path.abspath(file_path))
            norm_upload = os.path.normcase(os.path.abspath(self.upload_dir)) + os.sep
            if norm_fp.startswith(norm_upload):
                os.remove(file_path)

        # 物理移除注册
        self.registry.purge(doc_id)
        # 同步清掉进程内的已知哈希，否则 purge 后同文件立刻重传仍会被
        # prescreen 拦截（known_hashes 只在构造时从 registry 构建一次）
        if self.data_pipeline is not None:
            fh = doc_info.get('file_hash', '')
            if fh:
                self.data_pipeline.prescreen.known_hashes.pop(fh, None)
        logger.info(f"文档已物理删除: {doc_info.get('filename', doc_id)}")
        return True

    def update_tags(self, doc_id: str, tags: List[str]) -> bool:
        """更新文档标签"""
        doc_info = self.registry.get(doc_id)
        if not doc_info:
            return False
        self.registry.update(doc_id, {'tags': tags})
        return True

    def update_access(self, doc_id: str, access_level: str,
                      allowed_groups: str = None) -> bool:
        """
        更新文档权限（骨架层一期固化）
        
        注意：chunk 权限继承 - 更新注册表后，Milvus 中对应 chunk 需同步更新
        :param doc_id: 文档 ID
        :param access_level: 新访问级别
        :param allowed_groups: 新允许组（逗号分隔字符串）
        :return: True 成功
        """
        doc_info = self.registry.get(doc_id)
        if not doc_info:
            return False

        updates = {
            'access_level': access_level or 'internal',
            'version': doc_info.get('version', 1) + 1,
        }
        if allowed_groups is not None:
            updates['allowed_groups'] = allowed_groups

        self.registry.update(doc_id, updates)
        logger.info(f"[DocumentManager] 文档权限已更新: {doc_id} -> {access_level}")
        return True

    def get_all_chunks(self, doc_ids: List[str] = None) -> List[ParsedChunk]:
        """
        获取所有文档的 chunks（用于构建索引，跳过已删除文档）
        :param doc_ids: 指定文档ID列表，None 则获取全部
        :return: ParsedChunk 列表
        """
        all_chunks = []

        if doc_ids is None:
            # 只获取未删除的文档
            doc_ids = [
                doc_id for doc_id, doc in self.registry.documents.items()
                if not doc.get('is_deleted')
            ]

        for doc_id in doc_ids:
            doc = self._load_parsed_doc(doc_id)
            if doc and doc.status == "success":
                # 修正 chunk 的 doc_id 为 registry 的 doc_id
                # （parsed cache 中的 chunk doc_id 可能与 registry 不一致）
                for chunk in doc.chunks:
                    chunk.doc_id = doc_id
                    chunk.chunk_id = f"{doc_id}_chunk_{chunk.chunk_index:05d}"
                all_chunks.extend(doc.chunks)

        return all_chunks

    def get_stats(self) -> Dict:
        """获取文档库统计信息"""
        stats = self.registry.get_stats()

        # 解析器状态
        parser_status = self.dispatcher.get_parser_status()
        stats['parser_availability'] = {
            name: info['available']
            for name, info in parser_status.items()
        }

        return stats

    # ============================================================
    # 持久化
    # ============================================================
    def _save_parsed_doc(self, parsed_doc: ParsedDocument):
        """保存解析结果到磁盘"""
        cache_path = os.path.join(PARSED_CACHE_DIR, f"{parsed_doc.doc_id}.json")

        data = {
            'document': parsed_doc.to_dict(),
            'chunks': [
                {
                    'chunk_id': c.chunk_id,
                    'doc_id': c.doc_id,
                    'content': c.content,
                    'chunk_index': c.chunk_index,
                    'title': c.title,
                    'page_number': c.page_number,
                    'bbox': c.bbox,
                    'chunk_type': c.chunk_type,
                    'language': c.language,
                    'metadata': c.metadata
                }
                for c in parsed_doc.chunks
            ]
        }

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_parsed_doc(self, doc_id: str) -> Optional[ParsedDocument]:
        """从磁盘加载解析结果"""
        cache_path = os.path.join(PARSED_CACHE_DIR, f"{doc_id}.json")

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            doc_data = data['document']
            chunks = [
                ParsedChunk(
                    chunk_id=c['chunk_id'],
                    doc_id=c['doc_id'],
                    content=c['content'],
                    chunk_index=c['chunk_index'],
                    title=c.get('title', ''),
                    page_number=c.get('page_number', -1),
                    bbox=c.get('bbox'),
                    chunk_type=c.get('chunk_type', 'text'),
                    language=c.get('language', 'zh'),
                    metadata=c.get('metadata', {})
                )
                for c in data['chunks']
            ]

            return ParsedDocument(
                doc_id=doc_data['doc_id'],
                filename=doc_data['filename'],
                file_path=doc_data.get('file_path', ''),
                file_type=doc_data['file_type'],
                file_size=doc_data['file_size'],
                file_hash=doc_data['file_hash'],
                parser_used=doc_data['parser_used'],
                parse_time=doc_data['parse_time'],
                chunks=chunks,
                metadata=doc_data.get('metadata', {}),
                status=doc_data.get('status', 'success'),
                error_msg=doc_data.get('error_msg', '')
            )

        except Exception as e:
            logger.error(f"加载解析结果失败 [{doc_id}]: {e}")
            return None

    def _pipeline_result_to_parsed_doc(self, pipeline_result: 'PipelineResult') -> ParsedDocument:
        """
        将新管线的 PipelineResult 转换为旧的 ParsedDocument 格式（兼容下游代码）
        """
        from pipeline import PipelineResult

        chunks = []
        for i, chunk in enumerate(pipeline_result.chunks):
            chunks.append(ParsedChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                title=chunk.parent_title,
                page_number=chunk.page_number,
                chunk_type=chunk.chunk_type,
                metadata={
                    'parser_source': chunk.parser_source,
                    'quality_score': chunk.quality_score,
                    'truncated': chunk.truncated,
                    'tiny_chunk': chunk.tiny_chunk,
                    **chunk.metadata,
                }
            ))

        # 状态映射
        status_map = {
            'success': 'success',
            'failed': 'failed',
            'blocked': 'partial',
            'duplicate': 'success',
        }
        status = status_map.get(pipeline_result.status, 'failed')

        return ParsedDocument(
            doc_id=pipeline_result.prescreen.sha256[:16] if pipeline_result.prescreen else 'unknown',
            filename=pipeline_result.filename,
            file_path=pipeline_result.file_path,
            file_type=os.path.splitext(pipeline_result.filename)[1].lstrip('.'),
            file_size=os.path.getsize(pipeline_result.file_path) if os.path.exists(pipeline_result.file_path) else 0,
            file_hash=pipeline_result.prescreen.sha256 if pipeline_result.prescreen else '',
            parser_used=pipeline_result.routing.resolve_parser() if pipeline_result.routing else 'unknown',
            parse_time=pipeline_result.elapsed_time,
            chunks=chunks,
            metadata={
                'pipeline_version': 'v2',
                'quality_score': pipeline_result.quality.overall_score if pipeline_result.quality else 0,
                'quality_status': pipeline_result.quality.status if pipeline_result.quality else '',
                'version_tag': pipeline_result.prescreen.version_tag if pipeline_result.prescreen else '',
            },
            status=status,
            error_msg=pipeline_result.error_msg,
        )

    # ============================================================
    # 文档搜索（基于元数据）
    # ============================================================
    def search_documents(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        基于文档元数据的搜索（文件名、标签、类型）
        注意：内容搜索请使用 retriever 的向量检索
        """
        query_lower = query.lower()
        results = []

        for doc_info in self.registry.list_all():
            score = 0
            filename = doc_info.get('filename', '').lower()
            tags = doc_info.get('tags', [])

            # 文件名匹配
            if query_lower in filename:
                score += 5

            # 标签匹配
            for tag in tags:
                if query_lower in tag.lower():
                    score += 3

            # 文件类型匹配
            if query_lower == doc_info.get('file_type', '').lower():
                score += 1

            if score > 0:
                results.append({**doc_info, 'search_score': score})

        results.sort(key=lambda x: x['search_score'], reverse=True)
        return results[:top_k]




