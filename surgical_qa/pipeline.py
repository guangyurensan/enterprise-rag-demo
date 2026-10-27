# -*- coding: utf-8 -*-
"""
企业级知识库 - 数据清洗管线主编排器

完整流程:
文件前置粗筛 → 路由调度 → 解析 → 归一适配 → 差异化清洗 →
文档质检打分 → 智能分块 → Chunk去重 → 写入向量库

旁路: 指标埋点、失败任务队列、人工重试台
"""

import os
import re
import json
import time
import logging
import shutil
import hashlib
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

from prescreen import FilePrescreen, PrescreenResult
from router import RouterDispatcher, RoutingResult
from block_adapter import BlockAdapter, UnifiedBlock
from cleaner import DifferentialCleaner
from smart_chunker import SmartChunker, Chunk
from pipeline_config import (
    PIPELINE_METRICS, MANUAL_RETRY_DIR,
    QUALITY_SCORING, DEDUP,
)
from resilience import ingestion_dlq

logger = logging.getLogger(__name__)


# ============================================================
# 管线执行结果
# ============================================================
@dataclass
class PipelineResult:
    """管线执行结果"""
    file_path: str
    filename: str
    status: str                     # success / failed / blocked / duplicate
    stage: str = ""                 # 最后执行到的阶段
    error_msg: str = ""

    # 各阶段结果
    prescreen: Optional[PrescreenResult] = None
    routing: Optional[RoutingResult] = None
    quality: Optional["QualityReport"] = None

    # 最终产出
    chunks: List[Chunk] = field(default_factory=list)
    blocks_before_chunking: int = 0

    # 统计
    elapsed_time: float = 0.0
    metrics: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "file_path": self.file_path,
            "filename": self.filename,
            "status": self.status,
            "stage": self.stage,
            "error_msg": self.error_msg,
            "chunk_count": len(self.chunks),
            "blocks_before_chunking": self.blocks_before_chunking,
            "elapsed_time": self.elapsed_time,
            "prescreen": {
                "passed": self.prescreen.passed if self.prescreen else None,
                "sha256": self.prescreen.sha256[:16] if self.prescreen else "",
                "is_scanned_pdf": self.prescreen.is_scanned_pdf if self.prescreen else False,
            },
            "routing": {
                "parser": self.routing.resolve_parser() if self.routing else "",
                "rule": self.routing.rule_name if self.routing else "",
            },
            "quality": self.quality.to_dict() if self.quality else None,
        }


# ============================================================
# 文档质检打分（原 quality_scorer.py）
# ============================================================
@dataclass
class QualityReport:
    """质检报告"""
    doc_id: str
    filename: str
    overall_score: float
    status: str                   # pass / warn / block
    total_blocks: int
    valid_blocks: int
    invalid_blocks: int
    avg_block_quality: float
    deductions: List[str] = field(default_factory=list)
    bonuses: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id, "filename": self.filename,
            "overall_score": self.overall_score, "status": self.status,
            "total_blocks": self.total_blocks, "valid_blocks": self.valid_blocks,
            "invalid_blocks": self.invalid_blocks, "avg_block_quality": self.avg_block_quality,
            "deductions": self.deductions, "bonuses": self.bonuses,
            "recommendations": self.recommendations,
        }


class QualityScorer:
    """文档质检打分器"""

    def __init__(self):
        self.cfg = QUALITY_SCORING

    def score_document(self, doc_id: str, filename: str,
                       blocks: List[UnifiedBlock],
                       original_text: str = "") -> QualityReport:
        score = 100.0
        deductions, bonuses, recommendations = [], [], []

        # 1. 乱码检测
        garbled_count = sum(
            len(re.findall(r'[\ufffd\u200b-\u200f\u2028\u2029]', b.content))
            for b in blocks
        )
        if garbled_count > 0:
            penalty = garbled_count * self.cfg["penalty_per_garbled_char"]
            score -= penalty
            deductions.append(f"检测到 {garbled_count} 个乱码字符，扣 {penalty} 分")
            if garbled_count > 20:
                recommendations.append("乱码较多，建议检查 PDF 质量或使用 OCR 增强模式")

        # 2. 噪声块
        noise_count = sum(1 for b in blocks if b.is_noise)
        if noise_count > 0:
            penalty = noise_count * self.cfg["penalty_per_noise_block"]
            score -= penalty
            deductions.append(f"检测到 {noise_count} 个噪声块，扣 {penalty} 分")

        # 3. 空白占比
        if original_text:
            blank_count = sum(1 for c in original_text if c in ' \t\n\r')
            blank_ratio = blank_count / max(len(original_text), 1)
            if blank_ratio > 0.5:
                penalty = self.cfg["penalty_empty_ratio"]
                score -= penalty
                deductions.append(f"空白占比过高 ({blank_ratio:.0%})，扣 {penalty} 分")
                recommendations.append("文档空白区域较多，可能是扫描件或排版问题")

        # 4. 结构完整性
        has_titles = any(b.parent_title for b in blocks)
        sections = set(b.parent_title for b in blocks if b.parent_title)
        has_sections = len(sections) > 1

        if not has_titles:
            penalty = self.cfg["penalty_no_structure"]
            score -= penalty
            deductions.append(f"无标题结构，扣 {penalty} 分")
            recommendations.append("文档缺少章节标题，建议检查解析器是否正确识别结构")
        else:
            bonus = self.cfg["bonus_has_title"]
            score += bonus
            bonuses.append(f"有标题层级，加 {bonus} 分")

        # 5. 表格加分
        if any(b.block_type == "table" for b in blocks):
            bonus = self.cfg["bonus_has_table"]
            score += bonus
            bonuses.append(f"包含表格，加 {bonus} 分")

        # 6. 章节数加分
        if has_sections:
            bonus = min(len(sections) * self.cfg["bonus_per_section"], 20)
            score += bonus
            bonuses.append(f"共 {len(sections)} 个章节，加 {bonus} 分")

        # 7. 块质量评估
        block_qualities = [self._score_single_block(b) for b in blocks]
        avg_quality = sum(block_qualities) / max(len(block_qualities), 1)
        low_quality_count = sum(1 for q in block_qualities if q < 0.5)
        if low_quality_count > len(blocks) * 0.3:
            penalty = 15
            score -= penalty
            deductions.append(f"低质量块占比过高 ({low_quality_count}/{len(blocks)})，扣 {penalty} 分")

        # 8. 块数量过少
        if len(blocks) < 3:
            penalty = 10
            score -= penalty
            deductions.append(f"有效块数过少 ({len(blocks)})，扣 {penalty} 分")
            recommendations.append("文档解析出的内容块太少，可能解析不完整")

        score = max(0, min(100, score))
        if score >= self.cfg["pass_threshold"]:
            status = "pass"
        elif score >= self.cfg["warn_threshold"]:
            status = "warn"
            recommendations.append("文档质量偏低，建议人工复核后入库")
        else:
            status = "block"
            recommendations.append("文档质量过低，已阻断入库，请送人工重试台处理")

        logger.info(f"[QualityScorer] {filename}: 得分={score:.0f}, 状态={status}")
        return QualityReport(
            doc_id=doc_id, filename=filename, overall_score=score, status=status,
            total_blocks=len(blocks),
            valid_blocks=sum(1 for q in block_qualities if q >= 0.5),
            invalid_blocks=sum(1 for q in block_qualities if q < 0.5),
            avg_block_quality=avg_quality,
            deductions=deductions, bonuses=bonuses, recommendations=recommendations,
        )

    def _score_single_block(self, block: UnifiedBlock) -> float:
        score = 1.0
        content = block.content
        if len(content) < 10:
            score -= 0.5
        elif len(content) < 30:
            score -= 0.2
        if re.match(r'^[\d\s\.\,\;\:]+$', content):
            score -= 0.5
        garbled_ratio = len(re.findall(r'[\ufffd\u200b-\u200f]', content)) / max(len(content), 1)
        if garbled_ratio > 0.1:
            score -= 0.3
        cjk_ratio = len(re.findall(r'[\u4e00-\u9fff]', content)) / max(len(content), 1)
        if cjk_ratio < 0.1 and block.language == "zh":
            score -= 0.2
        if block.is_noise:
            score -= 0.3
        return max(0, min(1, score))


# ============================================================
# Chunk 级近似去重（原 dedup.py）
# ============================================================
class SimHash:
    """SimHash: 64 位指纹，海明距离判相似度"""

    def __init__(self, hash_bits: int = 64):
        self.hash_bits = hash_bits

    def compute(self, text: str) -> int:
        text = re.sub(r'\s+', '', text)
        tokens = [(text[i:i+2], 1) for i in range(len(text)-1)] if len(text) > 1 else [(text, 1)]
        v = [0] * self.hash_bits
        for token, weight in tokens:
            h = int(hashlib.md5(token.encode('utf-8')).hexdigest()[:16], 16) % (2 ** self.hash_bits)
            for i in range(self.hash_bits):
                v[i] += weight if (h >> i) & 1 else -weight
        return sum((1 << i) for i in range(self.hash_bits) if v[i] > 0)

    def distance(self, h1: int, h2: int) -> int:
        return bin(h1 ^ h2).count('1')


class MinHash:
    """MinHash: 多 hash 函数最小值估计 Jaccard 相似度"""

    def __init__(self, num_perm: int = 128):
        self.num_perm = num_perm
        import random
        random.seed(42)
        self.a = [random.randint(1, 2**31 - 1) for _ in range(num_perm)]
        self.b = [random.randint(0, 2**31 - 1) for _ in range(num_perm)]
        self.prime = 2**31 - 1

    def compute(self, text: str) -> List[int]:
        text = re.sub(r'\s+', ' ', text)
        tokens = set(text[i:i+3] for i in range(len(text)-2))
        if not tokens:
            return [2**32] * self.num_perm
        sig = [2**32] * self.num_perm
        for token in tokens:
            th = int(hashlib.md5(token.encode()).hexdigest()[:8], 16)
            for i in range(self.num_perm):
                sig[i] = min(sig[i], (self.a[i] * th + self.b[i]) % self.prime)
        return sig

    def similarity(self, sig1: List[int], sig2: List[int]) -> float:
        return sum(1 for a, b in zip(sig1, sig2) if a == b) / self.num_perm


class ChunkDeduplicator:
    """Chunk 级近似去重器"""

    def __init__(self, method: str = None, threshold: float = None):
        self.cfg = DEDUP
        self.method = method or self.cfg["method"]
        self.threshold = threshold or self.cfg["threshold"]
        self.simhash = SimHash() if self.method == "simhash" else None
        self.minhash = MinHash() if self.method != "simhash" else None

    @staticmethod
    def _is_table_chunk(content: str) -> bool:
        """含 ≥2 行 markdown 表格行的块视为表格块。

        表格数据行结构高度相似、只有数字不同（如城市住宿标准 600/450/350），
        SimHash/MinHash 对这类差异不敏感，会把它们误判为重复块整行删除，
        导致关键数值在入库时静默丢失（2026-10-24 差旅制度实测）。
        表格块一律豁免去重。
        """
        return sum(1 for line in content.split("\n")
                   if line.count("|") >= 2 and line.strip().startswith("|")) >= 2

    def deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return chunks
        logger.info(f"[Dedup] 开始去重: {len(chunks)} chunks, 方法={self.method}")
        if self.method == "simhash":
            return self._dedup_simhash(chunks)
        return self._dedup_minhash(chunks)

    def _dedup_simhash(self, chunks: List[Chunk]) -> List[Chunk]:
        threshold = self.cfg["threshold"]
        hashes = [self.simhash.compute(c.content) for c in chunks]
        table = [self._is_table_chunk(c.content) for c in chunks]
        dup_indices = set()
        for i in range(len(hashes)):
            if i in dup_indices or table[i]:
                continue
            for j in range(i + 1, len(hashes)):
                if j not in dup_indices and not table[j] \
                        and self.simhash.distance(hashes[i], hashes[j]) <= threshold:
                    dup_indices.add(j)
        logger.info(f"[Dedup] SimHash: {len(dup_indices)} 个重复")
        if self.cfg["action"] == "remove":
            return [c for i, c in enumerate(chunks) if i not in dup_indices]
        for i in dup_indices:
            chunks[i].metadata["is_duplicate"] = True
        return chunks

    def _dedup_minhash(self, chunks: List[Chunk]) -> List[Chunk]:
        threshold = self.cfg["minhash_threshold"]
        sigs = [self.minhash.compute(c.content) for c in chunks]
        table = [self._is_table_chunk(c.content) for c in chunks]
        dup_indices = set()
        for i in range(len(sigs)):
            if i in dup_indices or table[i]:
                continue
            for j in range(i + 1, len(sigs)):
                if j not in dup_indices and not table[j] \
                        and self.minhash.similarity(sigs[i], sigs[j]) >= threshold:
                    dup_indices.add(j)
        logger.info(f"[Dedup] MinHash: {len(dup_indices)} 个重复")
        if self.cfg["action"] == "remove":
            return [c for i, c in enumerate(chunks) if i not in dup_indices]
        for i in dup_indices:
            chunks[i].metadata["is_duplicate"] = True
        return chunks


# ============================================================
# 管线编排器
# ============================================================
class DataPipeline:
    """
    数据清洗管线编排器

    串联所有处理阶段，管理旁路（指标、失败队列、人工重试）
    """

    def __init__(self, known_hashes: Dict[str, str] = None):
        self.prescreen = FilePrescreen(known_hashes)
        self.router = RouterDispatcher()
        self.adapter = BlockAdapter()
        self.cleaner = DifferentialCleaner()
        self.scorer = QualityScorer()
        self.chunker = SmartChunker()
        self.dedup = ChunkDeduplicator()

        # 旁路
        self.metrics_dir = PIPELINE_METRICS["log_dir"]
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(MANUAL_RETRY_DIR, exist_ok=True)

    def process(self, file_path: str, parser_override: str = None,
                sample_text: str = "",
                parse_func=None) -> PipelineResult:
        """
        执行完整管线

        :param file_path: 文件路径
        :param parser_override: 强制指定解析器（跳过路由）
        :param sample_text: 文件采样文本
        :param parse_func: 解析函数 (file_path, parser) -> (doc_id, blocks, original_text)
                           如果不传，使用默认的 document_parser 模块
        :return: PipelineResult
        """
        start_time = time.time()
        filename = os.path.basename(file_path)
        result = PipelineResult(file_path=file_path, filename=filename, status="running")

        try:
            # ========================================
            # 阶段1: 文件前置粗筛
            # ========================================
            logger.info(f"[Pipeline] 阶段1: 前置粗筛 - {filename}")
            prescreen_result = self.prescreen.prescreen(file_path)
            result.prescreen = prescreen_result
            result.stage = "prescreen"

            if not prescreen_result.passed:
                result.status = "failed"
                result.error_msg = prescreen_result.reject_reason
                self._log_failure(result)
                return result

            if prescreen_result.is_duplicate:
                result.status = "duplicate"
                result.error_msg = prescreen_result.reject_reason
                return result

            # ========================================
            # 阶段2: 路由调度
            # ========================================
            logger.info(f"[Pipeline] 阶段2: 路由调度 - {filename}")
            if parser_override:
                routing_result = RoutingResult(
                    file_tag=parser_override,
                    rule_name="手动指定",
                    confidence=1.0,
                    reasons=[f"手动指定解析器: {parser_override}"],
                )
            else:
                routing_result = self.router.route(prescreen_result, sample_text)
            result.routing = routing_result
            result.stage = "routing"

            # ========================================
            # 阶段3: 解析 + 归一适配
            # ========================================
            parser_name = routing_result.resolve_parser()
            logger.info(f"[Pipeline] 阶段3: 解析 - {filename} → {parser_name}")
            result.stage = "parsing"

            if parse_func:
                doc_id, blocks, original_text = parse_func(
                    file_path, parser_name
                )
            else:
                doc_id, blocks, original_text = self._default_parse(
                    file_path, parser_name, prescreen_result
                )

            result.stage = "adapting"

            # ========================================
            # 阶段4: 差异化清洗
            # ========================================
            logger.info(f"[Pipeline] 阶段4: 差异化清洗 - {len(blocks)} 个块")
            cleaned_blocks = self.cleaner.clean_blocks(blocks)
            result.stage = "cleaning"

            # ========================================
            # 阶段5: 文档质检打分
            # ========================================
            logger.info(f"[Pipeline] 阶段5: 质检打分 - {filename}")
            quality_report = self.scorer.score_document(
                doc_id, filename, cleaned_blocks, original_text
            )
            result.quality = quality_report
            result.stage = "quality_scoring"

            # 低分阻断
            if quality_report.status == "block":
                result.status = "blocked"
                result.error_msg = (
                    f"质检不通过 (得分: {quality_report.overall_score:.0f})，"
                    f"已送人工重试台"
                )
                self._send_to_manual_retry(result, cleaned_blocks)
                return result

            # ========================================
            # 阶段6: 智能分块
            # ========================================
            logger.info(f"[Pipeline] 阶段6: 智能分块 - {len(cleaned_blocks)} 个块")
            result.blocks_before_chunking = len(cleaned_blocks)
            chunks = self.chunker.chunk_blocks(doc_id, cleaned_blocks)
            result.stage = "chunking"

            # ========================================
            # 阶段7: Chunk 去重
            # ========================================
            logger.info(f"[Pipeline] 阶段7: Chunk去重 - {len(chunks)} 个chunk")
            deduped_chunks = self.dedup.deduplicate(chunks)
            result.stage = "deduplication"

            # mark 模式：dedup 给重复块打 is_duplicate 标记，此处统一过滤。
            # （去重器内已豁免表格块，表格数据行不会被标记）
            final_chunks = deduped_chunks if DEDUP["action"] == "remove" else [
                c for c in deduped_chunks
                if not c.metadata.get("is_duplicate", False)
            ]

            # ========================================
            # 完成
            # ========================================
            result.chunks = final_chunks
            result.status = "success"
            result.elapsed_time = time.time() - start_time

            # 注册哈希
            self.prescreen.register_hash(
                prescreen_result.sha256, filename
            )

            # 记录指标
            self._log_metrics(result)

            logger.info(
                f"[Pipeline] 完成: {filename} → "
                f"{len(final_chunks)} chunks, "
                f"耗时 {result.elapsed_time:.1f}s"
            )
            return result

        except Exception as e:
            result.status = "failed"
            result.error_msg = str(e)
            result.elapsed_time = time.time() - start_time
            logger.error(f"[Pipeline] 失败: {filename} → {e}")
            self._log_failure(result)
            return result

    def _default_parse(self, file_path: str, parser: str,
                       prescreen: PrescreenResult) -> Tuple[str, List[UnifiedBlock], str]:
        """
        默认解析逻辑（使用现有 document_parser 模块）
        :return: (doc_id, blocks, original_text)
        """
        try:
            from document_parser import MultiPipelineDispatcher
            dispatcher = MultiPipelineDispatcher()
            parsed_doc = dispatcher.parse(file_path, parser=parser)

            doc_id = parsed_doc.doc_id
            original_text = ""
            blocks = []

            if parsed_doc.status == "success" and parsed_doc.chunks:
                # 收集原始文本
                original_text = "\n".join(c.content for c in parsed_doc.chunks)
                # 归一适配
                blocks = self.adapter.adapt_from_parsed_chunks(
                    doc_id, parsed_doc.chunks, parser
                )

            return doc_id, blocks, original_text

        except Exception as e:
            logger.error(f"[Pipeline] 解析失败: {e}")
            raise

    def _log_metrics(self, result: PipelineResult):
        """记录管线指标"""
        if not PIPELINE_METRICS["enabled"]:
            return

        log_file = os.path.join(self.metrics_dir, "pipeline_metrics.jsonl")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "filename": result.filename,
            "status": result.status,
            "elapsed_time": result.elapsed_time,
            "chunk_count": len(result.chunks),
            "blocks_before_chunking": result.blocks_before_chunking,
            "parser": result.routing.resolve_parser() if result.routing else "",
            "quality_score": result.quality.overall_score if result.quality else 0,
            "version_tag": result.prescreen.version_tag if result.prescreen else "",
        }

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def _log_failure(self, result: PipelineResult):
        """记录失败任务到死信队列"""
        ingestion_dlq.push({
            "filename": result.filename,
            "file_path": result.file_path,
            "stage": result.stage,
            "error_msg": result.error_msg,
            "status": result.status,
        })
        logger.warning(
            f"[Pipeline] 失败任务已入死信队列: {result.filename} "
            f"(stage={result.stage}, DLQ size={len(ingestion_dlq.peek_all())})"
        )

    def retry_failed(self) -> Dict:
        """
        重跑死队列中的失败任务
        :return: {"retried": N, "succeeded": M, "still_failed": K}
        """
        items = ingestion_dlq.list_items()
        if not items:
            return {"retried": 0, "succeeded": 0, "still_failed": 0}

        retried = 0
        succeeded = 0
        still_failed = 0

        for item in items:
            file_path = item.get('file_path')
            if not file_path or not os.path.exists(file_path):
                logger.warning(f"[Pipeline] 死信任务文件不存在，跳过: {file_path}")
                still_failed += 1
                continue

            retried += 1
            try:
                result = self.process_file(file_path)
                if result.status == "success":
                    succeeded += 1
                    # 从 DLQ 移除成功任务
                    ingestion_dlq.remove(item.get('id', ''))
                    logger.info(f"[Pipeline] 重跑成功: {item.get('filename')}")
                else:
                    still_failed += 1
                    logger.warning(f"[Pipeline] 重跑仍失败: {item.get('filename')} - {result.error_msg}")
            except Exception as e:
                still_failed += 1
                logger.error(f"[Pipeline] 重跑异常: {item.get('filename')} - {e}")

        return {"retried": retried, "succeeded": succeeded, "still_failed": still_failed}

    def _send_to_manual_retry(self, result: PipelineResult,
                              blocks: List[UnifiedBlock]):
        """发送到低分文档人工重试台"""
        retry_file = os.path.join(
            MANUAL_RETRY_DIR,
            f"{result.filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        data = {
            "result": result.to_dict(),
            "blocks": [b.to_dict() for b in blocks],
            "submitted_at": datetime.now().isoformat(),
        }

        with open(retry_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"[Pipeline] 已送人工重试台: {retry_file}")

    def get_pipeline_stats(self) -> Dict:
        """获取管线统计信息"""
        metrics_file = os.path.join(self.metrics_dir, "pipeline_metrics.jsonl")
        if not os.path.exists(metrics_file):
            return {"message": "暂无管线指标数据"}

        entries = []
        with open(metrics_file, 'r', encoding='utf-8') as f:
            for line in f:
                entries.append(json.loads(line.strip()))

        if not entries:
            return {"message": "暂无管线指标数据"}

        total = len(entries)
        success = sum(1 for e in entries if e["status"] == "success")
        failed = sum(1 for e in entries if e["status"] == "failed")
        blocked = sum(1 for e in entries if e["status"] == "blocked")
        total_chunks = sum(e.get("chunk_count", 0) for e in entries)
        avg_time = sum(e.get("elapsed_time", 0) for e in entries) / total
        avg_quality = sum(
            e.get("quality_score", 0) for e in entries
        ) / total

        return {
            "total_files": total,
            "success": success,
            "failed": failed,
            "blocked": blocked,
            "total_chunks": total_chunks,
            "avg_elapsed_time": round(avg_time, 2),
            "avg_quality_score": round(avg_quality, 1),
        }



