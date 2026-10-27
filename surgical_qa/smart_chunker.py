# -*- coding: utf-8 -*-
"""
企业级知识库 - 纯确定性智能分块模块

设计原则：
- 不做语义理解、不写复杂正则、不调用 LLM
- 输入：清洗质检后的 Block（UnifiedBlock）
- 输出：Chunk + 固定 meta
- 完全可预测、可测试

处理策略：
- text：LTP 分句 → token 看门滑动窗口 → 标点回退截断 → 短碎片合并
- table：保护表头，按行组分块，不跑分句
- equation / image_ocr：整块保留，不切

三项增强：
- Context Enrichment：标题前缀注入 chunk content
- 无重叠滑动窗口（step = len(picked_sents)）
- 段落边界感知（遇 \\n\\n 强制切分）
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from block_adapter import UnifiedBlock
from pipeline_config import SMART_CHUNKING

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================
@dataclass
class Chunk:
    """切片输出，送入 Embedding"""
    chunk_id: str
    doc_id: str
    content: str               # 已注入标题前缀的最终文本
    chunk_index: int
    chunk_type: str            # text / table / equation / image_ocr
    parent_title: str
    title_level: int
    page_number: int
    parser_source: str
    truncated: bool = False    # 主干唯一标记：是否经过截断
    # 预留扩展字段（主干不填充）
    tiny_chunk: bool = False
    clause_id: Optional[str] = None
    quality_score: Optional[float] = None
    # 向后兼容字段
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "chunk_type": self.chunk_type,
            "parent_title": self.parent_title,
            "title_level": self.title_level,
            "page_number": self.page_number,
            "parser_source": self.parser_source,
            "truncated": self.truncated,
            "tiny_chunk": self.tiny_chunk,
            "metadata": self.metadata,
        }


# ============================================================
# 工具函数
# ============================================================
def count_token(text: str) -> int:
    """
    轻量 token 估算：len(text) * ratio
    中文约 1 字 ≈ 1-1.5 token，取 0.55 留余量
    """
    ratio = SMART_CHUNKING["token_estimate_ratio"]
    return max(1, int(len(text) * ratio))


def token_slice(text: str, max_token: int) -> str:
    """取前 max_token 个 token 对应的文本（估算）"""
    ratio = SMART_CHUNKING["token_estimate_ratio"]
    char_pos = min(int(max_token / ratio), len(text))
    return text[:char_pos]


def ltp_stn_split(text: str) -> List[str]:
    """
    分句：优先 LTP，降级到规则分句
    不修改 LTP 结果，主干信任分句输出
    """
    # 尝试 LTP
    try:
        from ltp import LTP
        ltp = LTP()
        result = ltp.sent_split([text])
        sentences = []
        for group in result:
            for sent in group:
                s = sent.strip()
                if s:
                    sentences.append(s)
        if sentences:
            return sentences
    except (ImportError, Exception):
        pass

    # 降级：规则分句
    return _rule_sent_split(text)


def _rule_sent_split(text: str) -> List[str]:
    """规则分句：按句子结束符切分，保留结束符"""
    endings = set('。！？；\n.!?;')
    sentences = []
    current = []

    for char in text:
        current.append(char)
        if char in endings:
            sent = ''.join(current).strip()
            if sent:
                sentences.append(sent)
            current = []

    if current:
        remaining = ''.join(current).strip()
        if remaining:
            sentences.append(remaining)

    return sentences


def find_last_punct(text: str, punct_order: List[str]) -> Optional[int]:
    """
    在 text 中按优先级找最后一个标点的位置
    :return: 标点位置（0-based），None 表示没找到
    """
    best_pos = -1
    for punct in punct_order:
        pos = text.rfind(punct)
        if pos > best_pos:
            best_pos = pos
    return best_pos if best_pos >= 0 else None


def parse_markdown_table(table_md: str) -> Tuple[List[str], List[str]]:
    """
    解析 Markdown 表格 → (表头行列表, 数据行列表)
    只处理标准 | 格式，非标准则整块当一行返回
    """
    lines = [l.strip() for l in table_md.strip().split('\n') if l.strip()]
    if not lines:
        return [], []

    # 检查是否为标准 markdown 表格（至少有 | 分隔）
    if not all('|' in l for l in lines[:2]):
        # 非标准格式，整块返回
        return [], lines

    header = [lines[0]]
    data_start = 1

    # 跳过分隔行 (|---|---|)
    if len(lines) > 1 and re.match(r'^\|[\s\-:|]+\|$', lines[1]):
        header.append(lines[1])
        data_start = 2

    rows = lines[data_start:]
    return header, rows


def build_md_table(header_lines: List[str], rows: List[str]) -> str:
    """用表头行 + 数据行重建 Markdown 表格"""
    return '\n'.join(header_lines + rows)


def enrich_context(content: str, parent_title: str, title_level: int) -> str:
    """
    Context Enrichment：将标题前缀注入 chunk content
    让 embedding 向量携带层级上下文
    """
    if not SMART_CHUNKING.get("context_enrichment", True):
        return content
    if not parent_title:
        return content
    return f"[{parent_title}] {content}"


# ============================================================
# 核心分块函数
# ============================================================
def split_block(block: UnifiedBlock, doc_id: str,
                chunk_start_idx: int) -> List[Chunk]:
    """
    总入口：按 block_type 分发
    """
    if block.block_type == "text":
        return split_text_block(block, doc_id, chunk_start_idx)
    elif block.block_type == "table":
        return split_table_block(block, doc_id, chunk_start_idx)
    elif block.block_type in ("equation", "image_ocr"):
        # 整块直接输出，不切
        content = enrich_context(block.content, block.parent_title, block.title_level)
        return [Chunk(
            chunk_id=f"{doc_id}_chunk_{chunk_start_idx:05d}",
            doc_id=doc_id,
            content=content,
            chunk_index=chunk_start_idx,
            chunk_type=block.block_type,
            parent_title=block.parent_title,
            title_level=block.title_level,
            page_number=block.page_number,
            parser_source=block.parser_source,
        )]
    else:
        return []


def split_text_block(block: UnifiedBlock, doc_id: str,
                     chunk_start_idx: int) -> List[Chunk]:
    """
    文本分块主干：
    Step1: LTP 分句
    Step2: 句子滑动窗口组装（token 看门，动态收缩，段落边界感知）
    Step3: 超长兜底 → 标点回退截断
    Step4: 短碎片就近合并
    """
    raw_text = block.content
    if not raw_text.strip():
        return []

    parent_title = block.parent_title
    title_level = block.title_level
    page = block.page_number

    MAX_TOKEN = SMART_CHUNKING["max_embed_token"]
    MIN_TOKEN = SMART_CHUNKING["min_token"]
    WIN_MAX = SMART_CHUNKING["win_size_max"]
    PUNCT_ORDER = SMART_CHUNKING["punct_order"]

    # Step1: 分句
    sentences = ltp_stn_split(raw_text)
    if not sentences:
        return []

    # Step2: 句子滑动窗口组装
    chunks: List[Chunk] = []
    idx = chunk_start_idx
    i = 0
    n = len(sentences)

    while i < n:
        # 从最大窗口尝试，逐步缩小到 1
        current_win = WIN_MAX
        picked_sents = []

        while current_win >= 1:
            candidate = sentences[i:i + current_win]
            joined = "".join(candidate)
            if count_token(joined) <= MAX_TOKEN:
                picked_sents = candidate
                break
            current_win -= 1

        if not picked_sents:
            # 单句本身就超限 → 标点回退截断
            sent = sentences[i]
            sub_parts = truncate_by_punct(sent, MAX_TOKEN, PUNCT_ORDER)
            for part in sub_parts:
                content = enrich_context(part, parent_title, title_level)
                chunks.append(Chunk(
                    chunk_id=f"{doc_id}_chunk_{idx:05d}",
                    doc_id=doc_id,
                    content=content,
                    chunk_index=idx,
                    chunk_type="text",
                    parent_title=parent_title,
                    title_level=title_level,
                    page_number=page,
                    parser_source=block.parser_source,
                    truncated=True,
                ))
                idx += 1
            i += 1
            continue

        # 段落边界检查：如果 picked_sents 中包含段落分隔，在最接近窗口末尾的段落边界处切
        chunk_text_raw = "".join(picked_sents)
        if '\n\n' in chunk_text_raw:
            # 在段落边界处切分：只取第一个段落内的句子
            para_sents = []
            for s in picked_sents:
                if '\n\n' in s:
                    # 句子本身包含段落分隔，拆开
                    parts = s.split('\n\n')
                    para_sents.append(parts[0])
                    break
                para_sents.append(s)
            if para_sents:
                picked_sents = para_sents

        # 组装 chunk
        chunk_text = "".join(picked_sents)
        content = enrich_context(chunk_text, parent_title, title_level)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_chunk_{idx:05d}",
            doc_id=doc_id,
            content=content,
            chunk_index=idx,
            chunk_type="text",
            parent_title=parent_title,
            title_level=title_level,
            page_number=page,
            parser_source=block.parser_source,
            truncated=False,
        ))
        idx += 1

        # 无重叠步进：step = len(picked_sents)
        i += len(picked_sents)

    # Step3: 短碎片就近合并
    final_chunks = merge_tiny_chunks(chunks, MIN_TOKEN, MAX_TOKEN)

    # 标记 tiny_chunk
    for c in final_chunks:
        if count_token(c.content) < MIN_TOKEN:
            c.tiny_chunk = True

    return final_chunks


def truncate_by_punct(sent: str, max_token: int,
                      punct_order: List[str]) -> List[str]:
    """
    标点回退截断（超长兜底）
    不丢后半段，递归处理直到切完
    只在 。！？；：处断开，不切逗号、顿号
    """
    res = []
    remaining = sent

    while True:
        if count_token(remaining) <= max_token:
            res.append(remaining)
            break

        # 截到 token 上限对应的文本
        prefix = token_slice(remaining, max_token)

        # 在 prefix 中向前找最高优先级标点
        cut_pos = find_last_punct(prefix, punct_order)

        if cut_pos is None:
            # 整个 prefix 里没有合法标点，硬切在 token 边界
            cut_pos = len(prefix)

        part = remaining[:cut_pos + 1]
        res.append(part)
        remaining = remaining[cut_pos + 1:].strip()

        if not remaining:
            break

    return res


def merge_tiny_chunks(chunks: List[Chunk], min_token: int,
                      max_token: int) -> List[Chunk]:
    """
    短碎片就近合并
    约束：只在同 parent_title 内合并（输入天然满足）
    合并后做 token 校验，超限则放弃合并
    """
    if len(chunks) <= 1:
        return chunks

    out = [chunks[0]]
    for ck in chunks[1:]:
        last = out[-1]
        last_tokens = count_token(last.content)

        if last_tokens < min_token:
            # 尝试合并
            combined_text = last.content + ck.content
            if count_token(combined_text) <= max_token:
                # 安全，合并
                merged = Chunk(
                    chunk_id=last.chunk_id,
                    doc_id=last.doc_id,
                    content=combined_text,
                    chunk_index=last.chunk_index,
                    chunk_type=last.chunk_type,
                    parent_title=last.parent_title,
                    title_level=last.title_level,
                    page_number=last.page_number,
                    parser_source=last.parser_source,
                    truncated=last.truncated or ck.truncated,
                )
                out[-1] = merged
                continue

        out.append(ck)

    return out


def split_table_block(block: UnifiedBlock, doc_id: str,
                      chunk_start_idx: int) -> List[Chunk]:
    """
    表格分块：保护表头，按行组分块，不跑分句
    """
    table_cfg = SMART_CHUNKING["table"]
    group_size = table_cfg["group_size"]
    table_md = block.content

    header_lines, rows = parse_markdown_table(table_md)

    # 小表（行数 <= group_size）整块保留
    if len(rows) <= group_size:
        content = enrich_context(table_md, block.parent_title, block.title_level)
        return [Chunk(
            chunk_id=f"{doc_id}_chunk_{chunk_start_idx:05d}",
            doc_id=doc_id,
            content=content,
            chunk_index=chunk_start_idx,
            chunk_type="table",
            parent_title=block.parent_title,
            title_level=block.title_level,
            page_number=block.page_number,
            parser_source=block.parser_source,
        )]

    # 大表：按行分组，每组携带完整表头
    chunks = []
    idx = chunk_start_idx
    for i in range(0, len(rows), group_size):
        group_rows = rows[i:i + group_size]
        sub_table = build_md_table(header_lines, group_rows)
        content = enrich_context(sub_table, block.parent_title, block.title_level)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_chunk_{idx:05d}",
            doc_id=doc_id,
            content=content,
            chunk_index=idx,
            chunk_type="table",
            parent_title=block.parent_title,
            title_level=block.title_level,
            page_number=block.page_number,
            parser_source=block.parser_source,
        ))
        idx += 1

    return chunks


# ============================================================
# 对外接口（兼容 pipeline 调用）
# ============================================================
class SmartChunker:
    """
    智能分块器（纯确定性）

    对外接口保持不变：chunk_blocks(doc_id, blocks) -> List[Chunk]
    """

    def chunk_blocks(self, doc_id: str,
                     blocks: List[UnifiedBlock]) -> List[Chunk]:
        """
        对清洗后的块列表执行智能分块
        """
        all_chunks = []
        chunk_idx = 0

        for block in blocks:
            if block.is_noise or not block.content.strip():
                continue

            new_chunks = split_block(block, doc_id, chunk_idx)
            all_chunks.extend(new_chunks)
            chunk_idx += len(new_chunks)

        logger.info(
            f"[SmartChunker] 分块完成: {len(blocks)} 个块 → "
            f"{len(all_chunks)} 个 chunk"
        )
        return all_chunks
