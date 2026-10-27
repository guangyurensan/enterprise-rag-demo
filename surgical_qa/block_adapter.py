# -*- coding: utf-8 -*-
"""
企业级知识库 - 归一适配器

将各解析器（MinerU/Docling/Unstructured）的输出
统一为标准化 Block 结构，携带完整元数据
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# 统一 Block 数据结构
# ============================================================
@dataclass
class UnifiedBlock:
    """
    归一化后的文档块

    所有解析器的输出都统一为此结构
    """
    block_id: str                    # 块唯一ID
    doc_id: str                      # 所属文档ID
    content: str                     # 文本内容
    block_type: str                  # text / table / equation / image_ocr / email / code
    block_index: int                 # 块序号

    # 层级结构
    parent_title: str = ""           # 所属章节标题
    title_level: int = 0             # 标题层级 (0=无标题, 1=H1, 2=H2, ...)

    # 位置信息
    page_number: int = -1            # 页码
    bbox: Optional[List] = None      # 边界框 [x1, y1, x2, y2]

    # 清洗标记
    is_header: bool = False          # 是否为页眉
    is_footer: bool = False          # 是否为页脚
    is_noise: bool = False           # 是否为噪声（业务黑名单）
    noise_type: str = ""             # 噪声类型

    # 来源追踪
    parser_source: str = ""          # 来源解析器
    original_element_type: str = ""  # 原始 element 类型

    # 语言
    language: str = "zh"             # zh / en / mixed

    # 质量标记
    quality_score: float = 1.0       # 块质量分 (0-1)

    def to_dict(self) -> Dict:
        return {
            "block_id": self.block_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "block_type": self.block_type,
            "block_index": self.block_index,
            "parent_title": self.parent_title,
            "title_level": self.title_level,
            "page_number": self.page_number,
            "bbox": self.bbox,
            "is_header": self.is_header,
            "is_footer": self.is_footer,
            "is_noise": self.is_noise,
            "noise_type": self.noise_type,
            "parser_source": self.parser_source,
            "original_element_type": self.original_element_type,
            "language": self.language,
            "quality_score": self.quality_score,
        }


# ============================================================
# 归一适配器
# ============================================================
class BlockAdapter:
    """
    归一适配器

    将各解析器的输出转换为统一的 UnifiedBlock 结构
    """

    def __init__(self, parser_source: str = ""):
        self.parser_source = parser_source

    def adapt_from_parsed_chunks(self, doc_id: str, chunks: list,
                                 parser_source: str = "") -> List[UnifiedBlock]:
        """
        从现有 ParsedChunk 列表转换（兼容旧管线）

        :param doc_id: 文档ID
        :param chunks: ParsedChunk 列表
        :param parser_source: 解析器来源
        :return: UnifiedBlock 列表
        """
        blocks = []
        for i, chunk in enumerate(chunks):
            block = UnifiedBlock(
                block_id=getattr(chunk, "chunk_id", f"{doc_id}_block_{i:05d}"),
                doc_id=doc_id,
                content=chunk.content,
                block_type=self._map_chunk_type(getattr(chunk, "chunk_type", "text")),
                block_index=i,
                parent_title=getattr(chunk, "title", ""),
                page_number=getattr(chunk, "page_number", -1),
                bbox=getattr(chunk, "bbox"),
                parser_source=parser_source or self.parser_source,
                original_element_type=getattr(chunk, "chunk_type", "text"),
                language=getattr(chunk, "language", "zh"),
            )
            blocks.append(block)
        return blocks

    def adapt_from_markdown(self, doc_id: str, markdown_content: str,
                            parser_source: str = "") -> List[UnifiedBlock]:
        """
        从 Markdown 内容转换（适用于 MinerU/Docling 输出）

        :param doc_id: 文档ID
        :param markdown_content: Markdown 文本
        :param parser_source: 解析器来源
        :return: UnifiedBlock 列表
        """
        blocks = []
        current_title = ""
        current_title_level = 0
        current_content = []
        block_idx = 0

        lines = (markdown_content or "").split("\n")

        for line in lines:
            stripped = line.strip()

            # 检测标题
            title_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            if title_match:
                # 保存之前的内容
                if current_content:
                    content_text = "\n".join(current_content).strip()
                    if content_text:
                        block = UnifiedBlock(
                            block_id=f"{doc_id}_block_{block_idx:05d}",
                            doc_id=doc_id,
                            content=content_text,
                            block_type=self._detect_block_type(content_text),
                            block_index=block_idx,
                            parent_title=current_title,
                            title_level=current_title_level,
                            parser_source=parser_source or self.parser_source,
                        )
                        blocks.append(block)
                        block_idx += 1

                current_title = title_match.group(2).strip()
                current_title_level = len(title_match.group(1))
                current_content = []
                continue

            # 检测代码块
            if stripped.startswith("```"):
                if current_content:
                    content_text = "\n".join(current_content).strip()
                    if content_text:
                        block = UnifiedBlock(
                            block_id=f"{doc_id}_block_{block_idx:05d}",
                            doc_id=doc_id,
                            content=content_text,
                            block_type="code",
                            block_index=block_idx,
                            parent_title=current_title,
                            title_level=current_title_level,
                            parser_source=parser_source or self.parser_source,
                        )
                        blocks.append(block)
                        block_idx += 1
                    current_content = []
                continue

            # 检测表格（Markdown 表格行）
            if re.match(r'^\|.*\|.*\|', stripped):
                # 收集连续表格行
                table_lines = [stripped]
                continue

            current_content.append(line)

        # 处理最后一段
        if current_content:
            content_text = "\n".join(current_content).strip()
            if content_text:
                block = UnifiedBlock(
                    block_id=f"{doc_id}_block_{block_idx:05d}",
                    doc_id=doc_id,
                    content=content_text,
                    block_type=self._detect_block_type(content_text),
                    block_index=block_idx,
                    parent_title=current_title,
                    title_level=current_title_level,
                    parser_source=parser_source or self.parser_source,
                )
                blocks.append(block)

        return blocks

    def adapt_from_elements(self, doc_id: str, elements: list,
                            parser_source: str = "") -> List[UnifiedBlock]:
        """
        从 Unstructured elements 转换

        :param doc_id: 文档ID
        :param elements: Unstructured element 列表
        :param parser_source: 解析器来源
        :return: UnifiedBlock 列表
        """
        blocks = []
        current_title = ""
        current_content = []
        block_idx = 0

        for elem in elements:
            elem_type = type(elem).__name__

            if elem_type in ("Title", "SectionHeader"):
                # 保存之前的内容
                if current_content:
                    content_text = "\n".join(current_content).strip()
                    if content_text:
                        block = UnifiedBlock(
                            block_id=f"{doc_id}_block_{block_idx:05d}",
                            doc_id=doc_id,
                            content=content_text,
                            block_type="text",
                            block_index=block_idx,
                            parent_title=current_title,
                            parser_source=parser_source or self.parser_source,
                            original_element_type="paragraph",
                        )
                        blocks.append(block)
                        block_idx += 1
                    current_content = []

                current_title = str(elem)
                continue

            if elem_type == "Table":
                # 保存之前内容
                if current_content:
                    content_text = "\n".join(current_content).strip()
                    if content_text:
                        block = UnifiedBlock(
                            block_id=f"{doc_id}_block_{block_idx:05d}",
                            doc_id=doc_id,
                            content=content_text,
                            block_type="text",
                            block_index=block_idx,
                            parent_title=current_title,
                            parser_source=parser_source or self.parser_source,
                        )
                        blocks.append(block)
                        block_idx += 1
                    current_content = []

                # 表格单独成块
                block = UnifiedBlock(
                    block_id=f"{doc_id}_block_{block_idx:05d}",
                    doc_id=doc_id,
                    content=str(elem),
                    block_type="table",
                    block_index=block_idx,
                    parent_title=current_title,
                    parser_source=parser_source or self.parser_source,
                    original_element_type="Table",
                )
                blocks.append(block)
                block_idx += 1
                continue

            if elem_type == "Image":
                block = UnifiedBlock(
                    block_id=f"{doc_id}_block_{block_idx:05d}",
                    doc_id=doc_id,
                    content=str(elem),
                    block_type="image_ocr",
                    block_index=block_idx,
                    parent_title=current_title,
                    parser_source=parser_source or self.parser_source,
                    original_element_type="Image",
                )
                blocks.append(block)
                block_idx += 1
                continue

            # 普通文本
            current_content.append(str(elem))

        # 处理最后一段
        if current_content:
            content_text = "\n".join(current_content).strip()
            if content_text:
                block = UnifiedBlock(
                    block_id=f"{doc_id}_block_{block_idx:05d}",
                    doc_id=doc_id,
                    content=content_text,
                    block_type="text",
                    block_index=block_idx,
                    parent_title=current_title,
                    parser_source=parser_source or self.parser_source,
                )
                blocks.append(block)

        return blocks

    def _detect_block_type(self, text: str) -> str:
        """检测块类型"""
        if re.search(r'\\begin\{(equation|align|matrix)', text):
            return "equation"
        if re.match(r'^\|.*\|.*\|', text) or text.count('\t') > 3:
            return "table"
        if text.startswith("图") or text.startswith("Fig"):
            return "image_ocr"
        return "text"

    def _map_chunk_type(self, chunk_type: str) -> str:
        """映射旧的 chunk_type 到新的 block_type"""
        mapping = {
            "text": "text",
            "table": "table",
            "formula": "equation",
            "image": "image_ocr",
            "email": "email",
        }
        return mapping.get(chunk_type, "text")



