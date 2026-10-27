# -*- coding: utf-8 -*-
"""
企业级知识库 - 差异化清洗算子

按 block_type 执行不同清洗策略：
- text: 合并断行、去页眉页脚、压缩空白、清理乱码
- table: 修复表格格式、去空行
- equation: 保留原始格式，仅清理前后空白
- image_ocr: 去 OCR 噪声、修复断行
- email: 提取正文、去邮件头

业务黑名单：可配置标记/剔除
"""

import re
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

from block_adapter import UnifiedBlock
from pipeline_config import CLEANER_NOISE_PATTERNS, NOISE_REMOVE_ENABLED, LINE_MERGE_ENABLED

logger = logging.getLogger(__name__)


@dataclass
class CleaningStats:
    """清洗统计"""
    total_blocks: int = 0
    cleaned_blocks: int = 0
    removed_blocks: int = 0
    noise_marked: int = 0
    noise_removed: int = 0
    header_footer_removed: int = 0
    chars_cleaned: int = 0
    lines_merged: int = 0


class DifferentialCleaner:
    """
    差异化清洗算子

    根据 block_type 执行不同清洗策略
    """

    def __init__(self):
        self.stats = CleaningStats()
        self._compile_noise_patterns()

    def _compile_noise_patterns(self):
        """预编译噪声模式正则"""
        self.noise_patterns = {}
        for noise_type, patterns in CLEANER_NOISE_PATTERNS.items():
            compiled = []
            for p in patterns:
                try:
                    compiled.append(re.compile(p, re.MULTILINE | re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"噪声模式编译失败 [{noise_type}]: {p} → {e}")
            self.noise_patterns[noise_type] = compiled

    def clean_blocks(self, blocks: List[UnifiedBlock]) -> List[UnifiedBlock]:
        """
        对块列表执行差异化清洗

        :param blocks: UnifiedBlock 列表
        :return: 清洗后的块列表（可能减少）
        """
        self.stats = CleaningStats()
        self.stats.total_blocks = len(blocks)

        result = []
        for block in blocks:
            cleaned = self._clean_single_block(block)

            if cleaned is None:
                # 块被移除
                self.stats.removed_blocks += 1
                continue

            if cleaned.is_noise:
                self.stats.noise_marked += 1
                if NOISE_REMOVE_ENABLED.get(cleaned.noise_type, False):
                    self.stats.noise_removed += 1
                    continue

            if cleaned.content != block.content:
                self.stats.cleaned_blocks += 1

            result.append(cleaned)

        logger.info(
            f"[Cleaner] 清洗完成: "
            f"总块数={self.stats.total_blocks}, "
            f"清洗={self.stats.cleaned_blocks}, "
            f"移除={self.stats.removed_blocks}, "
            f"噪声标记={self.stats.noise_marked}"
        )
        return result

    def _clean_single_block(self, block: UnifiedBlock) -> UnifiedBlock:
        """
        清洗单个块

        :return: 清洗后的块，None 表示应移除
        """
        # 复制块（避免修改原始数据）
        cleaned = UnifiedBlock(
            block_id=block.block_id,
            doc_id=block.doc_id,
            content=block.content,
            block_type=block.block_type,
            block_index=block.block_index,
            parent_title=block.parent_title,
            title_level=block.title_level,
            page_number=block.page_number,
            bbox=block.bbox,
            is_header=block.is_header,
            is_footer=block.is_footer,
            is_noise=block.is_noise,
            noise_type=block.noise_type,
            parser_source=block.parser_source,
            original_element_type=block.original_element_type,
            language=block.language,
            quality_score=block.quality_score,
        )

        # 按类型分发清洗
        if block.block_type == "text":
            cleaned.content = self._clean_text(block.content)
        elif block.block_type == "table":
            cleaned.content = self._clean_table(block.content)
        elif block.block_type == "equation":
            cleaned.content = self._clean_equation(block.content)
        elif block.block_type == "image_ocr":
            cleaned.content = self._clean_image_ocr(block.content)
        elif block.block_type == "email":
            cleaned.content = self._clean_email(block.content)
        elif block.block_type == "code":
            cleaned.content = self._clean_code(block.content)
        else:
            cleaned.content = self._clean_text(block.content)

        # 检测页眉页脚
        if self._is_header_footer(cleaned.content):
            cleaned.is_header = True
            cleaned.is_footer = True
            self.stats.header_footer_removed += 1
            return None  # 移除页眉页脚

        # 检测业务黑名单
        noise_type = self._check_blacklist(cleaned.content)
        if noise_type:
            cleaned.is_noise = True
            cleaned.noise_type = noise_type

        # 空块移除
        if not cleaned.content.strip():
            return None

        return cleaned

    # ============================================================
    # 各类型清洗方法
    # ============================================================

    def _clean_text(self, text: str) -> str:
        """清洗文本块"""
        if not text:
            return ""

        # 1. 清理控制字符（保留换行和制表符）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # 2. 清理乱码（常见 PDF 解析产生的乱码）
        text = self._remove_garbled(text)

        # 3. 合并断行（中文行尾无标点 + 下一行非大写开头）
        if LINE_MERGE_ENABLED:
            text = self._merge_broken_lines(text)

        # 4. 压缩空白
        text = self._compress_whitespace(text)

        # 5. 规范化标点
        text = self._normalize_punctuation(text)

        return text.strip()

    def _clean_table(self, text: str) -> str:
        """清洗表格块"""
        if not text:
            return ""

        # 去除连续空行
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 去除行首行尾空白
        lines = [line.strip() for line in text.split('\n')]

        # 去除全空行
        lines = [line for line in lines if line]

        return '\n'.join(lines)

    def _clean_equation(self, text: str) -> str:
        """清洗公式块（保留原始格式）"""
        return text.strip()

    def _clean_image_ocr(self, text: str) -> str:
        """清洗 OCR 文本"""
        if not text:
            return ""

        # 1. 去控制字符
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

        # 2. 合并 OCR 断行
        text = self._merge_broken_lines(text)

        # 3. 压缩空白
        text = self._compress_whitespace(text)

        # 4. 修复常见 OCR 错误（空格插入中文字符间）
        text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', text)

        return text.strip()

    def _clean_email(self, text: str) -> str:
        """清洗邮件（提取正文）"""
        if not text:
            return ""

        # 尝试跳过头部
        lines = text.split('\n')
        body_start = 0
        for i, line in enumerate(lines):
            if line.strip() == '' and i > 0:
                body_start = i + 1
                break

        body = '\n'.join(lines[body_start:])
        return self._clean_text(body)

    def _clean_code(self, text: str) -> str:
        """清洗代码块"""
        return text.strip()

    # ============================================================
    # 通用清洗方法
    # ============================================================

    def _remove_garbled(self, text: str) -> str:
        """清理乱码字符"""
        # 替换常见乱码模式
        replacements = [
            (r'[\ufffd]', ''),           # Unicode 替换字符
            (r'[\u200b-\u200f]', ''),    # 零宽字符
            (r'[\u2028\u2029]', '\n'),   # 行分隔符 → 换行
            (r'[\u00a0]', ' '),          # 不间断空格 → 普通空格
            (r'[\u3000]', ' '),          # 全角空格 → 半角空格
        ]
        for pattern, replacement in replacements:
            text = re.sub(pattern, replacement, text)
        return text

    def _merge_broken_lines(self, text: str) -> str:
        """
        合并断行

        规则：如果一行末尾无标点，且下一行开头非大写/非数字，
        则认为是一句话被拆断，合并两行
        """
        lines = text.split('\n')
        if len(lines) <= 1:
            return text

        result = [lines[0]]
        merge_count = 0

        for i in range(1, len(lines)):
            prev = result[-1]
            curr = lines[i]

            if not prev or not curr:
                result.append(curr)
                continue

            prev_end = prev.rstrip()[-1] if prev.rstrip() else ''
            curr_start = curr.lstrip()[0] if curr.lstrip() else ''

            # 判断是否需要合并
            # 前一行末尾不是句子结束符
            not_sentence_end = prev_end not in '。！？；.!?;\n'
            # 当前行开头不是大写/数字/列表标记
            not_new_start = not (
                curr_start.isupper() or
                curr_start.isdigit() or
                curr.lstrip().startswith(('#', '-', '*', '1.', '2.', '3.'))
            )

            if not_sentence_end and not_new_start:
                # 合并：去掉前一行末尾空格，直接拼接
                result[-1] = prev.rstrip() + curr.lstrip()
                merge_count += 1
            else:
                result.append(curr)

        self.stats.lines_merged += merge_count
        return '\n'.join(result)

    def _compress_whitespace(self, text: str) -> str:
        """压缩空白"""
        # 多个空格 → 单空格
        text = re.sub(r'[ \t]+', ' ', text)
        # 多个空行 → 双空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text

    def _normalize_punctuation(self, text: str) -> str:
        """规范化标点（全角/半角统一）"""
        # 英文标点 → 中文标点（仅在中文语境下）
        # 这里只做简单处理：连续两个句号 → 一个句号
        text = re.sub(r'。。+', '。', text)
        text = re.sub(r'，，+', '，', text)
        return text

    def _is_header_footer(self, text: str) -> bool:
        """检测是否为页眉页脚"""
        text = text.strip()
        if not text or len(text) > 100:
            return False

        for pattern in self.noise_patterns.get("header_footer", []):
            if pattern.match(text):
                return True
        return False

    def _check_blacklist(self, text: str) -> str:
        """
        检查业务黑名单
        :return: 噪声类型，空字符串表示未命中
        """
        for noise_type, patterns in self.noise_patterns.items():
            if noise_type == "header_footer":
                continue  # 页眉页脚单独处理
            for pattern in patterns:
                if pattern.search(text):
                    return noise_type
        return ""



