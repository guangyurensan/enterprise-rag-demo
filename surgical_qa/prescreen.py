# -*- coding: utf-8 -*-
"""
企业级知识库 - 文件前置粗筛模块

功能:
1. SHA256 指纹计算 + 去重
2. 文件损坏检测（magic bytes 校验）
3. 文件准入权限（扩展名、大小限制）
4. 版本标记（基于文件名/哈希）
"""

import os
import re
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from datetime import datetime

from pipeline_config import FILE_PRESCREEN

logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================
@dataclass
class PrescreenResult:
    """前置粗筛结果"""
    file_path: str
    filename: str
    file_size: int
    sha256: str
    file_ext: str
    passed: bool                    # 是否通过粗筛
    reject_reason: str = ""         # 拒绝原因
    is_duplicate: bool = False      # 是否重复文件
    is_corrupted: bool = False      # 是否疑似损坏
    is_scanned_pdf: bool = False    # 是否为扫描件 PDF
    is_image_heavy: bool = False    # 是否为图片型 PDF（非扫描件但图片密集）
    is_complex_excel: bool = False  # 是否为复杂 Excel（多 sheet）
    zip_subtype: str = ""           # ZIP 系文件子类型: docx / pptx / xlsx / ""
    version_tag: str = ""           # 版本标记
    metadata: Dict = field(default_factory=dict)


# ============================================================
# 文件前置粗筛器
# ============================================================
class FilePrescreen:
    """
    文件前置粗筛器

    流程:
    1. 扩展名准入检查
    2. 文件大小检查
    3. SHA256 指纹计算
    4. 重复文件检测
    5. 文件损坏检测（magic bytes）
    6. PDF 类型探测（原生/扫描）
    7. Excel 复杂度探测
    8. 版本标记
    """

    def __init__(self, known_hashes: Dict[str, str] = None):
        """
        :param known_hashes: 已知文件哈希 {sha256: filename}，用于去重
        """
        self.known_hashes = known_hashes or {}
        self.cfg = FILE_PRESCREEN

    def prescreen(self, file_path: str) -> PrescreenResult:
        """
        执行前置粗筛
        :param file_path: 文件路径
        :return: PrescreenResult
        """
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        file_size = os.path.getsize(file_path)

        result = PrescreenResult(
            file_path=file_path,
            filename=filename,
            file_size=file_size,
            sha256="",
            file_ext=file_ext,
            passed=True,
        )

        # Step 1: 扩展名准入
        if not self._check_extension(file_ext, result):
            return result

        # Step 2: 文件大小检查
        if not self._check_file_size(file_size, result):
            return result

        # Step 3: SHA256 指纹
        result.sha256 = self._compute_sha256(file_path)

        # Step 4: 重复检测
        if self._check_duplicate(result):
            return result

        # Step 5: 损坏检测（magic bytes）
        if not self._check_integrity(file_path, file_ext, result):
            return result

        # Step 6: ZIP 系文件子类型探测（区分 docx/pptx/xlsx）
        if file_ext in (".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls"):
            self._detect_zip_subtype(file_path, file_ext, result)

        # Step 7: PDF 类型探测（扫描件 + 图片型）
        if file_ext == ".pdf":
            self._detect_pdf_type(file_path, result)

        # Step 8: Excel 复杂度探测
        if file_ext in (".xlsx", ".xls"):
            self._detect_excel_complexity(file_path, result)

        # Step 9: 版本标记
        result.version_tag = self._generate_version_tag(filename, result.sha256)

        return result

    def _check_extension(self, ext: str, result: PrescreenResult) -> bool:
        """检查扩展名是否在允许列表中"""
        allowed = self.cfg["allowed_extensions"]
        if ext not in allowed:
            result.passed = False
            result.reject_reason = f"不支持的文件类型: {ext}"
            logger.warning(f"[Prescreen] 拒绝 {result.filename}: {result.reject_reason}")
            return False
        return True

    def _check_file_size(self, size: int, result: PrescreenResult) -> bool:
        """检查文件大小"""
        min_size = self.cfg["min_file_size"]
        max_size = self.cfg["max_size_mb"] * 1024 * 1024

        if size < min_size:
            result.passed = False
            result.reject_reason = f"文件过小 ({size} bytes < {min_size})，疑似空文件/损坏"
            result.is_corrupted = True
            logger.warning(f"[Prescreen] 拒绝 {result.filename}: {result.reject_reason}")
            return False

        if size > max_size:
            result.passed = False
            result.reject_reason = f"文件过大 ({size / 1024 / 1024:.1f}MB > {self.cfg['max_size_mb']}MB)"
            logger.warning(f"[Prescreen] 拒绝 {result.filename}: {result.reject_reason}")
            return False

        return True

    def _compute_sha256(self, file_path: str) -> str:
        """计算文件 SHA256 指纹"""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _check_duplicate(self, result: PrescreenResult) -> bool:
        """检查是否为重复文件"""
        if result.sha256 in self.known_hashes:
            result.passed = False
            result.is_duplicate = True
            result.reject_reason = (
                f"重复文件（SHA256 匹配），已有: {self.known_hashes[result.sha256]}"
            )
            logger.info(f"[Prescreen] 跳过重复: {result.filename}")
            return True
        return False

    def _check_integrity(self, file_path: str, ext: str,
                         result: PrescreenResult) -> bool:
        """检查文件完整性（magic bytes）"""
        magic_list = self.cfg["magic_bytes"].get(ext)
        if not magic_list:
            return True  # 无 magic bytes 要求的类型

        try:
            with open(file_path, "rb") as f:
                header = f.read(16)

            for magic in magic_list:
                if header.startswith(magic):
                    return True

            result.is_corrupted = True
            result.passed = False
            result.reject_reason = f"文件头不匹配（疑似损坏），期望: {magic_list[0].hex()}"
            logger.warning(f"[Prescreen] 损坏检测: {result.filename}")
            return False

        except Exception as e:
            result.passed = False
            result.reject_reason = f"文件读取失败: {e}"
            return False

    def _detect_zip_subtype(self, file_path: str, ext: str,
                            result: PrescreenResult):
        """
        探测 ZIP 系 Office 文件的真实子类型

        DOCX/PPTX/XLSX 共享同一 magic bytes（PK\x03\x04），
        通过解压后检查内部 XML 结构来区分：
        - DOCX → word/document.xml
        - PPTX → ppt/presentation.xml
        - XLSX → xl/workbook.xml

        旧版 DOC/PPT/XLS（OLE2 格式）不做此探测，直接由扩展名决定。
        """
        # 旧版 Office（OLE2 格式）不是 ZIP 包，跳过
        if ext in (".doc", ".ppt", ".xls"):
            result.metadata["zip_subtype"] = f"ole2_{ext.lstrip('.')}"
            return

        try:
            import zipfile
            with zipfile.ZipFile(file_path, 'r') as zf:
                names = zf.namelist()

            if any(n.startswith("word/") for n in names):
                result.zip_subtype = "docx"
                result.metadata["zip_subtype"] = "docx"
            elif any(n.startswith("ppt/") for n in names):
                result.zip_subtype = "pptx"
                result.metadata["zip_subtype"] = "pptx"
            elif any(n.startswith("xl/") for n in names):
                result.zip_subtype = "xlsx"
                result.metadata["zip_subtype"] = "xlsx"
            else:
                result.zip_subtype = "unknown_zip"
                result.metadata["zip_subtype"] = "unknown_zip"
                logger.warning(
                    f"[Prescreen] ZIP 系文件无法识别子类型: {result.filename}"
                )

        except Exception as e:
            result.zip_subtype = "error"
            result.metadata["zip_subtype"] = f"error: {e}"
            logger.warning(f"[Prescreen] ZIP 子类型探测失败: {e}")

    def _detect_pdf_type(self, file_path: str, result: PrescreenResult):
        """
        探测 PDF 类型：原生文本 vs 扫描件 vs 图片型
        方法：读取文件内容，检查文本密度和图片标记
        """
        try:
            with open(file_path, "rb") as f:
                raw = f.read(20480)  # 读取 20KB 采样

            # 检查可打印字符比例（扫描件文本层极少）
            printable = sum(1 for b in raw if 32 <= b <= 126 or b in (10, 13))
            ratio = printable / max(len(raw), 1)

            if ratio < 0.3:
                result.is_scanned_pdf = True
                result.metadata["pdf_type"] = "scanned"
            else:
                # 原生 PDF，进一步检查是否图片密集
                # 通过检测 PDF 图片标记（/Image、/XObject 等）
                image_markers = raw.count(b"/Image") + raw.count(b"/XObject")
                text_blocks = raw.count(b"BT") + raw.count(b"ET")  # 文本块标记

                # 图片标记多且文本块少 → 图片型 PDF
                if image_markers > 5 and text_blocks < 3:
                    result.is_image_heavy = True
                    result.metadata["pdf_type"] = "image_heavy"
                else:
                    result.metadata["pdf_type"] = "native"

        except Exception as e:
            logger.warning(f"[Prescreen] PDF 类型探测失败: {e}")
            result.metadata["pdf_type"] = "unknown"

    def _detect_excel_complexity(self, file_path: str, result: PrescreenResult):
        """
        探测 Excel 复杂度：简单台账 vs 复杂多 sheet 报表
        """
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
            sheet_count = len(wb.sheetnames)
            total_rows = 0
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                total_rows += ws.max_row or 0
            wb.close()

            result.metadata["excel_sheets"] = sheet_count
            result.metadata["excel_rows"] = total_rows

            # 多 sheet 或行数多 → 复杂
            if sheet_count >= 3 or total_rows >= 500:
                result.is_complex_excel = True
                result.metadata["excel_type"] = "complex"
            else:
                result.metadata["excel_type"] = "simple"

        except ImportError:
            logger.warning("[Prescreen] openpyxl 未安装，Excel 复杂度探测跳过")
            result.metadata["excel_type"] = "unknown"
        except Exception as e:
            logger.warning(f"[Prescreen] Excel 复杂度探测失败: {e}")
            result.metadata["excel_type"] = "unknown"

    def _generate_version_tag(self, filename: str, sha256: str) -> str:
        """
        生成版本标记
        格式: {文件名去后缀}_{SHA256前8位}_{日期}
        """
        base = os.path.splitext(filename)[0]
        date_str = datetime.now().strftime("%Y%m%d")
        return f"{base}_{sha256[:8]}_{date_str}"

    def register_hash(self, sha256: str, filename: str):
        """注册新文件哈希（用于后续去重）"""
        self.known_hashes[sha256] = filename



