# -*- coding: utf-8 -*-
"""
企业级知识库 - 三管线文档解析器
MinerU:       论文、财报、中文密集、含大量公式的 PDF、图文密集/扫描件
Docling:      普通合同、英文文档、docx/pptx 混合文档
Unstructured: PDF、Word、Excel、HTML、邮件等通用办公文档

根据文档类型/内容特征自动选择最优解析管线，也可手动指定。
"""

import os
import re
import json
import hashlib
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# HTML 表格 → Markdown 管道表格
# ============================================================
def _render_html_table(fragment: str) -> str:
    """把单个 <table>…</table> 片段渲染成 Markdown 管道表格"""
    from html.parser import HTMLParser

    class _Collector(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.rows, self.row, self.cell = [], None, None

        def handle_starttag(self, tag, attrs):
            if tag == 'tr':
                self._flush_row()
                self.row = []
            elif tag in ('td', 'th'):
                self._flush_cell()
                self.cell = []
            elif tag == 'br' and self.cell is not None:
                self.cell.append(' ')

        def handle_endtag(self, tag):
            if tag == 'tr':
                self._flush_row()
            elif tag in ('td', 'th'):
                self._flush_cell()

        def handle_data(self, data):
            if self.cell is not None:
                text = ' '.join(data.split())
                if text:
                    self.cell.append(text)

        def _flush_cell(self):
            if self.cell is not None and self.row is not None:
                self.row.append(' '.join(self.cell).strip())
            self.cell = None

        def _flush_row(self):
            self._flush_cell()
            if self.row and any(c for c in self.row):
                self.rows.append(self.row)
            self.row = None

    c = _Collector()
    try:
        c.feed(fragment)
        c._flush_row()
    except Exception as e:
        logger.warning(f"[Parser] HTML 表格解析失败，丢弃该表: {e}")
        return ''

    rows = c.rows
    if not rows:
        return ''
    width = max(len(r) for r in rows)
    lines = []
    for i, r in enumerate(rows):
        cells = [x.replace('|', '\\|') for x in (r + [''] * (width - len(r)))]
        lines.append('| ' + ' | '.join(cells) + ' |')
        if i == 0:
            lines.append('|' + ' --- |' * width)
    return '\n'.join(lines)


def html_tables_to_markdown(text: str) -> str:
    """把文本里的 HTML 表格转成 Markdown 管道表格，非表格内容原样保留。

    MinerU 返回的 Markdown 对复杂表格仍用 HTML <table> 输出，而分块只认
    管道表（BaseParser._chunk_table），标签会原样切进 chunk。实测 4 份财报
    PDF 的标签占正文 54%~73%，检索命中的内容一半是 <td>。这里在解析出口
    统一规范化，避免「重新启用 mineru 就复发」。
    """
    if not text or '<table' not in text.lower():
        return text

    lower = text.lower()
    out, pos = [], 0
    while True:
        start = lower.find('<table', pos)
        if start < 0:
            out.append(text[pos:])
            break

        # 配对 </table>，按嵌套深度计数（PDF 表格少见嵌套，但不能假设没有）
        depth, i, end = 0, start, -1
        while i < len(text):
            nxt_open = lower.find('<table', i)
            nxt_close = lower.find('</table>', i)
            if nxt_close < 0:
                break
            if 0 <= nxt_open < nxt_close:
                depth += 1
                i = nxt_open + len('<table')
            else:
                depth -= 1
                i = nxt_close + len('</table>')
                if depth == 0:
                    end = i
                    break

        if end < 0:  # 没有闭合标签，剩余内容原样保留
            out.append(text[pos:])
            break

        out.append(text[pos:start])
        out.append(_render_html_table(text[start:end]))
        pos = end

    return ''.join(out)


# ============================================================
from config import CHUNK_SIZE, CHUNK_OVERLAP, PARSER_DISABLED, PDF_OCR_DEFAULT


# 数据结构
# ============================================================
@dataclass
class ParsedChunk:
    """解析后的文档块"""
    chunk_id: str              # 唯一标识
    doc_id: str                # 所属文档ID
    content: str               # 文本内容
    chunk_index: int           # 块序号
    metadata: Dict = field(default_factory=dict)
    title: str = ""            # 所属章节标题
    page_number: int = -1      # 页码
    bbox: Optional[List] = None
    chunk_type: str = "text"   # text / table / formula / image / email
    language: str = "zh"       # zh / en / mixed


@dataclass
class ParsedDocument:
    """解析后的文档"""
    doc_id: str
    filename: str
    file_path: str
    file_type: str             # pdf / docx / pptx / xlsx / html / eml ...
    file_size: int
    file_hash: str
    parser_used: str           # mineru / docling / unstructured
    parse_time: float
    chunks: List[ParsedChunk] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    status: str = "success"    # success / failed / partial
    error_msg: str = ""

    def to_dict(self) -> Dict:
        return {
            'doc_id': self.doc_id,
            'filename': self.filename,
            'file_path': self.file_path,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'file_hash': self.file_hash,
            'parser_used': self.parser_used,
            'parse_time': self.parse_time,
            'metadata': self.metadata,
            'status': self.status,
            'error_msg': self.error_msg,
            'chunk_count': len(self.chunks),
            'parsed_at': datetime.now().isoformat()
        }


# ============================================================
# 文档分类器 - 自动选择最优解析管线
# ============================================================
class DocumentClassifier:
    """
    根据文档特征自动分类，决定使用哪个解析器

    路由规则:
    - MinerU:       PDF + (含公式 | 中文密集 | 论文/财报特征 | 图文密集 | 扫描件)
    - Openpyxl:     Excel（合并单元格/多工作表专用，回退 docling）
    - Unstructured: HTML / 邮件 / 通用办公文档
    - Docling:      docx / pptx / 普通英文 PDF / 合同 / CSV
    """

    FORMULA_PATTERNS = [
        re.compile(r'\\[a-zA-Z]+\{'),
        re.compile(r'\$\$.*?\$\$', re.DOTALL),
        re.compile(r'\$[^$]+\$'),
        re.compile(r'\\begin\{(equation|align|matrix)'),
        re.compile(r'[∑∏∫∂√∞≈≠≤≥±×÷]'),
        re.compile(r'[A-Z]\s*=\s*[A-Z].*[+\-*/]'),
    ]

    CJK_PATTERN = re.compile(r'[\u4e00-\u9fff]')

    ACADEMIC_KEYWORDS = [
        '摘要', 'abstract', '关键词', 'keywords', '参考文献', 'references',
        '引言', 'introduction', '方法', 'method', '实验', 'experiment',
        '结论', 'conclusion', '财务报告', '资产负债表', '利润表', '现金流量',
        '营业收入', '净利润', '每股收益', '审计报告',
    ]

    # 图文密集特征
    IMAGE_HEAVY_KEYWORDS = [
        '图', '表', '示意', '如图', '下图', '上图',
        'figure', 'table', 'fig.', 'illustration', 'diagram',
    ]

    def classify(self, file_path: str, sample_text: str = "") -> str:
        """
        分类文档，返回推荐解析器
        :return: "mineru" / "docling" / "unstructured"
        """
        ext = os.path.splitext(file_path)[1].lower()

        # ===== 规则1: 非 PDF 文件按扩展名分流 =====
        # PARSER_DISABLED 中的解析器不可用时统一回退 docling（docling 支持 html/xlsx 等结构化输入）
        def route(parser):
            if parser in PARSER_DISABLED:
                return "docling"
            return parser

        # Excel / HTML / 邮件 → Unstructured
        if ext in ('.html', '.htm', '.eml', '.msg', '.xml', '.json'):
            return route("unstructured")

        # 纯文本 / Markdown / CSV → Docling
        # （Unstructured 对这类文件无优势，且依赖重）
        if ext in ('.txt', '.md', '.rst', '.csv'):
            return "docling"

        # Excel → Openpyxl 专用解析器（docling 的 markdown 转换会畸变合并单元格，
        # 实测工资表 18 块全是标题残片；失败时经 FALLBACK_ORDER 回退 docling）
        if ext in ('.xlsx', '.xls'):
            return route("openpyxl")

        # Word / PPT → Docling
        if ext in ('.docx', '.pptx', '.doc', '.ppt'):
            return "docling"

        # 非 PDF 兜底
        if ext != '.pdf':
            return route("unstructured")

        # ===== 规则2: PDF 文件智能分析 =====
        if not sample_text:
            # 无采样文本 = 扫描件/图片型 PDF。mineru 会把文件上传到云端，
            # 默认走本地 PDF_OCR_DEFAULT；数据可出域时才在 config 里改回 "mineru"
            return route(PDF_OCR_DEFAULT)

        # 公式检测
        formula_score = sum(len(p.findall(sample_text)) for p in self.FORMULA_PATTERNS)
        has_heavy_formula = formula_score >= 3

        # 中文密度
        cjk_chars = len(self.CJK_PATTERN.findall(sample_text))
        total_chars = max(len(sample_text), 1)
        cjk_ratio = cjk_chars / total_chars
        is_chinese_dense = cjk_ratio > 0.3

        # 学术/财报关键词
        text_lower = sample_text[:3000].lower()
        keyword_hits = sum(1 for kw in self.ACADEMIC_KEYWORDS if kw in text_lower)
        is_academic = keyword_hits >= 2

        # 图文密集检测
        image_hits = sum(1 for kw in self.IMAGE_HEAVY_KEYWORDS if kw in text_lower)
        is_image_heavy = image_hits >= 3

        # ===== 综合判断 =====
        # 优先级: MinerU(公式/中文/学术/图文) > Docling(普通)
        # MinerU 为云端解析，被 PARSER_DISABLED 时 route() 会回退 docling
        if has_heavy_formula or is_chinese_dense or is_academic or is_image_heavy:
            return route("mineru")
        else:
            return "docling"

    def classify_with_info(self, file_path: str, sample_text: str = "") -> Tuple[str, Dict]:
        """分类并返回详细信息"""
        ext = os.path.splitext(file_path)[1].lower()
        parser = self.classify(file_path, sample_text)

        reasons = []
        if ext in ('.xlsx', '.xls'):
            reasons.append(f"文件类型 {ext} → Openpyxl 专用（合并单元格/多工作表）")
        elif ext in ('.docx', '.pptx', '.doc', '.ppt'):
            reasons.append(f"文件类型 {ext} 适合 Docling")
        elif ext == '.pdf' and sample_text:
            formula_score = sum(len(p.findall(sample_text)) for p in self.FORMULA_PATTERNS)
            cjk_chars = len(self.CJK_PATTERN.findall(sample_text))
            cjk_ratio = cjk_chars / max(len(sample_text), 1)
            if formula_score >= 3:
                reasons.append(f"检测到大量公式 ({formula_score} 处) → MinerU")
            if cjk_ratio > 0.3:
                reasons.append(f"中文密集 (CJK占比 {cjk_ratio:.0%}) → MinerU")
            keyword_hits = sum(1 for kw in self.ACADEMIC_KEYWORDS
                               if kw in sample_text[:3000].lower())
            if keyword_hits >= 2:
                reasons.append(f"学术/财报特征 ({keyword_hits} 个关键词) → MinerU")
            image_hits = sum(1 for kw in self.IMAGE_HEAVY_KEYWORDS
                             if kw in sample_text.lower())
            if image_hits >= 3:
                reasons.append(f"图文密集 ({image_hits} 处图表引用) → MinerU")
            if not reasons:
                reasons.append("普通 PDF → Docling")
        elif ext == '.pdf':
            reasons.append("PDF 文档，默认 MinerU")

        return parser, {
            'file_type': ext,
            'reasons': reasons,
            'recommended': parser
        }


# ============================================================
# 解析器基类
# ============================================================
class BaseParser(ABC):
    """文档解析器基类"""

    @abstractmethod
    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()

    @staticmethod
    def generate_doc_id(file_path: str) -> str:
        stat = os.stat(file_path)
        raw = f"{file_path}_{stat.st_mtime}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    @staticmethod
    def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
        return f"{doc_id}_chunk_{chunk_index:05d}"

    @staticmethod
    def _detect_language(text: str) -> str:
        cjk = len(re.findall(r'[\u4e00-\u9fff]', text))
        eng = len(re.findall(r'[a-zA-Z]', text))
        if cjk > eng * 0.5:
            return "zh"
        elif eng > cjk * 2:
            return "en"
        return "mixed"

    @staticmethod
    def _detect_chunk_type(text: str) -> str:
        if re.search(r'\\begin\{(equation|align|matrix)', text):
            return "formula"
        if re.search(r'\|.*\|.*\|', text) or text.count('\t') > 3:
            return "table"
        if text.startswith("图") or text.startswith("Fig"):
            return "image"
        return "text"

    TABLE_ROW_RE = re.compile(r'^\s*\|.*\|\s*$')
    TABLE_SEP_RE = re.compile(r'^\s*\|[\s:|\-]+\|\s*$')

    def _chunk_table(self, text: str, doc_id: str, chunk_size: int,
                     title: str = "") -> List[ParsedChunk]:
        """
        Markdown 表格按行对齐切块，每块重复表头
        固定字数滑动窗口会把一行数据截成两半（金额与部门分属不同块），检索到也无意义
        """
        lines = [ln.rstrip() for ln in text.split('\n') if ln.strip()]
        if (len(lines) < 3 or not self.TABLE_ROW_RE.match(lines[0])
                or not self.TABLE_SEP_RE.match(lines[1])):
            return []

        header, rows = lines[:2], lines[2:]
        head_len = sum(len(x) for x in header)
        chunks: List[ParsedChunk] = []
        buf: List[str] = []
        size = head_len

        def emit():
            nonlocal buf, size
            body = '\n'.join(header + buf)
            chunks.append(ParsedChunk(
                chunk_id=self.generate_chunk_id(doc_id, len(chunks)),
                doc_id=doc_id,
                content=body.strip(),
                chunk_index=len(chunks),
                title=title,
                chunk_type='table',
                language=self._detect_language(body),
            ))
            buf, size = [], head_len

        for row in rows:
            if buf and size + len(row) > chunk_size:
                emit()
            buf.append(row)
            size += len(row)
        if buf:
            emit()
        return chunks

    def _chunk_text(self, text: str, doc_id: str,
                    chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP,
                    title: str = "") -> List[ParsedChunk]:
        """通用分块：Markdown 表格按行对齐切分，其余走滑动窗口"""
        if not text or not text.strip():
            return []

        table = self._chunk_table(text, doc_id, chunk_size, title)
        if table:
            return table

        if len(text) <= chunk_size:
            return [ParsedChunk(
                chunk_id=self.generate_chunk_id(doc_id, 0),
                doc_id=doc_id,
                content=text.strip(),
                chunk_index=0,
                title=title,
                chunk_type=self._detect_chunk_type(text),
                language=self._detect_language(text)
            )]

        chunks = []
        idx = 0
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(ParsedChunk(
                chunk_id=self.generate_chunk_id(doc_id, idx),
                doc_id=doc_id,
                content=chunk_text.strip(),
                chunk_index=idx,
                title=title,
                chunk_type=self._detect_chunk_type(chunk_text),
                language=self._detect_language(chunk_text)
            ))
            idx += 1
            start += chunk_size - overlap

        return chunks

    # 中文公文章节编号：一、 / （一） / (一)
    CN_SECTION_RE = re.compile(
        r'^\s*(?:[一二三四五六七八九十]+、|（[一二三四五六七八九十]+）|\([一二三四五六七八九十]+\))'
    )

    @classmethod
    def _split_text_sections(cls, text: str) -> List[Dict]:
        """
        纯文本按中文编号分节（一、/（一））
        用于没有 Markdown 标题的纯文本（.txt、docx 导出文本），
        否则按字数硬切会把一条完整规定腰斩在块边界上
        """
        sections: List[Dict] = []
        cur_title, cur_lines = '', []
        for line in (text or '').split('\n'):
            if cls.CN_SECTION_RE.match(line):
                body = '\n'.join(cur_lines).strip()
                if body:
                    sections.append({'title': cur_title, 'content': body})
                cur_title, cur_lines = line.strip(), []
            else:
                cur_lines.append(line)
        body = '\n'.join(cur_lines).strip()
        if body:
            sections.append({'title': cur_title, 'content': body})
        return sections

    @staticmethod
    def _is_section_heading(title: str, content: str) -> bool:
        """
        判断 title 是真章节标题，还是解析器误当标题的正文首句 / 表格首格

        PDF 表格常被解析成 content="[项目] 2025年 2024年"、title="项目" 这种伪标题。
        若把它当章节边界，一张表会被切成几十个十几字的碎片，在向量与 rerank 里
        反而因为短而拿到高分，挤占真正制度条款的位置。
        """
        if not title or len(title) > 40:
            return False
        head = content.lstrip()
        return not (head.startswith(title) or head.startswith(f'[{title}]'))

    def _chunk_by_sections(self, sections: List[Dict], doc_id: str,
                           chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[ParsedChunk]:
        """
        按章节分块：每个 section 是 {"title", "content", "page_number", "type"}

        规则：
        1. 同一章节内相邻段落累积到 chunk_size，不跨章节合并
        2. 伪标题（表格首格/正文首句）不构成边界，继续累积，避免表格碎成几十块
        3. 表格与正文不混块；超长章节按滑动窗口切分
        """
        chunks: List[ParsedChunk] = []
        idx = 0
        group_title, group_type, group_page = '', 'text', -1
        parts: List[str] = []
        size = 0

        def flush():
            nonlocal idx, group_title, group_type, group_page, parts, size
            body = '\n'.join(parts).strip()
            if body:
                for c in self._chunk_text(body, doc_id, chunk_size, overlap, group_title):
                    c.chunk_index = idx
                    c.chunk_id = self.generate_chunk_id(doc_id, idx)
                    c.page_number = group_page
                    if group_type != 'text':
                        c.chunk_type = group_type
                    chunks.append(c)
                    idx += 1
            parts, size = [], 0

        for sec in sections:
            title = (sec.get('title') or '').strip()
            content = (sec.get('content') or '').strip()
            if not content:
                continue
            # 兜底：docling 的 Table 元素 str(elem) 同样是 HTML，这里统一规范化
            if '<table' in content.lower():
                content = html_tables_to_markdown(content)
            sec_type = sec.get('type', 'text')
            page = sec.get('page_number', -1)
            real_heading = self._is_section_heading(title, content)

            is_boundary = bool(parts) and (
                sec_type != group_type or (real_heading and title != group_title)
            )
            if is_boundary or (parts and size + len(content) > chunk_size):
                flush()

            if not parts:
                group_title = title if real_heading else ''
                group_type, group_page = sec_type, page
            elif real_heading and not group_title:
                group_title = title

            parts.append(content)
            size += len(content)

        flush()
        return chunks


# ============================================================
# 1. MinerU 解析器 - 论文/财报/中文密集/公式（云端 API 模式）
# ============================================================
class MinerUParser(BaseParser):
    """
    MinerU 云端 API 解析器
    擅长: 论文、财报、中文密集文档、含大量公式的 PDF
    特点: 调用 mineru.net 云端 API，无需本地模型，数据不出域需切换为自托管
    
    环境变量:
        MINERU_API_KEY: MinerU API 密钥（从 OpenDataLab 获取）
    """

    def __init__(self):
        from config import MINERU_API_KEY
        self.api_key = MINERU_API_KEY
        self.api_base = "https://mineru.net/api/v4"

    def is_available(self) -> bool:
        """检查 API Key 是否配置"""
        return bool(self.api_key)

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """
        调用 MinerU 云端 API 解析 PDF
        
        流程:
        1. 申请文件上传 URL
        2. 上传文件到预签名 URL
        3. 系统自动提交解析任务
        4. 轮询任务状态直到完成
        5. 下载并解压结果包，提取 Markdown
        """
        import time
        import requests
        start = time.time()
        doc_id = self.generate_doc_id(file_path)
        file_hash = self.compute_file_hash(file_path)
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        chunk_size = kwargs.get('chunk_size', 500)
        chunk_overlap = kwargs.get('chunk_overlap', 50)

        try:
            # 步骤 1: 申请上传 URL
            batch_id, upload_url = self._request_upload_url(filename)
            logger.info(f"[MinerU] 申请上传 URL 成功: {filename}, batch_id={batch_id}")

            # 步骤 2: 上传文件（系统会自动提交解析任务）
            self._upload_file(file_path, upload_url)
            logger.info(f"[MinerU] 文件上传成功: {filename}")

            # 步骤 3: 轮询任务状态
            result = self._poll_task(batch_id)
            logger.info(f"[MinerU] 任务完成: batch_id={batch_id}")

            # 步骤 4: 下载并提取 Markdown 内容
            markdown_content = self._extract_markdown_from_zip(result)
            # MinerU 的复杂表格是 HTML <table>，先转成管道表，否则标签会切进 chunk
            markdown_content = html_tables_to_markdown(markdown_content)

            # 按标题分块；无 Markdown 标题时回退到中文编号分节
            sections = self._split_markdown_sections(markdown_content)
            if len(sections) <= 1:
                cn_sections = self._split_text_sections(markdown_content)
                if len(cn_sections) > 1:
                    sections = cn_sections
            chunks = self._chunk_by_sections(sections, doc_id, chunk_size, chunk_overlap)

            if not chunks:
                chunks = self._chunk_text(markdown_content or "", doc_id, chunk_size, chunk_overlap)

            return ParsedDocument(
                doc_id=doc_id, filename=filename, file_path=file_path,
                file_type="pdf", file_size=file_size, file_hash=file_hash,
                parser_used="mineru", parse_time=time.time() - start,
                chunks=chunks,
                metadata={'batch_id': batch_id, 'api_mode': 'cloud'},
                status="success"
            )
        except Exception as e:
            logger.error(f"MinerU 解析失败 [{filename}]: {e}")
            return ParsedDocument(
                doc_id=doc_id, filename=filename, file_path=file_path,
                file_type="pdf", file_size=file_size, file_hash=file_hash,
                parser_used="mineru", parse_time=time.time() - start,
                status="failed", error_msg=str(e)
            )

    def _request_upload_url(self, filename: str) -> Tuple[str, str]:
        """申请文件上传 URL，系统会自动提交解析任务"""
        import requests
        url = f"{self.api_base}/file-urls/batch"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data_id = "rag_" + hashlib.md5(filename.encode()).hexdigest()[:8]
        payload = {
            "files": [
                {"name": filename, "data_id": data_id}
            ],
            "model_version": "pipeline"
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if data.get('code') != 0:
            raise Exception(f"申请上传 URL 失败: {data.get('msg', '未知错误')}")

        batch_id = data['data']['batch_id']
        file_urls = data['data']['file_urls']

        if not file_urls:
            raise Exception("未获取到上传 URL")

        upload_url = file_urls[0]

        return batch_id, upload_url

    def _upload_file(self, file_path: str, upload_url: str):
        """上传文件到预签名 URL"""
        import requests
        with open(file_path, 'rb') as f:
            resp = requests.put(upload_url, data=f, timeout=120)
        resp.raise_for_status()

    def _poll_task(self, batch_id: str, max_wait: int = 600) -> Dict:
        """轮询批量任务状态，返回解析结果"""
        import requests
        import time
        url = f"{self.api_base}/extract-results/batch/{batch_id}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        start = time.time()
        while time.time() - start < max_wait:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if data.get('code') != 0:
                raise Exception(f"查询任务失败: {data.get('msg', '未知错误')}")

            # 批量结果在 data.extract_result 数组中
            extract_results = data.get('data', {}).get('extract_result', [])
            if not extract_results:
                # 还未生成结果，继续等待
                time.sleep(5)
                continue

            # 取第一个文件的结果
            result = extract_results[0]
            state = result.get('state')

            if state == 'done':
                return result
            elif state == 'failed':
                raise Exception(f"任务失败: {result.get('err_msg', '未知错误')}")

            # 等待 5 秒后重试
            time.sleep(5)

        raise Exception(f"任务超时（等待 {max_wait} 秒）")

    def _extract_markdown_from_zip(self, result: Dict) -> str:
        """从结果中下载 zip 包并提取 Markdown 内容"""
        import requests
        import zipfile
        import io
        
        full_zip_url = result.get('full_zip_url')
        if not full_zip_url:
            raise Exception("未获取到结果下载链接")
        
        # 下载 zip 包
        resp = requests.get(full_zip_url, timeout=120)
        resp.raise_for_status()
        
        # 解压并提取 markdown
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # 查找 .md 文件
            md_files = [f for f in zf.namelist() if f.endswith('.md')]
            if not md_files:
                raise Exception("zip 包中未找到 Markdown 文件")
            
            # 读取第一个 markdown 文件
            with zf.open(md_files[0]) as f:
                return f.read().decode('utf-8')

    @staticmethod
    def _split_markdown_sections(md_text: str) -> List[Dict]:
        """将 Markdown 按标题拆分为章节"""
        sections = []
        current_title = ""
        current_content = ""
        for line in (md_text or "").split('\n'):
            m = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if m:
                if current_content.strip():
                    sections.append({'title': current_title, 'content': current_content.strip()})
                current_title = m.group(2).strip()
                current_content = ""
            else:
                current_content += line + "\n"
        if current_content.strip():
            sections.append({'title': current_title, 'content': current_content.strip()})
        return sections


# ============================================================
# 2. Docling 解析器 - 合同/docx/pptx/英文文档
# ============================================================
class DoclingParser(BaseParser):
    """
    Docling 解析器
    擅长: 普通合同、英文文档、docx/pptx 混合文档
    特点: 多格式支持, 优秀表格提取, 文档结构树解析, 轻量部署
    """

    def __init__(self):
        self._converter = None

    def is_available(self) -> bool:
        try:
            from docling.document_converter import DocumentConverter
            return True
        except ImportError:
            return False

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        import time
        start = time.time()
        doc_id = self.generate_doc_id(file_path)
        file_hash = self.compute_file_hash(file_path)
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        chunk_size = kwargs.get('chunk_size', 500)
        chunk_overlap = kwargs.get('chunk_overlap', 50)

        try:
            if self._converter is None:
                from docling.document_converter import DocumentConverter
                self._converter = DocumentConverter()

            result = self._converter.convert(file_path)
            document = result.document
            md_content = document.export_to_markdown()

            # 按标题分块；无 Markdown 标题时回退到中文编号分节
            sections = MinerUParser._split_markdown_sections(md_content)
            if len(sections) <= 1:
                cn_sections = self._split_text_sections(md_content)
                if len(cn_sections) > 1:
                    sections = cn_sections
            chunks = self._chunk_by_sections(sections, doc_id, chunk_size, chunk_overlap)

            if not chunks:
                chunks = self._chunk_text(md_content or "", doc_id, chunk_size, chunk_overlap)

            return ParsedDocument(
                doc_id=doc_id, filename=filename, file_path=file_path,
                file_type=ext.lstrip('.'), file_size=file_size, file_hash=file_hash,
                parser_used="docling", parse_time=time.time() - start,
                chunks=chunks,
                metadata={'page_count': len(document.pages) if hasattr(document, 'pages') else -1},
                status="success"
            )
        except Exception as e:
            logger.error(f"Docling 解析失败 [{filename}]: {e}")
            return ParsedDocument(
                doc_id=doc_id, filename=filename, file_path=file_path,
                file_type=ext.lstrip('.'), file_size=file_size, file_hash=file_hash,
                parser_used="docling", parse_time=time.time() - start,
                status="failed", error_msg=str(e)
            )


# ============================================================
# 3. Unstructured 解析器 - Excel/HTML/邮件/通用办公
# ============================================================
class UnstructuredParser(BaseParser):
    """
    Unstructured 解析器
    擅长: PDF、Word、Excel、HTML、邮件等通用办公文档
    特点: 格式支持最广, 内置多种 element 类型识别, 企业级 ETL 管线
    安装: pip install "unstructured[all-docs]"
    """

    def is_available(self) -> bool:
        try:
            from unstructured.partition.auto import partition
            return True
        except ImportError:
            return False

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        import time
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
        start = time.time()
        doc_id = self.generate_doc_id(file_path)
        file_hash = self.compute_file_hash(file_path)
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        chunk_size = kwargs.get('chunk_size', 500)
        chunk_overlap = kwargs.get('chunk_overlap', 50)
        strategy = kwargs.get('strategy', 'auto')  # auto / hi_res / fast / ocr_only
        timeout_seconds = kwargs.get('timeout', 120)  # 默认 120 秒超时

        def _do_parse():
            from unstructured.partition.auto import partition
            return partition(
                filename=file_path,
                strategy=strategy,
                languages=["chi_sim", "eng"],
            )

        try:
            # 带超时保护的解析
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_do_parse)
                elements = future.result(timeout=timeout_seconds)

            # 将 elements 转为结构化 sections
            sections = []
            current_title = ""
            current_content = ""

            for elem in elements:
                elem_type = type(elem).__name__

                if elem_type in ('Title', 'SectionHeader'):
                    # 遇到标题，保存之前的内容
                    if current_content.strip():
                        sections.append({
                            'title': current_title,
                            'content': current_content.strip(),
                            'page_number': getattr(elem.metadata, 'page_number', -1)
                            if hasattr(elem, 'metadata') else -1
                        })
                    current_title = str(elem)
                    current_content = ""
                elif elem_type == 'Table':
                    # 表格单独成块
                    if current_content.strip():
                        sections.append({'title': current_title, 'content': current_content.strip()})
                        current_content = ""
                    sections.append({
                        'title': current_title,
                        'content': str(elem),
                        'type': 'table',
                        'page_number': getattr(elem.metadata, 'page_number', -1)
                        if hasattr(elem, 'metadata') else -1
                    })
                else:
                    current_content += str(elem) + "\n"

            # 最后一段
            if current_content.strip():
                sections.append({'title': current_title, 'content': current_content.strip()})

            # 解析器没识别出任何标题时（常见于 .txt），回退到中文编号分节
            if len(sections) <= 1:
                all_text = "\n".join(str(e) for e in elements)
                cn_sections = self._split_text_sections(all_text)
                if len(cn_sections) > 1:
                    sections = cn_sections

            chunks = self._chunk_by_sections(sections, doc_id, chunk_size, chunk_overlap)

            if not chunks:
                all_text = "\n".join(str(e) for e in elements)
                chunks = self._chunk_text(all_text, doc_id, chunk_size, chunk_overlap)

            return ParsedDocument(
                doc_id=doc_id, filename=filename, file_path=file_path,
                file_type=ext.lstrip('.'), file_size=file_size, file_hash=file_hash,
                parser_used="unstructured", parse_time=time.time() - start,
                chunks=chunks,
                metadata={'element_count': len(elements), 'strategy': strategy},
                status="success"
            )
        except FuturesTimeoutError:
            error_msg = f"解析超时（>{timeout_seconds}秒），已自动终止"
            logger.error(f"Unstructured 解析超时 [{filename}]: {error_msg}")
            return ParsedDocument(
                doc_id=doc_id, filename=filename, file_path=file_path,
                file_type=ext.lstrip('.'), file_size=file_size, file_hash=file_hash,
                parser_used="unstructured", parse_time=time.time() - start,
                status="failed", error_msg=error_msg
            )
        except Exception as e:
            logger.error(f"Unstructured 解析失败 [{filename}]: {e}")
            return ParsedDocument(
                doc_id=doc_id, filename=filename, file_path=file_path,
                file_type=ext.lstrip('.'), file_size=file_size, file_hash=file_hash,
                parser_used="unstructured", parse_time=time.time() - start,
                status="failed", error_msg=str(e)
            )


# ============================================================
# 4. Openpyxl 解析器 - Excel 专用（合并单元格/多工作表/表头保护）
# ============================================================
class OpenpyxlParser(BaseParser):
    """
    Excel 专用解析器：openpyxl 直读单元格，不经过 docling 的 markdown 转换。

    为什么不用 docling 解析 Excel（2026-10-24 实测工资表）：
    docling 会把合并单元格的标题复制到每一列（一行跨 9 列的表头膨胀成
    2000+ 字符），再经 500 字切块后，18 个块全是标题残片，员工数据
    一行都没进块。这不是切块参数问题，是 markdown 转换层的信息畸变，
    所以在这里直接读单元格，机制型绕开。

    处理规则：
    - 合并单元格：左上角值填充到整个合并区域（openpyxl 只给左上角一个值）
    - 装饰标题行（合并大标题）：剥离进 metadata，不混进数据表
    - 分块：按行累加，超过 chunk_size 封块，每块都带表头（与 CSV 机制同源）
    """

    def is_available(self) -> bool:
        try:
            import openpyxl  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _fmt_cell(cell) -> str:
        """单元格 → 字符串。按 number_format 渲染，与 Excel 显示值对齐。

        为什么不能直接 str(value)：xlsx 里占比列是公式 float，
        Excel 显示 20.59%，str() 却输出 0.2059075939248601 ——
        全精度尾巴既让块不可读，16 位连续小数还会撞上脱敏规则
        的银行卡号检测（实测 2026-10-24，占比列整列被打码）。
        """
        value = cell.value
        if value is None:
            return ""
        if isinstance(value, datetime):
            fmt = cell.number_format or ""
            if "yy" in fmt or "dd" in fmt:
                return value.strftime("%Y-%m-%d")
            return value.strftime("%Y-%m-%d")
        if isinstance(value, float):
            fmt = cell.number_format or "General"
            if "%" in fmt:
                head = fmt.split("%")[0]
                decimals = len(m.group(1)) if (m := re.search(r"\.(0+)", head)) else 0
                return f"{value * 100:.{decimals}f}%"
            m = re.search(r"\.(0+)", fmt)
            if m:  # 明确的小数位数（如 0.00）
                return f"{value:.{len(m.group(1))}f}"
            if value == int(value) and abs(value) < 1e15:
                return str(int(value))
            # General 浮点：截到 6 位小数再去尾零，杜绝全精度尾巴
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value).strip()

    @staticmethod
    def _sheet_to_rows(ws) -> Tuple[List[List[str]], List[str]]:
        """工作表 → (数据行二维数组, 装饰/信息行列表)

        列头行的判定：从上往下第一行「非空格数 ≥ 2 且无重复值」的行。
        合并单元格被左上角值填充后，同一值会在一行内出现多次——
        装饰大标题（整行单值）和「单位/日期/制表」信息行（多个合并区域）
        都带这个特征，全部剥出表体，避免膨胀表头或错选表头。
        """
        from collections import Counter

        merged_map = {}
        for rng in ws.merged_cells.ranges:
            top_left = ws.cell(rng.min_row, rng.min_col)  # 存 cell 本身，保留 number_format
            # 左上角值填充到合并区域内所有格子
            for r in range(rng.min_row, rng.max_row + 1):
                for c in range(rng.min_col, rng.max_col + 1):
                    merged_map[(r, c)] = top_left

        max_col = ws.max_column
        titles, body = [], []
        header_found = False
        for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=max_col):
            cells = []
            for cell in row:
                src = merged_map.get((cell.row, cell.column), cell)
                cells.append(OpenpyxlParser._fmt_cell(src))
            non_empty = [c for c in cells if c]
            if not non_empty:
                continue  # 空行直接跳过
            if not header_found:
                # 列头行之前：带合并填充特征（整行单值或任一值重复≥2）的行 = 装饰/信息行
                dup = max(Counter(non_empty).values())
                if len(non_empty) == 1 or dup >= 2:
                    titles.append(non_empty[0] if len(non_empty) == 1
                                  else "；".join(dict.fromkeys(non_empty)))
                    continue
                header_found = True
            body.append(cells)
        return body, titles

    @staticmethod
    def _chunk_rows(header: List[str], rows: List[List[str]], doc_id: str,
                    sheet_name: str, chunk_size: int, chunk_overlap: int) -> List["ParsedChunk"]:
        """表头 + 数据行 → markdown 管道表块，每块都带表头"""
        def render(rows_part):
            lines = ["| " + " | ".join(header) + " |",
                     "|" + "|".join(["---"] * len(header)) + "|"]
            lines += ["| " + " | ".join(r) + " |" for r in rows_part]
            return "\n".join(lines)

        chunks = []
        cur, cur_len = [], 0
        for row in rows:
            row_len = len("| " + " | ".join(row) + " |")
            if cur and cur_len + row_len > chunk_size:
                chunks.append((cur, cur_len))
                cur, cur_len = [], 0
            cur.append(row)
            cur_len += row_len
        if cur:
            chunks.append((cur, cur_len))

        out = []
        for i, (rows_part, _) in enumerate(chunks):
            content = render(rows_part)
            out.append(ParsedChunk(
                chunk_id=BaseParser.generate_chunk_id(doc_id, i),
                doc_id=doc_id, chunk_index=i, content=content,
                chunk_type="table",
                title=sheet_name,
                metadata={"sheet": sheet_name, "rows": len(rows_part),
                          "with_header": True},
            ))
        return out

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        import time
        start = time.time()
        doc_id = self.generate_doc_id(file_path)
        file_hash = self.compute_file_hash(file_path)
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        chunk_size = kwargs.get('chunk_size', 500)

        try:
            import openpyxl
            # data_only=True：公式单元格取上次保存时的缓存值
            wb = openpyxl.load_workbook(file_path, data_only=True)

            chunks = []
            sheet_notes = []
            for ws in wb.worksheets:
                body, titles = self._sheet_to_rows(ws)
                if not body:
                    continue
                if len(body) == 1:
                    # 只剩一行数据也保留，表头即它自己
                    header, data = body[0], [body[0]]
                else:
                    header, data = body[0], body[1:]
                # 表头去重相邻空列（docling 式的全空列在这里直接过滤）
                chunks += self._chunk_rows(header, data, doc_id, ws.title,
                                           chunk_size, kwargs.get('chunk_overlap', 50))
                note = f"{ws.title}({len(data)}行)"
                if titles:
                    note += f" 标题:{titles[0][:30]}"
                sheet_notes.append(note)

            if not chunks:
                raise ValueError("所有工作表均无数据行")

            return ParsedDocument(
                doc_id=doc_id, filename=filename, file_path=file_path,
                file_type=ext.lstrip('.'), file_size=file_size, file_hash=file_hash,
                parser_used="openpyxl", parse_time=time.time() - start,
                chunks=chunks,
                metadata={"sheets": sheet_notes,
                          "sheet_count": len(sheet_notes)},
                status="success"
            )
        except Exception as e:
            logger.error(f"Openpyxl 解析失败 [{filename}]: {e}")
            return ParsedDocument(
                doc_id=doc_id, filename=filename, file_path=file_path,
                file_type=ext.lstrip('.'), file_size=file_size, file_hash=file_hash,
                parser_used="openpyxl", parse_time=time.time() - start,
                status="failed", error_msg=str(e)
            )


# ============================================================
# 三管线智能调度器
# ============================================================
class MultiPipelineDispatcher:
    """
    三管线智能调度器

    自动路由:
    ┌────────────────────────────────────────────────────┐
    │  文件输入 → 分类器 → 选择最优解析器 → 解析 → 分块    │
    │                                                    │
    │  论文/公式/中文密集/图文/扫描件 → MinerU              │
    │  合同/docx/pptx                → Docling            │
    │  Excel/HTML/邮件               → Unstructured        │
    │                                                    │
    │  失败时自动回退到备选解析器                            │
    └────────────────────────────────────────────────────┘

    使用:
        dispatcher = MultiPipelineDispatcher()
        result = dispatcher.parse("paper.pdf")                 # 自动选择
        result = dispatcher.parse("paper.pdf", parser="mineru") # 手动指定
    """

    # 回退顺序：某个解析器失败时尝试的备选（构造时过滤掉 PARSER_DISABLED）
    FALLBACK_ORDER = {
        k: [p for p in v if p not in PARSER_DISABLED]
        for k, v in {
            'mineru':       ['docling', 'unstructured'],
            'docling':      ['unstructured', 'mineru'],
            'unstructured': ['docling', 'mineru'],
            'openpyxl':     ['docling'],
        }.items()
        if k not in PARSER_DISABLED
    }

    def __init__(self, default_chunk_size: int = CHUNK_SIZE,
                 default_chunk_overlap: int = CHUNK_OVERLAP):
        self.classifier = DocumentClassifier()
        self.default_chunk_size = default_chunk_size
        self.default_chunk_overlap = default_chunk_overlap
        self._parsers: Dict[str, BaseParser] = {}

    def _get_parser(self, name: str) -> BaseParser:
        if name not in self._parsers:
            if name == "mineru":
                self._parsers[name] = MinerUParser()
            elif name == "docling":
                self._parsers[name] = DoclingParser()
            elif name == "unstructured":
                self._parsers[name] = UnstructuredParser()
            elif name == "openpyxl":
                self._parsers[name] = OpenpyxlParser()
            else:
                raise ValueError(f"未知解析器: {name}")
        return self._parsers[name]

    def parse(self, file_path: str, parser: str = None,
              sample_text: str = "", **kwargs) -> ParsedDocument:
        """
        解析文档（带自动回退）
        :param file_path: 文件路径
        :param parser: 指定解析器 (None=自动)
        :param sample_text: 采样文本（用于自动分类）
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        kwargs.setdefault('chunk_size', self.default_chunk_size)
        kwargs.setdefault('chunk_overlap', self.default_chunk_overlap)

        # 显式指定的解析器也要服从 PARSER_DISABLED。
        # 否则 pipeline 的路由规则（PDF → mineru）会绕过环境变量：只要
        # MINERU_API_KEY 还在环境里，is_available() 就是 True，文件会真被传到
        # mineru.net 云端——与「默认全本地」的配置承诺相反。
        if parser is not None and parser in PARSER_DISABLED:
            logger.warning(
                f"解析器 {parser} 已在 PARSER_DISABLED 中，改为自动选择: "
                f"{os.path.basename(file_path)}"
            )
            parser = None

        # 自动选择
        if parser is None:
            parser, info = self.classifier.classify_with_info(file_path, sample_text)
            logger.info(f"文档分类: {info}")

        # 尝试解析 + 回退
        tried = set()
        current = parser

        while current and current not in tried:
            tried.add(current)
            parser_instance = self._get_parser(current)

            if not parser_instance.is_available():
                logger.warning(f"{current} 不可用，尝试回退...")
                current = self._next_fallback(parser, tried)
                continue

            print(f"[MultiPipeline] 使用 {current} 解析: {os.path.basename(file_path)}")
            result = parser_instance.parse(file_path, **kwargs)

            if result.status == "success" and result.chunks:
                return result

            # 解析失败，尝试回退
            if result.status == "failed":
                logger.warning(f"{current} 解析失败: {result.error_msg}，尝试回退...")
                current = self._next_fallback(parser, tried)
                if current:
                    print(f"[MultiPipeline] 回退到 {current}")
                continue

            return result

        # 所有解析器都失败
        return ParsedDocument(
            doc_id=BaseParser.generate_doc_id(file_path),
            filename=os.path.basename(file_path),
            file_path=file_path,
            file_type=os.path.splitext(file_path)[1].lstrip('.'),
            file_size=os.path.getsize(file_path),
            file_hash=BaseParser.compute_file_hash(file_path),
            parser_used=parser or "auto",
            parse_time=0,
            status="failed",
            error_msg=f"所有解析器均失败，已尝试: {', '.join(tried)}"
        )

    def _next_fallback(self, original_parser: str, tried: set) -> Optional[str]:
        """获取下一个回退解析器"""
        fallbacks = self.FALLBACK_ORDER.get(original_parser)
        if fallbacks is None:
            # 起点解析器被禁用时，它的 key 已从 FALLBACK_ORDER 里被过滤掉，
            # 直接 .get(..., []) 会拿不到任何备选，导致「无回退可用」。
            fallbacks = [p for p in ('docling', 'mineru', 'unstructured')
                         if p not in PARSER_DISABLED]
        for fb in fallbacks:
            if fb not in tried:
                return fb
        return None

    def parse_batch(self, file_paths: List[str], parser: str = None,
                    progress_callback=None, **kwargs) -> List[ParsedDocument]:
        """批量解析"""
        results = []
        total = len(file_paths)
        for i, fp in enumerate(file_paths):
            if progress_callback:
                progress_callback(i + 1, total, os.path.basename(fp))
            try:
                doc = self.parse(fp, parser=parser, **kwargs)
                results.append(doc)
                print(f"[MultiPipeline] ({i+1}/{total}) {doc.filename} → "
                      f"{len(doc.chunks)} chunks via {doc.parser_used}, {doc.parse_time:.1f}s")
            except Exception as e:
                logger.error(f"解析失败 [{fp}]: {e}")
                results.append(ParsedDocument(
                    doc_id="error", filename=os.path.basename(fp),
                    file_path=fp, file_type=os.path.splitext(fp)[1].lstrip('.'),
                    file_size=0, file_hash="", parser_used=parser or "auto",
                    parse_time=0, status="failed", error_msg=str(e)
                ))
        return results

    def get_parser_status(self) -> Dict:
        """获取所有解析器的可用状态"""
        status = {}
        for name in [n for n in ["mineru", "docling", "unstructured", "openpyxl"] if n not in PARSER_DISABLED]:
            try:
                p = self._get_parser(name)
                status[name] = {
                    'available': p.is_available(),
                    'class': p.__class__.__name__,
                    'description': {
                        'mineru': '论文/财报/中文密集/公式/图文/扫描件 PDF',
                        'docling': '合同/docx/pptx/英文文档',
                        'unstructured': 'HTML/邮件/通用办公',
                        'openpyxl': 'Excel 专用（合并单元格/多工作表/表头保护）',
                    }.get(name, '')
                }
            except Exception:
                status[name] = {'available': False, 'class': 'Error', 'description': ''}
        return status



