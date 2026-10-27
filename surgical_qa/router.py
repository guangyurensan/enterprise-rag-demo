# -*- coding: utf-8 -*-
"""
企业级知识库 - 路由调度器

顺序匹配、命中即终止，优先级从上到下
第一层：后缀过滤
第二层：PDF 内部探测（原生/扫描）+ 业务标签
"""

import os
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from pipeline_config import ROUTING_RULES, PARSER_MAP
from prescreen import PrescreenResult

logger = logging.getLogger(__name__)


@dataclass
class RoutingResult:
    """路由结果（解耦：只存 file_tag，解析器通过 resolve_parser() 查映射表）"""
    file_tag: str                 # 文件标签（对应 PARSER_MAP 的 key）
    rule_name: str                # 命中的规则名
    confidence: float = 1.0       # 路由置信度
    reasons: List[str] = field(default_factory=list)  # 路由原因
    metadata: Dict = field(default_factory=dict)

    def resolve_parser(self) -> str:
        """通过 PARSER_MAP 映射表查解析器名"""
        return PARSER_MAP.get(self.file_tag, "unstructured")


class RouterDispatcher:
    """
    路由调度器

    规则按优先级从高到低排列，顺序匹配，命中即终止
    """

    def __init__(self, rules: List[Dict] = None):
        self.rules = rules or ROUTING_RULES
        # 按优先级降序排列
        self.rules.sort(key=lambda r: r.get("priority", 0), reverse=True)

    def route(self, prescreen: PrescreenResult,
              sample_text: str = "") -> RoutingResult:
        """
        根据前置粗筛结果和采样文本，路由到目标解析器

        :param prescreen: 前置粗筛结果
        :param sample_text: 文件采样文本（前 3000 字）
        :return: RoutingResult
        """
        file_ext = prescreen.file_ext
        filename = prescreen.filename.lower()

        for rule in self.rules:
            match_result = self._match_rule(
                rule, file_ext, filename, sample_text, prescreen
            )
            if match_result:
                reasons = [f"命中规则: {rule['name']}"] + match_result
                file_tag = rule.get("parser_key", "unstructured")
                result = RoutingResult(
                    file_tag=file_tag,
                    rule_name=rule["name"],
                    confidence=rule.get("priority", 50) / 100.0,
                    reasons=reasons,
                )
                logger.info(
                    f"[Router] {filename} → tag={file_tag} "
                    f"(规则: {rule['name']})"
                )
                return result

        # 兜底：所有规则都没命中
        result = RoutingResult(
            file_tag="unstructured",
            rule_name="兜底",
            confidence=0.1,
            reasons=["未命中任何规则，使用兜底解析器"],
        )
        logger.warning(f"[Router] {filename} → unstructured (兜底)")
        return result

    def _match_rule(self, rule: Dict, file_ext: str, filename: str,
                    sample_text: str, prescreen: PrescreenResult) -> Optional[List[str]]:
        """
        匹配单条规则，返回匹配原因列表，None 表示不匹配
        """
        match_cfg = rule.get("match", {})
        reasons = []

        # 1. 文件类型匹配
        file_type_cfg = match_cfg.get("file_type")
        if file_type_cfg:
            if isinstance(file_type_cfg, list):
                if file_ext not in file_type_cfg:
                    return None
            elif file_ext != file_type_cfg:
                return None
            reasons.append(f"文件类型 {file_ext} 匹配")

        # 2. 文件名关键词匹配
        fn_keywords = match_cfg.get("filename_keywords", [])
        if fn_keywords:
            fn_hits = [kw for kw in fn_keywords if kw.lower() in filename]
            if not fn_hits:
                return None
            reasons.append(f"文件名关键词: {fn_hits}")

        # 3. 内容关键词匹配
        ct_keywords = match_cfg.get("content_keywords", [])
        if ct_keywords and sample_text:
            ct_hits = [kw for kw in ct_keywords if kw in sample_text[:3000]]
            if len(ct_hits) < 1:
                return None
            reasons.append(f"内容关键词: {ct_hits[:3]}")

        # 4. 内容正则模式匹配
        ct_patterns = match_cfg.get("content_patterns", [])
        if ct_patterns and sample_text:
            pattern_hits = sum(
                len(re.findall(p, sample_text[:5000]))
                for p in ct_patterns
            )
            threshold = match_cfg.get("pattern_threshold", 1)
            if pattern_hits < threshold:
                return None
            reasons.append(f"模式匹配: {pattern_hits} 处 (阈值: {threshold})")

        # 5. 扫描件 PDF 检测
        if match_cfg.get("is_scanned"):
            if not prescreen.is_scanned_pdf:
                return None
            reasons.append("扫描件 PDF")

        # 6. 图片型 PDF 检测
        if match_cfg.get("is_image_heavy"):
            if not prescreen.is_image_heavy:
                return None
            reasons.append("图片型 PDF")

        # 7. 复杂 Excel 检测
        if match_cfg.get("is_complex_excel"):
            if not prescreen.is_complex_excel:
                return None
            reasons.append("复杂 Excel (多sheet/大表格)")

        return reasons if reasons else None

    def get_route_info(self, prescreen: PrescreenResult,
                       sample_text: str = "") -> Dict:
        """获取路由详细信息（用于调试/日志）"""
        result = self.route(prescreen, sample_text)
        return {
            "filename": prescreen.filename,
            "file_tag": result.file_tag,
            "parser": result.resolve_parser(),
            "rule": result.rule_name,
            "confidence": result.confidence,
            "reasons": result.reasons,
            "all_rules": [
                {"name": r["name"], "parser_key": r.get("parser_key", ""), "priority": r["priority"]}
                for r in self.rules
            ],
        }
