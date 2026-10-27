# -*- coding: utf-8 -*-
"""
企业级知识库 - 安全防护模块

核心职责：
1. Prompt 注入检测（规则引擎，无需额外模型）
2. 上下文脱敏（PII 正则替换）
3. 输入边界隔离

设计原则：
- 骨架层一期固化：检测函数签名稳定，后续只增强规则库
- 策略层可插拔：脱敏正则可配置，检测阈值可调
"""

import re
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


# ============================================================
# Prompt 注入检测
# ============================================================

# 注入模式规则库（骨架层一期固化，后续只增不减）
INJECTION_PATTERNS = [
    # 1. 忽略前指令
    (r"(?i)ignore\s+(all\s+)?previous\s+instructions", "ignore_previous"),
    (r"(?i)忽略(所有|上面|之前)(的)?(指令|指示|提示)", "ignore_previous"),
    (r"(?i)forget\s+(all\s+)?(previous|above)\s+(instructions|context)", "ignore_previous"),

    # 2. 系统提示词泄露
    (r"(?i)(show|print|output|display|reveal)\s+(me\s+)?(the\s+)?system\s+prompt", "leak_system"),
    (r"(?i)(输出|显示|打印)(系统|system)\s*(提示词|prompt)", "leak_system"),
    (r"(?i)what\s+(is|are)\s+your\s+(instructions|system\s+prompt|rules)", "leak_system"),

    # 3. 角色劫持
    (r"(?i)you\s+are\s+now\s+(a|an)\s+", "role_hijack"),
    (r"(?i)(act|pretend|behave)\s+(as|like)\s+(a|an|if)", "role_hijack"),
    (r"(?i)(你现在|你从现在开始)(是|扮演|假装)", "role_hijack"),

    # 4. 分隔符注入
    (r"---+\s*system\s*:", "separator_inject"),
    (r"```\s*system\s*:", "separator_inject"),
    (r"\[INST\]|\[/INST\]|<\|im_start\|>", "separator_inject"),

    # 5. 编码绕过尝试
    (r"(?i)(base64|hex|unicode|rot13)\s*(decode|encode)", "encoding_bypass"),
    (r"(?i)(用|使用)\s*(base64|hex|unicode)\s*(解码|编码|解密)", "encoding_bypass"),

    # 6. 输出格式化攻击（试图让模型输出结构化数据）
    (r"(?i)output\s+in\s+JSON\s+format\s+including\s+system", "data_exfil"),
    (r"(?i)repeat\s+the\s+above\s+from\s+the\s+beginning", "repeat_attack"),
]

# 编译正则（性能优化）
_COMPILED_PATTERNS = [(re.compile(pattern), label) for pattern, label in INJECTION_PATTERNS]


class InjectionDetector:
    """Prompt 注入检测器（规则引擎）"""

    def __init__(self, enabled: bool = True, threshold: float = 0.5):
        """
        :param enabled: 是否启用检测
        :param threshold: 触发阈值（命中规则数 / 总规则数），超过则判定为注入
        """
        self.enabled = enabled
        self.threshold = threshold

    def detect(self, text: str) -> Tuple[bool, List[str], float]:
        """
        检测文本是否包含注入攻击

        :param text: 待检测文本（用户 query）
        :return: (is_injection, triggered_labels, confidence)
        """
        if not self.enabled or not text:
            return False, [], 0.0

        triggered = []
        for pattern, label in _COMPILED_PATTERNS:
            if pattern.search(text):
                triggered.append(label)

        if not triggered:
            return False, [], 0.0

        # 置信度：命中任意一条即视为注入
        # 多条命中时置信度更高
        confidence = min(1.0, 0.6 + 0.1 * (len(triggered) - 1))

        is_injection = True  # 命中任意一条即告警

        if is_injection:
            logger.warning(
                f"[Security] Prompt injection detected! "
                f"labels={triggered}, confidence={confidence:.2f}, "
                f"text_preview='{text[:80]}...'"
            )

        return is_injection, triggered, confidence

    def sanitize(self, text: str) -> str:
        """
        清洗 query：移除可疑的控制指令片段，保留正常查询部分
        如果检测到注入，返回清洗后的安全版本；否则原样返回
        """
        is_injection, labels, _ = self.detect(text)

        if not is_injection:
            return text

        # 尝试移除可疑片段
        cleaned = text
        for pattern, label in _COMPILED_PATTERNS:
            cleaned = pattern.sub('', cleaned)

        # 清理多余空白
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        if not cleaned:
            # 整个 query 都是注入，返回空
            return ""

        logger.info(f"[Security] Query sanitized: '{text[:50]}' -> '{cleaned[:50]}'")
        return cleaned


# ============================================================
# 上下文脱敏（PII 正则替换）
# ============================================================

# PII 正则模式（单一数据源，供脱敏和审计共用）
PII_PATTERNS = {
    "phone_cn": re.compile(r'(?<!\d)1[3-9]\d{9}(?!\d)'),
    "id_card_cn": re.compile(r'(?<!\d)\d{17}[\dXx](?!\d)'),
    "email": re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
    # 银行卡：16~19 位纯数字。排除「前一个字符是小数点」的情况——
    # 那是表格里全精度浮点的小数部分（如 0.2059075939248601），
    # 银行卡号前不会出现小数点，此收紧对真实卡号零损失（2026-10-24 表格数据实测误伤后收紧）
    "bank_card": re.compile(r'(?<![.\d])\d{16,19}(?!\d)'),
    "ip_address": re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
}

# 脱敏规则（骨架层一期固化，后续可扩展）
DESENSITIZATION_RULES = [
    {"name": "phone_cn",    "pattern": PII_PATTERNS["phone_cn"],    "replacement": "***手机号***"},
    {"name": "id_card_cn",  "pattern": PII_PATTERNS["id_card_cn"],  "replacement": "***身份证号***"},
    {"name": "email",       "pattern": PII_PATTERNS["email"],       "replacement": "***邮箱***"},
    {"name": "bank_card",   "pattern": PII_PATTERNS["bank_card"],   "replacement": "***银行卡号***"},
    {"name": "ip_address",  "pattern": PII_PATTERNS["ip_address"],  "replacement": "***IP地址***"},
]


class ContextDesensitizer:
    """上下文脱敏器"""

    def __init__(self, enabled: bool = True, rules: List[dict] = None):
        """
        :param enabled: 是否启用脱敏
        :param rules: 自定义规则列表，None 则使用默认规则
        """
        self.enabled = enabled
        self.rules = rules if rules is not None else DESENSITIZATION_RULES

    def desensitize(self, text: str) -> Tuple[str, List[str]]:
        """
        对文本进行脱敏处理

        :param text: 待脱敏文本
        :return: (desensitized_text, triggered_rules)
        """
        if not self.enabled or not text:
            return text, []

        triggered = []
        result = text

        for rule in self.rules:
            pattern = rule["pattern"]
            replacement = rule["replacement"]
            name = rule["name"]

            matches = pattern.findall(result)
            if matches:
                triggered.append(name)
                result = pattern.sub(replacement, result)

        if triggered:
            logger.info(f"[Security] Desensitized: rules={triggered}")

        return result, triggered

    def desensitize_records(self, results: List[dict]) -> List[dict]:
        """
        对检索结果中的 content 字段进行批量脱敏

        :param results: 检索结果列表
        :return: 脱敏后的结果列表（原地修改）
        """
        if not self.enabled:
            return results

        for item in results:
            content = item.get('content', '')
            if content:
                item['content'], _ = self.desensitize(content)

            # record 中的 answer 也脱敏
            record = item.get('record', {})
            if 'answer' in record:
                record['answer'], _ = self.desensitize(record['answer'])

        return results


# ============================================================
# 统一安全网关
# ============================================================
class SecurityGateway:
    """
    安全网关 — 统一入口

    使用方式：
        gateway = SecurityGateway()

        # 检索前：检测 query 注入
        is_safe, clean_query = gateway.check_query(user_query)
        if not is_safe:
            return "检测到不安全的输入，请重新提问"

        # 组装上下文前：脱敏
        safe_results = gateway.desensitize_context(results)
    """

    def __init__(self, injection_enabled: bool = True,
                 desensitization_enabled: bool = True):
        self.detector = InjectionDetector(enabled=injection_enabled)
        self.desensitizer = ContextDesensitizer(enabled=desensitization_enabled)

    def check_query(self, query: str) -> Tuple[bool, str]:
        """
        检查用户 query 安全性

        :return: (is_safe, sanitized_query)
            - is_safe=True: 安全，sanitized_query 为清洗后的 query
            - is_safe=False: 检测到注入，sanitized_query 为空
        """
        is_injection, labels, confidence = self.detector.detect(query)

        if is_injection:
            # 尝试清洗，保留安全部分
            cleaned = self.detector.sanitize(query)
            if not cleaned:
                return False, ""
            return True, cleaned

        return True, query

    def desensitize_context(self, results: List[dict]) -> List[dict]:
        """对检索结果进行上下文脱敏"""
        return self.desensitizer.desensitize_records(results)


# 全局单例
security_gateway = SecurityGateway()
