# -*- coding: utf-8 -*-
"""
企业级知识库 - 数据清洗管线配置

集中管理：路由规则、黑名单、质量阈值、分块参数等
所有配置项均可按业务需求热调整
"""

import os

# ============================================================
# 文件前置粗筛
# ============================================================

# 允许的文件类型及大小限制
FILE_PRESCREEN = {
    "max_size_mb": 200,
    "allowed_extensions": {
        ".pdf", ".docx", ".pptx", ".doc", ".ppt",
        ".xlsx", ".xls", ".html", ".htm", ".eml", ".msg",
        ".csv", ".txt", ".md", ".rst", ".json", ".xml",
    },
    # 损坏检测：文件头 magic bytes
    "magic_bytes": {
        ".pdf": [b"%PDF"],
        ".docx": [b"PK\x03\x04"],
        ".pptx": [b"PK\x03\x04"],
        ".xlsx": [b"PK\x03\x04"],
        ".doc": [b"\xd0\xcf\x11\xe0"],
        ".ppt": [b"\xd0\xcf\x11\xe0"],
    },
    # 最小文件大小（字节），低于此值视为空文件/损坏
    "min_file_size": 100,
}

# ============================================================
# 解析器映射表（集中管理，换解析器只改这里）
# ============================================================
PARSER_MAP = {
    "mineru":       "mineru",
    "docling":      "docling",
    "unstructured": "unstructured",
    "excel_native": "openpyxl",  # Excel 全量走 OpenpyxlParser（2026-10-24 起）
    "openpyxl":     "openpyxl",
}

# ============================================================
# 路由调度器规则（顺序匹配，命中即终止）
# ============================================================

# 业务标签关键词 → 路由映射
# 在文件内容/文件名中检测到关键词 → 路由到指定解析器
# parser_key 对应 PARSER_MAP 中的 key，不直接写死解析器名
ROUTING_RULES = [
    # 规则0: 财务/年报/含表格密集 PDF → MinerU（表格提取质量更高）
    {
        "name": "财务报告",
        "match": {
            "filename_keywords": ["年度报告", "年报", "经营快报", "财务报表",
                                  "审计报告", "半年度", "季度报告", "利润表",
                                  "资产负债表", "现金流量"],
            "content_keywords": ["营业收入", "净利润", "总资产", "负债",
                                "资产负债表", "利润表", "现金流量表"],
        },
        "parser_key": "mineru",
        "priority": 105,
    },
    # 规则1: 合同类 → MinerU
    {
        "name": "合同类",
        "match": {
            "filename_keywords": ["合同", "contract", "协议", "agreement"],
            "content_keywords": ["甲方", "乙方", "合同编号", "签章", "违约", "条款"],
        },
        "parser_key": "mineru",
        "priority": 100,
    },
    # 规则2: 可研报告 → MinerU
    {
        "name": "可研报告",
        "match": {
            "filename_keywords": ["可行性", "可研", "feasibility"],
            "content_keywords": ["可行性研究", "投资估算", "经济评价", "项目背景"],
        },
        "parser_key": "mineru",
        "priority": 95,
    },
    # 规则3: 含公式 → MinerU
    {
        "name": "含公式文档",
        "match": {
            "content_patterns": [
                r"\\[a-zA-Z]+\{",
                r"\$\$.*?\$\$",
                r"\\begin\{(equation|align)",
                r"[∑∏∫∂√∞≈≠≤≥]",
            ],
            "pattern_threshold": 3,  # 至少匹配3处
        },
        "parser_key": "mineru",
        "priority": 90,
    },
    # 规则4: 扫描件 PDF → MinerU
    {
        "name": "扫描件PDF",
        "match": {
            "file_type": ".pdf",
            "is_scanned": True,  # 由 prescreen 检测
        },
        "parser_key": "mineru",
        "priority": 85,
    },
    # 规则5: 图片型 PDF（非扫描件但图片密集）→ MinerU
    {
        "name": "图片型PDF",
        "match": {
            "file_type": ".pdf",
            "is_image_heavy": True,  # 由 prescreen 检测
        },
        "parser_key": "mineru",
        "priority": 80,
    },
    # 规则6: 制度/常规报告 → Docling
    {
        "name": "制度报告",
        "match": {
            "filename_keywords": ["制度", "规范", "报告", "总结", "计划", "通知"],
            "content_keywords": ["第一章", "第一条", "目的", "范围", "职责", "流程"],
        },
        "parser_key": "docling",
        "priority": 70,
    },
    # 规则7: 复杂 Excel → openpyxl 专用
    {
        "name": "复杂Excel",
        "match": {
            "file_type": ".xlsx",
            "is_complex_excel": True,  # 多 sheet / 大表格
        },
        "parser_key": "excel_native",
        "priority": 60,
    },
    # 规则8: 简单台账 Excel → openpyxl 专用
    # （2026-10-24 起不再走 unstructured：docling 的 markdown 转换会畸变
    #   合并单元格，统一交给 OpenpyxlParser，最终解析仍由 dispatcher 兜底）
    {
        "name": "简单Excel",
        "match": {
            "file_type": ".xlsx",
        },
        "parser_key": "excel_native",
        "priority": 55,
    },
    # 规则9: Word/PPT → Docling
    {
        "name": "Word/PPT",
        "match": {
            "file_type": [".docx", ".pptx", ".doc", ".ppt"],
        },
        "parser_key": "docling",
        "priority": 45,
    },
    # 规则10: 杂格式 → Unstructured
    {
        "name": "杂格式",
        "match": {
            "file_type": [".eml", ".html", ".htm", ".csv", ".txt",
                          ".md", ".json", ".xml"],
        },
        "parser_key": "unstructured",
        "priority": 40,
    },
    # 兜底: PDF → Docling
    {
        "name": "默认PDF",
        "match": {"file_type": ".pdf"},
        "parser_key": "docling",
        "priority": 10,
    },
]



# ============================================================
# 差异化清洗 - 业务黑名单
# ============================================================

# 噪声模式（正则），匹配到的内容会被标记/剔除
CLEANER_NOISE_PATTERNS = {
    # 页眉页脚常见模式
    "header_footer": [
        r"^第\s*\d+\s*页\s*(共\s*\d+\s*页)?$",
        r"^Page\s*\d+\s*(of\s*\d+)?$",
        r"^\d+\s*/\s*\d+$",
        r"^—\s*\d+\s*—$",
        r"^\d+\s*$",  # 纯页码
    ],
    # 机密/版权声明
    "legal_notice": [
        r"(本文件为.*?机密|内部资料.*?严禁外传)",
        r"(Copyright|版权所有).*?\d{4}",
        r"(Confidential|机密文件)",
    ],
    # 自定义业务黑名单（可按需扩展）
    "business_blacklist": [
        # r"此内容已废弃",
        # r"测试数据.*?请勿使用",
    ],
}

# 是否剔除黑名单匹配内容（False = 仅标记不剔除）
NOISE_REMOVE_ENABLED = {
    "header_footer": True,
    "legal_notice": True,
    "business_blacklist": False,  # 仅标记，留存日志
}

# 合并断行的阈值：行尾无标点 + 下一行首字非大写/非数字 → 合并
LINE_MERGE_ENABLED = True

# ============================================================
# 文档质检打分
# ============================================================

QUALITY_SCORING = {
    # 质量分阈值
    "pass_threshold": 60,    # ≥60 分放行
    "warn_threshold": 40,    # 40-60 分标记警告
    # <40 分 → 阻断入库，送人工重试台

    # 扣分项
    "penalty_per_garbled_char": 2,     # 每个乱码字符扣 2 分
    "penalty_per_noise_block": 5,      # 每个噪声块扣 5 分
    "penalty_empty_ratio": 30,         # 空白占比超 50% 扣 30 分
    "penalty_no_structure": 10,        # 无任何结构（无标题/段落）扣 10 分

    # 加分项
    "bonus_has_title": 10,             # 有标题层级
    "bonus_has_table": 5,              # 含表格
    "bonus_per_section": 2,            # 每个章节加 2 分（上限 20）
}

# ============================================================
# 智能分块参数（纯确定性分割）
# ============================================================

SMART_CHUNKING = {
    # Embedding 模型 token 上限（留余量）
    "max_embed_token": 900,
    # 短碎片合并阈值（token 数）
    "min_token": 15,
    # 滑动窗口配置
    "win_size_max": 6,       # 最大窗口（句子数）
    "step": 0,               # 0 = 无重叠（step = len(picked_sents)）

    # 标点回退优先级（逗号、顿号不作为断开点）
    "punct_order": ["。", "！", "？", "；", "："],

    # token 估算系数（中文约 1 字 ≈ 1-1.5 token，取 0.55 留余量）
    "token_estimate_ratio": 0.55,

    # 表格分块
    "table": {
        "group_size": 8,             # 每 N 行一组
        "header_copy": True,         # 每个子 chunk 携带完整表头
    },

    # Context Enrichment：标题前缀注入 chunk content
    "context_enrichment": True,      # 将 parent_title 拼入 chunk 开头

    # 图片/公式：整块保留，不分
    "image_ocr": {"split": False},
    "equation": {"split": False},
}

# ============================================================
# Chunk 去重
# ============================================================

DEDUP = {
    "method": "simhash",       # simhash / minhash
    "threshold": 3,            # SimHash 海明距离 ≤ 3 视为重复
    "minhash_threshold": 0.8,  # MinHash Jaccard 相似度 ≥ 0.8 视为重复
    "action": "mark",          # mark（标记）/ remove（移除）
}

# ============================================================
# 向量库写入
# ============================================================

VECTOR_STORE = {
    "is_valid_default": True,   # 新写入 chunk 默认有效
    "soft_delete": True,        # 删除时设 is_valid=False 而非物理删除
}

# ============================================================
# 旁路：指标埋点 / 失败队列 / 人工重试
# ============================================================

PIPELINE_METRICS = {
    "enabled": True,
    "log_dir": os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pipeline_logs"
    ),
}

# 人工重试台目录
MANUAL_RETRY_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "manual_retry"
)


