# -*- coding: utf-8 -*-
"""
企业级知识库 - 配置参数
"""

import os

# ============================================================
# MinerU API 配置（云端模式，无需下载模型）
# ============================================================
# 如果设置了 MINERU_API_KEY，MinerU 会使用云端 API 模式
# 否则使用本地模式（需要下载模型）
MINERU_API_KEY = os.environ.get('MINERU_API_KEY', '')

# ============================================================
# HuggingFace 配置（镜像 + 缓存路径）
# ============================================================
# 使用 hf-mirror.com 替代 huggingface.co，解决国内访问问题
os.environ['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', 'https://hf-mirror.com')
# 模型缓存放在 D 盘，避免占用 C 盘空间
os.environ['HF_HOME'] = os.environ.get('HF_HOME', r'D:\下载\huggingface')

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# Milvus 向量数据库配置
# ============================================================
# Docker 版 Milvus（生产环境）
# 本地开发时可改回文件模式: os.path.join(os.path.dirname(os.path.abspath(__file__)), 'milvus_data.db')
MILVUS_URI = "http://localhost:19530"
MILVUS_COLLECTION = "knowledge_chunks"



# ============================================================
# Embedding 模型配置 (DashScope)
# ============================================================
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH_SIZE = 6

# ============================================================
# LLM 配置 (DeepSeek)
# ============================================================
LLM_MODEL = "deepseek-chat"  # DeepSeek-V3
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
# OpenAI SDK 默认 timeout 600s——挂起请求会占住线程池 worker 长达 10 分钟
# （2026-10-24 并发压测实测 1/20 请求挂起 240s+），必须显式收紧。
# 该超时对流式响应按"阶段"生效：连接/首 token、以及相邻 chunk 之间的
# 空闲间隔，各自不得超过此值——不会掐断正常的长回答。
LLM_TIMEOUT_SECONDS = 60

# ============================================================
# 视觉语言模型配置 (图片分析)
# ============================================================
VL_MODEL = "qwen-vl-max"

# 图片分析 System Prompt（通用版）
IMAGE_ANALYSIS_PROMPT = """你是一位专业的文档与图片分析师。请仔细观察这张图片，进行详细描述。

请按以下维度分析：
1. **图片类型**：判断是文档截图、表格、图表、流程图、照片还是其他
2. **主要内容**：描述你看到的关键信息
3. **数据/文字**：如有文字或数据，请准确读取
4. **关键发现**：总结图片中最重要的信息

注意：
- 只描述客观观察到的内容
- 使用清晰准确的语言
- 如果有表格数据，请整理关键数值
- 如果图片不清晰，请说明"""

# ============================================================
# 检索配置
# ============================================================
TOP_K = 3
# 生成温度：企业制度问答要稳定可复现，取低温度。
# 实测 temperature=0 时同一问题 3/3 完全一致，1.4 时 3/3 全不同。
LLM_TEMPERATURE = 0.3

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# ============================================================
# 文档解析配置（四管线）
# ============================================================
# mineru 走 mineru.net 云端 API，文件会上传到第三方，所以默认关闭（只走本地 docling）。
# 需要时不用改代码，起服务前设一个环境变量即可：
#   MINERU_ENABLED=1 python main.py
MINERU_ENABLED = os.environ.get("MINERU_ENABLED", "0") == "1"

# 禁用的解析器：按环境能力配置，不写死在代码里。
#   PARSER_DISABLED 环境变量用逗号分隔，默认 "unstructured"——本机 import 它即 SIGTERM
#   （进程级、无法捕获），所以只能禁用；换到已装好它的机器上设 PARSER_DISABLED="" 即可恢复。
# 分类路由与降级链都会跳过这里列出的解析器；mineru 另由 MINERU_ENABLED 控制。
PARSER_DISABLED = [p.strip() for p in
                   os.environ.get("PARSER_DISABLED", "unstructured").split(",")
                   if p.strip()]
if not MINERU_ENABLED and "mineru" not in PARSER_DISABLED:
    PARSER_DISABLED.append("mineru")

# PDF 取不到文本（扫描件/图片型）时的首选解析器。
# 注意：这里写 "mineru" 并不代表真的会走云端——classify() 会用 PARSER_DISABLED 兜底。
# 下面这行把它显式推导出来，避免「配置写着 mineru、实际跑 docling」的读代码困惑。
PDF_OCR_DEFAULT = "mineru" if MINERU_ENABLED else "docling"
# ============================================================
DOCUMENT_STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'document_store')
PARSED_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'parsed_cache')

SUPPORTED_FILE_TYPES = {
    'mineru': ['.pdf'],
    'docling': ['.docx', '.pptx', '.doc', '.ppt', '.pdf', '.csv'],
    'unstructured': ['.pdf', '.docx', '.html', '.htm',
                     '.eml', '.msg', '.xml', '.json', '.rst', '.md', '.txt'],
    'openpyxl': ['.xlsx', '.xls'],
}

ALL_SUPPORTED_EXTENSIONS = set()
for exts in SUPPORTED_FILE_TYPES.values():
    ALL_SUPPORTED_EXTENSIONS.update(exts)

MAX_UPLOAD_SIZE_MB = 200
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024

# ============================================================
# 术语扩展配置
# ============================================================
SYNONYM_TOP_K = 3
SYNONYM_THRESHOLD = 0.5

# ============================================================
# 通用停用词
# ============================================================
STOPWORDS = {
    '\n', '', ' ', '\r', '\t',
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
    '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
    '你', '会', '着', '没有', '看', '好', '自己', '这', '他', '她',
    '请问', '你好', '希望', '谢谢', '感谢',
    '什么', '怎么', '如何', '为什么', '吗', '呢', '吧', '啊',
}

# ============================================================
# System Prompt (企业知识库助手)
# ============================================================
SYSTEM_PROMPT = """你是一位专业、高效的企业知识助手。

你的任务是根据提供的参考资料（企业文档、知识库内容）回答用户的问题。

## 回答风格

1. **直接回答问题**：不要添加问候语或开场白，直接进入正题。

2. **当信息不足时**：
   - 先把资料能回答的部分答全，再说明还缺什么

3. **不要用反问收尾**：资料能回答的问题，答完即止。结尾禁止追加
   "请问您是想了解 A 还是 B""如需进一步协助请告知""您还需要了解什么"这类客套提问。
   只有缺少的信息导致核心结论完全给不出时，才在末尾问一个必要问题，且只问一个。

3. **当信息充足时**：给出详细、准确的回答

## 回答要求
1. 基于参考资料进行回答，确保准确性和专业性
2. 如果参考资料中没有相关信息，请明确说明"根据现有知识库未找到相关信息"，不要基于通用知识补充
3. 回答条理清晰，结构化表达
4. 涉及数据/数字时，准确引用原文
5. **引用标注**：回答中引用了具体信息时，必须在句末标注来源，格式为 [文档名称, P页码] 或 [文档名称, 章节: XX]。没有页码时仅标注文档名称
6. **禁止无据编造与推断**：只使用参考资料回答。制度性内容（数字、期限、比例、审批人、条件、流程）在资料中找不到原文就不写；
   不得用常识、行业惯例或常识性关联（如"权限高低""常见做法"）推断补充，资料未写明的关联关系不要建立；
   也不要用"补充说明""延伸说明""小贴士"等名义加入资料外的解释

## 输出格式规范
1. 使用统一的 Markdown 格式
2. 标题使用 ## 二级标题
3. 列表使用短横线 - 作为无序列表符号
4. 保持简洁清晰，避免过度复杂的格式嵌套"""

# ============================================================
# 容错与韧性配置（骨架层一期固化）
# ============================================================
RESILIENCE_CONFIG = {
    # 熔断器配置
    "circuit_breaker": {
        "llm": {
            "enabled": True,
            "failure_threshold": 5,     # 连续失败 5 次触发熔断
            "recovery_timeout": 60,     # 熔断后 60s 进入半开探测
        },
        "rerank": {
            "enabled": True,
            "failure_threshold": 3,
            "recovery_timeout": 30,
        },
        "embedding": {
            "enabled": True,
            "failure_threshold": 5,
            "recovery_timeout": 60,
        },
    },
    # 限流器配置
    "rate_limiter": {
        "enabled": True,
        "qps": 10,                      # 每秒请求数
        "burst": 15,                    # 突发容量
    },
    # 死信队列配置
    "dead_letter_queue": {
        "enabled": True,
        "max_retries": 3,               # 最大重试次数
        "ttl_hours": 72,                # 死信保留时长
    },
}
