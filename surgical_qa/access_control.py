# -*- coding: utf-8 -*-
"""
企业级知识库 - 权限控制模块（骨架层）

核心职责：
1. 用户上下文定义（UserContext）
2. 权限判定（AccessPolicy.check_access 四分支齐全）
3. Milvus 过滤表达式生成（AccessPolicy.build_filter_expr）
4. 组名清洗（clean_groups）
5. 审计日志（AuditLogger）

设计原则：
- 骨架层一期固化，函数签名稳定，后续只改内部判断逻辑
- ACL 过滤在召回阶段（Rerank 前）执行，不可绕过
"""

import os
import json
import logging
import threading
from datetime import datetime
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


# ============================================================
# 用户上下文
# ============================================================
class UserContext:
    """
    用户上下文 — 检索时传入，用于 ACL 过滤

    :param role: 用户角色 "admin" | "manager" | "employee" | "guest"
    :param groups: 用户所属组列表，如 ["人事部", "全员"]
    """

    VALID_ROLES = {"admin", "manager", "employee", "guest"}

    def __init__(self, role: str = "employee", groups: Optional[List[str]] = None,
                 user_id: str = "", username: str = "", display_name: str = ""):
        if role not in self.VALID_ROLES:
            logger.warning(f"[AccessControl] 未知角色 '{role}'，降级为 employee")
            role = "employee"
        self.role = role
        self.groups = groups or ["全员"]
        # 身份字段（2026-10-25 SSO 地基）：来自登录令牌，审计落 user_id。
        # 旧代码 UserContext(role, groups) 位置传参不受影响
        self.user_id = user_id
        self.username = username
        self.display_name = display_name

    def __repr__(self):
        return f"UserContext(role={self.role}, groups={self.groups}, user_id={self.user_id!r})"


# ============================================================
# 权限判定策略
# ============================================================
class AccessPolicy:
    """
    权限判定策略（四分支齐全，骨架层一期固化）

    密级层级：public(0) < internal(1) < restricted(2) < confidential(3)
    角色上限：guest=0, employee=1, manager=2, admin=3

    判定规则：
    - admin        -> 全部可见
    - guest        -> 仅 public
    - employee     -> public + internal
    - manager      -> public + internal + restricted（需组匹配）
    - confidential -> 仅 admin
    """

    LEVELS = {"public": 0, "internal": 1, "restricted": 2, "confidential": 3}
    ROLE_MAX = {"guest": 0, "employee": 1, "manager": 2, "admin": 3}

    @classmethod
    def check_access(cls, doc_level: str, doc_groups: str, user_ctx: UserContext) -> bool:
        """
        判定用户是否有权访问该文档

        :param doc_level: 文档密级 "public"|"internal"|"restricted"|"confidential"
        :param doc_groups: 文档允许组（逗号分隔字符串，"all" 表示所有组）
        :param user_ctx: 用户上下文
        :return: True 可访问，False 不可访问
        """
        # 标准化
        doc_level = doc_level or "internal"
        doc_groups = doc_groups or "all"

        # admin 全部可见
        if user_ctx.role == "admin":
            return True

        # 获取角色可达最高密级
        user_max = cls.ROLE_MAX.get(user_ctx.role, 0)
        doc_min = cls.LEVELS.get(doc_level, 1)

        # 密级超出角色上限
        if doc_min > user_max:
            return False

        # restricted 需要额外组匹配
        if doc_level == "restricted" and doc_groups != "all":
            allowed = set(g.strip() for g in doc_groups.split(",") if g.strip())
            user_groups = set(user_ctx.groups)
            if not user_groups & allowed:
                return False

        return True

    @classmethod
    def build_filter_expr(cls, user_ctx: UserContext) -> str:
        """
        生成 Milvus 过滤表达式（引擎侧过滤）

        :param user_ctx: 用户上下文
        :return: Milvus 过滤表达式字符串，空字符串表示不过滤
                 始终追加 is_deleted == False
        """
        # 基础 ACL 条件
        if user_ctx.role == "admin":
            acl_expr = ''  # admin 无 ACL 限制
        elif user_ctx.role == "manager":
            # manager: public + internal + restricted（组匹配在内存侧做，引擎侧只过滤密级）
            acl_expr = '(access_level == "public" or access_level == "internal" or access_level == "restricted")'
        elif user_ctx.role == "employee":
            acl_expr = '(access_level == "public" or access_level == "internal")'
        else:  # guest
            acl_expr = 'access_level == "public"'

        # 始终追加 is_deleted 条件
        deleted_expr = 'is_deleted == false'

        if acl_expr:
            return f'{acl_expr} and {deleted_expr}'
        else:
            return deleted_expr


# ============================================================
# 组名清洗
# ============================================================
def clean_groups(raw_str: str) -> str:
    """
    清洗组名字符串

    :param raw_str: 原始输入，如 " 人事部 , 财务部 , , 人事部 "
    :return: 清洗后的逗号分隔字符串，如 "人事部,财务部"
             非法组名只告警，不阻断
    """
    if not raw_str or not raw_str.strip():
        return "all"

    # 分割、strip、去空、去重（保持顺序）
    seen = set()
    cleaned = []
    for g in raw_str.split(","):
        g = g.strip()
        if g and g not in seen:
            seen.add(g)
            cleaned.append(g)

    if not cleaned:
        return "all"

    result = ",".join(cleaned)
    logger.info(f"[AccessControl] 组名清洗: '{raw_str}' -> '{result}'")
    return result


# ============================================================
# 自动定密（2026-10-24 新增）
# ============================================================
# 背景：ACL 四级密级机制上线后从未有文档被真正定密——所有入库文档
# 默认 internal（普通员工可见），导致工资表、劳动合同等敏感文档对
# 全员泄露（2026-10-24 用户以员工身份查到总经理工资，实测复现）。
# 本函数是密级分配的单一事实源：
#   - add_document 未显式指定密级时自动调用（堵新入口）
#   - apply_acl.py 存量刷新调用（修旧账）
# 规则按顺序首条命中即生效；宁高勿低（错标 internal 是泄露，错标
# restricted 只是多一次权限申请）。
import re as _re

CLASSIFY_RULES = [
    # 个人薪酬数据：仅管理员
    (r'工资|薪酬|薪资|工资条|offer', 'confidential', 'all'),
    # 个人隐私文档（本人与人事之外不应可见）
    (r'劳动合同|竞业|保密协议|入职登记|简历', 'restricted', '人事部,管理层'),
    # 个人单据（请假/报销单据含个人明细）
    (r'请假申请单|报销单', 'restricted', '财务部,管理层'),
    # 财务与经营数据（利润/审计/经营分析/预算/台账/账龄/研发投入）
    (r'利润表|审计报告|经营分析|经营快报|年度报告|预算表|账龄|研发项目投入|台账|报销明细',
     'restricted', '管理层,财务部'),
    # 组织与人事数据（编制/招聘）
    (r'组织架构|人员编制|招聘需求', 'restricted', '人事部,管理层'),
    # 对外合同文本与价格（合同评审"办法"是制度，排除）
    (r'合同(?!评审)|价格表', 'restricted', '法务部,销售部,管理层'),
    # 对外公开资料
    (r'公司简介|技术规格书|产品手册|产品介绍', 'public', 'all'),
]
_CLASSIFY_DEFAULT = ('internal', 'all')


def classify_document(filename: str) -> tuple:
    """
    按文件名自动定密

    :param filename: 文档文件名
    :return: (access_level, allowed_groups) 二元组
    """
    for pattern, level, groups in CLASSIFY_RULES:
        if _re.search(pattern, filename):
            return (level, groups)
    return _CLASSIFY_DEFAULT


# ============================================================
# 审计日志（骨架层一期固化：持久化 + 脱敏 + 扩展字段）
# ============================================================

# PII 脱敏正则 — 单一数据源在 security.py，此处引用
from security import PII_PATTERNS as _SECURITY_PII_PATTERNS

# 审计日志用的脱敏模式（身份证号优先，避免与银行卡冲突）
_PII_PATTERNS = [
    (_SECURITY_PII_PATTERNS["id_card_cn"], '***身份证***'),
    (_SECURITY_PII_PATTERNS["phone_cn"],   '***手机号***'),
    (_SECURITY_PII_PATTERNS["email"],      '***邮箱***'),
    (_SECURITY_PII_PATTERNS["bank_card"],  '***银行卡号***'),
]

# 审计日志默认存储路径
_AUDIT_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'audit_logs'
)


def desensitize_query(text: str) -> str:
    """
    对查询文本进行 PII 脱敏（用于审计日志记录）
    不影响原始查询，仅用于日志输出
    """
    if not text:
        return text
    result = text
    for pattern, replacement in _PII_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


class AuditLogger:
    """
    审计日志（骨架层一期固化结构）

    固定输出字段：
    - timestamp: 时间戳
    - event_type: 事件类型 (search / doc_access / document_op)
    - user_role: 用户角色
    - user_groups: 用户组列表
    - query: 查询文本（已脱敏）
    - query_raw_hash: 原始查询摘要（SHA256 前8位，用于关联分析）
    - milvus_filter: Milvus 过滤表达式
    - result_count: 命中结果数
    - filtered_count: 被权限过滤掉的数量
    - latency_ms: 检索耗时（毫秒）
    - channels: 召回通道列表 ["vector", "bm25"]
    - hit_docs: 命中文档摘要 [{doc_name, chunk_id, score}]
    - doc_id / doc_level / access_granted: 文档访问事件专用

    持久化：JSONL 文件，按天分割
    脱敏：查询文本自动替换 PII
    """

    def __init__(self, logger_name: str = "audit", log_dir: str = None):
        self.audit_logger = logging.getLogger(logger_name)
        self.log_dir = log_dir or _AUDIT_LOG_DIR
        os.makedirs(self.log_dir, exist_ok=True)
        self._lock = threading.Lock()

    def _get_log_file(self) -> str:
        """按天分割的日志文件路径"""
        date_str = datetime.now().strftime('%Y-%m-%d')
        return os.path.join(self.log_dir, f'audit_{date_str}.jsonl')

    def _persist(self, entry: Dict):
        """持久化到 JSONL 文件（线程安全）"""
        log_file = self._get_log_file()
        with self._lock:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def log_search(
        self,
        user_ctx: UserContext,
        filter_expr: str,
        result_count: int,
        filtered_count: int = 0,
        query: str = "",
        latency_ms: float = 0,
        channels: List[str] = None,
        hit_docs: List[Dict] = None,
    ):
        """
        记录一次检索的审计日志

        :param user_ctx: 用户上下文
        :param filter_expr: Milvus 过滤表达式
        :param result_count: 命中结果数
        :param filtered_count: 被权限过滤的数量
        :param query: 原始查询文本（自动脱敏后存储）
        :param latency_ms: 检索耗时（毫秒）
        :param channels: 召回通道，如 ["vector", "bm25"]
        :param hit_docs: 命中文档摘要列表
        """
        # 脱敏处理
        safe_query = desensitize_query(query)
        query_hash = self._short_hash(query)

        # 命中密级汇总：内审/等保的核心诉求是"谁在什么时间查了涉密内容"，
        # 把密级提升为顶层字段，查询侧无需 join registry 即可筛选
        _SENSITIVE_LEVELS = ("confidential", "restricted")
        hit_levels = [
            h.get("access_level", "unknown")
            for h in (hit_docs or [])
            if h.get("access_level")
        ]
        sensitive_hit = any(lv in _SENSITIVE_LEVELS for lv in hit_levels)

        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "search",
            "user_id": getattr(user_ctx, "user_id", ""),
            "user_role": user_ctx.role,
            "user_groups": user_ctx.groups,
            "query": safe_query,
            "query_hash": query_hash,
            "milvus_filter": filter_expr,
            "result_count": result_count,
            "filtered_count": filtered_count,
            "latency_ms": round(latency_ms, 1),
            "channels": channels or [],
            "hit_docs": (hit_docs or [])[:5],  # 最多记录前5条
            "hit_levels": hit_levels[:5],
            "sensitive_hit": sensitive_hit,
        }

        # Python logging（摘要）
        self.audit_logger.info(
            f"[AUDIT] search | "
            f"time={entry['timestamp']} | "
            f"role={user_ctx.role} | "
            f"groups={user_ctx.groups} | "
            f"filter={filter_expr!r} | "
            f"results={result_count} | "
            f"filtered={filtered_count} | "
            f"latency={entry['latency_ms']}ms | "
            f"sensitive={sensitive_hit} | "
            f"query={safe_query[:50]!r}"
        )

        # 持久化
        self._persist(entry)

    def log_answer(
        self,
        user_ctx: UserContext,
        query: str,
        answer_chars: int = 0,
        refused: bool = False,
        citation_docs: List[str] = None,
        latency_ms: float = 0,
        llm_called: bool = True,
    ):
        """
        记录一次问答的答案侧审计（等保可追溯性：系统最终返回了什么）

        与 search 事件通过 query_hash 关联：search 记"检索了什么"，
        answer 记"返回了什么"，两者拼起来才是完整的问答链路。

        :param user_ctx: 用户上下文
        :param query: 原始查询（自动脱敏后存储）
        :param answer_chars: 答案长度（不存答案全文：正文已含脱敏上下文，
            且全文落审计有二次泄露风险，长度+拒答标志足够审计用）
        :param refused: 是否拒答（零召回统一口径 / LLM 拒答）
        :param citation_docs: 引用文档名列表（去重）
        :param latency_ms: 检索+生成总耗时
        :param llm_called: 是否实际调用了 LLM（零召回短路时不调用）
        """
        safe_query = desensitize_query(query)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "answer",
            "user_id": getattr(user_ctx, "user_id", ""),
            "user_role": user_ctx.role,
            "user_groups": user_ctx.groups,
            "query": safe_query,
            "query_hash": self._short_hash(query),
            "answer_chars": answer_chars,
            "refused": refused,
            "llm_called": llm_called,
            "citation_docs": (citation_docs or [])[:10],
            "latency_ms": round(latency_ms, 1),
        }

        self.audit_logger.info(
            f"[AUDIT] answer | "
            f"time={entry['timestamp']} | "
            f"role={user_ctx.role} | "
            f"refused={refused} | "
            f"chars={answer_chars} | "
            f"citations={len(entry['citation_docs'])} | "
            f"latency={entry['latency_ms']}ms | "
            f"query={safe_query[:50]!r}"
        )

        self._persist(entry)

    def log_document_access(
        self,
        user_ctx: UserContext,
        doc_id: str,
        doc_level: str,
        access_granted: bool,
    ):
        """记录文档访问审计"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "doc_access",
            "user_id": getattr(user_ctx, "user_id", ""),
            "user_role": user_ctx.role,
            "user_groups": user_ctx.groups,
            "doc_id": doc_id,
            "doc_level": doc_level,
            "access_granted": access_granted,
        }

        self.audit_logger.info(
            f"[AUDIT] doc_access | "
            f"time={entry['timestamp']} | "
            f"role={user_ctx.role} | "
            f"doc_id={doc_id} | "
            f"level={doc_level} | "
            f"granted={access_granted}"
        )

        self._persist(entry)

    def log_document_operation(
        self,
        user_ctx: UserContext,
        operation: str,
        doc_id: str,
        success: bool,
        detail: str = "",
    ):
        """
        记录文档操作审计（上传/更新/删除/恢复）

        :param user_ctx: 用户上下文
        :param operation: 操作类型 "upload"/"update"/"delete"/"restore"
        :param doc_id: 文档 ID
        :param success: 是否成功
        :param detail: 附加信息
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": "document_op",
            "user_id": getattr(user_ctx, "user_id", ""),
            "user_role": user_ctx.role,
            "user_groups": user_ctx.groups,
            "operation": operation,
            "doc_id": doc_id,
            "success": success,
            "detail": detail[:200],
        }

        self.audit_logger.info(
            f"[AUDIT] doc_op | "
            f"time={entry['timestamp']} | "
            f"role={user_ctx.role} | "
            f"op={operation} | "
            f"doc_id={doc_id} | "
            f"success={success}"
        )

        self._persist(entry)

    @staticmethod
    def _short_hash(text: str) -> str:
        """生成文本短摘要（SHA256 前8位）"""
        if not text:
            return ""
        import hashlib
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]

    # ---- 日志查询 API（供 app.py 调用） ----

    def query_logs(
        self,
        date: str = None,
        event_type: str = None,
        role: str = None,
        limit: int = 50,
        sensitive_only: bool = False,
        level: str = None,
    ) -> List[Dict]:
        """
        查询审计日志

        :param date: 日期，格式 "YYYY-MM-DD"，默认今天
        :param event_type: 事件类型过滤，None 表示不过滤
        :param role: 角色过滤，None 表示不过滤
        :param limit: 最大返回条数
        :param sensitive_only: 仅看命中涉密文档（confidential/restricted）的检索
        :param level: 按命中密级筛选，如 "confidential"
        :return: 日志条目列表
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')

        log_file = os.path.join(self.log_dir, f'audit_{date}.jsonl')
        if not os.path.exists(log_file):
            return []

        results = []
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # 过滤
                if event_type and entry.get('event_type') != event_type:
                    continue
                if role and entry.get('user_role') != role:
                    continue
                if sensitive_only and not entry.get('sensitive_hit'):
                    continue
                if level and level not in entry.get('hit_levels', []):
                    continue

                results.append(entry)
                if len(results) >= limit:
                    break

        return results

    def get_stats(self, date: str = None) -> Dict:
        """
        获取审计统计

        :param date: 日期，默认今天
        :return: {"total": N, "by_event": {...}, "by_role": {...}, "avg_latency_ms": X}
        """
        logs = self.query_logs(date=date, limit=10000)
        if not logs:
            return {"total": 0, "by_event": {}, "by_role": {}, "avg_latency_ms": 0,
                    "sensitive_hits": 0, "by_hit_level": {}, "refused": 0}

        by_event = {}
        by_role = {}
        by_hit_level = {}
        total_latency = 0
        latency_count = 0
        sensitive_hits = 0
        refused_count = 0

        for entry in logs:
            et = entry.get('event_type', 'unknown')
            by_event[et] = by_event.get(et, 0) + 1

            r = entry.get('user_role', 'unknown')
            by_role[r] = by_role.get(r, 0) + 1

            if entry.get('sensitive_hit'):
                sensitive_hits += 1
            for lv in entry.get('hit_levels', []):
                by_hit_level[lv] = by_hit_level.get(lv, 0) + 1
            if entry.get('refused'):
                refused_count += 1

            lat = entry.get('latency_ms', 0)
            if lat > 0:
                total_latency += lat
                latency_count += 1

        return {
            "total": len(logs),
            "by_event": by_event,
            "by_role": by_role,
            "avg_latency_ms": round(total_latency / latency_count, 1) if latency_count else 0,
            "sensitive_hits": sensitive_hits,
            "by_hit_level": by_hit_level,
            "refused": refused_count,
        }


# 全局审计日志实例
audit_logger = AuditLogger()



