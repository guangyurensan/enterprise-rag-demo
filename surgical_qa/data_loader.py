# -*- coding: utf-8 -*-
"""
企业级知识库 - 数据加载模块
支持两种数据源:
1. CSV 问答数据（可选）
2. 企业文档（PDF/Word/Excel/HTML/邮件等，通过四管线解析器）
"""

import os
import csv
from typing import List, Dict, Optional
from config import DOCUMENT_STORE_DIR

# CSV 文件路径（可选，不存在则跳过）
CSV_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'enterprise_qa.csv'
)


# ============================================================
# CSV 数据加载（可选）
# ============================================================
def load_csv_data(csv_path=None):
    """
    加载CSV数据，提取结构化问答记录
    :param csv_path: CSV文件路径，默认使用配置中的路径
    :return: 问答记录列表 list[dict]
    """
    if csv_path is None:
        csv_path = CSV_FILE

    if not os.path.exists(csv_path):
        print(f"[WARN] CSV 文件不存在: {csv_path}，跳过 CSV 加载")
        return []

    records = []

    for enc in ['gb18030', 'utf-8', 'gbk']:
        try:
            with open(csv_path, 'r', encoding=enc) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    department = row.get('department', '').strip()
                    title = row.get('title', '').strip()
                    ask = row.get('ask', '').strip()
                    answer = row.get('answer', '').strip()

                    if not ask or not answer:
                        continue

                    question = f"{title} {ask}".strip() if title else ask

                    records.append({
                        'department': department,
                        'question': question,
                        'answer': answer,
                        'full_text': f"{question} {answer}",
                        'source': 'csv'
                    })

            print(f"[OK] CSV 数据加载成功，编码: {enc}")
            print(f"     共 {len(records)} 条有效问答记录")
            return records

        except (UnicodeDecodeError, UnicodeError):
            continue

    raise ValueError(f"无法用任何已知编码读取文件: {csv_path}")


# ============================================================
# 企业文档加载
# ============================================================
def load_documents_as_records(doc_manager, doc_ids: List[str] = None) -> List[Dict]:
    """
    将已解析的企业文档转换为检索记录格式
    :param doc_manager: DocumentManager 实例
    :param doc_ids: 指定文档ID列表，None 则加载全部
    :return: 记录列表（兼容原有检索格式）
    """
    chunks = doc_manager.get_all_chunks(doc_ids=doc_ids)

    # 构建 doc_id → 文档元数据映射（用于 doc_name 和权限字段）
    doc_meta_map = {}
    for doc_id, doc_info in doc_manager.registry.documents.items():
        if not doc_info.get('is_deleted'):
            doc_meta_map[doc_id] = {
                'doc_name': doc_info.get('filename', ''),
                # 干净的制度名（去掉序号前缀与版本后缀），用于块级上下文
                'base_title': doc_info.get('base_title', ''),
                'version_label': doc_info.get('version_label', ''),
                'access_level': doc_info.get('access_level', 'internal'),
                'allowed_groups': doc_info.get('allowed_groups', 'all'),
                'version': doc_info.get('version', 1),
            }

    records = []

    for chunk in chunks:
        meta = doc_meta_map.get(chunk.doc_id, {})
        # 块级上下文：制度名 + 章节标题 + 正文，让向量与 BM25 都能看到"这段属于哪份文件哪一节"
        header = ' '.join(x for x in (meta.get('base_title', ''), chunk.title) if x)
        records.append({
            'department': chunk.metadata.get('department', '文档库'),
            'question': chunk.title or chunk.content[:100],
            'answer': chunk.content,
            'full_text': f"{header} {chunk.content}".strip(),
            'source': 'document',
            'doc_id': chunk.doc_id,
            'chunk_id': chunk.chunk_id,
            'chunk_index': chunk.chunk_index,
            'page_number': chunk.page_number,
            'chunk_type': chunk.chunk_type,
            'language': chunk.language,
            'parent_title': chunk.title or '',
            'truncated': chunk.metadata.get('truncated', False),
            # 权限字段（骨架层一期固化，从文档继承）
            'doc_name': doc_meta_map.get(chunk.doc_id, {}).get('doc_name', ''),
            # 引用标注用干净制度名（003_员工考勤…_V2.1_现行最新.txt 对用户不可读）
            'base_title': meta.get('base_title', ''),
            'version_label': meta.get('version_label', ''),
            'access_level': doc_meta_map.get(chunk.doc_id, {}).get('access_level', 'internal'),
            'allowed_groups': doc_meta_map.get(chunk.doc_id, {}).get('allowed_groups', 'all'),
            'is_deleted': False,
            'version': doc_meta_map.get(chunk.doc_id, {}).get('version', 1),
        })

    print(f"[OK] 文档数据加载完成: {len(records)} 个 chunks "
          f"(来自 {len(set(c.doc_id for c in chunks))} 个文档)")
    return records


def load_all_records(csv_path=None, doc_manager=None) -> List[Dict]:
    """
    加载所有数据源（CSV + 企业文档）
    :param csv_path: CSV 文件路径
    :param doc_manager: DocumentManager 实例（可选）
    :return: 合并后的记录列表
    """
    all_records = []

    # 1. 加载 CSV 数据
    csv_records = load_csv_data(csv_path)
    all_records.extend(csv_records)

    # 2. 加载企业文档
    if doc_manager is not None:
        doc_records = load_documents_as_records(doc_manager)
        all_records.extend(doc_records)

    print(f"\n[总计] 共加载 {len(all_records)} 条记录 "
          f"(CSV: {len(csv_records)}, 文档: {len(all_records) - len(csv_records)})")

    return all_records


def get_department_stats(records):
    """统计各部门/来源的问答数量"""
    stats = {}
    for r in records:
        dept = r.get('department', '未知')
        stats[dept] = stats.get(dept, 0) + 1
    return stats


def get_all_departments(records):
    """获取所有部门/来源列表（按数量降序）"""
    stats = get_department_stats(records)
    return [dept for dept, _ in sorted(stats.items(), key=lambda x: -x[1])]


def filter_by_department(records, department):
    """按部门/来源过滤记录"""
    if not department or department == '全部':
        return records
    return [r for r in records if r.get('department') == department]



