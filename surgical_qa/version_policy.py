# -*- coding: utf-8 -*-
"""
版本治理模块（通用规则，与具体文件无关）
==========================================

解决：同一制度的多版本文档（V1.0/V2.0/V2.1_现行最新）同时入库且都能被召回。

设计原则（防过拟合）：
1. 不写死任何 doc_id / 文件名清单 —— 一切从文件名动态解析：
   - 制度名（base_title）：去掉 "NNN_" 序号前缀 和 "_Vx.y[_现行最新]" 版本后缀
   - 版本号（version_label）：V(\d+(\.\d+)*) 模式
   - 显式现行标记：文件名含 "_现行最新"
2. 现行版判定规则（对任何新数据同样生效）：
   - 同制度只有 1 个版本 → 现行
   - 多版本：显式"现行最新"标记优先；无标记时取最高版本号
   - 无版本概念的文档（CSV/SOP/PDF 报告等）→ 现行
3. 旧版处理：软删除（deleted_reason='superseded'），可随时 restore；
   用户手动删除（无 deleted_reason）的文档不参与现行版竞选，也不被本模块触碰
4. 幂等：重复执行结果一致；恢复某旧版会自动顶替原现行版（ 二者互斥自动维持）
"""

import os
import re
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

VERSION_RE = re.compile(r'_V(\d+(?:\.\d+)*)(_(现行最新))?', re.I)
PREFIX_RE = re.compile(r'^\d+_')
EXPLICIT_MARK_RE = re.compile(r'现行最新')

# 不是版本号、但常被写在文件名末尾的修饰词。
# 不处理它们会导致同一个制度被当成两个：「员工手册_final.pdf」和「员工手册_V2.0.pdf」
# 的 base_title 分别是「员工手册_final」和「员工手册」，两份会同时判为现行。
NOISE_SUFFIX_RE = re.compile(
    r'[_\-—\s]*(?:final|最终版?|最新版?|现行最新|现行版?|终版|定稿|修订版?|草案|草稿'
    r'|old|new|副本|copy|\(\d+\)|（\d+）)\s*$',
    re.I,
)


def _normalize_title(title: str) -> str:
    """反复剥掉末尾的修饰词，直到不再变化。

    只剥“末尾”——制度名中间出现「修订」不动，避免把「制度修订管理办法」拆坏。
    """
    prev = None
    while prev != title:
        prev = title
        title = NOISE_SUFFIX_RE.sub('', title).strip('_-— ')
    return title


def parse_doc_meta(filename: str) -> Dict:
    """
    从文件名解析文档元信息（纯函数，无状态）

    :param filename: 如 "003_员工考勤与假期管理规范_V2.1_现行最新.txt"
    :return: {"base_title": "员工考勤与假期管理规范",
              "version_label": "2.1",
              "explicit_current": True}
             无版本号时 version_label=""，base_title 为去掉序号前缀的文件名主干
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = VERSION_RE.search(stem)
    if not m:
        # 「员工手册_现行最新.txt」这类只标了现行、没写版本号的，标记要认出来，
        # 否则它会被当成无标记文档，跟 V2.0 比版本号时白白输掉
        explicit = bool(EXPLICIT_MARK_RE.search(stem))
        return {
            "base_title": _normalize_title(PREFIX_RE.sub('', stem).strip('_ ')),
            "version_label": "",
            "explicit_current": explicit,
        }
    version = m.group(1)
    explicit = bool(m.group(3))
    # m.start() 指向 "_Vx.y" 的下划线处，"_现行最新" 位于 m.end() 之后，
    # 因此制度名始终是 stem[:m.start()]（此前对 explicit 额外截 5 字符是错误的）
    base_title = _normalize_title(PREFIX_RE.sub('', stem[:m.start()]).strip('_ '))
    return {"base_title": base_title, "version_label": version, "explicit_current": explicit}


def _version_key(v: str) -> Tuple[int, ...]:
    """'2.1' -> (2, 1)；空版本号视为最小"""
    if not v:
        return (0,)
    try:
        return tuple(int(x) for x in v.split('.'))
    except ValueError:
        return (0,)


def _is_candidate(info: Dict) -> bool:
    """参与现行版竞选的文档：未删除，或仅因版本更替而软删除"""
    if not info.get('is_deleted'):
        return True
    return info.get('deleted_reason') == 'superseded'


def resolve_current_map(registry_docs: Dict[str, Dict], warnings: Optional[List[Dict]] = None) -> Dict[str, str]:
    """
    计算每个制度名的现行版 doc_id

    竞选规则（按优先级）：
      1. 同制度只有 1 个版本 → 现行
      2. 有「_现行最新」显式标记 → 在标记的里面选，**但若存在版本号更高的未标记文档，
         改判版本号高者并记一条 warnings**。理由：版本号是客观事实，标记是人手写的、
         容易忘记更新；旧的「标记优先」会让刚上传的新版被静默判成旧版。
      3. 无标记 → 取最高版本号；版本号相同（或都没有）时取 updated_at 最新的

    :param registry_docs: document_registry.json 的 documents 字典
    :param warnings: 传入 list 时，冲突会追加进去（供 CLI / 上传接口提示用户）
    :return: {base_title: current_doc_id}
    """
    groups: Dict[str, List[str]] = {}
    for doc_id, info in registry_docs.items():
        if not _is_candidate(info):
            continue
        meta = parse_doc_meta(info.get('filename', ''))
        groups.setdefault(meta['base_title'], []).append(doc_id)

    def rank(doc_id: str):
        """版本号优先，同版本号时用更新时间兜底"""
        info = registry_docs[doc_id]
        meta = parse_doc_meta(info.get('filename', ''))
        stamp = info.get('updated_at') or info.get('created_at') or ''
        return (_version_key(meta['version_label']), stamp)

    current_map = {}
    for base_title, ids in groups.items():
        if len(ids) == 1:
            current_map[base_title] = ids[0]
            continue

        explicit = [d for d in ids if parse_doc_meta(registry_docs[d]['filename'])['explicit_current']]
        pool = explicit if explicit else ids
        winner = max(pool, key=rank)

        # 冲突：有人标了「现行最新」，但同时存在版本号更高的文档。
        # 只在两侧版本号都能比较时才推翻标记——若标记文档本身没写版本号
        # （如「员工手册_现行最新.txt」），无法判断谁更新，尊重人的标记。
        highest = max(ids, key=rank)
        winner_v, highest_v = rank(winner)[0], rank(highest)[0]
        if highest != winner and winner_v != (0,) and highest_v > winner_v:
            if warnings is not None:
                warnings.append({
                    'base_title': base_title,
                    'marked_current': registry_docs[winner].get('filename'),
                    'higher_version': registry_docs[highest].get('filename'),
                    'resolution': 'higher_version_wins',
                })
            winner = highest

        current_map[base_title] = winner
    return current_map


def enforce_version_policy(registry_docs: Dict[str, Dict], now: str = None) -> Dict:
    """
    执行版本治理（原地修改 registry_docs，幂等）

    返回报告：
    {
        "current_marked": [...],       # 标记为现行的 doc_id
        "newly_superseded": [...],     # 本次新退场的旧版（需要清 Milvus 向量）
        "already_superseded": [...],   # 之前已退场的旧版（Milvus 向量也应存在清理兜底）
        "restored": [...],             # 因恢复而被重新立为现行的 doc_id
    }
    """
    now = now or datetime.now().isoformat()
    conflicts: List[Dict] = []
    current_map = resolve_current_map(registry_docs, warnings=conflicts)
    current_ids = set(current_map.values())

    report = {"current_marked": [], "newly_superseded": [],
              "already_superseded": [], "restored": [], "conflicts": conflicts}

    for doc_id, info in registry_docs.items():
        meta = parse_doc_meta(info.get('filename', ''))
        info['base_title'] = meta['base_title']
        info['version_label'] = meta['version_label']

        if doc_id in current_ids:
            # 现行版：若之前因版本更替被退场（如用户恢复了它），自动复活
            if info.get('is_deleted') and info.get('deleted_reason') == 'superseded':
                info['is_deleted'] = False
                info.pop('deleted_at', None)
                info.pop('deleted_reason', None)
                info.pop('superseded_by', None)
                info['updated_at'] = now
                report['restored'].append(doc_id)
            info['is_current'] = True
            report['current_marked'].append(doc_id)
        else:
            info['is_current'] = False
            if not _is_candidate(info):
                continue  # 用户手动删除的文档：不触碰
            if not info.get('is_deleted'):
                # 本次新退场
                info['is_deleted'] = True
                info['deleted_at'] = now
                info['updated_at'] = now
                report['newly_superseded'].append(doc_id)
            else:
                report['already_superseded'].append(doc_id)
            info['deleted_reason'] = 'superseded'
            info['superseded_by'] = current_map.get(meta['base_title'], '')

    return report


def cleanup_milvus_vectors(vector_index, doc_ids: List[str]) -> int:
    """
    清理 Milvus 中被退场文档的向量（幂等，删不存在的 id 也安全）
    :param vector_index: VectorIndex 实例（需有 delete_by_doc_id 方法）
    :param doc_ids: 被退场的文档 ID 列表
    :return: 成功删除的文档数
    """
    ok = 0
    for doc_id in doc_ids:
        try:
            vector_index.delete_by_doc_id(doc_id)
            ok += 1
        except Exception as e:
            print(f"  [WARN] Milvus 清理失败 {doc_id}: {e}")
    return ok


# ============================================================
# registry 备份（自动治理与手动脚本共用）
# ============================================================
BACKUP_KEEP = 5

def list_registry_backups(registry_path: str) -> List[str]:
    """按时间戳升序列出 registry 备份（文件名 YYYYmmdd_HHMMSS，字典序即时间序）"""
    d = os.path.dirname(registry_path) or '.'
    prefix = os.path.basename(registry_path) + '.bak_'
    try:
        return sorted(f for f in os.listdir(d) if f.startswith(prefix))
    except OSError:
        return []


def prune_registry_backups(registry_path: str, keep: int = BACKUP_KEEP,
                           verbose: bool = False, dry_run: bool = False) -> List[str]:
    """只保留最近 keep 份备份，返回被删除（或 dry_run 时将被删除）的文件名列表"""
    files = list_registry_backups(registry_path)
    stale = files[:-keep] if keep > 0 else files
    if keep <= 0:
        stale = []
    if not stale:
        if verbose:
            print(f"[备份] 现有 {len(files)} 份，未超过 {keep} 份上限，无需清理")
        return []

    if verbose:
        print(f"[备份] 现有 {len(files)} 份，将删除最早的 {len(stale)} 份（保留最近 {keep} 份）:")
        for f in stale:
            print(f"       - {f}")

    if dry_run:
        if verbose:
            print("       加 --apply 执行删除")
        return stale

    removed = []
    d = os.path.dirname(registry_path) or '.'
    for f in stale:
        try:
            os.remove(os.path.join(d, f))
            removed.append(f)
        except OSError as e:
            print(f"[WARN] 删除失败 {f}: {e}")
    if removed:
        print(f"[备份] 已清理 {len(removed)} 份旧备份，保留最近 {keep} 份")
    return removed


def backup_registry(registry_path: str, keep: int = BACKUP_KEEP,
                    verbose: bool = False) -> Optional[str]:
    """改写 registry 前留一份快照，并顺带清理旧备份。

    上传触发的自动治理与 apply_version_policy.py 都走这里，
    保证任何写入路径都有回滚点（误判版本号时能救回来）。
    :return: 备份文件路径；失败或源文件不存在时返回 None
    """
    if not os.path.exists(registry_path):
        return None
    backup = f"{registry_path}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copy2(registry_path, backup)
    except OSError as e:
        print(f"[WARN] registry 备份失败: {e}")
        return None
    prune_registry_backups(registry_path, keep=keep, verbose=verbose)
    return backup


# ============================================================
# 展示名（引用、文档列表、Gradio 三处共用）
# ============================================================
# 早期入库记录里 base_title 可能带着扩展名或重复上传加的时间戳后缀，
# 例如「示例智能2026年三季度经营快报_1792565911.pdf」。
# 这类脏值不该让用户看见，但也不值得改 registry（会牵动 hash 追踪），
# 所以在读取时统一洗一遍。
_DISPLAY_EXT_RE = re.compile(r'\.(pdf|docx?|xlsx?|pptx?|txt|md|csv|html?)$', re.I)
# 同名重复上传时会加 `_<int(time.time())>`，即 10 位秒级时间戳。
# 只认 9~11 位：`报销制度_20260101` 这类日期后缀（8 位）是名称的一部分，不能剥。
_DUP_SUFFIX_RE = re.compile(r'[_\-—]\d{9,11}$')


def build_display_name(base_title: str = '', version_label: str = '',
                       fallback: str = '') -> str:
    """把内部字段拼成给人看的名字。

    规则只有这一份：引用、文档列表、Gradio 都调它，避免同一份文档在
    不同界面显示成不同名字（曾出现：回答里是「三季度经营快报」，
    文档管理页却是「..._1792565911.pdf」）。

    :param base_title: 注册表里的制度名（可能带脏后缀）
    :param version_label: 版本号，如 "2.1" / "V2.1"，空表示无版本概念
    :param fallback: 兜底文件名（base_title 为空时使用）
    """
    def _clean(value: str) -> str:
        value = _DISPLAY_EXT_RE.sub('', (value or '').strip())
        value = _DUP_SUFFIX_RE.sub('', value).strip(' _-—')
        return _normalize_title(value)

    name = _clean(base_title)
    if not name:
        name = _clean(os.path.splitext(os.path.basename(fallback or ''))[0])

    ver = str(version_label or '').strip()
    if name and ver:
        name = f"{name} {ver if ver[:1] in ('V', 'v') else 'V' + ver}"
    return name
