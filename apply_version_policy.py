# -*- coding: utf-8 -*-
"""
版本治理执行器（一次性修复存量数据 + 之后可随时重跑）
====================================================

用法（在项目根目录执行）：
    python apply_version_policy.py             # 预览将要发生的变化（不写入）
    python apply_version_policy.py --apply     # 真正执行：更新 registry + 软删除旧版
    python apply_version_policy.py --apply --cleanup
                                               # 同时清理 Milvus 中旧版向量（需 Milvus 运行中）
    python apply_version_policy.py --purge     # 预览：归档旧版文件 + 删解析缓存
    python apply_version_policy.py --apply --purge
                                               # 真正清理磁盘：旧版原件移入 superseded/、删缓存
    python apply_version_policy.py --prune-backups            # 预览：将清理哪些旧备份
    python apply_version_policy.py --apply --prune-backups    # 真正删除，只留最近 5 份

每次 --apply 写回 registry 前都会备份一份 `.bak_时间戳`，写回后自动只保留最近
BACKUP_KEEP（5）份；备份不参与检索，删掉不影响任何数据。

--purge 不删除 registry 记录：记录要留着，否则 superseded_by 会指向不存在的文档、
审计链断裂。原始文件移入 document_store/superseded/ 而不是直接删除，需要时可恢复。

新文档上传后的版本治理已在 retriever.py 中自动触发，本脚本用于存量修复与手动兜底。
"""

import os
import sys
import json
import shutil
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'surgical_qa'))

REGISTRY_PATH = os.path.join(BASE_DIR, 'surgical_qa', 'document_store', 'document_registry.json')
ARCHIVE_DIR = os.path.join(BASE_DIR, 'surgical_qa', 'document_store', 'superseded')
CACHE_DIR = os.path.join(BASE_DIR, 'surgical_qa', 'parsed_cache')

# 备份与清理逻辑放在 version_policy 里，上传触发的自动治理与本文脚本共用同一套
from version_policy import backup_registry, prune_registry_backups  # noqa: E402


def purge_superseded_files(docs, apply_mode=False):
    """归档旧版原件 + 删除其解析缓存。

    - 原件移入 document_store/superseded/，并在 registry 记 archived_to，供 restore 找回
    - 解析缓存直接删（可由原件重建，留着纯占空间）
    - registry 记录不动：软删除状态是审计与恢复的依据
    """
    targets = [(d, i) for d, i in docs.items()
               if i.get('is_deleted') and i.get('deleted_reason') == 'superseded']
    moved = removed = missing = 0

    print("\n" + "=" * 80)
    print(f"{'归档旧版原件':<20}{'解析缓存':<12}文件名")
    print("-" * 80)
    for doc_id, info in targets:
        src = info.get('file_path', '')
        cache = os.path.join(CACHE_DIR, f"{doc_id}.json")
        file_state = "在" if src and os.path.exists(src) else "缺失"
        cache_state = "在" if os.path.exists(cache) else "-"

        if apply_mode:
            if src and os.path.exists(src):
                os.makedirs(ARCHIVE_DIR, exist_ok=True)
                dst = os.path.join(ARCHIVE_DIR, os.path.basename(src))
                if os.path.exists(dst):
                    dst = os.path.join(ARCHIVE_DIR, f"{doc_id[:8]}_{os.path.basename(src)}")
                shutil.move(src, dst)
                info['archived_to'] = dst
                moved += 1
            else:
                missing += 1
            if os.path.exists(cache):
                os.remove(cache)
                removed += 1
        print(f"{file_state:<20}{cache_state:<12}{info.get('filename', '?')}")

    print("-" * 80)
    if apply_mode:
        print(f"原件归档 {moved} 个（→ document_store/superseded/）| 缓存删除 {removed} 个 | 原件缺失 {missing} 个")
    else:
        print(f"共 {len(targets)} 个旧版待处理（预览模式未改动，加 --apply --purge 执行）")
    return moved, removed


def main():
    args = sys.argv[1:]
    apply_mode = '--apply' in args
    cleanup = '--cleanup' in args
    purge = '--purge' in args
    prune_only = '--prune-backups' in args

    # 只清理旧备份：不读 registry、不改文档状态，可随时单独执行
    if prune_only:
        prune_registry_backups(REGISTRY_PATH, verbose=True, dry_run=not apply_mode)
        return

    with open(REGISTRY_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    docs = data['documents']

    from version_policy import enforce_version_policy, cleanup_milvus_vectors
    report = enforce_version_policy(docs)

    # ---- 预览打印 ----
    print("=" * 80)
    print(f"{'判定':<18}{'doc_id':<20}文件名")
    print("-" * 80)

    id2file = {d: docs[d].get('filename', '?') for d in docs}
    for d in report['current_marked']:
        cur_tag = " [现行]" if docs[d].get('is_current') else ""
        print(f"{'现行版':<18}{d:<20}{id2file[d]}{cur_tag}")
    for d in report['newly_superseded']:
        print(f"{'旧版退场(本次)':<18}{d:<20}{id2file[d]}  → superseded_by={docs[d].get('superseded_by','')}")
    for d in report['already_superseded']:
        print(f"{'旧版(已退场)':<18}{d:<20}{id2file[d]}")
    for d in report['restored']:
        print(f"{'恢复为现行':<18}{d:<20}{id2file[d]}")
    print("=" * 80)
    print(f"现行 {len(report['current_marked'])} | 本次新退场 {len(report['newly_superseded'])} | "
          f"已退场 {len(report['already_superseded'])} | 恢复 {len(report['restored'])}")

    if report.get('conflicts'):
        print("\n[注意] 以下文档标了「现行最新」，但同时存在版本号更高的文件。")
        print("       已按「版本号高者为准」处理——如果更高版本其实是未生效的草稿，请删掉它或补上标记：")
        for c in report['conflicts']:
            print(f"  {c['base_title']}: 标记为现行的是 {c['marked_current']}，"
                  f"但 {c['higher_version']} 版本号更高")

    if not apply_mode:
        if purge:
            purge_superseded_files(docs, apply_mode=False)
        print("\n[预览模式] 未做任何修改。确认无误后执行: python apply_version_policy.py --apply")
        return

    # ---- 磁盘清理（先做，archived_to 随 registry 一起写回）----
    if purge:
        purge_superseded_files(docs, apply_mode=True)

    # ---- 备份后写回 ----
    backup = backup_registry(REGISTRY_PATH, verbose=True)
    data['last_updated'] = datetime.now().isoformat()
    with open(REGISTRY_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] registry 已更新（备份: {os.path.basename(backup) if backup else '无'}）")

    # ---- Milvus 清理 ----
    to_purge = report['newly_superseded'] + report['already_superseded']
    if not cleanup:
        if to_purge:
            print(f"\n[提示] 有 {len(to_purge)} 个旧版文档的向量仍在 Milvus 中（字段 is_deleted 仍为 false），")
            print("       向量检索通道尚未被挡住。请带 --cleanup 重跑，或启动应用后由版本治理自动清理：")
            print("       python apply_version_policy.py --apply --cleanup")
        return

    try:
        from vector_index import VectorIndex
        vi = VectorIndex()
        if not vi.load_index():
            print("[WARN] Milvus 索引加载失败，跳过向量清理。请确认 Milvus 已启动。")
            return
        ok = cleanup_milvus_vectors(vi, to_purge)
        print(f"\n[OK] Milvus 旧版向量清理完成: {ok}/{len(to_purge)} 个文档")
    except Exception as e:
        print(f"[WARN] Milvus 连接/清理失败: {e}")
        print("       registry 已更新（BM25/内存通道已生效），Milvus 通道请在服务启动后重跑 --cleanup")


if __name__ == '__main__':
    main()
