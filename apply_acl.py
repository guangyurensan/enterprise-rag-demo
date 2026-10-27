# -*- coding: utf-8 -*-
"""
存量文档密级刷新（2026-10-24 ACL 泄露修复）

背景：ACL 上线后从未定密，全部文档 internal -> 员工可见工资表（实测泄露）。
本脚本：
  1. 对 registry 中全部活跃文档按 access_control.CLASSIFY_RULES 自动定密
  2. 更新 registry（DocumentManager.update_access）
  3. 同步 Milvus chunk 行（VectorIndex.update_doc_access，查询->删除->重插）
  4. 输出定密前后对照 + Milvus 分密级统计

用法（必须在项目根目录，Python 3.10）：
    python apply_acl.py            # 预览模式，只打印不修改
    python apply_acl.py --apply    # 实际执行
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'surgical_qa'))

from access_control import classify_document  # noqa: E402


def main():
    apply = '--apply' in sys.argv

    from document_manager import DocumentManager
    from vector_index import VectorIndex

    dm = DocumentManager()
    vi = VectorIndex()  # 默认 collection knowledge_chunks（主库）

    active = {k: d for k, d in dm.registry.documents.items() if not d.get('is_deleted')}
    print(f"活跃文档 {len(active)} 份\n")

    changes, unchanged, errors = [], [], []
    for doc_id, doc in sorted(active.items(), key=lambda kv: kv[1].get('filename', '')):
        name = doc.get('filename', '')
        new_level, new_groups = classify_document(name)
        old_level = doc.get('access_level') or 'internal'
        old_groups = doc.get('allowed_groups') or 'all'

        if old_level == new_level and old_groups == new_groups:
            unchanged.append((name, new_level, new_groups))
            continue
        changes.append((doc_id, name, old_level, old_groups, new_level, new_groups))

    # ---- 打印变更表 ----
    print(f"{'文件名':<44} {'旧':<14} {'新':<14} 允许组")
    print('-' * 96)
    for _, name, ol, og, nl, ng in changes:
        print(f"{name:<44} {ol:<14} {nl:<14} {ng}")
    print('-' * 96)
    from collections import Counter
    after = Counter(l for _, _, _, _, l, _ in changes) + Counter(l for _, l, _ in unchanged)
    print(f"变更 {len(changes)} 份 | 不变 {len(unchanged)} 份 | 定密后分布: {dict(after)}\n")

    if not apply:
        print("（预览模式，未修改任何数据。加 --apply 执行）")
        return 0

    # ---- 执行 ----
    for doc_id, name, _, _, nl, ng in changes:
        try:
            ok = dm.update_access(doc_id, nl, ng)   # registry + 版本号
            n = vi.update_doc_access(doc_id, nl, ng)  # Milvus 同步
            print(f"  ✓ {name[:40]:<42} {nl:<14} Milvus chunks={n} registry={'ok' if ok else 'FAIL'}")
            if n == 0:
                print(f"    ⚠️ Milvus 中未找到该 doc_id 的 chunk（可能未入索引）")
        except Exception as e:
            errors.append((name, str(e)))
            print(f"  ✗ {name[:40]} 失败: {e}")

    if errors:
        print(f"\n{len(errors)} 份失败，请检查：")
        for n, e in errors:
            print(f"  {n}: {e}")
        return 1

    # ---- 验证：Milvus 分密级统计 ----
    print("\n=== Milvus 密级分布验证 ===")
    for level in ('public', 'internal', 'restricted', 'confidential'):
        rows = vi.client.query(
            collection_name=vi.collection_name,
            filter=f'access_level == "{level}"',
            output_fields=['count(*)'],
        )
        cnt = rows[0].get('count(*)', 0) if rows else 0
        print(f"  {level:<14} {cnt} chunks")
    return 0


if __name__ == '__main__':
    sys.exit(main())
