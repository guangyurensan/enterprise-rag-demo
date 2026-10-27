#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文档批量重命名：registry 元数据 + Milvus chunk 元数据同步。

场景：语料脱敏（如某主体名替换）后，需要让「文件名 / base_title / Milvus doc_name」
一起变更，否则评测集按 base_title 动态解析期望文档会失败——而失败是静默的
（期望列表被清空后 recall 兜底返回 1.0），会制造"永久通过"的假绿用例。

设计取舍：**只改元数据，不重新解析文档**。
chunk 与向量保持原样，因此检索行为完全不变，指标可与历史基线直接对比；
若走"删了重新入库"，分块边界可能漂移，指标变化将混入两种原因，无法归因。

用法：
    python scripts/rename_documents.py --old 旧名 --new 新名            # 预演（默认）
    python scripts/rename_documents.py --old 旧名 --new 新名 --apply    # 执行

可重复传 --old/--new 对，按顺序依次应用（例如同时处理两个主体名）。
"""
import argparse
import json
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "surgical_qa"))

REGISTRY_PATH = os.path.join(BASE_DIR, "surgical_qa", "document_store",
                             "document_registry.json")


def load_registry():
    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(data):
    tmp = REGISTRY_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, REGISTRY_PATH)


def plan(registry, pairs):
    """算出需要变更的条目（不落盘）"""
    changes = []
    for doc_id, info in registry["documents"].items():
        fname = info.get("filename", "") or ""
        btitle = info.get("base_title", "") or ""
        fpath = info.get("file_path", "") or ""

        new_fname, new_btitle, new_fpath = fname, btitle, fpath
        for old, new in pairs:
            new_fname = new_fname.replace(old, new)
            new_btitle = new_btitle.replace(old, new)
            new_fpath = new_fpath.replace(old, new)

        if (new_fname, new_btitle, new_fpath) == (fname, btitle, fpath):
            continue
        changes.append({
            "doc_id": doc_id,
            "old_filename": fname, "new_filename": new_fname,
            "old_base_title": btitle, "new_base_title": new_btitle,
            "old_path": fpath, "new_path": new_fpath,
            "exists": os.path.exists(fpath) if fpath else False,
            "level": info.get("access_level", ""),
        })
    return changes


def rename_files(changes, apply):
    """磁盘文件重命名；返回实际成功数"""
    done = 0
    for c in changes:
        if not c["exists"] or c["old_path"] == c["new_path"]:
            continue
        if apply:
            if os.path.exists(c["new_path"]):
                print(f"  [跳过] 目标已存在: {os.path.basename(c['new_path'])}")
                continue
            os.rename(c["old_path"], c["new_path"])
        done += 1
    return done


def update_milvus(changes, pairs, apply):
    """同步 Milvus doc_name（查询 → 改字段 → 删除 → 重插，auto_id 表不支持原地更新）"""
    from vector_index import VectorIndex  # noqa: WPS433 (延迟导入，仅 --apply 时需要)

    vi = VectorIndex()
    if not vi.client.has_collection(vi.collection_name):
        print("  Milvus 集合不存在，跳过")
        return {}

    report = {}
    for c in changes:
        doc_id = c["doc_id"]
        rows = vi.client.query(
            collection_name=vi.collection_name,
            filter=f'doc_id == "{doc_id}"',
            output_fields=["*"],
        )
        report[doc_id] = {"rows": len(rows), "updated": 0}
        if not rows:
            continue

        new_rows, seen = [], set()
        for r in rows:
            r.pop("id", None)
            r.pop("distance", None)
            name = r.get("doc_name", "") or ""
            for old, new in pairs:
                name = name.replace(old, new)
            r["doc_name"] = name
            cid = r.get("chunk_id", "")
            if cid and cid in seen:
                continue      # 去重防御：历史重复行不被原样继承
            seen.add(cid)
            new_rows.append(r)

        if apply:
            vi.client.delete(collection_name=vi.collection_name,
                             filter=f'doc_id == "{doc_id}"')
            vi.client.insert(collection_name=vi.collection_name, data=new_rows)
        report[doc_id]["updated"] = len(new_rows)
    return report


def main():
    ap = argparse.ArgumentParser(description="文档批量重命名（元数据）")
    ap.add_argument("--old", action="append", required=True, help="原名称（可多次）")
    ap.add_argument("--new", action="append", required=True, help="新名称（可多次）")
    ap.add_argument("--apply", action="store_true", help="真正执行（默认仅预演）")
    ap.add_argument("--skip-milvus", action="store_true", help="只改 registry 与文件")
    args = ap.parse_args()

    if len(args.old) != len(args.new):
        sys.exit("--old 与 --new 数量必须一致")
    pairs = list(zip(args.old, args.new))

    print(f"{'执行' if args.apply else '预演'}模式 | 替换规则: "
          + ", ".join(f"{o}→{n}" for o, n in pairs))

    registry = load_registry()
    before_total = registry.get("total")
    changes = plan(registry, pairs)
    print(f"受影响文档: {len(changes)} / {len(registry['documents'])}")
    for c in changes:
        mark = "有原件" if c["exists"] else "无原件(仅元数据)"
        print(f"  {c['doc_id']} [{mark}] {c['old_filename']}")
        print(f"      → {c['new_filename']}  (level={c['level']})")
        if c["old_base_title"] != c["new_base_title"]:
            print(f"      base_title: {c['old_base_title']} → {c['new_base_title']}")

    if not changes:
        print("无需变更")
        return

    n_files = rename_files(changes, args.apply)
    print(f"文件重命名: {n_files} 个{'（已执行）' if args.apply else '（待执行）'}")

    if not args.skip_milvus:
        rep = update_milvus(changes, pairs, args.apply)
        total_rows = sum(v["rows"] for v in rep.values())
        total_upd = sum(v["updated"] for v in rep.values())
        print(f"Milvus 行: {total_rows} → {total_upd}"
              f"{'（已重插）' if args.apply else '（待重插）'}")
        for doc_id, v in rep.items():
            if v["rows"] == 0:
                print(f"  [警告] {doc_id} 在 Milvus 中无 chunk")

    if args.apply:
        now = datetime.now().isoformat()
        for c in changes:
            info = registry["documents"][c["doc_id"]]
            info["filename"] = c["new_filename"]
            info["base_title"] = c["new_base_title"]
            if c["new_path"]:
                info["file_path"] = c["new_path"]
            info["updated_at"] = now
        registry["last_updated"] = now
        registry["total"] = before_total
        save_registry(registry)
        print("registry 已保存")
        print("下一步：python run_regress.py --level 1 验证指标回到基线")
    else:
        print("（预演结束，加 --apply 执行）")


if __name__ == "__main__":
    main()
