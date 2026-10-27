# -*- coding: utf-8 -*-
"""
索引重建：按章节重切 + 强制重新解析 + 重建向量
================================================

背景（实测问题）：
- PDF 表格被切成十几字的碎片（年报 113 块、平均 30 字），在向量与 rerank 里因短而虚高，
  挤占真正制度条款的位置
- 制度文本按 500 字硬切，一条完整规定常被切在块边界上（如"不报销跨城网约车费用"）

本脚本强制重新解析，不复用 parsed_cache —— 旧缓存里存的是碎片化结果，
复用会把问题原样带回来。

用法：
    python rebuild_index.py --dry-run                # 只打印改前改后对照，不写任何数据
    python rebuild_index.py --apply                  # 真实重建（覆盖缓存 + 重建 Milvus）
    python rebuild_index.py --dry-run --only 差旅     # 只处理文件名含关键词的文档
"""

import os
import sys
import time
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'surgical_qa'))


def collect_docs(dm, only=None, limit=0):
    """收集待重建的现行文档（旧版本已由版本治理挡在检索之外，无需重建）"""
    docs = []
    for doc_id, info in dm.registry.documents.items():
        if info.get('is_deleted') or not info.get('is_current'):
            continue
        path = info.get('file_path', '')
        if not path or not os.path.exists(path):
            print(f"  [跳过] 源文件不存在: {info.get('filename', '')}")
            continue
        if only and only not in info.get('filename', ''):
            continue
        docs.append((doc_id, info))
    docs.sort(key=lambda x: x[1].get('filename', ''))
    return docs[:limit] if limit else docs


def stats(chunks):
    """(块数, 平均字数, 最短字数, 标题覆盖率)"""
    if not chunks:
        return 0, 0, 0, 0.0
    lens = [len(c.content) for c in chunks]
    titled = sum(1 for c in chunks if c.title)
    return len(chunks), sum(lens) / len(lens), min(lens), titled / len(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='真实写入（默认干跑）')
    ap.add_argument('--dry-run', action='store_true', help='干跑：只打印对照（默认行为，写出来更明确）')
    ap.add_argument('--only', default=None, help='只处理文件名含该关键词的文档')
    ap.add_argument('--limit', type=int, default=0, help='最多处理 N 个文档')
    args = ap.parse_args()

    from document_manager import DocumentManager
    from document_parser import MultiPipelineDispatcher

    dm = DocumentManager()
    docs = collect_docs(dm, args.only, args.limit)

    print('=' * 92)
    print(f"{'索引重建' if args.apply else '干跑（不写任何数据）'}  |  待处理 {len(docs)} 个现行文档")
    print('=' * 92)
    print(f"{'文档':<38}{'改前块':>7}{'改后块':>7}{'平均字':>7}{'最短':>6}{'标题覆盖':>9}{'耗时':>8}")
    print('-' * 92)

    dispatcher = MultiPipelineDispatcher(default_chunk_size=500, default_chunk_overlap=50)
    done, failed = [], []
    old_total = 0

    for doc_id, info in docs:
        fname = info.get('filename', '')
        t0 = time.time()
        parsed = dispatcher.parse(info['file_path'])
        cost = time.time() - t0

        if parsed.status != 'success' or not parsed.chunks:
            print(f"{fname[:37]:<38}{'-':>7}{'失败':>7}  {parsed.error_msg[:40]}")
            failed.append(fname)
            continue

        # doc_id 由「文件路径 + mtime」生成，文件被移动或覆盖后会漂移。
        # 必须对齐 registry 的 doc_id，否则新结果会写到另一个缓存文件，
        # 检索时读到的还是旧的分块结果。
        if parsed.doc_id != doc_id:
            parsed.doc_id = doc_id
            for c in parsed.chunks:
                c.doc_id = doc_id
                c.chunk_id = f"{doc_id}_chunk_{c.chunk_index:05d}"

        old = dm._load_parsed_doc(doc_id)
        o_n, _, _, _ = stats(old.chunks if old else [])
        n_n, n_avg, n_min, n_title = stats(parsed.chunks)
        old_total += o_n
        print(f"{fname[:37]:<38}{o_n:>7}{n_n:>7}{n_avg:>7.0f}{n_min:>6}"
              f"{n_title:>9.0%}{cost:>7.1f}s")
        done.append((doc_id, info, parsed))

    if not done:
        print('\n没有可重建的文档。')
        return

    new_total = sum(len(p.chunks) for _, _, p in done)
    print('-' * 92)
    print(f"合计: {old_total} 块 → {new_total} 块   "
          f"（成功 {len(done)} 个，失败 {len(failed)} 个）")

    if not args.apply:
        print('\n干跑结束，未写入任何数据。确认切块符合预期后，加 --apply 执行。')
        return

    # ---- 1. 覆盖解析缓存 + 刷新 registry 元数据 ----
    for doc_id, info, parsed in done:
        dm._save_parsed_doc(parsed)
        info['chunk_count'] = len(parsed.chunks)
        info['parser_used'] = parsed.parser_used
        info['parse_status'] = parsed.status
        info['parse_time'] = round(parsed.parse_time, 3)
        info['updated_at'] = datetime.now().isoformat()
    dm.registry._save()
    print(f"\n[1/2] 解析缓存与 registry 已更新（{len(done)} 个文档）")

    # ---- 2. 重建向量索引 ----
    from data_loader import load_all_records
    from vector_index import VectorIndex

    records = load_all_records(doc_manager=dm)
    VectorIndex().build_index(records)
    print(f"[2/2] 向量索引已重建: {len(records)} chunks")

    print("\n重建完成。BM25 为内存索引，检索器初始化时自动按新 chunks 构建。")
    print("请重跑两套评测确认不退化：")
    print("  python run_baseline.py")
    print("  python run_persona_eval.py")


if __name__ == '__main__':
    main()
