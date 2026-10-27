# -*- coding: utf-8 -*-
"""
规模压测：解析/清洗/分块吞吐（不落主库）

隔离原则——主库零接触：
- 直接调 DataPipeline（解析→清洗→分块→去重），不写 document registry
- 向量阶段（--vector）写独立 collection "stress_chunks"，不碰 knowledge_chunks
- 用途：验证语料扩容后管线的吞吐与失败率，产出压测报告

用法:
    python run_stress_ingest.py --dir stress_corpus/synthetic --limit 60
    python run_stress_ingest.py --dir stress_corpus/cail/docs --vector
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "surgical_qa"))

from pipeline import DataPipeline  # noqa: E402

STRESS_COLLECTION = "stress_chunks"
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_reports")


def collect_files(root: str, exts=(".docx", ".xlsx", ".csv", ".md", ".txt", ".pdf")):
    out = []
    for r, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                out.append(os.path.join(r, fn))
    return sorted(out)


def stage1_pipeline(files, n_segments=4):
    """阶段1：管线吞吐（解析→清洗→分块→去重），按文件顺序分 n_segments 段统计"""
    dp = DataPipeline()
    stats = Counter()
    chunk_total, chunk_hist, elapsed_hist = 0, [], []
    failures, all_chunks = [], []
    seg_docs = [0] * n_segments
    seg_chunks = [0] * n_segments
    seg_time = [0.0] * n_segments
    n = len(files)
    t0 = time.time()

    for i, fp in enumerate(files, 1):
        seg = min((i - 1) * n_segments // n, n_segments - 1)
        ts = time.time()
        try:
            res = dp.process(fp)
            stats[res.status] += 1
            seg_time[seg] += res.elapsed_time
            if res.status == "success":
                chunk_total += len(res.chunks)
                chunk_hist.append(len(res.chunks))
                elapsed_hist.append(res.elapsed_time)
                all_chunks.extend((os.path.basename(fp), c) for c in res.chunks)
                seg_docs[seg] += 1
                seg_chunks[seg] += len(res.chunks)
        except Exception as e:  # 管线自身未捕获的异常也计入
            stats["exception"] += 1
            failures.append((os.path.basename(fp), f"exception: {e}"))
            seg_time[seg] += time.time() - ts
        if i % 100 == 0 or i == n:
            dt = time.time() - t0
            print(f"  进度 {i}/{n} | 成功 {stats['success']} 失败 {stats['failed']} "
                  f"异常 {stats['exception']} | 累计 {dt:.1f}s ({i / dt:.1f} docs/s)")
    segments = [{"segment": f"{k + 1}/{n_segments}",
                 "docs": seg_docs[k], "chunks": seg_chunks[k],
                 "docs_per_sec": round(seg_docs[k] / seg_time[k], 2) if seg_time[k] else 0}
                for k in range(n_segments)]
    return stats, chunk_total, chunk_hist, elapsed_hist, failures, time.time() - t0, all_chunks, segments


def stage2_vector(chunks_sample, max_chunks=300):
    """阶段2：嵌入 + 插入独立 collection（限流抽样，控制成本）"""
    from vector_index import VectorIndex, get_client, get_embeddings

    vi = VectorIndex(collection_name=STRESS_COLLECTION)
    vi._ensure_collection()          # 独立 collection 首次创建（不碰主库 knowledge_chunks）
    client = get_client()
    sample = chunks_sample[:max_chunks]
    texts = [c.content for _, c in sample]
    t0 = time.time()
    vecs = get_embeddings(client, texts)
    dt = time.time() - t0
    rows = [{
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "doc_name": fname,           # 原始文件名——召回评测命中判定用
        "chunk_index": c.chunk_index,
        "chunk_type": c.chunk_type,
        "parent_title": c.parent_title[:512],
        "answer": c.content[:60000],
        "department": "压测",
        "question": "",
        "source": "stress",
        "page_number": c.page_number,
        "truncated": False,
        "access_level": "internal",
        "allowed_groups": "all",
        "is_deleted": False,
        "version": 1,
        "embedding": v,
    } for fname, c, v in zip([f for f, _ in sample], [c for _, c in sample], vecs)]
    t1 = time.time()
    vi.client.insert(collection_name=STRESS_COLLECTION, data=rows)
    vi.client.flush(STRESS_COLLECTION)
    dt_ins = time.time() - t1
    return len(sample), dt, dt_ins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vector", action="store_true", help="追加向量阶段（独立 collection）")
    ap.add_argument("--vector-chunks", type=int, default=300)
    args = ap.parse_args()

    files = collect_files(args.dir)
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"目录无可用文件: {args.dir}")
        return 1
    print(f"[压测] {len(files)} 份文件 | {args.dir}")
    print(f"[隔离] 管线模式：不写 registry；向量模式目标 collection: {STRESS_COLLECTION}")

    stats, chunk_total, chunk_hist, elapsed_hist, failures, total_dt, all_chunks, segments = stage1_pipeline(files)

    # ---- 阶段2（可选）：嵌入 + 插入独立 collection ----
    vec_report = {}
    if args.vector:
        try:
            n, dt_emb, dt_ins = stage2_vector(all_chunks, max_chunks=args.vector_chunks)
            vec_report = {
                "vector_collection": STRESS_COLLECTION,
                "vector_chunks": n,
                "embed_seconds": round(dt_emb, 2),
                "embed_chunks_per_sec": round(n / dt_emb, 1) if dt_emb else 0,
                "insert_seconds": round(dt_ins, 2),
            }
            print(f"[向量] {n} 块 → {STRESS_COLLECTION} | 嵌入 {dt_emb:.1f}s 插入 {dt_ins:.1f}s")
        except Exception as e:
            vec_report = {"vector_error": str(e)[:200]}
            print(f"[向量] 失败: {e}")

    # ---- 汇总 ----
    n_ok = stats["success"]
    chunk_hist.sort()
    p = lambda q: chunk_hist[int(len(chunk_hist) * q) - 1] if chunk_hist else 0
    elapsed_hist.sort()
    lat = lambda q: elapsed_hist[int(len(elapsed_hist) * q) - 1] if elapsed_hist else 0

    report = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "corpus_dir": os.path.abspath(args.dir),
        "files_total": len(files),
        "status_counter": dict(stats),
        "chunks_total": chunk_total,
        "chunks_per_doc_avg": round(chunk_total / n_ok, 1) if n_ok else 0,
        "chunk_p50": p(0.5), "chunk_p95": p(0.95),
        "latency_ms_p50": round(lat(0.5) * 1000, 1),
        "latency_ms_p95": round(lat(0.95) * 1000, 1),
        "throughput_docs_per_sec": round(len(files) / total_dt, 2) if total_dt else 0,
        "throughput_chunks_per_sec": round(chunk_total / total_dt, 1) if total_dt else 0,
        "vector": vec_report,
        "scale_segments": segments,
        "failures": failures[:20],
    }

    print("\n========== 压测结果 ==========")
    for k, v in report.items():
        if k not in ("failures",):
            print(f"{k}: {v}")
    if failures:
        print("失败样例:")
        for fn, err in failures[:5]:
            print(f"  {fn}: {err[:100]}")

    os.makedirs(REPORT_DIR, exist_ok=True)
    rp = os.path.join(REPORT_DIR, f"stress_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写: {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
