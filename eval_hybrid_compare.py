# -*- coding: utf-8 -*-
"""
检索架构对比实验：同一批 QA，三档策略量化 hit@k

  A 纯向量      —— Milvus top-5
  B 混合 RRF    —— 向量 top-20 + BM25 top-20 → RRF(k=60) → top-5
  C 混合+rerank —— B 的融合候选 top-30 → DashScope gte-rerank → top-5

数据：stress_chunks collection（与生产同款 schema/分块/嵌入模型）；
组件：生产同款 BM25Index（jieba）与 DashScopeReranker——非玩具复刻。

用法:
    python eval_hybrid_compare.py --limit 300
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "surgical_qa"))

STRESS_COLLECTION = "stress_chunks"


def load_stress_rows(client):
    """拉取压测块（source=='stress'），并清掉早期无 source 的残留行"""
    client.delete(collection_name=STRESS_COLLECTION,
                  filter='doc_name == "stress_sample"')  # 首轮小批量残留（150 块）
    rows, offset = [], 0
    while True:
        batch = client.query(
            collection_name=STRESS_COLLECTION,
            filter='source == "stress"',
            output_fields=["doc_name", "answer", "parent_title", "chunk_type", "id"],
            limit=1000, offset=offset,
        )
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if offset >= 10000:
            break
    return rows


def rrf_fuse(vector_hits, bm25_hits, k=60, alpha=0.5):
    """与生产 HybridSearcher 同款 RRF 融合"""
    v_rank = {idx: r for r, (idx, _) in enumerate(vector_hits)}
    b_rank = {idx: r for r, (idx, _) in enumerate(bm25_hits)}
    scores = {}
    for idx in set(v_rank) | set(b_rank):
        scores[idx] = ((1 - alpha) / (k + b_rank.get(idx, 999))
                       + alpha / (k + v_rank.get(idx, 999)))
    return sorted(scores, key=scores.get, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="stress_corpus/synthetic/qa.jsonl")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--pool", type=int, default=20, help="每路召回池大小")
    ap.add_argument("--rerank-pool", type=int, default=30, help="rerank 候选池")
    args = ap.parse_args()

    from pymilvus import MilvusClient
    from vector_index import get_client, get_embeddings, MILVUS_URI
    from hybrid_search import BM25Index
    from reranker import DashScopeReranker

    qas = [json.loads(l) for l in open(args.qa, encoding="utf-8") if l.strip()]
    step = max(1, len(qas) // args.limit)
    qas = qas[::step][:args.limit]
    print(f"[对比实验] {len(qas)} 条 QA | top-{args.topk} | 池 {args.pool}×2 | rerank 池 {args.rerank_pool}")

    # ---- 数据准备 ----
    client = MilvusClient(uri=MILVUS_URI)
    rows = load_stress_rows(client)
    print(f"[数据] stress 块 {len(rows)} 条")
    bm25 = BM25Index()
    bm25.build([{"question": "", "full_text": r["answer"]} for r in rows])   # 生产同款 BM25
    row_idx_by_id = {r["id"]: i for i, r in enumerate(rows)}   # Milvus 自增 id → 行号
    reranker = DashScopeReranker()

    questions = [qa["q"] for qa in qas]
    vecs = get_embeddings(get_client(), questions)
    print(f"[数据] 问题嵌入完成 {len(vecs)} 条")

    # ---- 三档评测 ----
    hits = {arm: Counter() for arm in ("A_pure_vector", "B_hybrid_rrf", "C_hybrid_rerank")}
    latency = {arm: [] for arm in hits}
    miss = {"B_hybrid_rrf": [], "C_hybrid_rerank": []}

    for i, (qa, vec) in enumerate(zip(qas, vecs)):
        evidence = os.path.basename(qa["evidence"])

        # A：纯向量
        ts = time.time()
        res = client.search(collection_name=STRESS_COLLECTION, data=[vec],
                            limit=args.topk, output_fields=["doc_name"],
                            search_params={"metric_type": "COSINE"})[0]
        latency["A_pure_vector"].append(time.time() - ts)
        a_docs = [r["entity"]["doc_name"] for r in res]
        for k in range(1, args.topk + 1):
            if evidence in a_docs[:k]:
                hits["A_pure_vector"][k] += 1

        # B：混合 RRF（向量池 + BM25 池；Milvus id 必须映射回行号再融合）
        ts = time.time()
        res_pool = client.search(collection_name=STRESS_COLLECTION, data=[vec],
                                 limit=args.pool, output_fields=["doc_name"],
                                 search_params={"metric_type": "COSINE"})[0]
        v_hits = [(row_idx_by_id.get(r["id"], -1), r["distance"]) for r in res_pool]
        v_hits = [(idx, s) for idx, s in v_hits if idx >= 0]
        b_hits = bm25.search(qa["q"], top_k=args.pool)
        fused = rrf_fuse(v_hits, b_hits)[:args.topk]
        b_docs = [rows[idx]["doc_name"] for idx in fused if idx < len(rows)]
        latency["B_hybrid_rrf"].append(time.time() - ts)
        for k in range(1, args.topk + 1):
            if evidence in b_docs[:k]:
                hits["B_hybrid_rrf"][k] += 1
        if evidence not in b_docs and len(miss["B_hybrid_rrf"]) < 8:
            miss["B_hybrid_rrf"].append({"q": qa["q"], "evidence": evidence, "got": b_docs[:3]})

        # C：混合 + rerank（融合候选 top-30 → gte-rerank → top-5）
        ts = time.time()
        cand_idx = rrf_fuse(v_hits, b_hits)[:args.rerank_pool]
        cands = [{"content": rows[idx]["answer"], "doc_name": rows[idx]["doc_name"],
                  "parent_title": rows[idx]["parent_title"], "index": idx}
                 for idx in cand_idx if idx < len(rows)]
        reranked = reranker.rerank(qa["q"], cands, top_n=args.topk)
        c_docs = [rows[d["index"]]["doc_name"] for d in reranked if d.get("index") is not None]
        latency["C_hybrid_rerank"].append(time.time() - ts)
        for k in range(1, args.topk + 1):
            if evidence in c_docs[:k]:
                hits["C_hybrid_rerank"][k] += 1
        if evidence not in c_docs and len(miss["C_hybrid_rerank"]) < 8:
            miss["C_hybrid_rerank"].append({"q": qa["q"], "evidence": evidence, "got": c_docs[:3]})
        if (i + 1) % 50 == 0:
            print(f"  进度 {i + 1}/{len(qas)}")

    # ---- 汇总 ----
    n = len(qas)
    report = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "collection": STRESS_COLLECTION, "rows": len(rows),
        "qa_count": n, "topk": args.topk, "pool": args.pool,
        "arms": {},
    }
    print("\n========== 三档对比结果 ==========")
    print(f"{'策略':<18}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'p50(ms)':>10}{'p95(ms)':>10}")
    for arm in ("A_pure_vector", "B_hybrid_rrf", "C_hybrid_rerank"):
        h = hits[arm]
        ls = sorted(latency[arm])
        lat = lambda q: ls[min(int(len(ls) * q), len(ls) - 1)] * 1000
        report["arms"][arm] = {
            "hit@1": round(h[1] / n, 4), "hit@3": round(h[3] / n, 4),
            "hit@5": round(h[5] / n, 4),
            "latency_ms_p50": round(lat(0.5), 1), "latency_ms_p95": round(lat(0.95), 1),
        }
        print(f"{arm:<18}{h[1]/n:>8.1%}{h[3]/n:>8.1%}{h[5]/n:>8.1%}"
              f"{lat(0.5):>10.1f}{lat(0.95):>10.1f}")
    report["miss_examples"] = miss

    out = os.path.join("eval_reports",
                       f"hybrid_compare_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
