# -*- coding: utf-8 -*-
"""
BM25 融合调优实验：α 权重扫描 + 向量主/兜底通道 + 最优配置 rerank

背景（六点十六）：等权 RRF（α=0.5）下 BM25 无净收益（hit@5 76.7% vs
纯向量 79.0%）。本实验回答：α 调到多少能兑现 BM25 收益？兜底通道
（低置信时才听 BM25 的）是否优于加权融合？

效率设计：每条查询只算一次向量池(20) + BM25 池(20)，
所有 α 变体在内存里做排名算术；仅对最优配置跑 rerank。

用法:
    python eval_alpha_sweep.py --limit 300
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
ALPHAS = [0.3, 0.5, 0.7, 0.85, 0.95]          # α = 向量在 RRF 中的权重
FALLBACK_TAUS = [0.45, 0.55, 0.65]            # 向量 top1 相似度低于 τ → 触发兜底


def rrf_fuse(vector_hits, bm25_hits, k=60, alpha=0.5):
    """与生产 HybridSearcher 同款 RRF（alpha=向量权重）"""
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
    ap.add_argument("--pool", type=int, default=20)
    ap.add_argument("--rerank-pool", type=int, default=30)
    args = ap.parse_args()

    from pymilvus import MilvusClient
    from vector_index import get_client, get_embeddings, MILVUS_URI
    from hybrid_search import BM25Index
    from reranker import DashScopeReranker

    qas = [json.loads(l) for l in open(args.qa, encoding="utf-8") if l.strip()]
    step = max(1, len(qas) // args.limit)
    qas = qas[::step][:args.limit]
    n = len(qas)
    print(f"[α 扫描] {n} 条 QA | top-{args.topk} | 池 {args.pool}×2")

    client = MilvusClient(uri=MILVUS_URI)
    client.delete(collection_name=STRESS_COLLECTION, filter='doc_name == "stress_sample"')
    rows, offset = [], 0
    while True:
        batch = client.query(collection_name=STRESS_COLLECTION, filter='source == "stress"',
                             output_fields=["doc_name", "answer", "parent_title", "id"],
                             limit=1000, offset=offset)
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if offset >= 10000:
            break
    row_idx_by_id = {r["id"]: i for i, r in enumerate(rows)}
    print(f"[数据] stress 块 {len(rows)} 条")

    bm25 = BM25Index()
    bm25.build([{"question": "", "full_text": r["answer"]} for r in rows])
    reranker = DashScopeReranker()

    questions = [qa["q"] for qa in qas]
    vecs = get_embeddings(get_client(), questions)
    print(f"[数据] 问题嵌入完成 {n} 条")

    arms = {f"rrf_a{a}": Counter() for a in ALPHAS}
    arms.update({f"fallback_t{t}": Counter() for t in FALLBACK_TAUS})
    arms["baseline_pure_vector"] = Counter()

    fused_cache = {}          # (i, alpha) -> 融合后 idx 列表（给 rerank 复用）
    fallback_count = {t: 0 for t in FALLBACK_TAUS}
    t0 = time.time()

    for i, (qa, vec) in enumerate(zip(qas, vecs)):
        evidence = os.path.basename(qa["evidence"])
        res_pool = client.search(collection_name=STRESS_COLLECTION, data=[vec],
                                 limit=args.pool, output_fields=["doc_name"],
                                 search_params={"metric_type": "COSINE"})[0]
        v_hits = [(row_idx_by_id[r["id"]], r["distance"]) for r in res_pool
                  if r["id"] in row_idx_by_id]
        b_hits = bm25.search(qa["q"], top_k=args.pool)
        v_docs = [rows[idx]["doc_name"] for idx, _ in v_hits]
        b_docs = [rows[idx]["doc_name"] for idx, _ in b_hits]

        for k in range(1, args.topk + 1):
            if evidence in v_docs[:k]:
                arms["baseline_pure_vector"][k] += 1

        for a in ALPHAS:
            fused = rrf_fuse(v_hits, b_hits, alpha=a)
            fused_cache[(i, a)] = fused
            docs = [rows[idx]["doc_name"] for idx in fused[:args.topk]]
            for k in range(1, args.topk + 1):
                if evidence in docs[:k]:
                    arms[f"rrf_a{a}"][k] += 1

        # 兜底通道：向量 top1 相似度 < τ → 用 BM25 top-k 顶替
        top1_sim = v_hits[0][1] if v_hits else 0.0
        for t in FALLBACK_TAUS:
            docs = v_docs if top1_sim >= t else b_docs
            if top1_sim < t:
                fallback_count[t] += 1
            for k in range(1, args.topk + 1):
                if evidence in docs[:k]:
                    arms[f"fallback_t{t}"][k] += 1

        if (i + 1) % 100 == 0:
            print(f"  进度 {i + 1}/{n} | {time.time() - t0:.0f}s")

    # ---- 最优 α + rerank ----
    best_alpha = max(ALPHAS, key=lambda a: arms[f"rrf_a{a}"][args.topk])
    print(f"\n[最优 α] {best_alpha} → 对其融合池跑 rerank")
    rerank_hits = Counter()
    for i, qa in enumerate(qas):
        evidence = os.path.basename(qa["evidence"])
        fused = fused_cache[(i, best_alpha)][:args.rerank_pool]
        cands = [{"content": rows[idx]["answer"], "doc_name": rows[idx]["doc_name"],
                  "parent_title": rows[idx].get("parent_title", ""), "index": idx}
                 for idx in fused if idx < len(rows)]
        reranked = reranker.rerank(qa["q"], cands, top_n=args.topk)
        docs = [rows[d["index"]]["doc_name"] for d in reranked if d.get("index") is not None]
        for k in range(1, args.topk + 1):
            if evidence in docs[:k]:
                rerank_hits[k] += 1
        if (i + 1) % 100 == 0:
            print(f"  rerank 进度 {i + 1}/{n}")

    # ---- 汇总 ----
    arms["best_alpha_plus_rerank"] = rerank_hits
    print(f"\n========== 调优结果（hit@1 / hit@3 / hit@5）==========")
    report = {"run_at": datetime.now().isoformat(timespec="seconds"),
              "rows": len(rows), "qa_count": n, "alphas": ALPHAS,
              "fallback_trigger_count": fallback_count, "arms": {}}
    order = (["baseline_pure_vector"] + [f"rrf_a{a}" for a in ALPHAS]
             + [f"fallback_t{t}" for t in FALLBACK_TAUS] + ["best_alpha_plus_rerank"])
    print(f"{'策略':<26}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}")
    for arm in order:
        h = arms[arm]
        report["arms"][arm] = {f"hit@{k}": round(h[k] / n, 4) for k in (1, 3, 5)}
        print(f"{arm:<26}{h[1]/n:>8.1%}{h[3]/n:>8.1%}{h[5]/n:>8.1%}")

    out = os.path.join("eval_reports", f"alpha_sweep_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
