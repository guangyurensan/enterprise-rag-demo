# -*- coding: utf-8 -*-
"""
压测语料召回评测：用合成 QA 标签测向量召回命中率

前提：run_stress_ingest.py --vector 已把压测语料写入独立 collection
stress_chunks（doc_name = 原始文件名）。

判定：对每条 QA，question 嵌入后检索 top-k，若任一命中块的
doc_name == basename(qa.evidence) 则记为命中（doc 级命中）。
不经过主服务/BM25/rerank——这是纯向量通道的召回衰减基线。

用法:
    python eval_stress_recall.py --limit 300 --topk 5
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "surgical_qa"))

STRESS_COLLECTION = "stress_chunks"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="stress_corpus/synthetic/qa.jsonl")
    ap.add_argument("--limit", type=int, default=300, help="评测 QA 条数（0=全部）")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--report-dir", default="eval_reports")
    args = ap.parse_args()

    from pymilvus import MilvusClient
    from vector_index import get_client, get_embeddings, MILVUS_URI

    qas = [json.loads(l) for l in open(args.qa, encoding="utf-8") if l.strip()]
    if args.limit:
        # 等距抽样，覆盖不同文档类型
        step = max(1, len(qas) // args.limit)
        qas = qas[::step][:args.limit]
    print(f"[召回评测] {len(qas)} 条 QA | collection={STRESS_COLLECTION} | top-{args.topk}")

    client = MilvusClient(uri=MILVUS_URI)
    if not client.has_collection(STRESS_COLLECTION):
        print("stress_chunks 不存在——先跑 run_stress_ingest.py --vector")
        return 1
    n_rows = client.get_collection_stats(STRESS_COLLECTION)["row_count"]
    print(f"[召回评测] collection 共 {n_rows} 块")

    client_load = client
    questions = [qa["q"] for qa in qas]
    t0 = time.time()
    vecs = get_embeddings(get_client(), questions)
    print(f"[召回评测] 问题嵌入完成 {len(vecs)} 条，{time.time() - t0:.1f}s")

    hits_at = {k: 0 for k in range(1, args.topk + 1)}
    miss_examples = []
    t1 = time.time()
    lat = []
    for qa, vec in zip(qas, vecs):
        evidence = os.path.basename(qa["evidence"])
        ts = time.time()
        results = client_load.search(
            collection_name=STRESS_COLLECTION,
            data=[vec],
            limit=args.topk,
            output_fields=["doc_name", "chunk_type"],
            search_params={"metric_type": "COSINE"},
        )[0]
        lat.append(time.time() - ts)
        doc_names = [r["entity"]["doc_name"] for r in results]
        for k in range(1, args.topk + 1):
            if evidence in doc_names[:k]:
                hits_at[k] += 1
        if evidence not in doc_names and len(miss_examples) < 10:
            miss_examples.append({"q": qa["q"], "a": qa["a"],
                                  "evidence": evidence, "got": doc_names[:3]})

    n = len(qas)
    lat.sort()
    report = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "collection": STRESS_COLLECTION,
        "collection_rows": n_rows,
        "qa_count": n,
        "topk": args.topk,
        "hit_rate": {f"hit@{k}": round(hits_at[k] / n, 4) for k in hits_at},
        "search_latency_ms_p50": round(lat[n // 2] * 1000, 1),
        "search_latency_ms_p95": round(lat[int(n * 0.95) - 1] * 1000, 1),
        "miss_examples": miss_examples,
    }
    print("\n========== 召回评测结果 ==========")
    for k, v in report.items():
        if k != "miss_examples":
            print(f"{k}: {v}")
    if miss_examples:
        print("\n未命中样例（前 5）:")
        for m in miss_examples[:5]:
            print(f"  Q: {m['q']}\n    证据: {m['evidence']} | 实际返回: {m['got']}")

    os.makedirs(args.report_dir, exist_ok=True)
    rp = os.path.join(args.report_dir, f"recall_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写: {rp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
