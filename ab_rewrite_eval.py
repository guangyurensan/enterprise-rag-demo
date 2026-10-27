# -*- coding: utf-8 -*-
"""
Query 改写链路 A/B 评测
==========================================================
目的：量化 retrieve_with_rewrite（线上路径，含 Agently 改写）相对 retrieve_hybrid
      （评测路径，无改写）的真实收益与代价。

指标：Recall@5 / MRR / 命中率 / 平均延迟 / 改写触发率 / 空子查询率
用法：python ab_rewrite_eval.py [--limit N]
"""

import os
import sys
import json
import time
import logging
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'surgical_qa'))


class RewriteRetrieverProxy:
    """把 retrieve_hybrid 透明替换为 retrieve_with_rewrite，供 EvalRunner 复用"""

    def __init__(self, retriever):
        self._r = retriever
        self.doc_manager = retriever.doc_manager
        self.records = retriever.records
        self.infos = {}

    def retrieve_hybrid(self, query, top_k=5, user_ctx=None, eval_mode=True, **kw):
        results, info = self._r.retrieve_with_rewrite(
            query, conversation_history="", top_k=top_k, user_ctx=user_ctx
        )
        self.infos[query] = info
        return results


def summarize(report, label):
    m = report.get("metrics", {})
    return {
        "label": label,
        "count": report.get("total", 0),
        "recall@5": m.get("avg_recall@k", 0),
        "mrr": m.get("avg_mrr", 0),
        "hit_rate": m.get("hit_rate", 0),
        "keyword_hit": m.get("avg_keyword_hit", 0),
        "latency_ms": m.get("avg_latency_ms", 0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0=全部")
    ap.add_argument("--set", default="seed_persona")
    args = ap.parse_args()

    logging.disable(logging.INFO)

    from evaluator import EvalSetManager, EvalRunner
    from document_manager import DocumentManager
    from data_loader import load_all_records
    from retriever import KnowledgeRetriever

    items = EvalSetManager().load_set(args.set)
    items = [i for i in items if not i.get("expect_refusal")]
    if args.limit:
        items = items[:args.limit]
    print(f"[1/3] 可答题 {len(items)} 条")

    dm = DocumentManager()
    records = load_all_records(doc_manager=dm)
    retriever = KnowledgeRetriever(records, use_vector_index=True, doc_manager=dm)
    print(f"[2/3] 索引就绪，records={len(records)}")

    print("[3/3] A 组：无改写（评测路径）")
    base_report = EvalRunner(retriever).run_eval(items, top_k=5)

    print("      B 组：带改写（线上路径）")
    proxy = RewriteRetrieverProxy(retriever)
    rw_report = EvalRunner(proxy).run_eval(items, top_k=5)

    # 改写行为统计
    infos = proxy.infos
    triggered = [q for q, i in infos.items()
                 if i.get("rewrite_type") not in ("无需改写", "跳过改写", "空查询", "", None)
                 or len(i.get("sub_queries", [])) > 1]
    empty_sub = [q for q, i in infos.items()
                 if any(not s.strip() for s in i.get("sub_queries", []))]
    multi = [q for q, i in infos.items() if len(i.get("sub_queries", [])) > 1]

    # 逐条差异
    base_by_id = {d["id"]: d for d in base_report.get("details", [])}
    diffs = []
    for d in rw_report.get("details", []):
        b = base_by_id.get(d["id"], {})
        delta = round(d["recall@k"] - b.get("recall@k", 0), 3)
        if delta != 0 or d["hit_any"] != b.get("hit_any"):
            diffs.append({
                "id": d["id"],
                "query": d["query"][:40],
                "base_recall": b.get("recall@k"),
                "rw_recall": d["recall@k"],
                "delta": delta,
                "sub_queries": infos.get(d["query"], {}).get("sub_queries"),
            })

    out = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "eval_set": args.set,
        "baseline": summarize(base_report, "retrieve_hybrid（无改写）"),
        "rewrite": summarize(rw_report, "retrieve_with_rewrite（含 Agently 改写）"),
        "rewrite_behavior": {
            "触发改写": len(triggered),
            "多子查询": len(multi),
            "含空子查询": len(empty_sub),
            "总条数": len(infos),
        },
        "diffs": diffs,
    }

    os.makedirs(os.path.join(BASE_DIR, "surgical_qa", "eval_reports"), exist_ok=True)
    path = os.path.join(BASE_DIR, "surgical_qa", "eval_reports",
                        f"ab_rewrite_{out['timestamp']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 66)
    print(f"{'指标':<14}{'无改写':>12}{'带改写':>12}{'差异':>12}")
    print("-" * 66)
    for k in ("recall@5", "mrr", "hit_rate", "keyword_hit", "latency_ms"):
        b, r = out["baseline"][k], out["rewrite"][k]
        print(f"{k:<14}{b:>12}{r:>12}{round(r - b, 4):>12}")
    print("-" * 66)
    print(f"改写触发 {len(triggered)}/{len(infos)}，多子查询 {len(multi)}，"
          f"含空子查询 {len(empty_sub)}")
    print(f"逐条差异 {len(diffs)} 条")
    for d in diffs[:10]:
        print(f"  {d['id']} recall {d['base_recall']}→{d['rw_recall']} "
              f"sub={d['sub_queries']} | {d['query']}")
    print(f"\n报告: {path}")


if __name__ == "__main__":
    main()
