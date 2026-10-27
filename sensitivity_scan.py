# -*- coding: utf-8 -*-
"""
参数敏感度扫描：判断指标是不是靠某个特定数值撑起来的。

扫四个旋钮（单变量，其余保持基线值）：
  max_per_doc  1 / 2 / 3 / off      单文档配额
  min_score    0.25 / 0.30 / 0.35(基线) / 0.40 / 0.45   向量相似度下限
  top_n        4 / 6(基线) / 8       rerank 输出条数
  alpha        0.3 / 0.5(基线) / 0.7  RRF 融合权重

数据：seed_v2（26 条）+ persona 可答题（脚本自动剔除拒答与期望无法解析的用例）
只看检索侧指标，不调用生成，所以很快（约 12 分钟跑完 14 组）。

判读方法：
  - 指标在邻域内平缓 → 说明是机制，不是调出来的魔法值
  - 只有某一个值明显高、两侧都掉 → 说明踩中数据分布，需要警惕

用法：
  python sensitivity_scan.py
  python sensitivity_scan.py --only max_per_doc
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "surgical_qa"))
logging.disable(logging.INFO)

import retriever as R
from access_control import UserContext
from data_loader import load_all_records
from document_manager import DocumentManager
from evaluator import EvalMetrics
from version_policy import resolve_current_map


def load_cases():
    dm = DocumentManager()
    current = resolve_current_map(dm.registry.documents)
    cases = []

    for fname in ("seed_v2.json", "seed_persona.json"):
        items = json.load(open(os.path.join(BASE, "surgical_qa/eval_sets", fname),
                               encoding="utf-8"))["items"]
        for it in items:
            if it.get("expect_refusal"):
                continue
            ids = [current[t] for t in it.get("expected_base_titles", []) if t in current]
            if ids:
                cases.append({"q": it["query"], "ids": ids, "src": fname})
    return dm, cases


def measure(retriever, cases, ctx):
    metrics = EvalMetrics()
    recalls, mrrs, hits, lat, top_scores = [], [], [], [], []
    for c in cases:
        t = time.time()
        try:
            res = retriever.retrieve_hybrid(c["q"], top_k=5, user_ctx=ctx, eval_mode=True)
        except Exception as e:
            print(f"  [跳过] {c['q'][:20]} -> {e}")
            res = []
        lat.append((time.time() - t) * 1000)
        recalls.append(metrics.recall_at_k(res, c["ids"], k=5))
        mrrs.append(metrics.mrr(res, c["ids"]))
        hits.append(1.0 if metrics.hit_any(res, c["ids"]) else 0.0)
        if res:
            top_scores.append(res[0].get("rerank_score", res[0].get("score", 0)) or 0)
    n = max(len(cases), 1)
    return {
        "recall@5": round(sum(recalls) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "hit_rate": round(sum(hits) / n, 4),
        "latency_ms": round(sum(lat) / len(lat), 1),
        "avg_top1_score": round(sum(top_scores) / max(len(top_scores), 1), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", choices=["", "max_per_doc", "min_score", "top_n", "alpha"])
    a = ap.parse_args()

    dm, cases = load_cases()
    retriever = R.KnowledgeRetriever(load_all_records(doc_manager=dm),
                                     use_vector_index=True, doc_manager=dm)
    ctx = UserContext(role="admin", groups=["全员"])

    # 基线值快照（每次扫描前恢复，保证单变量）
    base = {
        "DIVERSITY": dict(R.DIVERSITY),
        "VECTOR_RECALL": dict(R.VECTOR_RECALL),
        "RERANK": dict(R.RERANK),
        "RRF": dict(R.RRF),
    }

    def reset():
        R.DIVERSITY = dict(base["DIVERSITY"])
        R.VECTOR_RECALL = dict(base["VECTOR_RECALL"])
        R.RERANK = dict(base["RERANK"])
        R.RRF = dict(base["RRF"])

    print(f"数据集：{len(cases)} 条（seed_v2 + persona 可答）")
    print(f"基线：max_per_doc={base['DIVERSITY'].get('max_per_doc')}, "
          f"min_score={base['VECTOR_RECALL']['min_score']}, "
          f"top_n={base['RERANK']['top_n']}, alpha={base['RRF']['alpha']}\n")

    grids = {
        "max_per_doc": [1, 2, 3, "off"],
        "min_score": [0.25, 0.30, 0.35, 0.40, 0.45],
        "top_n": [4, 6, 8],
        "alpha": [0.3, 0.5, 0.7],
    }
    dims = [a.only] if a.only else list(grids)

    results = {}
    for dim in dims:
        results[dim] = []
        print(f"===== {dim} =====")
        for v in grids[dim]:
            reset()
            if dim == "max_per_doc":
                if v == "off":
                    R.DIVERSITY = {**base["DIVERSITY"], "enabled": False}
                else:
                    R.DIVERSITY = {**base["DIVERSITY"], "enabled": True, "max_per_doc": v}
            elif dim == "min_score":
                R.VECTOR_RECALL = {**base["VECTOR_RECALL"], "min_score": v}
            elif dim == "top_n":
                R.RERANK = {**base["RERANK"], "top_n": v}
            elif dim == "alpha":
                R.RRF = {**base["RRF"], "alpha": v}

            m = measure(retriever, cases, ctx)
            m["value"] = v
            results[dim].append(m)
            print(f"  {str(v):<6} recall@5={m['recall@5']:.4f}  mrr={m['mrr']:.4f}  "
                  f"hit={m['hit_rate']:.4f}  top1={m['avg_top1_score']}  {m['latency_ms']}ms")
        print()

    reset()

    # ---- Markdown 报告
    lines = [f"# 参数敏感度扫描 {datetime.now():%Y-%m-%d %H:%M}", "",
             f"数据集：{len(cases)} 条（seed_v2 + persona 可答），top_k=5，仅检索侧指标", ""]
    for dim, rows in results.items():
        lines += [f"## {dim}", "",
                  "| 取值 | Recall@5 | MRR | hit_rate | 平均 top1 分数 | 延迟(ms) |",
                  "|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['value']} | {r['recall@5']:.4f} | {r['mrr']:.4f} | "
                         f"{r['hit_rate']:.4f} | {r['avg_top1_score']} | {r['latency_ms']} |")
        vals = [r["recall@5"] for r in rows]
        lines += ["", f"Recall@5 波动范围：{min(vals):.4f} ~ {max(vals):.4f}"
                      f"（跨度 {max(vals) - min(vals):.4f}）", ""]

    out_dir = os.path.join(BASE, "surgical_qa", "eval_reports")
    md = os.path.join(out_dir, f"sensitivity_{datetime.now():%Y%m%d_%H%M%S}.md")
    js = md.replace(".md", ".json")
    open(md, "w", encoding="utf-8").write("\n".join(lines))
    json.dump(results, open(js, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n".join(lines))
    print(f"\n报告：{md}")


if __name__ == "__main__":
    main()
