# -*- coding: utf-8 -*-
"""
基线校准 + A/B 对比（审查报告行动项 4）
==========================================

1. min_score 阈值扫描：VECTOR_RECALL.min_score ∈ {0.25, 0.30, 0.35, 0.40, 0.45}
   用 Recall@5 / MRR / 命中率选最优阈值（当前线上值 0.35 是拍脑袋值）
2. Rerank A/B：enabled=True（基线） vs False
3. RRF alpha 对比：当前值 vs 0.5

全部走真实检索主干（向量+BM25+RRF+Rerank），评测集 seed_v2（28 条，带黄金标准）。

用法：
    python calibrate_ab.py            # 全部跑完约 6 分钟（含 rerank 网络调用）
    python calibrate_ab.py --quick    # 只跑 min_score 扫描
"""

import os
import sys
import json
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'surgical_qa'))


def run_once(retriever, items, label, overrides):
    """应用配置覆盖 → 跑评测 → 恢复配置，返回精简结果"""
    import retrieval_config as rc

    original = {}
    for path, value in overrides.items():
        section, key = path.split('.')
        original[path] = getattr(rc, section)[key]
        getattr(rc, section)[key] = value

    try:
        from evaluator import EvalRunner
        report = EvalRunner(retriever).run_eval(items, top_k=5)
    finally:
        for path, value in original.items():
            section, key = path.split('.')
            getattr(rc, section)[key] = value

    m = report['metrics']
    row = {
        'label': label,
        'recall': m['avg_recall@k'],
        'mrr': m['avg_mrr'],
        'hit_rate': m['hit_rate'],
        'latency_ms': m['avg_latency_ms'],
    }
    print(f"  {label:<28} Recall@5={row['recall']:.2%}  MRR={row['mrr']:.3f}  "
          f"命中={row['hit_rate']:.2%}  延迟={row['latency_ms']:.0f}ms")
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick', action='store_true', help='只跑 min_score 扫描')
    args = parser.parse_args()

    print('[1/3] 加载评测集与检索主干...')
    from evaluator import EvalSetManager
    items = EvalSetManager().load_set('seed_v2')

    from document_manager import DocumentManager
    from data_loader import load_all_records
    from retriever import KnowledgeRetriever

    dm = DocumentManager()
    records = load_all_records(doc_manager=dm)
    retriever = KnowledgeRetriever(records, use_vector_index=True, doc_manager=dm)

    results = []

    # ---- 1. min_score 扫描 ----
    print('\n[2/3] min_score 阈值扫描（当前线上: retrieval_config.VECTOR_RECALL.min_score）:')
    from retrieval_config import VECTOR_RECALL
    current_ms = VECTOR_RECALL['min_score']
    print(f'  （当前配置值: {current_ms}）')
    for ms in [0.25, 0.30, 0.35, 0.40, 0.45]:
        results.append(run_once(
            retriever, items, f'min_score={ms}{" (当前)" if ms == current_ms else ""}',
            {'VECTOR_RECALL.min_score': ms},
        ))

    if not args.quick:
        # ---- 2. Rerank A/B ----
        print('\n[3/3] Rerank A/B（当前 min_score 下）:')
        results.append(run_once(retriever, items, 'rerank=ON (基线)',
                                {'VECTOR_RECALL.min_score': current_ms}))
        results.append(run_once(retriever, items, 'rerank=OFF',
                                {'VECTOR_RECALL.min_score': current_ms, 'RERANK.enabled': False}))

        # ---- 3. RRF alpha 对比 ----
        from retrieval_config import RRF
        print('\nRRF alpha 对比（rerank=ON）:')
        for alpha in [RRF['alpha'], 0.5]:
            tag = f'alpha={alpha}{" (当前)" if alpha == RRF["alpha"] else ""}'
            results.append(run_once(
                retriever, items, tag,
                {'VECTOR_RECALL.min_score': current_ms, 'RRF.alpha': alpha},
            ))

    # ---- 保存 ----
    from evaluator import save_report
    out = {
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'eval_set': 'seed_v2',
        'note': '行动项4：min_score校准 + rerank A/B + alpha对比',
        'rows': results,
    }
    path = save_report(out, name='calibration_ab')
    print(f'\n结果已保存: {path}')

    # ---- 结论提示 ----
    best = max(results, key=lambda r: (r['recall'], r['mrr']))
    print(f"Recall 最高: {best['label']} ({best['recall']:.2%})")
    print("提示: min_score 过低会放进噪声（上游过滤失效），过高会截断真命中；")
    print("      选 Recall 持平区间内最高的值，并对比 rerank ON/OFF 的增益是否值得其延迟。")


if __name__ == '__main__':
    main()
