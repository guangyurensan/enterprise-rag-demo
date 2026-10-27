# -*- coding: utf-8 -*-
"""
基线评测执行器（完整检索主干：向量+BM25+RRF+Rerank）
====================================================

前置条件：
1. Milvus 已启动（docker ps 检查）
2. 已执行过: python apply_version_policy.py --apply --cleanup（旧版向量已清）
3. 环境变量已设置: DEEPSEEK_API_KEY（rerank 用 DashScope key，按 reranker.py 要求配置）

用法：
    python run_baseline.py              # 对 seed_v2 跑基线，报告存 eval_reports/
    python run_baseline.py --set seed   # 对旧 seed 跑（不建议，expected 全空指标失真）

产出：
    - 终端打印 Markdown 报告
    - surgical_qa/eval_reports/eval_report_<时间戳>.json
"""

import os
import sys
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'surgical_qa'))


def main():
    set_name = 'seed_v2' if '--set' not in sys.argv else sys.argv[sys.argv.index('--set') + 1]

    if not os.environ.get('DEEPSEEK_API_KEY'):
        print("[ERROR] 请先设置环境变量 DEEPSEEK_API_KEY")
        print("  Windows CMD : set DEEPSEEK_API_KEY=sk-xxx")
        print("  PowerShell  : $env:DEEPSEEK_API_KEY=\"sk-xxx\"")
        sys.exit(1)

    print("[1/4] 加载评测集:", set_name)
    from evaluator import EvalSetManager
    items = EvalSetManager().load_set(set_name)
    if not items:
        print(f"[ERROR] 评测集 {set_name} 为空或不存在")
        sys.exit(1)
    n_bound = sum(1 for i in items if i.get('expected_base_titles') or i.get('expected_doc_ids'))
    print(f"      共 {len(items)} 条，其中 {n_bound} 条绑定期望（未绑定期望的条目 recall 恒为 1.0，无诊断价值）")

    print("[2/4] 构建完整检索主干（向量+BM25+RRF+Rerank，首次启动较慢）...")
    from document_manager import DocumentManager
    from data_loader import load_all_records
    from retriever import KnowledgeRetriever

    dm = DocumentManager()
    records = load_all_records(doc_manager=dm)
    retriever = KnowledgeRetriever(records, use_vector_index=True, doc_manager=dm)

    print("[3/4] 执行评测（每条查询走真实主干，含 rerank 网络调用，请耐心等待）...")
    from evaluator import EvalRunner, save_report, format_report_md
    runner = EvalRunner(retriever)
    report = runner.run_eval(items, top_k=5)

    print("[4/4] 保存报告")
    path = save_report(report, name=f"baseline_{set_name}")
    print()
    print(format_report_md(report))
    print(f"\n报告已保存: {path}")
    print("\n建议：把本次各 category 的指标存为基线，之后每次调整（换embedding/分块/rerank参数）后重跑对比。")


if __name__ == '__main__':
    main()
