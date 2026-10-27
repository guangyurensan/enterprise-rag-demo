# -*- coding: utf-8 -*-
"""
人设化评测：模拟 6 类真实用户的提问方式 + 无答案拒答检测
==========================================================

解决的问题（单人测试的盲区）：
1. 一个人写 query 会不自觉使用文档用词 → 测不出口语化/黑话的召回损失
2. 一个人用 admin 身份提问 → 测不到权限边界（本集不覆盖，需真实多身份）
3. 一个人只问"库里有答案的" → 测不出无答案时会不会编造（拒答类专门测这个）

评测集：eval_sets/seed_persona.json
- 可答题（41 条）：走完整主干，计算 Recall@5 / MRR / 答案要点命中率，按人设分组
- 拒答题（8 条）：走检索 + LLM 生成，检查是否老实承认"没有相关信息"

用法：
    python run_persona_eval.py
"""

import os
import sys
import json
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'surgical_qa'))

# 拒答判定标记（答案中出现任一即认为"老实拒答"）
REFUSAL_MARKERS = [
    "没有找到", "未找到", "无法回答", "未提及", "没有相关", "未涉及",
    "不包含", "没有规定", "无相关规定", "暂无", "不清楚", "建议咨询",
    "没有明确", "未明确", "无法确定", "资料中没有",
]


def is_refusal(answer: str) -> bool:
    return any(m in answer for m in REFUSAL_MARKERS)


def main():
    logging.disable(logging.INFO)

    print("[1/4] 加载人设评测集...")
    from evaluator import EvalSetManager
    items = EvalSetManager().load_set('seed_persona')
    # ACL 用例（acl=true）：多身份权限回归——敏感文档不得命中 + 护栏防过度封锁
    acl_items = [i for i in items if i.get('acl')]
    answerable = [i for i in items if not i.get('expect_refusal') and not i.get('acl')]
    refusal = [i for i in items if i.get('expect_refusal')]
    print(f"      可答 {len(answerable)} 条 / 拒答 {len(refusal)} 条 / ACL 权限 {len(acl_items)} 条")

    print("[2/4] 构建检索主干...")
    from document_manager import DocumentManager
    from data_loader import load_all_records
    from retriever import KnowledgeRetriever

    dm = DocumentManager()
    records = load_all_records(doc_manager=dm)
    retriever = KnowledgeRetriever(records, use_vector_index=True, doc_manager=dm)

    print("[3/4] 可答题评测（完整主干，含 rerank）...")
    from evaluator import EvalRunner
    report = EvalRunner(retriever).run_eval(answerable, top_k=5)

    print("[4/5] 拒答题检测（检索 + LLM 生成）...")
    from access_control import UserContext
    user_ctx = UserContext(role="admin", groups=["全员"])

    refusal_details = []
    for item in refusal:
        results = retriever.retrieve_hybrid(item['query'], top_k=5, user_ctx=user_ctx)
        answer = "".join(retriever.generate_answer(item['query'], results)).strip()
        refused = is_refusal(answer)
        top_score = round(results[0].get('rerank_score', results[0].get('score', 0)) or 0, 4) if results else 0
        refusal_details.append({
            'id': item['id'],
            'query': item['query'],
            'refused': refused,
            'top_score': top_score,
            'top_doc': results[0].get('doc_name', '') if results else '',
            'answer_head': answer[:150],
        })
        print(f"      {'✅ 拒答' if refused else '❌ 编造'} | top1={top_score:.3f} | {item['query']}")

    print("[5/5] ACL 权限回归（多身份：敏感文档不得命中 + 护栏不过度封锁）...")
    from access_control import UserContext as _UC
    acl_details = []
    for item in acl_items:
        ctx = _UC(role=item.get('role', 'employee'), groups=item.get('groups', ['全员']))
        results = retriever.retrieve_hybrid(item['query'], top_k=5, user_ctx=ctx)
        hit_docs = [r.get('doc_name', '') for r in results]
        leaked = [f for f in item.get('expect_forbidden_titles', [])
                  if any(f in d for d in hit_docs)]
        missing = [e for e in item.get('expected_base_titles', [])
                   if not any(e in d for d in hit_docs)]
        passed = not leaked and not missing
        acl_details.append({
            'id': item['id'], 'role': ctx.role, 'query': item['query'],
            'passed': passed, 'leaked_docs': leaked, 'missing_expected': missing,
            'hit_docs': hit_docs[:5], 'note': item.get('note', ''),
        })
        if passed:
            print(f"      ✅ {ctx.role:<8}| {item['query']}")
        else:
            why = (f"泄露 {leaked}" if leaked else "") + \
                  (f" 未命中预期 {missing}" if missing else "")
            print(f"      ❌ {ctx.role:<8}| {item['query']} —— {why}")
            print(f"         命中: {[d[:28] for d in hit_docs[:3]]}")

    n_acl_ok = sum(1 for d in acl_details if d['passed'])

    # ---- 汇总 ----
    print("\n" + "=" * 88)
    print("一、按人设的检索质量（Recall@5 / MRR / 答案要点命中率）")
    print("=" * 88)
    print(f"{'人设':<12}{'条数':<6}{'Recall@5':<12}{'MRR':<10}{'要点命中':<10}")
    print("-" * 88)
    for cat, v in report['category_summary'].items():
        # 补充 MRR 与要点命中（category_summary 只有 recall/hit_rate）
        sub = [d for d in report['details'] if d['category'] == cat]
        mrr = sum(d['mrr'] for d in sub) / len(sub) if sub else 0
        kw = sum(d['keyword_hit'] for d in sub) / len(sub) if sub else 0
        print(f"{cat:<12}{v['count']:<6}{v['avg_recall']:<12.2%}{mrr:<10.3f}{kw:<10.2%}")

    m = report['metrics']
    print("-" * 88)
    print(f"{'整体':<12}{report['total']:<6}{m['avg_recall@k']:<12.2%}{m['avg_mrr']:<10.3f}{m['avg_keyword_hit']:<10.2%}")
    print(f"平均延迟: {m['avg_latency_ms']:.0f}ms")

    print("\n" + "=" * 88)
    print("二、无答案拒答检测（库里没有时会不会编）")
    print("=" * 88)
    n_refused = sum(1 for d in refusal_details if d['refused'])
    print(f"拒答率: {n_refused}/{len(refusal_details)} = {n_refused / len(refusal_details):.1%}")
    for d in refusal_details:
        if not d['refused']:
            print(f"  ❌ {d['id']} {d['query']}")
            print(f"     答: {d['answer_head'][:100]}")

    print("\n" + "=" * 88)
    print("三、ACL 权限回归（多身份：敏感文档不得命中 + 护栏不过度封锁）")
    print("=" * 88)
    print(f"通过率: {n_acl_ok}/{len(acl_details)}")
    for d in acl_details:
        if not d['passed']:
            print(f"  ❌ {d['id']} [{d['role']}] {d['query']}")
            print(f"     泄露: {d['leaked_docs']} | 未命中预期: {d['missing_expected']}")

    print("\n" + "=" * 88)
    print("四、阈值可分性分析（能否用 top1 分数直接拦住无答案问题）")
    print("=" * 88)
    ans_scores = [d['top_score'] for d in report['details']]
    rej_scores = [d['top_score'] for d in refusal_details]
    print(f"可答题 top1 分数: 平均 {sum(ans_scores)/len(ans_scores):.3f}  "
          f"最低 {min(ans_scores):.3f}  最高 {max(ans_scores):.3f}")
    print(f"拒答题 top1 分数: 平均 {sum(rej_scores)/len(rej_scores):.3f}  "
          f"最低 {min(rej_scores):.3f}  最高 {max(rej_scores):.3f}")
    if min(ans_scores) > max(rej_scores):
        print(f"→ 存在清晰分界：阈值可设在 {max(rej_scores):.2f} ~ {min(ans_scores):.2f} 之间")
    else:
        print("→ 分数区间重叠，无法仅靠阈值区分：无答案问题必须依赖 LLM 拒答行为兜底")

    # ---- 保存 ----
    out = {
        'eval_set': 'seed_persona',
        'answerable_report': report,
        'refusal_details': refusal_details,
        'refusal_rate': round(n_refused / len(refusal_details), 4),
        'acl_details': acl_details,
        'acl_pass_rate': round(n_acl_ok / len(acl_details), 4) if acl_details else None,
    }
    from evaluator import save_report
    path = save_report(out, name='persona_eval')
    print(f"\n报告已保存: {path}")


if __name__ == '__main__':
    main()
