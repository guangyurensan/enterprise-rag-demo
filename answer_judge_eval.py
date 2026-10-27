# -*- coding: utf-8 -*-
"""
答案级评测：检索 → 生成 → LLM 裁判

与 evaluator.py 的区别：
  evaluator.py 只跑到「检索结果」，验的是「找没找到」；
  本脚本跑到「最终回答」，验的是「答得对不对、引用支不支持、该拒答有没有拒」。

用法：
  python answer_judge_eval.py --set seed_persona --limit 5
  python answer_judge_eval.py --set seed_persona --ids p-001,p-042
  python answer_judge_eval.py --set seed_v2 --limit 3 --no-refusal
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

from access_control import UserContext
from data_loader import load_all_records
from document_manager import DocumentManager
from retriever import KnowledgeRetriever
from task_executor import executor

# 裁判输出契约：五个判定槽 + 证据摘录
JUDGE_SCHEMA = {
    "verdict": (
        str,
        "grounded：每个具体断言都能在资料中找到依据；partial：部分有依据但缺失或夹杂无依据内容；"
        "ungrounded：关键结论在资料中找不到依据；refused：回答明确表示资料不足；"
        "should_refuse_but_answered：资料明显不含答案，但回答仍给出了具体数字、期限或条款",
        "not_null",
    ),
    "faithfulness": (int, "0-100 整数。回答中能被资料原文支持的具体断言占比", "not_null"),
    "completeness": (int, "0-100 整数。问题问到的关键要点是否都答到了", "not_null"),
    "citation_ok": (bool, "回答标注的引用是否真的包含所引内容", "not_null"),
    # 不用列表：Agently 的 not_null 会展开成 items[*]，空数组永远无法满足，
    # 而「没有无依据断言」恰恰是合法的空情况 —— 实测 4/4 全部重试失败。
    "unsupported_claims": (str, "资料中找不到依据的具体断言，摘录原句并用；分隔；没有则填「无」", "not_null"),
    "reason": (str, "一句话说明判定依据，必须点出具体断言或缺失点", "not_null"),
}

JUDGE_SYSTEM = (
    "你是企业知识库问答的审核员。你只依据用户提供的【给定资料】判断回答，"
    "严禁使用你自己的常识或行业经验补充。资料里没写的内容，一律视为无依据。"
)


def load_items(set_name, ids=None, limit=None, include_refusal=True):
    path = os.path.join(BASE, "surgical_qa", "eval_sets", f"{set_name}.json")
    with open(path, encoding="utf-8") as f:
        items = json.load(f)["items"]
    if ids:
        want = {i.strip() for i in ids.split(",") if i.strip()}
        items = [x for x in items if x.get("id") in want]
    if not include_refusal:
        items = [x for x in items if not x.get("expect_refusal")]
    return items[:limit] if limit else items


def build_judge_input(query, context, answer, expect_refusal):
    refusal_rule = ""
    if expect_refusal:
        refusal_rule = (
            "\n本题的正确答案是「资料不足以回答」。"
            "如果回答仍然给出了具体数字、期限、条款或流程，判为 should_refuse_but_answered，faithfulness 记 0。"
        )
    return f"""【用户问题】
{query}

【给定资料】
{context[:6000]}

【系统回答】
{answer}

【判定要求】{refusal_rule}
逐条核对回答中的具体断言（数字、期限、审批人、金额、流程步骤），
每一条都要能在【给定资料】里找到出处；找不到的写进 unsupported_claims。
不要因为「说得合理」就判它有依据。"""


def run(set_name, ids, limit, include_refusal, top_k):
    dm = DocumentManager()
    retriever = KnowledgeRetriever(
        load_all_records(doc_manager=dm), use_vector_index=True, doc_manager=dm
    )
    ctx = UserContext(role="admin", groups=["全员"])
    items = load_items(set_name, ids, limit, include_refusal)

    details = []
    for i, item in enumerate(items, 1):
        qid = item.get("id", f"q{i}")
        query = item.get("query", "")
        expect_refusal = bool(item.get("expect_refusal"))

        t0 = time.time()
        results = retriever.retrieve_hybrid(query, top_k=top_k, user_ctx=ctx, eval_mode=True)
        answer = "".join(retriever.generate_answer(query, results)).strip()
        context = retriever.format_context(results)
        gen_sec = time.time() - t0

        try:
            verdict = executor.run_structured(
                user_input=build_judge_input(query, context, answer, expect_refusal),
                output_schema=JUDGE_SCHEMA,
                system_prompt=JUDGE_SYSTEM,
                max_retries=2,
                temperature=0,
            )
            err = None
        except Exception as e:
            verdict = {"verdict": "judge_error", "faithfulness": -1, "completeness": -1,
                       "citation_ok": False, "unsupported_claims": [], "reason": str(e)[:200]}
            err = type(e).__name__

        total_sec = time.time() - t0
        raw = verdict.get("unsupported_claims")
        raw_claims = ";".join(raw) if isinstance(raw, list) else str(raw or "")
        claims = [
            c.strip() for c in raw_claims.replace("；", ";").split(";")
            if c.strip() and c.strip() != "无"
        ]
        detail = {
            "id": qid,
            "query": query,
            "expect_refusal": expect_refusal,
            "answer": answer,
            "answer_chars": len(answer),
            "top_docs": [r.get("doc_name", "") for r in results[:3]],
            "verdict": verdict.get("verdict"),
            "faithfulness": verdict.get("faithfulness"),
            "completeness": verdict.get("completeness"),
            "citation_ok": verdict.get("citation_ok"),
            "unsupported_claims": claims,
            "reason": verdict.get("reason"),
            "judge_error": err,
            "gen_sec": round(gen_sec, 2),
            "total_sec": round(total_sec, 2),
        }
        details.append(detail)

        print(f"[{i}/{len(items)}] {qid} {verdict.get('verdict')} "
              f"faith={verdict.get('faithfulness')} comp={verdict.get('completeness')} "
              f"cite={verdict.get('citation_ok')} {total_sec:.1f}s")
        print(f"    Q: {query}")
        print(f"    A: {answer[:120].replace(chr(10), ' ')}")
        print(f"    判: {verdict.get('reason')}")
        for c in claims[:3]:
            print(f"    无依据: {c}")
        print()

    answered = [d for d in details if not d["expect_refusal"]]
    refusals = [d for d in details if d["expect_refusal"]]
    ok = [d for d in details if d["verdict"] not in ("judge_error", None)]

    metrics = {
        "count": len(details),
        "judge_ok": len(ok),
        "grounded_rate": round(sum(1 for d in answered if d["verdict"] == "grounded") / len(answered), 4) if answered else None,
        "avg_faithfulness": round(sum(d["faithfulness"] for d in answered) / len(answered), 1) if answered else None,
        "avg_completeness": round(sum(d["completeness"] for d in answered) / len(answered), 1) if answered else None,
        "citation_ok_rate": round(sum(1 for d in answered if d["citation_ok"]) / len(answered), 4) if answered else None,
        "refusal_correct": sum(1 for d in refusals if d["verdict"] == "refused"),
        "refusal_total": len(refusals),
        "over_answer": sum(1 for d in refusals if d["verdict"] == "should_refuse_but_answered"),
        "avg_gen_sec": round(sum(d["gen_sec"] for d in details) / len(details), 2),
        "avg_total_sec": round(sum(d["total_sec"] for d in details) / len(details), 2),
    }

    report = {
        "set": set_name,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "top_k": top_k,
        "metrics": metrics,
        "details": details,
    }

    out_dir = os.path.join(BASE, "surgical_qa", "eval_reports")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"answer_judge_{set_name}_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"报告: {out}")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--set", default="seed_persona")
    p.add_argument("--ids", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--no-refusal", action="store_true")
    a = p.parse_args()
    run(a.set, a.ids, a.limit, not a.no_refusal, a.top_k)
