# -*- coding: utf-8 -*-
"""
分层回归：按改动范围选择要跑的层，避免每次都全量。

  L0  冒烟      约 20s    改任何代码都跑：检索 6 条 + 生成 1 条 + 拒答 1 条 + 版本治理快检
  L1  检索全量  约 1.5min 改解析/切块/检索/rerank 后跑：seed_v2 26 + persona 可答 61
  L2  答案级    约 60s    改 prompt/模型后跑：答案级裁判抽样 20 条
  L3  全部      约 5min   发版前跑：L1 + 答案级全量 26 条

用法：
  python run_regress.py --level 0
  python run_regress.py --level 1
  python run_regress.py --level 3 --update-baseline    # 确认指标变化可接受后，刷新基线

基线存在 surgical_qa/eval_reports/regress_baseline.json；
指标相对基线下跌超过阈值时，输出 ⚠ 提示（不阻断，由人判断）。
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
REPORT_DIR = os.path.join(BASE, "surgical_qa", "eval_reports")
BASELINE_FILE = os.path.join(REPORT_DIR, "regress_baseline.json")

# 允许的波动：低于基线超过这些值就告警
TOLERANCE = {"recall": 0.02, "mrr": 0.03, "grounded": 0.05}


def newest(pattern, after_ts):
    """取本次运行新产生的最新报告"""
    files = [f for f in glob.glob(pattern) if os.path.getmtime(f) >= after_ts - 1]
    return max(files, key=os.path.getmtime) if files else None


def run_cmd(args, label):
    print(f"\n{'=' * 64}\n{label}\n{'=' * 64}")
    t0 = time.time()
    r = subprocess.run(args, cwd=BASE, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-2000:])
        print(r.stderr[-2000:])
        raise SystemExit(f"[失败] {label}")
    return time.time() - t0


def load_metrics(path):
    if not path or not os.path.exists(path):
        return {}
    d = json.load(open(path, encoding="utf-8"))
    return d.get("metrics", d)


# ---------------------------------------------------------------- L0
def level0():
    sys.path.insert(0, os.path.join(BASE, "surgical_qa"))
    import logging
    logging.disable(logging.INFO)

    from access_control import UserContext
    from data_loader import load_all_records
    from document_manager import DocumentManager
    from retriever import KnowledgeRetriever
    from version_policy import resolve_current_map

    t0 = time.time()
    checks = []

    dm = DocumentManager()
    current = resolve_current_map(dm.registry.documents)
    names = list(current.keys())
    checks.append(("版本治理：现行制度", len(names), 1, all(n.strip() for n in names)))

    retriever = KnowledgeRetriever(load_all_records(doc_manager=dm),
                                   use_vector_index=True, doc_manager=dm)
    ctx = UserContext(role="admin", groups=["全员"])

    seed = json.load(open(os.path.join(BASE, "surgical_qa/eval_sets/seed_v2.json"),
                          encoding="utf-8"))["items"][:6]
    hits = 0
    for item in seed:
        ids = [current[t] for t in item.get("expected_base_titles", []) if t in current]
        res = retriever.retrieve_hybrid(item["query"], top_k=5, user_ctx=ctx, eval_mode=True)
        got = {r.get("doc_id") for r in res}
        if not ids or got & set(ids):
            hits += 1
    checks.append(("检索命中（6 条）", f"{hits}/6", ">=5", hits >= 5))

    first = seed[0]
    res = retriever.retrieve_hybrid(first["query"], top_k=5, user_ctx=ctx, eval_mode=True)
    answer = "".join(retriever.generate_answer(first["query"], res)).strip()
    checks.append(("生成答案非空", f"{len(answer)} 字", ">50", len(answer) > 50))

    persona = json.load(open(os.path.join(BASE, "surgical_qa/eval_sets/seed_persona.json"),
                             encoding="utf-8"))["items"]
    refusal = next(x for x in persona if x.get("expect_refusal"))
    res = retriever.retrieve_hybrid(refusal["query"], top_k=5, user_ctx=ctx, eval_mode=True)
    ans = "".join(retriever.generate_answer(refusal["query"], res)).strip()
    markers = ("未找到", "没有", "无法", "不包含", "未涉及", "资料不足", "暂无")
    checks.append(("拒答（库外问题）", "命中拒答词" if any(m in ans for m in markers) else "未拒答",
                   "需拒答", any(m in ans for m in markers)))

    print(f"\n{'检查项':<22}{'实测':<16}{'期望':<10}结果")
    ok = True
    for name, actual, expect, passed in checks:
        ok = ok and passed
        print(f"{name:<22}{str(actual):<16}{str(expect):<10}{'PASS' if passed else 'FAIL'}")

    print(f"\nL0 {'全部通过' if ok else '存在失败'}，耗时 {time.time() - t0:.1f}s")
    return ok


# ---------------------------------------------------------------- L1/L2/L3
def run_level(level, update_baseline):
    t_start = time.time()
    result = {}

    if level in (1, 3):
        ts = time.time()
        run_cmd([PY, "run_baseline.py"], "L1-a 检索基线 seed_v2（26 条）")
        m = load_metrics(newest(os.path.join(REPORT_DIR, "baseline_seed_v2_*.json"), ts))
        result["seed_v2"] = {"recall": m.get("avg_recall@k"), "mrr": m.get("avg_mrr"),
                             "latency_ms": m.get("avg_latency_ms"),
                             "missing_expectation": m.get("missing_expectation_count", 0),
                             "empty_expectation": m.get("empty_expectation_count", 0)}

        ts = time.time()
        run_cmd([PY, "run_persona_eval.py"], "L1-b 人设集（78 条：61 可答 + 11 拒答 + 6 ACL）")
        p = newest(os.path.join(REPORT_DIR, "persona_eval_*.json"), ts)
        pm = load_metrics(p).get("answerable_report", {}).get("metrics", {})
        refusal_rate = load_metrics(p).get("refusal_rate")
        result["persona"] = {"recall": pm.get("avg_recall@k"), "mrr": pm.get("avg_mrr"),
                             "refusal_rate": refusal_rate,
                             "acl_pass_rate": load_metrics(p).get("acl_pass_rate"),
                             "latency_ms": pm.get("avg_latency_ms"),
                             "missing_expectation": pm.get("missing_expectation_count", 0),
                             "empty_expectation": pm.get("empty_expectation_count", 0)}

    if level in (2, 3):
        ts = time.time()
        limit = ["--limit", "20"] if level == 2 else []
        run_cmd([PY, "answer_judge_eval.py", "--set", "seed_v2"] + limit,
                f"L{level} 答案级裁判 seed_v2（{'抽样 20' if level == 2 else '全量 26'}）")
        m = load_metrics(newest(os.path.join(REPORT_DIR, "answer_judge_seed_v2_*.json"), ts))
        result["judge"] = {"grounded": m.get("grounded_rate"),
                           "faithfulness": m.get("avg_faithfulness"),
                           "completeness": m.get("avg_completeness"),
                           "citation_ok": m.get("citation_ok_rate")}

    # ---- 与基线对比
    baseline = {}
    if os.path.exists(BASELINE_FILE):
        baseline = json.load(open(BASELINE_FILE, encoding="utf-8"))

    print(f"\n{'=' * 64}\n指标对比（{'★ 新基线' if update_baseline else 'vs 基线'}）\n{'=' * 64}")
    warnings = []
    for section, vals in result.items():
        old = baseline.get(section, {})
        for k, v in vals.items():
            if v is None:
                continue
            o = old.get(k)
            flag = ""
            if o is not None and not update_baseline:
                key = "recall" if "recall" in k else ("grounded" if k == "grounded" else k)
                tol = TOLERANCE.get(key, 0)
                # 除 refusal_rate / faithfulness / completeness 外，数值越大越好
                if key in ("recall", "mrr", "grounded", "citation_ok", "refusal_rate",
                           "faithfulness", "completeness", "acl_pass_rate"):
                    if v < o - tol:
                        flag = " ⚠"
                        warnings.append(f"{section}.{k}: {o} → {v}")
                elif k == "latency_ms" and v > o * 1.5:
                    flag = " ⚠"
                    warnings.append(f"{section}.{k}: {o} → {v}ms")
            print(f"  {section:<10}{k:<16}{str(v):<12}" +
                  (f"基线 {o}{flag}" if o is not None else "(无基线)"))

    # 期望解析失败：与基线无关，任何时候都不允许——期望文档解析不到时该期望项被
    # 静默跳过，若全空则 recall 兜底返回 1.0，用例变成"永久通过"的空壳（假绿）。
    # 2026-10-27 语料改名未同步 registry 就是这样骗过了一轮回归。
    for section, vals in result.items():
        miss = vals.get("missing_expectation") or 0
        empty = vals.get("empty_expectation") or 0
        if miss:
            msg = f"{section}.missing_expectation: {miss} 条用例的期望文档无法解析"
            if empty:
                msg += f"，其中 {empty} 条期望全空（recall 恒为 1.0，已失去检验能力）"
            warnings.append(msg)

    if update_baseline:
        baseline.update(result)
        json.dump(baseline, open(BASELINE_FILE, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"\n基线已更新：{BASELINE_FILE}")

    if warnings:
        print("\n⚠ 相对基线退化：")
        for w in warnings:
            print("  -", w)
    else:
        print("\n无退化。")

    print(f"\n总耗时 {time.time() - t_start:.1f}s")
    return not warnings


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--update-baseline", action="store_true")
    a = ap.parse_args()
    if a.level == 0:
        sys.exit(0 if level0() else 1)
    sys.exit(0 if run_level(a.level, a.update_baseline) else 1)
