# -*- coding: utf-8 -*-
"""
load_test.py —— 服务级并发压测
对运行中的主服务（默认 127.0.0.1:8765）做真实 HTTP 并发压测：
  Phase 1  /api/search  检索全链路（嵌入 API + BM25 + rerank API，无 LLM 生成）
  Phase 2  /api/chat    完整问答（检索 + LLM 生成）
并发档位 1/10/20/50，统计吞吐、p50/p95 延迟、错误率、空结果率；
压测前后拉取 /api/metrics 对比服务端计数（llm_error / search_empty 等）。

用法：
  python load_test.py --base http://127.0.0.1:8765
报告落盘 eval_reports/load_<ts>.json
"""
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE, "eval_reports")
ROLE = "admin"
GROUPS = ["全员"]


def pct(values, p):
    if not values:
        return 0.0
    vs = sorted(values)
    idx = min(len(vs) - 1, max(0, round(p / 100 * len(vs)) - 1))
    return vs[idx]


def fire_one(endpoint, query, timeout):
    """发单请求，返回 (ok, latency, err_type, extra)"""
    t0 = time.time()
    try:
        if endpoint == "search":
            payload = {"query": query, "user_role": ROLE, "user_groups": GROUPS, "top_k": 5}
            r = requests.post(f"{_base}/api/search", json=payload, timeout=timeout)
            ok_flag = (r.status_code == 200) and bool(r.json().get("results"))
            extra = len(r.json().get("results", [])) if r.status_code == 200 else -1
        else:
            payload = {"query": query, "user_role": ROLE, "user_groups": GROUPS, "top_k": 5}
            r = requests.post(f"{_base}/api/chat", json=payload, timeout=timeout)
            if r.status_code != 200:
                ok_flag, extra = False, -1
            else:
                ans = r.json().get("answer", "")
                ok_flag = bool(ans.strip()) and "没有找到" not in ans and "未能找到" not in ans
                extra = len(ans)
        return ok_flag, time.time() - t0, None, extra
    except requests.exceptions.Timeout:
        return False, time.time() - t0, "timeout", -1
    except requests.exceptions.ConnectionError:
        return False, time.time() - t0, "connection", -1
    except Exception as e:
        return False, time.time() - t0, f"http:{type(e).__name__}", -1


_base = "http://127.0.0.1:8765"


def run_level(endpoint, queries, conc, n_req, timeout):
    """一个并发档位：conc 个 worker 跑 n_req 个请求"""
    latencies, errors = [], []
    ok_cnt, empty_cnt = 0, 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as pool:
        futures = [pool.submit(fire_one, endpoint, queries[i % len(queries)], timeout)
                   for i in range(n_req)]
        for f in futures:
            ok, lat, err, _ = f.result()
            latencies.append(lat)
            if ok:
                ok_cnt += 1
            elif err:
                errors.append(err)
            else:
                empty_cnt += 1  # HTTP 200 但结果为空
    wall = time.time() - t0
    return {
        "concurrency": conc, "requests": n_req, "wall_s": round(wall, 2),
        "rps": round(n_req / wall, 2),
        "ok": ok_cnt, "empty_200": empty_cnt,
        "errors": dict((e, errors.count(e)) for e in set(errors)),
        "error_rate": round(len(errors) / n_req, 4),
        "p50_ms": round(pct(latencies, 50) * 1000, 1),
        "p95_ms": round(pct(latencies, 95) * 1000, 1),
        "max_ms": round(max(latencies) * 1000, 1),
    }


def get_metrics_snapshot():
    try:
        r = requests.get(f"{_base}/api/metrics", timeout=10)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def main():
    global _base
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--skip-chat", action="store_true", help="只压 search，不压 chat（省 LLM 费用）")
    args = ap.parse_args()
    _base = args.base.rstrip("/")

    # 载荷：persona 评测集 72 条真实问题
    qs = json.load(open(os.path.join(BASE, "surgical_qa/eval_sets/seed_persona.json"), encoding="utf-8"))
    items = qs if isinstance(qs, list) else qs.get("items", [])
    queries = [it["query"] for it in items]
    print(f"载荷就绪：{len(queries)} 条真实问题；目标 {_base}")

    m_before = get_metrics_snapshot()
    report = {"base": _base, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "search": [], "chat": []}

    # 预热 1 发（模型/连接池冷启动不计入统计）
    ok, lat, err, _ = fire_one("search", queries[0], timeout=120)
    print(f"预热：ok={ok} {lat:.2f}s")

    # ---- Phase 1: /api/search ----
    print("\n== Phase 1: /api/search（检索全链路）==")
    for conc, n in [(1, 10), (10, 30), (20, 40), (50, 50)]:
        row = run_level("search", queries, conc, n, timeout=90)
        report["search"].append(row)
        print(f"  c={conc:>2} | RPS {row['rps']:>6} | p50 {row['p50_ms']:>7}ms | p95 {row['p95_ms']:>7}ms "
              f"| 空200 {row['empty_200']} | 错误 {row['errors']}")

    # ---- Phase 2: /api/chat ----
    if not args.skip_chat:
        print("\n== Phase 2: /api/chat（含 LLM 生成）==")
        for conc, n in [(1, 5), (10, 20), (20, 20), (50, 20)]:
            row = run_level("chat", queries, conc, n, timeout=240)
            report["chat"].append(row)
            print(f"  c={conc:>2} | RPS {row['rps']:>6} | p50 {row['p50_ms']:>7}ms | p95 {row['p95_ms']:>7}ms "
                  f"| 空200 {row['empty_200']} | 错误 {row['errors']}")

    m_after = get_metrics_snapshot()
    report["metrics_before"] = m_before
    report["metrics_after"] = m_after
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(REPORT_DIR, f"load_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump(report, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n报告已写入 {out}")


if __name__ == "__main__":
    sys.exit(main())
