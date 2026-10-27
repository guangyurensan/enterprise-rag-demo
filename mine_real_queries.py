# -*- coding: utf-8 -*-
"""
真实问法挖矿：从审计日志里捞"该进评测集的真实问法"

背景（2026-10-25）：seed_persona 78 条都是自己写的问法——出题人和
答题人同源，容易越测越顺。真实用户的问法才是评测集该有的样子。

数据源：audit_logs/audit_*.jsonl（每条真实检索天然留痕，无需埋点）

四类高价值信号（按优先级）：
  zero_hit      零命中/极低命中——知识缺口，或问法与文档措辞差太远
  low_confidence 最高分低于阈值——检索到但不确定，匹配质量弱
  acl_blocked   被权限过滤比例高——有真实需求但无权访问（运营信号，
                不进评测集，该给业务方看）
  repeated      同一问法被反复问——高频问题，值得进回归集保住质量

产出：eval_reports/real_queries_<date>.md 人工审核清单
      （机器只提名，人工决定是否进 seed_persona——问法是否有代表性、
       期望命中是否合理，需要懂业务的人判断）

用法：
  python mine_real_queries.py                 # 全部历史
  python mine_real_queries.py --days 7        # 最近 7 天
  python mine_real_queries.py --min-score 0.5 # 低置信阈值
"""

import argparse
import json
import os
import glob
from collections import Counter, defaultdict
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
AUDIT_DIR = os.path.join(BASE, "audit_logs")
REPORT_DIR = os.path.join(BASE, "surgical_qa", "eval_reports")

LOW_SCORE_DEFAULT = 0.5   # rerank 分低于此视为"检索到但不自信"
TOP_N = 30                # 每类最多列出多少条


def load_entries(days: int):
    """读取审计文件（最近 N 天，N<=0 表示全部）"""
    files = sorted(glob.glob(os.path.join(AUDIT_DIR, "audit_*.jsonl")))
    if days > 0:
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        files = [f for f in files
                 if os.path.basename(f)[6:16] >= cutoff]
    entries = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("event_type") == "search":
                    entries.append(e)
    return entries


def classify(entries, min_score):
    """按四类信号分组（同一查询可能命中多类，重复有价值就重复列）"""
    zero_hit, low_conf, acl_blocked, repeated = [], [], [], []
    freq = Counter(e.get("query_hash") for e in entries)
    by_hash = {}
    for e in entries:
        qh = e.get("query_hash")
        if qh not in by_hash:
            by_hash[qh] = e

    for e in entries:
        scores = [h.get("score", 0) for h in e.get("hit_docs", [])]
        top = max(scores) if scores else 0

        if e.get("result_count", 0) == 0 or top == 0:
            zero_hit.append(e)
        elif top < min_score:
            low_conf.append(e)

        if e.get("filtered_count", 0) >= 2:
            acl_blocked.append(e)

    for qh, n in freq.items():
        if n >= 3 and qh in by_hash:
            repeated.append(by_hash[qh])

    return zero_hit, low_conf, acl_blocked, repeated, freq, by_hash


def fmt(e, freq=None):
    q = (e.get("query") or "").replace("\n", " ")[:40]
    top = max((h.get("score", 0) for h in e.get("hit_docs", [])), default=0)
    extra = f" ×{freq}" if freq else ""
    return (f"| {e.get('timestamp', '')[:16].replace('T', ' ')} "
            f"| {e.get('user_role', '')} | {q}{extra} "
            f"| top {top:.2f} | 滤 {e.get('filtered_count', 0)} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=0, help="只看最近 N 天（0=全部）")
    ap.add_argument("--min-score", type=float, default=LOW_SCORE_DEFAULT)
    args = ap.parse_args()

    entries = load_entries(args.days)
    if not entries:
        print("审计日志中没有 search 事件——先积累几天真实使用再来挖。")
        return

    zero_hit, low_conf, acl_blocked, repeated, freq, _ = classify(
        entries, args.min_score)

    os.makedirs(REPORT_DIR, exist_ok=True)
    out = os.path.join(
        REPORT_DIR, f"real_queries_{datetime.now().strftime('%Y%m%d_%H%M')}.md")

    lines = [
        f"# 真实问法候选清单（挖矿时间 {datetime.now():%Y-%m-%d %H:%M}）",
        "",
        f"数据范围：{'最近 %d 天' % args.days if args.days else '全部历史'}，"
        f"共 {len(entries)} 条检索事件，去重后 {len(freq)} 个不同问法。",
        "",
        "> 机器只提名，是否进 seed_persona 由人工判断。",
        "> 进集时补字段：role（用哪个身份问）、expect_forbidden_titles / expected_base_titles。",
        "",
    ]

    sections = [
        ("零命中——知识缺口或问法差距", zero_hit,
         "查一下知识库是否真没有相关文档；有的话说明问法与措辞差距大，是最有价值的评测题。"),
        (f"低置信（top 分 < {args.min_score}）——检索到但不确定", low_conf,
         "通常涉及口语化问法 vs 制度文书措辞的鸿沟，适合进集并观察 rerank 表现。"),
        ("反复出现（≥3 次）——高频问题", repeated,
         "用户真正关心的问题，进回归集守住质量底线。"),
        ("被 ACL 过滤多——权限外的真实需求", acl_blocked,
         "不进评测集。汇总给业务方：是文档定密过严，还是该开放给更多角色。"),
    ]

    for title, items, note in sections:
        lines += [f"## {title}（{len(items)} 条）", "", note, ""]
        if not items:
            lines += ["（无）", ""]
            continue
        seen = set()
        shown = 0
        lines += ["| 时间 | 身份 | 问法 | 最高分 | 被滤 |", "|---|---|---|---|---|"]
        for e in sorted(items, key=lambda x: x.get("timestamp", ""), reverse=True):
            qh = e.get("query_hash")
            if qh in seen:
                continue
            seen.add(qh)
            lines.append(fmt(e, freq.get(qh) if freq.get(qh, 0) > 1 else None))
            shown += 1
            if shown >= TOP_N:
                lines.append(f"| … | | （同类共 {len(items)} 条，仅列最近 {TOP_N} 条） | | |")
                break
        lines.append("")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"共扫描 {len(entries)} 条检索事件（{len(freq)} 个不同问法）")
    for title, items, _ in sections:
        print(f"  {title}: {len(items)} 条")
    print(f"清单已写入: {out}")


if __name__ == "__main__":
    main()
