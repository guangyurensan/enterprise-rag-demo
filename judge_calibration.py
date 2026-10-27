# -*- coding: utf-8 -*-
"""
答案级裁判的人工校准：导出打分表 → 人工填写 → 计算一致性（Cohen's κ）

裁判和生成用的是同一个模型（DeepSeek），存在自偏好；只有和人工判定对过账，
这个指标才能当门禁用。

用法：
  1. 导出（默认取最新一份答案级评测报告，按判定分层抽 20 条）
       python judge_calibration.py --export
     两类样本大致各半，但报告本身若是 19 grounded + 1 refused，
     导出也就是 19:1 —— 这时 κ 会被样本不平衡抬高，脚本会打印提示。
     想要有说服力的 κ，先跑覆盖拒答题的评测：python answer_judge_eval.py --set seed_persona
  2. 打开生成的 csv，在 human_verdict 列填：grounded / partial / ungrounded / refused
     （判定口径见 csv 顶部说明，或直接看同一行 judge_reason）
  3. 算一致性
       python judge_calibration.py --score surgical_qa/eval_reports/judge_calibration_xxx.csv

判读：
  κ ≥ 0.6  一致性可接受，裁判可作为趋势指标并逐步用于门禁
  0.4~0.6  勉强，需要补充判定口径说明或多标几次
  < 0.4    不可用，改 prompt 或换模型（建议换一个模型当裁判，避免自偏好）
  前提：两类样本各 ≥ 5 条，否则 κ 的区间太宽，数字本身没有意义。
"""

import argparse
import csv
import glob
import json
import os
import random
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(BASE, "surgical_qa", "eval_reports")

VERDICTS = ["grounded", "partial", "ungrounded", "refused", "should_refuse_but_answered"]

GUIDE = """判定口径（只判断回答是否被【引用片段】支持，不判断业务上是否最优）：
grounded                  每个具体断言（数字/期限/审批人/条件/流程）都能在片段中找到出处
partial                   部分有出处，部分无出处，或明显漏答关键项
ungrounded                关键结论在片段中找不到出处（含用常识推断制度的表述）
refused                   明确说明资料不足（这在拒答题上是正确行为）
should_refuse_but_answered 资料里没有答案，但回答仍给出具体数字/条款
"""


def load_latest():
    files = sorted(glob.glob(os.path.join(REPORT_DIR, "answer_judge_*.json")), key=os.path.getmtime)
    if not files:
        raise SystemExit("没有找到答案级评测报告，先跑 python answer_judge_eval.py")
    return files[-1], json.load(open(files[-1], encoding="utf-8"))


MIN_PER_CLASS = 5
# κ 是二分类一致性指标，两类里样本少的那一类不足 5 条时，
# 一两条分歧就能让 κ 从 0.9 掉到 0.4，结论不可用。


def export(n=20):
    src, report = load_latest()
    details = report["details"]
    bad = [d for d in details if d["verdict"] != "grounded"]
    good = [d for d in details if d["verdict"] == "grounded"]

    # 分层抽样：两类大致各占一半。
    # 原实现是「bad 全取、good 补足」，bad 一多就把 grounded 挤成 0；
    # 样本单边时 κ 恒等于 0 或 1，看着很好实际没测到东西。
    take_bad = min(len(bad), max(n // 2, n - len(good)))
    take_good = min(len(good), n - take_bad)
    picked = bad[:take_bad] + good[:take_good]
    # 打乱顺序，避免「排在前面的就是有问题的」这种顺序暗示，固定种子保证可复现
    random.Random(0).shuffle(picked)

    out = os.path.join(REPORT_DIR, f"judge_calibration_{os.path.basename(src).replace('.json', '')}.csv")
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# " + line for line in GUIDE.strip().split("\n")])
        w.writerow([])
        w.writerow(["id", "query", "answer", "judge_verdict", "judge_faithfulness",
                    "judge_completeness", "judge_reason", "human_verdict", "human_note"])
        for d in picked:
            w.writerow([d["id"], d["query"], d["answer"], d["verdict"], d["faithfulness"],
                        d["completeness"], d["reason"], "", ""])

    print(f"导出 {len(picked)} 条（非 grounded {take_bad}/{len(bad)}，grounded {take_good}/{len(good)}）")
    print(f"来源报告：{src}")
    print(f"打分表：{out}")
    if min(take_bad, take_good) < MIN_PER_CLASS:
        print(f"\n[注意] 有一类样本不足 {MIN_PER_CLASS} 条，κ 的置信区间会很宽，"
              f"一两条分歧就能让结论翻盘；这份表只够验口径，不能当门禁依据。")
        print(f"       想拿到可用结论，先跑覆盖面更广的答案级评测（含拒答题）：")
        print(f"       python answer_judge_eval.py --set seed_persona")
    print(f"\n把 human_verdict 列填成 {VERDICTS[:4]} 之一，然后：")
    print(f"  python judge_calibration.py --score \"{out}\"")


def cohen_kappa(pairs):
    """二分类 κ：grounded vs 非 grounded"""
    n = len(pairs)
    if n == 0:
        raise SystemExit("没有有效标注")
    a = sum(1 for h, j in pairs if h == "grounded" and j == "grounded")
    b = sum(1 for h, j in pairs if h == "grounded" and j != "grounded")
    c = sum(1 for h, j in pairs if h != "grounded" and j == "grounded")
    d = sum(1 for h, j in pairs if h != "grounded" and j != "grounded")
    po = (a + d) / n
    ph = (a + b) / n
    pj = (a + c) / n
    pe = ph * pj + (1 - ph) * (1 - pj)
    return (po - pe) / (1 - pe) if pe != 1 else 0.0, po, {"一致 grounded": a, "人工 grounded 裁判否": b,
                                                          "人工否 裁判 grounded": c, "一致非 grounded": d}


def score(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            rows.append(line)
    reader = csv.DictReader(rows)
    filled, disagree = [], []
    for r in reader:
        h = (r.get("human_verdict") or "").strip().lower()
        j = (r.get("judge_verdict") or "").strip().lower()
        if not h or h not in VERDICTS:
            continue
        filled.append((h, j))
        hb = "grounded" if h == "grounded" else "非grounded"
        jb = "grounded" if j == "grounded" else "非grounded"
        if hb != jb:
            disagree.append((r["id"], r["query"][:30], f"人工={h} 裁判={j}", (r.get("judge_reason") or "")[:80]))

    if not filled:
        raise SystemExit("表里 human_verdict 还没填")

    kappa, po, table = cohen_kappa(filled)
    dist = Counter(h for h, _ in filled)
    n_grounded = sum(1 for h, _ in filled if h == "grounded")
    minority = min(n_grounded, len(filled) - n_grounded)
    print(f"已标注 {len(filled)} 条，人工判定分布：{dict(dist)}")
    print(f"一致率（observed agreement）：{po:.3f}")
    print(f"Cohen's κ：{kappa:.3f}")
    print(f"列联：{table}")
    if minority < MIN_PER_CLASS:
        # 样本单边时 κ 会逼近 1（两类都几乎只有 grounded，随机一致的基线也高），
        # 这时给「可作门禁」的结论是误导，宁可不给。
        print(f"[注意] 较少的一类只有 {minority} 条（建议 ≥{MIN_PER_CLASS}）：κ 在这个样本上不稳定，"
              f"只能看方向，不要拿来当门禁阈值。")
        print("→ 样本量不足，暂不下一致性结论；补齐另一类样本后重跑")
    elif kappa >= 0.6:
        print("→ 一致性可接受，裁判可作趋势指标，并逐步用于门禁")
    elif kappa >= 0.4:
        print("→ 勉强，建议补充判定口径或多标一轮")
    else:
        print("→ 不可用：建议换一个模型当裁判（避免同模型自偏好），或改写判定 prompt")
    if disagree:
        print(f"\n分歧 {len(disagree)} 条（优先看这些，它们说明判定口径没对齐）：")
        for d in disagree:
            print("  -", " | ".join(str(x) for x in d))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--score", default="")
    ap.add_argument("-n", type=int, default=20)
    a = ap.parse_args()
    if a.score:
        score(a.score)
    elif a.export:
        export(a.n)
    else:
        ap.print_help()
