# -*- coding: utf-8 -*-
"""
CAIL2018 法律文书抽样转档（方案 B）

从 CAIL2018 zip 中流式抽取 N 份刑事判决文书，转成单文件 txt：
- 不全量解压（train.json 解压后 2GB），直接流式读 zip 成员
- 每份文书 = 一个 txt（含事实描述 + 罪名/法条元数据头），保持行文为
  真实法律文本风格——用于压测异构长文本的解析与分块
- 抽样步长固定 → 同参数重跑结果一致（幂等）

用法:
    python cail_sampler.py --count 100 --out stress_corpus/cail/docs
"""
import argparse
import io
import json
import os
import sys
import zipfile

ZIP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "stress_corpus", "cail", "CAIL2018_ALL_DATA.zip")
MEMBER = "final_all_data/final_test.json"   # 50MB 最小的主数据文件，够抽


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--out", default="stress_corpus/cail/docs")
    ap.add_argument("--stride", type=int, default=50, help="每 N 行取 1 份")
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    existing = [f for f in os.listdir(out) if f.endswith(".txt")]
    if len(existing) >= args.count:
        print(f"[checkpoint] 已有 {len(existing)} 份 ≥ 目标 {args.count}，跳过")
        return 0
    if not os.path.exists(ZIP_PATH):
        print(f"未找到数据包: {ZIP_PATH}")
        return 1

    made, idx, kept = 0, 0, 0
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open(MEMBER) as raw:
            reader = io.TextIOWrapper(raw, encoding="utf-8")
            for line in reader:
                idx += 1
                if idx % args.stride != 1:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fact = doc.get("fact", "").strip()
                if len(fact) < 200:          # 过滤过短文书，保证压测有效负载
                    continue
                meta = doc.get("meta", {}) or {}
                accusations = "、".join(meta.get("accusation", [])) or "未标注"
                laws = "、".join(map(str, meta.get("law", []))) or "未标注"
                impr = json.dumps(meta.get("imprisonment"), ensure_ascii=False)
                content = (
                    f"案件类型：刑事判决文书（CAIL2018 抽样）\n"
                    f"罪名：{accusations}\n"
                    f"适用法条：{laws}\n"
                    f"刑期标注：{impr}\n"
                    f"\n【事实描述】\n{fact}\n"
                )
                path = os.path.join(out, f"CAIL_doc_{kept:05d}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                made += 1
                kept += 1
                if kept >= args.count:
                    break
    print(f"扫描 {idx} 行，抽样写出 {made} 份 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
