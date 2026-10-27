"""解析健康检查：扫描 parsed_cache 里所有块，找出污染与碎片。

起因：MinerU 返回的 Markdown 对复杂表格用 HTML <table> 输出，而切块器只识别
Markdown 管道表格（| ... |），HTML 标签被原样写进 chunk 正文。4 份 PDF 的
70 个块里 48 个中招，最高一份 73% 的字是标签。

这类污染不报错、不影响块数，但会：
  - 稀释向量（500 字的块里可能只有 150 字有效内容）
  - 污染 BM25 词表（td / colspan / rowspan 成了高频词）
  - 浪费 rerank 与 prompt 的 token 预算

用法：
    python parse_health_check.py              # 全库体检
    python parse_health_check.py --detail     # 附带每个脏块的首行样例
退出码：有严重污染（任一块标签占比 > 30%）时返回 1，可直接接进 CI。
"""
import argparse
import json
import os
import re

REG = "surgical_qa/document_store/document_registry.json"
CACHE_DIR = "surgical_qa/parsed_cache"

TAG_RE = re.compile(r"<[^>]{0,80}>")
DIRTY_MARK_RE = re.compile(r"</?(?:td|tr|th|table|div|p)\b|colspan|rowspan", re.I)


def scan_chunk(content):
    total = len(content)
    if total == 0:
        return 0, 0
    tag_chars = sum(len(m) for m in TAG_RE.findall(content))
    return tag_chars, len(DIRTY_MARK_RE.findall(content))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detail", action="store_true", help="打印脏块样例")
    ap.add_argument("--threshold", type=float, default=0.30, help="单块标签占比告警线")
    args = ap.parse_args()

    registry = json.load(open(REG, encoding="utf-8"))
    docs = list(registry["documents"].values())

    rows = []
    total_chunks = total_tag = severe = 0
    for doc in docs:
        path = os.path.join(CACHE_DIR, f"{doc['doc_id']}.json")
        if not os.path.exists(path):
            continue
        chunks = json.load(open(path, encoding="utf-8"))["chunks"]
        doc_tag = 0
        doc_chars = 0
        doc_severe = 0
        samples = []
        for c in chunks:
            content = c.get("content") or ""
            tag_chars, marks = scan_chunk(content)
            doc_chars += len(content)
            doc_tag += tag_chars
            if len(content) and tag_chars / len(content) > args.threshold:
                doc_severe += 1
                if len(samples) < 2:
                    samples.append(content[:110].replace("\n", " "))
            total_chunks += 1
        total_tag += doc_tag
        severe += doc_severe
        rows.append({
            "filename": doc["filename"],
            "parser": doc.get("parser_used", "?"),
            "chunks": len(chunks),
            "chars": doc_chars,
            "tag_ratio": doc_tag / max(doc_chars, 1),
            "severe": doc_severe,
            "samples": samples,
        })

    rows.sort(key=lambda r: r["tag_ratio"], reverse=True)
    print(f"{'文件':34} {'解析器':9} {'块':>3} {'字数':>7} {'标签占比':>8} {'脏块':>7}")
    print("-" * 78)
    for r in rows:
        flag = "  ←" if r["severe"] else ""
        print(f"{r['filename'][:32]:34} {r['parser']:9} {r['chunks']:3d} "
              f"{r['chars']:7d} {r['tag_ratio'] * 100:7.0f}% {r['severe']:3d}/{r['chunks']:<3}{flag}")

    print("-" * 78)
    print(f"合计 {len(rows)} 份 / {total_chunks} 块，标签占全部正文 {total_tag / max(sum(r['chars'] for r in rows), 1) * 100:.1f}%，"
          f"严重污染块 {severe} 个")

    if args.detail:
        for r in rows:
            if r["samples"]:
                print(f"\n[{r['filename']}] 脏块样例：")
                for s in r["samples"]:
                    print("   ", s)

    return 1 if severe else 0


if __name__ == "__main__":
    raise SystemExit(main())
