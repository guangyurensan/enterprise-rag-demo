"""对比同一份 PDF 在 mineru 与 docling 下的解析结果差异。

用途：确认关闭 mineru（MINERU_ENABLED=False）后，PDF 解析质量的实际损失。
只做只读对比，不写入 registry / Milvus。
"""
import io
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "surgical_qa"))
logging.disable(logging.INFO)

from document_parser import DoclingParser  # noqa: E402

REG = "surgical_qa/document_store/document_registry.json"


def registry_chunks_by_doc():
    """从 registry 反查每份文档当时解析出的块数（没存正文，只存统计）。"""
    d = json.load(open(REG, encoding="utf-8"))
    return {x["filename"]: x for x in d["documents"].values()}


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "示例智能2025年年度报告.pdf"
    reg = registry_chunks_by_doc()
    meta = reg.get(target)
    if not meta:
        print(f"未在 registry 找到 {target}")
        return
    path = meta["file_path"]
    print(f"目标: {target}")
    print(f"mineru 结果: {meta.get('chunk_count')} 块, status={meta.get('parse_status')}")

    parser = DoclingParser()
    if not parser.is_available():
        print("docling 不可用")
        return

    t0 = time.time()
    doc = parser.parse(path)
    cost = time.time() - t0
    text = "\n".join((c.content or "") for c in doc.chunks)
    print(f"docling 结果: {len(doc.chunks)} 块, 正文 {len(text)} 字, 耗时 {cost:.1f}s")

    # 质量粗检：中文占比、替换字符、空白块
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    bad = text.count("\ufffd")
    empty = sum(1 for c in doc.chunks if len((c.content or "").strip()) < 10)
    print(f"中文 {cjk} 字 ({cjk / max(len(text),1) * 100:.1f}%) | 乱码符 {bad} | 空/极短块 {empty}/{len(doc.chunks)}")
    lens = sorted(len((c.content or "")) for c in doc.chunks)
    if lens:
        print(f"块长: min {lens[0]} / 中位 {lens[len(lens)//2]} / max {lens[-1]}")

    print("--- docling 前 3 块 ---")
    for c in doc.chunks[:3]:
        title = getattr(c, "title", "") or ""
        print(f"[{title[:30]}] {(c.content or '')[:120].replace(chr(10), ' ')}")

    out = f"eval_reports/pdf_parser_compare_{target.replace('.pdf','')}.txt"
    os.makedirs("eval_reports", exist_ok=True)
    with io.open(out, "w", encoding="utf-8") as f:
        f.write(f"== {target} docling 全文 ==\n")
        for i, c in enumerate(doc.chunks):
            f.write(f"\n### chunk {i} | title={getattr(c,'title','')}\n{c.content}\n")
    print("全文已写入", out)


if __name__ == "__main__":
    main()
