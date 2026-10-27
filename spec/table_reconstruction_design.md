# 表格重建模块设计

> **状态：设计草案，未实施。** 后续改在源头解决——xlsx 用 `OpenpyxlParser` 直接读单元格（见 `surgical_qa/document_parser.py`），PDF 表格交给 docling 的结构化输出。LLM 重建每份财报要多几十次调用，且会给数值再加一层幻觉面，收益不抵成本，故未进主链路。保留本文用于记录取舍过程。

## Context

PDF 解析器（Docling/MinerU）无法可靠保留表格结构，多列数值被合并为扁平文本（如 `[西南] 5,210.010.81%`）。这导致检索时 embedding 无法匹配语义，BM25 也无法精确命中关键词。

当前 pipeline 中 `block_adapter` 已有 Markdown 表格检测能力，`smart_chunker` 已有 `split_table_block()` 处理 Markdown 表格。问题在于解析器没有输出 Markdown 格式，表格以 `block_type="text"` 进入 pipeline。

**目标**：在清洗后、分块前，用 Agently 结构化输出将扁平文本块重建为 Markdown 表格，使其能被现有 chunker 正确处理。

## 方案：规则快筛 + LLM 重建

### 架构位置

```
解析 → BlockAdapter → DifferentialCleaner → [表格重建] → SmartChunker
                                              ↑ 新增
```

在 `pipeline.py` 的 `clean_blocks()` 之后、`chunk_blocks()` 之前插入新步骤。

### 新增文件：`table_reconstructor.py`

```
class TableReconstructor:
    """表格重建算子：规则快筛 + Agently 结构化输出"""
    
    def reconstruct(self, blocks: List[UnifiedBlock]) -> List[UnifiedBlock]:
        """
        主入口：扫描 blocks，检测表格组，调用 LLM 重建为 Markdown 表格
        
        流程：
        1. 遍历 blocks，用规则检测哪些块像"表格碎片"
        2. 将同页连续的表格碎片分组
        3. 对每组调用 Agently 重建为 Markdown 表格
        4. 用新的 block_type="table" 的 UnifiedBlock 替换原始碎片
        5. 非表格 blocks 原样返回
        """
```

### Step 1：规则快筛 — 检测表格碎片

检测条件（满足任意 2 项即视为表格碎片）：

| 条件 | 正则/逻辑 | 说明 |
|------|----------|------|
| 含 `[xxx]` 标题模式 | `^\[.+?\]` | 如 `[华东]`、`[合计]` |
| 含密集数值 | 数值占比 > 40% | `[\d,]+\.?\d*` 字符占内容比 |
| 含百分比 | `\d+\.?\d*%` | 如 `53.20%` |
| 数值间无分隔符 | `\d\d+\.?\d*%` 或 `\d\.\d+\d\.\d+` | 如 `5,210.010.81%`（两列合并） |
| 短文本 | `len(content) < 200` | 表格碎片通常很短 |

### Step 2：分组 — 同页连续碎片

```python
# 分组条件：
# 1. 同 page_number
# 2. 连续 chunk_index（或间隔 ≤ 1）
# 3. 都通过了 Step 1 的表格检测
```

分组后的每组代表一张被拍平的表格。

### Step 3：LLM 重建

对每组碎片调用 `TaskExecutor.run_structured()`：

```python
output_schema = {
    "markdown_table": (str, "还原后的完整 Markdown 表格，包含表头和数据行"),
}

system_prompt = """你是一个表格结构还原专家。
输入是一组从 PDF 表格中拍平的文本碎片。
你的任务是根据数值模式、标题标签和上下文，推断原始表格的列结构，并还原为 Markdown 表格。

规则：
1. 第一行必须是表头（用 | 分隔）
2. 第二行必须是分隔线 (|---|---|)
3. 后续行是数据行
4. 如果无法确定列名，用"列1"、"列2"等占位
5. 不要编造数据，只使用输入中出现的数值
6. 保留原始数值精度"""

# 输入：将同组碎片按 chunk_index 排序拼接
user_input = f"章节标题：{group_title}\n页码：{page_number}\n\n表格碎片：\n{combined_text}"
```

### Step 4：替换

将 LLM 返回的 Markdown 表格创建为新的 `UnifiedBlock`：

```python
new_block = UnifiedBlock(
    block_id=f"{doc_id}_table_{table_idx:03d}",
    doc_id=doc_id,
    content=markdown_table,        # LLM 返回的 Markdown 表格
    block_type="table",            # 关键：标记为 table
    block_index=original_index,
    parent_title=group_title,      # 继承章节标题
    page_number=page_number,
)
```

后续的 `smart_chunker.split_table_block()` 会自动处理 Markdown 表格分块。

### 修改文件清单

| 文件 | 改动 |
|------|------|
| **新增** `table_reconstructor.py` | 表格重建算子（~150 行） |
| `pipeline.py` | 在 clean_blocks() 后插入 `reconstructor.reconstruct(blocks)` 调用（~5 行） |
| `pipeline_config.py` | 新增 `TABLE_RECONSTRUCTION` 配置节（开关、检测阈值） |

### pipeline_config.py 新增配置

```python
TABLE_RECONSTRUCTION = {
    "enabled": True,              # 总开关
    "min_numeric_ratio": 0.4,     # 数值占比阈值
    "max_chunk_len": 200,         # 碎片最大长度
    "group_max_gap": 2,           # 同页分组允许的最大 chunk_index 间隔
}
```

### 数据流示例

```
输入 blocks (来自 cleaner):
  block_031: type=text, "[华东] 25,640.053.20%"
  block_032: type=text, "[华南] 10,220.021.20%"
  block_033: type=text, "[华北] 7,130.014.79%"
  block_034: type=text, "[西南] 5,210.010.81%"

↓ 规则快筛：4 个块都满足表格碎片条件
↓ 分组：同 page=3，连续 chunk_index → 一组
↓ Agently 重建：

输出 blocks:
  block_table_001: type=table, content=
    "| 区域 | 收入(万元) | 占比 |\n|---|---|---|\n| 华东 | 25,640.0 | 53.20% |\n| 华南 | 10,220.0 | 21.20% |\n| 华北 | 7,130.0 | 14.79% |\n| 西南 | 5,210.0 | 10.81% |"

↓ smart_chunker.split_table_block() 自动处理
```

### 容错设计

- LLM 调用失败 → 保留原始 blocks，不阻断 pipeline
- LLM 返回空/无效 Markdown → 检测格式合法性，不合法则回退
- 配置 `enabled=False` → 完全跳过，零影响
- 每次 LLM 调用记录日志（输入碎片、输出表格、耗时）

### 验证方案

1. 用当前年度报告数据测试：重新解析 → 检查表格 chunk 是否为 Markdown 格式
2. 在 UI 中搜索"西南地区收入"：验证能否命中重建后的表格 chunk
3. 运行现有测试：`python -m pytest tests/ -v` 确认无回归
4. 新增 `tests/test_table_reconstructor.py`：测试规则快筛、分组、LLM 调用（mock）
