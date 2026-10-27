# 改动台账

> 目的：回答一个问题——「这还是不是我的项目」。
> 本文件记录每个文件的来源与改动原因，配合 git 历史可逐行核对。
> 基线提交：`baseline: 检索链路接通后的可用状态`（122 个文件，1.1M）

分类只有三种：**原始未动** / **已修改**（在原文件上改） / **新增**（本轮新建）。
判断标准是「行为是否改变」，不是「有没有被我打开过」。

---

## 一、原始未动（项目骨架，约 3,600 行）

这些模块一行没改，它们是系统能跑起来的原因：

| 文件 | 行数 | 职责 |
|---|---|---|
| `surgical_qa/vector_index.py` | 439 | Milvus 向量索引与检索 |
| `surgical_qa/hybrid_search.py` | 274 | BM25 + 向量混合检索 |
| `surgical_qa/access_control.py` | 465 | ACL 策略与 engine-side 过滤表达式 |
| `surgical_qa/security.py` | 265 | PII 脱敏与安全网关 |
| `surgical_qa/pipeline.py` | 674 | 文档入库流水线 |
| `surgical_qa/cache_manager.py` | 119 | 解析缓存 |
| `surgical_qa/task_executor.py` | 99 | Agently 结构化输出封装 |
| `surgical_qa/app.py` | 1,070 | Gradio 界面 |
| `surgical_qa/smart_chunker.py` | 493 | 旧版切块器（当前主流程未使用，保留） |
| `tests/` | — | 原有单元测试（ACL / BM25 / 安全 / 评测 / 熔断） |

`milvus_data.db`（0 字节占位）、`models/`、`volumes/` 均为运行时资产，已加入 `.gitignore`。

---

## 二、已修改（函数级改动，约 250 行，接口全部不变）

| 文件 | 改了什么 | 为什么 |
|---|---|---|
| `surgical_qa/retriever.py` | ① `_diversify()` 单文档配额<br>⑧ 向量通道异常降级为仅 BM25 并置 `degraded` 标记<br>⑨ 生成温度读 `LLM_TEMPERATURE`<br>② `_merge_dedup()` 回填 `doc_name`<br>③ `generate_answer()` 改为纯流式生成器<br>④ `add/update/restore` 后触发版本治理<br>⑤ `_record_to_candidate()` 缩进修复<br>⑥ 空查询短路返回空<br>⑦ 改写链路：原查询必参与召回 + 跨查询 RRF 融合 + 无历史时跳过改写 | ① 年报碎片挤占 top-k（详见四）<br>② Milvus payload 无 doc_name 导致引用为空<br>③ 原函数有 yield 却保留 `stream=False` 分支，非流式时返回生成器对象<br>④ 新文档入库后版本规则自动生效，不依赖手工脚本<br>⑤ 原代码缩进错误，backfill 分支实际不可用 |
| `surgical_qa/reranker.py` | 新增 `_document_text()`：候选文本 = 文档名 + 章节标题 + 正文（截 2000 字） | 表格/列表碎片脱离语境后误命中（如「3 天以上：部门负责人→人事部审批」被任何含「审批」的问题命中） |
| `surgical_qa/document_parser.py` | ① 新增 `_split_text_sections()` / `_is_section_heading()` / `_chunk_table()`<br>② 重写 `_chunk_by_sections()`<br>③ `classify()` 将 `.txt/.md/.rst/.csv/.xlsx` 路由到 docling | ① 按章节切块 + 伪标题识别 + 表格按行对齐并重复表头<br>② 原切块固定 500 字窗口，把表格行和条款拦腰截断<br>③ `unstructured` 在本机 import 即 SIGTERM，原路由会把 txt/csv 送进崩溃路径 |
| `surgical_qa/evaluator.py` | `_resolve_expected()` 支持按制度名动态解析；`details` 增加 `top_score` | 原期望值绑死 `doc_id`，文件一换全量对不上，评测跑出来是空转 |
| `surgical_qa/data_loader.py` | `full_text = 制度名 + 章节标题 + 正文` | 与 reranker 同理，让向量化时带上语境 |
| `surgical_qa/config.py` | SYSTEM_PROMPT：删「可基于通用知识补充」「追问 1~2 个问题」；新增 `PARSER_DISABLED`、`MINERU_ENABLED=False`（云端解析默认关闭）、`LLM_TEMPERATURE=0.3` | 答案级评测 28 条实测：7 条 partial 中 5 条是模型把常识当制度写（病根就是这两句 prompt）；unstructured import 即崩无法运行时探测 |
| `surgical_qa/retrieval_config.py` | 新增 `DIVERSITY` 配置块（enabled / max_per_doc=2）；新增 `REWRITE` 配置块 | 把配额与改写开关做成可配置，不再散在代码里 |
| `surgical_qa/query_rewriter.py` | 三次串行调用（上下文改写→指代消解→多意图分解）合并为**一次结构化调用**；schema 第三槽改用 `"not_null"`；保留 `_sanitize()` 兜底 | ①串行三次使多轮检索 4.04s，合并后 2.26s 且改写与拆解同源不矛盾；②实测 `True` 只校验键存在，5 次有 2 次返回空元素，`"not_null"` 后 0 次（框架自动重试）；③详见 `AGENTLY.md` |

---

## 三、新增（约 1,145 行，占全项目 9%）

| 文件 | 行数 | 职责 | 为什么需要 |
|---|---|---|---|
| `surgical_qa/version_policy.py` | 208 | 文件名解析制度名与版本号，判定现行版本；含冲突检测与命名归一 | 同一制度 V1.0/V2.0/V2.1 同时入库且都能召回，答出旧标准 |
| `apply_version_policy.py` | 151 | 治理 CLI（`--apply/--cleanup/--purge`），`--purge` 归档旧版原件并删解析缓存 | 存量数据一次性纠正；旧版文件与缓存需要真正离开主目录 |
| `rebuild_index.py` | 147 | 强制重解析并重建索引（`--dry-run/--apply/--only`） | 原脚本存在 doc_id 依赖 mtime 的问题，会写出孤儿缓存 |
| `api_server.py` | 257 | FastAPI：`/api/chat`、`/api/chat/stream`(SSE)、`/api/search`、`/api/documents` | 前后端对接入口，前端不必等后端改完再开工 |
| `verify_version_fix.py` | 148 | 17 项版本解析验证（含未见过的文件名、V10.2、.docx/.pdf） | 证明版本规则不依赖现有这 47 份文件 |
| `run_baseline.py` | 69 | seed_v2 基线评测 | 可重复的量化基线 |
| `run_persona_eval.py` | 137 | persona 集评测（含拒答判定） | 按不同身份的问法检验，而不只用出题人的问法 |
| `calibrate_ab.py` | 125 | 参数校准（alpha / top_k） | 验证参数不是拍脑袋定的 |
| `surgical_qa/eval_sets/seed_v2.json` | 28 条 | 检索评测集，绑制度名而非 doc_id | 换文件不失效 |
| `surgical_qa/eval_sets/seed_persona.json` | 70 条 | 五类身份问题（59 可答 / 11 应拒答） | 覆盖真实问法差异与「无答案该拒答」 |
| `ab_rewrite_eval.py` | 152 | 改写链路 A/B（无改写 vs 带改写，逐条差异 + 延迟） | 量化「改写到底有没有用」，而不是凭感觉 |
| `AGENTLY.md` | 110 | Agently 能力评估：已用/候选/不适用 + 实测数据 | 记录为什么只用契约层，以及 `True` / `not_null` / 列表 `[*]` 三种写法的边界 |
| `run_regress.py` | 210 | 分层回归入口 L0/L1/L2/L3，与基线自动对比 | 改一行代码不该跑全量 6 分钟；基线存在 `eval_reports/regress_baseline.json` |
| `sensitivity_scan.py` | 165 | 参数敏感度扫描（max_per_doc/min_score/top_n/alpha） | 证明指标不是靠某个特定数值撑起来的 |
| `parse_health_check.py` | 88 | 解析健康检查：逐块统计 HTML 标签占比，超标退出码 1 | 查证「关掉 mineru 损失多少」时发现 4 份 PDF 的块被 HTML 表格标签污染（最高 73%），此前无任何指标能看见 |
| `pdf_parser_compare.py` | 90 | 同一份 PDF 用 docling 重解析并与 mineru 结果对比 | 用数据回答「少两个解析器损失多少」，而不是凭印象 |
| `judge_calibration.py` | 150 | 裁判人工校准：导出打分表 + 算 Cohen's κ | 同模型自评必须和人工对过账才能当门禁 |
| `archive/`（13 个脚本） | 1,553 | 未接入主链的探索脚本归档（附 README 说明每个的用途） | 主链路的可达性从 86% 提升到「除归档外全部可达」 |
| `answer_judge_eval.py` | 185 | 答案级评测：检索→生成→LLM 裁判（忠实性/完整性/引用/拒答） | 召回指标只验「找到没找到」，无法回答「答得对不对」。28 条实测暴露了 7 条召回指标看不见的问题 |

---

## 三点五、版本治理的两次加固（2026-10-22 晚，用户实际试用后）

用户问「不需要的版本能不能删」时做了一轮冲突场景验证，发现两个会导致**看不到最新制度**的缺陷：

| 场景 | 修复前 | 修复后 |
|---|---|---|
| 上传 `员工手册_V3.0`，但 `V2.1_现行最新` 没改名 | **V2.1 继续当现行，V3.0 被静默判为旧版** | 版本号高者胜出，并在 CLI / 上传响应里告警 |
| `员工手册_final.pdf` 与 `员工手册_V2.0.pdf` 并存 | base_title 不同（`员工手册_final` vs `员工手册`），**两份同时判为现行** | 归一化掉 final/最新版/终版/定稿等修饰词，合并为同一制度 |
| `员工手册_现行最新.txt`（无版本号）+ `员工手册_V2.0.txt` | 标记未被识别，V2.0 胜出 | 无版本号时也认「现行最新」；两版无法比较时尊重人的标记 |

配套改动：
- `api_server.POST /api/documents` 新增 `version_status` / `superseded_others` / `version_conflicts` / `notice`，
  上传即刻告知「你是现行版」还是「你被判为旧版」，不再只能事后翻 registry。
- `apply_version_policy.py --purge`：旧版原件移入 `document_store/superseded/` 并记 `archived_to`（可恢复），
  parsed_cache 直接删（可由原件重建）。**registry 记录保留**——删了会让 `superseded_by` 指向不存在的文档、审计链断裂。

存量清理结果：uploads 21 个旧版原件归档、21 份缓存删除，目录对齐为 `26 文件 = 26 现行版`。
另在 `document_store/orphans/` 隔离 2 个从未入库的重复副本（hash 与 registry 不匹配，未删）。

验证：`verify_version_fix.py` 17/17 通过；新增 10 项冲突/命名场景全部 PASS；真实 registry 前后现行版集合完全一致（26 → 26）。

## 四、三个真实问题的根因与修法（是否过拟合）

| 问题 | 现象 | 修法 | 是否依赖本批数据 |
|---|---|---|---|
| PDF 碎片挤占 top-k | 人设集 75.61%，比预期低 21pt | 单文档配额 + rerank 输入带语境 | 否，机制型 |
| 表格被 500 字窗口拦腰截断 | CSV 金额与部门落在不同块，检索到也无意义 | 按行累加、每块重复表头 | 否，只依赖 Markdown 表格语法 |
| 公式/碎片被当章节标题 | 一张表切成几十个十几字碎片，短文本反而高分 | 伪标题识别（长度 + 正文前缀校验） | 否，启发式通用 |

三个都是**机制型补丁**：一次消掉一整类问题，不会因单例复发。全项目没有一处 `if 某关键词 in query` 式的特例分支。

---

## 五、已知遗留与处置（如实记录，2026-10-23 复核更新）

> 本节按「现状」维护，不再保留已解决项的旧描述。历史过程见 git log。

1. ~~**`unstructured` 仍在降级链里**~~ **已修**：`config.PARSER_DISABLED = ['unstructured']`（import 即 SIGTERM，无法运行时探测，只能显式禁用）；`FALLBACK_ORDER` 构造时过滤、`classify()` 回退 docling、`get_parser_status()` 跳过。
2. ~~**PDF 默认走 mineru 云端**~~ **已关闭，且污染已根治**：`MINERU_ENABLED=False`；根因是 mineru 对复杂表格输出 HTML `<table>`，切块器只认 Markdown 管道表，标签被写进正文（4 份 PDF 共 70 块，50 块严重污染，标签占 54%~73%）。
   - 已给 `document_parser.py` 加 `html_tables_to_markdown()`（标准库 HTMLParser），4 份 PDF 用 docling 重解析：**70→49 块，全库 237→216 块，标签占比 0.0%，脏块 0**（`parse_health_check.py` 复检）。
   - 现在若要重新启用 mineru，转换函数已在，污染不会复发。
3. ~~**评测集标注错误 2 条**~~ **已修**：`v2-015`/`v2-017` 移入 persona 作拒答题（72 条，拒答 13），seed_v2 剩 26 条全应可答。
4. ~~**裁判为同模型自评，κ 未出前不当门禁**~~ **已校准（2026-10-23）**：人工复核 20 条（10 grounded / 9 refused / 1 partial，两类各 ≥5 满足门槛），与裁判判定 **20/20 全一致，Cohen's κ = 1.000，一致率 1.000**，0 分歧。
   - 已做：裁判温度固定 0、生成温度统一 0.3（此前同配置两次差 15.4pt 即由此而来）；导出改分层抽样 + `MIN_PER_CLASS=5`，避免 19:1 样本算出假 κ=1.000。
   - 复核留痕：`judge_calibration_review_20261023.md`（人工逐条判断）+ `judge_calibration_answer_judge_seed_persona_20261023_112000.csv`（含 human_verdict 列）。
   - 附注：κ=1.000 的效力限于这批样本与当前口径——20 条全部人工认同裁判，意味着没有测到裁判出错的方向；边界案例（p-004「推算=无依据」的人工认同）确认的是口径本身，而非裁判的抗错性。裁判可作趋势指标并逐步用于门禁，换裁判模型（避免自偏好）仍是后续加固方向。
5. ~~**Agently 改写链路未被评测**~~ **已测**：单轮改写负收益（98.31%→96.61%，+1.7s），多轮有效（4/5→5/5）；已收窄为「仅有历史时触发」，三次串行调用合并为一次结构化调用，多轮延迟 4.04s→2.26s。
6. ~~**`alpha` 双来源 / `chunk_size` 散落**~~ **已修**：`alpha` 读 `RRF["alpha"]`；`chunk_size/overlap` 全项目只在 `config.py` 定义一次，`document_manager`、`document_parser`、`DocumentManager.__init__` 均从 config 取。
7. **语义鸿沟个例（未解，已知边界）**：p-037「照片发朋友圈」、ext-005「文档转第三方」、ext-006「微信发资料」被《对外沟通话术规范》带偏。
   - 2026-10-23 定量确认：`微信传输保密信息` rank1=0.9069、`可以用微信传文件吗` rank1=0.6695，但口语原句 `资料我直接微信发你们对接人，可以吧` 最高仅 0.4253、正确块不在前 6 → 是**口语化短句的召回边界**，不是代码缺陷；单轮改写实测负收益，故不动。
8. ~~**孤岛脚本 1,071 行**~~ **已归档**：13 个早期探索脚本移入 `archive/`（含 README 说明），主目录只留主链路与评测工具。

---

## 五点五、存量文件的最终处置（2026-10-23）

| 对象 | 结论 | 依据 |
|---|---|---|
| `document_store/orphans/` 两份（`示例智能2026年三季度经营快报.pdf`、`…半年度经营分析报告_1792486916.pdf`） | **已删除（回收站，可还原）** | md5 与 uploads 现存文件字节级相同（`b7a2f916…` / `0e8b23b3…`），是同名重复上传的副本，无独立内容。`orphans/` 已空 |
| `document_store/superseded/` 21 份旧版原件 | **保留** | 21 文件 ↔ 21 条 `archived_to` ↔ 21 条软删除记录，一一对应；总计仅 92KB，是 `restore` 的实体来源。删掉省不到空间，却让 restore 失去原件、`archived_to` 变悬空；真要删必须同时清 `archived_to`，等于放弃「可恢复」承诺 |
| `judge_calibration.py` 的 κ | **待人工作业** | 机器自评替代不了人工标注，复核单已备（见第五节第 4 条） |

---

## 六、本轮审查修复（2026-10-22 晚，全项目复查）

按「用户能否感知」排序，前三条是会直接出错或被用户看见的。

| # | 问题 | 根因 | 修法 |
|---|---|---|---|
| 1 | **关了 mineru 却仍会走云端** | `pipeline` 的路由规则把 PDF 指向 mineru，而 `dispatcher.parse(parser='mineru')` 只查 `is_available()`，不查 `PARSER_DISABLED`；只要 `MINERU_API_KEY` 还在环境里就为 True → 文件被真传到 mineru.net | `MultiPipelineDispatcher.parse()` 入口加禁用过滤，显式指定的解析器被禁用时退回自动分类；`_next_fallback()` 对「起点被禁用」补通用备选（原先 key 被过滤后返回空，等于无回退） |
| 2 | **4 份 PDF 索引仍是 HTML 标签** | mineru 的复杂表格是 HTML `<table>`，而切块器只认 Markdown 管道表 | 新增 `html_tables_to_markdown()`（标准库 `HTMLParser`，支持嵌套与未闭合兜底），在 mineru 出口与 `_chunk_by_sections()` 两处调用；随后用 docling 重解析这 4 份：**70 块 → 49 块，标签占比 45.5% → 0.0%，严重污染块 50 → 0**，全库 237 → 216 块 |
| 3 | **检索为空仍在调模型** | 空上下文照样进 LLM，只能靠 Prompt 让它说「未找到」 | `app.py` 与 `/api/chat`、`/api/chat/stream` 三处加零召回短路，由代码返回统一口径，顺带省一次生成调用 |
| 4 | **信息充足仍以反问收尾** | 旧 Prompt 写「最多 1 个追问」，模型把它理解成「可以追问」；实测 2/2 都有（「请问您是想了解…还是…？」） | 改为显式禁例：资料能答就答完即止，列出三种禁止的客套句式；只有核心结论给不出时才问，且只问一个 |
| 5 | **引用显示原始文件名** | 引用用 `doc_name`（`003_…_V2.1_现行最新.txt`） | 改用 `base_title` + 版本号（`员工考勤与假期管理规范 V2.1`）；Milvus payload 缺这两个字段，在 `_merge_dedup()` 统一从内存 records 补齐；`/api/chat` 的 references 新增 `display_name` |
| 6 | **API 无 CORS** | 前端跨域会被浏览器拦掉 | 加 `CORSMiddleware`，来源由 `CORS_ORIGINS` 控制（默认 `*`） |
| 7 | **上传全量读内存 + 阻塞事件循环** | `await file.read()` 一次读全量；`add_document` 是同步的（docling 单份可跑 200s） | 改为 1MB 分块落盘 + 200MB 上限校验 + 后缀白名单；解析丢 `run_in_threadpool` |
| 8 | **HybridSearcher 白占一份 BM25** | 构造时 `bm25.build(records)`，但主链路只用 `self.bm25`，它仅服务「策略对比」面板；且文档增删后不重建，面板看的是旧数据 | 改懒加载，`_refresh_document_records()` 时失效；实测启动日志中 BM25 只构建一次 |
| 9 | **重建索引后缓存仍命中** | cache key 只有 `query+department`，不知道底层语料变了 | key 加入语料指纹（`chunk_id` 集合摘要），文档增删/重建索引后旧缓存自动失效；同时清空了当时残留的 26 条 |
| 10 | **`chunk_size` 仍有一处硬编码** | `DocumentManager.__init__(chunk_size=500, overlap=50)`，且自行重算了一遍 `DOCUMENT_STORE_DIR` | 默认值改在调用时读 `config`，路径改为 `from config import` |
| 11 | **README 是两份项目拼在一起** | 前半是企业知识库，后半是旧医疗问答；还引用了不存在的 `knowledge_manager.py`、`architecture.png` | 重写为当前真实状态：启动方式、链路、目录、环境变量、评测命令、已知边界 |
| 12 | **运行时文件入库** | `document_registry.json.bak_*`（registry 手动备份）与 `surgical_qa/cache/qa_cache.json`（问答缓存）都被 git 跟踪，每次提问都会产生 diff | `.gitignore` 加 `*.bak_*` / `*.bak` / `surgical_qa/cache/`，对应文件 `git rm --cached` |

**查过但不改的**（附理由，避免下次重复怀疑）：

- **API 无鉴权、`user_role` 由调用方自报**：改动会波及前后端契约，且正确做法是由网关/会话解析身份。已在 `api_server.py` 文件头与 README「已知边界」写明——可信内网可用，公网暴露前必须加认证。
- **PII 脱敏把 16~19 位纯数字当银行卡号**：全库 216 块实测 0 命中，当前无误伤。收紧规则会降低脱敏能力，不敢在无样本时改。
- **两条入库路径切块器不同**：`rebuild_index.py` 走 `document_parser._chunk_by_sections()`，正常上传走 `pipeline → smart_chunker`。都能产出合格块，统一需重构入库管线，收益不明，先记录。
- **2 个孤儿文件**（`document_store/orphans/`：`示例智能2026年三季度经营快报.pdf`、`..._1792486916.pdf`）：hash 与 registry 不匹配，已隔离未删，等用户确认。

---

## 六点五、registry 状态治理（2026-10-22 深夜）

起因：`document_registry.json` 被 git 跟踪这件事是否合理，以及 `*.bak_*` 无限累积。

**结论一：registry 保留在版本库，不移出。**
它是知识库唯一的状态源（doc_id、版本、权限、解析器、软删除、`superseded_by`）。
删掉它 `uploads/` 里的 26 份原件不会被自动识别，知识库直接变空 —— `rebuild_index.py`
收集现行文档也是从它取。移出版本库省下的是「多人协作时的合并冲突」，
这个场景当前不存在；丢掉的是「误删文档/误治理可回滚」，这个是现在就会用到的。
取舍已写进 README「已知边界」，并写明：将来真要多人共享，正确做法是移出版本库、
由服务端独占管理并配套导出/导入，而不是手工合并 JSON。

**结论二：修复自动治理没有回滚点。**

| # | 问题 | 改法 |
|---|---|---|
| 13 | 上传触发的自动版本治理 `retriever._enforce_version_policy()` 直接 `registry._save()` 写盘，不留备份。文件名版本号解析错了，旧版被静默退场，只能靠 git —— 而 git 未必有这中间状态的快照 | 抽出 `version_policy.backup_registry()`，自动治理（有 `newly_superseded`/`restored` 时）与 `apply_version_policy.py` 共用；写盘前先快照 |
| 14 | 备份无清理，每次治理一份，长期累积 | `prune_registry_backups()` 只保留最近 `BACKUP_KEEP=5` 份，`backup_registry()` 内自动调用；另加 `python apply_version_policy.py [--apply] --prune-backups` 手动入口 |
| 15 | `DocumentRegistry._save()` 直接 `open('w')` 覆盖写。中途断电或写满会把 registry 截断，而它是唯一状态源，截断等于整库状态丢失且无原文可重建 | 改临时文件 + `flush`/`fsync` + `os.replace()` 原子替换；实测无 `.tmp` 残留 |

备份与清理的实现只有一份（`surgical_qa/version_policy.py`），治理脚本改为 import，
不重复维护两套规则。

---

## 六点六、裁判校准的抽样修正（2026-10-23 上午）

起因：想算一次 κ，导出的 20 条里 19 条 grounded、1 条 refused，κ 直接算出 1.000。

**问题不在数字，在抽样。**

| # | 问题 | 改法 |
|---|---|---|
| 16 | `export()` 是「非 grounded 全取 + grounded 补足」。报告里非 grounded 一多，grounded 就被挤成 0；样本单边时 κ 恒等于 0 或 1，看着漂亮，实际什么都没测 | 改分层抽样：两类大致各取一半；用固定种子打乱顺序，避免「排前面的就是有问题的」这种顺序暗示 |
| 17 | 样本极度不平衡时仍照常输出「κ≥0.6 可作门禁」的结论，前后矛盾（19:1 的样本给 1.000 还建议当门禁） | 加入 `MIN_PER_CLASS=5`：较少一类不足 5 条时，只报 κ 数值、明确写「暂不下一致性结论」，不给门禁建议；导出时同样提前警告并提示先跑 `answer_judge_eval.py --set seed_persona` |

为什么门槛是 5：κ 是二分类一致性，少的一类只有 1~2 条时，一两条分歧就能把 κ 从 0.9 打到 0.4，
置信区间比结论本身还宽。这类数字写进报告只会误导决策，不如不给。

实测：19:1 样本 → 打印警告 + 「样本量不足，暂不下结论」；造 10:10 均衡样本 → 正常给出可接受结论。

---

## 六点七、展示名统一（2026-10-23 中午）

起因：文档管理页与 `/api/documents` 显示的是 `示例智能2026年三季度经营快报_1792565911.pdf`，
而对话引用已经是「示例智能2026年三季度经营快报」。同一份文档在两个界面叫两个名字。

根因不是某一处写错，而是**同一套命名规则被抄了四份**（API 引用、API 列表、Gradio 引用、生成上下文），
每份的兜底和清洗程度还不一样。

**改法**：`surgical_qa/version_policy.py` 新增 `build_display_name(base_title, version_label, fallback)`，
四处全部改为调用它。规则只此一份：

| 脏值 | 来源 | 处理 |
|---|---|---|
| 扩展名（`…快报_1792565911.pdf`） | 部分路径把完整文件名当兜底传进来 | 剥扩展名（防御性处理，当前 26 份的 base_title 实测均无扩展名） |
| `_1792565911` | 同名重复上传时 `int(time.time())` 落的后缀 | 剥 9~11 位纯数字后缀 |
| `_final` / `_最新版` / `_副本` | 人工命名习惯 | 复用已有的 `_normalize_title()` |
| 版本号 `2.1` / `V2.1` | 注册表 | 统一成 `V2.1` 追加在名字后 |

**为什么只认 9~11 位数字**：`报销制度_20260101` 这种日期后缀是名称的一部分（8 位），
最初写成 `\d{6,}` 会把它一起剥掉，两份不同月份的表就变成同名了。实测用例已覆盖。

**为什么不在 registry 里改**：只改了读取侧。`base_title` 里的脏值是历史数据，
改它要动 registry，会牵到 file_hash 与版本判定；而展示层洗一遍就能解决，没必要碰状态源。

顺带修好的：回答正文里模型自己写的引注（`[示例智能2026年三季度经营快报_1792565911, 章节：…]`）。
模型是照抄喂进去的上下文，所以清洗点放在 `retriever._build_context()` 的 `cite_name` 上，
不是事后替换文本——那样正则又得多一份。

**验证**：26 份现行文档显示名全部无时间戳残留；`报销制度_20260101`、`ISO9001_2015…` 等边界用例未被误伤；
L0 全过（12.1s，现行 26 / 检索 6/6 / 生成 473 字 / 拒答）。

---

## 六点八、企业测试数据批次①：Excel 解析攻坚（2026-10-24）

起因：用户引入「企业知识库测试数据」数据集（37 文件，30 份新语料：docx/xlsx/csv/md/pdf），
按「一次打穿一类」节奏先攻 9 份 xlsx。预检即命中致命问题。

**问题 1：docling 解析 Excel 全灭。** docling 把合并单元格标题复制到每一列
（一行表头膨胀 9 倍、2000+ 字符），经 500 字切块后工资表 18 块全是标题残片，
**员工数据一行都没进块**，且解析状态是 success——静默失败。

**修法**：`surgical_qa/document_parser.py` 新增 `OpenpyxlParser`（约 190 行）：
- openpyxl 直读单元格（`data_only=True` 取公式缓存值），合并单元格左上角值填充全区域
- 列头行判定：从上往下第一行「非空格数 ≥2 且无重复值」；合并填充特征行
  （整行单值或任一值重复 ≥2，覆盖装饰大标题与「单位/日期/制表」信息行）剥出表体
- 数值按 `number_format` 渲染（百分比 `0.2059075939248601`→`20.59%`），杜绝全精度尾巴
- 按行累加分块、每块重复表头；多工作表全解析
- 挂载：classifier 路由 `.xlsx/.xls` → openpyxl；`PARSER_MAP["excel_native"]` 改指 openpyxl；
  `FALLBACK_ORDER` openpyxl→docling；`config.SUPPORTED_FILE_TYPES` 注册
- **环境注意**：docling 装在系统 Python 3.10，managed 3.13 没有——解析相关命令必须用 3.10

**问题 2：占比列整列被脱敏打码。** 上面的全精度小数 `0.2059075939248601`
小数点后正好 16 位连续数字，撞上 `bank_card` 规则（`\d{16,19}`），
检索结果里占比全部变成 `***银行卡号***`。CHANGES 六点五记录过「216 块实测 0 命中、
无样本不敢改」——**现在样本来了**。
修法（双保险）：根因由 OpenpyxlParser 的格式化渲染消除；`security.py` 的 bank_card
正则收紧为 `(?<![.\d])\d{16,19}(?!\d)`——数字前是小数点必是浮点小数部分，
真实卡号前不会出现小数点，对真卡号零损失。`tests/test_security.py` 23 用例全过。

**问题 3：API 上传丢失原始文件名。** 上传先落盘到 `kb_upload_<uuid>.xlsx`，
registry 的 filename/base_title/版本治理全取 basename，导致 9 份文档全被注册成 uuid 名。
修法：落盘到 `kb_upload_<uuid>/原始文件名` 子目录，finally 清理整个目录。

**问题 4：双 hash 体系打架，幂等检查形同虚设。** `add_document` 用 MD5 查重，
registry 的 `file_hash` 却是 prescreen 写入的 SHA256——永远查不中。后果链条：
重复上传漏到 prescreen 才被拦 → 失败结果仍被 `registry.register` 无条件注册 →
doc_id（=SHA256 前 16 位）相同，**把正常记录覆盖成 parse_status=failed / chunk_count=0 的空壳**。
修法三处：① `add_document` 幂等检查改用 SHA256；② 解析失败不注册（不再覆盖好记录）；
③ `purge_document` 同步清 `data_pipeline.prescreen.known_hashes`（否则 purge 后
同文件立刻重传仍被进程内快照拦截）。

**验证**：9 份 xlsx 全部入库（68 块，工资表 18 脏块→4 净块）；registry 文件名全部正确；
重复上传同文件幂等返回已有 doc_id；检索实测「示例精密机械应收账款占比」→ `18.22%`/`20.59%`
完整可读；23 项安全测试全过。库现状：35 文档 / 284 块。

---

## 六点九、企业测试数据批次②：docx 入库 + SimHash 去重误删表格行（2026-10-24 中午）

16 份 docx（合同 5 / 人事行政 5 / 制度流程 5 / 立项报告 1）预检干净（docling 对 docx
的表格、层级标题处理正常），批量入库。随后做同域冲突验证时发现严重问题。

**问题 5：ChunkDeduplicator 把表格数据行当重复块删除。** 症状：
《差旅与费用报销管理制度》库内仅 4 块且索引跳跃（0,1,4,6），住宿费标准表格
（一线 600 / 二线 450 / 其他 350 元/晚）**整段数字丢失**，而解析原文完整——
静默失败第二例。根因：入库链路（DataPipeline 阶段 7）的 SimHash 近似去重
（海明距离 ≤3 判重复）对表格数据行失效——`| 一线城市 | 600 |`、`| 省会 | 450 |`、
`| 其他 | 350 |` 结构完全相同、只有数字不同，SimHash 对数字差异不敏感，整行被判重复。
mark 模式打标后统一过滤（注释还写着"保留所有"，自相矛盾）。

**受影响面**：批量对比 16 份 docx 库内块数 vs 修复后解析块数，**16/16 全部丢块**。
表单类最惨：新员工入职登记表 2→7 块（丢 71%）、请假申请单 3→7、报销单 3→6。
合同/制度类丢 1~6 块。批次① 的 xlsx 不受影响（OpenpyxlParser 每块重复表头，
表头差异足够大未被误判，属侥幸）。

**修法**：`pipeline.py` 的 `ChunkDeduplicator` 新增 `_is_table_chunk()`
（≥2 行 markdown 表格行即表格块），SimHash/MinHash 两条去重路径均豁免表格块；
修正 mark 模式的误导注释。修复后该制度 4→9 块，600/450/350 完整入库。

**重传**：停服 → purge 16 条 → 起服 → 重传（known_hashes 是进程启动快照，
外部 purge 必须配合重启，见六点八问题 4 ③）。16 份全部按修复后块数入库。

**验证**：检索「出差住宿费报销标准」→ 新制度 600/450/350 与旧 V1.8 的 400/300/250
同时召回（同域竞争如预期，留给 42 题口径辨析题检验）；「张思远什么学历」→
教育经历表格块命中。库现状：51 文档 / 504 块。

---

## 六点十、企业测试数据批次③ + 42 题基准评测（2026-10-24 下午）

**批次③**：5 份入库——产品价格表/供应商名录/员工通讯录（csv，共 47 块）、
公司简介（md，20 块）、XS-100 技术规格书 PDF（docling 106s，13 块，参数表完整）。
**至此 30 份新语料全部入库，三个批次收官。**

**42 题测试集转换**：外部提供的 42 题测试 md（未随仓库分发，含业务信息）用一次性脚本
转成评测集 JSON。转换三坑：同行多字段需 lookahead 到下一个 `**`、
行尾吞下一个章节标题（`## xxx`）、关键词用 `/` 分隔需保留「450元/晚」类组合词。
类目：事实 22 / 表格 10 / 计算 5 / 干扰辨析 1 / 口径辨析 1 / 口径溯源 1 / 幻觉抑制 2；
难度 easy 9 / medium 26 / hard 7；ext-041/042 标 `expect_refusal`。

**基准成绩（首轮，2026-10-24 11:36 评测报告）**：

| 指标 | 值 |
|---|---|
| grounded 率 | 0.925（37/40，拒答题除外） |
| faithfulness / completeness | 98.5 / 98.3 |
| citation 正确率 | 100% |
| 幻觉抑制 | 2/2 正确拒答，0 over_answer |
| 平均生成耗时 | 2.1s |

**3 条非满分定性（全部是召回/切块缺口，零编造）**：

1. **ext-020 误拒**：「订单金额最高的客户」——「按客户统计」工作表在库、检索 top8
   也能召回（rank3），但单文档配额把明细页块优先装进上下文，统计页被挤出，
   模型只能如实说"资料不足"。改进方向：表格汇总类块（sheet 级聚合）加权。
2. **ext-028 partial**：合同有效期条款（第二条 2.2）未进召回，模型如实答"未找到"。
3. **ext-032 partial**：租金单价 38.5 元/㎡ 落在块边界被截断，模型引用了截断值。

**遗留观察项（已全部处理，2026-10-24 下午）**：
- ~~2 条活跃文档文件名带时间戳后缀~~ → 已重命名为干净名（10 条：9 xlsx + 1 pdf），
  filename/file_path/base_title 三处同步，API 列表 0 脏名；
- ~~Milvus 745 向量 vs BM25 584 块~~ → 已按 parsed_cache 全量重建（drop+insert，
  块结构不动），584 向量与 BM25 完全一致，161 条孤儿清零；
- uploads 顺带清理：18 个重复实体（同 hash 三代副本）移入 `_uploads_trash_20261024/`；
- 新发现机制缺口待修：`purge_document` 只删 registry 与缓存，**不清 uploads 实体文件**，
  这是重复实体越积越多的根源。

---

## 六点十一、72 题 persona 回归：同域竞争的首次实测（2026-10-24 下午）

三批新语料入库后跑老基线回归（report: `answer_judge_seed_persona_20261024_134715.json`）：

| 指标 | 10-23 基线 | 本次回归 | 判读 |
|---|---|---|---|
| grounded 率 | 0.8983 | **0.9322** | +3.4pt，无退步 |
| faithfulness | 99.2 | 97.5 | 微降（裁判严格度波动内） |
| completeness | 96.2 | 95.0 | 微降 |
| citation 正确率 | 100% | 100% | 持平 |
| 拒答正确 | 13/13 | 11/13 | **2 条漏拒，见下** |
| over_answer | 0 | 2 | 与漏拒同源 |

**2 条漏拒（p-047 竞业补偿金 / v2-017 商务舱）定性：预期标签过期，非系统退步。**
逐条款核验：p-047 回答引用的「30% / 24 个月 / 每月 15 日前支付」在今日新导入的
《保密与竞业限制协议_李维安.docx》第 4.1/4.2 条**原文存在**；v2-017 引用的
「经济舱上限 / 1.5 倍规则」在《差旅与费用报销管理制度 V3.0》3.3 条**原文存在**。
两条回答事实上正确且全部有据——旧库无这些制度所以当年标了 expect_refusal，
新语料补上真实答案后拒答前提消失。这正是「同域语料竞争」的另一面：
**库扩了，拒答边界会移动，评测集标签必须跟着语料版本走。**

**处置**：seed_persona.json 中 p-047 / v2-017 的 `expect_refusal` 改 False、
补 expected_base_titles，条目 note 记录标签更新原因，metadata.label_updates 留痕。
拒答题从 13 → 11。回归结论：**今日全部改动（解析器/去重/脱敏/上传/hash）验收通过，
persona 新基线 0.9322。**

---

## 六点十二、存储与索引卫生收口（2026-10-24 下午）

目标：不清算块结构（保住 0.9322 基线），只清存储垃圾与索引孤儿。

**1. 文件名清洗（10 条）。** registry 与 uploads 实体中带时间戳后缀的文件
（`*_1792565911.pdf` 等，uploads 冲突重命名的历史产物）全部重命名为干净名，
filename / file_path / base_title 三处同步。完成后 API 文档列表 0 脏名。

**2. 重复实体清理（18 个文件 → trash）。** uploads 里同一 xlsx 存在三份实体
（`kb_upload_<uuid>.xlsx` uuid 时代 / 干净名 / `*_1792810364` 时间戳重命名副本，
三者 hash 相同），是批次①当天反复 purge+重传留下的。以 registry 引用的实体为准，
其余 18 个文件移入 `_uploads_trash_20261024/`（不直接删除，确认无误后可清空）。
教训：purge_document 只删 registry+cache，**不清 uploads 实体**——以后清理要带文件层。

**3. Milvus 孤儿向量清零。** 向量索引按现有 parsed_cache 全量重建
（`build_index` 先 drop 再全量插入，块结构不变）：745 → **584**，
与 BM25 584 块完全一致。161 条已删文档的残留向量清除。

**终态**：56 文档 / 584 块 / 584 向量 / 0 脏名 / 0 重复实体；检索抽查 doc_name
已带干净文件名。过程中学到的运维细节：Windows 路径比对必须 normcase
（`C:` vs `c:` 大小写差异曾让整目录被误判孤儿）。

---

## 六点十二、召回配额重构 + purge 实体清理 + 最小监控（2026-10-24 下午）

### 1. ext-020 表格汇总块修复（召回配额两级化 + rerank 池扩容）

**病灶追了三层**：汇总块被挤出不是配额一家的锅——
1. rerank `top_n=6` 太小：订单明细块（含客户名+金额）rerank 分天然高于汇总块，
   6 个名额全被明细占掉，配额层无材料可选 → 扩到 15（rerank API 对候选池全体
   打分，扩大 top_n 不增加调用成本，token 预算兜底）；
2. 配额粒度从「文档」细化到「文档×区块（parent_title/sheet）」：多工作表 Excel
   的汇总 sheet 与明细 sheet 结构迥异，按文档统一配额必然牺牲一方；
3. 区块配额 2→3：同一 sheet 被切块后各块是**互补**关系（工资表 3 明细块+合计块），
   配 2 会把 rerank 高分（0.753，全局第 2）的合计块压进 overflow 挤死（ext-019 差点
   回归）；文档级总量上限 5 防多区块洪水。

**双基线验证**（配置生效后全量回归）：
- 外部 42 题：grounded 0.925（与上午持平）但成分更优——ext-020 refused→grounded、
  ext-032 partial→grounded；faith 99.1 / citation 100% / 拒答 2/2
- persona 72 题：grounded 0.9344（上午 0.9322） / 拒答 11/11 / over_answer 0

**残留 3 条 partial（定性归档，不再逐题调参）**：
- ext-024：docx 勾选框表单字段（请假类型 ☑）解析层丢失——需表单感知解析，与
  扫描件 OCR 同属"难格式攻坚"批次；
- ext-028 / ext-031：条款类文档被切成 ~14 个细粒度块，目标条款块排名靠后未进
  rerank 池/被文档配额拦下——系统性解法是**短块合并或条款窗口扩展**（分块策略
  课题），不是配额参数问题。

### 2. purge_document 补实体清理

`file_path.startswith(self.upload_dir)` 字符串前缀判断因相对/绝对路径差异永不命中
→ purge 只删 registry 不删 uploads 实体（重复实体的根源）。改归一化绝对路径判断。
端到端验证：注册→purge→registry/实体/缓存三清。
**事故记录**：purge 测试用副本与《公司简介.md》同 sha256（同 doc_id），purge 清掉了
原件——已从测试数据集重新导入恢复（56/584）。教训：**同内容文件的 doc_id 相同，
purge 前必须确认目标是副本还是唯一实体**。

### 3. 最小监控 + 看护进程（服务化第一步）

- `surgical_qa/metrics.py`：进程内线程安全计数器 + 滚动延迟样本（p95）；
  埋点：检索（次数/延迟/空结果/仅BM25）、LLM（成功/错误/延迟）、解析（成败/延迟）
- `GET /api/metrics`：指标出口，看 search_empty / llm_error / parse_fail 三个信号
- `serve.py`：看护进程——拉起 api_server 子进程 + 健康轮询（15s 间隔、3 次失败
  判死）+ 崩溃自愈（进程死亡 5s 退避重启、指数退避上限 300s），
  可注册 Windows 任务计划程序实现开机自启
- 实测：杀掉子进程 → 8s 内自动恢复（`logs/service_watchdog.log` 留痕）；
  /api/metrics 出数正常（search_latency avg 2811ms 首查含 BM25 冷启动）
- 93 项单测全过

---

## 六点十三、崩溃现场留痕：子进程日志落盘（2026-10-24 14:41）

**起因**：排查看护日志中 14:27:03 的"exit code=1"——结论是上轮自愈验收测试
的 taskkill /F（Windows 强杀退出码即 1），**不是真实崩溃**，服务无异常。
但排查暴露缺口：`serve.py` 子进程 stdout/stderr 直通看护终端不落盘，
`api_server.py` 日志也只配 stdout——真崩溃时 traceback 会随终端关闭丢失，
根因永远无法回溯。

**修复**：
- `serve.py`：子进程 stdout/stderr 重定向追加写入 `logs/service.log`
  （行缓冲，每次启动写分隔头，退出时关闭句柄）；
  看护日志的崩溃行新增"详情见 logs/service.log"指引
- 已重启看护进程验证：健康检查/业务请求正常写入 service.log

**遗留**：`api_server.py` 的 logging 仍为 stdout 单出口——依赖 serve.py
兜底已可回溯崩溃现场；若未来脱离看护独立部署（如 uvicorn 直跑/容器化），
需补文件 Handler 或结构化日志。

---

## 六点十四、规模压测工具链：合成语料 + CAIL 公开语料（2026-10-24 15:08）

**背景**：语料 56 份/1.7MB 对标企业级远远不够，且无真实数据。选定
A+B 双路线（全部零成本）：A=基于现有模板合成虚构企业文档；B=CAIL2018
免费公开法律文书（984MB zip 下载，流式抽样，不触碰百度 API/爬虫）。

**新增三个工具**：
- `generate_corpus.py`：合成企业语料生成器——每家虚构企业 7 份
  （简介 md / 销售台账 xlsx / 工资表 xlsx（合并单元格+number_format）/
  制度 docx（章条结构）/ 合同 docx（标的表）/ 表单 docx / 供应商 csv）。
  按企业序号派生随机种子 → 幂等可复现；progress.jsonl 断点续传。
  16 秒生成 203 份。
- `cail_sampler.py`：从 CAIL2018 zip 流式抽样判决文书转 txt
  （不全量解压 2GB 的 train.json），固定步长幂等，100 份 1.6KB/份。
- `run_stress_ingest.py`：压测执行器——阶段1 管线吞吐（DataPipeline
  全流程，不写 registry）；阶段2（--vector）嵌入+插入独立 collection
  `stress_chunks`，**主库 knowledge_chunks 与评测基线零接触**。

**小批量压测结果（303 份全成功）**：
- 合成语料 203 份：1.35 docs/s，1150 块，块延迟 p50 699ms / p95 1074ms
- CAIL 文书 100 份：1.05 docs/s，321 块，块延迟 p50 609ms / p95 951ms
- 向量阶段：150 块嵌入 14.9 块/s（DashScope text-embedding-v4）、
  Milvus 插入 3.75s——独立 collection 创建/插入链路打通
- 瓶颈：docling 解析（尤其 docx），放量到万级需考虑并发或换轻量解析

**踩坑记录**：向量插入两次失败——① 未显式调 `_ensure_collection`；
② schema 标量字段（department/question/page_number/truncated/version）
必须全量显式赋值，Milvus 的 schema default_value 对 insert 不生效。

**遗留**：压测语料未做检索质量评测（无问答标签）——放量后需合成
QA 对才能测召回衰减，这是下一阶段的课题。

---

## 六点十五、放量 1204 份 + QA 标签 + 向量召回基线（2026-10-24 15:52）

**放量**：generate_corpus.py 重构——各生成函数同步产出 QA 真值
（checkpoint 随行记录，汇总写 qa.jsonl），1204 份 / 2665 条 QA，75s。
QA 抽样验证：工资表 QA 与文档实际值逐字段一致。

**全量压测（1204 份 / 7075 块）**：
- 管线 1204/1204 全成功，1.65 docs/s，块延迟 p50 663ms / p95 1014ms
- **规模曲线平稳**：四分段 1.50 → 1.73 → 1.72 → 1.70 docs/s，无衰减
- 向量全量入库：7075 块嵌入 16.2 块/s（436s）、Milvus 插入 8.6s
- 独立 collection stress_chunks 共 7225 块（含前次小批量残留 150 块）

**向量召回基线（300 条 QA，纯向量通道，无 BM25/rerank）**：
- hit@1 64.0% / hit@3 75.3% / hit@5 79.0%
- 检索延迟 p50 7ms / p95 12ms（7225 块规模，Milvus 无压力）
- **未命中模式高度集中**：① 同结构表格跨公司混淆（台账 QA 大量
  返回别家公司的同结构台账——向量对"结构相同仅数字不同"无区分力）；
  ② 近名人名混淆（周慕言↔孙慕言）。**这正是主系统混合检索
  （BM25 词级精确匹配）设计要覆盖的场景**——下一层验证应把
  BM25 通道与 rerank 拉进来对比 hit@5 提升幅度。

**修复**：stage2 向量插入元组解包遗漏（本次重跑发现并修复）。

---

## 六点十六、检索架构对比实验：向量 vs 混合 vs 混合+rerank（2026-10-24 16:08）

**实验设计**：同一批 300 条 QA（带真值），7225→7075 块压测库，
生产同款组件（BM25Index/jieba + DashScope gte-rerank），三档策略：
A 纯向量 top-5；B 向量+BM25 等权 RRF；C = B + rerank(池 30→5)。

**结果**：

| 策略 | hit@1 | hit@3 | hit@5 | 延迟 p50 |
|---|---|---|---|---|
| A 纯向量 | 65.0% | 75.7% | 79.0% | 7ms |
| B 混合 RRF | 66.7% | 73.3% | 76.7% | 309ms |
| C 混合+rerank | **78.0%** | **80.7%** | **82.0%** | 370ms |

**结论**：
1. **rerank 的价值被量化确认**：hit@1 +13pp（65→78）、hit@5 +3pp，
   三档中全面最优——精排把"对的块"排到最前面，hit@1 已逼近 hit@5。
2. **BM25 等权融合在本文料上无净收益**（B 比 A 低 2.3pp）——
   预期的"订单编号/人名精确匹配"收益被等权 RRF 稀释。改进方向：
   α 权重向向量倾斜、BM25 改为兜底通道、或 jieba 专名/编号切分优化。
   不等于"BM25 没用"，是"当前融合配置下未兑现"。
3. 结论适用范围：压测语料 + 模板化 QA（问题含公司名/人名/字段名），
   与真实用户问题分布有差异，放量验证时需重新校验。

**踩坑（实验自身）**：首版把 Milvus 自增 id 与 BM25 行号两个 ID 空间
直接 RRF 融合 → B/C 虚低（67%/71.7%），加 id→行号映射修复后数据才可信。
教训：跨系统融合实验必须先对齐标识空间。

**新增**：`eval_hybrid_compare.py`（三档对比 + 未命中样例留存）。

---

## 六点十七、BM25 融合调优实验：α 扫描 + 兜底通道 + 一场虚惊（2026-10-24 17:55）

**实验一（压测语料，有效）**：`eval_alpha_sweep.py`，300 条 QA，
每查询一次向量池+BM25 池、内存内做 α 算术。结果：hit@5 随 α 增大
恢复——α=0.3 最差（72.3%）、0.5 等权 76.7%、**α∈[0.7,0.95] 稳定
79.7%**；hit@3 在 α=0.85 最优（77.0%，+1.3pp）。兜底通道（向量
top1 相似度 < τ 才听 BM25）**完全无效：τ=0.65 仅触发 1/300**——
向量 miss 是"高置信错误"（同构表格块相似度照样 0.9+），阈值拦不住。
α=0.7+rerank = 78.3/80.7/82.0，与等权+rerank 持平（rerank 是主杠杆）。

**实验二（主库 L1 回归，结论已被后续复查修正）**：α=0.85/0.7 两轮
"退化"（0.934/0.902）经 18:05 充值恢复后复查，**大部分是 artifacts**：
- **α=0.7 那轮（17:33）才是 BM25-only 降级轮**——账号异常起点在
  17:28~17:33 之间而非 17:33 之后，此前判断"0.85/0.7 两轮有效"有误；
- **α=0.85 轮（17:28）有效，但与充值后 α=0.5 复测（18:03）61 题逐位
  相同**——同进程探针证实融合层确实响应 α（候选集随 α 变化），但
  **rerank 会把融合顺序差异洗平**：只要候选集大体稳定，rerank 按语义
  分重排后最终输出与 α 无关。**生产结论：α 旋钮被 rerank 中和，保持
  0.5 即可；α 只在无 rerank 的纯融合场景有意义（压测实验一的结论
  适用范围即此）**；
- "退化"的另一半来自**基线过期**：regress_baseline.json 录于 10-22
  （59 可答题/26 文档/旧检索参数），今天扩容 56 文档 + 拒答标签
  13→11（可答题变 61）+ rerank top_n 15 等变更后未刷新。0.9344 与
  当日上午全量评测验证值一致，并非真实退化。

**DashScope 账号异常（已解决）**：17:33~17:57 嵌入/rerank 返回 400
"Access denied"（当日高频嵌入调用触发欠费/限额），用户充值后探针验证
恢复。服务重启后 L1 复测确认向量通道与 rerank 正常，**基线已按纪律
刷新**（seed_v2 recall 0.9808/mrr 0.9038；persona recall 0.9344/mrr
0.7874；拒答 11/11）。教训保留：向量通道降级是静默的，metrics 应加
"向量召回占比"告警；L1 基线在语料/评测集变更后必须同步刷新，否则
会把"合法变更"误报为"退化"。

**排查过程记录**：三轮 L1 数字逐位相同 → 先怀疑 qa_cache.json（证伪，
且确认 eval 路径全程零缓存读写）→ 逐条 diff 报告（0/61 差异）→ 同进程
monkeypatch 探针抓到 DashScope 400 真身；充值后再次探针发现 α 失效
→ 融合层候选集验证 → 锁定 rerank 中和效应与过期基线两个真因。

---

## 六点十八、并发压测 + 向量通道健康告警（2026-10-24 19:50）

**并发压测（`load_test.py`，真实 HTTP，72 条 persona 问题做载荷）**：

| 端点 | 并发 | RPS | p50 | p95 | 错误 |
|---|---|---|---|---|---|
| /api/search | 1 | 0.71 | 1132ms | 3193ms | 0 |
| /api/search | 10 | 2.98 | 1645ms | 3296ms | 0 |
| /api/search | 20 | 4.27 | 2180ms | 4506ms | 0 |
| /api/search | 50 | 4.72 | 2758ms | 4887ms | 0 |
| /api/chat | 1 | 0.31 | 2845ms | 4245ms | 0 |
| /api/chat | 10 | 2.32 | 3050ms | 5150ms | 0 |
| /api/chat | 20 | — | 3753ms | — | **1 请求挂起 240s** |
| /api/chat | 50 | 3.44 | 4093ms | 5530ms | 0 |

结论：**并发扩展性健康**——零错误率（除一次挂起）、延迟随并发优雅
劣化（search p50 1.1s→2.8s）、RPS 在 c=20~50 饱和于 ~4.7（瓶颈是
DashScope 嵌入+rerank 的 API 往返，非本地资源）。50 并发下 uvicorn
线程池未被占满（chat 仍 3.44 RPS）。

**唯一异常**：c=20 chat 一档，1/20 请求服务端挂起整整 240s 直到
客户端超时，其余 19 个正常，且 c=50 全清——非系统性限流，疑似
DashScope SDK 内部退避重试或瞬时网络停顿。**暴露改进项：LLM 调用
（`generate_answer`）无服务端超时**，挂起请求会占住线程池 worker
（anyio 默认 40 线程，50 并发下尚有余量，但生产应补超时+重试语义）。

**向量通道健康告警（修六点十七的遗留教训）**：
- `retriever._vector_recall` 异常路径新增 `vector_recall_error` 计数
  （与 `search_bm25_only` 的"真没命中"区分开）；
- `metrics.snapshot()` 新增 `health_signals` 派生段：
  `vector_channel_status`（unknown/ok/degraded，样本 <10 为 unknown，
  BM25-only 占比 >50% 判 degraded）、`vector_recall_ratio`、
  `vector_recall_errors`——/api/metrics 直接可读，降级不再静默；
- 端到端验证：重启后 12 发真实检索 → ratio 1.0 / status ok。
- 93 项单测全过；新增 `load_test.py`（可复跑，支持 --skip-chat 省费）。

---

**LLM 调用超时收紧（修六点十八抓到的挂起问题）**：
- 根因：OpenAI SDK 默认 timeout **600s**——服务端挂起时请求会占住
  线程池 worker 长达 10 分钟（压测实测 240s+）；
- 修复：`config.LLM_TIMEOUT_SECONDS = 60`，客户端级显式设置。对流式
  响应按阶段生效（连接/首 token、相邻 chunk 空闲间隔各自 ≤60s），
  不会掐断正常的长回答；
- `generate_answer` 新增 `APITimeoutError` 专项分支：计入
  `llm_timeout`（与一般 llm_error 区分），照常走既有重试+熔断链路；
- 验证：93 单测全过；黑洞地址实测超时按预期触发（挂起→报错→重试，
  不再永久阻塞）；重启后真实 chat 走通，回答与引用正常。

---

**企业级 Web 前端（替代 Gradio 作为用户界面）**：
- `static/`（index.html + app.js + style.css，零构建原生三件套，
  无 node/打包链依赖），`api_server.py` 末尾 mount `/` 同源服务
  （API 路由优先，无 CORS 问题），根路径直接打开界面；
- 三视图：**智能问答**（SSE 流式打字机 + 可展开引用卡片 + 身份切换
  演示 ACL + 建议问题）/ **文档管理**（列表 + 上传含版本治理提示
  supersedes/旧版退场 + 删除二次确认）/ **运维状态**（health 卡片 +
  向量通道告警灯 ok/样本不足/降级 + 计数器与延迟指标，15s 自刷新）；
- 定位分工：Gradio（surgical_qa/app.py 六 Tab）保留为开发运维台；
  新前端面向企业最终用户，只暴露问答/文档/状态三类能力；
- 验证：index/app.js/style.css 全部 200；SSE 帧格式（references →
  token* → done）与前端解析契约逐字段核对通过。

---

**修复三（严重级）**：`update_access` 只改 registry 不同步 Milvus——
而 ACL 引擎侧过滤读的是 Milvus 行上的 access_level，改了注册表
过滤器照旧命中（原代码注释里"Milvus 需同步更新"的 TODO 从未实现）。
新增 `VectorIndex.update_doc_access()`：查询→删除→重插（auto_id
主键不支持 update），并按 chunk_id 去重防御历史重复行。

**存量定密**：`apply_acl.py`（--apply 执行/默认预览）对 56 份活跃
文档自动定密，26 份升级：工资表→confidential（仅管理员）；合同/
竞业协议/入职登记/台账/财务经营报表等 23 份→restricted（按组分
人事部/财务部/法务部/管理层）；公司简介/技术规格书→public；30 份
制度/SOP 保持 internal。顺手清掉《公司简介》20 行历史重复向量
（584 恢复对齐）。

**入口加固**：`add_document` 未显式指定密级时自动调用
`classify_document()` 按文件名定密，不再一律默认 internal——
新上传文档自动继承正确密级。

**端到端验证（7 场景全过）**：员工查总经理工资→工资表被过滤（修
前实测泄露命中）；员工查劳动合同/台账→restricted 被过滤；管理员
查工资→正常命中；经理查工资→confidential 拦截（符合设计）；访客
仅见 public。93 单测全过；L1 回归与基线逐位一致（admin 全见无退化）。

---

**ACL 回归用例（2026-10-24 追加）**：seed_persona 新增 6 条 `acl`
用例（78 条总量）：员工查工资/劳动合同/台账→敏感文档不得命中，
经理查工资→confidential 拦截，访客可见 public、员工可见 internal
两条护栏防过度封锁。`run_persona_eval.py` 新增多身份 ACL 执行段
（acl_pass_rate 出报告），`run_regress.py` L1 将其纳入回归门禁
（基线 1.0，越低越警）。评测验证 6/6 全过，原有指标与基线逐位一致。

**前端密级可视化**：/api/documents 增加 access_level/allowed_groups
字段；文档管理页新增"密级"列——公开(绿)/内部(蓝)/受限(琥珀)/机密(红)
四色徽章 + restricted 文档显示允许组标签，摘要行显示密级分布，
便于管理员人工核对定密合理性。

---

## 六点十九、审计日志补全：敏感命中标记 + 答案侧留痕（2026-10-24 21:20）

**背景**：审计体系（`AuditLogger`，JSONL 按天 + PII 脱敏 + 查询 API）
骨架层一期已建，本次对照等保/内审要求补齐三个缺口：

**① 敏感命中标记（等保核心诉求："谁在什么时间查了涉密内容"）**：
- `retriever` 构建审计 hit_docs 时带上 `access_level`（两路召回的
  record 均携带，无需事后 join registry）；
- `log_search` 顶层新增 `hit_levels`（命中密级列表）与
  `sensitive_hit`（布尔，confidential/restricted 命中即 True）；
- `query_logs` 新增 `sensitive_only` / `level` 筛选参数；
- `get_stats` 新增 `sensitive_hits` / `by_hit_level` / `refused`。

**② 答案侧审计（search 记"检索了什么"，answer 记"返回了什么"）**：
- 新增 `AuditLogger.log_answer()`：refused（是否拒答）、answer_chars
  （只记长度不记全文——正文含脱敏上下文，全文落审计有二次泄露风险）、
  citation_docs（引用文档去重）、llm_called、latency_ms；与 search
  事件通过 query_hash 关联成完整问答链路；
- `api_server` 的 /api/chat 与 /api/chat/stream 两条路径都落痕，
  流式用 finally 保证中断/异常也不丢审计；审计写入失败只 warning
  不影响主流程响应。

**③ 审计出口**：
- 新增 `GET /api/audit/summary`：今日统计 + 最近 10 条敏感命中明细
  （时间/角色/查询/命中密级），生产暴露前须在网关层限管理员；
- 前端运维页新增"审计概览（今日）"卡片：事件总数按类型、涉密命中
  告警灯（>0 变红并列明细）、拒答次数，随 15s 自刷新。

**验证**：临时目录单测全过（敏感标记/PII 脱敏/筛选/统计逐项断言）；
端到端三发真实请求——管理员查工资表→`sensitive_hit=true` 且
hit_docs 带密级（confidential 工资表留痕）、员工查年假→false、
日志行数 342→348（每问 search+answer 各一条）。

**仍欠的账（后续）**：user_id 字段缺位（等 SSO 落地一起补，当前只有
模拟 role）；JSONL 无防篡改（哈希链/异地归档，等保三级要求）；
保留期轮转策略未配置。

---

## 六点二十、SSO 身份集成（一期）：登录 + JWT + 服务端解身份（2026-10-25 11:00）

**动机**：此前身份由前端请求体传入（`user_role` 字段），任何人改请求
即可冒充 admin——ACL 是真护栏但门是虚掩的。本节把"门"装上锁。

**新模块 `surgical_qa/auth.py`（零第三方依赖，纯标准库）**：
- 密码：PBKDF2-HMAC-SHA256 十万轮 + 随机盐，常数时间比较防时序侧信道
- 令牌：JWT HS256 三段式，8 小时有效期；密钥环境变量
  `AUTH_JWT_SECRET` 优先，否则自动生成持久化到 `auth_secret.key`
- 用户表：`auth_users.json` 首次运行播种 4 个演示账号
  （admin/manager/employee/guest，密码 = 用户名+123，上线必须改）
- **SSO 适配器预留**：`AuthProvider` 接口 + `LocalPasswordProvider`
  实现。接企业微信/钉钉/LDAP 时只需新增 Provider（OAuth 码 →
  Identity），登录路由/签发/校验/审计链路零改动

**API 层（api_server.py）**：
- `POST /api/auth/login`（签发令牌）、`GET /api/auth/me`（校验身份）
- 所有业务端点身份改由 `Depends(resolve_identity)` 从 Bearer 令牌
  解出；`AUTH_REQUIRED=0` 评测兼容模式回退请求体身份（本地
  load_test/压测用，生产严禁）
- `/api/audit/summary` 收紧为仅 admin（403）
- 上传/删除文档补 `log_document_operation` 审计（API 层此前漏记）

**审计闭环补全**：`UserContext` 增加 user_id/username/display_name
（位置传参向后兼容），四类审计事件（search/answer/doc_access/
doc_op）全部落 `user_id`——"谁"在"何时"查了"什么"终于三要素齐了。

**前端（static/）**：
- 新增登录层（用户名/密码/错误提示），令牌存 localStorage，
- 统一 `api()` 封装自动带 Authorization 头，401 自动清会话回登录页
- 侧边栏"身份下拉"退役，换成用户卡片（头像/姓名/角色 + 退出）——
  身份切换演示功能下线，这正是 SSO 的意义
- 非管理员访问审计卡片友好降级为提示文案

**验证**：
- 单测全过：散列往返、错误密码、不存在用户、JWT 签发/校验、
  **篡改检测**（改载荷 role=admin 签名不匹配即拒）、过期令牌
- 端到端 7/7：无令牌 401 / 错密码 401 / 登录签发 / me 解身份 /
  带令牌检索 200 / **伪造请求体提权 401（旧通道封死）** /
  employee 查审计 403
- 审计落痕：user_id=u-admin/u-employee 进入审计文件
- 评测模式：AUTH_REQUIRED=0 请求体身份仍生效（HTTP 评测工具兼容）

**仍欠的账**：用户表/密钥与代码同机（上库+KMS 待做）；JWT 无吊销
黑名单；演示密码未改；真实 SSO Provider（企业微信 OAuth）待公司
审批后接入。

---

## 六点二十一、JWT 吊销黑名单 + 真实问法挖矿工具（2026-10-25 10:55）

**① 令牌吊销黑名单（还六点二十的账）**：
- `auth.py` 新增 `revoke_token` / `is_token_revoked` /
  `verify_token_active`：jti 黑名单 + 过期自动清理 + 文件持久化
  （`auth_revoked.json`，重启后吊销不丢）
- `POST /api/auth/logout`：登出即吊销当前令牌——被偷走的令牌也
  立即失效，弥补 JWT 无状态缺陷；鉴权入口统一切
  `verify_token_active`
- 前端登出先通知服务端再清本地（失败不阻塞）
- 边界声明：单副本进程内方案，多副本部署须换 Redis 共享存储
- 验证：单测 7 项全过（吊销立即生效/其他令牌 jti 隔离不受影响/
  **模拟重启后吊销仍生效**/过期条目自动清理）；端到端 4 步全过
  （登出→原令牌 401→重新登录 200→黑名单文件落盘）

**② 真实问法挖矿工具 `mine_real_queries.py`**：
- 数据源：审计日志（每条真实检索天然留痕，零埋点成本）
- 四类信号：零命中（知识缺口）、低置信 top 分 < 0.5（问法与措辞
  差距）、反复出现 ≥3 次（高频，值得进回归集）、被 ACL 过滤多
  （权限外真实需求，汇总给业务方而非进评测集）
- 产出 markdown 人工审核清单（机器提名、人工决定），支持 --days/
  --min-score 参数
- 首跑实测：890 条检索事件 / 303 个不同问法 → 低置信 219、高频 90、
  ACL 拦截 661——**样本几乎全是压测与自测载荷**（"带宠物上班 ×19"
  即 load_test 重复请求），今日清单是噪音，无挖掘价值。工具价值
  在系统给真实用户用之后：积累一周即可跑出第一份有意义的候选清单

---

## 六点二十二、语料改名引出评测假绿：期望解析门禁（2026-10-27 12:55）

**背景**：GitHub 开源化隐私审查时把演示语料的主体名统一替换为虚构名
（示例智能 / 示例OA），评测集 `expected_base_titles` 随之更新，但
`document_store/document_registry.json` 的 `base_title` / `filename` 与
Milvus 行上的 `doc_name` 没有同步——评测按标题动态解析现行版 `doc_id`，
标题对不上就直接跳过该期望项。

**现象**：L1 回归报告"轻微退化"：

```text
seed_v2  recall 0.9615  基线 0.9808  (-0.019)
seed_v2  mrr    0.7788  基线 0.9038  (-0.125) ⚠
persona  recall 0.9344  基线 0.9344  (持平)
persona  mrr    0.7546  基线 0.7874  (-0.033) ⚠
```

**真相**：7 条用例（seed_v2 4 + persona 3）受影响，形式分两种——

- **5 条彻底变空壳**：唯一的期望标题解析失败 → 期望列表为空 →
  `recall_at_k` 兜底 `if not expected_doc_ids: return 1.0` → **recall 恒为
  1.0，用例从此"永久通过"**，只有 mrr 掉到 0 露出马脚
- **2 条检验强度减半**：期望 2 项被砍成 1 项，其中一项没命中 → recall 0.0

指标上的"轻微退化"是假象，实质是**评测能力被削掉 6.7%**，而且削掉的方式
是静默的。这条与 2026-10-24 的向量通道静默降级（DashScope 欠费后
BM25-only 不报错）是同一类病：**失败被兜底值掩盖**。

**定位**：v2-026（"年假申请在OA系统里具体怎么操作？"）三个阶段数字对照，
一眼看清机制：

| 时点 | recall | mrr | 状态 |
|---|---|---|---|
| 10-25 基线 | 0.5 | 1.0 | 期望 2 项、top5 命中 1 项 |
| 12:37 改名未同步 | 0.0 | 0.0 | 期望被砍成 1 项，且未命中 |
| 12:46 改名同步后 | 1.0 | 1.0 | 期望 2 项全命中 |

第三行的 recall 1.0 **不是质量提升**：重插后 HNSW 索引重建，top5 尾部
多挤进"员工考勤与假期管理规范"（本来就该在），mrr 两次都是 0.9038
（首条排序未变）。指标变好也要归因，不能当成绩。

**修复**：新增 `scripts/rename_documents.py`，把 registry 的
`filename` / `base_title` / `file_path` 与 Milvus 行 `doc_name` 一并改名。
关键取舍：**只改元数据、不重新解析文档**——chunk 与向量保持原样，
指标可与历史基线直接对比；若走"删了重新入库"，分块边界可能漂移，
指标变化会混进两种原因，无法归因。执行结果：6 个文档、64 行 chunk
（行数不变）、零重复 chunk_id、全库零残留。修复后指标：

```text
seed_v2  mrr 0.9038 = 基线   persona mrr 0.7874 = 基线
persona  recall 0.9344 = 基线  拒答 1.0 / ACL 1.0 = 基线
```

**门禁**：`EvalRunner._resolve_expected` 改为返回
`(expected_ids, missing_titles)`，报告新增三处：

| 字段 | 含义 |
|---|---|
| `missing_expectation_count` | 至少一个期望标题解析失败的用例数 |
| `empty_expectation_count` | 期望全空、recall 恒 1.0 的空壳用例数 |
| `missing_expectation` | 明细（用例 id / query / 缺失标题） |

`run_regress.py` 对两者**与基线无关、非零即判失败**——因为这不是指标
波动，是评测集出现了洞。

**验证**（不只是"能输出 0"，要证明会红）：往 v2-023 注入不存在的制度名
后跑评测，report 里 `recall` 仍是 **1.0**（假绿现场），但
`missing_expectation_count=1`、`empty_expectation_count=1`，明细精确
指出 v2-023 与缺失标题。评测集随后恢复原状，基线以干净数据刷新。

**教训**：任何"失败被兜底值掩盖"的地方都要加显式信号。兜底 1.0 这种
"友好默认值"在评测里就是毒药——它把"测不了"伪装成"测满分"。

---

## 六点二十三、开源前一致性审查：文档说错话比代码有 bug 更贵（2026-10-27 13:30）

公开仓库的第一读者是陌生人，他只会按 README 做一遍。所以这轮只做一件事：
**把文档里每一句可验证的话，拿代码和数据对一遍。** 结果 6 处不实：

1. **门禁用例数三处说法不一**：README/ARCHITECTURE/CHANGELOG 写 112，EVALUATION 写
   "seed_v2 61 + persona 45"；实际 `seed_v2.json` 26 条、`seed_persona.json` 78 条
   （61 可答 + 11 拒答 + 6 ACL），合计 104。数字错不是排版问题——面试官顺手数一遍就露。
   改法：统一 104，并写出四项构成，谁都能复核。
2. **基线写的是旧值**：README 仍写 recall 0.9808，基线文件早已是 1.000 / MRR 0.9038。
   顺手补了判读提示：seed 集只有 26 题、56 份语料，recall 已饱和，**排序质量看 MRR**，
   别把构造集的满分当战绩。
3. **演示步骤跑不通**：README 让人用 employee 问"总经理的工资"，但 `sample_docs/`
   根本没有工资表——employee 查不到是因为库里没有，admin 同样答不出，演示当场穿帮。
   改成用 `007_2024年11月差旅报销明细表.csv`（文件名含"报销明细"→自动定密 restricted），
   并实测：employee 的召回里没有这张表，manager/admin 有。结论按实测写，不按设想写。
4. **断链与忽略规则**：CHANGELOG 指向 `DEVLOG.md`（实际在 `docs/`）、`query_rewriter.py`
   注释指向改名前的 `CHANGES.md`；`.gitignore` 的 `eval_reports/` 会把门禁基线一起忽略，
   新克隆拿不到对比基准 → 改成 `surgical_qa/eval_reports/*` + 显式放行 `regress_baseline.json`。
5. **鉴权模块零单测**：JWT/吊销是这一版最"危险"的代码，却只有手工验证，没有一条回归。
   补 `tests/test_auth.py` 28 项：口令散列、载荷/签名篡改、过期、jti 隔离、黑名单重启
   恢复、禁用账号、二次启动不覆盖已改密码。顺带发现 `verify_token(None)` 会抛
   `AttributeError`（不属于它承诺的"不抛异常"），加一行类型守卫。全套 121 项通过。
6. **未实施的设计混在规格里**：`spec/table_reconstruction_design.md` 描述了一个没写的模块。
   补状态行说明为什么放弃（xlsx 走 OpenpyxlParser 从源头解决、PDF 走 docling，
   LLM 重建的调用成本与幻觉面不划算），把"没做"变成"权衡过"。

另外把外部测试集的编号统一为 `ext-*`、来源表述改为"外部提供、未随仓库分发"，
避免公开仓库里留下客户/数据集的痕迹。

**这一轮的结论**：一致性检查不需要聪明，需要的是不信自己写过的字——每句都回到
`git grep` 和实跑结果里核一遍。

---

## 七、如何自查

```bash
git log --oneline                     # 看历史
git show --stat HEAD                  # 每次改动的文件清单
git diff HEAD -- surgical_qa/retriever.py   # 逐行看改了什么
git checkout <commit> -- <file>       # 单文件回滚
```

回滚点时注意：`.gitignore` 已排除 `volumes/`（Milvus 数据）与 `parsed_cache/`，代码可回滚，索引不会跟着回滚——恢复旧代码后若结构变化，需要跑一次 `rebuild_index.py --apply`。
