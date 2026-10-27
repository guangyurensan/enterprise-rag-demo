# 企业知识库智能问答系统（RAG）

面向企业内部知识（制度 / 规章 / 财报 / 表格）的问答系统：上传文档后自动解析、分块、建索引；回答问题时可追溯到「哪份文件、哪一章」，并按登录身份过滤可见范围。

> 仓库内语料均为**虚构企业「示例科技」**的演示数据（制度、报销明细、OA 操作 SOP），不含任何真实公司信息。

## 功能亮点

| 维度 | 实现 |
|---|---|
| 混合检索 | BM25 + 向量双路召回，RRF 融合，rerank 语义重排；三轮架构对比实验确定最优组合（seed 集 recall@5 = 1.000、MRR 0.904） |
| 文档权限（ACL） | 四级密级（public / internal / restricted / confidential），入库自动定密；检索层 Milvus 表达式过滤，密级台账与向量库双写一致 |
| 身份鉴权 | 账号密码（PBKDF2）+ JWT（8h）；身份由服务端从令牌解出，请求体无法伪造角色；登出即吊销（jti 黑名单，重启不丢） |
| 审计留痕 | 检索 / 答案 / 文档操作三类事件按天落盘（JSONL），PII 自动脱敏，敏感密级命中单独标记 |
| 质量门禁 | L1 回归 104 条用例：召回/MRR、拒答、ACL、多轮 persona 四项指标与基线逐位对比 |
| 拒答优先 | 检索为空不调用模型，统一口径拒答；答案强制携带引用溯源 |
| 稳定性 | LLM 显式超时 + 熔断重试；向量通道健康监测与前端告警灯，外部服务静默降级可见 |
| 多轮对话 | 查询改写消解指代（"那它的报销流程呢"），persona 回归保障多轮召回一致 |
| 版本治理 | 同名制度仅保留现行版，旧版自动退场归档，可恢复、留审计 |

## 架构总览

```
入库：文档 → 解析清洗 → 智能分块(~600 token) → 向量化 → Milvus（块级密级标签）
                                        └→ 自动定密 → registry 台账（单一事实源，双写一致）

问答：提问(带JWT身份) → 查询改写 → BM25 ∥ 向量召回 → RRF融合(α=0.5) → rerank
      → ACL过滤 → 零召回拒答 / 拼prompt流式生成(带引用) → 审计留痕
```

详细设计见 [ARCHITECTURE.md](ARCHITECTURE.md)；评测方法见 [docs/EVALUATION.md](docs/EVALUATION.md)；安全设计见 [docs/SECURITY.md](docs/SECURITY.md)；变更历史见 [CHANGELOG.md](CHANGELOG.md)。

## 快速启动

### 0. 前置依赖

- Python ≥ 3.10、Docker
- 启动 Milvus standalone（仓库自带 docker-compose）：

```bash
docker compose up -d        # etcd + minio + milvus
```

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置密钥

```bash
export DEEPSEEK_API_KEY=sk-...      # 答案生成 + 查询改写（DeepSeek-chat）
export DASHSCOPE_API_KEY=sk-...     # 向量 embedding + rerank（DashScope）
```

> Windows PowerShell：`$env:DEEPSEEK_API_KEY="sk-..."`（当前会话生效，勿把密钥写进仓库）。

### 3. 启动

```bash
python api_server.py --port 8000
```

浏览器打开 `http://localhost:8000`，用演示账号登录（首次登录时自动播种）：

| 账号 | 密码 | 可见范围 |
|---|---|---|
| `admin` | `admin123` | 全部文档 + 审计概览 |
| `manager` | `manager123` | 加受限文档（按组） |
| `employee` | `employee123` | 公开 + 内部制度 |
| `guest` | `guest123` | 仅公开 |

> ⚠️ 演示账号仅用于本地体验。实际部署前必须修改密码、更换 JWT 密钥，并按 docs/SECURITY.md 核对边界清单。

### 4. 导入演示文档并体验权限过滤

登录后在「文档管理」页上传 `sample_docs/` 下的 5 份虚构演示文档（考勤制度、报销办法、保密规范、报销明细 csv、OA 操作 SOP），即可体验完整流程：

`007_2024年11月差旅报销明细表.csv` 按文件名规则自动定为 restricted（管理层/财务部可见），用它看权限边界最直观：

- 用 `employee` 问「2024年11月研发部张三的差旅报销合计是多少」→ 明细表不会出现在召回结果里（Milvus 表达式已过滤），回答只能落在制度条文上，拿不到具体金额
- 用 `manager` 或 `admin` 问同一句 → 明细表被召回，正常回答并带引用；「运维状态」页审计概览出现敏感命中记录
- 把同一份文档改个版本号重传 → 旧版自动退场，仅现行版可答

Gradio 兼容界面（旧入口）：`cd surgical_qa && python app.py`（端口 7860）

## 回归评测

```bash
python run_regress.py --level 1     # L1 全量：召回/MRR + 拒答 + ACL + persona
python -m pytest tests -q           # 121 项单测：鉴权/ACL/脱敏/评测/熔断，约 13s
```

当前基线（26 条 seed_v2 + 78 条 persona，共 104 条）：seed recall@5 **1.000** / MRR **0.904** ｜ persona recall **0.934** / MRR 0.787 ｜ 拒答 **11/11** ｜ ACL **6/6**。
seed 集语料规模小、recall 已饱和，**排序质量看 MRR**（同一份报告里 `missing_expectation` 必须为 0，否则说明期望文档没解析到、用例已退化成空壳）。
指标与 `surgical_qa/eval_reports/regress_baseline.json` 逐位对比，任何退化立即失败退出——这是所有检索改动的合并门禁。方法详见 docs/EVALUATION.md。

## 目录导览

```
├── api_server.py            # FastAPI 入口：登录/问答(SSE)/文档管理/审计接口
├── static/                  # 零构建前端：问答 / 文档管理 / 运维监控 三视图
├── surgical_qa/             # 核心包
│   ├── document_parser.py   #   多格式解析（txt/docx/pdf/csv/xlsx）
│   ├── smart_chunker.py     #   智能分块
│   ├── vector_index.py      #   Milvus 索引与 ACL 同步
│   ├── hybrid_search.py     #   BM25+向量混合召回
│   ├── reranker.py          #   语义重排
│   ├── retriever.py         #   检索编排 + 生成 + 熔断
│   ├── access_control.py    #   密级定密/过滤/审计日志
│   ├── auth.py              #   JWT 鉴权与吊销黑名单
│   ├── version_policy.py    #   版本治理
│   ├── document_manager.py  #   入库/删除/台账
│   ├── metrics.py           #   指标与健康信号
│   └── eval_sets/           #   评测集（seed_v2 / persona）
├── sample_docs/             # 虚构演示文档
├── docker-compose.yml       # Milvus standalone
├── docs/                    # 架构/评测/安全文档与完整开发日志
├── spec/                    # 早期设计规格（含未实施方案，作决策留痕）
│
├── run_regress.py           # 回归门禁入口（L0 冒烟 / L1 检索 / L2 答案级 / L3 全量）
├── run_baseline.py / run_persona_eval.py / answer_judge_eval.py   # 各评测集执行器
├── eval_*.py / sensitivity_scan.py / calibrate_ab.py             # 专项实验：架构对比 / α 扫描 / 裁判校准
├── apply_acl.py / apply_version_policy.py / rebuild_index.py     # 数据治理：定密刷新 / 版本退场 / 索引重建
├── load_test.py / run_stress_ingest.py                           # 压测：问答并发 / 入库吞吐
├── mine_real_queries.py / scripts/rename_documents.py            # 真实问法挖矿 / 文档批量改名
└── serve.py                 # 服务看护（健康轮询 + 异常重启）
```

## 常用配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `AUTH_REQUIRED` | `1` | 设 `0` 进入评测兼容模式（信任请求体身份），仅限内网评测/压测 |
| `AUTH_JWT_SECRET` | 首次启动自动生成 | JWT 签名密钥；未显式设置时写入 `surgical_qa/auth_secret.key`（已 gitignore） |
| `MINERU_ENABLED` | `0` | 设 `1` 启用 MinerU 云端解析（文件会上传第三方，默认关闭） |
| `PARSER_DISABLED` | `unstructured` | 逗号分隔禁用列表 |
| `CORS_ORIGINS` | `*` | API 允许的前端来源，生产环境收紧为具体域名 |
| `LLM_TEMPERATURE` | `0.3` | 生成温度；评测需可复现时设 `0` |

## 已知边界（诚实清单）

- 单机单副本设计：JWT 吊销黑名单与用户表为本地文件，多副本部署需换 Redis/数据库
- 审计为 JSONL 追加文件，无防篡改哈希链；规模化后建议迁 SQLite（触发条件见 ARCHITECTURE.md）
- 演示账号密码为公开默认值，部署前必须更换
- 评测集为构造数据；真实用户问法采集工具已内置（mine_real_queries.py），需真实流量积累

## License

[MIT](LICENSE)
