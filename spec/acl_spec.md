# ACL 权限控制规格文档

> 版本: 1.0 | 日期: 2026-10-20 | 状态: 骨架层一期固化

## 1. 密级模型

### 1.1 四级密级

| 密级 | 数值 | 说明 |
|------|------|------|
| public | 0 | 公开，所有人可见 |
| internal | 1 | 内部，员工可见 |
| restricted | 2 | 受限，需组匹配 |
| confidential | 3 | 机密，仅 admin |

### 1.2 角色权限矩阵

| 角色 | 可达密级上限 | 说明 |
|------|-------------|------|
| guest | 0 (public) | 仅公开文档 |
| employee | 1 (internal) | 公开 + 内部 |
| manager | 2 (restricted) | 公开 + 内部 + 受限(需组匹配) |
| admin | 3 (confidential) | 全部可见 |

## 2. 权限判定 (AccessPolicy.check_access)

### 2.1 判定流程

```
输入: (doc_level, doc_groups, user_ctx)
  |
  v
admin? --YES--> 返回 True
  |
  NO
  v
doc_level 数值 > 角色上限? --YES--> 返回 False
  |
  NO
  v
restricted 且 doc_groups != "all"? --YES--> 用户组与文档组有交集? --NO--> 返回 False
  |                                                                     |
  NO                                                                    YES
  v                                                                     v
返回 True                                                         返回 True
```

### 2.2 函数签名

```python
AccessPolicy.check_access(doc_level: str, doc_groups: str, user_ctx: UserContext) -> bool
```

## 3. Milvus 引擎侧过滤 (AccessPolicy.build_filter_expr)

### 3.1 生成规则

| 角色 | 生成的 ACL 表达式 |
|------|-------------------|
| admin | (空，无 ACL 限制) |
| manager | `access_level in ["public", "internal", "restricted"]` |
| employee | `access_level in ["public", "internal"]` |
| guest | `access_level == "public"` |

### 3.2 强制追加

所有角色均追加 `is_deleted == false`，确保软删除文档不可见。

### 3.3 应用位置

- **向量召回**: 引擎侧过滤 (filter_expr)，ACL 不可绕过
- **BM25 召回**: 内存侧过滤 (check_access)

## 4. 组名清洗 (clean_groups)

```
输入: " 人事部 , 财务部 , , 人事部 "
  |
  v
分割 + strip + 去空 + 去重(保序)
  |
  v
输出: "人事部,财务部"
```

- 空输入 -> "all"
- 非法组名只告警，不阻断

## 5. 用户上下文 (UserContext)

```python
@dataclass
class UserContext:
    role: str       # "admin" | "manager" | "employee" | "guest"
    groups: List[str]  # ["人事部", "全员"]
```

- 非法角色降级为 "employee" 并告警
- 默认 groups = ["全员"]

## 6. 审计日志 (AuditLogger)

### 6.1 事件类型

| 事件 | 触发时机 | 关键字段 |
|------|----------|----------|
| search | 每次检索 | query(脱敏), result_count, latency_ms, channels, hit_docs |
| doc_access | 文档访问 | doc_id, doc_level, access_granted |
| document_op | 文档操作 | operation(upload/update/delete/restore), doc_id, success |

### 6.2 持久化

- **格式**: JSONL，按天分割 (`audit_logs/audit_YYYY-MM-DD.jsonl`)
- **线程安全**: threading.Lock
- **脱敏**: query 自动 PII 脱敏后存储
- **查询**: 按日期/事件类型/角色过滤

## 7. 一期边界 (TODO 二期)

- [ ] 用户认证系统 (当前硬编码 admin)
- [ ] restricted 组匹配在 Milvus 引擎侧执行 (当前在内存侧)
- [ ] confidential 写入管控 (当前仅告警不阻断)
- [ ] 审计日志实时告警
