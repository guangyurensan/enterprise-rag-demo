# -*- coding: utf-8 -*-
"""
企业级知识库 - Query 改写
检索前用一次结构化调用完成三件事：上下文补全、指代消解、多意图拆解

设计依据（实测，见 docs/DEVLOG.md 与 ab_rewrite_eval.py）：
1. 一次调用而非三次。原实现串行调 3 次模型（上下文改写 → 指代消解 → 多意图分解），
   多轮单次检索 4.04s；这三件事本质是同一个判断"这句话离开上文是否完整、含几个意图"，
   合并成一次结构化调用后约 1.5s，且改写结果与子查询来自同一份判断，不会互相矛盾。
2. schema 用 "not_null" 而非 True。实测同一问题重复 5 次：
   [(str, "子问题", True)]      → 5 次里 2 次返回 ['...', '']（True 只校验键存在）
   [(str, "子问题", "not_null")] → 5/5 完整（空串会触发框架自动重试）
3. 单轮不调用模型。59 条人设集实测：单轮改写把原问题换成同义问法，丢原文关键词，
   Recall@5 98.31%→96.61%，延迟 +1.7s，收益为负；多轮追问场景 4/5→5/5，才是它的价值。
"""

from task_executor import executor

# TaskExecutor.run 在模型调用失败时返回固定文案，不能当作改写结果使用
EXECUTOR_FAILURE_PREFIX = "[执行失败"

# 结构化契约：一次调用同时产出"完整问题"与"检索子问题"
REWRITE_SCHEMA = {
    "rewritten_query": (str, "补全主语与指代后的独立问题；无需改写时原样返回", "not_null"),
    "sub_queries": [(str, "可独立检索的子问题；单意图时只返回一条", "not_null")],
}

REWRITE_INSTRUCTION = """你是企业知识库的查询整理器。根据对话历史和当前问题，完成两件事：

1. rewritten_query：当前问题如果省略了主语/宾语，或含"它、这个、那个、这种情况"等指代，
   用对话历史补全成一个脱离上下文也能看懂的完整问题；如果本来就完整，原样返回。
2. sub_queries：如果当前问题包含多个互不相同的检索意图（例如"报销标准是什么，网约车能报吗"），
   拆成能独立检索的子问题；只有一个意图时只返回一条，内容与 rewritten_query 一致。

规则：
- 不改变用户意图，不扩写、不猜测、不添加原文没有的限定条件
- 保留用户原话里的关键词，尤其是制度名、部门名、系统名
- 只输出结构化字段，不要回答问题"""


class QueryRewriter:
    """企业 Query 改写器"""

    @staticmethod
    def _sanitize(text, fallback):
        """兜底：空串 / 模型失败文案 → 退回原查询（框架重试失败后仍可能返回空）"""
        if not isinstance(text, str):
            return fallback
        text = text.strip()
        if not text or text.startswith(EXECUTOR_FAILURE_PREFIX):
            return fallback
        return text

    def auto_rewrite(self, query, conversation_history=""):
        """
        自动改写（统一入口）

        :return: {
            "original_query": 原始问题,
            "rewritten_query": 改写后的完整问题,
            "sub_queries": [可独立检索的子问题],
            "rewrite_type": "上下文改写 / 多意图分解 / 无需改写 / 跳过改写 / 空查询",
        }
        """
        query = (query or "").strip()
        if not query:
            return {
                "original_query": query, "rewritten_query": query,
                "sub_queries": [], "rewrite_type": "空查询",
            }

        # 无对话历史时跳过：单轮改写实测为负收益
        if not (conversation_history or "").strip():
            return {
                "original_query": query, "rewritten_query": query,
                "sub_queries": [query], "rewrite_type": "跳过改写",
            }

        try:
            data = executor.run_structured(
                user_input=f"对话历史：\n{conversation_history}\n\n当前问题：{query}",
                output_schema=REWRITE_SCHEMA,
                system_prompt=REWRITE_INSTRUCTION,
                max_retries=2,
            )
        except Exception as e:
            print(f"[QueryRewriter] 改写失败，回退原查询: {type(e).__name__}: {e}")
            return {
                "original_query": query, "rewritten_query": query,
                "sub_queries": [query], "rewrite_type": "改写失败",
            }

        rewritten = self._sanitize(data.get("rewritten_query"), query)
        sub_queries, seen = [], set()
        for s in data.get("sub_queries") or []:
            s = self._sanitize(s, "")
            if s and s not in seen:
                seen.add(s)
                sub_queries.append(s)
        if not sub_queries:
            sub_queries = [rewritten]

        types = []
        if rewritten.rstrip("？?。.") != query.rstrip("？?。."):
            types.append("上下文改写")
        if len(sub_queries) > 1:
            types.append("多意图分解")

        return {
            "original_query": query,
            "rewritten_query": rewritten,
            "sub_queries": sub_queries,
            "rewrite_type": "+".join(types) if types else "无需改写",
        }

    def format_history(self, history):
        """
        将 Gradio 对话历史格式化为文本
        :param history: [{"role": "user", "content": "..."}, ...]
        :return: 最近 5 轮的文本
        """
        if not history:
            return ""

        lines = []
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                lines.append(f"用户：{content}")
            elif role == "assistant":
                summary = content[:100] + "..." if len(content) > 100 else content
                lines.append(f"助手：{summary}")

        return "\n".join(lines[-10:])
