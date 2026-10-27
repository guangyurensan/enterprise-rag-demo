# -*- coding: utf-8 -*-
"""
企业级知识库 - Agently 任务执行器
使用 Agently 框架实现结构化输出，避免手动解析 JSON
"""

import os
from typing import Optional
from agently import Agently

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, LLM_MODEL


# ── 初始化 Agently 配置（使用 DeepSeek API）────────────────────────────────────
Agently.set_settings("OpenAICompatible", {
    "base_url": DEEPSEEK_BASE_URL,
    "model": LLM_MODEL,
    "auth": DEEPSEEK_API_KEY,
})


class TaskExecutor:
    """
    任务执行器 - 封装 Agently 调用
    职责：接收任务，调用模型，返回结果
    """
    
    def run(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> str:
        """
        文本型任务。
        
        Args:
            user_input: 任务描述
            system_prompt: 角色定义（可选）
            context: 背景资料，格式为 {"标签": "内容"}（可选）
        
        Returns:
            str: 模型返回的文本
        """
        try:
            agent = Agently.create_agent()
            if system_prompt:
                agent = agent.system(system_prompt)
            if context:
                for label, content in context.items():
                    agent = agent.info(label, content)
            result = agent.input(user_input).get_text()
            return result if result and result.strip() else "[无内容返回]"
        except Exception as e:
            print(f"[TaskExecutor.run] {type(e).__name__}: {e}")
            return "[执行失败，请检查模型服务]"
    
    def run_structured(
        self,
        user_input: str,
        output_schema: dict,
        system_prompt: Optional[str] = None,
        context: Optional[dict] = None,
        ensure_keys: Optional[list] = None,
        max_retries: int = 2,
        temperature: Optional[float] = None,
    ) -> dict:
        """
        结构化输出任务。失败时抛出 RuntimeError。
        
        Args:
            user_input: 任务描述
            output_schema: Agently dict schema
            system_prompt: 角色定义（可选）
            context: 背景资料（可选）
            ensure_keys: 必须存在的字段路径列表，缺失时自动重试
            max_retries: 最大重试次数（默认 2）
        
        Returns:
            dict: 结构化结果
        """
        try:
            agent = Agently.create_agent()
            if temperature is not None:
                # Agently 的温度入口是 agent.options()，不是 set_settings('request_options')
                agent = agent.options({"temperature": temperature})
            if system_prompt:
                agent = agent.system(system_prompt)
            if context:
                for label, content in context.items():
                    agent = agent.info(label, content)
            return (
                agent
                .input(user_input)
                .output(output_schema)
                .start(ensure_keys=ensure_keys, max_retries=max_retries)
            )
        except Exception as e:
            raise RuntimeError(f"结构化请求失败：{e}") from e


# 全局实例
executor = TaskExecutor()
