# -*- coding: utf-8 -*-
"""
企业级知识库 - 容错与韧性模块

核心组件：
1. CircuitBreaker — 熔断器（保护下游 API：LLM / Rerank / Embedding）
2. RateLimiter — 令牌桶限流器（控制 QPS）
3. DeadLetterQueue — 死信队列（失败任务持久化，可重跑）

设计原则：
- 轻量实现，不引入外部依赖（无 Redis / RabbitMQ）
- 每个下游服务独立熔断器实例
- 熔断状态机：CLOSED → OPEN → HALF_OPEN → CLOSED
- 死信队列基于 JSON 文件，支持重跑和统计
"""

import os
import json
import time
import logging
import threading
from enum import Enum
from typing import Callable, Any, Optional, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# 熔断器
# ============================================================
class CircuitState(Enum):
    CLOSED = "closed"        # 正常通行
    OPEN = "open"            # 熔断中，拒绝请求
    HALF_OPEN = "half_open"  # 半开，放行一个探测请求


class CircuitBreaker:
    """
    熔断器 — 保护下游 API 调用

    状态机：
    - CLOSED: 正常状态，失败计数达到阈值时切换到 OPEN
    - OPEN: 拒绝所有请求，等待 recovery_timeout 后切换到 HALF_OPEN
    - HALF_OPEN: 放行一个请求，成功则切换到 CLOSED，失败则回到 OPEN

    使用方式：
        breaker = CircuitBreaker(name="llm", failure_threshold=5, recovery_timeout=60)

        @breaker
        def call_llm(query):
            ...

        # 或手动调用
        result = breaker.call(call_llm, query)
    """

    def __init__(self, name: str, failure_threshold: int = 5,
                 recovery_timeout: float = 60.0, half_open_max: int = 1):
        """
        :param name: 熔断器名称（用于日志和监控）
        :param failure_threshold: 连续失败次数阈值，达到后熔断
        :param recovery_timeout: 熔断恢复等待时间（秒）
        :param half_open_max: 半开状态允许放行的探测请求数
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max = half_open_max

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_count = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """获取当前状态（自动检查是否应从 OPEN 切换到 HALF_OPEN）"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    self._half_open_count = 0
                    logger.info(f"[CircuitBreaker:{self.name}] OPEN -> HALF_OPEN")
            return self._state

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        通过熔断器调用函数

        :param func: 要调用的函数
        :param args: 位置参数
        :param kwargs: 关键字参数
        :return: 函数返回值
        :raises: CircuitOpenError 如果熔断器处于 OPEN 状态
        """
        current_state = self.state

        if current_state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit breaker '{self.name}' is OPEN, "
                f"rejecting request. Retry after {self.recovery_timeout}s"
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        """调用成功"""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
                    logger.info(f"[CircuitBreaker:{self.name}] HALF_OPEN -> CLOSED (recovered)")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # 成功则重置失败计数

    def _on_failure(self):
        """调用失败"""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # 探测失败，回到 OPEN
                self._state = CircuitState.OPEN
                self._success_count = 0
                logger.warning(f"[CircuitBreaker:{self.name}] HALF_OPEN -> OPEN (probe failed)")
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        f"[CircuitBreaker:{self.name}] CLOSED -> OPEN "
                        f"(failures: {self._failure_count}/{self.failure_threshold})"
                    )

    def reset(self):
        """手动重置熔断器"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            logger.info(f"[CircuitBreaker:{self.name}] manually reset to CLOSED")

    def get_stats(self) -> Dict:
        """获取熔断器统计"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
        }

    def __call__(self, func):
        """装饰器用法"""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        wrapper.__name__ = func.__name__
        return wrapper


class CircuitOpenError(Exception):
    """熔断器打开时抛出的异常"""
    pass


# ============================================================
# 令牌桶限流器
# ============================================================
class RateLimiter:
    """
    令牌桶限流器

    使用方式：
        limiter = RateLimiter(qps=10)  # 每秒最多 10 个请求

        if limiter.acquire():
            # 执行请求
            ...
        else:
            # 被限流
            ...
    """

    def __init__(self, qps: float = 10.0, burst: int = None):
        """
        :param qps: 每秒允许的请求数
        :param burst: 突发上限（默认 = qps）
        """
        self.qps = qps
        self.burst = burst if burst is not None else int(qps)
        self._tokens = float(self.burst)
        self._last_time = time.time()
        self._lock = threading.Lock()
        self._total_allowed = 0
        self._total_rejected = 0

    def acquire(self) -> bool:
        """
        尝试获取一个令牌

        :return: True 获取成功，False 被限流
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_time
            self._last_time = now

            # 补充令牌
            self._tokens = min(
                float(self.burst),
                self._tokens + elapsed * self.qps
            )

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                self._total_allowed += 1
                return True
            else:
                self._total_rejected += 1
                return False

    def wait(self, timeout: float = 5.0) -> bool:
        """
        等待直到获取到令牌或超时

        :param timeout: 最大等待时间（秒）
        :return: True 获取成功，False 超时
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.acquire():
                return True
            time.sleep(1.0 / self.qps)  # 等待一个令牌周期
        return False

    def get_stats(self) -> Dict:
        """获取限流统计"""
        return {
            "qps": self.qps,
            "burst": self.burst,
            "total_allowed": self._total_allowed,
            "total_rejected": self._total_rejected,
        }


# ============================================================
# 死信队列
# ============================================================
class DeadLetterQueue:
    """
    死信队列 — 失败任务持久化

    基于 JSON 文件，支持：
    - 写入失败任务
    - 读取待重试任务
    - 重跑后移除
    - 统计

    使用方式：
        dlq = DeadLetterQueue("ingestion_failures")
        dlq.push({"file": "test.pdf", "error": "parse failed"})

        # 查看待重试
        tasks = dlq.peek_all()

        # 重跑成功后移除
        dlq.remove(task_id)
    """

    def __init__(self, name: str, storage_dir: str = None):
        """
        :param name: 队列名称
        :param storage_dir: 存储目录，默认为项目根目录下 dead_letters/
        """
        self.name = name
        if storage_dir is None:
            storage_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "dead_letters"
            )
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self._file_path = os.path.join(self.storage_dir, f"{name}.json")
        self._lock = threading.Lock()

    def _load(self) -> Dict:
        """加载队列数据"""
        if not os.path.exists(self._file_path):
            return {"tasks": [], "created_at": datetime.now().isoformat()}
        try:
            with open(self._file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"tasks": [], "created_at": datetime.now().isoformat()}

    def _save(self, data: Dict):
        """保存队列数据"""
        with open(self._file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def push(self, task: Dict, task_id: str = None) -> str:
        """
        写入失败任务

        :param task: 任务数据
        :param task_id: 任务 ID，不传则自动生成
        :return: 任务 ID
        """
        if task_id is None:
            task_id = f"{self.name}_{int(time.time() * 1000)}"

        entry = {
            "task_id": task_id,
            "data": task,
            "failed_at": datetime.now().isoformat(),
            "retry_count": 0,
        }

        with self._lock:
            data = self._load()
            data["tasks"].append(entry)
            self._save(data)

        logger.info(f"[DLQ:{self.name}] Task added: {task_id}")
        return task_id

    def peek_all(self) -> List[Dict]:
        """查看所有待重试任务"""
        with self._lock:
            data = self._load()
        return data.get("tasks", [])

    def remove(self, task_id: str) -> bool:
        """移除已完成的任务"""
        with self._lock:
            data = self._load()
            original_count = len(data["tasks"])
            data["tasks"] = [t for t in data["tasks"] if t["task_id"] != task_id]
            if len(data["tasks"]) < original_count:
                self._save(data)
                logger.info(f"[DLQ:{self.name}] Task removed: {task_id}")
                return True
        return False

    def increment_retry(self, task_id: str):
        """增加任务的重试计数"""
        with self._lock:
            data = self._load()
            for task in data["tasks"]:
                if task["task_id"] == task_id:
                    task["retry_count"] = task.get("retry_count", 0) + 1
                    break
            self._save(data)

    def clear(self):
        """清空队列"""
        with self._lock:
            self._save({"tasks": [], "created_at": datetime.now().isoformat()})

    def get_stats(self) -> Dict:
        """获取队列统计"""
        tasks = self.peek_all()
        return {
            "name": self.name,
            "pending_count": len(tasks),
            "total_retries": sum(t.get("retry_count", 0) for t in tasks),
        }


# ============================================================
# 全局实例（从 config 读取配置）
# ============================================================
from config import RESILIENCE_CONFIG

_cb_cfg = RESILIENCE_CONFIG.get("circuit_breaker", {})
_rl_cfg = RESILIENCE_CONFIG.get("rate_limiter", {})

# 三路熔断器（LLM / Rerank / Embedding）
llm_breaker = CircuitBreaker(
    name="llm",
    failure_threshold=_cb_cfg.get("llm", {}).get("failure_threshold", 5),
    recovery_timeout=_cb_cfg.get("llm", {}).get("recovery_timeout", 60),
)
rerank_breaker = CircuitBreaker(
    name="rerank",
    failure_threshold=_cb_cfg.get("rerank", {}).get("failure_threshold", 3),
    recovery_timeout=_cb_cfg.get("rerank", {}).get("recovery_timeout", 30),
)
embedding_breaker = CircuitBreaker(
    name="embedding",
    failure_threshold=_cb_cfg.get("embedding", {}).get("failure_threshold", 5),
    recovery_timeout=_cb_cfg.get("embedding", {}).get("recovery_timeout", 60),
)

# 限流器（DashScope QPM 限制）
api_rate_limiter = RateLimiter(
    qps=_rl_cfg.get("qps", 10),
    burst=_rl_cfg.get("burst", 15),
)

# 死信队列
ingestion_dlq = DeadLetterQueue("ingestion_failures")
query_dlq = DeadLetterQueue("query_failures")



