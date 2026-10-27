# -*- coding: utf-8 -*-
"""
容错与韧性模块测试

覆盖:
- CircuitBreaker 状态机 (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- RateLimiter 令牌桶
- DeadLetterQueue 持久化与重跑
"""

import os
import time
import json
import pytest
import tempfile
import shutil

from resilience import (
    CircuitBreaker, CircuitState, CircuitOpenError,
    RateLimiter, DeadLetterQueue,
)


# ============================================================
# CircuitBreaker
# ============================================================
class TestCircuitBreaker:
    def test_initial_state_closed(self):
        breaker = CircuitBreaker("test", failure_threshold=3, recovery_timeout=1)
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_open is False

    def test_call_success(self):
        breaker = CircuitBreaker("test")
        result = breaker.call(lambda: 42)
        assert result == 42
        assert breaker.state == CircuitState.CLOSED

    def test_call_failure_increments_count(self):
        breaker = CircuitBreaker("test", failure_threshold=3, recovery_timeout=1)

        def failing():
            raise ValueError("fail")

        for _ in range(2):
            try:
                breaker.call(failing)
            except ValueError:
                pass

        assert breaker.state == CircuitState.CLOSED  # 还没到阈值

    def test_closed_to_open_on_threshold(self):
        breaker = CircuitBreaker("test", failure_threshold=3, recovery_timeout=1)

        def failing():
            raise ValueError("fail")

        for _ in range(3):
            try:
                breaker.call(failing)
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open is True

    def test_open_rejects_calls(self):
        breaker = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60)

        def failing():
            raise ValueError("fail")

        try:
            breaker.call(failing)
        except ValueError:
            pass

        assert breaker.is_open is True

        with pytest.raises(CircuitOpenError):
            breaker.call(lambda: 42)

    def test_open_to_half_open_after_timeout(self):
        breaker = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.5)

        def failing():
            raise ValueError("fail")

        try:
            breaker.call(failing)
        except ValueError:
            pass

        assert breaker.is_open is True

        time.sleep(0.6)
        assert breaker.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        breaker = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.5)

        def failing():
            raise ValueError("fail")

        try:
            breaker.call(failing)
        except ValueError:
            pass

        time.sleep(0.6)
        assert breaker.state == CircuitState.HALF_OPEN

        result = breaker.call(lambda: 42)
        assert result == 42
        assert breaker.state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        breaker = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.5)

        def failing():
            raise ValueError("fail")

        try:
            breaker.call(failing)
        except ValueError:
            pass

        time.sleep(0.6)
        assert breaker.state == CircuitState.HALF_OPEN

        try:
            breaker.call(failing)
        except ValueError:
            pass

        assert breaker.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        breaker = CircuitBreaker("test", failure_threshold=3, recovery_timeout=60)

        def failing():
            raise ValueError("fail")

        for _ in range(2):
            try:
                breaker.call(failing)
            except ValueError:
                pass

        breaker.call(lambda: 42)  # 成功，重置计数
        assert breaker._failure_count == 0

    def test_manual_reset(self):
        breaker = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60)

        def failing():
            raise ValueError("fail")

        try:
            breaker.call(failing)
        except ValueError:
            pass

        assert breaker.is_open is True
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED

    def test_decorator_usage(self):
        breaker = CircuitBreaker("test", failure_threshold=5)

        @breaker
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_get_stats(self):
        breaker = CircuitBreaker("test", failure_threshold=5, recovery_timeout=60)
        stats = breaker.get_stats()
        assert stats["name"] == "test"
        assert stats["state"] == "closed"


# ============================================================
# RateLimiter
# ============================================================
class TestRateLimiter:
    def test_acquire_within_limit(self):
        limiter = RateLimiter(qps=10, burst=10)
        assert limiter.acquire() is True

    def test_acquire_exceeds_burst(self):
        limiter = RateLimiter(qps=10, burst=2)
        assert limiter.acquire() is True
        assert limiter.acquire() is True
        assert limiter.acquire() is False  # 超出 burst

    def test_tokens_replenish(self):
        limiter = RateLimiter(qps=100, burst=1)
        limiter.acquire()  # 用完
        time.sleep(0.05)  # 等 50ms，应补充 5 个 token
        assert limiter.acquire() is True

    def test_get_stats(self):
        limiter = RateLimiter(qps=10, burst=2)
        limiter.acquire()
        limiter.acquire()
        limiter.acquire()  # rejected
        stats = limiter.get_stats()
        assert stats["total_allowed"] == 2
        assert stats["total_rejected"] == 1


# ============================================================
# DeadLetterQueue
# ============================================================
class TestDeadLetterQueue:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.dlq = DeadLetterQueue("test", storage_dir=self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_push_and_peek(self):
        task_id = self.dlq.push({"file": "test.pdf", "error": "parse failed"})
        tasks = self.dlq.peek_all()
        assert len(tasks) == 1
        assert tasks[0]["data"]["file"] == "test.pdf"

    def test_remove(self):
        task_id = self.dlq.push({"file": "test.pdf"})
        assert self.dlq.remove(task_id) is True
        assert len(self.dlq.peek_all()) == 0

    def test_remove_nonexistent(self):
        assert self.dlq.remove("nonexistent_id") is False

    def test_increment_retry(self):
        task_id = self.dlq.push({"file": "test.pdf"})
        self.dlq.increment_retry(task_id)
        tasks = self.dlq.peek_all()
        assert tasks[0]["retry_count"] == 1

    def test_clear(self):
        self.dlq.push({"file": "a.pdf"})
        self.dlq.push({"file": "b.pdf"})
        self.dlq.clear()
        assert len(self.dlq.peek_all()) == 0

    def test_get_stats(self):
        self.dlq.push({"file": "a.pdf"})
        task_id = self.dlq.push({"file": "b.pdf"})
        self.dlq.increment_retry(task_id)
        stats = self.dlq.get_stats()
        assert stats["pending_count"] == 2
        assert stats["total_retries"] == 1

    def test_persistence(self):
        """数据持久化到文件"""
        self.dlq.push({"file": "test.pdf"})
        # 重新创建实例，读同一文件
        dlq2 = DeadLetterQueue("test", storage_dir=self.tmpdir)
        tasks = dlq2.peek_all()
        assert len(tasks) == 1
