# -*- coding: utf-8 -*-
"""
进程内轻量指标（线程安全，无外部依赖）

设计原则：
- 只做计数器 + 滚动延迟样本，不上时序数据库——企业级监控的第一步是"有出口"，
  Prometheus/Grafana 接入可以后置，/api/metrics 就是那个出口
- 单例模式：retriever / document_manager / api_server 各处 import 同一份状态
- 异常绝不外抛：指标埋点失败不能影响主业务
"""
import threading
import time
import logging
from collections import deque

logger = logging.getLogger(__name__)

_START_TIME = time.time()
_LOCK = threading.Lock()

# 计数器：{key: int}
_counters = {}
# 滚动延迟样本：{key: deque(maxlen=200)}
_latencies = {}


def incr(key: str, n: int = 1):
    """计数器 +n"""
    try:
        with _LOCK:
            _counters[key] = _counters.get(key, 0) + n
    except Exception:  # noqa: BLE001
        pass


def observe(key: str, value_ms: float):
    """记录一次延迟样本（毫秒）"""
    try:
        with _LOCK:
            dq = _latencies.get(key)
            if dq is None:
                dq = _latencies[key] = deque(maxlen=200)
            dq.append(float(value_ms))
    except Exception:  # noqa: BLE001
        pass


def _lat_stats(key: str):
    dq = _latencies.get(key)
    if not dq:
        return None
    samples = sorted(dq)
    n = len(samples)
    p95 = samples[min(n - 1, int(n * 0.95))]
    return {
        "count": n,
        "avg_ms": round(sum(samples) / n, 1),
        "p95_ms": round(p95, 1),
        "max_ms": round(samples[-1], 1),
    }


def snapshot() -> dict:
    """全量指标快照（供 /api/metrics 输出）"""
    with _LOCK:
        counters = dict(_counters)
    return {
        "uptime_sec": int(time.time() - _START_TIME),
        "counters": counters,
        "latency": {
            k: _lat_stats(k) for k in sorted(_latencies) if _latencies.get(k)
        },
        "health_signals": _health_signals(counters),
    }


# 向量通道健康判定阈值：样本足够多且 BM25-only 占比过半即判降级
_DEGRADE_MIN_SAMPLES = 10
_DEGRADE_RATIO = 0.5


def _health_signals(counters: dict) -> dict:
    """派生健康信号——向量通道降级是静默的，必须在这里显式算出来"""
    total = counters.get("search_total", 0)
    bm25_only = counters.get("search_bm25_only", 0)
    vec_errors = counters.get("vector_recall_error", 0)
    if total < _DEGRADE_MIN_SAMPLES:
        status, ratio = "unknown", None
    else:
        ratio = round(1.0 - bm25_only / total, 4)  # 向量召回占比
        status = "degraded" if (1.0 - ratio) > _DEGRADE_RATIO else "ok"
    return {
        "vector_channel_status": status,
        "vector_recall_ratio": ratio,      # 近 200 次检索中向量通道有产出的占比
        "vector_recall_errors": vec_errors,  # 通道故障（欠费/限流/网络）次数
        "note": "degraded 表示向量通道产出占比低于 50%——优先检查 DashScope 账户/网络",
    }


# 埋点方统一 `from metrics import metrics` 后 metrics.incr/observe
from types import SimpleNamespace  # noqa: E402
metrics = SimpleNamespace(incr=incr, observe=observe)
