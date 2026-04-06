"""
Measures wall-clock time for async calls and computes latency statistics.
"""

import asyncio
import statistics
import time
from typing import Any

from src.metrics.schemas import LatencyStats


async def timed_call(coro) -> tuple[Any, float]:
    """
    Wrap an async coroutine and measure its wall-clock execution time.

    Args:
        coro: An awaitable coroutine.

    Returns:
        Tuple of (result, duration_ms).
    """
    start = time.perf_counter()
    result = await coro
    duration_ms = (time.perf_counter() - start) * 1000
    return result, duration_ms


def compute_latency_stats(durations: list[float]) -> LatencyStats:
    """
    Compute latency statistics from a list of durations in milliseconds.

    Args:
        durations: List of duration values in ms.

    Returns:
        LatencyStats with mean, median, min, max.
    """
    if not durations:
        return LatencyStats(mean_ms=0, median_ms=0, min_ms=0, max_ms=0)

    return LatencyStats(
        mean_ms=round(statistics.mean(durations), 2),
        median_ms=round(statistics.median(durations), 2),
        min_ms=round(min(durations), 2),
        max_ms=round(max(durations), 2),
    )
