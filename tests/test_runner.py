"""
Tests for src/runner/ module
Tests timing, retry logic, and timeout handling.
"""

import asyncio
import pytest

from src.runner.timing import timed_call, compute_latency_stats
from src.runner.retry import call_with_retry, AgentTimeoutError, AgentCallError
from src.metrics.schemas import LatencyStats


class TestTimedCall:
    """Tests for the timed_call function."""

    @pytest.mark.asyncio
    async def test_measures_time(self):
        async def slow_fn():
            await asyncio.sleep(0.05)
            return "done"

        result, duration_ms = await timed_call(slow_fn())
        assert result == "done"
        assert duration_ms >= 40  # At least ~50ms

    @pytest.mark.asyncio
    async def test_fast_call(self):
        async def fast_fn():
            return "instant"

        result, duration_ms = await timed_call(fast_fn())
        assert result == "instant"
        assert duration_ms < 50  # Should be nearly instant


class TestComputeLatencyStats:
    def test_normal_durations(self):
        durations = [100.0, 200.0, 300.0, 400.0, 500.0]
        stats = compute_latency_stats(durations)
        assert stats.mean_ms == 300.0
        assert stats.median_ms == 300.0
        assert stats.min_ms == 100.0
        assert stats.max_ms == 500.0

    def test_single_duration(self):
        stats = compute_latency_stats([42.0])
        assert stats.mean_ms == 42.0
        assert stats.median_ms == 42.0

    def test_empty_durations(self):
        stats = compute_latency_stats([])
        assert stats.mean_ms == 0
        assert stats.median_ms == 0


class TestCallWithRetry:
    """Tests for the retry logic."""

    @pytest.mark.asyncio
    async def test_success_first_try(self):
        async def good_fn(input):
            return f"response to {input}"

        result = await call_with_retry(good_fn, "hello", max_retries=3, timeout_seconds=5)
        assert result == "response to hello"

    @pytest.mark.asyncio
    async def test_success_after_retry(self):
        call_count = 0

        async def flaky_fn(input):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Temporary failure")
            return "success"

        result = await call_with_retry(flaky_fn, "test", max_retries=3, timeout_seconds=5)
        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_raises_error(self):
        async def slow_fn(input):
            await asyncio.sleep(10)
            return "too slow"

        with pytest.raises(AgentTimeoutError):
            await call_with_retry(slow_fn, "test", max_retries=1, timeout_seconds=0.1)

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        async def bad_fn(input):
            raise ValueError("Always fails")

        with pytest.raises(AgentCallError):
            await call_with_retry(bad_fn, "test", max_retries=2, timeout_seconds=5)
