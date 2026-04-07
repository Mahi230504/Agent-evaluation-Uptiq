"""
Async test execution engine.
Dispatches all test cases concurrently (with semaphore) and collects results.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from agents.base_agent import AbstractAgent
from src.config import Config
from src.metrics.schemas import TestCase, TestResult, EvalResult
from src.runner.retry import call_with_retry, AgentTimeoutError, AgentCallError
from src.runner.timing import timed_call
from src.evaluation.evaluator import evaluate
from src.evaluation.metrics.base import BaseEvalMetric
from src.reporting.logger import log_result


async def run_suite(
    agent: AbstractAgent,
    cases: list[TestCase],
    metrics: list[BaseEvalMetric] | None = None,
    log_path: Path | str | None = None,
) -> list[TestResult]:
    """
    Run the full test suite against an agent.

    Uses asyncio.Semaphore to cap concurrent requests and avoid rate limits.
    Writes intermediate results per-test for crash recovery.

    Args:
        agent: The agent to evaluate.
        cases: List of test cases to run.
        log_path: Optional path for intermediate JSONL logging.

    Returns:
        List of TestResult objects.
    """
    semaphore = asyncio.Semaphore(Config.MAX_CONCURRENT)

    _metrics = metrics or []

    async def _guarded_run(case: TestCase) -> TestResult:
        async with semaphore:
            return await _run_single(agent, case, _metrics)

    # Run all tests concurrently (bounded by semaphore)
    tasks = [_guarded_run(case) for case in cases]
    results: list[TestResult] = list(await asyncio.gather(*tasks, return_exceptions=False))

    # Log intermediate results
    if log_path:
        log_path = Path(log_path)
        for result in results:
            log_result(result, log_path)

    return results


async def _run_single(
    agent: AbstractAgent,
    case: TestCase,
    metrics: list[BaseEvalMetric] | None = None,
) -> TestResult:
    """
    Execute a single test case: run the agent, then evaluate the response.

    On agent error → returns TestResult with error field set and
    score=0 (excluded from aggregation by the metrics layer).
    """
    agent_response = ""
    duration_ms = 0.0
    error_msg: str | None = None

    try:
        # Call agent with retry + timeout, wrapped with timing
        agent_response, duration_ms = await timed_call(
            call_with_retry(
                fn=agent.run_agent,
                input=case.input,
                max_retries=Config.MAX_RETRIES,
                timeout_seconds=Config.AGENT_TIMEOUT_SECONDS,
            )
        )
    except AgentTimeoutError as e:
        error_msg = str(e)
        agent_response = ""
    except AgentCallError as e:
        error_msg = str(e)
        agent_response = ""
    except Exception as e:
        error_msg = f"Unexpected error: {type(e).__name__}: {e}"
        agent_response = ""

    # If agent errored out, create an error result (excluded from scoring)
    if error_msg:
        eval_result = EvalResult(
            score=0.0,
            passed=False,
            method="rule",
            rationale=f"Agent error: {error_msg}",
            error=error_msg,
        )
        return TestResult(
            test_case=case,
            agent_response=agent_response,
            eval_result=eval_result,
            duration_ms=duration_ms,
            error=error_msg,
        )

    # Evaluate the response through the metric pipeline
    eval_result = await evaluate(case, agent_response, metrics or [])

    return TestResult(
        test_case=case,
        agent_response=agent_response,
        eval_result=eval_result,
        duration_ms=duration_ms,
    )
