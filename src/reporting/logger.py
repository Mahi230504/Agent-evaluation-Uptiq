"""
Structured JSONL logger for observability.
One JSON line per test result, plus a final summary line.
"""

import json
from pathlib import Path

from src.metrics.schemas import TestResult, RunReport


def log_result(result: TestResult, log_path: Path) -> None:
    """
    Log a single TestResult as a JSON line.

    Args:
        result: The test result to log.
        log_path: Path to the JSONL log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "test_id": result.test_case.id,
        "category": result.test_case.category,
        "input": result.test_case.input[:200],  # Truncate for readability
        "agent_response": result.agent_response[:500],  # Truncate
        "score": result.eval_result.score,
        "passed": result.eval_result.passed,
        "method": result.eval_result.method,
        "rationale": result.eval_result.rationale,
        "duration_ms": round(result.duration_ms, 2),
        "error": result.error,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_run_summary(report: RunReport, log_path: Path) -> None:
    """
    Log the final RunReport summary as a JSON line.

    Args:
        report: The complete run report.
        log_path: Path to the JSONL log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "_type": "summary",
        "run_id": report.run_id,
        "timestamp": report.timestamp,
        "agent_name": report.agent_name,
        "total_tests": report.total_tests,
        "passed_tests": report.passed_tests,
        "failed_tests": report.failed_tests,
        "error_tests": report.error_tests,
        "safety_score": report.safety_score,
        "accuracy_score": report.accuracy_score,
        "robustness_score": report.robustness_score,
        "relevance_score": report.relevance_score,
        "overall_score": report.overall_score,
        "suite_passed": report.suite_passed,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
