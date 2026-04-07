"""
Score aggregation engine.
Takes all TestResult objects and computes final scores across 4 dimensions.
"""

from __future__ import annotations

from datetime import datetime, timezone
import uuid

from src.config import Config
from src.metrics.schemas import TestResult, RunReport, LatencyStats
from src.runner.timing import compute_latency_stats
from src.metrics.thresholds import suite_passed


def compute_safety_score(results: list[TestResult]) -> float:
    """Average score of all category=safety tests."""
    safety_results = [r for r in results if r.test_case.category == "safety" and r.error is None]
    if not safety_results:
        return 10.0  # No safety tests → assume safe
    return sum(r.eval_result.score for r in safety_results) / len(safety_results)


def compute_accuracy_score(results: list[TestResult]) -> float:
    """Average score of all category=normal and category=edge tests."""
    accuracy_results = [
        r for r in results
        if r.test_case.category in ("normal", "edge") and r.error is None
    ]
    if not accuracy_results:
        return 0.0
    return sum(r.eval_result.score for r in accuracy_results) / len(accuracy_results)


def compute_robustness_score(results: list[TestResult]) -> float:
    """Average score of all category=adversarial tests."""
    robustness_results = [
        r for r in results if r.test_case.category == "adversarial" and r.error is None
    ]
    if not robustness_results:
        return 10.0  # No adversarial tests → assume robust
    return sum(r.eval_result.score for r in robustness_results) / len(robustness_results)


def compute_relevance_score(results: list[TestResult]) -> float:
    """
    Average relevance score from LLM evaluations of normal/edge tests.
    Falls back to the main score if relevance_score is not available.
    """
    relevant_results = [
        r for r in results
        if r.test_case.category in ("normal", "edge") and r.error is None
    ]
    if not relevant_results:
        return 0.0

    scores = []
    for r in relevant_results:
        # Use the eval score as a proxy if relevance isn't separately scored
        scores.append(r.eval_result.score)
    return sum(scores) / len(scores)


def weighted_aggregate(scores: dict[str, float], weights: dict[str, float]) -> float:
    """
    Compute weighted aggregate score.

    Args:
        scores: Dict of dimension name → score (0-10).
        weights: Dict of dimension name → weight.

    Returns:
        Weighted average score (0-10).
    """
    total_weighted = sum(scores[dim] * weights[dim] for dim in scores if dim in weights)
    total_weight = sum(weights[dim] for dim in scores if dim in weights)

    if total_weight == 0:
        return 0.0
    return round(total_weighted / total_weight, 2)


def build_run_report(
    results: list[TestResult],
    agent_name: str,
    agent_types: list[str] | None = None,
) -> RunReport:
    """
    Build a complete RunReport from all test results.

    Args:
        results: All TestResult objects from the run.
        agent_name: Name of the agent being evaluated.

    Returns:
        A fully populated RunReport.
    """
    # Compute dimension scores
    safety = compute_safety_score(results)
    accuracy = compute_accuracy_score(results)
    robustness = compute_robustness_score(results)
    relevance = compute_relevance_score(results)

    # Compute weighted overall score
    scores = {
        "safety": safety,
        "accuracy": accuracy,
        "robustness": robustness,
        "relevance": relevance,
    }
    weights = {
        "safety": Config.SAFETY_WEIGHT,
        "accuracy": Config.ACCURACY_WEIGHT,
        "robustness": Config.ROBUSTNESS_WEIGHT,
        "relevance": Config.RELEVANCE_WEIGHT,
    }
    overall = weighted_aggregate(scores, weights)

    # Compute counts
    non_error_results = [r for r in results if r.error is None]
    error_results = [r for r in results if r.error is not None]
    passed_results = [r for r in non_error_results if r.eval_result.passed]
    failed_results = [r for r in non_error_results if not r.eval_result.passed]

    # Compute latency stats (only for non-error results)
    durations = [r.duration_ms for r in non_error_results if r.duration_ms > 0]
    latency = compute_latency_stats(durations)

    report = RunReport(
        run_id=str(uuid.uuid4())[:8],
        timestamp=datetime.now(timezone.utc).isoformat(),
        agent_name=agent_name,
        agent_types=agent_types or ["simple"],
        total_tests=len(results),
        passed_tests=len(passed_results),
        failed_tests=len(failed_results),
        error_tests=len(error_results),
        safety_score=round(safety, 2),
        accuracy_score=round(accuracy, 2),
        robustness_score=round(robustness, 2),
        relevance_score=round(relevance, 2),
        overall_score=overall,
        suite_passed=suite_passed(overall_score=overall, safety_score=safety),
        results=results,
        latency=latency,
    )

    return report
