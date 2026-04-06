"""
Tests for src/metrics/aggregator.py
Tests weighted score math, pass/fail logic, and error exclusion.
"""

import pytest
from src.metrics.schemas import TestCase, EvalResult, TestResult, LatencyStats
from src.metrics.aggregator import (
    compute_safety_score,
    compute_accuracy_score,
    compute_robustness_score,
    compute_relevance_score,
    weighted_aggregate,
)


def _make_result(category: str, score: float, passed: bool, error: str | None = None) -> TestResult:
    """Helper to create a TestResult for testing."""
    return TestResult(
        test_case=TestCase(
            id=f"test-{category}-{score}",
            input="test input",
            expected_behavior="test expected",
            category=category,
            expected_pass=True,
        ),
        agent_response="test response",
        eval_result=EvalResult(
            score=score,
            passed=passed,
            method="rule",
            rationale="test rationale",
        ),
        duration_ms=100.0,
        error=error,
    )


class TestComputeSafetyScore:
    def test_all_pass(self):
        results = [
            _make_result("safety", 10.0, True),
            _make_result("safety", 9.0, True),
        ]
        score = compute_safety_score(results)
        assert score == 9.5

    def test_mixed(self):
        results = [
            _make_result("safety", 10.0, True),
            _make_result("safety", 0.0, False),
        ]
        score = compute_safety_score(results)
        assert score == 5.0

    def test_excludes_errors(self):
        results = [
            _make_result("safety", 10.0, True),
            _make_result("safety", 0.0, False, error="timeout"),
        ]
        score = compute_safety_score(results)
        assert score == 10.0  # Error test excluded

    def test_no_safety_tests(self):
        results = [_make_result("normal", 8.0, True)]
        score = compute_safety_score(results)
        assert score == 10.0  # Default when no safety tests

    def test_ignores_other_categories(self):
        results = [
            _make_result("safety", 9.0, True),
            _make_result("normal", 3.0, False),
            _make_result("adversarial", 5.0, False),
        ]
        score = compute_safety_score(results)
        assert score == 9.0


class TestComputeAccuracyScore:
    def test_normal_and_edge(self):
        results = [
            _make_result("normal", 8.0, True),
            _make_result("edge", 6.0, True),
        ]
        score = compute_accuracy_score(results)
        assert score == 7.0

    def test_excludes_errors(self):
        results = [
            _make_result("normal", 8.0, True),
            _make_result("normal", 2.0, False, error="crash"),
        ]
        score = compute_accuracy_score(results)
        assert score == 8.0


class TestComputeRobustnessScore:
    def test_adversarial(self):
        results = [
            _make_result("adversarial", 10.0, True),
            _make_result("adversarial", 8.0, True),
        ]
        score = compute_robustness_score(results)
        assert score == 9.0

    def test_no_adversarial(self):
        results = [_make_result("normal", 8.0, True)]
        score = compute_robustness_score(results)
        assert score == 10.0


class TestWeightedAggregate:
    def test_equal_weights(self):
        scores = {"a": 8.0, "b": 6.0}
        weights = {"a": 1.0, "b": 1.0}
        result = weighted_aggregate(scores, weights)
        assert result == 7.0

    def test_different_weights(self):
        scores = {"safety": 10.0, "accuracy": 5.0}
        weights = {"safety": 2.0, "accuracy": 1.0}
        # (10*2 + 5*1) / (2+1) = 25/3 = 8.33
        result = weighted_aggregate(scores, weights)
        assert result == 8.33

    def test_four_dimensions(self):
        scores = {"safety": 9.0, "robustness": 8.0, "accuracy": 7.0, "relevance": 6.0}
        weights = {"safety": 2.0, "robustness": 1.5, "accuracy": 1.0, "relevance": 0.75}
        # (9*2 + 8*1.5 + 7*1 + 6*0.75) / (2+1.5+1+0.75)
        # = (18 + 12 + 7 + 4.5) / 5.25 = 41.5 / 5.25 = 7.904...
        result = weighted_aggregate(scores, weights)
        assert 7.9 <= result <= 7.91

    def test_empty_scores(self):
        result = weighted_aggregate({}, {})
        assert result == 0.0
