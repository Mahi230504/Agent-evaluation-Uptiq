"""
All Pydantic data models used across the framework.
Single source of truth for data structures.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """A single test case to evaluate an agent against."""
    id: str
    input: str
    expected_behavior: str
    category: Literal["normal", "edge", "adversarial", "safety"]
    expected_pass: bool
    weight: float = 1.0
    tags: list[str] = Field(default_factory=list)


class RuleResult(BaseModel):
    """Result from a deterministic rule-based check."""
    passed: bool
    confidence: Literal["high", "low"]
    matched_pattern: Optional[str] = None


class EvalResult(BaseModel):
    """Result from the evaluation cascade."""
    score: float = Field(ge=0, le=10, description="Score from 0-10")
    passed: bool
    method: Literal["rule", "llm_fast", "llm_slow", "rule+llm", "adversarial_rule", "adversarial_llm"]
    rationale: str
    model_used: Optional[str] = None
    error: Optional[str] = None  # Captures evaluation failures


class LLMEvalResult(BaseModel):
    """Detailed result from an LLM judge evaluation."""
    score: float = Field(ge=0, le=10)
    rationale: str
    model_used: str
    tokens_used: int = 0
    relevance_score: Optional[float] = Field(default=None, ge=0, le=10)


class TestResult(BaseModel):
    """Complete result for a single test case execution + evaluation."""
    test_case: TestCase
    agent_response: str
    eval_result: EvalResult
    duration_ms: float
    error: Optional[str] = None  # Agent-level errors (timeout, crash)


class LatencyStats(BaseModel):
    """Latency statistics across all test runs."""
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float


class RunReport(BaseModel):
    """Complete report for a full evaluation run."""
    run_id: str
    timestamp: str
    agent_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    error_tests: int
    safety_score: float
    accuracy_score: float
    robustness_score: float
    relevance_score: float
    overall_score: float
    suite_passed: bool
    results: list[TestResult]
    latency: LatencyStats
