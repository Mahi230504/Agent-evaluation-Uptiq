"""
All Pydantic data models used across the framework.
Single source of truth for data structures.
"""

from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A tool call made by an agent."""
    name: str
    input_parameters: Optional[dict] = None
    output: Optional[str] = None


class TestCase(BaseModel):
    """A single test case to evaluate an agent against."""
    id: str
    input: str
    expected_behavior: str
    category: Literal["normal", "edge", "adversarial", "safety"]
    expected_pass: bool
    weight: float = 1.0
    tags: list[str] = Field(default_factory=list)

    # RAG agent fields
    retrieval_context: list[str] | None = None  # chunks retrieved by the agent
    context: list[str] | None = None            # ground-truth reference context

    # Tool-using agent fields
    tools_called: list[ToolCall] | None = None       # actual calls made
    expected_tools: list[str] | None = None          # expected tool names


class RuleResult(BaseModel):
    """Result from a deterministic rule-based check."""
    passed: bool
    confidence: Literal["high", "low"]
    matched_pattern: Optional[str] = None


class MetricScore(BaseModel):
    """Score from a single named metric."""
    name: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    reason: str
    skipped: bool = False
    skip_reason: Optional[str] = None


class EvalResult(BaseModel):
    """Result from the evaluation pipeline."""
    score: float = Field(ge=0, le=10, description="Aggregated score 0-10")
    passed: bool
    method: Literal["rule", "llm", "rule+llm", "skipped"]
    rationale: str
    model_used: Optional[str] = None
    error: Optional[str] = None
    metric_scores: dict[str, MetricScore] = Field(default_factory=dict)


class TestResult(BaseModel):
    """Complete result for a single test case execution + evaluation."""
    test_case: TestCase
    agent_response: str
    eval_result: EvalResult
    duration_ms: float
    error: Optional[str] = None


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
    agent_types: list[str]          # e.g. ["simple", "rag"]
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
