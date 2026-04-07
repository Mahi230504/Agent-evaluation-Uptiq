"""
Metric Selector — maps agent types and test categories to the right metric set.
Agent types: "simple", "rag", "tool"
Test categories: "normal", "edge", "adversarial", "safety"
"""

from __future__ import annotations

from src.evaluation.metrics.base import BaseEvalMetric
from src.evaluation.metrics.answer_relevancy import AnswerRelevancyMetric
from src.evaluation.metrics.faithfulness import FaithfulnessMetric
from src.evaluation.metrics.hallucination import HallucinationMetric
from src.evaluation.metrics.toxicity import ToxicityMetric
from src.evaluation.metrics.pii_leakage import PIILeakageMetric
from src.evaluation.metrics.task_completion import TaskCompletionMetric
from src.evaluation.metrics.tool_correctness import ToolCorrectnessMetric
from src.evaluation.metrics.g_eval import GEvalMetric

# Agent type labels
AGENT_TYPE_SIMPLE = "simple"
AGENT_TYPE_RAG = "rag"
AGENT_TYPE_TOOL = "tool"

ALL_AGENT_TYPES = [AGENT_TYPE_SIMPLE, AGENT_TYPE_RAG, AGENT_TYPE_TOOL]

# Human-readable labels for the UI
AGENT_TYPE_LABELS: dict[str, str] = {
    AGENT_TYPE_SIMPLE: "Simple Chatbot",
    AGENT_TYPE_RAG:    "RAG Agent",
    AGENT_TYPE_TOOL:   "Tool-Using Agent",
}

# Metric descriptions for the UI
METRIC_DESCRIPTIONS: dict[str, str] = {
    "Answer Relevancy":        "Measures how relevant the response is to the user's query.",
    "Faithfulness":            "Checks if response is grounded in retrieved context (RAG only).",
    "Hallucination":           "Detects fabricated or false factual claims.",
    "Toxicity":                "Detects harmful, offensive, or inappropriate content.",
    "PII Leakage":             "Detects exposure of personally identifiable information.",
    "Task Completion":         "Evaluates whether the agent completed the intended task.",
    "Tool Correctness":        "Checks if the right tools were called (tool agents only).",
    "Custom Criteria (GEval)": "Evaluates against a custom user-defined rubric.",
}


def select_metrics(
    agent_types: list[str],
    custom_criteria: str | None = None,
    include_g_eval: bool = False,
) -> list[BaseEvalMetric]:
    """
    Return the appropriate metric set for the given agent types.

    Core metrics always included:
    - AnswerRelevancyMetric  (all)
    - ToxicityMetric         (all)
    - PIILeakageMetric       (all)

    RAG-specific (activated when "rag" in agent_types):
    - FaithfulnessMetric
    - HallucinationMetric

    Tool-specific (activated when "tool" in agent_types):
    - TaskCompletionMetric
    - ToolCorrectnessMetric

    Optional:
    - GEvalMetric (when include_g_eval=True)

    Args:
        agent_types: list of agent type strings (e.g. ["simple", "rag"])
        custom_criteria: custom rubric string for GEval (uses default if None)
        include_g_eval: whether to add the GEval custom metric

    Returns:
        Ordered list of metric instances to run.
    """
    metrics: list[BaseEvalMetric] = [
        AnswerRelevancyMetric(),
        ToxicityMetric(),
        PIILeakageMetric(),
    ]

    if AGENT_TYPE_RAG in agent_types:
        metrics.append(FaithfulnessMetric())
        metrics.append(HallucinationMetric())

    if AGENT_TYPE_TOOL in agent_types:
        metrics.append(TaskCompletionMetric())
        metrics.append(ToolCorrectnessMetric())
    elif AGENT_TYPE_SIMPLE in agent_types or AGENT_TYPE_RAG in agent_types:
        # Task completion is useful for non-tool agents too
        metrics.append(TaskCompletionMetric())

    if include_g_eval:
        metrics.append(GEvalMetric(criteria=custom_criteria or ""))

    return metrics


def metric_names_for_types(agent_types: list[str], include_g_eval: bool = False) -> list[str]:
    """Return the display names of metrics that would be activated for given agent types."""
    return [m.name for m in select_metrics(agent_types, include_g_eval=include_g_eval)]
