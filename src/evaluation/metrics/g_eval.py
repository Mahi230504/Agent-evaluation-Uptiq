"""
GEval Metric — Custom Criteria Evaluation.
A flexible LLM-as-judge metric that evaluates responses against
a user-defined criteria string. Inspired by G-Eval (Liu et al., 2023).
Score: 0 → 1
"""

from __future__ import annotations

from src.metrics.schemas import TestCase, MetricScore
from src.evaluation.metrics.base import (
    BaseEvalMetric, _call_gemini, _parse_json_response, _extract_score
)


_DEFAULT_CRITERIA = (
    "Evaluate the overall quality of the response: accuracy, helpfulness, "
    "clarity, and completeness relative to the user's query."
)

_PROMPT_TEMPLATE = """\
You are an expert evaluator. Evaluate the following AI assistant's response using the specified criteria.

EVALUATION CRITERIA:
{criteria}

USER INPUT:
{input}

EXPECTED BEHAVIOR:
{expected_behavior}

ASSISTANT RESPONSE:
{response}

TASK:
Evaluate the assistant's response strictly according to the criteria above.
Be objective and focus only on the specified criteria.

Return ONLY a JSON object with this exact format:
{{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one sentence explaining your evaluation according to the criteria>"
}}
"""


class GEvalMetric(BaseEvalMetric):
    """
    General evaluation metric with a configurable criteria string.
    Can be used for any custom evaluation dimension.
    """

    required_fields = ["input", "actual_output"]

    def __init__(self, criteria: str = _DEFAULT_CRITERIA, use_slow_model: bool = False):
        self.criteria = criteria
        self.use_slow_model = use_slow_model

    @property
    def name(self) -> str:
        return "Custom Criteria (GEval)"

    async def a_measure(self, case: TestCase, response: str) -> MetricScore:
        skip = self._check_fields(case, response)
        if skip:
            return self._skipped(skip)

        prompt = _PROMPT_TEMPLATE.format(
            criteria=self.criteria,
            input=case.input,
            expected_behavior=case.expected_behavior,
            response=response,
        )

        raw = await _call_gemini(prompt, self._get_model())
        data = _parse_json_response(raw)
        score, reason = _extract_score(data)
        return self._result(score, reason, threshold=0.5)
