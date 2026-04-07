"""
Answer Relevancy Metric.
Measures how relevant the agent's response is to the user's input.
Score: 0 (completely irrelevant) → 1 (fully relevant)
"""

from __future__ import annotations

from src.metrics.schemas import TestCase, MetricScore
from src.evaluation.metrics.base import (
    BaseEvalMetric, _call_gemini, _parse_json_response, _extract_score
)


_PROMPT_TEMPLATE = """\
You are an expert evaluator assessing the relevancy of an AI assistant's response to a user's query.

USER INPUT:
{input}

ASSISTANT RESPONSE:
{response}

TASK:
Evaluate how relevant the assistant's response is to the user's input.
A response is relevant if it directly addresses what was asked, stays on topic,
and provides useful information in relation to the query.

Return ONLY a JSON object with this exact format:
{{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one sentence explaining the score>"
}}

Scoring guide:
- 1.0 = Fully relevant, directly addresses the input
- 0.7 = Mostly relevant, minor off-topic content
- 0.5 = Partially relevant, significant portions off-topic
- 0.3 = Mostly irrelevant
- 0.0 = Completely irrelevant or empty
"""


class AnswerRelevancyMetric(BaseEvalMetric):
    """Assesses how relevant the agent's response is to the input query."""

    required_fields = ["input", "actual_output"]

    @property
    def name(self) -> str:
        return "Answer Relevancy"

    async def a_measure(self, case: TestCase, response: str) -> MetricScore:
        skip = self._check_fields(case, response)
        if skip:
            return self._skipped(skip)

        prompt = _PROMPT_TEMPLATE.format(
            input=case.input,
            response=response,
        )

        raw = await _call_gemini(prompt, self._get_model())
        data = _parse_json_response(raw)
        score, reason = _extract_score(data)
        return self._result(score, reason, threshold=0.5)
