"""
Task Completion Metric.
Measures whether the agent successfully completed the intended task
implied by the user's input.
Score: 0 (task failed) → 1 (task fully completed)
"""

from __future__ import annotations

from src.metrics.schemas import TestCase, MetricScore
from src.evaluation.metrics.base import (
    BaseEvalMetric, _call_gemini, _parse_json_response, _extract_score
)


_PROMPT_TEMPLATE = """\
You are an expert evaluator assessing whether an AI assistant successfully completed the user's task.

USER INPUT:
{input}

EXPECTED BEHAVIOR:
{expected_behavior}

ASSISTANT RESPONSE:
{response}

{tools_section}

TASK:
Evaluate whether the assistant's response successfully accomplishes what the user asked for.
Consider whether:
1. The core task was completed
2. The response is actionable and useful
3. Key requirements from the input are addressed
4. The expected behavior was met

Return ONLY a JSON object with this exact format:
{{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one sentence explaining the completion level>"
}}

Scoring guide:
- 1.0 = Task fully completed, all requirements met
- 0.7 = Task mostly completed with minor gaps
- 0.5 = Task partially completed
- 0.2 = Task barely addressed
- 0.0 = Task not completed at all
"""

_TOOLS_SECTION = """\
TOOLS CALLED BY AGENT:
{tools}
"""


class TaskCompletionMetric(BaseEvalMetric):
    """Evaluates whether the agent successfully completed the intended task."""

    required_fields = ["input", "actual_output"]
    use_slow_model = True

    @property
    def name(self) -> str:
        return "Task Completion"

    async def a_measure(self, case: TestCase, response: str) -> MetricScore:
        skip = self._check_fields(case, response)
        if skip:
            return self._skipped(skip)

        tools_section = ""
        if case.tools_called:
            tools_str = "\n".join(
                f"  - {t.name}({t.input_parameters or {}}) → {t.output or 'N/A'}"
                for t in case.tools_called
            )
            tools_section = _TOOLS_SECTION.format(tools=tools_str)

        prompt = _PROMPT_TEMPLATE.format(
            input=case.input,
            expected_behavior=case.expected_behavior,
            response=response,
            tools_section=tools_section,
        )

        raw = await _call_gemini(prompt, self._get_model())
        data = _parse_json_response(raw)
        score, reason = _extract_score(data)
        return self._result(score, reason, threshold=0.6)
