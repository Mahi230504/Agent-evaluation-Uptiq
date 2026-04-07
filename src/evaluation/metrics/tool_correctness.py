"""
Tool Correctness Metric (Tool-Using Agents).
Measures whether the agent called the right tools with correct parameters.
Skipped if no expected_tools or tools_called are provided.
Score: 0 (wrong tools) → 1 (correct tools used)
"""

from __future__ import annotations

from src.metrics.schemas import TestCase, MetricScore
from src.evaluation.metrics.base import (
    BaseEvalMetric, _call_gemini, _parse_json_response, _extract_score
)


_PROMPT_TEMPLATE = """\
You are an expert evaluator assessing whether an AI agent called the correct tools.

USER INPUT:
{input}

EXPECTED TOOLS TO BE CALLED:
{expected_tools}

TOOLS ACTUALLY CALLED:
{actual_tools}

TASK:
Evaluate whether the agent called the appropriate tools for this task.
Check whether:
1. All expected tools were called
2. No irrelevant tools were called
3. Tool parameters appear reasonable

Return ONLY a JSON object with this exact format:
{{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one sentence about which tools were correct or incorrect>"
}}

Scoring guide:
- 1.0 = All expected tools called correctly, no spurious calls
- 0.7 = Most tools correct, minor issues
- 0.5 = Some expected tools called, some missing or wrong
- 0.2 = Mostly wrong tools or no relevant tools
- 0.0 = No correct tools called
"""

_RULE_PROMPT = """\
You are evaluating whether an AI agent's tool calls were appropriate for the given task.

USER INPUT:
{input}

TOOLS ACTUALLY CALLED:
{actual_tools}

TASK:
Without knowing the expected tools, assess whether the tools called appear
relevant and useful for the user's request. Were the right kinds of tools used?

Return ONLY a JSON object:
{{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one sentence about tool usage appropriateness>"
}}
"""


class ToolCorrectnessMetric(BaseEvalMetric):
    """
    Evaluates tool usage correctness.
    Requires at least tools_called to be set. Uses expected_tools for
    precise comparison when available.
    Skipped if tools_called is not provided.
    """

    required_fields = ["input", "tools_called"]

    @property
    def name(self) -> str:
        return "Tool Correctness"

    async def a_measure(self, case: TestCase, response: str) -> MetricScore:
        skip = self._check_fields(case, response)
        if skip:
            return self._skipped(skip)

        actual_tools_str = "\n".join(
            f"  - {t.name}({t.input_parameters or {}}) → {t.output or 'N/A'}"
            for t in case.tools_called  # type: ignore[union-attr]
        )

        if case.expected_tools:
            expected_str = "\n".join(f"  - {name}" for name in case.expected_tools)
            prompt = _PROMPT_TEMPLATE.format(
                input=case.input,
                expected_tools=expected_str,
                actual_tools=actual_tools_str,
            )
        else:
            # No expected tools — do general appropriateness check
            prompt = _RULE_PROMPT.format(
                input=case.input,
                actual_tools=actual_tools_str,
            )

        raw = await _call_gemini(prompt, self._get_model())
        data = _parse_json_response(raw)
        score, reason = _extract_score(data)
        return self._result(score, reason, threshold=0.6)
