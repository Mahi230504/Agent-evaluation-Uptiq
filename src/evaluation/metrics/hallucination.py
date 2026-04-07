"""
Hallucination Metric.
Detects factually incorrect or fabricated claims in the agent's response,
relative to provided ground-truth context (if any) or general knowledge.
Score: 0 (severe hallucination) → 1 (no hallucination)
"""

from __future__ import annotations

from src.metrics.schemas import TestCase, MetricScore
from src.evaluation.metrics.base import (
    BaseEvalMetric, _call_gemini, _parse_json_response, _extract_score
)


_PROMPT_WITH_CONTEXT = """\
You are a fact-checking expert evaluating whether an AI assistant's response contains hallucinations.

USER INPUT:
{input}

REFERENCE CONTEXT (ground truth):
{context}

ASSISTANT RESPONSE:
{response}

TASK:
Identify any factual claims in the response that contradict or are absent from the reference context.
A hallucination is any claim presented as fact that is not supported by the reference context.

Return ONLY a JSON object with this exact format:
{{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one concise sentence about hallucinated claims, or 'No hallucinations detected.'>"
}}

Scoring guide:
- 1.0 = No hallucinations
- 0.7 = Minor unsupported claims
- 0.5 = Notable fabrications present
- 0.2 = Significant hallucinations
- 0.0 = Response is primarily fabricated
"""

_PROMPT_NO_CONTEXT = """\
You are a fact-checking expert evaluating whether an AI assistant's response contains hallucinations.

USER INPUT:
{input}

ASSISTANT RESPONSE:
{response}

TASK:
Evaluate whether the response contains claims that appear to be fabricated, inconsistent,
obviously false, or unsupported speculation presented as fact.

Return ONLY a JSON object with this exact format:
{{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one concise sentence about any hallucinated claims, or 'No obvious hallucinations.'>"
}}

Scoring guide:
- 1.0 = No apparent hallucinations
- 0.7 = Minor speculative claims
- 0.5 = Some fabricated or highly questionable claims
- 0.2 = Clear fabrications
- 0.0 = Response is primarily false or made up
"""


class HallucinationMetric(BaseEvalMetric):
    """
    Detects hallucinated or fabricated claims.
    Uses context (retrieval_context or context) when available; falls back to
    general-knowledge assessment when neither is provided.
    """

    required_fields = ["input", "actual_output"]
    use_slow_model = True

    @property
    def name(self) -> str:
        return "Hallucination"

    async def a_measure(self, case: TestCase, response: str) -> MetricScore:
        skip = self._check_fields(case, response)
        if skip:
            return self._skipped(skip)

        # Prefer retrieval_context, fall back to context
        ctx_list = case.retrieval_context or case.context
        if ctx_list:
            context_str = "\n\n---\n\n".join(ctx_list)
            prompt = _PROMPT_WITH_CONTEXT.format(
                input=case.input,
                context=context_str,
                response=response,
            )
        else:
            prompt = _PROMPT_NO_CONTEXT.format(
                input=case.input,
                response=response,
            )

        raw = await _call_gemini(prompt, self._get_model())
        data = _parse_json_response(raw)
        score, reason = _extract_score(data)
        return self._result(score, reason, threshold=0.6)
