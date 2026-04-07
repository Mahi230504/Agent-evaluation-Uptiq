"""
Faithfulness Metric (RAG).
Measures whether the agent's response is grounded in the retrieved context.
Detects hallucinations relative to provided retrieval context.
Score: 0 (fabricated) → 1 (fully grounded)
"""

from __future__ import annotations

from src.metrics.schemas import TestCase, MetricScore
from src.evaluation.metrics.base import (
    BaseEvalMetric, _call_gemini, _parse_json_response, _extract_score
)


_PROMPT_TEMPLATE = """\
You are an expert evaluator assessing whether an AI assistant's response is faithful to the retrieved context.

USER INPUT:
{input}

RETRIEVED CONTEXT:
{context}

ASSISTANT RESPONSE:
{response}

TASK:
Determine whether every factual claim in the assistant's response is supported by the retrieved context.
A faithful response only makes claims that are directly supported or inferable from the context.
It does NOT introduce external knowledge or fabricated information.

Return ONLY a JSON object with this exact format:
{{
  "score": <float between 0.0 and 1.0>,
  "reason": "<one sentence explaining which claims, if any, are not supported>"
}}

Scoring guide:
- 1.0 = All claims fully supported by the context
- 0.7 = Most claims supported, minor unsupported details
- 0.5 = Some claims unsupported or speculative
- 0.2 = Many fabricated claims
- 0.0 = Response is entirely fabricated or contradicts the context
"""


class FaithfulnessMetric(BaseEvalMetric):
    """
    RAG metric: checks whether the response is grounded in retrieval_context.
    Skipped if retrieval_context is not provided.
    """

    required_fields = ["input", "actual_output", "retrieval_context"]
    use_slow_model = True  # Faithfulness needs deeper reasoning

    @property
    def name(self) -> str:
        return "Faithfulness"

    async def a_measure(self, case: TestCase, response: str) -> MetricScore:
        skip = self._check_fields(case, response)
        if skip:
            return self._skipped(skip)

        context_str = "\n\n---\n\n".join(case.retrieval_context)  # type: ignore[arg-type]
        prompt = _PROMPT_TEMPLATE.format(
            input=case.input,
            context=context_str,
            response=response,
        )

        raw = await _call_gemini(prompt, self._get_model())
        data = _parse_json_response(raw)
        score, reason = _extract_score(data)
        return self._result(score, reason, threshold=0.6)
