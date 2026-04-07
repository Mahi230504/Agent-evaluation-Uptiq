"""
Base class for all evaluation metrics.
All metrics are self-contained: they use google.genai for LLM-as-judge
and return a normalised score in [0, 1].
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod

import google.genai as genai
from google.genai import types as genai_types

from src.config import Config
from src.metrics.schemas import TestCase, MetricScore

# ── Lazy Gemini client ─────────────────────────────────────────────────────────
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not Config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Cannot use LLM-as-judge metrics."
            )
        _client = genai.Client(api_key=Config.GEMINI_API_KEY)
    return _client


async def _call_gemini(prompt: str, model: str) -> str:
    """Make an async Gemini call and return raw text."""
    client = _get_client()
    response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=512,
        ),
    )
    return response.text or ""


def _parse_json_response(raw: str) -> dict:
    """
    Robustly extract a JSON object from an LLM response.
    Handles markdown fences and extra surrounding text.
    """
    # Strip markdown code fences
    md = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
    text = md.group(1).strip() if md else raw.strip()

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first JSON object
    obj = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if obj:
        try:
            return json.loads(obj.group())
        except json.JSONDecodeError:
            pass

    return {}


def _extract_score(data: dict, fallback: float = 0.5) -> tuple[float, str]:
    """Extract score (0-1) and reason from a parsed dict."""
    raw_score = data.get("score", fallback)
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = fallback
    score = max(0.0, min(1.0, score))
    reason = str(data.get("reason", data.get("rationale", "No reason provided.")))
    return score, reason


# ── Base Metric ────────────────────────────────────────────────────────────────

class BaseEvalMetric(ABC):
    """
    Abstract base for all evaluation metrics.

    Subclasses must implement:
    - `required_fields`: list of TestCase field names that must be present
    - `a_measure()`: async method returning a MetricScore
    """

    #: Fields that must be non-None on the TestCase to run this metric
    required_fields: list[str] = ["input"]

    #: Which Gemini model to use (fast = flash, slow = pro)
    use_slow_model: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name for this metric."""
        ...

    def _get_model(self) -> str:
        return Config.JUDGE_MODEL_SLOW if self.use_slow_model else Config.JUDGE_MODEL_FAST

    def _check_fields(self, case: TestCase, response: str) -> str | None:
        """
        Return a skip reason string if required fields are missing,
        or None if all fields are present.
        """
        for field in self.required_fields:
            if field == "actual_output":
                if not response:
                    return "agent_response is empty"
            else:
                val = getattr(case, field, None)
                if val is None or (isinstance(val, list) and len(val) == 0):
                    return f"required field '{field}' is missing from test case"
        return None

    @abstractmethod
    async def a_measure(self, case: TestCase, response: str) -> MetricScore:
        """Evaluate and return a MetricScore."""
        ...

    def _skipped(self, reason: str) -> MetricScore:
        return MetricScore(
            name=self.name,
            score=0.0,
            passed=False,
            reason=reason,
            skipped=True,
            skip_reason=reason,
        )

    def _result(self, score: float, reason: str, threshold: float = 0.5) -> MetricScore:
        return MetricScore(
            name=self.name,
            score=round(score, 4),
            passed=score >= threshold,
            reason=reason,
        )
