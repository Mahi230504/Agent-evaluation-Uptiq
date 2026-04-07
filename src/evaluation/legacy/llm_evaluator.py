"""
LLM-as-a-judge evaluator using Google Gemini (google-genai SDK).
Makes API calls to grade agent responses against rubrics.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import google.genai as genai
from google.genai import types as genai_types

from src.config import Config
from src.metrics.schemas import TestCase, LLMEvalResult


# Lazy-initialized Gemini client
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    """Lazily create and return a configured Gemini client."""
    global _client
    if _client is None:
        if not Config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Cannot use LLM judge. "
                "Set it in your .env file or environment variables."
            )
        _client = genai.Client(api_key=Config.GEMINI_API_KEY)
    return _client


def _load_rubric(rubric_path: str) -> str:
    """Load a rubric text file."""
    path = Path(rubric_path)
    if not path.exists():
        raise FileNotFoundError(f"Rubric file not found: {rubric_path}")
    return path.read_text().strip()


def _build_prompt(case: TestCase, response: str, rubric: str) -> str:
    """
    Build the judge prompt combining rubric, input, expected behavior, and response.
    """
    return f"""{rubric}

---

INPUT:
{case.input}

EXPECTED BEHAVIOR:
{case.expected_behavior}

AGENT'S RESPONSE:
{response}"""


def _parse_score(raw: str) -> tuple[float, str, Optional[float]]:
    """
    Parse the score, rationale, and optional relevance_score from LLM output.

    Handles:
    - Clean JSON
    - Markdown-wrapped JSON (```json ... ```)
    - Extra text before/after JSON
    - Score as string instead of int

    Returns:
        Tuple of (score, rationale, relevance_score).

    Raises:
        ValueError: If parsing fails completely.
    """
    json_str = raw.strip()

    # Remove markdown code fences if present
    md_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", json_str, re.DOTALL)
    if md_match:
        json_str = md_match.group(1).strip()

    # Try to find JSON object in the text
    json_match = re.search(r"\{[^{}]*\}", json_str)
    if json_match:
        json_str = json_match.group()

    try:
        data = json.loads(json_str)
        score = float(data.get("score", 0))
        rationale = str(data.get("rationale", "No rationale provided"))
        relevance_score: Optional[float] = data.get("relevance_score")
        if relevance_score is not None:
            relevance_score = float(relevance_score)

        # Clamp score to valid range
        score = max(0.0, min(10.0, score))
        if relevance_score is not None:
            relevance_score = max(0.0, min(10.0, relevance_score))

        return score, rationale, relevance_score

    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: try to extract score from text using regex
    score_match = re.search(r"(?:score|rating)[:\s]*(\d+(?:\.\d+)?)", raw, re.IGNORECASE)
    if score_match:
        score = float(score_match.group(1))
        score = max(0.0, min(10.0, score))
        return score, "Score extracted from text (parsing fallback)", None

    raise ValueError(f"Could not parse score from LLM response: {raw[:200]}")


async def judge(
    case: TestCase,
    response: str,
    rubric_path: str,
    model: str,
) -> LLMEvalResult:
    """
    Use an LLM to judge an agent's response against a rubric.

    Args:
        case: The test case.
        response: The agent's response.
        rubric_path: Path to the rubric text file.
        model: The Gemini model to use.

    Returns:
        LLMEvalResult with score, rationale, model used, and tokens.
    """
    client = _get_client()

    rubric = _load_rubric(rubric_path)
    prompt = _build_prompt(case, response, rubric)

    gen_response = await client.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=256,
        ),
    )

    raw_text: str = ""
    if gen_response.text:
        raw_text = gen_response.text

    tokens_used = 0
    if gen_response.usage_metadata and gen_response.usage_metadata.total_token_count:
        tokens_used = gen_response.usage_metadata.total_token_count

    try:
        score, rationale, relevance_score = _parse_score(raw_text)
    except ValueError as e:
        return LLMEvalResult(
            score=5.0,
            rationale=f"Parse error (defaulting to 5): {str(e)}",
            model_used=model,
            tokens_used=tokens_used,
            relevance_score=None,
        )

    return LLMEvalResult(
        score=score,
        rationale=rationale,
        model_used=model,
        tokens_used=tokens_used,
        relevance_score=relevance_score,
    )
