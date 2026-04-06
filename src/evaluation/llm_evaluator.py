"""
LLM-as-a-judge evaluator using Google Gemini.
Makes API calls to grade agent responses against rubrics.
"""

import json
import re
from pathlib import Path
from typing import Optional

import google.generativeai as genai

from src.config import Config
from src.metrics.schemas import TestCase, LLMEvalResult


# Configure Gemini on first use
_configured = False


def _ensure_configured():
    """Lazy-configure the Gemini API."""
    global _configured
    if not _configured:
        if not Config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Cannot use LLM judge. "
                "Set it in your .env file or environment variables."
            )
        genai.configure(api_key=Config.GEMINI_API_KEY)
        _configured = True


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
    # Try to extract JSON from the response
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
        relevance_score = data.get("relevance_score")
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
    _ensure_configured()

    rubric = _load_rubric(rubric_path)
    prompt = _build_prompt(case, response, rubric)

    # Call Gemini
    gen_model = genai.GenerativeModel(model)
    gen_response = await gen_model.generate_content_async(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=256,
        ),
    )

    raw_text = gen_response.text or ""
    tokens_used = 0

    # Try to get token count from usage metadata
    if hasattr(gen_response, "usage_metadata") and gen_response.usage_metadata:
        tokens_used = getattr(gen_response.usage_metadata, "total_token_count", 0)

    try:
        score, rationale, relevance_score = _parse_score(raw_text)
    except ValueError as e:
        # If parsing fails completely, return a low-confidence result
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
