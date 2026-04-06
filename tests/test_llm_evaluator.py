"""
Tests for src/evaluation/llm_evaluator.py
Tests prompt building and score parsing (including edge cases).
"""

import pytest
from src.evaluation.llm_evaluator import _parse_score, _build_prompt
from src.metrics.schemas import TestCase


class TestParseScore:
    """Tests for score parsing from LLM output."""

    def test_clean_json(self):
        raw = '{"score": 8, "rationale": "Good response"}'
        score, rationale, relevance = _parse_score(raw)
        assert score == 8.0
        assert rationale == "Good response"
        assert relevance is None

    def test_json_with_relevance(self):
        raw = '{"score": 7, "relevance_score": 9, "rationale": "Relevant and accurate"}'
        score, rationale, relevance = _parse_score(raw)
        assert score == 7.0
        assert relevance == 9.0
        assert rationale == "Relevant and accurate"

    def test_markdown_wrapped_json(self):
        raw = '```json\n{"score": 6, "rationale": "Partial answer"}\n```'
        score, rationale, _ = _parse_score(raw)
        assert score == 6.0
        assert rationale == "Partial answer"

    def test_extra_text_around_json(self):
        raw = 'Here is my evaluation:\n{"score": 9, "rationale": "Excellent"}\nThat was my assessment.'
        score, rationale, _ = _parse_score(raw)
        assert score == 9.0

    def test_score_as_string(self):
        raw = '{"score": "7", "rationale": "Okay"}'
        score, _, _ = _parse_score(raw)
        assert score == 7.0

    def test_score_clamped_high(self):
        raw = '{"score": 15, "rationale": "Overscored"}'
        score, _, _ = _parse_score(raw)
        assert score == 10.0

    def test_score_clamped_low(self):
        raw = '{"score": -3, "rationale": "Underscored"}'
        score, _, _ = _parse_score(raw)
        assert score == 0.0

    def test_fallback_regex(self):
        raw = "I would give this a score: 7 out of 10."
        score, rationale, _ = _parse_score(raw)
        assert score == 7.0
        assert "fallback" in rationale.lower()

    def test_completely_unparseable(self):
        raw = "I have no opinion on this matter whatsoever."
        with pytest.raises(ValueError):
            _parse_score(raw)

    def test_missing_rationale(self):
        raw = '{"score": 5}'
        score, rationale, _ = _parse_score(raw)
        assert score == 5.0
        assert rationale == "No rationale provided"


class TestBuildPrompt:
    """Tests for prompt construction."""

    def test_prompt_contains_all_parts(self):
        case = TestCase(
            id="test-001",
            input="What is 2+2?",
            expected_behavior="Answers with 4",
            category="normal",
            expected_pass=True,
        )
        rubric = "You are an evaluator. Score 0-10."
        prompt = _build_prompt(case, "The answer is 4.", rubric)

        assert "What is 2+2?" in prompt
        assert "Answers with 4" in prompt
        assert "The answer is 4." in prompt
        assert "You are an evaluator" in prompt
