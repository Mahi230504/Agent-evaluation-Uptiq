"""
Tests for src/loader/test_loader.py and src/loader/test_validator.py
Tests JSON loading, schema validation, and category filtering.
"""

import json
import tempfile
from pathlib import Path

import pytest
from src.loader.test_validator import validate, validate_batch
from src.loader.test_loader import _load_file


class TestValidator:
    """Tests for the test case validator."""

    def test_valid_test_case(self):
        raw = {
            "id": "test-001",
            "input": "What is 2+2?",
            "expected_behavior": "Answers with 4",
            "category": "normal",
            "expected_pass": True,
        }
        assert validate(raw) is True

    def test_valid_with_optional_fields(self):
        raw = {
            "id": "test-002",
            "input": "Test input",
            "expected_behavior": "Expected output",
            "category": "edge",
            "expected_pass": False,
            "weight": 1.5,
            "tags": ["edge", "language"],
        }
        assert validate(raw) is True

    def test_missing_required_field(self):
        raw = {
            "id": "test-003",
            "input": "Missing category",
            "expected_behavior": "Should fail",
            # Missing: category, expected_pass
        }
        import jsonschema
        with pytest.raises(jsonschema.ValidationError):
            validate(raw)

    def test_invalid_category(self):
        raw = {
            "id": "test-004",
            "input": "Bad category",
            "expected_behavior": "Should fail",
            "category": "unknown",
            "expected_pass": True,
        }
        import jsonschema
        with pytest.raises(jsonschema.ValidationError):
            validate(raw)

    def test_additional_properties_rejected(self):
        raw = {
            "id": "test-005",
            "input": "Extra field",
            "expected_behavior": "Should fail",
            "category": "normal",
            "expected_pass": True,
            "extra_field": "not allowed",
        }
        import jsonschema
        with pytest.raises(jsonschema.ValidationError):
            validate(raw)

    def test_validate_batch_filters_invalid(self):
        raws = [
            {"id": "good", "input": "x", "expected_behavior": "y", "category": "normal", "expected_pass": True},
            {"id": "bad"},  # Missing required fields
            {"id": "good2", "input": "a", "expected_behavior": "b", "category": "edge", "expected_pass": False},
        ]
        valid = validate_batch(raws)
        assert len(valid) == 2
        assert valid[0]["id"] == "good"
        assert valid[1]["id"] == "good2"


class TestLoadFile:
    """Tests for the _load_file helper."""

    def test_load_valid_json(self):
        data = [{"id": "test", "input": "hello"}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            loaded = _load_file(Path(f.name))
        assert len(loaded) == 1
        assert loaded[0]["id"] == "test"

    def test_load_nonexistent_file(self):
        loaded = _load_file(Path("/nonexistent/path/file.json"))
        assert loaded == []

    def test_load_non_array_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"not": "an array"}, f)
            f.flush()
            loaded = _load_file(Path(f.name))
        assert loaded == []
