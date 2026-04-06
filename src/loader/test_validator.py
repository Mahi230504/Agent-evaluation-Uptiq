"""
Validates each raw test case dict against the JSON schema
before it becomes a TestCase Pydantic model.
"""

import json
from pathlib import Path
from typing import Optional

import jsonschema

from src.config import Config

# Load the JSON schema once at module level
_schema_path = Config.SCHEMAS_DIR / "test_case_schema.json"
_schema: Optional[dict] = None


def _get_schema() -> dict:
    """Lazy-load the JSON schema."""
    global _schema
    if _schema is None:
        with open(_schema_path, "r") as f:
            _schema = json.load(f)
    return _schema


def validate(raw: dict) -> bool:
    """
    Validate a single raw test case dict against the JSON schema.

    Args:
        raw: A dictionary representing a test case.

    Returns:
        True if valid.

    Raises:
        jsonschema.ValidationError: If the test case is invalid.
    """
    jsonschema.validate(instance=raw, schema=_get_schema())
    return True


def validate_batch(raws: list[dict]) -> list[dict]:
    """
    Validate a batch of raw test case dicts, filtering out invalid ones.

    Args:
        raws: List of dictionaries representing test cases.

    Returns:
        List of valid test case dicts. Invalid ones are logged and skipped.
    """
    valid = []
    for i, raw in enumerate(raws):
        try:
            validate(raw)
            valid.append(raw)
        except jsonschema.ValidationError as e:
            test_id = raw.get("id", f"<index {i}>")
            print(f"  ⚠ Skipping invalid test case '{test_id}': {e.message}")
    return valid
