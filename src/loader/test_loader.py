"""
Reads and assembles test cases from /data/test_cases/.
Loads JSON files per category, validates, and converts to TestCase objects.
"""

import json
from pathlib import Path

from src.config import Config
from src.metrics.schemas import TestCase
from src.loader.test_validator import validate_batch


# Map category names to filenames
_CATEGORY_FILES = {
    "normal": "normal.json",
    "edge": "edge_cases.json",
    "adversarial": "adversarial.json",
    "safety": "safety.json",
}


def _load_file(path: Path) -> list[dict]:
    """Load and parse a single JSON test case file."""
    if not path.exists():
        print(f"  ⚠ Test case file not found: {path}")
        return []
    with open(path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"  ⚠ Expected a JSON array in {path}, got {type(data).__name__}")
        return []
    return data


def load_by_category(category: str) -> list[TestCase]:
    """
    Load test cases for a specific category.

    Args:
        category: One of 'normal', 'edge', 'adversarial', 'safety'.

    Returns:
        List of validated TestCase objects.

    Raises:
        ValueError: If category is not recognized.
    """
    if category not in _CATEGORY_FILES:
        raise ValueError(
            f"Unknown category '{category}'. "
            f"Valid categories: {list(_CATEGORY_FILES.keys())}"
        )

    filename = _CATEGORY_FILES[category]
    path = Config.TEST_CASES_DIR / filename
    raw_cases = _load_file(path)
    valid_cases = validate_batch(raw_cases)

    return [TestCase(**case) for case in valid_cases]


def load_all() -> list[TestCase]:
    """
    Load all test cases from all category files.

    Returns:
        Combined list of all validated TestCase objects.
    """
    all_cases: list[TestCase] = []
    for category, filename in _CATEGORY_FILES.items():
        path = Config.TEST_CASES_DIR / filename
        raw_cases = _load_file(path)
        valid_cases = validate_batch(raw_cases)
        cases = [TestCase(**case) for case in valid_cases]
        print(f"  ✓ Loaded {len(cases)} test cases from {filename}")
        all_cases.extend(cases)

    print(f"  ✓ Total: {len(all_cases)} test cases loaded")
    return all_cases
