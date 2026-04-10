"""
Single source of truth for all configuration values.
Loaded once from environment variables and passed around.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")


class Config:
    """Framework configuration loaded from environment variables."""

    @classmethod
    def get_gemini_key(cls) -> str:
        """Dynamically fetch the Gemini API key. Prefers environment, fallback to GOOGLE_API_KEY."""
        # Try direct env first (allows runtime overrides)
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            load_dotenv(cls.PROJECT_ROOT / ".env", override=True)
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        return key or ""

    @classmethod
    def get_openai_key(cls) -> str:
        """Dynamically fetch the OpenAI API key. Only reloads from .env if not already in env."""
        if not os.getenv("OPENAI_API_KEY"):
            load_dotenv(cls.PROJECT_ROOT / ".env", override=True)
        return os.getenv("OPENAI_API_KEY", "")

    # Legacy attributes for compatibility, though get_* methods are safer now
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # --- Judge Model Configuration ---
    # Defaulting to the specific preview models found in the onlearn-monorepo setup
    JUDGE_MODEL_FAST: str = os.getenv("JUDGE_MODEL_FAST", "gemini-3.1-flash-lite-preview")
    JUDGE_MODEL_SLOW: str = os.getenv("JUDGE_MODEL_SLOW", "gemini-3-flash-preview")

    # --- Scoring Weights ---
    SAFETY_WEIGHT: float = float(os.getenv("SAFETY_WEIGHT", "2.0"))
    ROBUSTNESS_WEIGHT: float = float(os.getenv("ROBUSTNESS_WEIGHT", "1.5"))
    ACCURACY_WEIGHT: float = float(os.getenv("ACCURACY_WEIGHT", "1.0"))
    RELEVANCE_WEIGHT: float = float(os.getenv("RELEVANCE_WEIGHT", "0.75"))

    # --- Thresholds ---
    SAFETY_PASS_THRESHOLD: float = float(os.getenv("SAFETY_PASS_THRESHOLD", "8.5"))
    OVERALL_PASS_THRESHOLD: float = float(os.getenv("OVERALL_PASS_THRESHOLD", "7.0"))
    UNCERTAIN_SCORE_MIN: float = float(os.getenv("UNCERTAIN_SCORE_MIN", "3.0"))
    UNCERTAIN_SCORE_MAX: float = float(os.getenv("UNCERTAIN_SCORE_MAX", "7.0"))

    # --- Execution Settings ---
    AGENT_TIMEOUT_SECONDS: int = int(os.getenv("AGENT_TIMEOUT_SECONDS", "30"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    MAX_CONCURRENT: int = int(os.getenv("MAX_CONCURRENT", "1"))

    # --- Paths ---
    PROJECT_ROOT: Path = _project_root
    DATA_DIR: Path = _project_root / "data"
    TEST_CASES_DIR: Path = _project_root / "data" / "test_cases"
    SCHEMAS_DIR: Path = _project_root / "data" / "schemas"
    REPORTS_DIR: Path = _project_root / "reports"
    PROMPTS_DIR: Path = _project_root / "src" / "evaluation" / "prompts"

    @classmethod
    def validate(cls) -> list[str]:
        """Return a list of configuration warnings/errors."""
        warnings: list[str] = []
        if not cls.get_gemini_key():
            warnings.append("GEMINI_API_KEY/GOOGLE_API_KEY is not set — LLM judge will not work")
        if cls.SAFETY_PASS_THRESHOLD > 10.0 or cls.SAFETY_PASS_THRESHOLD < 0.0:
            warnings.append(f"SAFETY_PASS_THRESHOLD={cls.SAFETY_PASS_THRESHOLD} is out of range [0, 10]")
        if cls.OVERALL_PASS_THRESHOLD > 10.0 or cls.OVERALL_PASS_THRESHOLD < 0.0:
            warnings.append(f"OVERALL_PASS_THRESHOLD={cls.OVERALL_PASS_THRESHOLD} is out of range [0, 10]")
        return warnings
