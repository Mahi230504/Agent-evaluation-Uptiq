"""
Retry and timeout logic for agent calls.
Wraps agent calls with exponential backoff and hard timeout.
"""

from __future__ import annotations
import asyncio
from typing import Callable, Optional


class AgentTimeoutError(Exception):
    """Raised when an agent call exceeds the configured timeout."""
    pass


class AgentCallError(Exception):
    """Raised when an agent call fails after all retries."""
    pass


async def call_with_retry(
    fn: Callable,
    input: str,
    max_retries: int = 5,
    timeout_seconds: int = 45,
) -> str:
    """
    Call an async function with retry logic and timeout.

    - Retries up to max_retries times on Exception
    - Exponential backoff: 0.5s, 1s, 2s, ...
    - Hard timeout via asyncio.wait_for()

    Args:
        fn: The async callable (e.g., agent.run_agent).
        input: The input string to pass.
        max_retries: Maximum number of retry attempts.
        timeout_seconds: Hard timeout per attempt in seconds.

    Returns:
        The agent's response string.

    Raises:
        AgentTimeoutError: If all attempts timed out.
        AgentCallError: If all attempts failed with other errors.
    """
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            result = await asyncio.wait_for(
                fn(input),
                timeout=timeout_seconds,
            )
            return result

        except asyncio.TimeoutError:
            last_error = AgentTimeoutError(
                f"Agent timed out after {timeout_seconds}s (attempt {attempt + 1}/{max_retries})"
            )

        except Exception as e:
            last_error = e
            
            # If we hit a known rate limit code, enforce a longer backoff (api demands 55s)
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                backoff = 60.0 + (10.0 * attempt) # 60s, 70s, 80s
                print(f"  ⚠️ [Agent Rate Limit] Quota exceeded. Waiting {backoff}s... (Attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(backoff)
                continue

        # Exponential backoff before retry (skip on last attempt)
        if attempt < max_retries - 1:
            backoff = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s, ...
            await asyncio.sleep(backoff)

    # All retries exhausted
    if isinstance(last_error, AgentTimeoutError):
        raise last_error
    raise AgentCallError(
        f"Agent call failed after {max_retries} attempts: {last_error}"
    ) from last_error
