"""
Abstract base class that all agent implementations must extend.
"""

from abc import ABC, abstractmethod


class AbstractAgent(ABC):
    """
    Base class for all agents under evaluation.

    Every agent must implement `run_agent()` as an async method
    that takes a string input and returns a string response.
    """

    agent_name: str = "unnamed_agent"

    @abstractmethod
    async def run_agent(self, input: str) -> str:
        """
        Process the input and return a response.

        Args:
            input: The user/test input string.

        Returns:
            The agent's response string.
        """
        ...

    async def setup(self) -> None:
        """Optional setup hook called before the test suite runs."""
        pass

    async def teardown(self) -> None:
        """Optional teardown hook called after the test suite completes."""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.agent_name}'>"
