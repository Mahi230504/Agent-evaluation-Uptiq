"""
OpenAI Chat Completions agent wrapper.
Requires OPENAI_API_KEY in environment.
"""

from openai import AsyncOpenAI

from agents.base_agent import AbstractAgent
from src.config import Config


class OpenAIAgent(AbstractAgent):
    """
    Agent that wraps the OpenAI Chat Completions API.
    Configurable model, temperature, and system prompt.
    """

    agent_name: str = "openai_agent"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self._client: AsyncOpenAI | None = None

    async def setup(self) -> None:
        """Initialize the OpenAI async client."""
        if not Config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Please set it in your .env file or environment variables."
            )
        self._client = AsyncOpenAI(api_key=Config.OPENAI_API_KEY)

    async def run_agent(self, input: str) -> str:
        """Send input to OpenAI and return the response."""
        if self._client is None:
            await self.setup()

        response = await self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": input},
            ],
        )
        return response.choices[0].message.content or ""

    async def teardown(self) -> None:
        """Close the OpenAI client."""
        if self._client:
            await self._client.close()
            self._client = None
