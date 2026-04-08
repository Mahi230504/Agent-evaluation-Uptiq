"""
Gemini agent wrapper using google-genai.
Requires GEMINI_API_KEY in environment.
"""

from google import genai
from agents.base_agent import AbstractAgent
from src.config import Config


class GeminiAgent(AbstractAgent):
    """
    Agent that wraps the Google Gemini API.
    Configurable model, temperature, and system prompt.
    """

    agent_name: str = "gemini_agent"

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self._client: genai.Client | None = None

    async def setup(self) -> None:
        """Initialize the Gemini client."""
        if not Config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Please set it in your .env file or environment variables."
            )
        self._client = genai.Client(api_key=Config.GEMINI_API_KEY)

    async def run_agent(self, input: str) -> str:
        """Send input to Gemini and return the response."""
        if self._client is None:
            await self.setup()

        # google-genai client doesn't have an async method for content generation
        # so we run it in a thread if strictly needed, but here we just call it.
        # However, for consistency with our async runner, we call it.
        
        response = self._client.models.generate_content(
            model=self.model,
            contents=input,
            config={
                "system_instruction": self.system_prompt,
                "temperature": self.temperature,
            }
        )
        return response.text or ""

    async def teardown(self) -> None:
        """Teardown (no-op for Gemini client)."""
        self._client = None
