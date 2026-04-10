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
        model: str = "gemini-3.1-flash-lite-preview",
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.model = model
        self.temperature = temperature
        self.system_prompt = system_prompt
        self._client: genai.Client | None = None

    async def setup(self) -> None:
        """Initialize the Gemini client."""
        key = Config.get_gemini_key()
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. "
                "Please set it in your .env file or environment variables."
            )
        self._client = genai.Client(api_key=key)

    async def run_agent(self, input: str) -> str:
        """Send input to Gemini and return the response."""
        if self._client is None:
            await self.setup()

        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=input,
                config={
                    "system_instruction": self.system_prompt,
                    "temperature": self.temperature,
                }
            )
            return response.text or ""
        except Exception as e:
            # Re-raise with more context
            error_msg = str(e)
            if "404" in error_msg:
                raise RuntimeError(f"Model '{self.model}' not found. Please check if this model is enabled for your API key or try gemini-1.5-flash-001 or gemini-1.5-pro-latest. Full error: {e}")
            elif "429" in error_msg:
                raise RuntimeError(f"Rate limit exceeded (429) for '{self.model}'. Please check your quota at ai.google.dev. Full error: {e}")
            raise RuntimeError(f"Gemini API error ({self.model}): {e}")

    @classmethod
    def list_available_models(cls) -> list[str]:
        """Utility to list all models available to the current key."""
        try:
            key = Config.get_gemini_key()
            if not key: return ["Key not found"]
            client = genai.Client(api_key=key)
            return [m.name.replace("models/", "") for m in client.models.list() if "generateContent" in m.supported_generation_methods]
        except Exception as e:
            return [f"Error listing models: {e}"]

    async def teardown(self) -> None:
        """Teardown (no-op for Gemini client)."""
        self._client = None
