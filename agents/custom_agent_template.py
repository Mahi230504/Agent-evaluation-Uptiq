"""
Custom Agent Template
=====================

Copy this file and implement your own agent.
Your agent must extend AbstractAgent and implement the async run_agent() method.

Usage:
    1. Copy this file to agents/my_agent.py
    2. Rename the class and set agent_name
    3. Implement run_agent() with your agent's logic
    4. Register in agents/agent_registry.py:
       from agents.my_agent import MyAgent
       register("my_agent", MyAgent)
    5. Run: python main.py --agent my_agent
"""

from agents.base_agent import AbstractAgent


class CustomAgent(AbstractAgent):
    """
    Your custom agent implementation.

    Replace this with your agent's logic — LangChain, LlamaIndex,
    REST API call, local model inference, etc.
    """

    agent_name: str = "custom_agent"  # Change this to your agent's name

    def __init__(self):
        # Initialize any resources your agent needs
        # e.g., API clients, model instances, configuration
        pass

    async def setup(self) -> None:
        """
        Called once before the test suite runs.
        Use this for expensive initialization (loading models, connecting to APIs).
        """
        # Example:
        # self.client = SomeAPIClient(api_key="...")
        # self.model = load_model("path/to/model")
        pass

    async def run_agent(self, input: str) -> str:
        """
        Process the input and return a response.

        This is the only method you MUST implement.
        The test runner will call this for each test case.

        Args:
            input: The test input string.

        Returns:
            Your agent's response string.
        """
        # ============================================
        # REPLACE THIS WITH YOUR AGENT'S LOGIC
        # ============================================
        #
        # Examples:
        #
        # --- LangChain ---
        # chain = self.prompt | self.llm | StrOutputParser()
        # return await chain.ainvoke({"input": input})
        #
        # --- REST API ---
        # async with httpx.AsyncClient() as client:
        #     resp = await client.post("https://my-api.com/chat", json={"query": input})
        #     return resp.json()["response"]
        #
        # --- Local Model ---
        # return await asyncio.to_thread(self.model.generate, input)
        #
        raise NotImplementedError("Implement run_agent() with your agent's logic")

    async def teardown(self) -> None:
        """
        Called once after the test suite completes.
        Use this for cleanup (closing connections, freeing resources).
        """
        # Example:
        # await self.client.close()
        pass
