"""
Central registry for all agent implementations.
Maps string names to agent classes for CLI lookup.
"""

from agents.base_agent import AbstractAgent


_registry: dict[str, type[AbstractAgent]] = {}


def register(name: str, agent_class: type[AbstractAgent]) -> None:
    """
    Register an agent class under a given name.

    Args:
        name: The CLI-friendly name for the agent.
        agent_class: The agent class (must extend AbstractAgent).

    Raises:
        TypeError: If agent_class doesn't extend AbstractAgent.
        ValueError: If name is already registered.
    """
    if not (isinstance(agent_class, type) and issubclass(agent_class, AbstractAgent)):
        raise TypeError(f"'{agent_class}' must be a subclass of AbstractAgent")
    if name in _registry:
        raise ValueError(f"Agent '{name}' is already registered")
    _registry[name] = agent_class


def get(name: str) -> AbstractAgent:
    """
    Instantiate and return an agent by its registered name.

    Args:
        name: The registered name of the agent.

    Returns:
        An instance of the requested agent.

    Raises:
        KeyError: If no agent is registered under that name.
    """
    if name not in _registry:
        available = ", ".join(sorted(_registry.keys())) or "(none)"
        raise KeyError(f"Unknown agent '{name}'. Available agents: {available}")
    return _registry[name]()


def list_agents() -> list[str]:
    """Return a sorted list of all registered agent names."""
    return sorted(_registry.keys())


def _auto_register() -> None:
    """Auto-register built-in agents."""
    from agents.simple_chatbot import SimpleChatbot
    from agents.openai_agent import OpenAIAgent

    register("simple_chatbot", SimpleChatbot)
    register("openai_agent", OpenAIAgent)


# Auto-register on import
_auto_register()
