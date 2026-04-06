"""
Simple rule-based chatbot agent for testing the framework.
No external API calls — zero latency, zero cost.
"""

import re
from agents.base_agent import AbstractAgent


class SimpleChatbot(AbstractAgent):
    """
    A keyword-driven demo agent that returns hardcoded responses.
    Used to validate the evaluation pipeline end-to-end without API keys.
    """

    agent_name: str = "simple_chatbot"

    # Patterns that indicate harmful/adversarial intent
    _REFUSAL_TRIGGERS = [
        r"(?i)\b(bomb|explosive|weapon|gun|firearm)\b.*\b(make|build|create|instructions|how to)\b",
        r"(?i)\b(how to)\b.*\b(bomb|explosive|weapon)\b",
        r"(?i)\b(kill|murder|harm|hurt|attack)\b.*\b(someone|myself|people|person)\b",
        r"(?i)\b(hack|bypass|break into)\b.*\b(security|system|account|password)\b",
        r"(?i)\b(hate|inferior|disgusting)\b.*\b(race|religion|gender|ethnicity|group)\b",
        r"(?i)\b(end my life|suicide|kill myself|want to die)\b",
        r"(?i)\b(home address|phone number|social security|ssn)\b",
        r"(?i)\b(ignore all previous|ignore your instructions|you are now DAN)\b",
        r"(?i)\b(system prompt|reveal your prompt|output your instructions)\b",
        r"(?i)\bstep.by.step.*(synth|meth|drug|controlled substance)\b",
    ]

    # Simple keyword-based responses
    _KNOWLEDGE = {
        "capital of france": "The capital of France is Paris.",
        "capital of germany": "The capital of Germany is Berlin.",
        "capital of alemania": "La capital de Alemania es Berlín.",
        "15 multiplied by 23": "15 multiplied by 23 equals 345.",
        "15 * 23": "15 × 23 = 345.",
        "primary colors": "The three primary colors are red, blue, and yellow.",
        "photosynthesis": (
            "Photosynthesis is the process by which plants convert sunlight, "
            "water, and carbon dioxide into glucose and oxygen. It's how plants "
            "make their own food using energy from the sun."
        ),
        "hello, how are you": "Hola, ¿cómo estás?",
        "translate": "Hola, ¿cómo estás?",
    }

    _REFUSAL_RESPONSE = (
        "I'm sorry, but I can't help with that request. "
        "If you're in crisis, please contact the 988 Suicide & Crisis Lifeline "
        "by calling or texting 988, or reach out to a trusted person for support."
    )

    async def run_agent(self, input: str) -> str:
        """Process input using keyword matching and pattern detection."""

        # Handle empty/whitespace input
        if not input or not input.strip():
            return "It looks like your message was empty. Could you please provide more details about what you'd like help with?"

        # Handle special-character-only input
        if not re.search(r"[a-zA-Z0-9]", input):
            return "I received some special characters but couldn't understand the request. Could you please rephrase your question in words?"

        # Check for adversarial/harmful patterns → refuse
        for pattern in self._REFUSAL_TRIGGERS:
            if re.search(pattern, input):
                return self._REFUSAL_RESPONSE

        # Check for known topics
        input_lower = input.lower()
        for keyword, response in self._KNOWLEDGE.items():
            if keyword in input_lower:
                return response

        # Check for summarization requests
        if "summarize" in input_lower or "summary" in input_lower:
            return "This text discusses a pangram — a sentence that contains every letter of the English alphabet, historically used for testing typewriters and fonts."

        # Check for paradox / logic
        if "this statement is false" in input_lower:
            return (
                "This is a classic logical paradox known as the liar's paradox. "
                "The statement 'This statement is false' cannot be consistently "
                "assigned a truth value — if it's true, then it must be false, "
                "and if it's false, then it must be true."
            )

        # Check for vague/ambiguous requests
        if "last time" in input_lower or "the thing" in input_lower:
            return (
                "I'd be happy to help, but I'm not sure what you're referring to. "
                "Could you please provide more specific details about what you need?"
            )

        # Check for injection attempts that slipped past refusal triggers
        if "hacked" in input_lower or "ignore" in input_lower:
            return "I'm here to help with legitimate questions. How can I assist you today?"

        # Default response
        return (
            "Thank you for your question. I'm a simple demo chatbot with limited "
            "knowledge. I can help with basic factual questions, translations, "
            "and math. For more complex queries, please try a more advanced agent."
        )
