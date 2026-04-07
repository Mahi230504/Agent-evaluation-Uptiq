# Agent Evaluation Framework

A modular, agent-agnostic testing and evaluation framework for AI systems. Evaluates agents across **4 dimensions** — safety, accuracy, robustness, and relevance — using a cascade evaluation engine (rule-based → LLM-as-judge) with Google Gemini.

## ✨ Features

- **Agent-Agnostic**: Test any AI agent through a simple async interface
- **Cascade Evaluation**: Rule-based checks first, LLM judge only when needed (cost-optimized)
- **4-Dimension Scoring**: Safety (×2.0), robustness (×1.5), accuracy (×1.0), relevance (×0.75)
- **Async Execution**: Concurrent test execution with semaphore-bounded rate limiting
- **Retry & Timeout**: Exponential backoff with configurable timeout per test
- **Rich Reporting**: Markdown reports, JSONL logs, and a Flask web dashboard
- **21 Built-in Test Cases**: Normal, edge, adversarial, and safety categories

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Run with the demo agent (no API keys needed for dry-run)
```bash
# Dry run — validate test cases only
python main.py --agent simple_chatbot --dry-run

# Full run (requires GEMINI_API_KEY for LLM judge)
python main.py --agent simple_chatbot
```

### 4. View results
```bash
# Check the generated Markdown report
cat reports/run_*/report.md

# Or launch the Streamlit dashboard (recommended)
streamlit run streamlit_app.py

# Or launch the minimal Flask dashboard
python main.py --dashboard
```

## 📋 CLI Usage

```bash
python main.py --agent <agent_name>              # Run all tests
python main.py --agent <agent_name> --category safety  # Run specific category
python main.py --agent <agent_name> --dry-run     # Validate only
python main.py --dashboard                        # Launch web dashboard
python main.py --list-agents                      # Show available agents
```

## 🏗️ Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full architecture documentation.

**Evaluation Cascade:**
- **Safety tests** → Rule check → LLM escalation if ambiguous
- **Adversarial tests** → Injection/jailbreak rules → LLM for ambiguous
- **Normal/Edge tests** → Fast Gemini model → Slow model if score is uncertain (3-7)

**Pass/Fail Criteria:**
- Overall score ≥ 7.0 **AND** safety score ≥ 8.5

## 🔌 Adding Your Own Agent

1. Copy `agents/custom_agent_template.py`
2. Implement the `run_agent()` method
3. Register in `agents/agent_registry.py`
4. Run: `python main.py --agent your_agent`

```python
from agents.base_agent import AbstractAgent

class MyAgent(AbstractAgent):
    agent_name = "my_agent"

    async def run_agent(self, input: str) -> str:
        # Your logic here
        return response
```

## 📊 Sample Report Structure

```
reports/
└── run_20241201_143022/
    ├── report.md       ← Human-readable Markdown report
    ├── summary.json    ← Machine-readable RunReport
    └── log.jsonl       ← One JSON line per test (observability)
```

## 🧪 Running Tests

```bash
python -m pytest tests/ -v
```

## 📁 Project Structure

```
├── main.py                  # CLI entry point
├── agents/                  # Agent implementations
│   ├── base_agent.py        # Abstract base (async)
│   ├── agent_registry.py    # Agent lookup registry
│   ├── simple_chatbot.py    # Demo agent (no API)
│   └── openai_agent.py      # OpenAI wrapper
├── src/
│   ├── config.py            # Configuration from .env
│   ├── loader/              # Test case loading & validation
│   ├── runner/              # Async execution engine
│   ├── evaluation/          # Cascade evaluator + rubrics
│   ├── metrics/             # Scoring & thresholds
│   └── reporting/           # Markdown, JSONL, dashboard
├── data/
│   ├── test_cases/          # JSON test case files
│   └── schemas/             # JSON schema for validation
└── tests/                   # Unit tests
```