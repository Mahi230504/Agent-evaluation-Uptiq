"""
Agent Evaluation Framework — Entry Point
CLI interface for running evaluations against AI agents.
"""

import argparse
import asyncio
import sys
import os
from datetime import datetime
from pathlib import Path

# Fix module resolution: add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure src is discoverable
src_path = os.path.join(str(PROJECT_ROOT), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.config import Config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Agent Evaluation Framework — Evaluate AI agents for safety, accuracy, robustness, and relevance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --agent simple_chatbot                          # Run all tests
  python main.py --agent simple_chatbot --category safety        # Safety tests only
  python main.py --agent simple_chatbot --dry-run                # Validate only
  python main.py --agent simple_chatbot --types rag tool         # RAG + Tool metrics
  python main.py --list-agents                                   # List registered agents
  streamlit run streamlit_app.py                                 # Launch UI
        """,
    )

    parser.add_argument("--agent", type=str, help="Agent name to evaluate")
    parser.add_argument(
        "--category",
        type=str,
        choices=["normal", "edge", "adversarial", "safety"],
        help="Run only a specific test category",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["simple", "rag", "tool"],
        default=["simple"],
        help="Agent type(s) — determines which metrics are activated (default: simple)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for reports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate test cases without running evaluation",
    )
    parser.add_argument(
        "--list-agents",
        action="store_true",
        help="List all registered agents and exit",
    )

    return parser.parse_args()


async def run_evaluation(args: argparse.Namespace) -> None:
    """Run the full evaluation pipeline."""
    from agents import agent_registry
    from src.loader import test_loader
    from src.runner import test_runner
    from src.metrics.aggregator import build_run_report
    from src.reporting import markdown_reporter, logger
    from src.evaluation.metric_selector import select_metrics

    # Validate config
    warnings = Config.validate()
    for w in warnings:
        print(f"  ⚠ {w}")

    # Instantiate the agent
    print(f"\n🤖 Loading agent: {args.agent}")
    agent = agent_registry.get(args.agent)
    await agent.setup()

    # Load test cases
    print("\n📂 Loading test cases...")
    if args.category:
        cases = test_loader.load_by_category(args.category)
    else:
        cases = test_loader.load_all()

    if not cases:
        print("  ❌ No test cases found!")
        await agent.teardown()
        return

    # Dry run — validate and exit
    if args.dry_run:
        print(f"\n✅ Dry run complete: {len(cases)} test cases loaded and validated.")
        cats: dict[str, int] = {}
        for c in cases:
            cats[c.category] = cats.get(c.category, 0) + 1
        for cat, count in sorted(cats.items()):
            print(f"  • {cat}: {count}")
        await agent.teardown()
        return

    # Select metrics based on agent types
    agent_types = args.types
    metrics = select_metrics(agent_types)
    print(f"\n📐 Agent types: {', '.join(agent_types)}")
    print(f"📏 Active metrics: {', '.join(m.name for m in metrics)}")

    # Prepare output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Config.REPORTS_DIR / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "log.jsonl"

    # Run the test suite
    print(f"\n🚀 Running {len(cases)} test cases against '{agent.agent_name}'...")
    print(f"   Max concurrent: {Config.MAX_CONCURRENT}")
    print(f"   Timeout per test: {Config.AGENT_TIMEOUT_SECONDS}s")

    results = await test_runner.run_suite(agent, cases, metrics=metrics, log_path=log_path)

    # Build the report
    print("\n📊 Computing scores...")
    report = build_run_report(results, agent.agent_name, agent_types=agent_types)

    # Log summary
    logger.log_run_summary(report, log_path)

    # Generate markdown report
    report_path = markdown_reporter.generate(report, output_dir)

    # Save machine-readable summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        f.write(report.model_dump_json(indent=2))

    # Print summary
    print(f"\n{'=' * 60}")
    emoji = "✅" if report.suite_passed else "❌"
    print(f" {emoji} SUITE {'PASSED' if report.suite_passed else 'FAILED'}")
    print(f"{'=' * 60}")
    print(f"  Overall:    {report.overall_score:.1f}/10")
    print(f"  Safety:     {report.safety_score:.1f}/10")
    print(f"  Accuracy:   {report.accuracy_score:.1f}/10")
    print(f"  Robustness: {report.robustness_score:.1f}/10")
    print(f"  Relevance:  {report.relevance_score:.1f}/10")
    print(f"{'=' * 60}")
    print(f"  Tests: {report.passed_tests} passed, {report.failed_tests} failed, {report.error_tests} errors")
    print(f"  Latency: {report.latency.mean_ms:.0f}ms mean, {report.latency.median_ms:.0f}ms median")
    print(f"{'=' * 60}")
    print(f"\n📄 Report:  {report_path}")
    print(f"📊 Summary: {summary_path}")
    print(f"📝 Log:     {log_path}")

    await agent.teardown()


def main():
    """Main entry point."""
    args = parse_args()

    if args.list_agents:
        from agents import agent_registry
        agents = agent_registry.list_agents()
        print("Registered agents:")
        for name in agents:
            print(f"  • {name}")
        return

    if not args.agent:
        print("Error: --agent is required. Use --list-agents to see options.")
        print("Tip: Use 'streamlit run streamlit_app.py' to launch the interactive UI.")
        sys.exit(1)

    asyncio.run(run_evaluation(args))


if __name__ == "__main__":
    main()
