"""
Minimal Flask dashboard serving an HTML view of the latest evaluation report.
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, render_template_string

from src.config import Config
from src.metrics.schemas import RunReport


_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Evaluation Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        h1 {
            font-size: 2rem;
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .meta { color: #94a3b8; margin-bottom: 2rem; }
        .verdict {
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            font-size: 1.5rem;
            font-weight: 700;
            text-align: center;
        }
        .verdict.pass { background: rgba(34, 197, 94, 0.15); border: 1px solid #22c55e; color: #4ade80; }
        .verdict.fail { background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #f87171; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 1.5rem; margin-bottom: 2rem; }
        .card {
            background: #1e293b;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid #334155;
        }
        .card .label { color: #94a3b8; font-size: 0.875rem; text-transform: uppercase; letter-spacing: 0.05em; }
        .card .value { font-size: 2.5rem; font-weight: 700; margin: 0.5rem 0; }
        .card .bar {
            height: 6px;
            background: #334155;
            border-radius: 3px;
            overflow: hidden;
            margin-top: 0.5rem;
        }
        .card .bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.8s ease;
        }
        .safety .value { color: #f97316; }
        .safety .bar-fill { background: linear-gradient(90deg, #ef4444, #f97316); }
        .accuracy .value { color: #60a5fa; }
        .accuracy .bar-fill { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
        .robustness .value { color: #a78bfa; }
        .robustness .bar-fill { background: linear-gradient(90deg, #7c3aed, #a78bfa); }
        .relevance .value { color: #34d399; }
        .relevance .bar-fill { background: linear-gradient(90deg, #059669, #34d399); }
        table {
            width: 100%;
            border-collapse: collapse;
            background: #1e293b;
            border-radius: 12px;
            overflow: hidden;
            margin-bottom: 2rem;
        }
        th {
            background: #0f172a;
            padding: 0.75rem 1rem;
            text-align: left;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #94a3b8;
        }
        td { padding: 0.75rem 1rem; border-top: 1px solid #334155; }
        tr:hover td { background: rgba(99, 102, 241, 0.05); }
        .badge {
            display: inline-block;
            padding: 0.125rem 0.5rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
        }
        .badge.pass { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
        .badge.fail { background: rgba(239, 68, 68, 0.2); color: #f87171; }
        .badge.error { background: rgba(234, 179, 8, 0.2); color: #fbbf24; }
        .score-cell { font-weight: 700; font-variant-numeric: tabular-nums; }
        .latency-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; }
        .latency-card { text-align: center; }
        .latency-card .lat-val { font-size: 1.5rem; font-weight: 700; color: #60a5fa; }
        .latency-card .lat-label { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; }
        h2 { font-size: 1.25rem; margin-bottom: 1rem; color: #cbd5e1; }
        .empty-state { text-align: center; padding: 3rem; color: #64748b; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧪 Agent Evaluation Dashboard</h1>
        {% if report %}
        <p class="meta">Run {{ report.run_id }} · {{ report.agent_name }} · {{ report.timestamp }}</p>

        <div class="verdict {{ 'pass' if report.suite_passed else 'fail' }}">
            {{ '✅ SUITE PASSED' if report.suite_passed else '❌ SUITE FAILED' }}
        </div>

        <div class="grid">
            <div class="card safety">
                <div class="label">Safety</div>
                <div class="value">{{ '%.1f' % report.safety_score }}</div>
                <div class="bar"><div class="bar-fill" style="width: {{ report.safety_score * 10 }}%"></div></div>
            </div>
            <div class="card accuracy">
                <div class="label">Accuracy</div>
                <div class="value">{{ '%.1f' % report.accuracy_score }}</div>
                <div class="bar"><div class="bar-fill" style="width: {{ report.accuracy_score * 10 }}%"></div></div>
            </div>
            <div class="card robustness">
                <div class="label">Robustness</div>
                <div class="value">{{ '%.1f' % report.robustness_score }}</div>
                <div class="bar"><div class="bar-fill" style="width: {{ report.robustness_score * 10 }}%"></div></div>
            </div>
            <div class="card relevance">
                <div class="label">Relevance</div>
                <div class="value">{{ '%.1f' % report.relevance_score }}</div>
                <div class="bar"><div class="bar-fill" style="width: {{ report.relevance_score * 10 }}%"></div></div>
            </div>
        </div>

        <h2>⏱️ Latency</h2>
        <div class="latency-grid" style="margin-bottom: 2rem;">
            <div class="latency-card">
                <div class="lat-val">{{ '%.0f' % report.latency.mean_ms }}ms</div>
                <div class="lat-label">Mean</div>
            </div>
            <div class="latency-card">
                <div class="lat-val">{{ '%.0f' % report.latency.median_ms }}ms</div>
                <div class="lat-label">Median</div>
            </div>
            <div class="latency-card">
                <div class="lat-val">{{ '%.0f' % report.latency.min_ms }}ms</div>
                <div class="lat-label">Min</div>
            </div>
            <div class="latency-card">
                <div class="lat-val">{{ '%.0f' % report.latency.max_ms }}ms</div>
                <div class="lat-label">Max</div>
            </div>
        </div>

        <h2>📋 Test Results ({{ report.total_tests }} tests)</h2>
        <table>
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Category</th>
                    <th>Score</th>
                    <th>Method</th>
                    <th>Status</th>
                    <th>Rationale</th>
                </tr>
            </thead>
            <tbody>
                {% for r in report.results %}
                <tr>
                    <td><code>{{ r.test_case.id }}</code></td>
                    <td>{{ r.test_case.category }}</td>
                    <td class="score-cell">{{ '%.1f' % r.eval_result.score }}</td>
                    <td>{{ r.eval_result.method }}</td>
                    <td>
                        {% if r.error %}
                        <span class="badge error">ERROR</span>
                        {% elif r.eval_result.passed %}
                        <span class="badge pass">PASS</span>
                        {% else %}
                        <span class="badge fail">FAIL</span>
                        {% endif %}
                    </td>
                    <td>{{ r.eval_result.rationale[:100] }}{{ '...' if r.eval_result.rationale|length > 100 else '' }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% else %}
        <div class="empty-state">
            <h2>No reports found</h2>
            <p>Run an evaluation first: <code>python main.py --agent simple_chatbot</code></p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""


def get_latest_report() -> RunReport | None:
    """
    Read the latest report from the reports directory.

    Returns:
        RunReport if found, None otherwise.
    """
    reports_dir = Config.REPORTS_DIR
    if not reports_dir.exists():
        return None

    # Find the latest run directory
    run_dirs = sorted(
        [d for d in reports_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        reverse=True,
    )

    if not run_dirs:
        return None

    summary_path = run_dirs[0] / "summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path, "r") as f:
        data = json.load(f)

    return RunReport(**data)


def create_app() -> Flask:
    """Create and configure the Flask dashboard app."""
    app = Flask(__name__)

    @app.route("/")
    def index():
        report = get_latest_report()
        return render_template_string(_DASHBOARD_HTML, report=report)

    @app.route("/report/<run_id>")
    def report_by_id(run_id: str):
        reports_dir = Config.REPORTS_DIR
        # Search for matching run directory
        for d in reports_dir.iterdir():
            if d.is_dir() and run_id in d.name:
                summary_path = d / "summary.json"
                if summary_path.exists():
                    with open(summary_path, "r") as f:
                        data = json.load(f)
                    report = RunReport(**data)
                    return render_template_string(_DASHBOARD_HTML, report=report)

        return render_template_string(_DASHBOARD_HTML, report=None)

    return app
