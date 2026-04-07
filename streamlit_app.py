"""
Streamlit UI for the Agent Evaluation Framework.
Run with: streamlit run streamlit_app.py
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import Config
from src.metrics.schemas import RunReport

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agent Evaluation Framework",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
      background: #0f172a;
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

  /* Main background */
  .stApp { background: #0f172a; }
  .main .block-container { padding-top: 2rem; }

  /* Cards */
  .metric-card {
      background: #1e293b;
      border-radius: 16px;
      padding: 1.5rem;
      border: 1px solid #334155;
      text-align: center;
  }
  .metric-label {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #94a3b8;
      margin-bottom: 0.5rem;
  }
  .metric-value {
      font-size: 2.75rem;
      font-weight: 700;
      line-height: 1;
  }

  /* Verdict banner */
  .verdict-pass {
      background: rgba(34,197,94,0.12);
      border: 1.5px solid #22c55e;
      border-radius: 14px;
      padding: 1.25rem 2rem;
      text-align: center;
      font-size: 1.4rem;
      font-weight: 700;
      color: #4ade80;
      margin-bottom: 1.5rem;
  }
  .verdict-fail {
      background: rgba(239,68,68,0.12);
      border: 1.5px solid #ef4444;
      border-radius: 14px;
      padding: 1.25rem 2rem;
      text-align: center;
      font-size: 1.4rem;
      font-weight: 700;
      color: #f87171;
      margin-bottom: 1.5rem;
  }

  /* Table styles */
  .stDataFrame { border-radius: 12px; overflow: hidden; }

  /* Badge colours for categories shown in table */
  .badge-normal { color: #60a5fa; }
  .badge-edge { color: #a78bfa; }
  .badge-adversarial { color: #fb923c; }
  .badge-safety { color: #34d399; }

  /* Headings */
  h1, h2, h3 { color: #f1f5f9 !important; }
  .stMarkdown p { color: #cbd5e1; }

  /* Input labels */
  label { color: #94a3b8 !important; }

  /* Buttons */
  .stButton > button {
      background: linear-gradient(135deg, #6366f1, #8b5cf6);
      color: #fff;
      border: none;
      border-radius: 10px;
      padding: 0.6rem 1.6rem;
      font-weight: 600;
      font-size: 0.9rem;
      transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_report(path: Path) -> RunReport | None:
    """Load a RunReport from a summary.json file."""
    try:
        with open(path) as f:
            return RunReport(**json.load(f))
    except Exception:
        return None


def list_reports() -> list[tuple[str, Path]]:
    """Return list of (run_id, summary_path) sorted newest-first."""
    reports_dir = Config.REPORTS_DIR
    if not reports_dir.exists():
        return []
    runs = sorted(
        [d for d in reports_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        reverse=True,
    )
    result = []
    for run_dir in runs:
        s = run_dir / "summary.json"
        if s.exists():
            result.append((run_dir.name, s))
    return result


def score_color(score: float) -> str:
    if score >= 8.0:
        return "#4ade80"
    if score >= 6.0:
        return "#facc15"
    return "#f87171"


def gauge_fig(value: float, title: str, color: str) -> go.Figure:
    """Create a compact Plotly gauge."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 36, "color": color}, "suffix": ""},
        title={"text": title, "font": {"size": 13, "color": "#94a3b8"}},
        gauge={
            "axis": {"range": [0, 10], "tickfont": {"color": "#cbd5e1"}, "tickcolor": "#334155"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1e293b",
            "bordercolor": "#334155",
            "steps": [
                {"range": [0, 5], "color": "rgba(239,68,68,0.15)"},
                {"range": [5, 7.5], "color": "rgba(234,179,8,0.15)"},
                {"range": [7.5, 10], "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.75,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        height=220,
        margin=dict(t=40, b=10, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
    )
    return fig


def bar_chart(report: RunReport) -> go.Figure:
    """Stacked bar: pass / fail / error per category."""
    cats = ["normal", "edge", "adversarial", "safety"]
    passed_counts = {c: 0 for c in cats}
    failed_counts = {c: 0 for c in cats}
    error_counts  = {c: 0 for c in cats}

    for r in report.results:
        cat = r.test_case.category
        if r.error:
            error_counts[cat] += 1
        elif r.eval_result.passed:
            passed_counts[cat] += 1
        else:
            failed_counts[cat] += 1

    fig = go.Figure(data=[
        go.Bar(name="Pass",  x=cats, y=[passed_counts[c] for c in cats], marker_color="#4ade80"),
        go.Bar(name="Fail",  x=cats, y=[failed_counts[c] for c in cats], marker_color="#f87171"),
        go.Bar(name="Error", x=cats, y=[error_counts[c]  for c in cats], marker_color="#fbbf24"),
    ])
    fig.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=20, b=20, l=10, r=10),
        height=260,
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155"),
    )
    return fig


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧪 Agent Evaluator")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🚀 Run Evaluation", "📊 Results", "📋 Test Details", "📁 Report History"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    # Report selector (shown on all pages except Run)
    all_reports = list_reports()
    selected_report_path: Path | None = None

    if all_reports and page != "🚀 Run Evaluation":
        run_labels = [r[0] for r in all_reports]
        sel = st.selectbox("Report", run_labels, index=0)
        selected_report_path = next(p for label, p in all_reports if label == sel)


# ── Page: Run Evaluation ──────────────────────────────────────────────────────
if page == "🚀 Run Evaluation":
    st.markdown("# 🚀 Run Evaluation")
    st.markdown("Configure and launch an evaluation run against your agent.")

    col1, col2 = st.columns(2)
    with col1:
        # Agent selector
        agents = ["simple_chatbot", "openai_agent"]
        # Try to auto-detect from registry
        try:
            from agents.agent_registry import list_agents as _list_agents
            agents = _list_agents()
        except Exception:
            pass
        agent_name = st.selectbox("Agent", agents)

    with col2:
        category = st.selectbox(
            "Test Category",
            ["all", "normal", "edge", "adversarial", "safety"],
        )

    col3, col4 = st.columns(2)
    with col3:
        dry_run = st.checkbox("Dry run (validate only)", value=False)
    with col4:
        gemini_key = st.text_input(
            "Gemini API Key (overrides .env)",
            type="password",
            placeholder="Leave blank to use .env value",
        )

    st.markdown("---")
    run_btn = st.button("▶ Run Evaluation", use_container_width=True)

    if run_btn:
        cmd = [sys.executable, "main.py", "--agent", agent_name]
        if category != "all":
            cmd += ["--category", category]
        if dry_run:
            cmd.append("--dry-run")

        env_override = {}
        if gemini_key:
            env_override["GEMINI_API_KEY"] = gemini_key

        st.markdown("### 🔄 Evaluation Output")
        output_box = st.empty()
        full_output: list[str] = []

        with st.spinner("Running evaluation…"):
            import os
            run_env = {**os.environ, **env_override}
            process = subprocess.Popen(
                cmd,
                cwd=str(_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=run_env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                full_output.append(line)
                output_box.code("".join(full_output), language="text")

            process.wait()

        if process.returncode == 0:
            st.success("✅ Evaluation complete! Go to **📊 Results** to view the report.")
        else:
            st.error("❌ Evaluation failed. Check output above for details.")


# ── Page: Results ─────────────────────────────────────────────────────────────
elif page == "📊 Results":
    st.markdown("# 📊 Evaluation Results")

    if not selected_report_path:
        st.info("No reports found yet. Run an evaluation first.")
        st.stop()

    report = load_report(selected_report_path)
    if not report:
        st.error("Could not load the selected report.")
        st.stop()

    # Verdict banner
    if report.suite_passed:
        st.markdown('<div class="verdict-pass">✅ SUITE PASSED</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="verdict-fail">❌ SUITE FAILED</div>', unsafe_allow_html=True)

    # Run metadata
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Run ID", report.run_id)
    mcol2.metric("Agent", report.agent_name)
    mcol3.metric("Timestamp", report.timestamp[:10])

    st.markdown("---")

    # Gauge charts
    st.markdown("### 🎯 Dimension Scores")
    g1, g2, g3, g4 = st.columns(4)
    gauges = [
        (g1, report.safety_score,     "Safety",     "#f97316"),
        (g2, report.robustness_score, "Robustness", "#a78bfa"),
        (g3, report.accuracy_score,   "Accuracy",   "#60a5fa"),
        (g4, report.relevance_score,  "Relevance",  "#34d399"),
    ]
    for col, val, title, color in gauges:
        col.plotly_chart(gauge_fig(val, title, color), use_container_width=True, config={"displayModeBar": False})

    # Overall score + counts
    st.markdown("---")
    st.markdown("### 📈 Summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Overall Score", f"{report.overall_score:.1f} / 10")
    s2.metric("Total Tests", report.total_tests)
    s3.metric("✅ Passed", report.passed_tests)
    s4.metric("❌ Failed", report.failed_tests)
    s5.metric("⚠️ Errors", report.error_tests)

    # Category breakdown chart
    st.markdown("---")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### 📊 Pass/Fail by Category")
        st.plotly_chart(bar_chart(report), use_container_width=True, config={"displayModeBar": False})

    with col_right:
        st.markdown("### ⏱️ Latency")
        lat = report.latency
        c1, c2 = st.columns(2)
        c1.metric("Mean", f"{lat.mean_ms:.0f} ms")
        c2.metric("Median", f"{lat.median_ms:.0f} ms")
        c1.metric("Min", f"{lat.min_ms:.0f} ms")
        c2.metric("Max", f"{lat.max_ms:.0f} ms")

        st.markdown("### 🔢 Score Weights")
        wdf = pd.DataFrame({
            "Dimension": ["Safety", "Robustness", "Accuracy", "Relevance"],
            "Score": [report.safety_score, report.robustness_score,
                      report.accuracy_score, report.relevance_score],
            "Weight": ["×2.0", "×1.5", "×1.0", "×0.75"],
        })
        st.dataframe(wdf, use_container_width=True, hide_index=True)


# ── Page: Test Details ────────────────────────────────────────────────────────
elif page == "📋 Test Details":
    st.markdown("# 📋 Test Details")

    if not selected_report_path:
        st.info("No reports found yet. Run an evaluation first.")
        st.stop()

    report = load_report(selected_report_path)
    if not report:
        st.error("Could not load the selected report.")
        st.stop()

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        status_filter = st.selectbox("Status", ["All", "Pass", "Fail", "Error"])
    with f2:
        cat_filter = st.selectbox("Category", ["All", "normal", "edge", "adversarial", "safety"])
    with f3:
        min_score, max_score = st.slider("Score range", 0.0, 10.0, (0.0, 10.0), 0.5)

    # Build DataFrame
    rows = []
    for r in report.results:
        if r.error:
            status = "Error"
        elif r.eval_result.passed:
            status = "Pass"
        else:
            status = "Fail"

        if status_filter != "All" and status != status_filter:
            continue
        if cat_filter != "All" and r.test_case.category != cat_filter:
            continue
        score = r.eval_result.score
        if not (min_score <= score <= max_score):
            continue

        rows.append({
            "ID": r.test_case.id,
            "Category": r.test_case.category,
            "Score": round(score, 1),
            "Status": status,
            "Method": r.eval_result.method,
            "Duration (ms)": round(r.duration_ms, 0),
            "Input": r.test_case.input[:80] + ("…" if len(r.test_case.input) > 80 else ""),
            "Rationale": r.eval_result.rationale[:120] + ("…" if len(r.eval_result.rationale) > 120 else ""),
        })

    if not rows:
        st.warning("No results match the current filters.")
    else:
        df = pd.DataFrame(rows)
        st.markdown(f"**{len(rows)} results**")

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=10, format="%.1f"
                ),
                "Status": st.column_config.TextColumn("Status"),
            },
        )

        # Expandable drilldown
        st.markdown("---")
        st.markdown("### 🔍 Drill Down")
        test_ids = [r.test_case.id for r in report.results]
        selected_id = st.selectbox("Select a test case to inspect", test_ids)
        matching = [r for r in report.results if r.test_case.id == selected_id]
        if matching:
            r = matching[0]
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**Input**")
                st.info(r.test_case.input)
                st.markdown("**Expected Behavior**")
                st.info(r.test_case.expected_behavior)
            with d2:
                st.markdown("**Agent Response**")
                st.warning(r.agent_response if r.agent_response else "(empty — agent error)")
                st.markdown("**Evaluation Rationale**")
                status_fn = st.success if r.eval_result.passed else st.error
                status_fn(f"Score {r.eval_result.score:.1f}/10 — {r.eval_result.rationale}")
            if r.error:
                st.error(f"⚠️ Agent Error: {r.error}")


# ── Page: Report History ──────────────────────────────────────────────────────
elif page == "📁 Report History":
    st.markdown("# 📁 Report History")

    all_reports = list_reports()
    if not all_reports:
        st.info("No reports found. Run an evaluation first.")
        st.stop()

    history_rows = []
    for label, path in all_reports:
        rpt = load_report(path)
        if not rpt:
            continue
        history_rows.append({
            "Run ID": rpt.run_id,
            "Timestamp": rpt.timestamp[:19].replace("T", " "),
            "Agent": rpt.agent_name,
            "Overall": round(rpt.overall_score, 2),
            "Safety": round(rpt.safety_score, 2),
            "Robustness": round(rpt.robustness_score, 2),
            "Accuracy": round(rpt.accuracy_score, 2),
            "Tests": rpt.total_tests,
            "Passed": rpt.passed_tests,
            "Verdict": "✅ Pass" if rpt.suite_passed else "❌ Fail",
        })

    df_hist = pd.DataFrame(history_rows)
    st.dataframe(
        df_hist,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Overall":    st.column_config.ProgressColumn("Overall",    min_value=0, max_value=10, format="%.2f"),
            "Safety":     st.column_config.ProgressColumn("Safety",     min_value=0, max_value=10, format="%.2f"),
            "Robustness": st.column_config.ProgressColumn("Robustness", min_value=0, max_value=10, format="%.2f"),
            "Accuracy":   st.column_config.ProgressColumn("Accuracy",   min_value=0, max_value=10, format="%.2f"),
        },
    )

    # Score trend over time
    if len(history_rows) > 1:
        st.markdown("---")
        st.markdown("### 📉 Score Trends")
        df_trend = df_hist[["Timestamp", "Overall", "Safety", "Robustness", "Accuracy"]].copy()
        df_trend = df_trend.iloc[::-1].reset_index(drop=True)  # oldest first

        trend_fig = go.Figure()
        colors = {"Overall": "#ffffff", "Safety": "#f97316", "Robustness": "#a78bfa", "Accuracy": "#60a5fa"}
        for col, color in colors.items():
            trend_fig.add_trace(go.Scatter(
                x=df_trend["Timestamp"], y=df_trend[col],
                name=col, line=dict(color=color, width=2),
                mode="lines+markers",
            ))
        trend_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0",
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=20, l=10, r=10),
            height=320,
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155", range=[0, 10]),
        )
        st.plotly_chart(trend_fig, use_container_width=True, config={"displayModeBar": False})
