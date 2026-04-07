"""
Agent Evaluation Framework — Production-Grade Streamlit UI
5 Pages: Dashboard · New Evaluation · Results · Test Inspector · History
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agent Eval Studio",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #13131f 50%, #0f0f1a 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12121e 0%, #1a1a2e 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.15);
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label { color: #94a3b8 !important; font-size: 0.75rem !important; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.06) 100%);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.5rem;
    transition: all 0.25s ease;
}
.metric-card:hover {
    border-color: rgba(99,102,241,0.45);
    background: linear-gradient(135deg, rgba(99,102,241,0.12) 0%, rgba(139,92,246,0.09) 100%);
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.15);
}
.metric-label { font-size: 0.72rem; font-weight: 600; color: #7c8db5; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.4rem; }
.metric-value { font-size: 1.9rem; font-weight: 700; color: #e2e8f0; line-height: 1; }
.metric-sub   { font-size: 0.75rem; color: #64748b; margin-top: 0.35rem; }

/* ── Verdict banners ── */
.verdict-pass {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.08));
    border: 1px solid rgba(16,185,129,0.35);
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    text-align: center;
}
.verdict-fail {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.08));
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 14px;
    padding: 1.2rem 1.6rem;
    text-align: center;
}
.verdict-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 0.2rem; }
.verdict-sub   { font-size: 0.85rem; opacity: 0.75; }

/* ── Section headers ── */
.section-header {
    font-size: 1.1rem; font-weight: 600; color: #a5b4fc;
    border-bottom: 1px solid rgba(99,102,241,0.2);
    padding-bottom: 0.5rem;
    margin: 1.2rem 0 0.8rem 0;
}

/* ── Tag badges ── */
.tag {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 6px;
    padding: 0.15rem 0.55rem;
    font-size: 0.7rem;
    font-weight: 500;
    color: #a5b4fc;
    margin-right: 0.25rem;
}
.tag-pass { background: rgba(16,185,129,0.15); border-color: rgba(16,185,129,0.3); color: #6ee7b7; }
.tag-fail { background: rgba(239,68,68,0.15);  border-color: rgba(239,68,68,0.3);  color: #fca5a5; }
.tag-skip { background: rgba(245,158,11,0.12); border-color: rgba(245,158,11,0.3); color: #fcd34d; }

/* ── Tables ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(99,102,241,0.15) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}
.stMultiSelect [data-baseweb="select"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 10px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.4) !important;
}

/* ── Nav pills ── */
.nav-pill {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    font-size: 0.85rem; font-weight: 500;
    cursor: pointer; margin-bottom: 0.25rem;
    transition: all 0.2s ease;
    color: #94a3b8; background: transparent;
    width: 100%;
}
.nav-pill:hover  { background: rgba(99,102,241,0.1); color: #a5b4fc; }
.nav-pill.active { background: rgba(99,102,241,0.18); color: #a5b4fc; border-left: 3px solid #6366f1; padding-left: 0.8rem; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
}

/* ── Divider ── */
hr { border-color: rgba(99,102,241,0.15) !important; }

/* ── Logo ── */
.logo-text {
    font-size: 1.35rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.logo-sub { font-size: 0.7rem; color: #475569; font-weight: 400; margin-top: -4px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def _score_color(score: float) -> str:
    if score >= 8.0:
        return "#10b981"
    elif score >= 6.0:
        return "#f59e0b"
    else:
        return "#ef4444"


def _gauge(score: float, title: str, max_val: float = 10.0) -> go.Figure:
    color = _score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 28, "color": color, "family": "Inter"}, "suffix": "/10"},
        title={"text": title, "font": {"size": 13, "color": "#94a3b8", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, max_val], "tickwidth": 0, "tickcolor": "#334155"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 6],  "color": "rgba(239,68,68,0.07)"},
                {"range": [6, 8],  "color": "rgba(245,158,11,0.07)"},
                {"range": [8, 10], "color": "rgba(16,185,129,0.07)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": score},
        },
    ))
    fig.update_layout(
        height=180,
        margin={"t": 40, "b": 0, "l": 20, "r": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    return fig


def _metric_bar_chart(metric_scores: dict) -> go.Figure:
    names, scores, colors = [], [], []
    for name, data in metric_scores.items():
        score = data.get("score", 0) * 10 if isinstance(data, dict) else 0
        skipped = data.get("skipped", False) if isinstance(data, dict) else False
        names.append(name)
        scores.append(score)
        if skipped:
            colors.append("rgba(100,116,139,0.5)")
        elif score >= 8:
            colors.append("rgba(16,185,129,0.75)")
        elif score >= 6:
            colors.append("rgba(245,158,11,0.75)")
        else:
            colors.append("rgba(239,68,68,0.75)")

    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h",
        marker={"color": colors, "line": {"width": 0}},
        text=[f"{s:.1f}" for s in scores],
        textposition="outside",
        textfont={"family": "Inter", "size": 12, "color": "#cbd5e1"},
    ))
    fig.update_layout(
        height=max(220, len(names) * 42),
        xaxis={"range": [0, 11], "showgrid": False, "zeroline": False, "tickcolor": "#334155", "color": "#64748b"},
        yaxis={"autorange": "reversed", "tickcolor": "#334155", "color": "#cbd5e1"},
        margin={"l": 0, "r": 60, "t": 10, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
        bargap=0.35,
    )
    return fig


def _load_all_runs() -> list[dict]:
    """Load all summary.json files from reports/."""
    runs = []
    for path in sorted(REPORTS_DIR.glob("**/summary.json"), reverse=True):
        try:
            with open(path) as f:
                data = json.load(f)
                data["_path"] = str(path)
                runs.append(data)
        except Exception:
            pass
    return runs


def _load_latest_run() -> dict | None:
    runs = _load_all_runs()
    return runs[0] if runs else None


def _run_evaluation_sync(agent_name: str, agent_types: list[str],
                          categories: list[str], custom_criteria: str,
                          use_g_eval: bool) -> dict | None:
    """Run the evaluation pipeline synchronously from Streamlit."""
    try:
        from agents import agent_registry
        from src.loader import test_loader
        from src.runner import test_runner
        from src.metrics.aggregator import build_run_report
        from src.reporting import markdown_reporter, logger
        from src.evaluation.metric_selector import select_metrics
        from src.config import Config

        agent = asyncio.get_event_loop().run_until_complete(
            _async_setup_agent(agent_registry, agent_name)
        )
        if not agent:
            return None

        if categories:
            cases = []
            for cat in categories:
                cases.extend(test_loader.load_by_category(cat))
        else:
            cases = test_loader.load_all()

        metrics = select_metrics(
            agent_types,
            custom_criteria=custom_criteria if use_g_eval else None,
            include_g_eval=use_g_eval,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Config.REPORTS_DIR / f"run_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "log.jsonl"

        loop = asyncio.new_event_loop()
        results = loop.run_until_complete(
            test_runner.run_suite(agent, cases, metrics=metrics, log_path=log_path)
        )
        loop.run_until_complete(agent.teardown())
        loop.close()

        report = build_run_report(results, agent_name, agent_types=agent_types)
        logger.log_run_summary(report, log_path)
        markdown_reporter.generate(report, output_dir)

        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            f.write(report.model_dump_json(indent=2))

        with open(summary_path) as f:
            return json.load(f)

    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        return None


async def _async_setup_agent(registry, agent_name: str):
    try:
        agent = registry.get(agent_name)
        await agent.setup()
        return agent
    except Exception as e:
        st.error(f"Failed to load agent '{agent_name}': {e}")
        return None


# ── Sidebar navigation ──────────────────────────────────────────────────────────
def _sidebar():
    with st.sidebar:
        st.markdown('<div class="logo-text">🧪 Agent Eval Studio</div>', unsafe_allow_html=True)
        st.markdown('<div class="logo-sub">Production Evaluation Framework</div>', unsafe_allow_html=True)
        st.markdown("---")

        pages = [
            ("🏠", "Dashboard",       "dashboard"),
            ("▶",  "New Evaluation",  "new_eval"),
            ("📊", "Results",         "results"),
            ("🔍", "Test Inspector",  "inspector"),
            ("📁", "History",         "history"),
        ]

        if "page" not in st.session_state:
            st.session_state.page = "dashboard"

        for icon, label, key in pages:
            active = "active" if st.session_state.page == key else ""
            if st.button(f"{icon}  {label}", key=f"nav_{key}",
                         use_container_width=True,
                         type="secondary" if active else "secondary"):
                st.session_state.page = key
                st.rerun()

        st.markdown("---")
        # Quick status
        runs = _load_all_runs()
        st.markdown(f'<div class="metric-label">Total Runs</div><div style="color:#a5b4fc;font-weight:600">{len(runs)}</div>', unsafe_allow_html=True)
        if runs:
            latest = runs[0]
            verdict = "✅ PASS" if latest.get("suite_passed") else "❌ FAIL"
            st.markdown(f'<div class="metric-label" style="margin-top:0.6rem">Last Run</div><div style="color:#94a3b8;font-size:0.8rem">{verdict} · {latest.get("agent_name","—")}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div style="color:#334155;font-size:0.68rem;text-align:center">© 2025 Agent Eval Studio</div>', unsafe_allow_html=True)


# ── Page 1: Dashboard ───────────────────────────────────────────────────────────
def page_dashboard():
    st.markdown("## 🏠 Dashboard")
    st.markdown('<div style="color:#64748b;margin-bottom:1.5rem">Real-time performance overview across all evaluation runs.</div>', unsafe_allow_html=True)

    runs = _load_all_runs()

    if not runs:
        st.info("No evaluation runs yet. Go to **▶ New Evaluation** to get started.", icon="💡")
        return

    # ── Latest run headline ─────────────────────────────────────────────────
    latest = runs[0]
    passed = latest.get("suite_passed", False)
    verdict_cls = "verdict-pass" if passed else "verdict-fail"
    verdict_txt = "✅ PASSED" if passed else "❌ FAILED"
    agent_name  = latest.get("agent_name", "Unknown")
    overall     = latest.get("overall_score", 0)

    st.markdown(f"""
    <div class="{verdict_cls}">
      <div class="verdict-title">{verdict_txt}</div>
      <div class="verdict-sub">Latest run · Agent: <strong>{agent_name}</strong> · Overall: <strong>{overall:.1f}/10</strong></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # ── 4 dimension gauges ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.plotly_chart(_gauge(latest.get("safety_score", 0),    "🛡 Safety"),    use_container_width=True)
    with c2: st.plotly_chart(_gauge(latest.get("accuracy_score", 0),  "🎯 Accuracy"),  use_container_width=True)
    with c3: st.plotly_chart(_gauge(latest.get("robustness_score", 0),"⚡ Robustness"),use_container_width=True)
    with c4: st.plotly_chart(_gauge(latest.get("relevance_score", 0), "💡 Relevance"), use_container_width=True)

    # ── Summary stat cards ──────────────────────────────────────────────────
    total  = latest.get("total_tests", 0)
    passed_n = latest.get("passed_tests", 0)
    failed_n = latest.get("failed_tests", 0)
    errors_n = latest.get("error_tests", 0)
    lat_mean = latest.get("latency", {}).get("mean_ms", 0)

    cols = st.columns(5)
    cards = [
        ("Total Tests",    str(total),           ""),
        ("Passed",         str(passed_n),         "color:#10b981"),
        ("Failed",         str(failed_n),         "color:#ef4444"),
        ("Errors",         str(errors_n),         "color:#f59e0b"),
        ("Avg Latency",    f"{lat_mean:.0f} ms",  ""),
    ]
    for col, (label, val, style) in zip(cols, cards):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="{style}">{val}</div>
        </div>""", unsafe_allow_html=True)

    # ── Score trend ─────────────────────────────────────────────────────────
    if len(runs) > 1:
        st.markdown('<div class="section-header">📈 Score Trends</div>', unsafe_allow_html=True)
        df = pd.DataFrame([{
            "Run": f'#{i+1} {r.get("agent_name","?")}',
            "Overall":    r.get("overall_score", 0),
            "Safety":     r.get("safety_score", 0),
            "Accuracy":   r.get("accuracy_score", 0),
            "Robustness": r.get("robustness_score", 0),
            "Relevance":  r.get("relevance_score", 0),
        } for i, r in enumerate(reversed(runs[-10:]))])

        fig = px.line(
            df.melt(id_vars="Run", value_vars=["Overall","Safety","Accuracy","Robustness","Relevance"]),
            x="Run", y="value", color="variable", markers=True,
            color_discrete_map={
                "Overall": "#818cf8", "Safety": "#10b981",
                "Accuracy": "#f59e0b", "Robustness": "#60a5fa", "Relevance": "#c084fc",
            },
        )
        fig.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend={"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#94a3b8"}},
            xaxis={"tickcolor":"#334155","color":"#64748b","gridcolor":"rgba(99,102,241,0.06)"},
            yaxis={"range":[0,10.5],"tickcolor":"#334155","color":"#64748b","gridcolor":"rgba(99,102,241,0.06)"},
            font={"family":"Inter"},
            margin={"l":0,"r":0,"t":10,"b":0},
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Recent runs table ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🕒 Recent Runs</div>', unsafe_allow_html=True)
    table_data = []
    for r in runs[:8]:
        table_data.append({
            "Run ID":    r.get("run_id", "?"),
            "Agent":     r.get("agent_name", "?"),
            "Types":     ", ".join(r.get("agent_types", [])),
            "Verdict":   "✅ PASS" if r.get("suite_passed") else "❌ FAIL",
            "Overall":   f'{r.get("overall_score", 0):.1f}',
            "Safety":    f'{r.get("safety_score", 0):.1f}',
            "Accuracy":  f'{r.get("accuracy_score", 0):.1f}',
            "Timestamp": r.get("timestamp", "")[:19].replace("T", " "),
        })
    st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


# ── Page 2: New Evaluation ─────────────────────────────────────────────────────
def page_new_eval():
    st.markdown("## ▶ New Evaluation")
    st.markdown('<div style="color:#64748b;margin-bottom:1.5rem">Configure and launch an evaluation run against your agent.</div>', unsafe_allow_html=True)

    from src.evaluation.metric_selector import (
        AGENT_TYPE_LABELS, metric_names_for_types, METRIC_DESCRIPTIONS
    )

    col_form, col_info = st.columns([1.6, 1])

    with col_form:
        st.markdown('<div class="section-header">🤖 Agent Configuration</div>', unsafe_allow_html=True)

        agent_name = st.text_input("Agent Name", placeholder="e.g. simple_chatbot, openai_agent", key="eval_agent")

        type_options = list(AGENT_TYPE_LABELS.values())
        type_keys    = list(AGENT_TYPE_LABELS.keys())
        selected_labels = st.multiselect(
            "Agent Type(s)",
            options=type_options,
            default=["Simple Chatbot"],
            help="Select one or more agent types. Metrics are automatically activated based on selection.",
        )
        selected_types = [k for k, v in AGENT_TYPE_LABELS.items() if v in selected_labels]

        st.markdown('<div class="section-header">🧪 Test Suite</div>', unsafe_allow_html=True)

        cat_cols = st.columns(4)
        categories = []
        with cat_cols[0]:
            if st.checkbox("Normal",      value=True):  categories.append("normal")
        with cat_cols[1]:
            if st.checkbox("Edge Case",   value=True):  categories.append("edge")
        with cat_cols[2]:
            if st.checkbox("Adversarial", value=True):  categories.append("adversarial")
        with cat_cols[3]:
            if st.checkbox("Safety",      value=True):  categories.append("safety")

        judge_model = st.selectbox(
            "Judge Model",
            ["gemini-2.0-flash", "gemini-2.0-pro-exp", "gemini-1.5-flash", "gemini-1.5-pro"],
            help="Gemini model used as the LLM-as-judge evaluator.",
        )
        os.environ["JUDGE_MODEL_FAST"] = judge_model
        os.environ["JUDGE_MODEL_SLOW"] = judge_model

        st.markdown('<div class="section-header">⚙ Advanced Options</div>', unsafe_allow_html=True)
        use_g_eval = st.checkbox("Enable Custom Criteria (GEval)", value=False)
        custom_criteria = ""
        if use_g_eval:
            custom_criteria = st.text_area(
                "Custom Evaluation Criteria",
                placeholder="e.g. Evaluate whether the response is concise—ideally under 100 words—while fully addressing the user's question.",
                height=90,
            )

        st.markdown("")
        run_btn = st.button("🚀 Launch Evaluation", use_container_width=True, type="primary")

    with col_info:
        st.markdown('<div class="section-header">📏 Active Metrics</div>', unsafe_allow_html=True)

        if selected_types:
            active = metric_names_for_types(selected_types, include_g_eval=use_g_eval)
            for metric_name in active:
                desc = METRIC_DESCRIPTIONS.get(metric_name, "")
                st.markdown(f"""
                <div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);border-radius:10px;padding:0.7rem 0.9rem;margin-bottom:0.5rem">
                  <div style="font-size:0.8rem;font-weight:600;color:#a5b4fc">{metric_name}</div>
                  <div style="font-size:0.7rem;color:#64748b;margin-top:0.2rem">{desc}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Select at least one agent type to see active metrics.")

        st.markdown('<div class="section-header">ℹ Scoring</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.75rem;color:#64748b;line-height:1.7">
        • Scores: <strong style="color:#94a3b8">0–10</strong> (higher = better)<br>
        • Suite passes when: <strong style="color:#94a3b8">Overall ≥ 7.0</strong> AND <strong style="color:#94a3b8">Safety ≥ 8.5</strong><br>
        • Skipped metrics: field missing from test case<br>
        • Safety weight: <strong style="color:#94a3b8">2.0×</strong> (highest priority)
        </div>""", unsafe_allow_html=True)

    if run_btn:
        if not agent_name:
            st.error("Please enter an agent name.")
            return
        if not selected_types:
            st.error("Please select at least one agent type.")
            return
        if not categories:
            st.error("Please select at least one test category.")
            return

        with st.spinner("🔄 Running evaluation — this may take a few minutes..."):
            progress = st.progress(0, text="Initialising...")
            progress.progress(10, text="Loading agent...")
            result = _run_evaluation_sync(
                agent_name, selected_types, categories, custom_criteria, use_g_eval
            )
            progress.progress(100, text="Complete!")

        if result:
            st.session_state["last_result"] = result
            st.success("✅ Evaluation complete! Switching to Results page...")
            st.session_state.page = "results"
            st.rerun()


# ── Page 3: Results ──────────────────────────────────────────────────────────--
def page_results():
    st.markdown("## 📊 Results")
    st.markdown('<div style="color:#64748b;margin-bottom:1rem">Detailed breakdown of the latest evaluation run.</div>', unsafe_allow_html=True)

    run = st.session_state.get("last_result") or _load_latest_run()
    if not run:
        st.info("No results yet. Run an evaluation first.", icon="💡")
        return

    # Header
    passed = run.get("suite_passed", False)
    verdict_cls = "verdict-pass" if passed else "verdict-fail"
    verdict_txt = "✅ SUITE PASSED" if passed else "❌ SUITE FAILED"
    st.markdown(f"""
    <div class="{verdict_cls}">
      <div class="verdict-title">{verdict_txt}</div>
      <div class="verdict-sub">
        Agent: <strong>{run.get("agent_name","?")}</strong> ·
        Types: {", ".join(run.get("agent_types",[]))} ·
        Overall: <strong>{run.get("overall_score",0):.1f}/10</strong> ·
        {run.get("passed_tests",0)} passed / {run.get("failed_tests",0)} failed / {run.get("error_tests",0)} errors
      </div>
    </div>""", unsafe_allow_html=True)
    st.markdown("")

    # Dimension gauges
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.plotly_chart(_gauge(run.get("safety_score", 0),    "🛡 Safety"),    use_container_width=True)
    with c2: st.plotly_chart(_gauge(run.get("accuracy_score", 0),  "🎯 Accuracy"),  use_container_width=True)
    with c3: st.plotly_chart(_gauge(run.get("robustness_score", 0),"⚡ Robustness"),use_container_width=True)
    with c4: st.plotly_chart(_gauge(run.get("relevance_score", 0), "💡 Relevance"), use_container_width=True)

    # Per-metric aggregate bar chart
    results = run.get("results", [])
    if results:
        # Aggregate metric scores across all test cases
        metric_agg: dict[str, list[float]] = {}
        for r in results:
            for mname, mdata in (r.get("eval_result", {}).get("metric_scores", {}) or {}).items():
                if not mdata.get("skipped"):
                    metric_agg.setdefault(mname, []).append(mdata.get("score", 0) * 10)

        if metric_agg:
            avg_metrics = {k: {"score": sum(v)/len(v)/10, "skipped": False} for k, v in metric_agg.items()}
            st.markdown('<div class="section-header">📐 Per-Metric Averages</div>', unsafe_allow_html=True)
            st.plotly_chart(_metric_bar_chart(avg_metrics), use_container_width=True)

    # Test case results table
    st.markdown('<div class="section-header">🗂 Test Cases</div>', unsafe_allow_html=True)

    rows = []
    for i, r in enumerate(results):
        tc = r.get("test_case", {})
        er = r.get("eval_result", {})
        rows.append({
            "#":          i + 1,
            "Input":      tc.get("input", "")[:60] + ("…" if len(tc.get("input","")) > 60 else ""),
            "Category":   tc.get("category", "?"),
            "Status":     "✅ Pass" if er.get("passed") else ("⚠ Error" if r.get("error") else "❌ Fail"),
            "Score":      f'{er.get("score", 0):.1f}',
            "Method":     er.get("method", "?"),
            "Duration":   f'{r.get("duration_ms", 0):.0f}ms',
        })

    df = pd.DataFrame(rows)
    st.session_state["result_rows"] = results
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Export
    col_dl1, col_dl2 = st.columns([1, 5])
    with col_dl1:
        st.download_button(
            "⬇ Download JSON",
            data=json.dumps(run, indent=2, default=str),
            file_name=f"eval_{run.get('run_id','?')}.json",
            mime="application/json",
        )


# ── Page 4: Test Inspector ─────────────────────────────────────────────────────
def page_inspector():
    st.markdown("## 🔍 Test Inspector")
    st.markdown('<div style="color:#64748b;margin-bottom:1rem">Drill down into individual test cases and metric rationales.</div>', unsafe_allow_html=True)

    run = st.session_state.get("last_result") or _load_latest_run()
    if not run:
        st.info("No results to inspect. Run an evaluation first.", icon="💡")
        return

    results = run.get("results", [])
    if not results:
        st.warning("No test results found in last run.")
        return

    # Selector
    options = [f"#{i+1} [{r.get('test_case',{}).get('category','?')}] {r.get('test_case',{}).get('input','')[:55]}" for i, r in enumerate(results)]
    selected = st.selectbox("Select Test Case", options)
    idx = options.index(selected)
    r = results[idx]

    tc = r.get("test_case", {})
    er = r.get("eval_result", {})
    passed = er.get("passed", False)

    # Status banner
    if r.get("error"):
        st.error(f"⚠ Agent Error: {r.get('error')}")
    elif passed:
        st.success(f"✅ PASSED · Score: {er.get('score',0):.1f}/10 · Method: {er.get('method','?')}")
    else:
        st.error(f"❌ FAILED · Score: {er.get('score',0):.1f}/10 · Method: {er.get('method','?')}")

    # Content comparison
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">📥 Input</div>', unsafe_allow_html=True)
        st.code(tc.get("input", "—"), language=None)
        st.markdown('<div class="section-header">🎯 Expected Behavior</div>', unsafe_allow_html=True)
        st.code(tc.get("expected_behavior", "—"), language=None)

    with c2:
        st.markdown('<div class="section-header">🤖 Agent Response</div>', unsafe_allow_html=True)
        resp = r.get("agent_response", "[no response]")
        st.code(resp if resp else "[empty]", language=None)
        st.markdown('<div class="section-header">💬 Rationale</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:rgba(0,0,0,0.2);border-radius:8px;padding:0.8rem;font-size:0.8rem;color:#94a3b8">{er.get("rationale","—")}</div>', unsafe_allow_html=True)

    # Per-metric breakdown
    metric_scores = er.get("metric_scores") or {}
    if metric_scores:
        st.markdown('<div class="section-header">📐 Metric Breakdown</div>', unsafe_allow_html=True)
        for mname, mdata in metric_scores.items():
            if isinstance(mdata, dict):
                score  = mdata.get("score", 0)
                s10    = score * 10
                mpassed= mdata.get("passed", False)
                reason = mdata.get("reason", "")
                skipped= mdata.get("skipped", False)

                if skipped:
                    badge = '<span class="tag tag-skip">⏭ Skipped</span>'
                elif mpassed:
                    badge = '<span class="tag tag-pass">✅ Passed</span>'
                else:
                    badge = '<span class="tag tag-fail">❌ Failed</span>'

                with st.expander(f"{mname}  •  {s10:.1f}/10  {badge if not skipped else '⏭ Skipped'}", expanded=not skipped):
                    if skipped:
                        st.markdown(f'<span style="color:#fcd34d;font-size:0.8rem">⏭ {mdata.get("skip_reason","Missing required fields")}</span>', unsafe_allow_html=True)
                    else:
                        bar_color = _score_color(s10)
                        st.progress(score, text="")
                        st.markdown(f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:0.4rem">{reason}</div>', unsafe_allow_html=True)

    # Metadata
    with st.expander("ℹ Test Case Metadata"):
        meta = {
            "ID":       tc.get("id", "?"),
            "Category": tc.get("category", "?"),
            "Weight":   tc.get("weight", 1.0),
            "Tags":     ", ".join(tc.get("tags", [])),
            "Duration": f'{r.get("duration_ms",0):.0f}ms',
            "Model":    er.get("model_used", "?"),
        }
        cols = st.columns(3)
        for i, (k, v) in enumerate(meta.items()):
            cols[i % 3].markdown(f'<div class="metric-label">{k}</div><div style="color:#e2e8f0;font-size:0.85rem">{v}</div>', unsafe_allow_html=True)


# ── Page 5: History ────────────────────────────────────────────────────────────
def page_history():
    st.markdown("## 📁 History")
    st.markdown('<div style="color:#64748b;margin-bottom:1rem">All past evaluation runs with trend analysis.</div>', unsafe_allow_html=True)

    runs = _load_all_runs()
    if not runs:
        st.info("No past runs found.", icon="💡")
        return

    # Summary table
    st.markdown('<div class="section-header">📋 All Runs</div>', unsafe_allow_html=True)
    rows = []
    for r in runs:
        rows.append({
            "Run ID":     r.get("run_id", "?"),
            "Timestamp":  r.get("timestamp", "")[:19].replace("T", " "),
            "Agent":      r.get("agent_name", "?"),
            "Types":      ", ".join(r.get("agent_types", [])),
            "Verdict":    "✅ PASS" if r.get("suite_passed") else "❌ FAIL",
            "Overall":    round(r.get("overall_score", 0), 2),
            "Safety":     round(r.get("safety_score", 0), 2),
            "Accuracy":   round(r.get("accuracy_score", 0), 2),
            "Robustness": round(r.get("robustness_score", 0), 2),
            "Relevance":  round(r.get("relevance_score", 0), 2),
            "Tests":      f'{r.get("passed_tests",0)}✓ / {r.get("failed_tests",0)}✗',
            "Avg Lat (ms)": round(r.get("latency", {}).get("mean_ms", 0)),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Trend chart
    if len(runs) > 1:
        st.markdown('<div class="section-header">📈 Score Trends Over Time</div>', unsafe_allow_html=True)
        trend_df = pd.DataFrame([{
            "Run":        f'#{i+1}',
            "Overall":    r.get("overall_score", 0),
            "Safety":     r.get("safety_score", 0),
            "Accuracy":   r.get("accuracy_score", 0),
            "Robustness": r.get("robustness_score", 0),
            "Relevance":  r.get("relevance_score", 0),
        } for i, r in enumerate(reversed(runs))])

        fig = px.line(
            trend_df.melt(id_vars="Run", value_vars=["Overall","Safety","Accuracy","Robustness","Relevance"]),
            x="Run", y="value", color="variable", markers=True,
            color_discrete_map={
                "Overall":"#818cf8","Safety":"#10b981",
                "Accuracy":"#f59e0b","Robustness":"#60a5fa","Relevance":"#c084fc",
            },
        )
        fig.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend={"bgcolor":"rgba(0,0,0,0)","font":{"color":"#94a3b8"}},
            xaxis={"tickcolor":"#334155","color":"#64748b","gridcolor":"rgba(99,102,241,0.06)"},
            yaxis={"range":[0,10.5],"tickcolor":"#334155","color":"#64748b","gridcolor":"rgba(99,102,241,0.06)"},
            font={"family":"Inter"},
            margin={"l":0,"r":0,"t":10,"b":0},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Download all
    st.download_button(
        "⬇ Export All Runs (JSON)",
        data=json.dumps(runs, indent=2, default=str),
        file_name="all_eval_runs.json",
        mime="application/json",
    )

    # Pass/fail distribution
    if len(runs) > 2:
        st.markdown('<div class="section-header">🥧 Pass / Fail Distribution</div>', unsafe_allow_html=True)
        pass_count = sum(1 for r in runs if r.get("suite_passed"))
        fail_count = len(runs) - pass_count
        pie = go.Figure(go.Pie(
            labels=["Passed", "Failed"],
            values=[pass_count, fail_count],
            marker_colors=["rgba(16,185,129,0.7)", "rgba(239,68,68,0.7)"],
            hole=0.5,
            textinfo="label+percent",
            textfont={"family":"Inter","size":13,"color":"#e2e8f0"},
        ))
        pie.update_layout(
            height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            margin={"l":0,"r":0,"t":10,"b":0},
            font={"family":"Inter"},
        )
        col_pie, _ = st.columns([1, 2])
        with col_pie:
            st.plotly_chart(pie, use_container_width=True)


# ── Router ─────────────────────────────────────────────────────────────────────
def main():
    _sidebar()
    page = st.session_state.get("page", "dashboard")
    {
        "dashboard": page_dashboard,
        "new_eval":  page_new_eval,
        "results":   page_results,
        "inspector": page_inspector,
        "history":   page_history,
    }.get(page, page_dashboard)()


if __name__ == "__main__":
    main()
