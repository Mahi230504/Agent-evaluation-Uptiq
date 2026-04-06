"""
Premium Streamlit Dashboard for Agent Evaluation Framework.
Provides a rich, interactive interface for viewing and analyzing evaluation results.
"""

import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configure page settings
st.set_page_config(
    page_title="Agent Eval Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .stAlert {
        border-radius: 10px;
    }
    .css-1r6slb0 {
        padding-top: 2rem;
    }
    .report-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    .dimension-badge {
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_stdio=True)

def load_reports():
    """Load all summary.json files from the reports directory."""
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return []
    
    summaries = []
    for summary_file in reports_dir.glob("run_*/summary.json"):
        try:
            with open(summary_file, "r") as f:
                data = json.load(f)
                data["_file_path"] = summary_file
                summaries.append(data)
        except Exception:
            continue
            
    # Sort by timestamp descending
    return sorted(summaries, key=lambda x: x.get("timestamp", ""), reverse=True)

def render_radar_chart(report):
    """Render a radar chart for the 4 evaluation dimensions."""
    categories = ['Safety', 'Accuracy', 'Robustness', 'Relevance']
    values = [
        report['safety_score'],
        report['accuracy_score'],
        report['robustness_score'],
        report['relevance_score']
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(0, 255, 128, 0.2)',
        line=dict(color='#00ff80', width=2),
        name=report['agent_name']
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], gridcolor="#30363d", tickfont=dict(color="white")),
            angularaxis=dict(gridcolor="#30363d", tickfont=dict(color="white")),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=40, b=40, l=40, r=40),
        height=400
    )
    return fig

def main():
    st.sidebar.title("🤖 Agent Eval")
    st.sidebar.markdown("---")
    
    reports = load_reports()
    
    if not reports:
        st.info("No evaluation reports found. Run an evaluation first!")
        st.code("python main.py --agent simple_chatbot", language="bash")
        return

    # Sidebar selection
    report_options = [f"{r['agent_name']} ({r['timestamp']})" for r in reports]
    selected_option = st.sidebar.selectbox("Select Evaluation Run", report_options)
    selected_idx = report_options.index(selected_option)
    report = reports[selected_idx]
    
    # Header Section
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"Evaluation: {report['agent_name']}")
        status = "PASSED" if report['suite_passed'] else "FAILED"
        status_color = "#00ff80" if report['suite_passed'] else "#ff4b4b"
        st.markdown(f"**Status:** <span style='color:{status_color}; font-weight:bold;'>{status}</span> | **Timestamp:** {report['timestamp']}", unsafe_allow_stdio=True)
    
    with col2:
        st.markdown(f"<div style='text-align: right; font-size: 2rem; font-weight: bold; color: {status_color};'>{report['overall_score']:.1f}/10</div>", unsafe_allow_stdio=True)
        st.markdown("<div style='text-align: right; color: grey;'>Overall Score</div>", unsafe_allow_stdio=True)

    st.markdown("---")

    # Metrics Section
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Safety", f"{report['safety_score']:.1f}")
    m2.metric("Accuracy", f"{report['accuracy_score']:.1f}")
    m3.metric("Robustness", f"{report['robustness_score']:.1f}")
    m4.metric("Relevance", f"{report['relevance_score']:.1f}")

    st.markdown("### 📊 Performance Analysis")
    col_chart, col_stats = st.columns([1, 1])
    
    with col_chart:
        st.plotly_chart(render_radar_chart(report), use_container_width=True)
    
    with col_stats:
        # Latency & Throughput
        st.markdown("#### Latency Statistics")
        l1, l2 = st.columns(2)
        l1.metric("Mean Latency", f"{report['latency']['mean_ms']:.0f}ms")
        l2.metric("Median Latency", f"{report['latency']['median_ms']:.0f}ms")
        
        # Test Case Breakdown
        st.markdown("#### Test Case Breakdown")
        df_stats = pd.DataFrame({
            "Metric": ["Passed", "Failed", "Errors"],
            "Count": [report['passed_tests'], report['failed_tests'], report['error_tests']]
        })
        fig_stats = px.pie(df_stats, values='Count', names='Metric', 
                         color='Metric', color_discrete_map={'Passed':'#00ff80', 'Failed':'#ff4b4b', 'Errors':'#ffa500'})
        fig_stats.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=False, height=250, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_stats, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📝 Test Case Results")

    # Filter Section
    f1, f2 = st.columns([2, 1])
    with f2:
        category_filter = st.selectbox("Category", ["All", "normal", "edge", "adversarial", "safety"])
    
    # Results Table
    results = report['results']
    if category_filter != "All":
        results = [r for r in results if r['test_case']['category'] == category_filter]

    for res in results:
        tc = res['test_case']
        ev = res['eval_result']
        
        status_icon = "✅" if ev['passed'] else "❌"
        if ev.get("error"):
            status_icon = "⚠️"
            
        with st.expander(f"{status_icon} {tc['id']} — {tc['category']} (Score: {ev['score']:.1f})"):
            c_left, c_right = st.columns(2)
            with c_left:
                st.markdown("**Test Input:**")
                st.info(tc['input'])
                st.markdown("**Expected Behavior:**")
                st.write(tc['expected_behavior'])
            
            with c_right:
                st.markdown("**Agent Response:**")
                st.success(res['agent_response'])
                st.markdown("**Judge Rationale:**")
                st.warning(ev['rationale'])
            
            st.markdown(f"*Evaluated using {ev['method']} (Model: {ev.get('model_used', 'N/A')}) | Latency: {res['duration_ms']:.0f}ms*")

if __name__ == "__main__":
    main()
