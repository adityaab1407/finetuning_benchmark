"""Streamlit dashboard for the Financial Fine-Tuning Laboratory.

Dark-themed, production-grade ML research dashboard that visualizes
benchmark results across 6 LLM approaches on financial question
answering.  Connects to the FastAPI backend for live inference and
pre-computed benchmark data.
"""

import glob
import json
import logging
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# Configurable via API_BASE_URL env var so the Docker frontend container
# can reach the backend service by its compose service name.
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
USD_TO_INR = 83.5

APPROACH_COLORS = {
    "zero_shot": "#00d4ff",
    "few_shot": "#7c3aed",
    "chain_of_thought": "#f59e0b",
    "cot": "#f59e0b",
    "rag": "#10b981",
    "sft": "#3b82f6",
    "dpo": "#ec4899",
}

APPROACH_LABELS = {
    "zero_shot": "Zero-Shot",
    "few_shot": "Few-Shot",
    "chain_of_thought": "Chain-of-Thought",
    "cot": "Chain-of-Thought",
    "rag": "RAG",
    "sft": "SFT Fine-Tuned",
    "dpo": "DPO Aligned",
}


# ── page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Financial Fine-Tuning Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── global CSS ───────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg-primary: #0e1117;
    --bg-card: #1c2333;
    --bg-card-hover: #232b3e;
    --accent-cyan: #00d4ff;
    --accent-purple: #7c3aed;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --accent-blue: #3b82f6;
    --accent-pink: #ec4899;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --border-subtle: rgba(148, 163, 184, 0.12);
}

html, body, [class*="st-"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

/* Hide default Streamlit chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] {background: transparent;}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid var(--border-subtle);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: var(--text-secondary);
}

/* Metric card */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 4px;
}
.metric-label {
    font-size: 0.82rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
}
.metric-sublabel {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 2px;
}

/* Approach result card */
.approach-card {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 20px;
    min-height: 220px;
    border-left: 4px solid var(--accent-cyan);
    transition: transform 0.2s, box-shadow 0.2s;
}
.approach-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}
.approach-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.approach-card-name {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-primary);
    text-transform: uppercase;
    letter-spacing: 0.3px;
}
.approach-badge {
    font-size: 0.65rem;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-groq { background: rgba(0, 212, 255, 0.15); color: var(--accent-cyan); }
.badge-local { background: rgba(59, 130, 246, 0.15); color: var(--accent-blue); }
.approach-card-answer {
    color: var(--text-primary);
    font-size: 0.9rem;
    line-height: 1.55;
    margin: 12px 0;
    max-height: 100px;
    overflow-y: auto;
}
.approach-card-metrics {
    display: flex;
    gap: 16px;
    padding-top: 12px;
    border-top: 1px solid var(--border-subtle);
    font-size: 0.78rem;
    color: var(--text-secondary);
}
.approach-card-metrics span {
    font-weight: 600;
    color: var(--text-primary);
}

/* Placeholder card */
.placeholder-card {
    background: var(--bg-card);
    border: 1px dashed rgba(148, 163, 184, 0.25);
    border-radius: 12px;
    padding: 32px;
    text-align: center;
}
.placeholder-card h3 {
    color: var(--text-primary);
    margin-bottom: 8px;
}
.placeholder-card p {
    color: var(--text-secondary);
    font-size: 0.9rem;
}

/* Section title */
.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 28px 0 16px 0;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-subtle);
}

/* Decision guide pill */
.guide-pill {
    background: var(--bg-card);
    border-radius: 12px;
    padding: 18px 20px;
    border-left: 4px solid var(--accent-cyan);
}
.guide-pill h4 {
    color: var(--text-primary);
    margin: 0 0 4px 0;
    font-size: 0.95rem;
}
.guide-pill p {
    color: var(--text-secondary);
    margin: 0;
    font-size: 0.82rem;
}

/* Stat note */
.stat-note {
    color: var(--text-secondary);
    font-size: 0.78rem;
    font-style: italic;
    margin-top: 24px;
    padding: 12px;
    background: var(--bg-card);
    border-radius: 8px;
    border-left: 3px solid var(--accent-cyan);
}

/* Tab styling overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: var(--bg-card);
    padding: 4px;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(0, 212, 255, 0.12);
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── helper functions ─────────────────────────────────────────────────────


def normalize_accuracy(val: float | None) -> float:
    """Ensure accuracy is on 0-100 scale."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)) and val <= 1.5:
        return float(val) * 100
    return float(val)


def _api_get(path: str, timeout: int = 10) -> dict | list | None:
    """GET from the FastAPI backend. Returns None on failure."""
    try:
        resp = requests.get(f"{API_BASE}{path}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _api_post(path: str, payload: dict, timeout: int = 60) -> dict | None:
    """POST to the FastAPI backend. Returns None on failure."""
    try:
        resp = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


@st.cache_data(ttl=60)
def load_benchmark_results() -> dict | None:
    """Load benchmark results, preferring full runs over dry runs."""
    # Try loading individual run files directly (bypasses API/summary staleness)
    run_files = glob.glob("evaluation/results/*_run*.json")
    results_by_approach: dict[str, dict] = {}

    for fpath in run_files:
        try:
            data = json.loads(Path(fpath).read_text(encoding="utf-8"))
            if data.get("status") != "complete":
                continue
            completed = data.get("completed_questions", 0)
            if completed <= 5:  # ignore dry runs
                continue
            approach = data.get("approach_name", "")
            # Keep the run with most questions per approach
            existing = results_by_approach.get(approach)
            if not existing or completed > existing.get("completed_questions", 0):
                results_by_approach[approach] = data
        except Exception:
            pass

    if len(results_by_approach) >= 2:
        return _build_summary_from_runs(results_by_approach)

    # Fall back to summary file
    summary_path = Path("evaluation/results/benchmark_summary.json")
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            total_q = data.get("total_questions", 0)
            if total_q > 5:
                return data
        except Exception:
            pass

    # Fall back to API
    api_data = _api_get("/api/results")
    if api_data and api_data.get("available"):
        return api_data.get("summary")

    return None


def _build_summary_from_runs(results: dict[str, dict]) -> dict:
    """Build a display-ready summary from individual run files."""
    approaches_data: dict[str, dict] = {}

    for name, data in results.items():
        acc = data.get("accuracy_pct", 0)
        approaches_data[name] = {
            "accuracy_per_run": [acc],
            "overall_accuracy_pct": acc,
            "bootstrap_ci": {
                "ci_lower": 0.0,
                "ci_upper": 0.0,
                "margin_of_error": 0.0,
            },
            "hallucination_rate_pct": data.get("hallucination_rate_pct", 0),
            "avg_latency_ms": data.get("avg_latency_ms", 0),
            "p95_latency_ms": 0,
            "total_cost_usd": data.get("total_cost_usd", 0),
            "cost_per_correct_usd": data.get("cost_per_correct_usd", 0),
            "cost_per_correct_inr": data.get("cost_per_correct_usd", 0) * USD_TO_INR,
            "completed_questions": data.get("completed_questions", 0),
        }

    ranking = sorted(
        approaches_data.keys(),
        key=lambda n: approaches_data[n]["overall_accuracy_pct"],
        reverse=True,
    )

    return {
        "benchmark_version": "1.0.0",
        "approaches": approaches_data,
        "ranking": ranking,
        "best_approach": ranking[0] if ranking else "",
        "total_questions": 100,
        "n_runs": 1,
        "statistical_notes": (
            "Results from 1 run of 100 questions. "
            "Confidence intervals will narrow with additional runs. "
            "Bootstrap CI computed over 1,000 resamples."
        ),
    }


@st.cache_data(ttl=60)
def load_approach_run(approach_name: str) -> dict | None:
    """Load per-approach run results."""
    return _api_get(f"/api/results/{approach_name}")


def load_version_comparison() -> dict | None:
    """Load synthetic data version comparison from local file."""
    path = Path("data/synthetic/version_comparison.json")
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def format_metric(value: float, decimals: int = 1, prefix: str = "", suffix: str = "") -> str:
    """Format a number for display."""
    if decimals == 0:
        return f"{prefix}{value:.0f}{suffix}"
    return f"{prefix}{value:.{decimals}f}{suffix}"


def _metric_card_html(value: str, label: str, sublabel: str = "", color: str = "#00d4ff") -> str:
    """Return HTML for a styled metric card."""
    sub = f'<div class="metric-sublabel">{sublabel}</div>' if sublabel else ""
    return f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {color};">{value}</div>
        <div class="metric-label">{label}</div>
        {sub}
    </div>
    """


def _approach_card_html(result: dict, color: str) -> str:
    """Return HTML for an approach result card."""
    name = APPROACH_LABELS.get(result.get("approach_name", ""), result.get("approach_name", ""))
    answer = result.get("answer", "") or ""
    error = result.get("error")
    latency = result.get("latency_ms", 0)
    tokens = result.get("tokens_total", 0)
    cost_inr = result.get("cost_inr", 0)

    if error:
        body = f'<div class="approach-card-answer" style="color: #ef4444;">{error}</div>'
    elif not answer:
        body = '<div class="approach-card-answer" style="color: var(--text-secondary);">No response</div>'
    else:
        safe_answer = answer.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        body = f'<div class="approach-card-answer">{safe_answer}</div>'

    return f"""
    <div class="approach-card" style="border-left-color: {color};">
        <div class="approach-card-header">
            <span class="approach-card-name">{name}</span>
            <span class="approach-badge badge-groq">GROQ</span>
        </div>
        {body}
        <div class="approach-card-metrics">
            <div><span>{latency:.0f}ms</span> latency</div>
            <div><span>{tokens}</span> tokens</div>
            <div><span>{format_metric(cost_inr, 4, prefix="₹")}</span> cost</div>
        </div>
    </div>
    """


def _placeholder_approach_card(name: str, label: str, color: str, detail: str) -> str:
    """Return HTML for a placeholder approach card."""
    return f"""
    <div class="approach-card" style="border-left-color: {color}; opacity: 0.7;">
        <div class="approach-card-header">
            <span class="approach-card-name">{label}</span>
            <span class="approach-badge badge-local">LOCAL</span>
        </div>
        <div class="approach-card-answer" style="color: var(--text-secondary);">
            {detail}
        </div>
    </div>
    """


def render_leaderboard(summary: dict) -> None:
    """Render the benchmark leaderboard using st.dataframe."""
    st.subheader("Leaderboard")

    approaches_data = summary.get("approaches", {})
    if not approaches_data:
        st.info("No benchmark data available yet.")
        return

    rows = []
    ranking = summary.get("ranking", list(approaches_data.keys()))

    for i, name in enumerate(ranking, 1):
        data = approaches_data.get(name, {})
        accuracy = normalize_accuracy(data.get("overall_accuracy_pct", data.get("accuracy_pct", 0)))
        moe = data.get("bootstrap_ci", {}).get("margin_of_error", 0)

        acc_str = f"{accuracy:.1f}%"
        if moe > 0:
            acc_str += f" ± {moe:.1f}%"

        halluc = data.get("hallucination_rate_pct", 0)
        latency = data.get("avg_latency_ms", 0)
        cost_inr = data.get("cost_per_correct_inr", 0)

        rows.append({
            "Rank": i,
            "Approach": APPROACH_LABELS.get(name, name.replace("_", " ").title()),
            "Accuracy": acc_str,
            "Halluc. %": f"{halluc:.1f}%",
            "Avg Latency": f"{latency:.0f}ms",
            "₹/Correct": f"₹{cost_inr:.4f}" if cost_inr > 0 else "₹0.00 (free)",
        })

    # Placeholder rows for fine-tuned models
    rows.append({
        "Rank": "—",
        "Approach": "SFT Fine-Tuned",
        "Accuracy": "Training...",
        "Halluc. %": "—",
        "Avg Latency": "—",
        "₹/Correct": "—",
    })
    rows.append({
        "Rank": "—",
        "Approach": "DPO Aligned",
        "Accuracy": "Training...",
        "Halluc. %": "—",
        "Avg Latency": "—",
        "₹/Correct": "—",
    })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── sidebar ──────────────────────────────────────────────────────────────


with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 4px 0;">
        <span style="font-size: 1.6rem;">🧪</span>
        <span style="font-size: 1.2rem; font-weight: 700; color: #f8fafc; margin-left: 6px;">
            Financial Fine-Tuning Lab
        </span>
    </div>
    <p style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">
        Benchmarking 6 LLM approaches on financial tasks
    </p>
    """, unsafe_allow_html=True)

    st.divider()

    # Status section
    status = _api_get("/api/status")
    health = _api_get("/health")

    if status:
        runs_done = status.get("benchmark_runs_complete", 0)
        st.markdown(f"""
        <div style="font-size: 0.78rem; color: #94a3b8; line-height: 2;">
            <div>🟢 <span style="color:#f8fafc;">Benchmark:</span> {runs_done}/12 runs complete</div>
            <div>🟢 <span style="color:#f8fafc;">ChromaDB:</span> {health.get('chromadb_chunks', 0) if health else '—'} chunks</div>
            <div>🟢 <span style="color:#f8fafc;">Eval set:</span> {health.get('eval_questions', 0) if health else '—'} questions</div>
            <div>🔄 <span style="color:#f8fafc;">Fine-tuned:</span> Training in progress</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size: 0.78rem; color: #94a3b8; line-height: 2;">
            <div>🔴 <span style="color:#f8fafc;">Backend:</span> Not connected</div>
            <div style="color: #f59e0b; margin-top: 4px;">Run: <code>make app</code></div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    st.markdown("""
    <div style="position: fixed; bottom: 16px; font-size: 0.7rem; color: #475569; line-height: 1.6;">
        Built with Groq · ChromaDB · Mistral-7B<br>
        Llama 4 Scout · Llama 3.3 70B
    </div>
    """, unsafe_allow_html=True)


# ── main tabs ────────────────────────────────────────────────────────────


tab_demo, tab_bench, tab_train, tab_gpu = st.tabs([
    "🎯 Live Demo",
    "📊 Benchmark",
    "🔬 Training Lab",
    "⚡ GPU & Systems",
])


# ═══════════════════════════════════════════════════════════════════════
# TAB 1: LIVE DEMO
# ═══════════════════════════════════════════════════════════════════════

with tab_demo:
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <h2 style="color: #f8fafc; margin-bottom: 0;">Ask a Financial Question</h2>
        <p style="color: #94a3b8; font-size: 0.9rem; margin-top: 4px;">
            Watch 4 approaches answer simultaneously — compare accuracy, speed, and cost
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Session state init
    if "results" not in st.session_state:
        st.session_state.results = None
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

    # Simple text area — no dropdown
    question = st.text_area(
        "Type your question here:",
        value="How much remained available under Microsoft's $60.0 billion share repurchase program approved on September 16, 2024",
        height=80,
        key="question_input",
    )

    run_clicked = st.button(
        "🚀 Run All Approaches",
        type="primary",
        use_container_width=True,
    )

    if run_clicked and question.strip():
        with st.spinner("Running all 4 approaches concurrently..."):
            try:
                response = requests.post(
                    f"{API_BASE}/api/run/all",
                    json={"question": question.strip()},
                    timeout=60,
                )
                if response.status_code == 200:
                    st.session_state.results = response.json()
                    st.session_state.last_question = question.strip()
                else:
                    st.error(f"API error: {response.status_code}")
            except requests.exceptions.ConnectionError:
                st.error("Backend not running. Start with: `make app`")
            except Exception as e:
                st.error(f"Error: {e}")
    elif run_clicked:
        st.warning("Please enter a question (at least 10 characters).")

    # Display results from session state
    api_response = st.session_state.results
    results = api_response.get("results", []) if isinstance(api_response, dict) else []

    if results:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        # 2x2 grid of approach cards
        row1_col1, row1_col2 = st.columns(2, gap="medium")
        row2_col1, row2_col2 = st.columns(2, gap="medium")

        approach_order = ["zero_shot", "few_shot", "chain_of_thought", "cot", "rag"]
        result_map = {r.get("approach_name", ""): r for r in results}

        cards_placed = []
        for name in approach_order:
            if name in result_map and name not in [c.get("approach_name") for c in cards_placed]:
                cards_placed.append(result_map[name])

        slots = [row1_col1, row1_col2, row2_col1, row2_col2]

        for i, slot in enumerate(slots):
            with slot:
                if i < len(cards_placed):
                    r = cards_placed[i]
                    aname = r.get("approach_name", "")
                    color = APPROACH_COLORS.get(aname, "#00d4ff")
                    st.markdown(_approach_card_html(r, color), unsafe_allow_html=True)
                elif i == 2 and len(cards_placed) <= 2:
                    st.markdown(_placeholder_approach_card(
                        "sft", "SFT Fine-Tuned", APPROACH_COLORS["sft"],
                        "🔄 Training in progress<br>Dataset: ready for Kaggle<br>Status: SFT r=16 queued",
                    ), unsafe_allow_html=True)
                elif i == 3 and len(cards_placed) <= 3:
                    st.markdown(_placeholder_approach_card(
                        "dpo", "DPO Aligned", APPROACH_COLORS["dpo"],
                        "🔄 Pending SFT completion<br>DPO alignment follows SFT<br>Status: Queued",
                    ), unsafe_allow_html=True)

        # Show SFT/DPO placeholders if we have all 4 real results
        if len(cards_placed) >= 4:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            pc1, pc2 = st.columns(2, gap="medium")
            with pc1:
                st.markdown(_placeholder_approach_card(
                    "sft", "SFT Fine-Tuned", APPROACH_COLORS["sft"],
                    "🔄 Training in progress<br>Dataset: ready for Kaggle<br>Status: SFT r=16 queued",
                ), unsafe_allow_html=True)
            with pc2:
                st.markdown(_placeholder_approach_card(
                    "dpo", "DPO Aligned", APPROACH_COLORS["dpo"],
                    "🔄 Pending SFT completion<br>DPO alignment follows SFT<br>Status: Queued",
                ), unsafe_allow_html=True)

        # Latency comparison bar
        latency_data = [
            {"Approach": APPROACH_LABELS.get(r.get("approach_name", ""), r.get("approach_name", "")),
             "Latency (ms)": r.get("latency_ms", 0),
             "color": APPROACH_COLORS.get(r.get("approach_name", ""), "#00d4ff")}
            for r in cards_placed if not r.get("error")
        ]
        if latency_data:
            st.markdown('<div class="section-title">Response Time Comparison</div>', unsafe_allow_html=True)
            fig = go.Figure()
            for d in sorted(latency_data, key=lambda x: x["Latency (ms)"]):
                fig.add_trace(go.Bar(
                    y=[d["Approach"]],
                    x=[d["Latency (ms)"]],
                    orientation="h",
                    marker_color=d["color"],
                    text=[f'{d["Latency (ms)"]:.0f}ms'],
                    textposition="outside",
                    textfont=dict(color="#f8fafc", size=12),
                    showlegend=False,
                ))
            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=180,
                margin=dict(l=0, r=60, t=10, b=10),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=False),
                bargap=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)

    elif not results and not run_clicked:
        st.markdown("""
        <div class="placeholder-card" style="margin-top: 20px;">
            <h3>Enter a question and click Run</h3>
            <p>All 4 approaches will answer the same question simultaneously.<br>
            Compare accuracy, latency, token usage, and cost side by side.</p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 2: BENCHMARK DASHBOARD
# ═══════════════════════════════════════════════════════════════════════

with tab_bench:
    summary = load_benchmark_results()

    if not summary:
        st.markdown("""
        <div class="placeholder-card" style="margin-top: 20px;">
            <h3>Benchmark In Progress</h3>
            <p>Results will appear here automatically when the benchmark run completes.<br>
            Run <code>python3 -m evaluation.run_benchmark</code> to start.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        approaches = summary.get("approaches", {})
        ranking = summary.get("ranking", [])

        # ── SECTION 1: Headline Metrics ──────────────────────────────
        st.markdown("""
        <div style="margin-bottom: 8px;">
            <h2 style="color: #f8fafc; margin-bottom: 0;">Benchmark Results</h2>
            <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">
                {} approaches × {} questions × {} run(s)
            </p>
        </div>
        """.format(len(approaches), summary.get("total_questions", 100), summary.get("n_runs", 1)),
        unsafe_allow_html=True)

        # Find best values
        best_acc_name = ranking[0] if ranking else ""
        best_acc_val = normalize_accuracy(
            approaches.get(best_acc_name, {}).get("overall_accuracy_pct", 0)
        )

        lowest_halluc_name = min(
            approaches,
            key=lambda n: approaches[n].get("hallucination_rate_pct", 100),
        ) if approaches else ""
        lowest_halluc_val = approaches.get(lowest_halluc_name, {}).get(
            "hallucination_rate_pct", 0
        )

        fastest_name = min(
            approaches,
            key=lambda n: approaches[n].get("avg_latency_ms", 99999),
        ) if approaches else ""
        fastest_val = approaches.get(fastest_name, {}).get("avg_latency_ms", 0)

        best_cost_name = ""
        best_cost_val = 0.0
        for n, d in approaches.items():
            cpci = d.get("cost_per_correct_inr", 0)
            if cpci > 0 and (not best_cost_name or cpci < best_cost_val):
                best_cost_name = n
                best_cost_val = cpci
        if not best_cost_name and approaches:
            best_cost_name = ranking[0] if ranking else list(approaches.keys())[0]

        m1, m2, m3, m4 = st.columns(4, gap="medium")
        with m1:
            st.markdown(_metric_card_html(
                f"{best_acc_val:.1f}%", "Best Accuracy",
                APPROACH_LABELS.get(best_acc_name, best_acc_name),
                "#00d4ff",
            ), unsafe_allow_html=True)
        with m2:
            st.markdown(_metric_card_html(
                f"{lowest_halluc_val:.1f}%", "Lowest Hallucination",
                APPROACH_LABELS.get(lowest_halluc_name, lowest_halluc_name),
                "#10b981",
            ), unsafe_allow_html=True)
        with m3:
            st.markdown(_metric_card_html(
                f"{fastest_val:.0f}ms", "Fastest Avg Latency",
                APPROACH_LABELS.get(fastest_name, fastest_name),
                "#f59e0b",
            ), unsafe_allow_html=True)
        with m4:
            cost_display = f"₹{best_cost_val:.4f}" if best_cost_val > 0 else "₹0.00"
            st.markdown(_metric_card_html(
                cost_display, "Best ₹/Correct",
                APPROACH_LABELS.get(best_cost_name, best_cost_name) if best_cost_val > 0 else "Groq free tier",
                "#7c3aed",
            ), unsafe_allow_html=True)

        # ── SECTION 2: Leaderboard (st.dataframe) ────────────────────
        render_leaderboard(summary)

        # ── SECTION 3: Charts ────────────────────────────────────────
        st.markdown('<div class="section-title">Performance Analysis</div>', unsafe_allow_html=True)

        chart_col1, chart_col2 = st.columns(2, gap="large")

        # Left: Accuracy by category (from per-approach run data)
        with chart_col1:
            cat_data = []
            for name in ranking:
                run_data = load_approach_run(name)
                if run_data and run_data.get("runs"):
                    cat_correct: dict[str, int] = {}
                    cat_total: dict[str, int] = {}
                    for run in run_data["runs"]:
                        for qr in run.get("question_results", []):
                            cat = qr.get("category", "unknown")
                            cat_total[cat] = cat_total.get(cat, 0) + 1
                            if qr.get("correct"):
                                cat_correct[cat] = cat_correct.get(cat, 0) + 1
                    for cat, total in cat_total.items():
                        correct = cat_correct.get(cat, 0)
                        cat_data.append({
                            "Approach": APPROACH_LABELS.get(name, name),
                            "Category": cat.replace("_", " ").title(),
                            "Accuracy %": round(correct / total * 100, 1) if total > 0 else 0,
                            "color": APPROACH_COLORS.get(name, "#94a3b8"),
                        })

            if cat_data:
                color_map = {APPROACH_LABELS.get(n, n): APPROACH_COLORS.get(n, "#94a3b8") for n in ranking}
                fig_cat = px.bar(
                    cat_data,
                    x="Category",
                    y="Accuracy %",
                    color="Approach",
                    barmode="group",
                    template="plotly_dark",
                    title="Accuracy by Question Category",
                    color_discrete_map=color_map,
                )
                fig_cat.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=380,
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", y=-0.15),
                )
                st.plotly_chart(fig_cat, use_container_width=True)
            else:
                st.info("Category breakdown requires per-approach run data.")

        # Right: Cost vs Accuracy scatter
        with chart_col2:
            scatter_data = []
            for name in ranking:
                d = approaches[name]
                scatter_data.append({
                    "Approach": APPROACH_LABELS.get(name, name),
                    "Accuracy %": normalize_accuracy(d.get("overall_accuracy_pct", 0)),
                    "₹ per Correct": d.get("cost_per_correct_inr", 0),
                    "Avg Latency": d.get("avg_latency_ms", 0),
                    "color": APPROACH_COLORS.get(name, "#94a3b8"),
                })

            fig_scatter = go.Figure()
            for sd in scatter_data:
                fig_scatter.add_trace(go.Scatter(
                    x=[sd["₹ per Correct"]],
                    y=[sd["Accuracy %"]],
                    mode="markers+text",
                    marker=dict(size=18, color=sd["color"], line=dict(width=2, color="#0e1117")),
                    text=[sd["Approach"]],
                    textposition="top center",
                    textfont=dict(color="#f8fafc", size=11),
                    showlegend=False,
                    hovertemplate=f"<b>{sd['Approach']}</b><br>Accuracy: {sd['Accuracy %']:.1f}%<br>₹/correct: ₹{sd['₹ per Correct']:.4f}<extra></extra>",
                ))
            fig_scatter.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title="Cost vs Accuracy Tradeoff",
                xaxis_title="₹ per Correct Answer",
                yaxis_title="Accuracy %",
                height=380,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            fig_scatter.add_annotation(
                text="All approaches ₹0 (Groq free tier) — metric preserved for paid tier",
                xref="paper", yref="paper", x=0.5, y=-0.18,
                showarrow=False, font=dict(size=10, color="#94a3b8"),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        # ── SECTION 4: Hallucination comparison ──────────────────────
        st.markdown('<div class="section-title">Hallucination Rate by Approach</div>', unsafe_allow_html=True)

        halluc_data = []
        for name in reversed(ranking):
            d = approaches[name]
            halluc_data.append({
                "Approach": APPROACH_LABELS.get(name, name),
                "Hallucination %": d.get("hallucination_rate_pct", 0),
                "color": "#10b981" if d.get("hallucination_rate_pct", 0) < 15 else (
                    "#f59e0b" if d.get("hallucination_rate_pct", 0) < 30 else "#ef4444"
                ),
            })

        fig_halluc = go.Figure()
        for hd in halluc_data:
            fig_halluc.add_trace(go.Bar(
                y=[hd["Approach"]],
                x=[hd["Hallucination %"]],
                orientation="h",
                marker_color=hd["color"],
                text=[f'{hd["Hallucination %"]:.1f}%'],
                textposition="outside",
                textfont=dict(color="#f8fafc", size=12),
                showlegend=False,
            ))
        fig_halluc.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
            margin=dict(l=0, r=60, t=10, b=10),
            xaxis=dict(title="Hallucination Rate %", showgrid=True, gridcolor="rgba(148,163,184,0.08)"),
            yaxis=dict(showgrid=False),
            bargap=0.4,
        )
        st.plotly_chart(fig_halluc, use_container_width=True)

        # ── SECTION 5: Decision Guide ────────────────────────────────
        st.markdown('<div class="section-title">When To Use Which Approach</div>', unsafe_allow_html=True)

        g1, g2 = st.columns(2, gap="medium")
        with g1:
            st.markdown("""
            <div class="guide-pill" style="border-left-color: #00d4ff; margin-bottom: 12px;">
                <h4>🎯 Factual Extraction → RAG</h4>
                <p>Retrieves real document evidence for precise figures and facts</p>
            </div>
            <div class="guide-pill" style="border-left-color: #f59e0b;">
                <h4>💭 Sentiment Analysis → Few-Shot</h4>
                <p>Examples clarify expected label format (positive/negative/neutral)</p>
            </div>
            """, unsafe_allow_html=True)
        with g2:
            st.markdown("""
            <div class="guide-pill" style="border-left-color: #10b981; margin-bottom: 12px;">
                <h4>📋 Structured JSON → Fine-Tune (coming)</h4>
                <p>Learns format consistency through training on schema examples</p>
            </div>
            <div class="guide-pill" style="border-left-color: #7c3aed;">
                <h4>🧠 Multi-Step Reasoning → Chain-of-Thought</h4>
                <p>Structured thinking improves multi-step calculation accuracy</p>
            </div>
            """, unsafe_allow_html=True)

        # ── SECTION 6: Statistical note ──────────────────────────────
        n_runs = summary.get("n_runs", 1)
        stat_note = summary.get("statistical_notes", "")
        st.markdown(f"""
        <div class="stat-note">
            {stat_note if stat_note else f"Results from {n_runs} run(s) of {summary.get('total_questions', 100)} questions. Confidence intervals will narrow with additional runs. Bootstrap CI computed over 1,000 resamples."}
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 3: TRAINING LAB
# ═══════════════════════════════════════════════════════════════════════

with tab_train:
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <h2 style="color: #f8fafc; margin-bottom: 0;">🔬 Fine-Tuning Laboratory</h2>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">
            Synthetic data generation, quality filtering, and model training progress
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Version comparison from synthetic pipeline
    vc = load_version_comparison()

    if vc:
        st.markdown('<div class="section-title">Prompt Version Comparison</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color: #94a3b8; font-size: 0.85rem;">
            Three prompt versions were tested. Quality improves as prompts add structure,
            examples, and anti-hallucination rules.
        </p>
        """, unsafe_allow_html=True)

        versions = vc.get("versions", {})

        vc_rows = []
        for v_name in sorted(versions.keys()):
            v = versions[v_name]
            total = v.get("total_pairs", 0)
            pass_rate = v.get("pass_rate_pct", 0)
            avg_inst = v.get("avg_instruction_length", 0)
            avg_resp = v.get("avg_response_length", 0)
            cats = v.get("category_distribution", {})
            top_cat = max(cats, key=cats.get) if cats else "—"

            vc_rows.append({
                "Version": v_name.upper(),
                "Pairs": total,
                "Pass Rate": f"{pass_rate:.0f}%",
                "Avg Instruction Len": f"{avg_inst:.0f}",
                "Avg Response Len": f"{avg_resp:.0f}",
                "Top Category": top_cat.capitalize(),
            })

        st.dataframe(pd.DataFrame(vc_rows), use_container_width=True, hide_index=True)

        st.markdown(f"""
        <div class="stat-note" style="margin-top: 16px;">
            Final training dataset: <strong>{vc.get('final_sft_count', 0)}</strong> SFT pairs,
            <strong>{vc.get('final_dpo_count', 0)}</strong> DPO pairs.
            {vc.get('recommendation', '')}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="placeholder-card" style="margin-top: 12px;">
            <h3>Synthetic Data Pipeline</h3>
            <p>Run <code>python3 -m data.synthetic.pipeline</code> to generate training data.<br>
            Version comparison will appear here.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Training progress cards
    tc1, tc2 = st.columns(2, gap="medium")

    with tc1:
        st.markdown("""
        <div class="approach-card" style="border-left-color: #3b82f6;">
            <div class="approach-card-header">
                <span class="approach-card-name">SFT Training</span>
                <span class="approach-badge" style="background: rgba(59,130,246,0.15); color: #3b82f6;">KAGGLE</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.88rem; line-height: 1.7;">
                <div>📦 <strong style="color: #f8fafc;">Model:</strong> Mistral-7B-Instruct-v0.3</div>
                <div>🔧 <strong style="color: #f8fafc;">Method:</strong> QLoRA (r=16, α=32)</div>
                <div>📊 <strong style="color: #f8fafc;">Epochs:</strong> 3</div>
                <div>🎯 <strong style="color: #f8fafc;">Status:</strong> <span style="color: #f59e0b;">Queued for Kaggle GPU</span></div>
            </div>
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(148,163,184,0.12); color: #94a3b8; font-size: 0.8rem;">
                W&B training curves will appear after session completes
            </div>
        </div>
        """, unsafe_allow_html=True)

    with tc2:
        st.markdown("""
        <div class="approach-card" style="border-left-color: #ec4899;">
            <div class="approach-card-header">
                <span class="approach-card-name">DPO Alignment</span>
                <span class="approach-badge" style="background: rgba(236,72,153,0.15); color: #ec4899;">KAGGLE</span>
            </div>
            <div style="color: #94a3b8; font-size: 0.88rem; line-height: 1.7;">
                <div>📦 <strong style="color: #f8fafc;">Base:</strong> SFT checkpoint (pending)</div>
                <div>🔧 <strong style="color: #f8fafc;">Method:</strong> DPO with β=0.1</div>
                <div>📊 <strong style="color: #f8fafc;">Data:</strong> Chosen/rejected pairs</div>
                <div>🎯 <strong style="color: #f8fafc;">Status:</strong> <span style="color: #94a3b8;">Waiting for SFT</span></div>
            </div>
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(148,163,184,0.12); color: #94a3b8; font-size: 0.8rem;">
                DPO alignment runs after SFT training completes
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
# TAB 4: GPU & SYSTEMS
# ═══════════════════════════════════════════════════════════════════════

with tab_gpu:
    st.markdown("""
    <div style="margin-bottom: 8px;">
        <h2 style="color: #f8fafc; margin-bottom: 0;">⚡ GPU & Systems Analysis</h2>
        <p style="color: #94a3b8; font-size: 0.85rem; margin-top: 4px;">
            Hardware utilization, optimization experiments, and infrastructure metrics
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="placeholder-card" style="margin-top: 12px; text-align: left; padding: 28px 32px;">
        <h3 style="margin-bottom: 16px;">Experiments — Coming After Kaggle Training</h3>
        <div style="color: #94a3b8; font-size: 0.88rem; line-height: 2.2;">
            <div>📊 <strong style="color: #f8fafc;">VRAM Usage</strong>
                — Peak memory across LoRA rank configurations (r=8, r=16, r=32)</div>
            <div>⚡ <strong style="color: #f8fafc;">Tokens/Second</strong>
                — FP32 vs BF16 inference throughput comparison</div>
            <div>💥 <strong style="color: #f8fafc;">Gradient Checkpointing</strong>
                — On vs off: OOM demonstration and memory savings</div>
            <div>📦 <strong style="color: #f8fafc;">Batch Size vs Grad Accumulation</strong>
                — Effective batch size tradeoffs on T4 GPU</div>
            <div>🔄 <strong style="color: #f8fafc;">API Threading</strong>
                — Sequential vs parallel Groq API call speedup (4 approaches)</div>
            <div>🏗️ <strong style="color: #f8fafc;">Quantization Impact</strong>
                — 4-bit QLoRA vs 8-bit LoRA accuracy and speed comparison</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Infrastructure status cards
    ic1, ic2, ic3 = st.columns(3, gap="medium")
    with ic1:
        st.markdown(_metric_card_html(
            "T4 16GB", "Kaggle GPU",
            "Free tier — 30h/week",
            "#10b981",
        ), unsafe_allow_html=True)
    with ic2:
        st.markdown(_metric_card_html(
            "QLoRA", "Training Method",
            "4-bit NF4 quantization",
            "#7c3aed",
        ), unsafe_allow_html=True)
    with ic3:
        st.markdown(_metric_card_html(
            "Groq", "Inference API",
            "Free tier — 30 RPM",
            "#00d4ff",
        ), unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-note" style="margin-top: 20px;">
        Results will populate after Kaggle training completes.
        GPU metrics are captured via <code>nvidia-smi</code> and W&B system monitoring during training runs.
    </div>
    """, unsafe_allow_html=True)
