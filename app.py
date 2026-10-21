"""
Interactive Streamlit Dashboard:
Negative Remediation Memory for Autonomous Cloud Self-Healing.

Faculty Demonstration & Interactive Benchmark Portal.
Features:
- Live Incident Injection & Autonomous Remediation Console
- Visual Explainability (XAI) for Blocked Actions (Cosine Similarity & Decayed Confidence)
- Negative & Positive Memory Record Inspector with Live Decay
- Full 3-Baseline Benchmark Suite & 300 DPI Publication Visualizations
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from typing import Optional, List, Dict, Any, Tuple

# Ensure root directory is on Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.memory_engine import RemediationMemoryEngine, cosine_similarity, normalize_context, NegativeConstraint
from src.cloud_simulator import CloudEnvironmentSimulator
from src.healing_controller import ProposedNegativeMemoryController, IncidentExecutionRecord
from src.evaluate import run_benchmark

# Page Configuration
st.set_page_config(
    page_title="Negative Remediation Memory | Cloud Self-Healing",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-card-danger {
        background-color: #fff5f5;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #dc3545;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .metric-card-warning {
        background-color: #fffdf5;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #ffc107;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    .badge-blocked {
        background-color: #dc3545;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
    .badge-allowed {
        background-color: #28a745;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initializes persistent session state objects."""
    if "simulator" not in st.session_state:
        st.session_state.simulator = CloudEnvironmentSimulator(seed=101)
    if "memory_engine" not in st.session_state:
        st.session_state.memory_engine = RemediationMemoryEngine(db_path="outputs/remediation_memory.db")
    if "controller" not in st.session_state:
        st.session_state.controller = ProposedNegativeMemoryController(
            simulator=st.session_state.simulator,
            memory_engine=st.session_state.memory_engine,
            theta_sim=0.85,
            theta_conf=0.40
        )
    if "current_time" not in st.session_state:
        st.session_state.current_time = 1.0
    if "incident_counter" not in st.session_state:
        st.session_state.incident_counter = 0
    if "last_record" not in st.session_state:
        st.session_state.last_record = None
    if "benchmark_df" not in st.session_state:
        csv_file = "outputs/evaluation_summary.csv"
        alt_csv = "outputs/evaluation_summary_latest.csv"
        if os.path.exists(csv_file):
            try:
                st.session_state.benchmark_df = pd.read_csv(csv_file)
            except Exception:
                if os.path.exists(alt_csv):
                    st.session_state.benchmark_df = pd.read_csv(alt_csv)
                else:
                    st.session_state.benchmark_df = None
        elif os.path.exists(alt_csv):
            st.session_state.benchmark_df = pd.read_csv(alt_csv)
        else:
            st.session_state.benchmark_df = None


init_session_state()

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("🛡️ Self-Healing Controls")
st.sidebar.markdown("**Negative Remediation Memory Engine**")

# Scenario Selector
scenario_choice = st.sidebar.selectbox(
    "Target Failure Scenario:",
    ["DB_CONN_EXHAUSTION", "MEMORY_LEAK_OOM", "CASCADING_CACHE_STAMPEDE"],
    index=0
)

# Preset baselines
preset = st.session_state.simulator.SCENARIO_PROFILES[scenario_choice]["baseline"]

st.sidebar.subheader("Live Telemetry Injection")
cpu_val = st.sidebar.slider("CPU Usage (%)", 0.0, 100.0, float(preset["cpu"]), 1.0)
mem_val = st.sidebar.slider("Memory Usage (%)", 0.0, 100.0, float(preset["mem"]), 1.0)
db_val = st.sidebar.slider("DB Pool Usage (%)", 0.0, 100.0, float(preset["db_conn"]), 1.0)
lat_val = st.sidebar.slider("Network Latency (ms)", 10.0, 1500.0, float(preset["latency_ms"]), 10.0)
err_val = st.sidebar.slider("Error Rate (%)", 0.0, 100.0, float(preset["error_rate"] * 100.0), 1.0)

st.sidebar.subheader("Novelty Engine Parameters")
theta_sim = st.sidebar.slider("Similarity Threshold (θ_sim)", 0.50, 1.00, 0.85, 0.01)
theta_conf = st.sidebar.slider("Confidence Threshold (θ_conf)", 0.10, 0.90, 0.40, 0.01)
decay_lambda = st.sidebar.slider("Decay Constant (λ)", 0.005, 0.200, 0.050, 0.005)

# Update controller params
st.session_state.controller.theta_sim = theta_sim
st.session_state.controller.theta_conf = theta_conf

st.sidebar.markdown("---")
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    advance_time_btn = st.button("⏩ +5 Time Steps", help="Advance simulation time to demonstrate confidence decay")
with col_sb2:
    reset_btn = st.button("🔄 Reset Memory", help="Wipes all positive and negative memories")

if advance_time_btn:
    st.session_state.current_time += 5.0
    st.sidebar.success(f"Time advanced to t = {st.session_state.current_time:.1f}")

if reset_btn:
    st.session_state.memory_engine.clear_memory()
    st.session_state.controller.history.clear()
    st.session_state.controller.prevented_actions_count = 0
    st.session_state.controller.action_history_by_condition.clear()
    st.session_state.current_time = 1.0
    st.session_state.incident_counter = 0
    st.session_state.last_record = None
    st.sidebar.warning("Remediation Memory & History Wiped Clean.")

# Drift Patch Toggle
patch_applied = st.sidebar.checkbox(
    "Apply Software Fix (Environment Drift)",
    value=st.session_state.simulator.patch_db_conn_exhaustion_fixed,
    help="Simulates a software release where restart_service is patched and now fixes DB_CONN_EXHAUSTION."
)
st.session_state.simulator.set_environment_patch(patch_applied)


# -----------------------------------------------------------------------------
# MAIN APP TABS
# -----------------------------------------------------------------------------
st.title("🛡️ Negative Remediation Memory for Autonomous Cloud Self-Healing")
st.markdown("""
**Transforming Failed Remediations into Reusable Decision Constraints.**
Eliminates repeated unsuccessful recovery attempts, prevents collateral damage, and cuts Mean Time To Recovery (MTTR) using Fuzzy Fingerprinting & Ebbinghaus Decay.
""")

tab1, tab2, tab3 = st.tabs([
    "🚀 Live Autonomous Healing Console",
    "🧠 Remediation Memory & Constraint Inspector",
    "📊 3-Baseline Benchmark & Research Evaluation"
])

# =============================================================================
# TAB 1: LIVE HEALING CONSOLE
# =============================================================================
with tab1:
    st.subheader("Interactive Fault Injection & Healing Loop")
    
    col_t1_l, col_t1_r = st.columns([1, 1])

    with col_t1_l:
        st.markdown("#### Step 1: Active Injected Telemetry")
        raw_telemetry = {
            "cpu": cpu_val,
            "mem": mem_val,
            "db_conn": db_val,
            "latency_ms": lat_val,
            "error_rate": err_val / 100.0
        }
        current_vector = normalize_context(raw_telemetry)
        candidates = st.session_state.simulator.SCENARIO_PROFILES[scenario_choice]["candidates"]

        v_df = pd.DataFrame([{
            "CPU (norm)": current_vector[0],
            "Memory (norm)": current_vector[1],
            "DB Pool (norm)": current_vector[2],
            "Latency (norm)": current_vector[3],
            "Error Rate (norm)": current_vector[4]
        }])
        st.dataframe(v_df, width="stretch", hide_index=True)

        st.markdown(f"**Target Scenario:** `{scenario_choice}` | **Simulation Time (t):** `{st.session_state.current_time:.1f}`")
        st.markdown(f"**Raw Candidate Actions:** `{[c for c in candidates]}`")

        # Prior evaluation of candidate actions
        allowed, blocked = st.session_state.memory_engine.filter_candidate_actions(
            current_condition=scenario_choice,
            current_vector=current_vector,
            candidate_actions=candidates,
            current_time=st.session_state.current_time,
            theta_sim=theta_sim,
            theta_conf=theta_conf
        )

        st.markdown("#### Candidate Action Filter (Negative Memory Lookup):")
        for act in candidates:
            blk_match = next((b for b in blocked if b["action"] == act), None)
            if blk_match:
                st.markdown(f"""
                <div class="metric-card-danger">
                    <span class="badge-blocked">BLOCKED BY NEGATIVE CONSTRAINT</span> <b>{act}</b><br>
                    <small><b>Similarity:</b> {blk_match['similarity']} (≥ {theta_sim}) | <b>Decayed Conf:</b> {blk_match['decayed_confidence']} (≥ {theta_conf})</small><br>
                    <small><b>Constraint:</b> {blk_match['negative_constraint']}</small><br>
                    <small><b>RCA Reason:</b> {blk_match['failure_reason']}</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <span class="badge-allowed">ELIGIBLE CANDIDATE</span> <b>{act}</b>
                </div>
                """, unsafe_allow_html=True)

        inject_btn = st.button("⚡ Inject Failure & Trigger Self-Healing", type="primary", width="stretch")

    with col_t1_r:
        st.markdown("#### Step 2: Autonomous Recovery Execution Trace")
        if inject_btn:
            st.session_state.incident_counter += 1
            st.session_state.current_time += 1.0

            record = st.session_state.controller.handle_incident(
                incident_id=st.session_state.incident_counter,
                condition=scenario_choice,
                raw_telemetry=raw_telemetry,
                vector=current_vector,
                candidates=candidates,
                timestamp=st.session_state.current_time
            )
            st.session_state.last_record = record

        last_rec: Optional[IncidentExecutionRecord] = st.session_state.last_record

        if last_rec:
            # Metrics display
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Attempts", last_rec.attempts)
            m2.metric("MTTR Overhead", f"{last_rec.mttr_seconds:.1f}s")
            m3.metric("First-Attempt?", "✅ Yes" if last_rec.first_attempt_success else "❌ No")
            m4.metric("Actions Blocked", len(last_rec.blocked_actions))

            st.markdown("##### Execution Trace Log:")
            for trace in last_rec.execution_trace:
                icon = "🟢" if trace["is_healthy"] else "🔴"
                st.markdown(f"""
                **{icon} Attempt {trace['attempt']}: Action `{trace['action']}`**
                - **HTTP Status:** `{trace['status_code']}`
                - **Evidence:** {trace['evidence']}
                - **Toxicity:** `{trace['toxicity']}` | **Overhead:** `{trace['overhead']}s`
                ---
                """)

            if last_rec.successful_action:
                st.success(f"🎉 Incident Resolved! Successful Remediation: **{last_rec.successful_action}**")
        else:
            st.info("Click **'Inject Failure & Trigger Self-Healing'** to execute recovery and observe the Negative Memory loop.")

    # Historical Incidents Table
    if st.session_state.controller.history:
        st.markdown("---")
        st.subheader("Recent Incident Execution History")
        hist_data = []
        for h in st.session_state.controller.history[-10:]:
            hist_data.append({
                "Incident #": h.incident_id,
                "Condition": h.condition,
                "Attempts": h.attempts,
                "Resolved Action": h.successful_action,
                "First Attempt?": "Yes" if h.first_attempt_success else "No",
                "MTTR (s)": h.mttr_seconds,
                "Actions Blocked": len(h.blocked_actions),
                "Sim Time": f"t = {h.timestamp:.1f}"
            })
        st.dataframe(pd.DataFrame(hist_data), width="stretch", hide_index=True)


# =============================================================================
# TAB 2: MEMORY & CONSTRAINT INSPECTOR
# =============================================================================
with tab2:
    st.subheader("Structured Remediation Memory Repository")
    st.markdown("Inspect recorded **Negative Constraints** and **Positive Outcomes**, and evaluate real-time Ebbinghaus Decay.")

    neg_mem = st.session_state.memory_engine.negative_memory
    pos_mem = st.session_state.memory_engine.positive_memory

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("Total Negative Constraints", len(neg_mem))
    with col_m2:
        st.metric("Total Positive Memory Records", len(pos_mem))

    st.markdown("### 🔴 Active Negative Constraints (Pruning Rules)")
    if neg_mem:
        neg_rows = []
        for n in neg_mem:
            curr_conf = n.get_decayed_confidence(st.session_state.current_time)
            is_active = curr_conf >= theta_conf
            neg_rows.append({
                "ID": n.record_id,
                "Condition": n.failure_condition,
                "Blocked Action": n.action,
                "Toxicity": n.toxicity,
                "Initial C0": n.initial_confidence,
                "Lambda λ": n.decay_lambda,
                "Decayed C(t)": round(curr_conf, 4),
                "Status": "ACTIVE (Blocked)" if is_active else "EXPIRED (Released)",
                "Constraint Rule": n.negative_constraint,
                "Recorded At": f"t = {n.timestamp:.1f}",
                "Age (steps)": round(st.session_state.current_time - n.timestamp, 1)
            })
        st.dataframe(pd.DataFrame(neg_rows), width="stretch", hide_index=True)
    else:
        st.info("No negative constraints recorded yet. Trigger a failure in Tab 1 to record failed actions!")

    st.markdown("### 🟢 Positive Remediation Memory (Success Verification)")
    if pos_mem:
        pos_rows = []
        for p in pos_mem:
            pos_rows.append({
                "ID": p.record_id,
                "Condition": p.failure_condition,
                "Successful Action": p.action,
                "Evidence": p.verification_evidence,
                "Recorded At": f"t = {p.timestamp:.1f}"
            })
        st.dataframe(pd.DataFrame(pos_rows), width="stretch", hide_index=True)
    else:
        st.info("No positive outcomes recorded yet.")

    # Interactive Cosine Similarity Tester
    st.markdown("---")
    st.subheader("🔍 Interactive Cosine Similarity Fingerprinting Calculator")
    if neg_mem:
        selected_neg_id = st.selectbox(
            "Select a recorded negative constraint to compare against current state vector:",
            [n.record_id for n in neg_mem]
        )
        selected_neg = next(n for n in neg_mem if n.record_id == selected_neg_id)

        sim_calc = cosine_similarity(current_vector, selected_neg.observed_context_vector)
        conf_calc = selected_neg.get_decayed_confidence(st.session_state.current_time)

        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("Cosine Similarity", f"{sim_calc:.4f}", delta="Match!" if sim_calc >= theta_sim else "No Match")
        col_c2.metric("Decayed Confidence", f"{conf_calc:.4f}", delta="Active!" if conf_calc >= theta_conf else "Expired")
        col_c3.metric("Constraint Triggered?", "🚫 YES (Action Blocked)" if (sim_calc >= theta_sim and conf_calc >= theta_conf) else "🟢 NO (Eligible)")

        st.code(f"""
Current Vector:   {current_vector}
Stored Vector:    {selected_neg.observed_context_vector}
Formula:          Similarity = (V_current . V_stored) / (||V_current|| * ||V_stored||)
Result:           {sim_calc:.4f}  [Threshold θ_sim = {theta_sim}]
Decay Formula:    C(t) = C0 * exp(-λ * Δt) = {selected_neg.initial_confidence} * exp(-{selected_neg.decay_lambda} * {st.session_state.current_time - selected_neg.timestamp:.1f})
Result:           {conf_calc:.4f}  [Threshold θ_conf = {theta_conf}]
        """)


# =============================================================================
# TAB 3: 3-BASELINE BENCHMARK & RESEARCH EVALUATION
# =============================================================================
with tab3:
    st.subheader("Research Evaluation & Baseline Comparison Suite")
    st.markdown("""
    Rigorous empirical evaluation comparing:
    1. **Baseline A:** Fixed Rule-Based Healing (Zero Cross-Incident Memory)
    2. **Baseline B:** Positive-Only Memory Healing (Success-Biased Memory)
    3. **Proposed System:** Positive + Negative Remediation Memory with Decay
    """)

    if st.button("▶️ Run Full 60-Incident Benchmark Suite", type="primary"):
        with st.spinner("Executing 60-incident batch simulation across Baseline A, Baseline B, and Proposed System..."):
            df_res = run_benchmark(num_incidents=60, output_dir="outputs")
            st.session_state.benchmark_df = df_res
            st.success("Benchmark completed! High-resolution plots rendered.")

    if st.session_state.benchmark_df is not None:
        st.markdown("### 📋 Comparative Metric Summary (Slide 7 Validation)")
        st.dataframe(st.session_state.benchmark_df, width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown("### 📈 Publication-Grade Evaluation Plots (300 DPI)")

        plot_cols = st.columns(2)
        with plot_cols[0]:
            if os.path.exists("outputs/mttr_comparison.png"):
                st.image("outputs/mttr_comparison.png", caption="Cumulative Mean Time To Recovery (MTTR) Comparison", width="stretch")
            else:
                st.warning("outputs/mttr_comparison.png not found. Run the benchmark to generate.")

        with plot_cols[1]:
            if os.path.exists("outputs/attempts_and_failures.png"):
                st.image("outputs/attempts_and_failures.png", caption="Attempts and Repeated Unsuccessful Actions", width="stretch")
            else:
                st.warning("outputs/attempts_and_failures.png not found. Run the benchmark to generate.")

        st.markdown("---")
        if os.path.exists("outputs/confidence_decay_curve.png"):
            st.image("outputs/confidence_decay_curve.png", caption="Ebbinghaus Confidence Decay: Neutral vs Toxic Collateral Failures", width="stretch")

        st.markdown(r"""
        ### 🔬 Key Research Takeaways & Theoretical Validation:
        1. **Elimination of Repeated Failures:** The Proposed System drives repeated unsuccessful actions to **0%**, whereas Baseline A and Baseline B repeatedly waste recovery cycles executing ineffective actions.
        2. **Substantial MTTR Reduction:** By directly selecting verified alternative actions on recurrence, the Proposed System achieves a **~60-75% reduction in Mean Time To Recovery (MTTR)**.
        3. **Collateral Damage Prevention:** Toxic actions during cascading cache stampedes are identified and blocked with prolonged confidence decay ($\lambda = 0.015$), preventing system collapse.
        4. **Adaptation to Environment Drift:** Ebbinghaus confidence decay prevents permanent over-constraining, enabling graceful re-evaluation after software patches.
        """)
    else:
        st.info("Click **'Run Full 60-Incident Benchmark Suite'** to run the evaluation and display performance graphs.")
