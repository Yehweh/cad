"""
Interactive Streamlit Dashboard:
Negative Remediation Memory for Autonomous Cloud Self-Healing.

Faculty Demonstration & Interactive Benchmark Portal.
Features:
- Live Incident Injection & Autonomous Remediation Console
- Persisted Incident Execution History (SQLite) with filters
- Visual Explainability (XAI) for Blocked Actions (Cosine Similarity & Decayed Confidence)
- Negative & Positive Memory Record Inspector with Live Decay
- Full 3-Baseline Benchmark Suite & 300 DPI Publication Visualizations

The dashboard ships a three-mode appearance selector (System Default / Light /
Dark). See ``src/theme.py`` for how each mode is applied, including the
browser-side bridge that mirrors the choice into Streamlit's native theme and
the documented Streamlit runtime-theming limitation.
"""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Ensure the project root is importable regardless of the current directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.memory_engine import (
    RemediationMemoryEngine,
    cosine_similarity,
    normalize_context,
    NegativeConstraint,
)
from src.cloud_simulator import CloudEnvironmentSimulator
from src.healing_controller import ProposedNegativeMemoryController, IncidentExecutionRecord
from src.history_store import IncidentHistoryStore
from src.evaluate import run_benchmark
from src.paths import OUTPUTS_DIR, get_default_db_path, resolve_path
from src.theme import (
    THEME_OPTIONS,
    THEME_MODES,
    SYSTEM,
    LIGHT,
    DARK,
    build_native_bridge_html,
    build_theme_css,
    normalize_mode,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cad.dashboard")

BENCHMARK_CSV = os.path.join(OUTPUTS_DIR, "evaluation_summary.csv")
BENCHMARK_CSV_ALT = os.path.join(OUTPUTS_DIR, "evaluation_summary_latest.csv")

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Negative Remediation Memory | Cloud Self-Healing",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def init_session_state() -> None:
    """Initializes persistent session state objects."""
    if "simulator" not in st.session_state:
        st.session_state.simulator = CloudEnvironmentSimulator(seed=101)
    if "memory_engine" not in st.session_state:
        st.session_state.memory_engine = RemediationMemoryEngine(db_path=get_default_db_path())
    if "history_store" not in st.session_state:
        st.session_state.history_store = IncidentHistoryStore(db_path=get_default_db_path())
    if "controller" not in st.session_state:
        st.session_state.controller = ProposedNegativeMemoryController(
            simulator=st.session_state.simulator,
            memory_engine=st.session_state.memory_engine,
            theta_sim=0.85,
            theta_conf=0.40,
            decay_lambda=0.05,
        )
    if "current_time" not in st.session_state:
        st.session_state.current_time = 1.0
    if "incident_counter" not in st.session_state:
        st.session_state.incident_counter = st.session_state.history_store.count_incidents()
    if "last_record" not in st.session_state:
        st.session_state.last_record = None
    if "benchmark_df" not in st.session_state:
        st.session_state.benchmark_df = _load_benchmark_csv()


def _load_benchmark_csv() -> Optional[pd.DataFrame]:
    for candidate in (BENCHMARK_CSV, BENCHMARK_CSV_ALT):
        if os.path.exists(candidate):
            try:
                return pd.read_csv(candidate)
            except Exception:
                logger.exception("Failed to read benchmark CSV at %s", candidate)
    return None


def overview_card(label: str, value: str, sub: str, accent: str = "primary") -> str:
    """Renders an overview card; ``accent`` is one of primary/success/warning/danger."""
    accent = accent if accent in {"primary", "success", "warning", "danger"} else "primary"
    return (
        f'<div class="overview-card overview-card--{accent}">'
        f'<div class="overview-label">{label}</div>'
        f'<div class="overview-value">{value}</div>'
        f'<div class="overview-sub">{sub}</div>'
        f"</div>"
    )


def render_theme_selector() -> str:
    """
    Renders the three-mode appearance selector at the top of the sidebar and
    returns the resolved mode id. The choice is persisted both in the widget's
    session state and in the URL query string (so it survives reloads and
    browser interactions).
    """
    labels = [label for _, label in THEME_OPTIONS]
    initial = normalize_mode(st.query_params.get("theme") or SYSTEM)
    try:
        index = THEME_MODES.index(initial)
    except ValueError:
        index = 0

    st.sidebar.markdown("### 🎨 Appearance")
    choice = st.sidebar.radio(
        "Theme",
        options=labels,
        index=index,
        key="theme_selector",
        help=(
            "System Default follows your OS/browser light or dark preference. "
            "Light and Dark override it."
        ),
    )
    mode = next((m for m, label in THEME_OPTIONS if label == choice), SYSTEM)
    if st.query_params.get("theme") != mode:
        st.query_params["theme"] = mode
    st.sidebar.caption(
        "System Default follows your OS setting; Light/Dark override it. "
        "Cards update immediately and native widgets/charts follow via a "
        "one-time theme refresh."
    )
    return mode


def render_constraint_card(action: str, blocked: Optional[Dict[str, Any]], theta_sim: float) -> str:
    if blocked:
        return (
            '<div class="metric-card-danger">'
            '<span class="badge-blocked">BLOCKED BY NEGATIVE CONSTRAINT</span> '
            f"<b>{action}</b><br>"
            f"<small><b>Similarity:</b> {blocked['similarity']} (≥ {theta_sim}) | "
            f"<b>Decayed Conf:</b> {blocked['decayed_confidence']}</small><br>"
            f"<small><b>Constraint:</b> {blocked['negative_constraint']}</small><br>"
            f"<small><b>RCA Reason:</b> {blocked['failure_reason']}</small>"
            "</div>"
        )
    return (
        '<div class="metric-card">'
        '<span class="badge-allowed">ELIGIBLE CANDIDATE</span> '
        f"<b>{action}</b>"
        "</div>"
    )


def active_native_theme() -> Optional[str]:
    """
    Returns Streamlit's *actual* active native theme ("light" or "dark"), or
    ``None`` when it cannot be determined (e.g. bare/unit context).

    ``st.context.theme`` is the supported public API and reflects what the user
    actually sees for Streamlit's native surface, including preferences that
    differ from the OS (a stale stored theme, or a choice made in Streamlit's
    own Settings menu).
    """
    try:
        context_theme = st.context.theme
    except Exception:
        return None
    if isinstance(context_theme, dict):
        value = context_theme.get("type")
    else:
        value = getattr(context_theme, "type", None)
    return value if value in (LIGHT, DARK) else None


def effective_plot_theme(mode: str) -> str:
    """
    Resolves the concrete light/dark palette for server-rendered charts.

    Explicit modes are direct. For System Default we consult the active native
    theme, falling back to light when it is unavailable.
    """
    if mode == DARK:
        return "dark"
    if mode == LIGHT:
        return "light"
    return "dark" if active_native_theme() == DARK else "light"


def _embed_theme_bridge(html: str) -> None:
    """
    Embeds the native-theme bridge in a same-origin iframe that allows scripts.
    Prefers the current ``st.iframe`` API and falls back to the older
    ``st.components.v1.html`` on Streamlit versions without it.
    """
    try:
        # st.iframe requires a positive height; the content is hidden anyway.
        st.iframe(html, height=1)
    except Exception:
        components.html(html, height=0)


init_session_state()

engine: RemediationMemoryEngine = st.session_state.memory_engine
history_store: IncidentHistoryStore = st.session_state.history_store
controller: ProposedNegativeMemoryController = st.session_state.controller

# -----------------------------------------------------------------------------
# THEME SELECTION (top of sidebar) + application
# -----------------------------------------------------------------------------
theme_mode = render_theme_selector()
# For System Default, key the custom-component palette to Streamlit's actual
# active theme so the cards can never disagree with the native surface.
st.markdown(build_theme_css(theme_mode, active_theme=active_native_theme()), unsafe_allow_html=True)
# Best-effort native-theme bridge: mirrors the selection into Streamlit's own
# theme preference so native widgets and charts follow. See src/theme.py.
_embed_theme_bridge(build_native_bridge_html(theme_mode))

if engine.last_error:
    st.sidebar.error(f"Memory persistence warning: {engine.last_error}")
if history_store.last_error:
    st.sidebar.error(f"History persistence warning: {history_store.last_error}")

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.title("🛡️ Self-Healing Controls")
st.sidebar.markdown("**Negative Remediation Memory Engine**")

scenario_choice = st.sidebar.selectbox(
    "Target Failure Scenario:",
    ["DB_CONN_EXHAUSTION", "MEMORY_LEAK_OOM", "CASCADING_CACHE_STAMPEDE"],
    index=0,
)
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
decay_lambda = st.sidebar.slider(
    "Decay Constant (λ)",
    0.005,
    0.200,
    0.050,
    0.005,
    help="Base Ebbinghaus decay rate for newly recorded NEUTRAL constraints.",
)
st.sidebar.caption(
    "**Decay policy:** λ sets the base decay for **new** constraints. Toxic constraints decay "
    "at 0.3×λ to preserve prolonged protection. Constraints already stored keep the λ they were "
    "created with, so historical confidence stays reproducible."
)

controller.theta_sim = theta_sim
controller.theta_conf = theta_conf
controller.decay_lambda = decay_lambda

st.sidebar.markdown("---")
st.sidebar.subheader("Simulation Time")
time_step = st.sidebar.number_input("Steps to advance", min_value=1, max_value=1000, value=5, step=1)
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    advance_time_btn = st.button("⏩ Advance Time", help="Advance simulation time to demonstrate confidence decay")
with col_sb2:
    reset_time_btn = st.button("⏱️ Reset Clock", help="Set simulation time back to t = 1.0")

if advance_time_btn:
    st.session_state.current_time += float(time_step)
    st.sidebar.success(f"Time advanced to t = {st.session_state.current_time:.1f}")
if reset_time_btn:
    st.session_state.current_time = 1.0
    st.sidebar.info("Simulation clock reset to t = 1.0")

patch_applied = st.sidebar.checkbox(
    "Apply Software Fix (Environment Drift)",
    value=st.session_state.simulator.patch_db_conn_exhaustion_fixed,
    help="Simulates a software release where restart_service is patched and now fixes DB_CONN_EXHAUSTION.",
)
st.session_state.simulator.set_environment_patch(patch_applied)

# -----------------------------------------------------------------------------
# Destructive actions: clearly separated, each requires explicit confirmation
# -----------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("⚠️ Destructive Actions")
with st.sidebar.expander("Reset / Delete controls"):
    st.markdown(
        "**Reset Learned Memory** deletes negative constraints and positive outcomes "
        "(what the system has learned). It **preserves** incident execution history and benchmark files.\n\n"
        "**Delete Incident History** removes execution logs only. It **preserves** learned "
        "remediation memory and benchmark files."
    )
    confirm_mem = st.checkbox("I understand: delete learned memory", key="confirm_mem")
    if st.button("🧠 Reset Learned Memory", disabled=not confirm_mem, width="stretch"):
        engine.clear_memory()
        controller.history.clear()
        controller.prevented_actions_count = 0
        controller.action_history_by_condition.clear()
        st.session_state.last_record = None
        st.sidebar.warning("Learned remediation memory cleared. Incident history preserved.")

    confirm_hist = st.checkbox("I understand: delete execution history", key="confirm_hist")
    if st.button("🗑️ Delete Incident History", disabled=not confirm_hist, width="stretch"):
        if history_store.clear_history():
            st.session_state.incident_counter = 0
            st.session_state.last_record = None
            st.sidebar.warning("Incident execution history deleted. Learned memory preserved.")
        else:
            st.sidebar.error("Failed to delete incident history. See logs for details.")

# -----------------------------------------------------------------------------
# MAIN TITLE + OVERVIEW
# -----------------------------------------------------------------------------
st.title("🛡️ Negative Remediation Memory for Autonomous Cloud Self-Healing")
st.markdown(
    """
**Transforming Failed Remediations into Reusable Decision Constraints.**
Reduces repeated unsuccessful recovery attempts, prevents collateral damage, and lowers
Mean Time To Recovery (MTTR) using Fuzzy Fingerprinting & Ebbinghaus Decay.
"""
)

st.subheader("Dashboard Overview")
metrics = history_store.aggregate_metrics()
neg_active = sum(1 for n in engine.negative_memory if n.is_active(st.session_state.current_time, theta_conf))
neg_total = len(engine.negative_memory)

if metrics["total_incidents"] == 0:
    st.info(
        "No incidents have been processed yet. Use the **Live Autonomous Healing Console** "
        "below to inject a failure — metrics will populate from persisted results."
    )
    o1, o2, o3, o4 = st.columns(4)
    o1.markdown(overview_card("Incidents Processed", "0", "No data yet", "primary"), unsafe_allow_html=True)
    o2.markdown(overview_card("Recovery Success", "—", "No data yet", "primary"), unsafe_allow_html=True)
    o3.markdown(overview_card("Mean Recovery Time", "—", "No data yet", "primary"), unsafe_allow_html=True)
    o4.markdown(overview_card("Actions Blocked", "0", "By negative memory", "primary"), unsafe_allow_html=True)
else:
    o1, o2, o3, o4 = st.columns(4)
    success_rate = metrics["success_rate"]
    o1.markdown(
        overview_card(
            "Incidents Processed",
            f"{metrics['total_incidents']}",
            f"{metrics['failed_recoveries']} unresolved · {metrics['total_attempts']} total attempts",
        ),
        unsafe_allow_html=True,
    )
    o2.markdown(
        overview_card(
            "Recovery Success",
            f"{success_rate:.1f}%" if success_rate is not None else "—",
            f"{metrics['successful_recoveries']} of {metrics['total_incidents']} incidents resolved",
            "success",
        ),
        unsafe_allow_html=True,
    )
    mrt = metrics["mean_recovery_time"]
    o3.markdown(
        overview_card(
            "Mean Recovery Time",
            f"{mrt:.1f}s" if mrt is not None else "—",
            "Averaged over successful recoveries",
        ),
        unsafe_allow_html=True,
    )
    o4.markdown(
        overview_card(
            "Actions Blocked",
            f"{metrics['actions_blocked']}",
            f"Across {metrics['incidents_with_blocks']} incidents · {neg_active}/{neg_total} constraints active",
            "warning",
        ),
        unsafe_allow_html=True,
    )

    recent = history_store.fetch_incidents(limit=1)
    if recent:
        r = recent[0]
        st.markdown(
            f'<div class="cad-caption">Most recent incident: <b>{r["incident_type"]}</b> at '
            f'sim t = {r["sim_time"]:.1f} · controller: {r["controller"]} · '
            f'outcome: {"resolved" if r["recovery_success"] else "unresolved"}.</div>',
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# MAIN APP TABS (three-tab structure retained)
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🚀 Live Autonomous Healing Console",
    "🧠 Remediation Memory & Constraint Inspector",
    "📊 3-Baseline Benchmark & Research Evaluation",
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
            "error_rate": err_val / 100.0,
        }
        current_vector = normalize_context(raw_telemetry)
        candidates = st.session_state.simulator.SCENARIO_PROFILES[scenario_choice]["candidates"]

        v_df = pd.DataFrame([{
            "CPU (norm)": current_vector[0],
            "Memory (norm)": current_vector[1],
            "DB Pool (norm)": current_vector[2],
            "Latency (norm)": current_vector[3],
            "Error Rate (norm)": current_vector[4],
        }])
        st.dataframe(v_df, width="stretch", hide_index=True)

        st.markdown(
            f"**Target Scenario:** `{scenario_choice}` | "
            f"**Simulation Time (t):** `{st.session_state.current_time:.1f}`"
        )
        st.markdown(f"**Raw Candidate Actions:** `{list(candidates)}`")

        try:
            allowed, blocked = engine.filter_candidate_actions(
                current_condition=scenario_choice,
                current_vector=current_vector,
                candidate_actions=candidates,
                current_time=st.session_state.current_time,
                theta_sim=theta_sim,
                theta_conf=theta_conf,
            )
        except Exception:
            logger.exception("Candidate filtering failed")
            allowed, blocked = list(candidates), []
            st.error("Could not evaluate negative memory for this telemetry; showing all candidates.")

        st.markdown("#### Candidate Action Filter (Negative Memory Lookup):")
        for act in candidates:
            blk_match = next((b for b in blocked if b["action"] == act), None)
            st.markdown(render_constraint_card(act, blk_match, theta_sim), unsafe_allow_html=True)

        inject_btn = st.button("⚡ Inject Failure & Trigger Self-Healing", type="primary", width="stretch")

    with col_t1_r:
        st.markdown("#### Step 2: Autonomous Recovery Execution Trace")
        if inject_btn:
            st.session_state.incident_counter += 1
            st.session_state.current_time += 1.0
            try:
                record = controller.handle_incident(
                    incident_id=st.session_state.incident_counter,
                    condition=scenario_choice,
                    raw_telemetry=raw_telemetry,
                    vector=current_vector,
                    candidates=candidates,
                    timestamp=st.session_state.current_time,
                )
                st.session_state.last_record = record
                # Persist actual simulation result to the shared SQLite database.
                if history_store.record_incident(record) is None:
                    st.warning("Recovery executed, but its history could not be persisted. See logs.")
            except Exception:
                logger.exception("Incident handling failed")
                st.error("Incident recovery could not be executed due to an internal error. See logs.")

        last_rec: Optional[IncidentExecutionRecord] = st.session_state.last_record

        if last_rec:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Attempts", last_rec.attempts)
            m2.metric("Recovery Time", f"{last_rec.mttr_seconds:.1f}s")
            m3.metric("First Attempt?", "Yes" if last_rec.first_attempt_success else "No")
            m4.metric("Actions Blocked", len(last_rec.blocked_actions))

            if last_rec.successful_action:
                st.success(f"Incident resolved. Successful remediation: **{last_rec.successful_action}**")
            else:
                st.error("Incident recovery failed: no candidate action restored health.")
                st.caption("This is a recovery failure, not an application/database error.")

            st.markdown("##### Execution Trace Log:")
            for trace in last_rec.execution_trace:
                status = "✅ healthy" if trace["is_healthy"] else "❌ failed"
                st.markdown(
                    f"**Attempt {trace['attempt']}: Action `{trace['action']}` — {status}**  \n"
                    f"- **HTTP Status:** `{trace['status_code']}`  \n"
                    f"- **Evidence:** {trace['evidence']}  \n"
                    f"- **Toxicity:** `{trace['toxicity']}` | **Overhead:** `{trace['overhead']}s`"
                )
                st.markdown("---")
        else:
            st.info("Click **'Inject Failure & Trigger Self-Healing'** to execute recovery and observe the Negative Memory loop.")

    # -------------------------------------------------------------------------
    # Persisted Incident Execution History (survives reruns and restarts)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("Persisted Incident Execution History")
    st.caption(
        "Stored in SQLite (`incident_history` table). Persists across browser reruns and "
        "application restarts, independent of learned remediation memory."
    )

    total_hist = history_store.count_incidents()
    if total_hist == 0:
        st.info("No incident execution history recorded yet.")
    else:
        f1, f2, f3, f4 = st.columns([2, 2, 1, 1])
        with f1:
            type_filter = st.multiselect(
                "Incident type", options=history_store.distinct_incident_types(), default=[]
            )
        with f2:
            outcome_filter = st.selectbox("Outcome", ["All", "Resolved", "Unresolved"])
        with f3:
            blocked_filter = st.checkbox("Blocked only")
        with f4:
            limit_filter = st.number_input("Limit", min_value=5, max_value=1000, value=50, step=5)

        success_param = None
        if outcome_filter == "Resolved":
            success_param = True
        elif outcome_filter == "Unresolved":
            success_param = False

        # Apply the type filter in Python since it is a multi-select.
        rows = history_store.fetch_incidents(
            success=success_param,
            blocked_only=blocked_filter,
            limit=int(limit_filter),
        )
        if type_filter:
            rows = [r for r in rows if r["incident_type"] in type_filter]

        if not rows:
            st.info("No incidents match the current filters.")
        else:
            hist_df = pd.DataFrame([{
                "Run ID": r["run_id"][:8],
                "Sim Time": f"t = {r['sim_time']:.1f}",
                "Incident Type": r["incident_type"],
                "Controller": r["controller"],
                "Attempts": r["attempts"],
                "Resolved": "Yes" if r["recovery_success"] else "No",
                "Recovery (s)": round(r["recovery_duration"], 1),
                "Actions Blocked": r["actions_blocked"],
                "Recorded (UTC)": (r["created_at"] or "")[:19],
            } for r in rows])
            st.dataframe(hist_df, width="stretch", hide_index=True)

            selected_run = st.selectbox(
                "Inspect a run:", options=[r["run_id"] for r in rows],
                format_func=lambda x: x[:8],
            )
            detail = next(r for r in rows if r["run_id"] == selected_run)
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**Input telemetry**")
                st.json(detail["telemetry"])
                st.markdown("**Attempted actions**")
                st.json(detail["attempted_actions"])
            with d2:
                st.markdown("**Blocked by memory**")
                st.json(detail["blocked_actions"] or [])
                st.markdown("**Failure / run details**")
                st.json(detail["failure_details"])


# =============================================================================
# TAB 2: MEMORY & CONSTRAINT INSPECTOR
# =============================================================================
with tab2:
    st.subheader("Structured Remediation Memory Repository")
    st.markdown(
        "Inspect recorded **Negative Constraints** and **Positive Outcomes**, and evaluate "
        "real-time Ebbinghaus Decay."
    )

    neg_mem = engine.negative_memory
    pos_mem = engine.positive_memory

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Total Negative Constraints", len(neg_mem))
    col_m2.metric("Active Constraints", neg_active)
    col_m3.metric("Total Positive Memory Records", len(pos_mem))

    st.markdown("### 🔴 Negative Constraints (Pruning Rules)")
    if neg_mem:
        neg_rows = []
        for n in neg_mem:
            info = n.get_decay_info(st.session_state.current_time, theta_conf)
            neg_rows.append({
                "ID": n.record_id,
                "Condition": n.failure_condition,
                "Blocked Action": n.action,
                "Toxicity": n.toxicity,
                "Initial C0": n.initial_confidence,
                "Lambda λ": n.decay_lambda,
                "Age (steps)": info["elapsed"],
                "Decayed C(t)": info["decayed_confidence"],
                "Status": "ACTIVE (blocking)" if info["is_active"] else "EXPIRED (released)",
                "Constraint Rule": n.negative_constraint,
                "Recorded At": f"t = {n.timestamp:.1f}",
            })
        st.dataframe(pd.DataFrame(neg_rows), width="stretch", hide_index=True)

        st.markdown("#### Confidence Decay Trajectory")
        st.caption(
            "Select a constraint to project how its confidence (and therefore its blocking "
            "influence) declines over future time steps. Values use the same decay calculation "
            "as the live blocking decision."
        )
        selected_neg_id = st.selectbox(
            "Constraint", options=[n.record_id for n in neg_mem],
            format_func=lambda rid: next(
                (f"{n.action} @ {n.failure_condition} ({rid})" for n in neg_mem if n.record_id == rid),
                rid,
            ),
            key="decay_traj_select",
        )
        selected_neg = next(n for n in neg_mem if n.record_id == selected_neg_id)
        horizon = st.slider("Projection horizon (steps ahead)", 5, 60, 30, 5)
        t0 = max(0.0, st.session_state.current_time - selected_neg.timestamp)
        times = np.linspace(t0, t0 + horizon, 61)
        confs = [selected_neg.get_decayed_confidence(selected_neg.timestamp + t) for t in times]
        traj_df = pd.DataFrame({
            "Δt from now": np.linspace(0, horizon, 61),
            "Confidence C(t)": confs,
        }).set_index("Δt from now")
        st.line_chart(traj_df, height=260)
        expiry_remaining = None
        for t, c in zip(times, confs):
            if c < theta_conf:
                expiry_remaining = t - t0
                break
        if expiry_remaining is None:
            st.info(f"Constraint remains above θ_conf={theta_conf} for the whole {horizon}-step horizon.")
        elif expiry_remaining <= 0:
            st.warning(f"Constraint is already expired (C(t) < θ_conf={theta_conf}).")
        else:
            st.info(f"Constraint is projected to expire in ~{expiry_remaining:.1f} more steps.")
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
                "Recorded At": f"t = {p.timestamp:.1f}",
            })
        st.dataframe(pd.DataFrame(pos_rows), width="stretch", hide_index=True)
    else:
        st.info("No positive outcomes recorded yet.")

    st.markdown("---")
    st.subheader("🔍 Interactive Cosine Similarity Fingerprinting Calculator")
    if neg_mem:
        selected_neg_id2 = st.selectbox(
            "Select a recorded negative constraint to compare against the current state vector:",
            [n.record_id for n in neg_mem],
            key="sim_calc_select",
        )
        selected_neg2 = next(n for n in neg_mem if n.record_id == selected_neg_id2)

        sim_calc = cosine_similarity(current_vector, selected_neg2.observed_context_vector)
        conf_calc = selected_neg2.get_decayed_confidence(st.session_state.current_time)

        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("Cosine Similarity", f"{sim_calc:.4f}", delta="Match" if sim_calc >= theta_sim else "No match")
        col_c2.metric("Decayed Confidence", f"{conf_calc:.4f}", delta="Active" if conf_calc >= theta_conf else "Expired")
        col_c3.metric(
            "Constraint Triggered?",
            "Yes (action blocked)" if (sim_calc >= theta_sim and conf_calc >= theta_conf) else "No (eligible)",
        )

        st.code(
            f"""
Current Vector:   {current_vector}
Stored Vector:    {selected_neg2.observed_context_vector}
Formula:          Similarity = (V_current . V_stored) / (||V_current|| * ||V_stored||)
Result:           {sim_calc:.4f}  [Threshold theta_sim = {theta_sim}]
Decay Formula:    C(t) = C0 * exp(-lambda * dt) = {selected_neg2.initial_confidence} * exp(-{selected_neg2.decay_lambda} * {st.session_state.current_time - selected_neg2.timestamp:.1f})
Result:           {conf_calc:.4f}  [Threshold theta_conf = {theta_conf}]
        """
        )
    else:
        st.info("Record a negative constraint to use the similarity calculator.")


# =============================================================================
# TAB 3: 3-BASELINE BENCHMARK & RESEARCH EVALUATION
# =============================================================================
with tab3:
    st.subheader("Research Evaluation & Baseline Comparison Suite")
    st.markdown(
        """
    Empirical evaluation comparing three controllers under identical scenarios:
    1. **Baseline A:** Fixed Rule-Based Healing (zero cross-incident memory)
    2. **Baseline B:** Positive-Only Memory Healing (success-biased memory)
    3. **Proposed System:** Positive + Negative Remediation Memory with decay

    All metrics below are **measured** from the actual simulation run. The benchmark uses an
    isolated in-memory memory engine, so it never modifies your live memory or incident history.
    """
    )

    b1, b2 = st.columns([2, 2])
    with b1:
        bench_n = st.number_input("Incidents to simulate", min_value=10, max_value=500, value=60, step=10)
    with b2:
        bench_seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1)

    if st.button("▶️ Run Benchmark Suite", type="primary"):
        try:
            with st.spinner(f"Executing {int(bench_n)}-incident benchmark across 3 controllers..."):
                df_res = run_benchmark(
                    num_incidents=int(bench_n),
                    output_dir=OUTPUTS_DIR,
                    seed=int(bench_seed),
                    save_plots=True,
                    save_csv=True,
                    plot_theme=effective_plot_theme(theme_mode),
                )
            st.session_state.benchmark_df = df_res
            st.success("Benchmark completed. Results and plots refreshed.")
        except Exception:
            logger.exception("Benchmark execution failed")
            st.error("The benchmark could not complete due to an internal error. See logs for details.")

    df = st.session_state.benchmark_df
    if df is not None and not df.empty:
        st.markdown("### 📋 Comparative Metric Summary")
        st.dataframe(df, width="stretch", hide_index=True)

        def _row(controller_name: str) -> Optional[pd.Series]:
            match = df[df["Controller"] == controller_name]
            return match.iloc[0] if not match.empty else None

        proposed = _row("Proposed (Negative Memory)")
        baseline_a = _row("Baseline A (Fixed Rule)")
        baseline_b = _row("Baseline B (Positive-Only)")

        st.markdown("---")
        st.markdown("### 📈 Evaluation Plots (300 DPI)")
        st.caption("Charts are generated from the most recent benchmark run and are measured, not illustrative.")

        plot_cols = st.columns(2)
        mttr_png = os.path.join(OUTPUTS_DIR, "mttr_comparison.png")
        fail_png = os.path.join(OUTPUTS_DIR, "attempts_and_failures.png")
        decay_png = os.path.join(OUTPUTS_DIR, "confidence_decay_curve.png")
        with plot_cols[0]:
            if os.path.exists(mttr_png):
                st.image(mttr_png, caption="Cumulative Mean Time To Recovery (MTTR) Comparison", width="stretch")
            else:
                st.warning("mttr_comparison.png not found. Run the benchmark to generate it.")
        with plot_cols[1]:
            if os.path.exists(fail_png):
                st.image(fail_png, caption="Attempts and Repeated Unsuccessful Actions", width="stretch")
            else:
                st.warning("attempts_and_failures.png not found. Run the benchmark to generate it.")

        if os.path.exists(decay_png):
            st.image(decay_png, caption="Ebbinghaus Confidence Decay: Neutral vs Toxic Collateral Failures", width="stretch")

        st.markdown("### 🔬 Measured Takeaways")
        if proposed is not None and baseline_a is not None:
            takeaways: List[str] = []
            n = int(proposed["Incidents"]) if pd.notna(proposed["Incidents"]) else 0
            takeaways.append(f"- **Sample size:** {n} incidents per controller, fixed random seed (reproducible).")

            rep = proposed.get("Repeated Unsuccessful Rate (%)")
            if pd.notna(rep):
                takeaways.append(f"- **Repeated unsuccessful actions (Proposed):** {rep}% of attempts.")

            red = proposed.get("MTTR Reduction vs Baseline A (%)")
            if pd.notna(red):
                direction = "reduction" if red >= 0 else "increase"
                takeaways.append(
                    f"- **MTTR:** {proposed['Mean MTTR (seconds)']}s vs Baseline A "
                    f"{baseline_a['Mean MTTR (seconds)']}s ({abs(red):.1f}% {direction})."
                )

            sr = proposed.get("Recovery Success Rate (%)")
            if pd.notna(sr):
                takeaways.append(f"- **Recovery success rate (Proposed):** {sr}%.")

            precision = proposed.get("Blocked-Action Precision (%)")
            if pd.notna(precision):
                takeaways.append(
                    f"- **Blocked-action precision:** {precision}% of memory-blocked actions would "
                    "genuinely have failed (no harmful blocks)."
                )
            else:
                takeaways.append("- **Blocked-action precision:** not applicable (no actions were blocked).")

            if baseline_b is not None and pd.notna(baseline_b.get("Recovery Success Rate (%)")):
                takeaways.append(
                    f"- **Baseline B (positive-only)** recovered {baseline_b['Recovery Success Rate (%)']}% "
                    f"of incidents; it cannot remember which actions fail."
                )

            st.markdown("\n".join(takeaways))
        else:
            st.info("Run the benchmark to generate measured takeaways.")

        with st.expander("How these metrics are defined"):
            st.markdown(
                """
- **Incidents:** number of incidents processed by each controller (sample size).
- **Recovery Success Rate:** incidents with a verified healthy post-action check ÷ incidents.
- **Avg Attempts / Incident:** total remediation attempts ÷ incidents.
- **First-Attempt Success Rate:** incidents resolved on the first action ÷ incidents.
- **Repeated Unsuccessful Rate:** attempts that repeat an action already known to fail ÷ total attempts.
- **Mean MTTR:** mean recovery duration (seconds) over successful recoveries.
- **MTTR Reduction vs Baseline A:** (Baseline A MTTR − controller MTTR) ÷ Baseline A MTTR.
- **Blocked-Action Precision:** blocked actions that would truly have failed ÷ blocked actions (n/a if none blocked).
                """
            )
    else:
        st.info("Click **'Run Benchmark Suite'** to run the evaluation and display measured results.")
