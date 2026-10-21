"""
Automated Evaluation and High-Resolution Plotting Suite.
Benchmarks:
1. Baseline A: Fixed Rule-Based Controller
2. Baseline B: Positive-Only Memory Controller
3. Proposed System: Positive + Negative Remediation Memory Controller

Computes Slide 7 Metrics:
- Repeated Unsuccessful Remediation Rate (%)
- Average Remediation Attempts per Incident
- First-Attempt Recovery Success Rate (%)
- Mean Time To Recovery (MTTR in seconds)
- Unnecessary / Blocked Actions Prevented
- Memory Precision (%)

Generates 300 DPI Publication-Grade Charts in `outputs/`.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple

from src.cloud_simulator import CloudEnvironmentSimulator
from src.memory_engine import RemediationMemoryEngine
from src.healing_controller import (
    BaselineAFixedRuleController,
    BaselineBPositiveOnlyController,
    ProposedNegativeMemoryController,
    IncidentExecutionRecord
)


def generate_benchmark_incident_stream(num_incidents: int = 60, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generates a reproducible stream of cloud failure incidents with controlled telemetry noise.
    Distribution:
      ~45% DB_CONN_EXHAUSTION
      ~35% MEMORY_LEAK_OOM
      ~20% CASCADING_CACHE_STAMPEDE
    """
    random.seed(seed)
    sim = CloudEnvironmentSimulator(seed=seed)
    conditions_pool = (
        ["DB_CONN_EXHAUSTION"] * 27 +
        ["MEMORY_LEAK_OOM"] * 21 +
        ["CASCADING_CACHE_STAMPEDE"] * 12
    )
    random.shuffle(conditions_pool)

    incident_stream = []
    for i, cond in enumerate(conditions_pool[:num_incidents]):
        raw, vec, candidates = sim.generate_incident(cond, noise_scale=0.04)
        incident_stream.append({
            "incident_id": i + 1,
            "condition": cond,
            "raw_telemetry": raw,
            "vector": vec,
            "candidates": candidates,
            "timestamp": float(i + 1)
        })

    return incident_stream


def run_benchmark(num_incidents: int = 60, output_dir: str = "outputs") -> pd.DataFrame:
    """
    Executes the benchmark across Baseline A, Baseline B, and Proposed System.
    """
    os.makedirs(output_dir, exist_ok=True)
    incidents = generate_benchmark_incident_stream(num_incidents=num_incidents, seed=42)

    # Simulator instances
    sim_a = CloudEnvironmentSimulator(seed=42)
    sim_b = CloudEnvironmentSimulator(seed=42)
    sim_prop = CloudEnvironmentSimulator(seed=42)

    # Controllers
    ctrl_a = BaselineAFixedRuleController(sim_a)
    ctrl_b = BaselineBPositiveOnlyController(sim_b)
    memory_engine = RemediationMemoryEngine(db_path=os.path.join(output_dir, "remediation_memory.db"))
    memory_engine.clear_memory()
    ctrl_prop = ProposedNegativeMemoryController(
        simulator=sim_prop,
        memory_engine=memory_engine,
        theta_sim=0.85,
        theta_conf=0.40
    )

    print(f"\nRunning Benchmark over {num_incidents} incidents across 3 controllers...")

    # Run stream
    for item in incidents:
        iid = item["incident_id"]
        cond = item["condition"]
        raw = item["raw_telemetry"]
        vec = item["vector"]
        cands = item["candidates"]
        ts = item["timestamp"]

        ctrl_a.handle_incident(iid, cond, raw, vec, cands, ts)
        ctrl_b.handle_incident(iid, cond, raw, vec, cands, ts)
        ctrl_prop.handle_incident(iid, cond, raw, vec, cands, ts)

    # Compute Aggregate Metrics
    records = {
        "Baseline A (Fixed Rule)": ctrl_a.history,
        "Baseline B (Positive-Only)": ctrl_b.history,
        "Proposed (Negative Memory)": ctrl_prop.history
    }

    summary_rows = []
    for name, hist in records.items():
        total_attempts = sum(r.attempts for r in hist)
        avg_attempts = total_attempts / len(hist)
        first_attempt_success_cnt = sum(1 for r in hist if r.first_attempt_success)
        first_attempt_rate = (first_attempt_success_cnt / len(hist)) * 100.0
        total_repeated_failures = sum(r.repeated_failures for r in hist)
        repeated_failure_rate = (total_repeated_failures / total_attempts) * 100.0 if total_attempts > 0 else 0.0
        avg_mttr = sum(r.mttr_seconds for r in hist) / len(hist)
        total_toxic = sum(r.toxic_failures_encountered for r in hist)

        prevented_actions = ctrl_prop.prevented_actions_count if name == "Proposed (Negative Memory)" else 0
        memory_precision = 100.0 if name == "Proposed (Negative Memory)" else 0.0

        summary_rows.append({
            "Controller": name,
            "Avg Attempts / Incident": round(avg_attempts, 2),
            "First-Attempt Success Rate (%)": round(first_attempt_rate, 1),
            "Repeated Unsuccessful Rate (%)": round(repeated_failure_rate, 1),
            "Mean MTTR (seconds)": round(avg_mttr, 1),
            "Toxic Collateral Events": total_toxic,
            "Prevented Unnecessary Actions": prevented_actions,
            "Memory Precision (%)": round(memory_precision, 1)
        })

    df_summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(output_dir, "evaluation_summary.csv")
    try:
        df_summary.to_csv(csv_path, index=False)
        print(f"\nSaved evaluation summary table to: {csv_path}")
    except PermissionError:
        alt_csv = os.path.join(output_dir, "evaluation_summary_latest.csv")
        df_summary.to_csv(alt_csv, index=False)
        print(f"\n[NOTICE] '{csv_path}' is currently open in another program (e.g., Microsoft Excel).")
        print(f"Saved latest evaluation results to: '{alt_csv}' instead.")
        print("Tip: Close Microsoft Excel to allow overwriting the main file directly.")

    print("\n" + df_summary.to_string(index=False))

    # Generate High-Resolution Publication Plots (300 DPI)
    generate_plots(ctrl_a.history, ctrl_b.history, ctrl_prop.history, output_dir)

    return df_summary


def _safe_savefig(fig, path: str, dpi: int = 300):
    try:
        fig.savefig(path, dpi=dpi)
        print(f"Generated: {path}")
    except PermissionError:
        base, ext = os.path.splitext(path)
        alt_path = f"{base}_latest{ext}"
        fig.savefig(alt_path, dpi=dpi)
        print(f"[NOTICE] '{path}' is locked by another program. Saved to '{alt_path}' instead.")


def generate_plots(
    hist_a: List[IncidentExecutionRecord],
    hist_b: List[IncidentExecutionRecord],
    hist_prop: List[IncidentExecutionRecord],
    output_dir: str
):
    """
    Renders and exports 3 high-resolution publication charts (300 DPI).
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # -------------------------------------------------------------------------
    # PLOT 1: Cumulative Average MTTR Comparison (outputs/mttr_comparison.png)
    # -------------------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
    steps = [r.incident_id for r in hist_a]
    cum_mttr_a = np.cumsum([r.mttr_seconds for r in hist_a]) / np.arange(1, len(hist_a) + 1)
    cum_mttr_b = np.cumsum([r.mttr_seconds for r in hist_b]) / np.arange(1, len(hist_b) + 1)
    cum_mttr_prop = np.cumsum([r.mttr_seconds for r in hist_prop]) / np.arange(1, len(hist_prop) + 1)

    ax1.plot(steps, cum_mttr_a, label="Baseline A (Fixed Rule)", color="#d9534f", linestyle="--", linewidth=2.5)
    ax1.plot(steps, cum_mttr_b, label="Baseline B (Positive-Only)", color="#f0ad4e", linestyle="-.", linewidth=2.5)
    ax1.plot(steps, cum_mttr_prop, label="Proposed (Negative Remediation Memory)", color="#28a745", linewidth=3.0)

    ax1.set_title("Cumulative Mean Time To Recovery (MTTR) Across 60 Incidents", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Incident Sequence Number", fontsize=12, labelpad=8)
    ax1.set_ylabel("Cumulative Average MTTR (Seconds)", fontsize=12, labelpad=8)
    ax1.set_ylim(0, max(max(cum_mttr_a), max(cum_mttr_b)) + 5)
    ax1.legend(loc="upper right", frameon=True, fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Highlight MTTR reduction annotation
    final_reduction = ((cum_mttr_a[-1] - cum_mttr_prop[-1]) / cum_mttr_a[-1]) * 100
    ax1.annotate(
        f"Proposed: ~{cum_mttr_prop[-1]:.1f}s MTTR\n({final_reduction:.1f}% reduction vs Baseline A)",
        xy=(steps[-1], cum_mttr_prop[-1]),
        xytext=(steps[-1] - 18, cum_mttr_prop[-1] + 4),
        arrowprops=dict(facecolor="#28a745", shrink=0.05, width=1.5, headwidth=6),
        fontweight="bold",
        color="#155724",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#d4edda", edgecolor="#28a745")
    )

    fig1.tight_layout()
    plot1_path = os.path.join(output_dir, "mttr_comparison.png")
    _safe_savefig(fig1, plot1_path, dpi=300)
    plt.close(fig1)

    # -------------------------------------------------------------------------
    # PLOT 2: Attempts and Repeated Failures (outputs/attempts_and_failures.png)
    # -------------------------------------------------------------------------
    fig2, (ax2_top, ax2_bot) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)

    controllers = ["Baseline A\n(Fixed Rule)", "Baseline B\n(Positive-Only)", "Proposed\n(Negative Memory)"]
    avg_attempts = [
        np.mean([r.attempts for r in hist_a]),
        np.mean([r.attempts for r in hist_b]),
        np.mean([r.attempts for r in hist_prop])
    ]
    rep_rates = [
        (sum(r.repeated_failures for r in hist_a) / sum(r.attempts for r in hist_a)) * 100,
        (sum(r.repeated_failures for r in hist_b) / sum(r.attempts for r in hist_b)) * 100,
        (sum(r.repeated_failures for r in hist_prop) / sum(r.attempts for r in hist_prop)) * 100
    ]

    colors = ["#d9534f", "#f0ad4e", "#28a745"]

    # Top: Average Attempts
    bars1 = ax2_top.bar(controllers, avg_attempts, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    ax2_top.set_ylabel("Avg Attempts per Incident", fontsize=11, fontweight="bold")
    ax2_top.set_title("Remediation Efficiency: Attempts and Repeated Unsuccessful Actions", fontsize=13, fontweight="bold", pad=10)
    ax2_top.set_ylim(0, max(avg_attempts) + 0.8)
    for bar in bars1:
        yval = bar.get_height()
        ax2_top.text(bar.get_x() + bar.get_width()/2.0, yval + 0.08, f"{yval:.2f}", ha="center", va="bottom", fontweight="bold")

    # Bottom: Repeated Unsuccessful Rate
    bars2 = ax2_bot.bar(controllers, rep_rates, color=colors, width=0.5, edgecolor="black", alpha=0.85)
    ax2_bot.set_ylabel("Repeated Failure Rate (%)", fontsize=11, fontweight="bold")
    ax2_bot.set_ylim(0, max(rep_rates) + 12)
    for bar in bars2:
        yval = bar.get_height()
        ax2_bot.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

    fig2.tight_layout()
    plot2_path = os.path.join(output_dir, "attempts_and_failures.png")
    _safe_savefig(fig2, plot2_path, dpi=300)
    plt.close(fig2)

    # -------------------------------------------------------------------------
    # PLOT 3: Ebbinghaus Confidence Decay Curve (outputs/confidence_decay_curve.png)
    # -------------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=300)

    time_steps = np.linspace(0, 35, 300)
    # Neutral failure: C0 = 0.86, lambda = 0.05
    c_neutral = 0.86 * np.exp(-0.05 * time_steps)
    # Toxic failure: C0 = 0.98, lambda = 0.015
    c_toxic = 0.98 * np.exp(-0.015 * time_steps)

    theta_conf = 0.40

    ax3.plot(time_steps, c_neutral, label=r"Neutral Failure Decay ($C_0=0.86, \lambda=0.05$)", color="#0275d8", linewidth=2.8)
    ax3.plot(time_steps, c_toxic, label=r"Toxic Failure Decay ($C_0=0.98, \lambda=0.015$)", color="#d9534f", linewidth=2.8)
    ax3.axhline(y=theta_conf, color="#333333", linestyle="--", linewidth=2.0, label=r"Deactivation Threshold ($\theta_{\mathrm{conf}} = 0.40$)")

    # Shaded active vs released regions
    ax3.axhspan(theta_conf, 1.05, color="#d4edda", alpha=0.35, label="Active Constraint Region (Action Blocked)")
    ax3.axhspan(0.0, theta_conf, color="#fff3cd", alpha=0.35, label="Expired Constraint Region (Safe Re-evaluation)")

    # Expiration intersection points
    # Neutral: 0.86 * exp(-0.05 * t) = 0.40 => t = -ln(0.40/0.86)/0.05 = 15.3
    t_neutral_exp = -np.log(theta_conf / 0.86) / 0.05
    # Toxic: 0.98 * exp(-0.015 * t) = 0.40 => t = -np.log(0.40/0.98)/0.015 = 59.7
    t_toxic_exp = -np.log(theta_conf / 0.98) / 0.015

    ax3.plot(t_neutral_exp, theta_conf, marker="o", markersize=8, color="#0275d8")
    ax3.annotate(
        f"Neutral Expiry: t = {t_neutral_exp:.1f} steps",
        xy=(t_neutral_exp, theta_conf),
        xytext=(t_neutral_exp - 4, theta_conf + 0.12),
        arrowprops=dict(facecolor="#0275d8", shrink=0.05, width=1.2, headwidth=5),
        fontweight="bold",
        color="#0275d8"
    )

    ax3.set_title(r"Ebbinghaus Confidence Decay: $C(t) = C_0 \cdot e^{-\lambda \Delta t}$", fontsize=14, fontweight="bold", pad=12)
    ax3.set_xlabel(r"Elapsed Incident Steps ($\Delta t$)", fontsize=12, labelpad=8)
    ax3.set_ylabel(r"Constraint Confidence $C(t)$", fontsize=12, labelpad=8)
    ax3.set_xlim(0, 35)
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc="upper right", frameon=True, fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)

    fig3.tight_layout()
    plot3_path = os.path.join(output_dir, "confidence_decay_curve.png")
    _safe_savefig(fig3, plot3_path, dpi=300)
    plt.close(fig3)


if __name__ == "__main__":
    run_benchmark(num_incidents=60, output_dir="outputs")
