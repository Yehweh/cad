"""
Automated Evaluation and High-Resolution Plotting Suite.
Benchmarks:
1. Baseline A: Fixed Rule-Based Controller
2. Baseline B: Positive-Only Memory Controller
3. Proposed System: Positive + Negative Remediation Memory Controller

All reported metrics are derived from the actual simulation results produced
by this run. Nothing is hardcoded. Controlled random seeds make the benchmark
reproducible.

The benchmark uses an *in-memory* remediation memory engine so it can never
corrupt or clear the live dashboard's persisted memory or incident history.

Generates 300 DPI charts in ``outputs/``.
"""

import logging
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.cloud_simulator import CloudEnvironmentSimulator
from src.memory_engine import RemediationMemoryEngine
from src.healing_controller import (
    BaselineAFixedRuleController,
    BaselineBPositiveOnlyController,
    ProposedNegativeMemoryController,
    IncidentExecutionRecord,
)
from src.paths import OUTPUTS_DIR, resolve_path

logger = logging.getLogger(__name__)

# Scenario mix used by the incident stream generator.
SCENARIO_MIX = {
    "DB_CONN_EXHAUSTION": 0.45,
    "MEMORY_LEAK_OOM": 0.35,
    "CASCADING_CACHE_STAMPEDE": 0.20,
}


def generate_benchmark_incident_stream(num_incidents: int = 60, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generates a reproducible stream of cloud failure incidents with controlled
    telemetry noise. Scenario proportions follow ``SCENARIO_MIX`` and scale to
    any ``num_incidents`` value.
    """
    if num_incidents <= 0:
        return []

    rng = random.Random(seed)
    sim = CloudEnvironmentSimulator(seed=seed)

    conditions_pool: List[str] = []
    for condition, fraction in SCENARIO_MIX.items():
        conditions_pool.extend([condition] * int(round(fraction * num_incidents)))
    # Correct for rounding so the pool length matches exactly.
    while len(conditions_pool) < num_incidents:
        conditions_pool.append("DB_CONN_EXHAUSTION")
    conditions_pool = conditions_pool[:num_incidents]
    rng.shuffle(conditions_pool)

    incident_stream: List[Dict[str, Any]] = []
    for i, cond in enumerate(conditions_pool):
        raw, vec, candidates = sim.generate_incident(cond, noise_scale=0.04)
        incident_stream.append({
            "incident_id": i + 1,
            "condition": cond,
            "raw_telemetry": raw,
            "vector": vec,
            "candidates": candidates,
            "timestamp": float(i + 1),
        })

    return incident_stream


def _safe_rate(numerator: float, denominator: float) -> Optional[float]:
    """Returns a percentage, or ``None`` when the denominator is zero/empty."""
    if not denominator:
        return None
    return (numerator / denominator) * 100.0


def _aggregate_controller(
    name: str,
    hist: List[IncidentExecutionRecord],
    baseline_mttr: Optional[float] = None,
    prevented_actions: int = 0,
    blocked_precision: Optional[float] = None,
) -> Dict[str, Any]:
    """Builds a single summary row from real controller history."""
    total_incidents = len(hist)
    successful = sum(1 for r in hist if r.successful_action)
    total_attempts = sum(r.attempts for r in hist)
    first_attempt = sum(1 for r in hist if r.first_attempt_success)
    repeated_failures = sum(r.repeated_failures for r in hist)
    mean_mttr = (sum(r.mttr_seconds for r in hist) / total_incidents) if total_incidents else None

    mttr_reduction = None
    if baseline_mttr and mean_mttr is not None and baseline_mttr > 0:
        mttr_reduction = ((baseline_mttr - mean_mttr) / baseline_mttr) * 100.0

    return {
        "Controller": name,
        "Incidents": total_incidents,
        "Successful Recoveries": successful,
        "Recovery Success Rate (%)": (
            round(_safe_rate(successful, total_incidents), 1)
            if total_incidents else None
        ),
        "Avg Attempts / Incident": (
            round(total_attempts / total_incidents, 2) if total_incidents else None
        ),
        "First-Attempt Success Rate (%)": (
            round(_safe_rate(first_attempt, total_incidents), 1)
            if total_incidents else None
        ),
        "Repeated Failures (count)": repeated_failures,
        "Repeated Unsuccessful Rate (%)": (
            round(_safe_rate(repeated_failures, total_attempts), 1)
            if total_attempts else None
        ),
        "Mean MTTR (seconds)": round(mean_mttr, 1) if mean_mttr is not None else None,
        "Toxic Collateral Events": sum(r.toxic_failures_encountered for r in hist),
        "Prevented Unnecessary Actions": prevented_actions,
        "MTTR Reduction vs Baseline A (%)": (
            round(mttr_reduction, 1) if mttr_reduction is not None else None
        ),
        "Blocked-Action Precision (%)": (
            round(blocked_precision, 1) if blocked_precision is not None else None
        ),
    }


def _blocked_action_precision(
    hist: List[IncidentExecutionRecord],
    probe: CloudEnvironmentSimulator,
) -> Optional[float]:
    """
    Fraction of memory-blocked actions that would genuinely have failed.
    Returns ``None`` when nothing was blocked (undefined, not 100%).
    """
    total = 0
    correct = 0
    for record in hist:
        for blocked in record.blocked_actions:
            action = blocked.get("action")
            if not action:
                continue
            total += 1
            if not probe.is_action_effective(record.condition, action):
                correct += 1
    return _safe_rate(correct, total)


def run_benchmark(
    num_incidents: int = 60,
    output_dir: str = OUTPUTS_DIR,
    seed: int = 42,
    save_plots: bool = True,
    save_csv: bool = True,
    plot_theme: str = "light",
) -> pd.DataFrame:
    """
    Executes the benchmark across Baseline A, Baseline B, and the Proposed
    System, returning a DataFrame of measured metrics.

    The benchmark is isolated: it uses an in-memory remediation memory engine
    and never touches the live database.
    """
    output_dir = resolve_path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    incidents = generate_benchmark_incident_stream(num_incidents=num_incidents, seed=seed)

    sim_a = CloudEnvironmentSimulator(seed=seed)
    sim_b = CloudEnvironmentSimulator(seed=seed)
    sim_prop = CloudEnvironmentSimulator(seed=seed)

    ctrl_a = BaselineAFixedRuleController(sim_a)
    ctrl_b = BaselineBPositiveOnlyController(sim_b)
    # In-memory only: the benchmark must not read/write the live application DB.
    memory_engine = RemediationMemoryEngine(in_memory_only=True)
    ctrl_prop = ProposedNegativeMemoryController(
        simulator=sim_prop,
        memory_engine=memory_engine,
        theta_sim=0.85,
        theta_conf=0.40,
        decay_lambda=0.05,
    )

    logger.info("Running benchmark over %d incidents across 3 controllers...", num_incidents)

    for item in incidents:
        ctrl_a.handle_incident(
            item["incident_id"], item["condition"], item["raw_telemetry"],
            item["vector"], item["candidates"], item["timestamp"],
        )
        ctrl_b.handle_incident(
            item["incident_id"], item["condition"], item["raw_telemetry"],
            item["vector"], item["candidates"], item["timestamp"],
        )
        ctrl_prop.handle_incident(
            item["incident_id"], item["condition"], item["raw_telemetry"],
            item["vector"], item["candidates"], item["timestamp"],
        )

    # Baseline A MTTR is the explicit reference for all MTTR reductions.
    baseline_a_mttr = (
        sum(r.mttr_seconds for r in ctrl_a.history) / len(ctrl_a.history)
        if ctrl_a.history else None
    )

    probe = CloudEnvironmentSimulator(seed=seed)
    precision = _blocked_action_precision(ctrl_prop.history, probe)

    summary_rows = [
        _aggregate_controller("Baseline A (Fixed Rule)", ctrl_a.history, baseline_a_mttr),
        _aggregate_controller("Baseline B (Positive-Only)", ctrl_b.history, baseline_a_mttr),
        _aggregate_controller(
            "Proposed (Negative Memory)",
            ctrl_prop.history,
            baseline_a_mttr,
            prevented_actions=ctrl_prop.prevented_actions_count,
            blocked_precision=precision,
        ),
    ]

    df_summary = pd.DataFrame(summary_rows)

    if save_csv:
        csv_path = os.path.join(output_dir, "evaluation_summary.csv")
        try:
            df_summary.to_csv(csv_path, index=False)
            logger.info("Saved evaluation summary table to: %s", csv_path)
        except PermissionError:
            alt_csv = os.path.join(output_dir, "evaluation_summary_latest.csv")
            df_summary.to_csv(alt_csv, index=False)
            logger.warning("'%s' locked; saved to '%s' instead.", csv_path, alt_csv)

    if save_plots:
        try:
            generate_plots(
                ctrl_a.history, ctrl_b.history, ctrl_prop.history, output_dir,
                plot_theme=plot_theme,
            )
        except Exception:
            logger.exception("Plot generation failed; summary metrics are still valid.")

    return df_summary


def _safe_savefig(fig, path: str, dpi: int = 300) -> None:
    try:
        fig.savefig(path, dpi=dpi)
        logger.info("Generated: %s", path)
    except PermissionError:
        base, ext = os.path.splitext(path)
        alt_path = f"{base}_latest{ext}"
        fig.savefig(alt_path, dpi=dpi)
        logger.warning("'%s' locked; saved to '%s' instead.", path, alt_path)


def _apply_plot_theme(theme: str) -> str:
    """
    Applies a light/dark matplotlib theme and returns the colors to use for
    foreground elements. Charts are static PNGs rendered server-side, so their
    theme is fixed at generation time (see README).
    """
    dark = str(theme).lower() == "dark"
    preferred = "seaborn-v0_8-darkgrid" if dark else "seaborn-v0_8-whitegrid"
    plt.style.use(preferred if preferred in plt.style.available else "default")
    fg = "#e8eaed" if dark else "#1f2430"
    bg = "#0e1117" if dark else "#ffffff"
    grid = "#3a4150" if dark else "#cccccc"
    plt.rcParams.update({
        "figure.facecolor": bg,
        "savefig.facecolor": bg,
        "axes.facecolor": bg,
        "axes.edgecolor": fg,
        "axes.labelcolor": fg,
        "text.color": fg,
        "xtick.color": fg,
        "ytick.color": fg,
        "grid.color": grid,
        "legend.facecolor": bg,
        "legend.edgecolor": grid,
    })
    return fg


def generate_plots(
    hist_a: List[IncidentExecutionRecord],
    hist_b: List[IncidentExecutionRecord],
    hist_prop: List[IncidentExecutionRecord],
    output_dir: str,
    plot_theme: str = "light",
) -> None:
    """Renders and exports publication charts (300 DPI) from real results."""
    fg = _apply_plot_theme(plot_theme)
    dark = str(plot_theme).lower() == "dark"
    n_incidents = len(hist_a)

    # -------------------------------------------------------------------------
    # PLOT 1: Cumulative Average MTTR Comparison
    # -------------------------------------------------------------------------
    if n_incidents:
        fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
        steps = [r.incident_id for r in hist_a]
        denom = np.arange(1, n_incidents + 1)

        cum_mttr_a = np.cumsum([r.mttr_seconds for r in hist_a]) / denom
        cum_mttr_b = np.cumsum([r.mttr_seconds for r in hist_b]) / np.arange(1, len(hist_b) + 1)
        cum_mttr_prop = np.cumsum([r.mttr_seconds for r in hist_prop]) / np.arange(1, len(hist_prop) + 1)

        ax1.plot(steps, cum_mttr_a, label="Baseline A (Fixed Rule)", color="#d9534f", linestyle="--", linewidth=2.5)
        ax1.plot(steps, cum_mttr_b, label="Baseline B (Positive-Only)", color="#f0ad4e", linestyle="-.", linewidth=2.5)
        ax1.plot(steps, cum_mttr_prop, label="Proposed (Negative Remediation Memory)", color="#28a745", linewidth=3.0)

        ax1.set_title(f"Cumulative Mean Time To Recovery (MTTR) Across {n_incidents} Incidents", fontsize=14, fontweight="bold", pad=12)
        ax1.set_xlabel("Incident Sequence Number", fontsize=12, labelpad=8)
        ax1.set_ylabel("Cumulative Average MTTR (Seconds)", fontsize=12, labelpad=8)
        ax1.set_ylim(0, max(max(cum_mttr_a), max(cum_mttr_b)) + 5)
        ax1.legend(loc="upper right", frameon=True, fontsize=11)
        ax1.grid(True, linestyle=":", alpha=0.6)

        if cum_mttr_a[-1] > 0:
            final_reduction = ((cum_mttr_a[-1] - cum_mttr_prop[-1]) / cum_mttr_a[-1]) * 100
            ax1.annotate(
                f"Proposed: {cum_mttr_prop[-1]:.1f}s MTTR\n({final_reduction:.1f}% vs Baseline A)",
                xy=(steps[-1], cum_mttr_prop[-1]),
                xytext=(max(steps[-1] - 18, 1), cum_mttr_prop[-1] + 4),
                arrowprops=dict(facecolor="#28a745", shrink=0.05, width=1.5, headwidth=6),
                fontweight="bold",
                color="#155724",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#d4edda", edgecolor="#28a745"),
            )

        fig1.tight_layout()
        _safe_savefig(fig1, os.path.join(output_dir, "mttr_comparison.png"), dpi=300)
        plt.close(fig1)

    # -------------------------------------------------------------------------
    # PLOT 2: Attempts and Repeated Failures
    # -------------------------------------------------------------------------
    if n_incidents:
        fig2, (ax2_top, ax2_bot) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)
        controllers = ["Baseline A\n(Fixed Rule)", "Baseline B\n(Positive-Only)", "Proposed\n(Negative Memory)"]
        histograms = [hist_a, hist_b, hist_prop]

        avg_attempts = [
            (sum(r.attempts for r in h) / len(h)) if h else 0.0 for h in histograms
        ]
        rep_rates = [
            ((sum(r.repeated_failures for r in h) / sum(r.attempts for r in h)) * 100) if sum(r.attempts for r in h) else 0.0
            for h in histograms
        ]
        colors = ["#d9534f", "#f0ad4e", "#28a745"]

        bars1 = ax2_top.bar(controllers, avg_attempts, color=colors, width=0.5, edgecolor="black", alpha=0.85)
        ax2_top.set_ylabel("Avg Attempts per Incident", fontsize=11, fontweight="bold")
        ax2_top.set_title("Remediation Efficiency: Attempts and Repeated Unsuccessful Actions", fontsize=13, fontweight="bold", pad=10)
        ax2_top.set_ylim(0, max(avg_attempts + [0]) + 0.8)
        for bar in bars1:
            yval = bar.get_height()
            ax2_top.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.08, f"{yval:.2f}", ha="center", va="bottom", fontweight="bold")

        bars2 = ax2_bot.bar(controllers, rep_rates, color=colors, width=0.5, edgecolor="black", alpha=0.85)
        ax2_bot.set_ylabel("Repeated Failure Rate (%)", fontsize=11, fontweight="bold")
        ax2_bot.set_ylim(0, max(rep_rates + [0]) + 12)
        for bar in bars2:
            yval = bar.get_height()
            ax2_bot.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")

        fig2.tight_layout()
        _safe_savefig(fig2, os.path.join(output_dir, "attempts_and_failures.png"), dpi=300)
        plt.close(fig2)

    # -------------------------------------------------------------------------
    # PLOT 3: Ebbinghaus Confidence Decay Curve
    # -------------------------------------------------------------------------
    from src.memory_engine import DEFAULT_NEUTRAL_CONFIDENCE, DEFAULT_NEUTRAL_DECAY, DEFAULT_TOXIC_CONFIDENCE, DEFAULT_TOXIC_DECAY

    fig3, ax3 = plt.subplots(figsize=(10, 6), dpi=300)
    time_steps = np.linspace(0, 35, 300)
    c_neutral = DEFAULT_NEUTRAL_CONFIDENCE * np.exp(-DEFAULT_NEUTRAL_DECAY * time_steps)
    c_toxic = DEFAULT_TOXIC_CONFIDENCE * np.exp(-DEFAULT_TOXIC_DECAY * time_steps)
    theta_conf = 0.40

    ax3.plot(time_steps, c_neutral, label=rf"Neutral Failure Decay ($C_0={DEFAULT_NEUTRAL_CONFIDENCE}, \lambda={DEFAULT_NEUTRAL_DECAY}$)", color="#0275d8", linewidth=2.8)
    ax3.plot(time_steps, c_toxic, label=rf"Toxic Failure Decay ($C_0={DEFAULT_TOXIC_CONFIDENCE}, \lambda={DEFAULT_TOXIC_DECAY}$)", color="#d9534f", linewidth=2.8)
    ax3.axhline(y=theta_conf, color=("#cccccc" if dark else "#333333"), linestyle="--", linewidth=2.0, label=rf"Deactivation Threshold ($\theta_{{\mathrm{{conf}}}} = {theta_conf}$)")

    ax3.axhspan(theta_conf, 1.05, color="#d4edda", alpha=0.35, label="Active Constraint Region (Action Blocked)")
    ax3.axhspan(0.0, theta_conf, color="#fff3cd", alpha=0.35, label="Expired Constraint Region (Safe Re-evaluation)")

    t_neutral_exp = -np.log(theta_conf / DEFAULT_NEUTRAL_CONFIDENCE) / DEFAULT_NEUTRAL_DECAY
    t_toxic_exp = -np.log(theta_conf / DEFAULT_TOXIC_CONFIDENCE) / DEFAULT_TOXIC_DECAY

    ax3.plot(t_neutral_exp, theta_conf, marker="o", markersize=8, color="#0275d8")
    ax3.annotate(
        f"Neutral Expiry: t = {t_neutral_exp:.1f} steps",
        xy=(t_neutral_exp, theta_conf),
        xytext=(max(t_neutral_exp - 4, 0), theta_conf + 0.12),
        arrowprops=dict(facecolor="#0275d8", shrink=0.05, width=1.2, headwidth=5),
        fontweight="bold",
        color="#0275d8",
    )

    ax3.set_title(r"Ebbinghaus Confidence Decay: $C(t) = C_0 \cdot e^{-\lambda \Delta t}$", fontsize=14, fontweight="bold", pad=12)
    ax3.set_xlabel(r"Elapsed Incident Steps ($\Delta t$)", fontsize=12, labelpad=8)
    ax3.set_ylabel(r"Constraint Confidence $C(t)$", fontsize=12, labelpad=8)
    ax3.set_xlim(0, 35)
    ax3.set_ylim(0, 1.05)
    ax3.legend(loc="upper right", frameon=True, fontsize=10)
    ax3.grid(True, linestyle=":", alpha=0.6)

    fig3.tight_layout()
    _safe_savefig(fig3, os.path.join(output_dir, "confidence_decay_curve.png"), dpi=300)
    plt.close(fig3)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_benchmark(num_incidents=60, output_dir=OUTPUTS_DIR)
