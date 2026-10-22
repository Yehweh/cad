"""Tests for the benchmark: dynamic metrics and edge-case handling."""

import pandas as pd
import pytest

from src.evaluate import run_benchmark, generate_benchmark_incident_stream, _safe_rate


def test_safe_rate_zero_denominator():
    assert _safe_rate(5, 0) is None
    assert _safe_rate(0, 0) is None
    assert _safe_rate(1, 2) == 50.0


def test_incident_stream_is_reproducible_and_sized():
    s1 = generate_benchmark_incident_stream(num_incidents=40, seed=7)
    s2 = generate_benchmark_incident_stream(num_incidents=40, seed=7)
    assert len(s1) == 40
    assert [i["condition"] for i in s1] == [i["condition"] for i in s2]
    assert [i["raw_telemetry"] for i in s1] == [i["raw_telemetry"] for i in s2]
    # The scenario mix (counts) is seed-independent; the order is shuffled.
    from collections import Counter
    s3 = generate_benchmark_incident_stream(num_incidents=40, seed=99)
    assert Counter(i["condition"] for i in s1) == Counter(i["condition"] for i in s3)


def test_benchmark_produces_measured_metrics(tmp_path):
    df = run_benchmark(num_incidents=30, output_dir=str(tmp_path), seed=11, save_plots=False, save_csv=True)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    expected_cols = {
        "Controller", "Incidents", "Successful Recoveries", "Recovery Success Rate (%)",
        "Avg Attempts / Incident", "First-Attempt Success Rate (%)",
        "Repeated Failures (count)", "Repeated Unsuccessful Rate (%)",
        "Mean MTTR (seconds)", "Toxic Collateral Events",
        "Prevented Unnecessary Actions", "MTTR Reduction vs Baseline A (%)",
        "Blocked-Action Precision (%)",
    }
    assert expected_cols.issubset(set(df.columns))
    assert (df["Incidents"] == 30).all()

    # Metrics must be derived from results, not hardcoded.
    proposed = df[df["Controller"] == "Proposed (Negative Memory)"].iloc[0]
    assert pd.notna(proposed["Mean MTTR (seconds)"])
    assert pd.notna(proposed["MTTR Reduction vs Baseline A (%)"])

    # CSV was written from real data.
    assert (tmp_path / "evaluation_summary.csv").exists()


def test_benchmark_empty_dataset_does_not_crash(tmp_path):
    df = run_benchmark(num_incidents=0, output_dir=str(tmp_path), seed=1, save_plots=False, save_csv=False)
    assert len(df) == 3
    assert (df["Incidents"] == 0).all()
    # Empty data must not be reported as a 100% success rate.
    assert df["Recovery Success Rate (%)"].isna().all()
    assert df["Mean MTTR (seconds)"].isna().all()
    assert df["MTTR Reduction vs Baseline A (%)"].isna().all()
