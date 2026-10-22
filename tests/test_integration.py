"""End-to-end integration: controller execution -> persisted history."""

from src.cloud_simulator import CloudEnvironmentSimulator
from src.memory_engine import RemediationMemoryEngine, normalize_context
from src.healing_controller import ProposedNegativeMemoryController
from src.history_store import IncidentHistoryStore


def test_controller_execution_persisted_and_reloaded(tmp_path):
    db = str(tmp_path / "shared.db")
    sim = CloudEnvironmentSimulator(seed=101)
    engine = RemediationMemoryEngine(db_path=db, in_memory_only=False)
    store = IncidentHistoryStore(db_path=db)
    ctrl = ProposedNegativeMemoryController(
        simulator=sim, memory_engine=engine, theta_sim=0.85, theta_conf=0.40, decay_lambda=0.05
    )

    raw = {"cpu": 42.0, "mem": 60.0, "db_conn": 100.0, "latency_ms": 850.0, "error_rate": 0.40}
    vec = normalize_context(raw)
    record = ctrl.handle_incident(
        incident_id=1,
        condition="DB_CONN_EXHAUSTION",
        raw_telemetry=raw,
        vector=vec,
        candidates=["restart_service", "connection_capacity_remediation", "scale_compute_instances"],
        timestamp=1.0,
    )
    assert store.record_incident(record) is not None

    # Simulated restart: brand-new store/engine instances on the same database.
    reopened_store = IncidentHistoryStore(db_path=db)
    reopened_engine = RemediationMemoryEngine(db_path=db, in_memory_only=False)

    rows = reopened_store.fetch_incidents()
    assert len(rows) == 1
    assert rows[0]["attempted_actions"] == record.tried_actions
    assert rows[0]["recovery_success"] == bool(record.successful_action)
    assert rows[0]["actions_blocked"] == len(record.blocked_actions)

    # Learned memory is also durable.
    assert len(reopened_engine.negative_memory) >= 1


def test_benchmark_is_reproducible(tmp_path):
    import pandas as pd
    from src.evaluate import run_benchmark

    a = run_benchmark(num_incidents=25, output_dir=str(tmp_path / "a"), seed=123, save_plots=False, save_csv=False)
    b = run_benchmark(num_incidents=25, output_dir=str(tmp_path / "b"), seed=123, save_plots=False, save_csv=False)
    # pandas treats NaN as equal in assert_frame_equal.
    pd.testing.assert_frame_equal(a, b)
