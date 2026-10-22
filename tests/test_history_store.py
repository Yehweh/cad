"""Tests for incident history persistence and separation from memory."""

import sqlite3

from src.memory_engine import RemediationMemoryEngine, normalize_context
from src.history_store import IncidentHistoryStore


class FakeRecord:
    """Lightweight stand-in for IncidentExecutionRecord."""

    def __init__(self, **overrides):
        self.incident_id = overrides.get("incident_id", 1)
        self.condition = overrides.get("condition", "DB_CONN_EXHAUSTION")
        self.controller_name = overrides.get("controller_name", "Proposed (Negative+Positive Memory)")
        self.telemetry_raw = overrides.get("telemetry_raw", {"cpu": 42.0, "mem": 60.0})
        self.telemetry_vector = overrides.get("telemetry_vector", [0.4, 0.6, 1.0, 0.85, 0.4])
        self.attempts = overrides.get("attempts", 2)
        self.tried_actions = overrides.get("tried_actions", ["restart_service", "connection_capacity_remediation"])
        self.successful_action = overrides.get("successful_action", "connection_capacity_remediation")
        self.blocked_actions = overrides.get("blocked_actions", [{"action": "restart_service"}])
        self.first_attempt_success = overrides.get("first_attempt_success", False)
        self.repeated_failures = overrides.get("repeated_failures", 0)
        self.mttr_seconds = overrides.get("mttr_seconds", 20.0)
        self.toxic_failures_encountered = overrides.get("toxic_failures_encountered", 0)
        self.timestamp = overrides.get("timestamp", 1.0)
        self.execution_trace = overrides.get("execution_trace", [
            {"attempt": 1, "action": "restart_service", "is_healthy": False, "status_code": 500,
             "evidence": "fail", "toxicity": "NEUTRAL", "overhead": 15.0},
        ])


def test_history_persists_across_restart(tmp_path):
    db = str(tmp_path / "hist.db")
    store = IncidentHistoryStore(db_path=db)
    run_id = store.record_incident(FakeRecord())
    assert run_id is not None
    assert store.count_incidents() == 1

    # Simulate an application restart by creating a fresh store instance.
    reopened = IncidentHistoryStore(db_path=db)
    rows = reopened.fetch_incidents()
    assert len(rows) == 1
    assert rows[0]["run_id"] == run_id
    assert rows[0]["incident_type"] == "DB_CONN_EXHAUSTION"
    assert rows[0]["recovery_success"] is True
    assert rows[0]["actions_blocked"] == 1
    assert rows[0]["telemetry"] == {"cpu": 42.0, "mem": 60.0}


def test_history_filters(tmp_path):
    db = str(tmp_path / "hist.db")
    store = IncidentHistoryStore(db_path=db)
    store.record_incident(FakeRecord(incident_id=1, condition="DB_CONN_EXHAUSTION", timestamp=1.0))
    store.record_incident(FakeRecord(
        incident_id=2, condition="MEMORY_LEAK_OOM", timestamp=2.0,
        successful_action="", blocked_actions=[], recovery_success=False,
    ))
    assert len(store.fetch_incidents(incident_type="DB_CONN_EXHAUSTION")) == 1
    assert len(store.fetch_incidents(success=False)) == 1
    assert len(store.fetch_incidents(blocked_only=True)) == 1
    assert set(store.distinct_incident_types()) == {"DB_CONN_EXHAUSTION", "MEMORY_LEAK_OOM"}


def test_aggregate_metrics_empty_dataset(tmp_path):
    store = IncidentHistoryStore(db_path=str(tmp_path / "hist.db"))
    metrics = store.aggregate_metrics()
    assert metrics["total_incidents"] == 0
    assert metrics["success_rate"] is None  # never 100% for empty data
    assert metrics["mean_recovery_time"] is None


def test_aggregate_metrics_values(tmp_path):
    store = IncidentHistoryStore(db_path=str(tmp_path / "hist.db"))
    store.record_incident(FakeRecord(incident_id=1, successful_action="a", mttr_seconds=5.0, timestamp=1.0))
    store.record_incident(FakeRecord(incident_id=2, successful_action="b", mttr_seconds=15.0, timestamp=2.0))
    store.record_incident(FakeRecord(
        incident_id=3, successful_action="", blocked_actions=[], timestamp=3.0,
    ))
    metrics = store.aggregate_metrics()
    assert metrics["total_incidents"] == 3
    assert metrics["successful_recoveries"] == 2
    assert metrics["success_rate"] == 2 / 3 * 100
    assert metrics["mean_recovery_time"] == 10.0  # (5 + 15) / 2, failures excluded
    assert metrics["actions_blocked"] == 2


def test_memory_reset_does_not_delete_history_and_vice_versa(tmp_path):
    db = str(tmp_path / "shared.db")
    engine = RemediationMemoryEngine(db_path=db, in_memory_only=False)
    store = IncidentHistoryStore(db_path=db)

    engine.record_negative_constraint(
        failure_condition="DB_CONN_EXHAUSTION",
        observed_context_raw={"cpu": 42.0, "mem": 60.0, "db_conn": 100.0, "latency_ms": 850.0, "error_rate": 0.4},
        observed_context_vector=normalize_context(
            {"cpu": 42.0, "mem": 60.0, "db_conn": 100.0, "latency_ms": 850.0, "error_rate": 0.4}
        ),
        action="restart_service",
        verification_evidence="HTTP 500",
        failure_reason="pool saturated",
        negative_constraint="avoid restart",
        timestamp=0.0,
    )
    store.record_incident(FakeRecord())

    # Clearing memory preserves history.
    engine.clear_memory()
    assert engine.negative_memory == []
    assert store.count_incidents() == 1

    # Re-add memory, then clear history: memory must be preserved.
    engine.record_negative_constraint(
        failure_condition="DB_CONN_EXHAUSTION",
        observed_context_raw={},
        observed_context_vector=[0.1] * 5,
        action="restart_service",
        verification_evidence="e",
        failure_reason="r",
        negative_constraint="c",
        timestamp=0.0,
    )
    assert store.clear_history() is True
    assert store.count_incidents() == 0
    reloaded = RemediationMemoryEngine(db_path=db, in_memory_only=False)
    assert len(reloaded.negative_memory) == 1


def test_history_schema_created_with_indexes(tmp_path):
    db = str(tmp_path / "hist.db")
    IncidentHistoryStore(db_path=db)
    conn = sqlite3.connect(db)
    indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
    conn.close()
    assert "idx_history_timestamp" in indexes
    assert "idx_history_incident_type" in indexes


def test_history_store_invalid_path_is_safe(tmp_path):
    store = IncidentHistoryStore(db_path=str(tmp_path))  # a directory
    assert store.last_error is not None
    assert store.aggregate_metrics()["total_incidents"] == 0
    assert store.record_incident(FakeRecord()) is None
