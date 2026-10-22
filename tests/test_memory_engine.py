"""Tests for the remediation memory engine: schema, persistence, decay."""

import math
import os
import sqlite3

import pytest

from src.memory_engine import (
    RemediationMemoryEngine,
    NegativeConstraint,
    normalize_context,
    cosine_similarity,
    DEFAULT_NEUTRAL_DECAY,
    DEFAULT_TOXIC_DECAY,
    TOXIC_DECAY_FACTOR,
)


def _constraint_kwargs(**overrides):
    base = dict(
        failure_condition="DB_CONN_EXHAUSTION",
        observed_context_raw={"cpu": 42.0, "mem": 60.0, "db_conn": 100.0, "latency_ms": 850.0, "error_rate": 0.4},
        observed_context_vector=normalize_context(
            {"cpu": 42.0, "mem": 60.0, "db_conn": 100.0, "latency_ms": 850.0, "error_rate": 0.4}
        ),
        action="restart_service",
        verification_evidence="HTTP 500",
        failure_reason="DB pool saturated",
        negative_constraint="Avoid restart_service",
        timestamp=0.0,
    )
    base.update(overrides)
    return base


def test_database_creation_and_schema(tmp_path):
    db = str(tmp_path / "mem.db")
    RemediationMemoryEngine(db_path=db, in_memory_only=False)
    conn = sqlite3.connect(db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "negative_constraints" in tables
    assert "positive_outcomes" in tables


def test_negative_constraint_persistence_roundtrip(tmp_path):
    db = str(tmp_path / "mem.db")
    engine = RemediationMemoryEngine(db_path=db, in_memory_only=False)
    engine.record_negative_constraint(**_constraint_kwargs())

    reloaded = RemediationMemoryEngine(db_path=db, in_memory_only=False)
    assert len(reloaded.negative_memory) == 1
    stored = reloaded.negative_memory[0]
    assert stored.action == "restart_service"
    assert stored.failure_condition == "DB_CONN_EXHAUSTION"
    assert stored.result == "FAILED"


def test_positive_outcome_persistence_roundtrip(tmp_path):
    db = str(tmp_path / "mem.db")
    engine = RemediationMemoryEngine(db_path=db, in_memory_only=False)
    engine.record_positive_outcome(
        failure_condition="MEMORY_LEAK_OOM",
        observed_context_raw={"cpu": 55.0, "mem": 98.0, "db_conn": 30.0, "latency_ms": 600.0, "error_rate": 0.5},
        observed_context_vector=normalize_context(
            {"cpu": 55.0, "mem": 98.0, "db_conn": 30.0, "latency_ms": 600.0, "error_rate": 0.5}
        ),
        action="restart_service",
        verification_evidence="HTTP 200",
        timestamp=1.0,
    )
    reloaded = RemediationMemoryEngine(db_path=db, in_memory_only=False)
    assert len(reloaded.positive_memory) == 1
    assert reloaded.positive_memory[0].result == "SUCCESS"


def test_confidence_decay_and_expiration():
    engine = RemediationMemoryEngine(in_memory_only=True)
    c = engine.record_negative_constraint(**_constraint_kwargs())
    # C(t) = C0 * exp(-lambda * dt)
    expected = c.initial_confidence * math.exp(-c.decay_lambda * 10.0)
    assert c.get_decayed_confidence(10.0) == pytest.approx(expected)
    # No time travel: negative elapsed is clamped to zero.
    assert c.get_decayed_confidence(-5.0) == pytest.approx(c.initial_confidence)
    # Active then expired.
    assert c.is_active(0.0, theta_conf=0.40)
    late = c.get_decayed_confidence(100.0)
    assert late < 0.40
    assert not c.is_active(100.0, theta_conf=0.40)


def test_decay_policy_slider_and_toxicity():
    engine = RemediationMemoryEngine(in_memory_only=True)
    neutral = engine.record_negative_constraint(**_constraint_kwargs(decay_lambda=0.10))
    assert neutral.decay_lambda == pytest.approx(0.10)

    toxic = engine.record_negative_constraint(
        **_constraint_kwargs(decay_lambda=0.10, toxicity="TOXIC", action="flush_cache")
    )
    assert toxic.decay_lambda == pytest.approx(0.10 * TOXIC_DECAY_FACTOR)
    assert toxic.initial_confidence == pytest.approx(0.98)

    # Defaults preserved when no override is supplied.
    default_neutral = engine.record_negative_constraint(**_constraint_kwargs(action="scale_compute_instances"))
    assert default_neutral.decay_lambda == pytest.approx(DEFAULT_NEUTRAL_DECAY)
    default_toxic = engine.record_negative_constraint(
        **_constraint_kwargs(action="restart_service", toxicity="TOXIC")
    )
    assert default_toxic.decay_lambda == pytest.approx(DEFAULT_TOXIC_DECAY)


def test_filter_blocks_previously_unsuccessful_action():
    engine = RemediationMemoryEngine(in_memory_only=True)
    vec = _constraint_kwargs()["observed_context_vector"]
    engine.record_negative_constraint(**_constraint_kwargs(timestamp=0.0))

    allowed, blocked = engine.filter_candidate_actions(
        current_condition="DB_CONN_EXHAUSTION",
        current_vector=vec,
        candidate_actions=["restart_service", "connection_capacity_remediation"],
        current_time=1.0,
        theta_sim=0.85,
        theta_conf=0.40,
    )
    assert "restart_service" not in allowed
    assert "connection_capacity_remediation" in allowed
    assert blocked and blocked[0]["action"] == "restart_service"


def test_filter_releases_action_after_expiry():
    engine = RemediationMemoryEngine(in_memory_only=True)
    vec = _constraint_kwargs()["observed_context_vector"]
    engine.record_negative_constraint(**_constraint_kwargs(timestamp=0.0))

    allowed, blocked = engine.filter_candidate_actions(
        current_condition="DB_CONN_EXHAUSTION",
        current_vector=vec,
        candidate_actions=["restart_service"],
        current_time=1000.0,
        theta_sim=0.85,
        theta_conf=0.40,
    )
    assert "restart_service" in allowed
    assert blocked == []


def test_invalid_telemetry_is_safe():
    assert normalize_context(None) == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert normalize_context({"cpu": "not-a-number"}) == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert normalize_context({"cpu": -50, "mem": 250}) == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert normalize_context({"cpu": float("nan")})[0] == 0.0
    assert cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0
    assert cosine_similarity([], [1.0]) == 0.0


def test_schema_migration_preserves_existing_records(tmp_path):
    """A DB created with an older/partial schema must be upgraded non-destructively."""
    db = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE negative_constraints (
            record_id TEXT PRIMARY KEY,
            failure_condition TEXT,
            observed_context_raw TEXT,
            observed_context_vector TEXT,
            action TEXT,
            result TEXT,
            verification_evidence TEXT,
            toxicity TEXT,
            initial_confidence REAL,
            decay_lambda REAL,
            timestamp REAL
        )
        """
    )
    # "type" column name is not reserved in SQLite; ensure it exists for realism.
    conn.execute(
        "INSERT INTO negative_constraints VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            "legacy1", "DB_CONN_EXHAUSTION", "{}", "[0.1,0.2,0.3,0.4,0.5]", "restart_service",
            "FAILED", "old evidence", "NEUTRAL", 0.86, 0.05, 0.0,
        ),
    )
    conn.commit()
    conn.close()

    engine = RemediationMemoryEngine(db_path=db, in_memory_only=False)
    assert engine.last_error is None
    assert any(c.record_id == "legacy1" for c in engine.negative_memory)

    conn = sqlite3.connect(db)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(negative_constraints)")}
    conn.close()
    assert "failure_reason" in cols
    assert "negative_constraint" in cols


def test_sqlite_error_is_handled_gracefully(tmp_path):
    # Pointing at a directory makes sqlite fail to open; engine must not crash.
    engine = RemediationMemoryEngine(db_path=str(tmp_path), in_memory_only=False)
    assert engine.last_error is not None
    assert engine.negative_memory == []
    assert engine.positive_memory == []


def test_clear_memory_only_clears_memory(tmp_path):
    db = str(tmp_path / "mem.db")
    engine = RemediationMemoryEngine(db_path=db, in_memory_only=False)
    engine.record_negative_constraint(**_constraint_kwargs())
    engine.clear_memory()
    assert engine.negative_memory == []
    reloaded = RemediationMemoryEngine(db_path=db, in_memory_only=False)
    assert reloaded.negative_memory == []
