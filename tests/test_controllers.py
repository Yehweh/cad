"""Tests for controller behavior: memory blocking, alternatives, failures."""

from src.cloud_simulator import CloudEnvironmentSimulator
from src.memory_engine import RemediationMemoryEngine, normalize_context
from src.healing_controller import (
    ProposedNegativeMemoryController,
    BaselineAFixedRuleController,
    BaselineBPositiveOnlyController,
)

RAW_DB = {"cpu": 42.0, "mem": 60.0, "db_conn": 100.0, "latency_ms": 850.0, "error_rate": 0.40}
CANDIDATES_DB = ["restart_service", "connection_capacity_remediation", "scale_compute_instances"]


def _seed_failed_restart(engine):
    engine.record_negative_constraint(
        failure_condition="DB_CONN_EXHAUSTION",
        observed_context_raw=RAW_DB,
        observed_context_vector=normalize_context(RAW_DB),
        action="restart_service",
        verification_evidence="HTTP 500",
        failure_reason="DB pool saturated across restarts",
        negative_constraint="Avoid restart_service for DB_CONN_EXHAUSTION",
        toxicity="NEUTRAL",
        timestamp=0.0,
    )


def test_negative_memory_blocks_action_and_controller_uses_alternative():
    sim = CloudEnvironmentSimulator(seed=101)
    engine = RemediationMemoryEngine(in_memory_only=True)
    _seed_failed_restart(engine)
    ctrl = ProposedNegativeMemoryController(
        simulator=sim, memory_engine=engine, theta_sim=0.85, theta_conf=0.40, decay_lambda=0.05
    )

    vec = normalize_context(RAW_DB)
    record = ctrl.handle_incident(
        incident_id=2,
        condition="DB_CONN_EXHAUSTION",
        raw_telemetry=RAW_DB,
        vector=vec,
        candidates=CANDIDATES_DB,
        timestamp=1.0,
    )

    assert record.successful_action == "connection_capacity_remediation"
    assert record.attempts == 1
    assert record.first_attempt_success is True
    assert "restart_service" not in record.tried_actions
    assert any(b["action"] == "restart_service" for b in record.blocked_actions)


def test_controller_reports_recovery_failure_when_no_action_helps():
    sim = CloudEnvironmentSimulator(seed=101)
    engine = RemediationMemoryEngine(in_memory_only=True)
    ctrl = ProposedNegativeMemoryController(
        simulator=sim, memory_engine=engine, theta_sim=0.85, theta_conf=0.40, decay_lambda=0.05
    )

    raw = {"cpu": 92.0, "mem": 85.0, "db_conn": 95.0, "latency_ms": 950.0, "error_rate": 0.75}
    vec = normalize_context(raw)
    record = ctrl.handle_incident(
        incident_id=1,
        condition="CASCADING_CACHE_STAMPEDE",
        raw_telemetry=raw,
        vector=vec,
        candidates=["restart_service", "flush_cache"],  # both toxic failures
        timestamp=1.0,
    )

    assert record.successful_action == ""
    assert record.attempts == 2
    # Two negative constraints were learned from the two failures.
    assert len(engine.negative_memory) == 2
    assert all(c.toxicity == "TOXIC" for c in engine.negative_memory)


def test_decay_lambda_override_affects_new_constraints():
    sim = CloudEnvironmentSimulator(seed=101)
    engine = RemediationMemoryEngine(in_memory_only=True)
    ctrl = ProposedNegativeMemoryController(
        simulator=sim, memory_engine=engine, theta_sim=0.85, theta_conf=0.40, decay_lambda=0.12
    )
    raw = {"cpu": 92.0, "mem": 85.0, "db_conn": 95.0, "latency_ms": 950.0, "error_rate": 0.75}
    vec = normalize_context(raw)
    ctrl.handle_incident(1, "CASCADING_CACHE_STAMPEDE", raw, vec, ["restart_service"], timestamp=1.0)
    assert engine.negative_memory
    # Toxic decay = 0.3 * base (0.12) = 0.036
    assert engine.negative_memory[0].decay_lambda == 0.12 * 0.3


def test_baseline_controllers_record_history():
    sim_a = CloudEnvironmentSimulator(seed=42)
    sim_b = CloudEnvironmentSimulator(seed=42)
    ctrl_a = BaselineAFixedRuleController(sim_a)
    ctrl_b = BaselineBPositiveOnlyController(sim_b)
    vec = normalize_context(RAW_DB)
    rec_a = ctrl_a.handle_incident(1, "DB_CONN_EXHAUSTION", RAW_DB, vec, CANDIDATES_DB, 1.0)
    rec_b = ctrl_b.handle_incident(1, "DB_CONN_EXHAUSTION", RAW_DB, vec, CANDIDATES_DB, 1.0)
    assert rec_a.successful_action == "connection_capacity_remediation"
    assert rec_b.successful_action == "connection_capacity_remediation"
    # Baseline B tries the globally-biased restart_service first and wastes an attempt.
    assert rec_b.attempts >= 2
