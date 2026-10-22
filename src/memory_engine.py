"""
Negative Remediation Memory Engine for Autonomous Cloud Self-Healing.
Implements:
1. NegativeConstraint & PositiveOutcome data schemas
2. Normalized context vectorization
3. Fuzzy Context Fingerprinting using Cosine Similarity
4. Ebbinghaus Confidence Decay (Self-Forgetting Memory)
5. Collateral Damage / Toxicity scoring
6. Candidate action filtering with XAI explainability
7. SQLite persistence
"""

import math
import sqlite3
import json
import os
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple, Optional

from src.paths import DEFAULT_DB_PATH, ensure_parent_dir, resolve_path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Decay / confidence policy constants
# ---------------------------------------------------------------------------
# A *neutral* failure (action failed but caused no collateral damage) starts
# with moderate confidence and forgets at a baseline rate. A *toxic* failure
# (action caused catastrophic collateral damage) starts with higher confidence
# and decays proportionally slower so that destructive actions are protected
# for longer. The toxic/neutral ratio is preserved from the original research
# methodology (0.015 / 0.05 = 0.3).
DEFAULT_NEUTRAL_CONFIDENCE: float = 0.86
DEFAULT_NEUTRAL_DECAY: float = 0.05
DEFAULT_TOXIC_CONFIDENCE: float = 0.98
DEFAULT_TOXIC_DECAY: float = 0.015
TOXIC_DECAY_FACTOR: float = DEFAULT_TOXIC_DECAY / DEFAULT_NEUTRAL_DECAY  # 0.3


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerces ``value`` to a finite float, returning ``default`` on failure."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result) or math.isinf(result):
        return default
    return result


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """
    Computes the Cosine Similarity between two normalized numerical vectors.
    Formula:
        Similarity(v1, v2) = (v1 . v2) / (||v1|| * ||v2||)
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_v1 = math.sqrt(sum(a * a for a in v1))
    norm_v2 = math.sqrt(sum(b * b for b in v2))
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0
    sim = dot_product / (norm_v1 * norm_v2)
    # Clamp within [-1.0, 1.0] to guard against floating point inaccuracies
    return float(max(-1.0, min(1.0, sim)))


def normalize_context(raw: Optional[Dict[str, float]]) -> List[float]:
    """
    Normalizes observed cloud operating telemetry into a 5-dimensional state vector:
    V_state = [CPU_Usage, Memory_Usage, DB_Connections, Network_Latency, Error_Rate]
    Each dimension is normalized to the range [0.0, 1.0].

    Non-numeric, missing, or non-finite inputs are treated as 0.0 rather than
    raising, so malformed telemetry can never crash the healing loop.
    """
    raw = raw if isinstance(raw, dict) else {}

    def _pct(key: str) -> float:
        # Accept either a 0-1 fraction or a 0-100 percentage.
        val = _safe_float(raw.get(key, 0.0), 0.0)
        return max(0.0, min(1.0, val / 100.0 if val > 1.0 else val))

    # CPU / Memory / DB connections are percentage-like metrics.
    cpu = _pct("cpu")
    mem = _pct("mem")
    db_conn = _pct("db_conn")

    # Network Latency: assume nominal max ~1000ms -> [0.0, 1.0]
    lat_raw = _safe_float(raw.get("latency_ms", 0.0), 0.0)
    latency = max(0.0, min(1.0, lat_raw / 1000.0 if lat_raw > 1.0 else lat_raw))

    # Error Rate: 0 - 1.0 (or 0 - 100%) -> [0.0, 1.0]
    error_rate = _pct("error_rate")

    return [round(cpu, 4), round(mem, 4), round(db_conn, 4), round(latency, 4), round(error_rate, 4)]


@dataclass
class NegativeConstraint:
    """
    Structured Negative Remediation Memory Record Schema.
    Represents a verified failed recovery attempt with operating telemetry context,
    root cause notes, and exponential decay parameters.
    """
    record_id: str
    failure_condition: str
    observed_context_raw: Dict[str, float]
    observed_context_vector: List[float]
    action: str
    result: str  # Always "FAILED" for negative constraints
    verification_evidence: str
    failure_reason: str
    negative_constraint: str
    toxicity: str  # "NEUTRAL" or "TOXIC"
    initial_confidence: float
    decay_lambda: float
    timestamp: float

    def get_decayed_confidence(self, current_time: float) -> float:
        """
        Computes active confidence C(t) using Ebbinghaus exponential decay:
        C(t) = C0 * exp(-lambda * max(0, t - t0))

        This is the single source of truth for confidence used by both the
        filtering engine and the dashboard display.
        """
        delta_t = max(0.0, _safe_float(current_time, 0.0) - self.timestamp)
        return float(self.initial_confidence * math.exp(-self.decay_lambda * delta_t))

    def is_active(self, current_time: float, theta_conf: float = 0.40) -> bool:
        """True while the constraint confidence is at/above the threshold."""
        return self.get_decayed_confidence(current_time) >= theta_conf

    def get_decay_info(self, current_time: float, theta_conf: float = 0.40) -> Dict[str, Any]:
        """
        Returns a consistent snapshot of the decay state for a given time.
        Used by the UI so displayed confidence and active/expired status are
        computed exactly the same way as the blocking decision.
        """
        delta_t = max(0.0, _safe_float(current_time, 0.0) - self.timestamp)
        confidence = self.get_decayed_confidence(current_time)
        return {
            "initial_confidence": self.initial_confidence,
            "decay_lambda": self.decay_lambda,
            "elapsed": round(delta_t, 4),
            "decayed_confidence": round(confidence, 6),
            "theta_conf": theta_conf,
            "is_active": confidence >= theta_conf,
        }


@dataclass
class PositiveOutcome:
    """
    Structured Positive Remediation Memory Record Schema.
    Tracks verified successful recovery actions to prioritize safe remediations.
    """
    record_id: str
    failure_condition: str
    observed_context_raw: Dict[str, float]
    observed_context_vector: List[float]
    action: str
    result: str  # Always "SUCCESS"
    verification_evidence: str
    timestamp: float


class RemediationMemoryEngine:
    """
    Remediation Memory & Constraint Engine.
    Manages negative constraints (with decay and cosine similarity), positive outcomes,
    and candidate action filtering with explainable diagnostic outputs.
    """

    def __init__(self, db_path: Optional[str] = DEFAULT_DB_PATH, in_memory_only: bool = False):
        self.db_path = resolve_path(db_path) if db_path else db_path
        self.in_memory_only = in_memory_only
        self.negative_memory: List[NegativeConstraint] = []
        self.positive_memory: List[PositiveOutcome] = []
        self.last_error: Optional[str] = None

        if not self.in_memory_only and self.db_path:
            try:
                ensure_parent_dir(self.db_path)
                self._init_sqlite()
                self._load_sqlite()
            except sqlite3.Error as exc:
                # Keep the engine usable in-memory even if persistence fails.
                self.last_error = f"SQLite initialization failed: {exc}"
                logger.exception("Failed to initialize remediation memory database at %s", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        """Opens a connection with sensible concurrency defaults."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_sqlite(self):
        """Initializes SQLite database tables for persistence (idempotent)."""
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS negative_constraints (
                    record_id TEXT PRIMARY KEY,
                    failure_condition TEXT,
                    observed_context_raw TEXT,
                    observed_context_vector TEXT,
                    action TEXT,
                    result TEXT,
                    verification_evidence TEXT,
                    failure_reason TEXT,
                    negative_constraint TEXT,
                    toxicity TEXT,
                    initial_confidence REAL,
                    decay_lambda REAL,
                    timestamp REAL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS positive_outcomes (
                    record_id TEXT PRIMARY KEY,
                    failure_condition TEXT,
                    observed_context_raw TEXT,
                    observed_context_vector TEXT,
                    action TEXT,
                    result TEXT,
                    verification_evidence TEXT,
                    timestamp REAL
                )
            """)
            # Safe, additive schema migration: older databases created before
            # these columns existed are upgraded without dropping any data.
            self._ensure_column(cur, "negative_constraints", "failure_reason", "TEXT")
            self._ensure_column(cur, "negative_constraints", "negative_constraint", "TEXT")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_neg_condition_action "
                "ON negative_constraints(failure_condition, action)"
            )
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _ensure_column(cur: sqlite3.Cursor, table: str, column: str, decl: str) -> None:
        """Adds a column only if it is missing (non-destructive migration)."""
        existing = {row[1] for row in cur.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def _load_sqlite(self):
        """Loads records from SQLite into memory."""
        conn = self._connect()
        try:
            cur = conn.cursor()

            cur.execute("SELECT record_id, failure_condition, observed_context_raw, observed_context_vector, action, result, verification_evidence, failure_reason, negative_constraint, toxicity, initial_confidence, decay_lambda, timestamp FROM negative_constraints")
            rows = cur.fetchall()
            self.negative_memory = []
            for r in rows:
                try:
                    self.negative_memory.append(NegativeConstraint(
                        record_id=r[0],
                        failure_condition=r[1],
                        observed_context_raw=json.loads(r[2]),
                        observed_context_vector=json.loads(r[3]),
                        action=r[4],
                        result=r[5],
                        verification_evidence=r[6],
                        failure_reason=r[7],
                        negative_constraint=r[8],
                        toxicity=r[9],
                        initial_confidence=float(r[10]),
                        decay_lambda=float(r[11]),
                        timestamp=float(r[12])
                    ))
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.warning("Skipping corrupt negative constraint row %r", r[0])

            cur.execute("SELECT record_id, failure_condition, observed_context_raw, observed_context_vector, action, result, verification_evidence, timestamp FROM positive_outcomes")
            rows = cur.fetchall()
            self.positive_memory = []
            for r in rows:
                try:
                    self.positive_memory.append(PositiveOutcome(
                        record_id=r[0],
                        failure_condition=r[1],
                        observed_context_raw=json.loads(r[2]),
                        observed_context_vector=json.loads(r[3]),
                        action=r[4],
                        result=r[5],
                        verification_evidence=r[6],
                        timestamp=float(r[7])
                    ))
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.warning("Skipping corrupt positive outcome row %r", r[0])
        finally:
            conn.close()

    def record_negative_constraint(
        self,
        failure_condition: str,
        observed_context_raw: Dict[str, float],
        observed_context_vector: List[float],
        action: str,
        verification_evidence: str,
        failure_reason: str,
        negative_constraint: str,
        toxicity: str = "NEUTRAL",
        initial_confidence: Optional[float] = None,
        decay_lambda: Optional[float] = None,
        timestamp: float = 0.0,
        record_id: Optional[str] = None
    ) -> NegativeConstraint:
        """
        Creates and stores a verified negative constraint.

        Decay policy (documented, consistent):
        - ``decay_lambda`` is the *base* decay constant supplied by the caller
          (e.g. the dashboard slider). It governs newly created NEUTRAL
          constraints.
        - If ``decay_lambda`` is omitted, the research default (0.05) is used.
        - TOXIC constraints always decay ``TOXIC_DECAY_FACTOR`` times slower
          than the base rate, preserving the original prolonged-protection
          semantics (0.015 / 0.05 = 0.3).
        - Existing constraints are never retroactively changed; each stores its
          own ``decay_lambda`` at creation time. Changing the slider therefore
          affects only *new* constraints, which keeps historical confidence
          reproducible.
        """
        if record_id is None:
            record_id = str(uuid.uuid4())[:8]

        # Base decay rate for this constraint (neutral baseline).
        base_decay = _safe_float(decay_lambda, DEFAULT_NEUTRAL_DECAY) if decay_lambda is not None else DEFAULT_NEUTRAL_DECAY
        base_decay = max(1e-6, base_decay)

        toxicity_upper = str(toxicity).upper()
        if toxicity_upper == "TOXIC":
            if initial_confidence is None:
                initial_confidence = DEFAULT_TOXIC_CONFIDENCE
            decay_lambda = base_decay * TOXIC_DECAY_FACTOR
        else:
            toxicity_upper = "NEUTRAL"
            if initial_confidence is None:
                initial_confidence = DEFAULT_NEUTRAL_CONFIDENCE
            decay_lambda = base_decay

        constraint = NegativeConstraint(
            record_id=record_id,
            failure_condition=failure_condition,
            observed_context_raw=observed_context_raw,
            observed_context_vector=observed_context_vector,
            action=action,
            result="FAILED",
            verification_evidence=verification_evidence,
            failure_reason=failure_reason,
            negative_constraint=negative_constraint,
            toxicity=toxicity_upper,
            initial_confidence=initial_confidence,
            decay_lambda=decay_lambda,
            timestamp=timestamp
        )

        self.negative_memory.append(constraint)

        if not self.in_memory_only and self.db_path:
            try:
                conn = self._connect()
                try:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT OR REPLACE INTO negative_constraints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        constraint.record_id,
                        constraint.failure_condition,
                        json.dumps(constraint.observed_context_raw),
                        json.dumps(constraint.observed_context_vector),
                        constraint.action,
                        constraint.result,
                        constraint.verification_evidence,
                        constraint.failure_reason,
                        constraint.negative_constraint,
                        constraint.toxicity,
                        constraint.initial_confidence,
                        constraint.decay_lambda,
                        constraint.timestamp
                    ))
                    conn.commit()
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                self.last_error = f"Failed to persist negative constraint: {exc}"
                logger.exception("Failed to persist negative constraint %s", constraint.record_id)

        return constraint

    def record_positive_outcome(
        self,
        failure_condition: str,
        observed_context_raw: Dict[str, float],
        observed_context_vector: List[float],
        action: str,
        verification_evidence: str,
        timestamp: float = 0.0,
        record_id: Optional[str] = None
    ) -> PositiveOutcome:
        """
        Records a verified successful remediation action.
        """
        if record_id is None:
            record_id = str(uuid.uuid4())[:8]

        outcome = PositiveOutcome(
            record_id=record_id,
            failure_condition=failure_condition,
            observed_context_raw=observed_context_raw,
            observed_context_vector=observed_context_vector,
            action=action,
            result="SUCCESS",
            verification_evidence=verification_evidence,
            timestamp=timestamp
        )

        self.positive_memory.append(outcome)

        if not self.in_memory_only and self.db_path:
            try:
                conn = self._connect()
                try:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT OR REPLACE INTO positive_outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        outcome.record_id,
                        outcome.failure_condition,
                        json.dumps(outcome.observed_context_raw),
                        json.dumps(outcome.observed_context_vector),
                        outcome.action,
                        outcome.result,
                        outcome.verification_evidence,
                        outcome.timestamp
                    ))
                    conn.commit()
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                self.last_error = f"Failed to persist positive outcome: {exc}"
                logger.exception("Failed to persist positive outcome %s", outcome.record_id)

        return outcome

    def filter_candidate_actions(
        self,
        current_condition: str,
        current_vector: List[float],
        candidate_actions: List[str],
        current_time: float,
        theta_sim: float = 0.85,
        theta_conf: float = 0.40
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        """
        Filters candidate actions against stored Negative Remediation Memories.
        
        Constraint Activation Rule:
            An action is blocked if and only if:
            Similarity(V_current, V_stored) >= theta_sim AND C(t) >= theta_conf
            
        Returns:
            (allowed_actions, blocked_details)
            where blocked_details contains explainable diagnostic metadata for XAI / human operators.
        """
        allowed_actions: List[str] = []
        blocked_details: List[Dict[str, Any]] = []

        for action in candidate_actions:
            is_blocked = False
            best_blocking_constraint = None
            max_similarity = -1.0
            blocking_confidence = 0.0

            # Scan stored negative constraints for this action and matching failure condition
            for stored in self.negative_memory:
                if stored.action == action and stored.failure_condition == current_condition:
                    sim = cosine_similarity(current_vector, stored.observed_context_vector)
                    conf = stored.get_decayed_confidence(current_time)

                    if sim >= theta_sim and conf >= theta_conf:
                        if sim > max_similarity:
                            max_similarity = sim
                            blocking_confidence = conf
                            best_blocking_constraint = stored
                            is_blocked = True

            if is_blocked and best_blocking_constraint:
                blocked_details.append({
                    "action": action,
                    "failure_condition": current_condition,
                    "similarity": round(max_similarity, 4),
                    "decayed_confidence": round(blocking_confidence, 4),
                    "initial_confidence": best_blocking_constraint.initial_confidence,
                    "decay_lambda": best_blocking_constraint.decay_lambda,
                    "toxicity": best_blocking_constraint.toxicity,
                    "record_id": best_blocking_constraint.record_id,
                    "elapsed_time": round(current_time - best_blocking_constraint.timestamp, 2),
                    "failure_reason": best_blocking_constraint.failure_reason,
                    "negative_constraint": best_blocking_constraint.negative_constraint,
                    "evidence": best_blocking_constraint.verification_evidence
                })
            else:
                allowed_actions.append(action)

        return allowed_actions, blocked_details

    def get_ranked_positive_actions(
        self,
        current_condition: str,
        candidate_actions: List[str]
    ) -> List[Tuple[str, int]]:
        """
        Ranks candidate actions based on historical positive success frequency for the condition.
        """
        success_counts: Dict[str, int] = {act: 0 for act in candidate_actions}
        for pos in self.positive_memory:
            if pos.failure_condition == current_condition and pos.action in success_counts:
                success_counts[pos.action] += 1
        
        # Sort descending by success count
        ranked = sorted(success_counts.items(), key=lambda item: item[1], reverse=True)
        return ranked

    def clear_memory(self):
        """
        Clears only the *learned remediation memory* (negative constraints and
        positive outcomes) from both RAM and SQLite.

        Incident execution history, benchmark artifacts, and any other tables
        in the shared database are deliberately preserved.
        """
        self.negative_memory.clear()
        self.positive_memory.clear()
        if not self.in_memory_only and self.db_path and os.path.exists(self.db_path):
            try:
                conn = self._connect()
                try:
                    cur = conn.cursor()
                    cur.execute("DELETE FROM negative_constraints")
                    cur.execute("DELETE FROM positive_outcomes")
                    conn.commit()
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                self.last_error = f"Failed to clear remediation memory: {exc}"
                logger.exception("Failed to clear remediation memory at %s", self.db_path)
