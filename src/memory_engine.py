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
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Tuple, Optional


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


def normalize_context(raw: Dict[str, float]) -> List[float]:
    """
    Normalizes observed cloud operating telemetry into a 5-dimensional state vector:
    V_state = [CPU_Usage, Memory_Usage, DB_Connections, Network_Latency, Error_Rate]
    Each dimension is normalized to the range [0.0, 1.0].
    """
    # CPU: 0 - 100% -> [0.0, 1.0]
    cpu = max(0.0, min(1.0, raw.get("cpu", 0.0) / 100.0 if raw.get("cpu", 0.0) > 1.0 else raw.get("cpu", 0.0)))
    
    # Memory: 0 - 100% -> [0.0, 1.0]
    mem = max(0.0, min(1.0, raw.get("mem", 0.0) / 100.0 if raw.get("mem", 0.0) > 1.0 else raw.get("mem", 0.0)))
    
    # DB Connections: 0 - 100% (or pool percentage) -> [0.0, 1.0]
    db_conn = max(0.0, min(1.0, raw.get("db_conn", 0.0) / 100.0 if raw.get("db_conn", 0.0) > 1.0 else raw.get("db_conn", 0.0)))
    
    # Network Latency: assume nominal max ~1000ms -> [0.0, 1.0]
    lat_raw = raw.get("latency_ms", 0.0)
    latency = max(0.0, min(1.0, lat_raw / 1000.0 if lat_raw > 1.0 else lat_raw))
    
    # Error Rate: 0 - 1.0 (or 0 - 100%) -> [0.0, 1.0]
    err_raw = raw.get("error_rate", 0.0)
    error_rate = max(0.0, min(1.0, err_raw / 100.0 if err_raw > 1.0 else err_raw))
    
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
        """
        delta_t = max(0.0, current_time - self.timestamp)
        return float(self.initial_confidence * math.exp(-self.decay_lambda * delta_t))


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

    def __init__(self, db_path: Optional[str] = "outputs/remediation_memory.db", in_memory_only: bool = False):
        self.db_path = db_path
        self.in_memory_only = in_memory_only
        self.negative_memory: List[NegativeConstraint] = []
        self.positive_memory: List[PositiveOutcome] = []
        
        if not self.in_memory_only and self.db_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
            self._init_sqlite()
            self._load_sqlite()

    def _init_sqlite(self):
        """Initializes SQLite database tables for persistence."""
        conn = sqlite3.connect(self.db_path)
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
        conn.commit()
        conn.close()

    def _load_sqlite(self):
        """Loads records from SQLite into memory."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        cur.execute("SELECT record_id, failure_condition, observed_context_raw, observed_context_vector, action, result, verification_evidence, failure_reason, negative_constraint, toxicity, initial_confidence, decay_lambda, timestamp FROM negative_constraints")
        rows = cur.fetchall()
        self.negative_memory = []
        for r in rows:
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
            
        cur.execute("SELECT record_id, failure_condition, observed_context_raw, observed_context_vector, action, result, verification_evidence, timestamp FROM positive_outcomes")
        rows = cur.fetchall()
        self.positive_memory = []
        for r in rows:
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
        Applies default parameters:
        - Neutral Failure: C0 = 0.86, lambda = 0.05
        - Toxic Failure: C0 = 0.98, lambda = 0.015
        """
        if record_id is None:
            record_id = str(uuid.uuid4())[:8]

        toxicity_upper = toxicity.upper()
        if toxicity_upper == "TOXIC":
            if initial_confidence is None:
                initial_confidence = 0.98
            if decay_lambda is None:
                decay_lambda = 0.015
        else:
            toxicity_upper = "NEUTRAL"
            if initial_confidence is None:
                initial_confidence = 0.86
            if decay_lambda is None:
                decay_lambda = 0.05

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
            conn = sqlite3.connect(self.db_path)
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
            conn.close()

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
            conn = sqlite3.connect(self.db_path)
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
            conn.close()

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
        """Clears in-memory lists and wipes SQLite database tables."""
        self.negative_memory.clear()
        self.positive_memory.clear()
        if not self.in_memory_only and self.db_path and os.path.exists(self.db_path):
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute("DELETE FROM negative_constraints")
            cur.execute("DELETE FROM positive_outcomes")
            conn.commit()
            conn.close()
