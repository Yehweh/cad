"""
Autonomous Self-Healing Controllers.
Implements:
1. Baseline A: Fixed Rule-Based Controller (Zero cross-incident memory)
2. Baseline B: Positive-Only Memory Controller (Success-biased memory)
3. Proposed System: Positive + Negative Remediation Memory Controller (Fuzzy matching + Ebbinghaus Decay + Toxicity awareness)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from src.memory_engine import RemediationMemoryEngine, NegativeConstraint, PositiveOutcome
from src.cloud_simulator import CloudEnvironmentSimulator, HealthCheckResult


@dataclass
class IncidentExecutionRecord:
    incident_id: int
    condition: str
    controller_name: str
    telemetry_raw: Dict[str, float]
    telemetry_vector: List[float]
    attempts: int
    tried_actions: List[str]
    successful_action: str
    blocked_actions: List[Dict[str, Any]]
    first_attempt_success: bool
    repeated_failures: int
    mttr_seconds: float
    toxic_failures_encountered: int
    timestamp: float
    execution_trace: List[Dict[str, Any]] = field(default_factory=list)


class BaselineAFixedRuleController:
    """
    Baseline A: Fixed Rule-Based Healing (No Persistent Memory).
    Always tries actions in a hard-coded static priority order.
    Repeats failed recovery attempts every time an incident recurs.
    """

    def __init__(self, simulator: CloudEnvironmentSimulator):
        self.simulator = simulator
        self.name = "Baseline A (Fixed Rule)"
        self.history: List[IncidentExecutionRecord] = []
        self.seen_failures: Dict[str, set] = {}

    def handle_incident(
        self,
        incident_id: int,
        condition: str,
        raw_telemetry: Dict[str, float],
        vector: List[float],
        candidates: List[str],
        timestamp: float
    ) -> IncidentExecutionRecord:
        attempts = 0
        tried_actions: List[str] = []
        mttr_seconds = 0.0
        toxic_count = 0
        successful_action = ""
        repeated_failures = 0
        trace: List[Dict[str, Any]] = []

        if condition not in self.seen_failures:
            self.seen_failures[condition] = set()

        for action in candidates:
            attempts += 1
            tried_actions.append(action)

            # Check if this action previously failed for this condition in past incidents
            if action in self.seen_failures[condition]:
                repeated_failures += 1

            result: HealthCheckResult = self.simulator.execute_and_verify(condition, action, raw_telemetry)
            mttr_seconds += result.overhead_seconds
            
            trace.append({
                "attempt": attempts,
                "action": action,
                "status_code": result.status_code,
                "is_healthy": result.is_healthy,
                "evidence": result.verification_evidence,
                "toxicity": result.toxicity,
                "overhead": result.overhead_seconds
            })

            if result.toxicity == "TOXIC":
                toxic_count += 1

            if result.is_healthy:
                successful_action = action
                break
            else:
                self.seen_failures[condition].add(action)

        record = IncidentExecutionRecord(
            incident_id=incident_id,
            condition=condition,
            controller_name=self.name,
            telemetry_raw=raw_telemetry,
            telemetry_vector=vector,
            attempts=attempts,
            tried_actions=tried_actions,
            successful_action=successful_action,
            blocked_actions=[],
            first_attempt_success=(attempts == 1),
            repeated_failures=repeated_failures,
            mttr_seconds=mttr_seconds,
            toxic_failures_encountered=toxic_count,
            timestamp=timestamp,
            execution_trace=trace
        )
        self.history.append(record)
        return record


class BaselineBPositiveOnlyController:
    """
    Baseline B: Positive-Only Memory Healing (Success-Biased Memory).
    Tracks actions that succeeded in the past and prioritizes globally frequent winners
    (e.g., restart_service). Has no negative memory to record what failed or why.
    Fails repeatedly when an incident requires avoiding the globally popular action.
    """

    def __init__(self, simulator: CloudEnvironmentSimulator):
        self.simulator = simulator
        self.name = "Baseline B (Positive-Only)"
        self.history: List[IncidentExecutionRecord] = []
        self.global_success_counts: Dict[str, int] = {
            # restart_service has initial bias because in general cloud ops it fixes transient bugs
            "restart_service": 5,
            "connection_capacity_remediation": 1,
            "scale_compute_instances": 1,
            "reallocate_worker_memory": 1,
            "flush_cache": 0,
            "enable_rate_limiting_and_circuit_breaker": 0
        }
        self.seen_failures: Dict[str, set] = {}

    def handle_incident(
        self,
        incident_id: int,
        condition: str,
        raw_telemetry: Dict[str, float],
        vector: List[float],
        candidates: List[str],
        timestamp: float
    ) -> IncidentExecutionRecord:
        attempts = 0
        tried_actions: List[str] = []
        mttr_seconds = 0.0
        toxic_count = 0
        successful_action = ""
        repeated_failures = 0
        trace: List[Dict[str, Any]] = []

        if condition not in self.seen_failures:
            self.seen_failures[condition] = set()

        # Sort candidate actions descending by global positive success count
        ordered_candidates = sorted(
            candidates,
            key=lambda act: self.global_success_counts.get(act, 0),
            reverse=True
        )

        for action in ordered_candidates:
            attempts += 1
            tried_actions.append(action)

            if action in self.seen_failures[condition]:
                repeated_failures += 1

            result: HealthCheckResult = self.simulator.execute_and_verify(condition, action, raw_telemetry)
            mttr_seconds += result.overhead_seconds

            trace.append({
                "attempt": attempts,
                "action": action,
                "status_code": result.status_code,
                "is_healthy": result.is_healthy,
                "evidence": result.verification_evidence,
                "toxicity": result.toxicity,
                "overhead": result.overhead_seconds
            })

            if result.toxicity == "TOXIC":
                toxic_count += 1

            if result.is_healthy:
                successful_action = action
                self.global_success_counts[action] = self.global_success_counts.get(action, 0) + 1
                break
            else:
                self.seen_failures[condition].add(action)

        record = IncidentExecutionRecord(
            incident_id=incident_id,
            condition=condition,
            controller_name=self.name,
            telemetry_raw=raw_telemetry,
            telemetry_vector=vector,
            attempts=attempts,
            tried_actions=tried_actions,
            successful_action=successful_action,
            blocked_actions=[],
            first_attempt_success=(attempts == 1),
            repeated_failures=repeated_failures,
            mttr_seconds=mttr_seconds,
            toxic_failures_encountered=toxic_count,
            timestamp=timestamp,
            execution_trace=trace
        )
        self.history.append(record)
        return record


class ProposedNegativeMemoryController:
    """
    Proposed System: Positive + Negative Remediation Memory with Decay.
    1. Fuzzy Context Fingerprinting: Matches operational telemetry via Cosine Similarity.
    2. Decayed Confidence: Evaluates Ebbinghaus self-forgetting confidence C(t).
    3. Action Filtering: Prunes blocked actions with explainable diagnostics.
    4. Safe Selection: Selects safer alternatives informed by Positive Memory.
    5. Post-Action Verification: Records verified Negative Constraints upon failure or Positive Outcomes upon success.
    """

    def __init__(
        self,
        simulator: CloudEnvironmentSimulator,
        memory_engine: Optional[RemediationMemoryEngine] = None,
        theta_sim: float = 0.85,
        theta_conf: float = 0.40,
        decay_lambda: float = 0.05
    ):
        self.simulator = simulator
        self.memory_engine = memory_engine or RemediationMemoryEngine(in_memory_only=True)
        self.theta_sim = theta_sim
        self.theta_conf = theta_conf
        # Base decay constant applied to newly recorded NEUTRAL constraints.
        # Existing constraints keep the lambda they were created with.
        self.decay_lambda = decay_lambda
        self.name = "Proposed (Negative+Positive Memory)"
        self.history: List[IncidentExecutionRecord] = []
        self.prevented_actions_count = 0
        self.action_history_by_condition: Dict[str, List[str]] = {}

    def handle_incident(
        self,
        incident_id: int,
        condition: str,
        raw_telemetry: Dict[str, float],
        vector: List[float],
        candidates: List[str],
        timestamp: float
    ) -> IncidentExecutionRecord:
        attempts = 0
        tried_actions: List[str] = []
        mttr_seconds = 0.0
        toxic_count = 0
        successful_action = ""
        repeated_failures = 0
        trace: List[Dict[str, Any]] = []

        if condition not in self.action_history_by_condition:
            self.action_history_by_condition[condition] = []

        # Step 1: Filter candidate actions through Negative Remediation Memory
        allowed_actions, blocked_details = self.memory_engine.filter_candidate_actions(
            current_condition=condition,
            current_vector=vector,
            candidate_actions=candidates,
            current_time=timestamp,
            theta_sim=self.theta_sim,
            theta_conf=self.theta_conf
        )

        self.prevented_actions_count += len(blocked_details)

        # Step 2: Prioritize remaining allowed actions using Positive Memory
        if allowed_actions:
            ranked_positives = self.memory_engine.get_ranked_positive_actions(condition, allowed_actions)
            # Maintain stable priority according to positive success frequency, falling back to original list order
            ordered_actions = [act for act, cnt in ranked_positives if act in allowed_actions]
        else:
            # Fallback if all actions are blocked (safe fail-soft mechanism: retry candidate with lowest confidence constraint)
            ordered_actions = list(candidates)

        remaining_candidates = list(ordered_actions)

        # Step 3: Execution & Verification Loop
        while remaining_candidates:
            action = remaining_candidates.pop(0)
            attempts += 1
            tried_actions.append(action)

            # Check if this is a repeated failure from this controller's previous incidents
            if action in self.action_history_by_condition[condition]:
                repeated_failures += 1

            result: HealthCheckResult = self.simulator.execute_and_verify(condition, action, raw_telemetry)
            mttr_seconds += result.overhead_seconds

            trace.append({
                "attempt": attempts,
                "action": action,
                "status_code": result.status_code,
                "is_healthy": result.is_healthy,
                "evidence": result.verification_evidence,
                "toxicity": result.toxicity,
                "overhead": result.overhead_seconds
            })

            if result.toxicity == "TOXIC":
                toxic_count += 1

            if result.is_healthy:
                successful_action = action
                # Record verified Positive Memory
                self.memory_engine.record_positive_outcome(
                    failure_condition=condition,
                    observed_context_raw=raw_telemetry,
                    observed_context_vector=vector,
                    action=action,
                    verification_evidence=result.verification_evidence,
                    timestamp=timestamp
                )
                break
            else:
                self.action_history_by_condition[condition].append(action)
                # Record verified Negative Constraint
                constraint_text = f"Avoid {action} for {condition} under similar operational saturation"
                self.memory_engine.record_negative_constraint(
                    failure_condition=condition,
                    observed_context_raw=raw_telemetry,
                    observed_context_vector=vector,
                    action=action,
                    verification_evidence=result.verification_evidence,
                    failure_reason=result.failure_reason,
                    negative_constraint=constraint_text,
                    toxicity=result.toxicity,
                    decay_lambda=self.decay_lambda,
                    timestamp=timestamp
                )

                # Re-filter remaining candidate actions in case the new constraint blocks others
                allowed_actions, new_blocks = self.memory_engine.filter_candidate_actions(
                    current_condition=condition,
                    current_vector=vector,
                    candidate_actions=remaining_candidates,
                    current_time=timestamp,
                    theta_sim=self.theta_sim,
                    theta_conf=self.theta_conf
                )
                remaining_candidates = allowed_actions

        record = IncidentExecutionRecord(
            incident_id=incident_id,
            condition=condition,
            controller_name=self.name,
            telemetry_raw=raw_telemetry,
            telemetry_vector=vector,
            attempts=attempts,
            tried_actions=tried_actions,
            successful_action=successful_action,
            blocked_actions=blocked_details,
            first_attempt_success=(attempts == 1),
            repeated_failures=repeated_failures,
            mttr_seconds=mttr_seconds,
            toxic_failures_encountered=toxic_count,
            timestamp=timestamp,
            execution_trace=trace
        )
        self.history.append(record)
        return record
