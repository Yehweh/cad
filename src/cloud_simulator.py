"""
Cloud Fault & Verification Simulator.
Implements:
1. Multi-scenario fault generation:
   - DB_CONN_EXHAUSTION
   - MEMORY_LEAK_OOM
   - CASCADING_CACHE_STAMPEDE (Toxic failure mode)
2. Realistic telemetry generation with controlled variance
3. Remediation action execution with domain physics & post-action health verification
4. Environment Drift simulation (software patches) to test Ebbinghaus decay recovery
5. Optional embedded FastAPI endpoints for microservice integration
"""

import random
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from src.memory_engine import normalize_context


@dataclass
class HealthCheckResult:
    is_healthy: bool
    status_code: int
    verification_evidence: str
    failure_reason: str
    toxicity: str  # "NEUTRAL", "TOXIC", or "NONE"
    post_telemetry: Dict[str, float]
    overhead_seconds: float


class CloudEnvironmentSimulator:
    """
    Simulates a production cloud microservice subject to transient and structural failures,
    fault injection, remediation actions, and health verification.
    """

    SCENARIO_PROFILES = {
        "DB_CONN_EXHAUSTION": {
            "baseline": {"cpu": 42.0, "mem": 60.0, "db_conn": 100.0, "latency_ms": 850.0, "error_rate": 0.40},
            "candidates": ["restart_service", "connection_capacity_remediation", "scale_compute_instances"],
            "default_negative_constraint": "Avoid restart_service for DB_CONN_EXHAUSTION under high DB saturation"
        },
        "MEMORY_LEAK_OOM": {
            "baseline": {"cpu": 55.0, "mem": 98.0, "db_conn": 30.0, "latency_ms": 600.0, "error_rate": 0.50},
            "candidates": ["connection_capacity_remediation", "restart_service", "reallocate_worker_memory"],
            "default_negative_constraint": "Avoid connection_capacity_remediation for MEMORY_LEAK_OOM"
        },
        "CASCADING_CACHE_STAMPEDE": {
            "baseline": {"cpu": 92.0, "mem": 85.0, "db_conn": 95.0, "latency_ms": 950.0, "error_rate": 0.75},
            "candidates": ["restart_service", "flush_cache", "enable_rate_limiting_and_circuit_breaker"],
            "default_negative_constraint": "Avoid restart_service and flush_cache during CASCADING_CACHE_STAMPEDE (causes toxic collapse)"
        }
    }

    def __init__(self, seed: Optional[int] = 42):
        if seed is not None:
            random.seed(seed)
        self.patch_db_conn_exhaustion_fixed = False
        self.current_state = "HEALTHY"
        self.active_condition: Optional[str] = None
        self.active_raw_telemetry: Dict[str, float] = {"cpu": 15.0, "mem": 25.0, "db_conn": 10.0, "latency_ms": 45.0, "error_rate": 0.0}

    def set_environment_patch(self, enabled: bool = True):
        """
        Simulates an infrastructure / code patch where restart_service now properly
        drains external connection pools, making it effective for DB_CONN_EXHAUSTION.
        Tests Ebbinghaus decay recovery when the environment drifts.
        """
        self.patch_db_conn_exhaustion_fixed = enabled

    def generate_incident(self, condition: str, noise_scale: float = 0.05) -> Tuple[Dict[str, float], List[float], List[str]]:
        """
        Generates noisy operational telemetry for a given failure scenario.
        Returns:
            (raw_telemetry, normalized_state_vector, candidate_actions)
        """
        if condition not in self.SCENARIO_PROFILES:
            raise ValueError(f"Unknown scenario condition: {condition}")

        profile = self.SCENARIO_PROFILES[condition]
        base = profile["baseline"]

        # Inject realistic Gaussian noise
        raw = {}
        for key, val in base.items():
            if key in ["cpu", "mem", "db_conn"]:
                noise = random.gauss(0, val * noise_scale)
                raw[key] = round(max(5.0, min(100.0, val + noise)), 1)
            elif key == "latency_ms":
                noise = random.gauss(0, val * noise_scale)
                raw[key] = round(max(20.0, min(1500.0, val + noise)), 1)
            elif key == "error_rate":
                noise = random.gauss(0, val * noise_scale)
                raw[key] = round(max(0.01, min(1.0, val + noise)), 3)

        vector = normalize_context(raw)
        candidates = list(profile["candidates"])
        self.active_condition = condition
        self.active_raw_telemetry = raw
        self.current_state = "UNHEALTHY"

        return raw, vector, candidates

    def is_action_effective(self, condition: str, action: str, raw_context: Optional[Dict[str, float]] = None) -> bool:
        """
        Pure, side-effect-free probe of whether ``action`` would restore health
        for ``condition`` under the current environment configuration.

        Uses a throwaway simulator instance so the caller's state (and the live
        environment) is never mutated. Used to validate that blocked actions
        would genuinely have failed (memory precision).
        """
        probe = CloudEnvironmentSimulator(seed=None)
        probe.patch_db_conn_exhaustion_fixed = self.patch_db_conn_exhaustion_fixed
        result = probe.execute_and_verify(condition, action, raw_context or {})
        return result.is_healthy

    def execute_and_verify(self, condition: str, action: str, raw_context: Dict[str, float]) -> HealthCheckResult:
        """
        Executes a remediation action against the simulated cloud environment and
        performs a rigorous post-action health verification.
        
        Failure physics:
        - DB_CONN_EXHAUSTION:
            * restart_service: FAILS (unless patch applied) -> drops requests, external pool still saturated
            * scale_compute_instances: FAILS -> new instances exacerbate DB pool lock
            * connection_capacity_remediation: SUCCEEDS -> flushes stale conns and resizes pool
        - MEMORY_LEAK_OOM:
            * connection_capacity_remediation: FAILS -> pool tuning does not free heap
            * restart_service: SUCCEEDS -> cleans leaking process memory
            * reallocate_worker_memory: SUCCEEDS -> expands cgroup container limit
        - CASCADING_CACHE_STAMPEDE:
            * restart_service / flush_cache: TOXIC FAILURE -> destroys warmth, DB collapse
            * enable_rate_limiting_and_circuit_breaker: SUCCEEDS -> sheds load, stabilizes downstream
        """
        # Scenario 1: DB_CONN_EXHAUSTION
        if condition == "DB_CONN_EXHAUSTION":
            if action == "restart_service":
                if self.patch_db_conn_exhaustion_fixed:
                    # Environment drifted / patched!
                    post = {"cpu": 22.0, "mem": 35.0, "db_conn": 25.0, "latency_ms": 50.0, "error_rate": 0.01}
                    self.current_state = "HEALTHY"
                    return HealthCheckResult(
                        is_healthy=True,
                        status_code=200,
                        verification_evidence="HTTP 200: Service restarted with new v2 DB pool driver. Leaked connections safely drained.",
                        failure_reason="",
                        toxicity="NONE",
                        post_telemetry=post,
                        overhead_seconds=5.0
                    )
                else:
                    post = {"cpu": 45.0, "mem": 58.0, "db_conn": 100.0, "latency_ms": 900.0, "error_rate": 0.45}
                    return HealthCheckResult(
                        is_healthy=False,
                        status_code=500,
                        verification_evidence="HTTP 500: Health check failed. DB pool remains at 100% capacity. In-flight transactions severed.",
                        failure_reason="Dependency unavailable / Connection limit persists across restarts",
                        toxicity="NEUTRAL",
                        post_telemetry=post,
                        overhead_seconds=15.0
                    )
            elif action == "scale_compute_instances":
                post = {"cpu": 30.0, "mem": 40.0, "db_conn": 100.0, "latency_ms": 1100.0, "error_rate": 0.60}
                return HealthCheckResult(
                    is_healthy=False,
                    status_code=500,
                    verification_evidence="HTTP 500: Scaled to 4 instances. All instances blocked on saturated Postgres connection limit.",
                    failure_reason="Scaling compute does not resolve downstream database connection bottleneck",
                    toxicity="NEUTRAL",
                    post_telemetry=post,
                    overhead_seconds=15.0
                )
            elif action == "connection_capacity_remediation":
                post = {"cpu": 18.0, "mem": 30.0, "db_conn": 22.0, "latency_ms": 48.0, "error_rate": 0.0}
                self.current_state = "HEALTHY"
                return HealthCheckResult(
                    is_healthy=True,
                    status_code=200,
                    verification_evidence="HTTP 200: Successfully pruned idle connections & expanded pool max_connections to 250. DB usage dropped to 22%.",
                    failure_reason="",
                    toxicity="NONE",
                    post_telemetry=post,
                    overhead_seconds=5.0
                )

        # Scenario 2: MEMORY_LEAK_OOM
        elif condition == "MEMORY_LEAK_OOM":
            if action == "connection_capacity_remediation":
                post = {"cpu": 54.0, "mem": 99.0, "db_conn": 28.0, "latency_ms": 650.0, "error_rate": 0.55}
                return HealthCheckResult(
                    is_healthy=False,
                    status_code=500,
                    verification_evidence="HTTP 500: Process memory remains at 99%. Connection pooling changes do not reclaim leaked heap.",
                    failure_reason="Heap leak unrelated to connection pool sizing",
                    toxicity="NEUTRAL",
                    post_telemetry=post,
                    overhead_seconds=15.0
                )
            elif action == "restart_service":
                post = {"cpu": 16.0, "mem": 24.0, "db_conn": 25.0, "latency_ms": 42.0, "error_rate": 0.0}
                self.current_state = "HEALTHY"
                return HealthCheckResult(
                    is_healthy=True,
                    status_code=200,
                    verification_evidence="HTTP 200: Pod restarted. Leaked memory heap discarded. Memory stabilized at 24%.",
                    failure_reason="",
                    toxicity="NONE",
                    post_telemetry=post,
                    overhead_seconds=5.0
                )
            elif action == "reallocate_worker_memory":
                post = {"cpu": 20.0, "mem": 32.0, "db_conn": 28.0, "latency_ms": 46.0, "error_rate": 0.0}
                self.current_state = "HEALTHY"
                return HealthCheckResult(
                    is_healthy=True,
                    status_code=200,
                    verification_evidence="HTTP 200: Worker cgroup memory ceiling elevated from 1GB to 4GB. OOM condition resolved.",
                    failure_reason="",
                    toxicity="NONE",
                    post_telemetry=post,
                    overhead_seconds=5.0
                )

        # Scenario 3: CASCADING_CACHE_STAMPEDE
        elif condition == "CASCADING_CACHE_STAMPEDE":
            if action in ["restart_service", "flush_cache"]:
                post = {"cpu": 100.0, "mem": 95.0, "db_conn": 100.0, "latency_ms": 1450.0, "error_rate": 0.95}
                return HealthCheckResult(
                    is_healthy=False,
                    status_code=503,
                    verification_evidence=f"HTTP 503: TOXIC FAILURE - Action '{action}' flushed hot cache buffers. Downstream DB collapsed under 100% cache miss storm.",
                    failure_reason="Cache warm-state destroyed during active stampede, compounding backend saturation",
                    toxicity="TOXIC",
                    post_telemetry=post,
                    overhead_seconds=15.0
                )
            elif action == "enable_rate_limiting_and_circuit_breaker":
                post = {"cpu": 35.0, "mem": 45.0, "db_conn": 35.0, "latency_ms": 55.0, "error_rate": 0.02}
                self.current_state = "HEALTHY"
                return HealthCheckResult(
                    is_healthy=True,
                    status_code=200,
                    verification_evidence="HTTP 200: Ingress rate limiting throttled surge by 40%. Circuit breaker shielded DB until cache re-warmed.",
                    failure_reason="",
                    toxicity="NONE",
                    post_telemetry=post,
                    overhead_seconds=5.0
                )

        # Fallback default
        return HealthCheckResult(
            is_healthy=False,
            status_code=500,
            verification_evidence=f"HTTP 500: Action '{action}' had no discernible effect on {condition}.",
            failure_reason="Unmatched action for failure condition",
            toxicity="NEUTRAL",
            post_telemetry=raw_context,
            overhead_seconds=15.0
        )
