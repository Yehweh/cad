"""
Persistent incident execution history store.

Execution history is written to the *same* SQLite database that backs the
remediation memory (``outputs/remediation_memory.db``) so that a single
database file contains the full system state. History lives in its own table
(``incident_history``) and is therefore independent of the learned remediation
memory: clearing memory does not delete history, and deleting history does not
touch memory.

All writes use parameterized SQL. The schema is created with
``CREATE TABLE IF NOT EXISTS`` and additive column migrations so that existing
databases (and their negative constraints / positive outcomes) are preserved.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.paths import DEFAULT_DB_PATH, ensure_parent_dir, resolve_path

logger = logging.getLogger(__name__)


class IncidentHistoryStore:
    """SQLite-backed store for incident execution records."""

    def __init__(self, db_path: Optional[str] = DEFAULT_DB_PATH):
        self.db_path = resolve_path(db_path) if db_path else db_path
        self.last_error: Optional[str] = None
        if self.db_path:
            try:
                ensure_parent_dir(self.db_path)
                self._init_schema()
            except sqlite3.Error as exc:
                self.last_error = f"SQLite initialization failed: {exc}"
                logger.exception("Failed to initialize incident history database at %s", self.db_path)

    # ------------------------------------------------------------------
    # Connection / schema
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        conn = self._connect()
        try:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS incident_history (
                    run_id TEXT PRIMARY KEY,
                    timestamp REAL,
                    sim_time REAL,
                    incident_type TEXT,
                    telemetry TEXT,
                    controller TEXT,
                    attempted_actions TEXT,
                    attempts INTEGER,
                    blocked_by_memory INTEGER,
                    actions_blocked INTEGER,
                    blocked_actions TEXT,
                    action_outcomes TEXT,
                    recovery_success INTEGER,
                    recovery_duration REAL,
                    failure_details TEXT,
                    created_at TEXT
                )
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_timestamp "
                "ON incident_history(timestamp)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_incident_type "
                "ON incident_history(incident_type)"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_history_controller "
                "ON incident_history(controller)"
            )
            # Non-destructive migration for databases created before this column.
            existing = {r[1] for r in cur.execute("PRAGMA table_info(incident_history)")}
            if "actions_blocked" not in existing:
                cur.execute("ALTER TABLE incident_history ADD COLUMN actions_blocked INTEGER")
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def record_incident(self, record: Any, run_id: Optional[str] = None) -> Optional[str]:
        """
        Persists a single incident execution record.

        ``record`` may be an ``IncidentExecutionRecord`` instance or any object
        exposing the same attributes. Returns the generated ``run_id`` on
        success, or ``None`` if persistence failed.
        """
        run_id = run_id or str(uuid.uuid4())
        blocked_actions = list(getattr(record, "blocked_actions", []) or [])
        execution_trace = list(getattr(record, "execution_trace", []) or [])
        successful_action = getattr(record, "successful_action", "") or ""

        row = {
            "run_id": run_id,
            "timestamp": float(getattr(record, "timestamp", 0.0) or 0.0),
            "sim_time": float(getattr(record, "timestamp", 0.0) or 0.0),
            "incident_type": getattr(record, "condition", "UNKNOWN"),
            "telemetry": json.dumps(getattr(record, "telemetry_raw", {}) or {}),
            "controller": getattr(record, "controller_name", "Unknown"),
            "attempted_actions": json.dumps(getattr(record, "tried_actions", []) or []),
            "attempts": int(getattr(record, "attempts", 0) or 0),
            "blocked_by_memory": 1 if blocked_actions else 0,
            "actions_blocked": len(blocked_actions),
            "blocked_actions": json.dumps(blocked_actions),
            "action_outcomes": json.dumps(execution_trace),
            "recovery_success": 1 if successful_action else 0,
            "recovery_duration": float(getattr(record, "mttr_seconds", 0.0) or 0.0),
            "failure_details": json.dumps({
                "successful_action": successful_action,
                "first_attempt_success": bool(getattr(record, "first_attempt_success", False)),
                "repeated_failures": int(getattr(record, "repeated_failures", 0) or 0),
                "toxic_failures_encountered": int(getattr(record, "toxic_failures_encountered", 0) or 0),
                "incident_id": getattr(record, "incident_id", None),
            }),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        try:
            conn = self._connect()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT OR REPLACE INTO incident_history (
                        run_id, timestamp, sim_time, incident_type, telemetry,
                        controller, attempted_actions, attempts, blocked_by_memory,
                        actions_blocked, blocked_actions, action_outcomes, recovery_success,
                        recovery_duration, failure_details, created_at
                    ) VALUES (
                        :run_id, :timestamp, :sim_time, :incident_type, :telemetry,
                        :controller, :attempted_actions, :attempts, :blocked_by_memory,
                        :actions_blocked, :blocked_actions, :action_outcomes, :recovery_success,
                        :recovery_duration, :failure_details, :created_at
                    )
                    """,
                    row,
                )
                conn.commit()
            finally:
                conn.close()
            return run_id
        except sqlite3.Error as exc:
            self.last_error = f"Failed to persist incident history: {exc}"
            logger.exception("Failed to persist incident history for run %s", run_id)
            return None

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        def _loads(value: Any, default: Any) -> Any:
            if value is None:
                return default
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return default

        return {
            "run_id": row["run_id"],
            "timestamp": row["timestamp"],
            "sim_time": row["sim_time"],
            "incident_type": row["incident_type"],
            "telemetry": _loads(row["telemetry"], {}),
            "controller": row["controller"],
            "attempted_actions": _loads(row["attempted_actions"], []),
            "attempts": row["attempts"],
            "blocked_by_memory": bool(row["blocked_by_memory"]),
            "actions_blocked": row["actions_blocked"] or 0,
            "blocked_actions": _loads(row["blocked_actions"], []),
            "action_outcomes": _loads(row["action_outcomes"], []),
            "recovery_success": bool(row["recovery_success"]),
            "recovery_duration": row["recovery_duration"],
            "failure_details": _loads(row["failure_details"], {}),
            "created_at": row["created_at"],
        }

    def fetch_incidents(
        self,
        incident_type: Optional[str] = None,
        controller: Optional[str] = None,
        success: Optional[bool] = None,
        blocked_only: bool = False,
        limit: int = 200,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Returns incident history rows matching the supplied filters."""
        if not self.db_path:
            return []
        clauses: List[str] = []
        params: List[Any] = []
        if incident_type:
            clauses.append("incident_type = ?")
            params.append(incident_type)
        if controller:
            clauses.append("controller = ?")
            params.append(controller)
        if success is not None:
            clauses.append("recovery_success = ?")
            params.append(1 if success else 0)
        if blocked_only:
            clauses.append("blocked_by_memory = 1")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            f"SELECT * FROM incident_history {where} "
            "ORDER BY timestamp DESC, created_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([max(0, int(limit)), max(0, int(offset))])

        try:
            conn = self._connect()
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(query, params).fetchall()
            finally:
                conn.close()
            return [self._row_to_dict(r) for r in rows]
        except sqlite3.Error as exc:
            self.last_error = f"Failed to read incident history: {exc}"
            logger.exception("Failed to read incident history")
            return []

    def count_incidents(self, incident_type: Optional[str] = None) -> int:
        if not self.db_path:
            return 0
        try:
            conn = self._connect()
            try:
                if incident_type:
                    row = conn.execute(
                        "SELECT COUNT(*) FROM incident_history WHERE incident_type = ?",
                        (incident_type,),
                    ).fetchone()
                else:
                    row = conn.execute("SELECT COUNT(*) FROM incident_history").fetchone()
            finally:
                conn.close()
            return int(row[0]) if row else 0
        except sqlite3.Error as exc:
            self.last_error = f"Failed to count incident history: {exc}"
            logger.exception("Failed to count incident history")
            return 0

    def distinct_incident_types(self) -> List[str]:
        if not self.db_path:
            return []
        try:
            conn = self._connect()
            try:
                rows = conn.execute(
                    "SELECT DISTINCT incident_type FROM incident_history "
                    "WHERE incident_type IS NOT NULL ORDER BY incident_type"
                ).fetchall()
            finally:
                conn.close()
            return [r[0] for r in rows]
        except sqlite3.Error:
            logger.exception("Failed to list incident types")
            return []

    def aggregate_metrics(self) -> Dict[str, Any]:
        """
        Computes overview metrics from persisted history.

        Returns ``success_rate=None`` and ``mean_recovery_time=None`` when
        there is no data, so an empty dataset is never reported as 100%.
        """
        empty = {
            "total_incidents": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "success_rate": None,
            "mean_recovery_time": None,
            "actions_blocked": 0,
            "incidents_with_blocks": 0,
            "total_attempts": 0,
            "last_incident_at": None,
        }
        if not self.db_path:
            return empty
        try:
            conn = self._connect()
            conn.row_factory = sqlite3.Row
            try:
                row = conn.execute(
                    """
                    SELECT
                        COUNT(*) AS total,
                        COALESCE(SUM(recovery_success), 0) AS successes,
                        COALESCE(SUM(blocked_by_memory), 0) AS blocked_incidents,
                        COALESCE(SUM(actions_blocked), 0) AS blocked_count,
                        COALESCE(SUM(attempts), 0) AS total_attempts,
                        AVG(CASE WHEN recovery_success = 1 THEN recovery_duration END) AS mean_duration,
                        MAX(timestamp) AS last_ts
                    FROM incident_history
                    """
                ).fetchone()
            finally:
                conn.close()

            total = int(row["total"] or 0)
            successes = int(row["successes"] or 0)
            blocked_count = int(row["blocked_count"] or 0)

            if total == 0:
                return empty

            return {
                "total_incidents": total,
                "successful_recoveries": successes,
                "failed_recoveries": total - successes,
                "success_rate": (successes / total) * 100.0,
                "mean_recovery_time": float(row["mean_duration"]) if row["mean_duration"] is not None else None,
                "actions_blocked": blocked_count,
                "incidents_with_blocks": int(row["blocked_incidents"] or 0),
                "total_attempts": int(row["total_attempts"] or 0),
                "last_incident_at": float(row["last_ts"]) if row["last_ts"] is not None else None,
            }
        except sqlite3.Error as exc:
            self.last_error = f"Failed to aggregate incident metrics: {exc}"
            logger.exception("Failed to aggregate incident metrics")
            return empty

    # ------------------------------------------------------------------
    # Destructive operations
    # ------------------------------------------------------------------
    def clear_history(self) -> bool:
        """
        Deletes ONLY incident execution history.

        Remediation memory (negative constraints / positive outcomes) and
        benchmark artifacts are preserved.
        """
        if not self.db_path:
            return False
        try:
            conn = self._connect()
            try:
                conn.execute("DELETE FROM incident_history")
                conn.commit()
            finally:
                conn.close()
            return True
        except sqlite3.Error as exc:
            self.last_error = f"Failed to clear incident history: {exc}"
            logger.exception("Failed to clear incident history")
            return False
