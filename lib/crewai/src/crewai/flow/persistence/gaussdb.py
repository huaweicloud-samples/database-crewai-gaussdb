"""GaussDB-backed implementation of flow state persistence."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import TYPE_CHECKING, Any

from crewai_core.lock_store import lock as store_lock
from pydantic import Field, PrivateAttr, model_validator
from typing_extensions import Self

from crewai.flow.persistence.base import FlowPersistence
from crewai.gaussdb.config import GaussDBConfig
from crewai.gaussdb.connection import cursor


if TYPE_CHECKING:
    import psycopg2.extensions

    from crewai.flow.async_feedback.types import PendingFeedbackContext


_CREATE_SCHEMA_SQL: tuple[str, ...] = (
    "CREATE SEQUENCE IF NOT EXISTS flow_states_id_seq START 1",
    """
    CREATE TABLE IF NOT EXISTS flow_states (
        id BIGINT PRIMARY KEY DEFAULT nextval('flow_states_id_seq'),
        flow_uuid VARCHAR(64) NOT NULL,
        method_name VARCHAR(255) NOT NULL,
        timestamp VARCHAR(64) NOT NULL,
        state_json TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_flow_states_uuid ON flow_states(flow_uuid)",
    """
    CREATE TABLE IF NOT EXISTS pending_feedback (
        flow_uuid VARCHAR(64) PRIMARY KEY,
        context_json TEXT NOT NULL,
        state_json TEXT NOT NULL,
        created_at VARCHAR(64) NOT NULL
    )
    """,
)

_INSERT_STATE_SQL = """
INSERT INTO flow_states (flow_uuid, method_name, timestamp, state_json)
VALUES (%s, %s, %s, %s)
"""

_LOAD_STATE_SQL = """
SELECT state_json FROM flow_states WHERE flow_uuid = %s ORDER BY id DESC LIMIT 1
"""

_SAVE_PENDING_SQL = """
MERGE INTO pending_feedback t
USING (SELECT %s AS flow_uuid, %s AS context_json, %s AS state_json, %s AS created_at) s
ON t.flow_uuid = s.flow_uuid
WHEN MATCHED THEN UPDATE
    SET context_json = s.context_json, state_json = s.state_json, created_at = s.created_at
WHEN NOT MATCHED THEN INSERT (flow_uuid, context_json, state_json, created_at)
    VALUES (s.flow_uuid, s.context_json, s.state_json, s.created_at)
"""

_LOAD_PENDING_SQL = """
SELECT state_json, context_json FROM pending_feedback WHERE flow_uuid = %s
"""

_CLEAR_PENDING_SQL = "DELETE FROM pending_feedback WHERE flow_uuid = %s"


def _to_state_dict(state_data: dict[str, Any] | Any) -> dict[str, Any]:
    """Convert state_data to a plain dict.

    Accepts dicts and any object exposing ``model_dump()`` (a superset of the
    SQLite backend's dict/BaseModel contract).
    """
    if isinstance(state_data, dict):
        return state_data
    dump = getattr(state_data, "model_dump", None)
    if callable(dump):
        dumped: dict[str, Any] = dump()
        return dumped
    raise ValueError(
        "state_data must be either a Pydantic BaseModel or dict, "
        f"got {type(state_data)}"
    )


def _save_state_sql(
    cur: psycopg2.extensions.cursor,
    flow_uuid: str,
    method_name: str,
    state_dict: dict[str, Any],
) -> None:
    """Execute the save-state INSERT without acquiring the lock."""
    cur.execute(
        _INSERT_STATE_SQL,
        (
            flow_uuid,
            method_name,
            datetime.now(timezone.utc).isoformat(),
            json.dumps(state_dict),
        ),
    )


class GaussDBFlowPersistence(FlowPersistence):
    """GaussDB-backed flow state persistence.

    Drop-in replacement for SQLiteFlowPersistence backed by a GaussDB
    (openGauss-compatible) database. Dialect notes: upsert via MERGE INTO
    (no ON CONFLICT in Oracle-compat mode); autoincrement via SEQUENCE
    (works on both centralized and distributed deployments).

    Example:
        ```python
        flow = MyFlow(persistence=GaussDBFlowPersistence())
        ```
    """

    persistence_type: str = Field(default="GaussDBFlowPersistence")
    config: GaussDBConfig = Field(default_factory=GaussDBConfig.from_env)
    _lock_name: str = PrivateAttr()

    @model_validator(mode="after")
    def _setup(self) -> Self:
        self._lock_name = (
            f"gaussdb:{self.config.host}:{self.config.port}:{self.config.database}"
        )
        self.init_db()
        return self

    def init_db(self) -> None:
        """Create tables/sequence/index if they don't exist."""
        with store_lock(self._lock_name), cursor(self.config) as cur:
            for statement in _CREATE_SCHEMA_SQL:
                cur.execute(statement)

    def save_state(
        self, flow_uuid: str, method_name: str, state_data: dict[str, Any] | Any
    ) -> None:
        """Save the current flow state to GaussDB.

        Args:
            flow_uuid: Unique identifier for the flow instance
            method_name: Name of the method that just completed
            state_data: Current state data (either dict or Pydantic model)
        """
        state_dict = _to_state_dict(state_data)
        with store_lock(self._lock_name), cursor(self.config) as cur:
            _save_state_sql(cur, flow_uuid, method_name, state_dict)

    def load_state(self, flow_uuid: str) -> dict[str, Any] | None:
        """Load the most recent state for a given flow UUID.

        Args:
            flow_uuid: Unique identifier for the flow instance

        Returns:
            The most recent state as a dictionary, or None if no state exists
        """
        with cursor(self.config) as cur:
            cur.execute(_LOAD_STATE_SQL, (flow_uuid,))
            row = cur.fetchone()
        if row:
            result = json.loads(row[0])
            return result if isinstance(result, dict) else None
        return None

    def save_pending_feedback(
        self,
        flow_uuid: str,
        context: PendingFeedbackContext,
        state_data: dict[str, Any] | Any,
    ) -> None:
        """Save state with a pending feedback marker.

        This method stores both the flow state and the pending feedback context,
        allowing the flow to be resumed later when feedback is received.

        Args:
            flow_uuid: Unique identifier for the flow instance
            context: The pending feedback context with all resume information
            state_data: Current state data
        """
        state_dict = _to_state_dict(state_data)
        with store_lock(self._lock_name), cursor(self.config) as cur:
            _save_state_sql(cur, flow_uuid, context.method_name, state_dict)
            cur.execute(
                _SAVE_PENDING_SQL,
                (
                    flow_uuid,
                    json.dumps(context.to_dict()),
                    json.dumps(state_dict),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def load_pending_feedback(
        self, flow_uuid: str
    ) -> tuple[dict[str, Any], PendingFeedbackContext] | None:
        """Load state and pending feedback context.

        Args:
            flow_uuid: Unique identifier for the flow instance

        Returns:
            Tuple of (state_data, pending_context) if pending feedback exists,
            None otherwise.
        """
        # Import here to avoid circular imports (mirrors the SQLite backend)
        from crewai.flow.async_feedback.types import PendingFeedbackContext

        with cursor(self.config) as cur:
            cur.execute(_LOAD_PENDING_SQL, (flow_uuid,))
            row = cur.fetchone()
        if row:
            state_dict = json.loads(row[0])
            context = PendingFeedbackContext.from_dict(json.loads(row[1]))
            return (state_dict, context)
        return None

    def clear_pending_feedback(self, flow_uuid: str) -> None:
        """Clear the pending feedback marker after successful resume.

        Args:
            flow_uuid: Unique identifier for the flow instance
        """
        with store_lock(self._lock_name), cursor(self.config) as cur:
            cur.execute(_CLEAR_PENDING_SQL, (flow_uuid,))
