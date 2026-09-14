"""GaussDB-backed storage for kickoff task outputs.

Same interface as
:class:`~crewai.memory.storage.kickoff_task_outputs_storage.KickoffTaskOutputsSQLiteStorage`
(add / update / load / delete_all), backed by GaussDB. Column order matches the
SQLite version exactly because ``load()`` maps columns by position.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from crewai_core.lock_store import lock as store_lock

from crewai.gaussdb.config import GaussDBConfig
from crewai.gaussdb.connection import cursor
from crewai.utilities.crew_json_encoder import CrewJSONEncoder
from crewai.utilities.errors import DatabaseError, DatabaseOperationError


if TYPE_CHECKING:
    import psycopg2.extensions

    from crewai.task import Task


logger = logging.getLogger(__name__)


_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS latest_kickoff_task_outputs (
    task_id VARCHAR(64) PRIMARY KEY,
    expected_output TEXT,
    output TEXT,
    task_index INTEGER,
    inputs TEXT,
    was_replayed BOOLEAN,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_ADD_SQL = """
MERGE INTO latest_kickoff_task_outputs t
USING (SELECT %s AS task_id, %s AS expected_output, %s AS output,
              %s AS task_index, %s AS inputs, %s AS was_replayed) s
ON t.task_id = s.task_id
WHEN MATCHED THEN UPDATE
    SET expected_output = s.expected_output, output = s.output,
        task_index = s.task_index, inputs = s.inputs, was_replayed = s.was_replayed
WHEN NOT MATCHED THEN INSERT (task_id, expected_output, output, task_index, inputs, was_replayed)
    VALUES (s.task_id, s.expected_output, s.output, s.task_index, s.inputs, s.was_replayed)
"""

_LOAD_SQL = """
SELECT task_id, expected_output, output, task_index, inputs, was_replayed, timestamp
FROM latest_kickoff_task_outputs
ORDER BY task_index
"""


def _add_task_output_sql(
    cur: psycopg2.extensions.cursor,
    task_id: str,
    expected_output: str | None,
    output_json: str,
    task_index: int,
    inputs_json: str,
    was_replayed: bool,
) -> None:
    """Execute the add-MERGE without acquiring the lock."""
    cur.execute(
        _ADD_SQL,
        (task_id, expected_output, output_json, task_index, inputs_json, was_replayed),
    )


class GaussDBKickoffTaskOutputsStorage:
    """GaussDB storage for kickoff task outputs (replay/audit trail).

    Drop-in replacement for KickoffTaskOutputsSQLiteStorage backed by a GaussDB
    (openGauss-compatible) database. Dialect note: upsert via MERGE INTO (no
    ON CONFLICT in Oracle-compat mode).
    """

    def __init__(self, config: GaussDBConfig | None = None) -> None:
        self.config = config or GaussDBConfig.from_env()
        self._lock_name = (
            f"gaussdb:{self.config.host}:{self.config.port}:"
            f"{self.config.database}:kickoff_outputs"
        )
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Create the latest_kickoff_task_outputs table if it doesn't exist.

        Raises:
            DatabaseOperationError: If database initialization fails.
        """
        try:
            with store_lock(self._lock_name), cursor(self.config) as cur:
                cur.execute(_CREATE_TABLE_SQL)
        except Exception as e:
            error_msg = DatabaseError.format_error(DatabaseError.INIT_ERROR, e)
            logger.error(error_msg)
            raise DatabaseOperationError(error_msg, e) from e

    def add(
        self,
        task: Task,
        output: dict[str, Any],
        task_index: int,
        was_replayed: bool = False,
        inputs: dict[str, Any] | None = None,
    ) -> None:
        """Add a new task output record (MERGE-overwrite by task_id).

        Args:
            task: The Task object containing task details.
            output: Dictionary containing the task's output data.
            task_index: Integer index of the task in the sequence.
            was_replayed: Boolean indicating if this was a replay execution.
            inputs: Dictionary of input parameters used for the task.

        Raises:
            DatabaseOperationError: If saving the task output fails.
        """
        inputs = inputs or {}
        try:
            with store_lock(self._lock_name), cursor(self.config) as cur:
                _add_task_output_sql(
                    cur,
                    str(task.id),
                    task.expected_output,
                    json.dumps(output, cls=CrewJSONEncoder),
                    task_index,
                    json.dumps(inputs, cls=CrewJSONEncoder),
                    was_replayed,
                )
        except Exception as e:
            error_msg = DatabaseError.format_error(DatabaseError.SAVE_ERROR, e)
            logger.error(error_msg)
            raise DatabaseOperationError(error_msg, e) from e

    def update(
        self,
        task_index: int,
        **kwargs: Any,
    ) -> None:
        """Update an existing task output record identified by task_index.

        Args:
            task_index: Integer index of the task to update.
            **kwargs: Arbitrary keyword arguments representing fields to update.
                     Values that are dictionaries will be JSON encoded.

        Raises:
            DatabaseOperationError: If updating the task output fails.
        """
        try:
            with store_lock(self._lock_name), cursor(self.config) as cur:
                fields = []
                values: list[Any] = []
                for key, value in kwargs.items():
                    fields.append(f"{key} = %s")
                    values.append(
                        json.dumps(value, cls=CrewJSONEncoder)
                        if isinstance(value, dict)
                        else value
                    )

                query = f"UPDATE latest_kickoff_task_outputs SET {', '.join(fields)} WHERE task_index = %s"  # nosec # noqa: S608
                values.append(task_index)

                cur.execute(query, tuple(values))

                if cur.rowcount == 0:
                    logger.warning(
                        f"No row found with task_index {task_index}. No update performed."
                    )
        except Exception as e:
            error_msg = DatabaseError.format_error(DatabaseError.UPDATE_ERROR, e)
            logger.error(error_msg)
            raise DatabaseOperationError(error_msg, e) from e

    def load(self) -> list[dict[str, Any]]:
        """Load all task output records ordered by task_index.

        Returns:
            List of dictionaries containing task output records. Each dictionary
            contains: task_id, expected_output, output, task_index, inputs,
            was_replayed, and timestamp.

        Raises:
            DatabaseOperationError: If loading task outputs fails.
        """
        try:
            with cursor(self.config) as cur:
                cur.execute(_LOAD_SQL)
                rows = cur.fetchall()
        except Exception as e:
            error_msg = DatabaseError.format_error(DatabaseError.LOAD_ERROR, e)
            logger.error(error_msg)
            raise DatabaseOperationError(error_msg, e) from e

        return [
            {
                "task_id": row[0],
                "expected_output": row[1],
                "output": json.loads(row[2]) if row[2] else None,
                "task_index": row[3],
                "inputs": json.loads(row[4]) if row[4] else None,
                "was_replayed": bool(row[5]),
                "timestamp": row[6].isoformat(sep=" ") if row[6] else None,
            }
            for row in rows
        ]

    def delete_all(self) -> None:
        """Delete all task output records from the table.

        Raises:
            DatabaseOperationError: If deleting task outputs fails.
        """
        try:
            with store_lock(self._lock_name), cursor(self.config) as cur:
                cur.execute("DELETE FROM latest_kickoff_task_outputs")
        except Exception as e:
            error_msg = DatabaseError.format_error(DatabaseError.DELETE_ERROR, e)
            logger.error(error_msg)
            raise DatabaseOperationError(error_msg, e) from e
