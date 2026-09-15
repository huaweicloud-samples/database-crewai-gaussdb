"""GaussDB state provider for checkpointing.

Same contract as :class:`~crewai.state.provider.sqlite_provider.SqliteProvider`,
backed by GaussDB. Differences from SQLite:

- ``ORDER BY rowid`` (insertion order) has no PostgreSQL-family equivalent, so
  the table carries an explicit ``seq BIGINT DEFAULT nextval(...)`` column.
- ``data`` is a JSONB column; writes cast ``%s::jsonb``, reads use
  ``data::text`` so callers always receive the raw JSON string (psycopg2 would
  otherwise auto-parse jsonb into a dict).
- Locations use the ``"gaussdb#<checkpoint_id>"`` scheme; connection details
  come from ``GAUSSDB_*`` env vars.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import os
from typing import Literal
import uuid

from pydantic import Field, model_validator
from typing_extensions import Self

from crewai.gaussdb.config import GaussDBConfig
from crewai.state.provider.core import BaseProvider


_CREATE_SCHEMA_SQL: tuple[str, ...] = (
    "CREATE SEQUENCE IF NOT EXISTS checkpoints_seq START 1",
    """
    CREATE TABLE IF NOT EXISTS checkpoints (
        id VARCHAR(64) PRIMARY KEY,
        seq BIGINT DEFAULT nextval('checkpoints_seq'),
        created_at VARCHAR(32) NOT NULL,
        parent_id VARCHAR(64),
        branch VARCHAR(64) NOT NULL DEFAULT 'main',
        data JSONB NOT NULL
    )
    """,
)

_INSERT_SQL = (
    "INSERT INTO checkpoints (id, created_at, parent_id, branch, data) "
    "VALUES (%s, %s, %s, %s, %s::jsonb)"
)
_SELECT_SQL = "SELECT data::text FROM checkpoints WHERE id = %s"
_PRUNE_SQL = """
DELETE FROM checkpoints WHERE branch = %s AND seq NOT IN (
    SELECT seq FROM checkpoints WHERE branch = %s ORDER BY seq DESC LIMIT %s
)
"""


def _make_id() -> tuple[str, str]:
    """Same ID scheme as the SQLite provider: ``<ts>_<uuid8>``."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    checkpoint_id = f"{ts}_{uuid.uuid4().hex[:8]}"
    return checkpoint_id, ts


class GaussDBProvider(BaseProvider):
    """Persists runtime state checkpoints in a GaussDB database."""

    provider_type: Literal["gaussdb"] = Field(default="gaussdb")
    config: GaussDBConfig = Field(default_factory=GaussDBConfig.from_env)

    @model_validator(mode="after")
    def _fill_password_from_env(self) -> Self:
        """Restored-from-payload providers carry an empty password (excluded
        from serialization); re-fill it from the environment, matching the
        connection-from-env design."""
        if not self.config.password:
            env_password = os.environ.get("GAUSSDB_PASSWORD", "")
            if env_password:
                self.config = self.config.model_copy(update={"password": env_password})
        return self

    def checkpoint(
        self,
        data: str,
        location: str,
        *,
        parent_id: str | None = None,
        branch: str = "main",
    ) -> str:
        """Write a checkpoint. *location* is accepted for interface parity but
        ignored; the returned location is ``"gaussdb#<checkpoint_id>"``."""
        from crewai.gaussdb.connection import cursor

        checkpoint_id, ts = _make_id()
        with cursor(self.config) as cur:
            for statement in _CREATE_SCHEMA_SQL:
                cur.execute(statement)
            cur.execute(_INSERT_SQL, (checkpoint_id, ts, parent_id, branch, data))
        return f"gaussdb#{checkpoint_id}"

    async def acheckpoint(
        self,
        data: str,
        location: str,
        *,
        parent_id: str | None = None,
        branch: str = "main",
    ) -> str:
        return await asyncio.to_thread(
            self.checkpoint, data, location, parent_id=parent_id, branch=branch
        )

    def prune(self, location: str, max_keep: int, *, branch: str = "main") -> int:
        from crewai.gaussdb.connection import cursor

        with cursor(self.config) as cur:
            cur.execute(_PRUNE_SQL, (branch, branch, max_keep))
            removed = cur.rowcount
        return max(removed, 0)

    def extract_id(self, location: str) -> str:
        return location.rsplit("#", 1)[1]

    def from_checkpoint(self, location: str) -> str:
        from crewai.gaussdb.connection import cursor

        checkpoint_id = location.rsplit("#", 1)[1]
        with cursor(self.config) as cur:
            cur.execute(_SELECT_SQL, (checkpoint_id,))
            row = cur.fetchone()
        if row is None:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        result: str = row[0]
        return result

    async def afrom_checkpoint(self, location: str) -> str:
        return await asyncio.to_thread(self.from_checkpoint, location)
