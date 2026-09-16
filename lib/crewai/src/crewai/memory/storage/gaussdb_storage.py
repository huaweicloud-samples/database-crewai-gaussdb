"""GaussDB storage backend for the unified memory system.

Contract and behavior mirror
:class:`~crewai.memory.storage.lancedb_storage.LanceDBStorage`: search
prefetches ``limit * 3`` when Python-side JSON filters are given, scope
filtering is a prefix ``LIKE``, and ``delete`` returns the number of removed
rows. Vectors live in a ``floatvector`` column ranked by ``<+>`` cosine
distance with ``score = clamp(1.0 - distance, 0.0, 1.0)``.

Schema notes (GaussDB O-mode facts verified in Plan 1):
- ``private`` is SMALLINT 0/1 because there is no implicit text→boolean
  assignment cast, so BOOLEAN columns cannot be written through
  ``upsert_via_merge``.
- ``content`` "" is stored as a single space: the O-mode driver turns empty
  strings into NULL, which the NOT NULL column would reject. The space is
  NOT restored on read.
- ``created_at``/``last_accessed`` are ISO-8601 text: lexicographic order
  equals chronological order, so ordering and range comparisons work.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
from typing import TYPE_CHECKING, Any

from crewai_core.lock_store import lock as store_lock

from crewai.gaussdb.config import GaussDBConfig
from crewai.gaussdb.connection import cursor
from crewai.gaussdb.vector import (
    _vector_literal,
    ensure_vector_index,
    upsert_via_merge,
)
from crewai.memory.storage.backend import EmbeddingDimensionMismatchError
from crewai.memory.types import MemoryRecord, ScopeInfo


if TYPE_CHECKING:
    # N812: aliasing the lowercase builtin-ish name `cursor` to PascalCase.
    from psycopg2.extensions import cursor as Psycopg2Cursor  # noqa: N812

_logger = logging.getLogger(__name__)

_TABLE = "memories"

# Identifiers are static schema constants; all values travel bound parameters.
_CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS memories ("
    "id VARCHAR(64) PRIMARY KEY, "
    "content TEXT NOT NULL, "
    "scope TEXT NOT NULL DEFAULT '/', "
    "categories TEXT NOT NULL DEFAULT '[]', "
    "metadata TEXT NOT NULL DEFAULT '{}', "
    "importance DOUBLE PRECISION NOT NULL DEFAULT 0.5, "
    "created_at VARCHAR(64) NOT NULL, "
    "last_accessed VARCHAR(64) NOT NULL, "
    "source VARCHAR(512), "
    "private SMALLINT NOT NULL DEFAULT 0, "
    "embedding floatvector({dim}) NOT NULL)"
)
_CREATE_SCOPE_INDEX = "CREATE INDEX IF NOT EXISTS idx_memories_scope ON memories(scope)"
_VECTOR_INDEX = "idx_memories_embedding"

# Reading order, shared by get_record/list_records/delete-scan. The embedding
# column is deliberately not selected: MemoryRecord.embedding stays None on
# read (the field is exclude=True and recall never needs the raw vector).
_COLUMNS = (
    "id, content, scope, categories, metadata, importance, "
    "created_at, last_accessed, source, private"
)


def _scope_like(scope_prefix: str | None) -> str | None:
    """Return the LIKE pattern for *scope_prefix*, or None for no filtering.

    None/root prefixes mean "all scopes" (LanceDB behavior). Prefixes are
    normalized to a leading slash; wildcards are escaped (see
    :meth:`GaussDBStorage._scope_pattern`). Like LanceDB, the trailing
    ``%`` also matches longer scope names sharing the prefix (``/test%``
    matches ``/testx``).
    """
    if scope_prefix is None or not scope_prefix.strip("/"):
        return None
    prefix = scope_prefix.rstrip("/")
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    return GaussDBStorage._scope_pattern(prefix)


class GaussDBStorage:
    """GaussDB-backed storage backend for the unified memory system.

    Behavior mirrors LanceDBStorage: search prefetches 3x when Python-side
    filters are given, scopes filter via prefix LIKE, delete returns the
    removed-row count. Vectors use floatvector with <+> cosine distance and
    score = clamp(1.0 - distance, 0, 1).
    """

    def __init__(self, config: GaussDBConfig | None = None) -> None:
        self.config = config or GaussDBConfig.from_env()
        # Dimension of the stored vectors; None until the first save/update
        # establishes it (and after reset drops the table). No DB connection
        # is opened here — the backend connects lazily on first use.
        self._dim: int | None = None
        self._lock_name = (
            f"gaussdb:{self.config.host}:{self.config.port}"
            f":{self.config.database}:memories"
        )

    # ---- schema helpers ----

    @staticmethod
    def _scope_pattern(prefix: str) -> str:
        """Escape LIKE wildcards in *prefix* and append the trailing '%'.

        GaussDB uses the backslash as the default LIKE escape character, so
        no ESCAPE clause is required.
        """
        escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return escaped + "%"

    def _table_exists(self, cur: Psycopg2Cursor) -> bool:
        cur.execute("SELECT 1 FROM pg_tables WHERE tablename = %s", (_TABLE,))
        return cur.fetchone() is not None

    def _ensure_schema(self, cur: Psycopg2Cursor, dim: int) -> None:
        """Create the table and vector index (idempotently).

        Caller must already hold ``store_lock(self._lock_name)``. When the
        table already exists and holds vectors of a different dimension the
        save is rejected with ``EmbeddingDimensionMismatchError``; an empty
        existing table skips the check (nothing stored to compare against).
        """
        if self._table_exists(cur):
            cur.execute("SELECT vector_dims(embedding) FROM memories LIMIT 1")
            row = cur.fetchone()
            if row is not None and int(row[0]) != dim:
                raise EmbeddingDimensionMismatchError(int(row[0]), dim)
        else:
            # .format() would trip over the literal '{}' JSON defaults below.
            cur.execute(_CREATE_TABLE.replace("{dim}", str(dim)))
        cur.execute(_CREATE_SCOPE_INDEX)
        ensure_vector_index(cur, _VECTOR_INDEX, _TABLE, "embedding", dim)
        self._dim = dim

    # ---- row mapping ----

    @staticmethod
    def _row_to_record(row: tuple[Any, ...]) -> MemoryRecord:
        return MemoryRecord(
            id=row[0],
            content=row[1],
            scope=row[2],
            categories=json.loads(row[3]) if row[3] else [],
            metadata=json.loads(row[4]) if row[4] else {},
            importance=float(row[5]),
            created_at=datetime.fromisoformat(row[6]),
            last_accessed=datetime.fromisoformat(row[7]),
            embedding=None,
            source=row[8],
            private=bool(row[9]),
        )

    @staticmethod
    def _passes_filters(
        record: MemoryRecord,
        categories: list[str] | None,
        metadata_filter: dict[str, Any] | None,
    ) -> bool:
        """LanceDB filter semantics: categories any-match, metadata all-match."""
        if categories and not any(c in record.categories for c in categories):
            return False
        if metadata_filter and not all(
            record.metadata.get(k) == v for k, v in metadata_filter.items()
        ):
            return False
        return True

    # ---- StorageBackend protocol ----

    def save(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        dim: int | None = None
        payload: list[dict[str, Any]] = []
        for rec in records:
            embedding = rec.embedding
            if not embedding:
                _logger.warning(
                    "Skipping memory record %s: no embedding to store", rec.id
                )
                continue
            if dim is None:
                dim = len(embedding)
            elif len(embedding) != dim:
                raise EmbeddingDimensionMismatchError(dim, len(embedding))
            payload.append(
                {
                    "id": rec.id,
                    "content": rec.content if rec.content != "" else " ",
                    "scope": rec.scope,
                    "categories": json.dumps(rec.categories),
                    "metadata": json.dumps(rec.metadata),
                    "importance": rec.importance,
                    "created_at": rec.created_at.isoformat(),
                    "last_accessed": rec.last_accessed.isoformat(),
                    "source": rec.source,
                    "private": 1 if rec.private else 0,
                    "embedding": embedding,
                }
            )
        if not payload:
            return
        with store_lock(self._lock_name):
            with cursor(self.config) as cur:
                self._ensure_schema(cur, len(payload[0]["embedding"]))
                upsert_via_merge(cur, _TABLE, ["id"], payload, "embedding")

    def update(self, record: MemoryRecord) -> None:
        """Update a record by ID (MERGE upsert; same semantics as save)."""
        self.save([record])

    def touch_records(
        self, record_ids: list[str], accessed_at: datetime | None = None
    ) -> int:
        """Update last_accessed for the given records (recall side-effect).

        Mirrors LanceDB's ``touch_records`` (same single-column UPDATE,
        ``store_lock`` held, no-op on empty ids) but parameterized — safer
        than LanceDB's string-concatenated WHERE. Returns the number of rows
        updated; a missing table returns 0, matching ``get_record``.

        The default timestamp is naive UTC (``datetime.utcnow``), same as
        LanceDB: ``last_accessed`` ISO strings must stay naive so
        lexicographic order keeps matching chronological order.
        """
        if not record_ids:
            return 0
        timestamp = (accessed_at or datetime.utcnow()).isoformat()
        with store_lock(self._lock_name), cursor(self.config) as cur:
            if not self._table_exists(cur):
                return 0
            cur.execute(
                "UPDATE memories SET last_accessed = %s WHERE id = ANY(%s)",
                (timestamp, record_ids),
            )
            return int(cur.rowcount)

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        if self._dim is not None and len(query_embedding) != self._dim:
            raise EmbeddingDimensionMismatchError(self._dim, len(query_embedding))
        pattern = _scope_like(scope_prefix)
        # Prefetch 3x only when Python-side JSON filters may drop rows
        # (LanceDB behavior); min_score alone never widens the fetch.
        prefetch = limit * 3 if (categories or metadata_filter) else limit
        sql = f"SELECT {_COLUMNS}, embedding <+> %s AS distance FROM {_TABLE}"  # noqa: S608
        params: list[Any] = [_vector_literal(query_embedding)]
        if pattern is not None:
            sql += " WHERE scope LIKE %s"
            params.append(pattern)
        sql += " ORDER BY distance LIMIT %s"
        params.append(prefetch)
        with cursor(self.config) as cur:
            if not self._table_exists(cur):
                return []
            # Set the probe GUCs for this session (same values as
            # ensure_vector_index): pooled recall-only processes never run
            # save(), so without this they search with default probes and
            # recall quality silently degrades. Unknown dim (no save yet in
            # this process) — skip; the default probes are functionally
            # correct, just less thorough.
            if self._dim is not None and self._dim > 1024:
                cur.execute("SET diskann_probe_ncandidates = 200")
            elif self._dim is not None:
                cur.execute("SET gsivfflat_probes = 25")
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        out: list[tuple[MemoryRecord, float]] = []
        for row in rows:
            distance = float(row[10])
            score = max(0.0, min(1.0, 1.0 - distance))
            if score < min_score:
                continue
            record = self._row_to_record(row)
            if not self._passes_filters(record, categories, metadata_filter):
                continue
            out.append((record, score))
            if len(out) >= limit:
                break
        return out

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        if record_ids is not None and not record_ids:
            return 0
        pattern = _scope_like(scope_prefix)
        with store_lock(self._lock_name), cursor(self.config) as cur:
            if not self._table_exists(cur):
                return 0
            if categories or metadata_filter:
                # categories/metadata are JSON text columns — scan then filter
                # in Python (LanceDB semantics), deleting the surviving ids.
                sql = f"SELECT {_COLUMNS} FROM {_TABLE}"  # noqa: S608
                params: list[Any] = []
                conditions: list[str] = []
                if pattern is not None:
                    conditions.append("scope LIKE %s")
                    params.append(pattern)
                if older_than is not None:
                    conditions.append("created_at < %s")
                    params.append(older_than.isoformat())
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                cur.execute(sql, tuple(params))
                ids = [
                    row[0]
                    for row in cur.fetchall()
                    if self._passes_filters(
                        self._row_to_record(row), categories, metadata_filter
                    )
                ]
                if not ids:
                    return 0
                cur.execute(
                    f"DELETE FROM {_TABLE} WHERE id IN %s",  # noqa: S608
                    (tuple(ids),),
                )
                return int(cur.rowcount)
            conditions = []
            params = []
            if pattern is not None:
                conditions.append("scope LIKE %s")
                params.append(pattern)
            if record_ids:
                conditions.append("id IN %s")
                params.append(tuple(record_ids))
            if older_than is not None:
                conditions.append("created_at < %s")
                params.append(older_than.isoformat())
            sql = f"DELETE FROM {_TABLE}"  # noqa: S608
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            cur.execute(sql, tuple(params) if params else None)
            return int(cur.rowcount)

    def get_record(self, record_id: str) -> MemoryRecord | None:
        with cursor(self.config) as cur:
            if not self._table_exists(cur):
                return None
            cur.execute(
                f"SELECT {_COLUMNS} FROM {_TABLE} WHERE id = %s",  # noqa: S608
                (record_id,),
            )
            row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        pattern = _scope_like(scope_prefix)
        sql = f"SELECT {_COLUMNS} FROM {_TABLE}"  # noqa: S608
        params: list[Any] = []
        if pattern is not None:
            sql += " WHERE scope LIKE %s"
            params.append(pattern)
        sql += " ORDER BY created_at DESC, id OFFSET %s LIMIT %s"
        params.extend([offset, limit])
        with cursor(self.config) as cur:
            if not self._table_exists(cur):
                return []
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        normalized = scope.rstrip("/") or "/"
        prefix = "" if normalized == "/" else normalized
        with cursor(self.config) as cur:
            if not self._table_exists(cur):
                rows: list[tuple[Any, ...]] = []
            elif prefix:
                cur.execute(
                    "SELECT scope, categories, created_at FROM memories"
                    " WHERE scope LIKE %s",
                    (GaussDBStorage._scope_pattern(prefix),),
                )
                rows = cur.fetchall()
            else:
                cur.execute("SELECT scope, categories, created_at FROM memories")
                rows = cur.fetchall()
        if not rows:
            return ScopeInfo(
                path=normalized or "/",
                record_count=0,
                categories=[],
                oldest_record=None,
                newest_record=None,
                child_scopes=[],
            )
        child_prefix = (prefix + "/") if prefix else "/"
        children: set[str] = set()
        categories: set[str] = set()
        oldest: str | None = None
        newest: str | None = None
        for row in rows:
            scope_path = str(row[0])
            if scope_path.startswith(child_prefix):
                first_component = scope_path[len(child_prefix) :].split("/", 1)[0]
                if first_component:
                    children.add(child_prefix + first_component)
            try:
                categories.update(json.loads(row[1] or "[]"))
            except (TypeError, ValueError):
                pass  # corrupt JSON: ignore the row's categories
            created = row[2]
            if created:
                # ISO text: lexicographic min/max == chronological min/max.
                if oldest is None or created < oldest:
                    oldest = created
                if newest is None or created > newest:
                    newest = created
        return ScopeInfo(
            path=normalized or "/",
            record_count=len(rows),
            categories=sorted(categories),
            oldest_record=datetime.fromisoformat(oldest) if oldest else None,
            newest_record=datetime.fromisoformat(newest) if newest else None,
            child_scopes=sorted(children),
        )

    def list_scopes(self, parent: str = "/") -> list[str]:
        parent = parent.rstrip("/") or ""
        prefix = (parent + "/") if parent else "/"
        with cursor(self.config) as cur:
            if not self._table_exists(cur):
                return []
            if prefix != "/":
                cur.execute(
                    "SELECT DISTINCT scope FROM memories WHERE scope LIKE %s",
                    (GaussDBStorage._scope_pattern(prefix),),
                )
            else:
                cur.execute("SELECT DISTINCT scope FROM memories")
            rows = cur.fetchall()
        root_path = prefix.rstrip("/") or "/"
        children: set[str] = set()
        for row in rows:
            scope_path = str(row[0])
            if scope_path.startswith(prefix) and scope_path != root_path:
                first_component = scope_path[len(prefix) :].split("/", 1)[0]
                if first_component:
                    children.add(prefix + first_component)
        return sorted(children)

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        pattern = _scope_like(scope_prefix)
        with cursor(self.config) as cur:
            if not self._table_exists(cur):
                return {}
            if pattern is not None:
                cur.execute(
                    "SELECT categories FROM memories WHERE scope LIKE %s", (pattern,)
                )
            else:
                cur.execute("SELECT categories FROM memories")
            rows = cur.fetchall()
        counts: dict[str, int] = {}
        for row in rows:
            try:
                parsed = json.loads(row[0] or "[]")
            except (TypeError, ValueError):
                continue
            for category in parsed:
                counts[category] = counts.get(category, 0) + 1
        return counts

    def count(self, scope_prefix: str | None = None) -> int:
        pattern = _scope_like(scope_prefix)
        with cursor(self.config) as cur:
            if not self._table_exists(cur):
                return 0
            if pattern is not None:
                cur.execute(
                    f"SELECT count(*) FROM {_TABLE} WHERE scope LIKE %s",  # noqa: S608
                    (pattern,),
                )
            else:
                cur.execute(f"SELECT count(*) FROM {_TABLE}")  # noqa: S608
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def reset(self, scope_prefix: str | None = None) -> None:
        with store_lock(self._lock_name):
            with cursor(self.config) as cur:
                if not self._table_exists(cur):
                    return
                if scope_prefix is None or not scope_prefix.strip("/"):
                    cur.execute(f"DROP TABLE IF EXISTS {_TABLE}")
                    self._dim = None
                    return
                cur.execute(
                    f"DELETE FROM {_TABLE} WHERE scope LIKE %s",  # noqa: S608
                    (GaussDBStorage._scope_pattern(scope_prefix.rstrip("/")),),
                )

    async def asave(self, records: list[MemoryRecord]) -> None:
        await asyncio.to_thread(self.save, records)

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        return await asyncio.to_thread(
            self.search,
            query_embedding,
            scope_prefix,
            categories,
            metadata_filter,
            limit,
            min_score,
        )

    async def adelete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self.delete,
            scope_prefix,
            categories,
            record_ids,
            older_than,
            metadata_filter,
        )
