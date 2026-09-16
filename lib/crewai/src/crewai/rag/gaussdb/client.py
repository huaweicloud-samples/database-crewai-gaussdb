"""GaussDB implementation of the RAG BaseClient protocol.

One collection = one ``crewai_rag_<name>`` table with a ``floatvector``
embedding column. Behavior anchors are replicated from ChromaDBClient
(the backend it replaces):

- default ``doc_id`` = sha256(content + "|" + json.dumps(metadata,
  sort_keys=True)) (chromadb utils.py:82-85);
- cosine ``score = clamp(1.0 - 0.5 * distance, 0, 1)`` (utils.py:180);
- ``add_documents`` is an upsert (same doc_id overwrites) in batches of
  100; a doc_id repeated inside one call keeps the LAST occurrence;
- an empty content string is a legal document (the content column is
  nullable; chromadb stores empty documents too).

GaussDB-specific facts (verified on GaussDB 507 O-mode):
- no implicit text→jsonb assignment cast, so the MERGE payload declares
  ``cast_columns={"metadata": "jsonb"}`` (see crewai.gaussdb.vector);
- the embedding dimension is persisted in a column COMMENT
  (``dim=<n>``) and read back via ``col_description`` on later calls;
- schema DDL races (create/drop vs create) are serialized with
  ``crewai_core.lock_store`` — see ensure_vector_index's contract.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import AbstractContextManager, asynccontextmanager, nullcontext
import hashlib
import json
import re
from typing import TYPE_CHECKING, Any

from crewai_core.lock_store import lock as store_lock
from typing_extensions import Unpack

from crewai.gaussdb.config import GaussDBConfig
from crewai.gaussdb.connection import cursor
from crewai.gaussdb.vector import (
    _vector_literal,
    ensure_vector_index,
    is_distributed,
    upsert_via_merge,
    validate_dimension,
)
from crewai.rag.core.base_client import (
    BaseClient,
    BaseCollectionAddParams,
    BaseCollectionParams,
    BaseCollectionSearchParams,
)
from crewai.rag.gaussdb.config import GaussDBRagConfig
from crewai.rag.types import BaseRecord, SearchResult


if TYPE_CHECKING:
    # N812: aliasing the lowercase builtin-ish name `cursor` to PascalCase.
    from psycopg2.extensions import cursor as Psycopg2Cursor  # noqa: N812

_TABLE_PREFIX = "crewai_rag_"
# GaussDB (like PostgreSQL) caps identifiers at 63 bytes.
_MAX_IDENTIFIER = 63
_DEFAULT_BATCH_SIZE = 100
_DEFAULT_LIMIT = 10
_DIM_PROBE_TEXT = "__dim__"


def _sanitize_table_name(name: str) -> str:
    """Normalize a collection name into a table-name suffix.

    Only ``[a-z0-9_]`` survive (everything else becomes ``_``), truncated so
    the full ``crewai_rag_<name>`` identifier stays within the 63-byte limit.
    """

    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name).lower()
    return sanitized[: _MAX_IDENTIFIER - len(_TABLE_PREFIX)]


def _distance_to_score(distance: float) -> float:
    """Cosine distance → similarity score (chromadb utils.py:180 anchor)."""

    return max(0.0, min(1.0, 1.0 - 0.5 * distance))


def _json_scalar_text(value: Any) -> str:
    """Render a metadata filter value the way jsonb ``->>`` renders it.

    ``->>`` returns the JSON scalar as text: booleans are ``true``/``false``
    (not Python's ``True``), numbers/strings use their JSON text form.
    """

    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class GaussDBClient(BaseClient):
    """BaseClient implementation backed by GaussDB floatvector tables.

    Attributes:
        client: Always ``None`` — this backend talks to the database through
            the shared pooled cursor (``crewai.gaussdb.connection``), so there
            is no SDK client object to hold. Present to satisfy the
            BaseClient protocol attribute contract.
        embedding_function: Callable turning a list of texts into embedding
            vectors. There is no local default embedder: passing ``None``
            makes add/search raise ``ValueError``.
    """

    def __init__(self, config: GaussDBRagConfig | None = None, **kwargs: Any) -> None:
        """Initialize the client from a RAG config.

        Args:
            config: GaussDB RAG configuration. Connection defaults mirror
                ``crewai.gaussdb.config.GaussDBConfig``.
            kwargs: Alternative escape hatch for ``embedding_function`` when
                no config object is supplied.
        """

        cfg = config or GaussDBRagConfig()
        # Any (not EmbeddingFunction): None is legal until add/search, which
        # raise a ValueError via _require_embedding_function.
        self.embedding_function: Any = cfg.embedding_function or kwargs.get(
            "embedding_function"
        )
        self._config = GaussDBConfig(
            host=cfg.host,
            port=cfg.port,
            user=cfg.user,
            password=cfg.password,
            database=cfg.database,
            min_connections=cfg.min_connections,
            max_connections=cfg.max_connections,
        )
        self.client: Any = None
        self._lock_name = f"gaussdb:{cfg.host}:{cfg.port}:{cfg.database}:rag"

    # ---- helpers ----

    @staticmethod
    def _table_name(collection_name: str) -> str:
        """Map a collection name to its physical table name."""

        return _TABLE_PREFIX + _sanitize_table_name(collection_name)

    def _locked(self) -> AbstractContextManager[None]:
        """Return a cross-process lock context manager, or nullcontext if no lock name."""

        return store_lock(self._lock_name) if self._lock_name else nullcontext()

    @asynccontextmanager
    async def _alocked(self) -> AsyncIterator[None]:
        """Async cross-process lock that acquires/releases in an executor."""

        if not self._lock_name:
            yield
            return
        lock_cm = store_lock(self._lock_name)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lock_cm.__enter__)
        try:
            yield
        finally:
            await loop.run_in_executor(None, lock_cm.__exit__, None, None, None)

    def _require_embedding_function(self) -> Any:
        if self.embedding_function is None:
            raise ValueError(
                "GaussDBClient requires an embedding_function (set "
                "GaussDBRagConfig.embedding_function or pass embedding_function=): "
                "no local default embedder is bundled for the GaussDB backend."
            )
        return self.embedding_function

    @staticmethod
    def _table_exists(cur: Psycopg2Cursor, table: str) -> bool:
        cur.execute("SELECT 1 FROM pg_tables WHERE tablename = %s", (table,))
        return cur.fetchone() is not None

    @staticmethod
    def _read_embedding_dim(cur: Psycopg2Cursor, table: str) -> int | None:
        """Read the stored embedding dimension, or None when unknown.

        Primary source is the ``dim=<n>`` COMMENT written at create time;
        pre-comment tables fall back to sampling one row with vector_dims.
        """

        cur.execute(
            # table is _sanitize_table_name output ([a-z0-9_])
            "SELECT col_description(a.attrelid, a.attnum) FROM pg_attribute a "  # noqa: S608
            f"WHERE a.attrelid = '\"{table}\"'::regclass AND a.attname = 'embedding'"
        )
        row = cur.fetchone()
        comment = str(row[0]) if row and row[0] is not None else ""
        if comment.startswith("dim="):
            try:
                return int(comment[4:])
            except ValueError:
                pass
        cur.execute(f"SELECT vector_dims(embedding) FROM {table} LIMIT 1")  # noqa: S608
        row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def _create_collection_table(self, cur: Psycopg2Cursor, table: str) -> int:
        """Create the table, dimension COMMENT and vector index (lock held)."""

        embedding_function = self._require_embedding_function()
        dim = len(embedding_function([_DIM_PROBE_TEXT])[0])
        validate_dimension(dim, is_distributed(cur))
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {table} ("
            "id VARCHAR(64) PRIMARY KEY, "
            "content TEXT, "  # nullable: an empty-string document is legal
            "metadata JSONB NOT NULL DEFAULT '{}', "
            f"embedding floatvector({dim}) NOT NULL)"
        )
        cur.execute(f"COMMENT ON COLUMN {table}.embedding IS 'dim={dim}'")
        ensure_vector_index(
            cur, f"idx_{table}"[:_MAX_IDENTIFIER], table, "embedding", dim
        )
        return dim

    def _ensure_collection(self, cur: Psycopg2Cursor, table: str) -> int | None:
        """Create the table when missing (lock held); return the stored dim."""

        if not self._table_exists(cur, table):
            return self._create_collection_table(cur, table)
        return self._read_embedding_dim(cur, table)

    @staticmethod
    def _validate_dim(stored_dim: int | None, vector: list[float], table: str) -> None:
        if stored_dim is not None and len(vector) != stored_dim:
            raise ValueError(
                f"Embedding dimension {len(vector)} does not match the stored "
                f"dimension {stored_dim} of table '{table}'. Use an embedding "
                f"function consistent with the one used to create the collection."
            )

    @staticmethod
    def _prepare_documents(
        documents: list[BaseRecord],
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Extract (doc_id, content, metadata) triples, chromadb-parity.

        doc_id resolution order matches chromadb utils.py: an explicit
        ``doc_id``, then a ``doc_id`` metadata key, then the sha256 of
        ``content + "|" + json.dumps(metadata, sort_keys=True)``. A repeated
        doc_id keeps the LAST occurrence (chromadb upsert semantics).
        """

        prepared: list[tuple[str, str, dict[str, Any]]] = []
        seen_ids: dict[str, int] = {}
        for doc in documents:
            metadata = doc.get("metadata")
            if "doc_id" in doc:
                doc_id = str(doc["doc_id"])
            elif metadata and isinstance(metadata, dict) and "doc_id" in metadata:
                doc_id = str(metadata["doc_id"])
            else:
                content_for_hash = doc["content"]
                if metadata:
                    metadata_str = json.dumps(metadata, sort_keys=True)
                    content_for_hash = f"{content_for_hash}|{metadata_str}"
                doc_id = hashlib.sha256(content_for_hash.encode()).hexdigest()

            if isinstance(metadata, list):
                processed_metadata = (
                    dict(metadata[0]) if metadata and metadata[0] else {}
                )
            elif metadata:
                processed_metadata = dict(metadata)
            else:
                processed_metadata = {}

            if doc_id in seen_ids:
                prepared[seen_ids[doc_id]] = (
                    doc_id,
                    doc["content"],
                    processed_metadata,
                )
            else:
                seen_ids[doc_id] = len(prepared)
                prepared.append((doc_id, doc["content"], processed_metadata))
        return prepared

    # ---- collection lifecycle ----

    def create_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Create a new collection (table + index) in GaussDB.

        Keyword Args:
            collection_name: Name of the collection to create. Must be unique.

        Raises:
            ValueError: If a collection with the same name already exists.
            ConnectionError: If unable to connect to GaussDB.
        """

        table = self._table_name(kwargs["collection_name"])
        with self._locked(), cursor(self._config) as cur:
            if self._table_exists(cur, table):
                raise ValueError(
                    f"Collection '{kwargs['collection_name']}' already exists "
                    f"(table {table})"
                )
            self._create_collection_table(cur, table)

    async def acreate_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Create a new collection in GaussDB asynchronously.

        Keyword Args:
            collection_name: Name of the collection to create. Must be unique.

        Raises:
            ValueError: If a collection with the same name already exists.
            ConnectionError: If unable to connect to GaussDB.
        """

        await asyncio.to_thread(self.create_collection, **kwargs)

    def get_or_create_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> Any:
        """Get an existing collection or create it if it doesn't exist.

        Keyword Args:
            collection_name: Name of the collection to get or create.

        Returns:
            The physical table name (``crewai_rag_<sanitized>``). Unlike the
            chromadb backend there is no collection object — the table name
            is the stable handle for subsequent operations.
        """

        table = self._table_name(kwargs["collection_name"])
        with self._locked(), cursor(self._config) as cur:
            if not self._table_exists(cur, table):
                self._create_collection_table(cur, table)
        return table

    async def aget_or_create_collection(
        self, **kwargs: Unpack[BaseCollectionParams]
    ) -> Any:
        """Get an existing collection or create it asynchronously.

        Keyword Args:
            collection_name: Name of the collection to get or create.

        Returns:
            The physical table name (``crewai_rag_<sanitized>``).
        """

        return await asyncio.to_thread(self.get_or_create_collection, **kwargs)

    def add_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to a collection.

        Performs an upsert — documents with an existing doc_id are updated.
        Embeddings are generated with the configured embedding function in
        batches (default 100). The collection is created implicitly when
        missing (chromadb get_or_create parity).

        Keyword Args:
            collection_name: The name of the collection to add documents to.
            documents: List of BaseRecord dicts containing ``content``
                (required), optional ``doc_id`` (auto-hashed when missing)
                and optional ``metadata``.
            batch_size: Batch size for the MERGE statements (default 100).

        Raises:
            ValueError: If documents is empty, no embedding function is
                configured, or an embedding dimension mismatches the table.
            ConnectionError: If unable to connect to GaussDB.
        """

        documents = kwargs["documents"]
        batch_size = kwargs.get("batch_size", _DEFAULT_BATCH_SIZE)

        if not documents:
            raise ValueError("Documents list cannot be empty")
        embedding_function = self._require_embedding_function()

        table = self._table_name(kwargs["collection_name"])
        prepared = self._prepare_documents(documents)

        with self._locked(), cursor(self._config) as cur:
            stored_dim = self._ensure_collection(cur, table)
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start : start + batch_size]
                vectors = embedding_function([content for _, content, _ in batch])
                rows = []
                for (doc_id, content, metadata), vector in zip(
                    batch, vectors, strict=True
                ):
                    self._validate_dim(stored_dim, vector, table)
                    rows.append(
                        {
                            "id": doc_id,
                            "content": content,
                            "metadata": json.dumps(metadata, ensure_ascii=False),
                            "embedding": vector,
                        }
                    )
                upsert_via_merge(
                    cur,
                    table,
                    ["id"],
                    rows,
                    "embedding",
                    cast_columns={"metadata": "jsonb"},
                )

    async def aadd_documents(self, **kwargs: Unpack[BaseCollectionAddParams]) -> None:
        """Add documents with their embeddings to a collection asynchronously.

        Keyword Args:
            collection_name: The name of the collection to add documents to.
            documents: List of BaseRecord dicts (see :meth:`add_documents`).
            batch_size: Batch size for the MERGE statements (default 100).
        """

        await asyncio.to_thread(self.add_documents, **kwargs)

    def search(
        self, **kwargs: Unpack[BaseCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query.

        Keyword Args:
            collection_name: The name of the collection to search in.
            query: The text query to search for.
            limit: Maximum number of results to return (default 10).
            metadata_filter: Optional dict of metadata equality filters
                (multi-key AND, values fully parameterized). Values are
                matched against the jsonb text form (booleans as
                ``true``/``false``).
            score_threshold: Optional minimum similarity score (0-1). Unlike
                the chromadb backend there is no implicit default — results
                are filtered only when a threshold is given.

        Returns:
            List of SearchResult dicts ordered by similarity score descending.

        Raises:
            ValueError: If the collection doesn't exist or no embedding
                function is configured.
            ConnectionError: If unable to connect to GaussDB.
        """

        query = kwargs["query"]
        limit = kwargs.get("limit", _DEFAULT_LIMIT)
        metadata_filter = kwargs.get("metadata_filter")
        score_threshold = kwargs.get("score_threshold")
        embedding_function = self._require_embedding_function()

        table = self._table_name(kwargs["collection_name"])
        with cursor(self._config) as cur:
            if not self._table_exists(cur, table):
                raise ValueError(
                    f"Collection '{kwargs['collection_name']}' does not exist "
                    f"(table {table})"
                )
            query_vector = embedding_function([query])[0]
            self._validate_dim(
                self._read_embedding_dim(cur, table), query_vector, table
            )

            sql = (
                f"SELECT id, content, metadata, embedding <+> %s AS distance "  # noqa: S608
                f"FROM {table}"
            )
            params: list[Any] = [_vector_literal(query_vector)]
            if metadata_filter:
                clauses = []
                for key, value in metadata_filter.items():
                    clauses.append("metadata->>%s = %s")
                    params.extend([str(key), _json_scalar_text(value)])
                sql += " WHERE " + " AND ".join(clauses)
            sql += " ORDER BY distance LIMIT %s"
            params.append(limit)
            cur.execute(sql, tuple(params))  # values are bound, not interpolated
            rows = cur.fetchall()

        results: list[SearchResult] = []
        for row in rows:
            score = _distance_to_score(float(row[3]))
            if score_threshold is not None and score < score_threshold:
                continue
            metadata = row[2]
            if isinstance(metadata, str):  # driver without jsonb auto-parse
                metadata = json.loads(metadata)
            results.append(
                {
                    "id": row[0],
                    "content": row[1] if row[1] is not None else "",
                    "metadata": dict(metadata) if metadata else {},
                    "score": score,
                }
            )
        return results

    async def asearch(
        self, **kwargs: Unpack[BaseCollectionSearchParams]
    ) -> list[SearchResult]:
        """Search for similar documents using a query asynchronously.

        Keyword Args:
            collection_name: The name of the collection to search in.
            query: The text query to search for.
            limit: Maximum number of results to return (default 10).
            metadata_filter: Optional metadata equality filter.
            score_threshold: Optional minimum similarity score (0-1).

        Returns:
            List of SearchResult dicts ordered by similarity score descending.
        """

        return await asyncio.to_thread(self.search, **kwargs)

    def delete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a collection and all its data (idempotent).

        Keyword Args:
            collection_name: Name of the collection to delete.
        """

        table = self._table_name(kwargs["collection_name"])
        with self._locked(), cursor(self._config) as cur:
            cur.execute(f"DROP TABLE IF EXISTS {table}")

    async def adelete_collection(self, **kwargs: Unpack[BaseCollectionParams]) -> None:
        """Delete a collection and all its data asynchronously.

        Keyword Args:
            collection_name: Name of the collection to delete.
        """

        await asyncio.to_thread(self.delete_collection, **kwargs)

    def reset(self) -> None:
        """Drop every ``crewai_rag_*`` table in the database.

        This operation is irreversible and removes all collections managed by
        this backend. Use with extreme caution in production environments.
        """

        with self._locked(), cursor(self._config) as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables WHERE tablename LIKE %s",
                (f"{_TABLE_PREFIX}%",),
            )
            # The LIKE wildcard '_' is re-checked in Python: only tables with
            # the literal prefix are dropped.
            tables = [
                str(row[0])
                for row in cur.fetchall()
                if str(row[0]).startswith(_TABLE_PREFIX)
            ]
            for table in tables:
                cur.execute(f"DROP TABLE IF EXISTS {table}")

    async def areset(self) -> None:
        """Drop every ``crewai_rag_*`` table in the database asynchronously."""

        await asyncio.to_thread(self.reset)
