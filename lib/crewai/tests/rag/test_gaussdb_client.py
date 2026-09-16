"""Tests for the GaussDB RAG client (BaseClient implementation).

Unit tests run against a fake cursor (no DB connection). Integration tests
are gated behind GAUSSDB_TEST=1 and hit a real GaussDB instance.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from typing import Any

import pytest
from crewai.gaussdb.config import GaussDBConfig
from pydantic import TypeAdapter

from crewai.rag.core.base_client import BaseClient
from crewai.rag.gaussdb import client as gaussdb_client_module
from crewai.rag.gaussdb.client import (
    GaussDBClient,
    _distance_to_score,
    _sanitize_table_name,
)
from crewai.rag.gaussdb.config import GaussDBRagConfig
from crewai.rag.types import BaseRecord


def _embedding_8(texts: list[str]) -> list[list[float]]:
    """Deterministic 8-dim embedding: every text maps to an 8-dim vector."""
    return [[0.125] * 8 for _ in texts]


class FakeCursor:
    """Cursor double dispatching fetch results by SQL substring (no DB)."""

    def __init__(
        self,
        fetchone_map: dict[str, tuple | None] | None = None,
        fetchall_map: dict[str, list[tuple]] | None = None,
    ) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self.fetchone_map = fetchone_map or {}
        self.fetchall_map = fetchall_map or {}

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.executed.append((sql, params or ()))

    def _pick(self, mapping: dict[str, Any]) -> Any:
        last = self.executed[-1][0] if self.executed else ""
        for key, value in mapping.items():
            if key in last:
                return value
        return None

    def fetchone(self) -> tuple | None:
        return self._pick(self.fetchone_map)

    def fetchall(self) -> list[tuple]:
        return self._pick(self.fetchall_map) or []


def _make_client(
    fake: FakeCursor,
    monkeypatch: pytest.MonkeyPatch,
    embedding_function: Any = _embedding_8,
    **config_kwargs: Any,
) -> GaussDBClient:
    """Build a GaussDBClient whose cursor/lock are the given fake."""

    @contextlib.contextmanager
    def fake_cursor(config):  # noqa: ANN001
        yield fake

    monkeypatch.setattr(gaussdb_client_module, "cursor", fake_cursor)
    monkeypatch.setattr(
        gaussdb_client_module, "store_lock", lambda name: contextlib.nullcontext()
    )
    config_kwargs.setdefault("embedding_function", embedding_function)
    cfg = GaussDBRagConfig(
        host="db", port=1, database="unit", user="u", **config_kwargs
    )
    return GaussDBClient(config=cfg)


def _merge_payloads(fake: FakeCursor) -> list[list[dict]]:
    """Decode the payload rows of every recorded MERGE statement."""
    payloads = []
    for sql, params in fake.executed:
        if "MERGE INTO" in sql:
            payloads.append(json.loads(params[0]))
    return payloads


# ---- sanitize / score helpers ----


class TestSanitizeTableName:
    def test_special_chars_and_case(self) -> None:
        assert _sanitize_table_name("My Docs/2024!") == "my_docs_2024_"

    def test_truncation_to_identifier_limit(self) -> None:
        long_name = "x" * 100
        sanitized = _sanitize_table_name(long_name)
        # 63-byte identifier limit minus the "crewai_rag_" prefix
        assert len(sanitized) == 63 - len("crewai_rag_")
        full = GaussDBClient._table_name(long_name)
        assert len(full) <= 63
        assert full.startswith("crewai_rag_")

    def test_underscores_kept(self) -> None:
        assert _sanitize_table_name("a_b-c") == "a_b_c"


class TestDistanceToScore:
    def test_anchors(self) -> None:
        assert _distance_to_score(0.0) == 1.0
        assert _distance_to_score(1.0) == 0.5
        assert _distance_to_score(2.0) == 0.0

    def test_clamp_bounds(self) -> None:
        assert _distance_to_score(-1.0) == 1.0
        assert _distance_to_score(3.0) == 0.0


# ---- doc_id hashing (chromadb utils.py parity) ----


class TestDocIdHash:
    def test_explicit_doc_id_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)})
        client = _make_client(fake, monkeypatch)
        docs: list[BaseRecord] = [{"doc_id": "given", "content": "c"}]
        client.add_documents(collection_name="docs", documents=docs)
        rows = _merge_payloads(fake)[0]
        assert rows[0]["id"] == "given"

    def test_default_hash_matches_chromadb_formula(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)})
        client = _make_client(fake, monkeypatch)
        metadata = {"b": 2, "a": "x"}
        docs: list[BaseRecord] = [{"content": "hello", "metadata": metadata}]
        client.add_documents(collection_name="docs", documents=docs)
        rows = _merge_payloads(fake)[0]
        expected = hashlib.sha256(
            f"hello|{json.dumps(metadata, sort_keys=True)}".encode()
        ).hexdigest()
        assert rows[0]["id"] == expected

    def test_default_hash_without_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)})
        client = _make_client(fake, monkeypatch)
        client.add_documents(collection_name="docs", documents=[{"content": "solo"}])
        rows = _merge_payloads(fake)[0]
        assert rows[0]["id"] == hashlib.sha256(b"solo").hexdigest()

    def test_metadata_doc_id_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)})
        client = _make_client(fake, monkeypatch)
        docs: list[BaseRecord] = [
            {"content": "c", "metadata": {"doc_id": "from-metadata"}}
        ]
        client.add_documents(collection_name="docs", documents=docs)
        rows = _merge_payloads(fake)[0]
        assert rows[0]["id"] == "from-metadata"

    def test_duplicate_doc_id_last_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)})
        client = _make_client(fake, monkeypatch)
        docs: list[BaseRecord] = [
            {"doc_id": "d", "content": "first"},
            {"doc_id": "d", "content": "second"},
        ]
        client.add_documents(collection_name="docs", documents=docs)
        payloads = _merge_payloads(fake)
        flat = [row for payload in payloads for row in payload]
        assert len(flat) == 1
        assert flat[0]["content"] == "second"


# ---- add_documents ----


class TestAddDocuments:
    def test_no_embedding_function_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor()
        client = _make_client(fake, monkeypatch, embedding_function=None)
        with pytest.raises(ValueError, match="embedding_function"):
            client.add_documents(
                collection_name="docs", documents=[{"content": "c"}]
            )
        assert fake.executed == []

    def test_empty_documents_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor()
        client = _make_client(fake, monkeypatch)
        with pytest.raises(ValueError, match="empty"):
            client.add_documents(collection_name="docs", documents=[])

    def test_batch_splitting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)})
        client = _make_client(fake, monkeypatch)
        docs: list[BaseRecord] = [{"content": f"c{i}"} for i in range(3)]
        client.add_documents(collection_name="docs", documents=docs, batch_size=2)
        payloads = _merge_payloads(fake)
        assert [len(p) for p in payloads] == [2, 1]

    def test_payload_shape(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)})
        client = _make_client(fake, monkeypatch)
        docs: list[BaseRecord] = [
            {"content": "hello", "metadata": {"k": "v", "n": 1}}
        ]
        client.add_documents(collection_name="docs", documents=docs)
        merge_sql, merge_params = next(
            (sql, params) for sql, params in fake.executed if "MERGE INTO" in sql
        )
        assert "MERGE INTO crewai_rag_docs" in merge_sql
        assert "(e->>'metadata')::jsonb" in merge_sql
        row = json.loads(merge_params[0])[0]
        assert row["id"] == hashlib.sha256(
            f"hello|{json.dumps({'k': 'v', 'n': 1}, sort_keys=True)}".encode()
        ).hexdigest()
        assert row["content"] == "hello"
        assert json.loads(row["metadata"]) == {"k": "v", "n": 1}
        assert row["embedding"] == "[" + ",".join(["0.125"] * 8) + "]"

    def test_create_table_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": None, "pgxc_node": (0,)})
        client = _make_client(fake, monkeypatch)
        client.add_documents(collection_name="docs", documents=[{"content": "c"}])
        sqls = [sql for sql, _ in fake.executed]
        create = next(sql for sql in sqls if "CREATE TABLE" in sql)
        assert "floatvector(8)" in create
        assert "metadata JSONB NOT NULL DEFAULT '{}'" in create
        assert any("COMMENT ON COLUMN crewai_rag_docs.embedding" in sql for sql in sqls)
        assert any("CREATE INDEX IF NOT EXISTS" in sql for sql in sqls)

    def test_dim_mismatch_on_existing_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,), "col_description": ("dim=8",)})
        client = _make_client(fake, monkeypatch, embedding_function=None)
        # 16-dim embedding function wired directly on the client
        client.embedding_function = lambda texts: [[0.5] * 16 for _ in texts]
        with pytest.raises(ValueError, match="dimension"):
            client.add_documents(collection_name="docs", documents=[{"content": "c"}])
        assert not any("MERGE" in sql for sql, _ in fake.executed)


# ---- search ----


def _search_row(
    rid: str,
    content: str | None,
    metadata: Any,
    distance: float,
) -> tuple:
    return (rid, content, metadata, distance)


class TestSearch:
    def test_sql_shape_and_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={"embedding <+>": [_search_row("r1", "c", {}, 0.0)]},
        )
        client = _make_client(fake, monkeypatch)
        results = client.search(collection_name="docs", query="q", limit=7)
        sql, params = next(
            (sql, p) for sql, p in fake.executed if "embedding <+>" in sql
        )
        assert "FROM crewai_rag_docs" in sql
        assert "ORDER BY distance LIMIT %s" in sql
        assert "WHERE" not in sql
        assert params[-1] == 7
        assert params[0].startswith("[")
        assert len(results) == 1
        assert results[0] == {
            "id": "r1",
            "content": "c",
            "metadata": {},
            "score": 1.0,
        }

    def test_metadata_filter_parameterized(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={"embedding <+>": [_search_row("r1", "c", {}, 0.0)]},
        )
        client = _make_client(fake, monkeypatch)
        client.search(
            collection_name="docs",
            query="q",
            metadata_filter={"env": "prod", "rank": 3},
        )
        sql, params = next(
            (sql, p) for sql, p in fake.executed if "embedding <+>" in sql
        )
        assert sql.count("metadata->>%s = %s") == 2
        assert " AND " in sql
        # every filter key/value travels as a bound parameter
        assert params[1:5] == ("env", "prod", "rank", "3")

    def test_metadata_filter_bool_normalization(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={"embedding <+>": []},
        )
        client = _make_client(fake, monkeypatch)
        client.search(collection_name="docs", query="q", metadata_filter={"f": True})
        _, params = next(
            (sql, p) for sql, p in fake.executed if "embedding <+>" in sql
        )
        assert params[1:3] == ("f", "true")

    def test_score_threshold_filters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={
                "embedding <+>": [
                    _search_row("a", "ca", {}, 0.0),  # score 1.0
                    _search_row("b", "cb", {}, 1.0),  # score 0.5
                    _search_row("c", "cc", {}, 2.0),  # score 0.0
                ]
            },
        )
        client = _make_client(fake, monkeypatch)
        results = client.search(
            collection_name="docs", query="q", score_threshold=0.5, limit=10
        )
        assert [r["id"] for r in results] == ["a", "b"]
        assert [r["score"] for r in results] == [1.0, 0.5]

    def test_null_content_and_string_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={
                "embedding <+>": [
                    _search_row("r1", None, '{"k": "v"}', 0.4),
                    _search_row("r2", "text", {"n": 2}, 0.8),
                ]
            },
        )
        client = _make_client(fake, monkeypatch)
        results = client.search(collection_name="docs", query="q")
        assert results[0]["content"] == ""  # NULL content -> ""
        assert results[0]["metadata"] == {"k": "v"}  # json text parsed
        assert results[1]["metadata"] == {"n": 2}  # dict passthrough

    def test_missing_collection_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": None})
        client = _make_client(fake, monkeypatch)
        with pytest.raises(ValueError, match="does not exist"):
            client.search(collection_name="ghost", query="q")

    def test_requires_embedding_function(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)})
        client = _make_client(fake, monkeypatch, embedding_function=None)
        with pytest.raises(ValueError, match="embedding_function"):
            client.search(collection_name="docs", query="q")


# ---- collection lifecycle ----


class TestCollectionLifecycle:
    def test_create_collection_new(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": None, "pgxc_node": (0,)})
        client = _make_client(fake, monkeypatch)
        client.create_collection(collection_name="docs")
        sqls = [sql for sql, _ in fake.executed]
        assert any("CREATE TABLE IF NOT EXISTS crewai_rag_docs" in sql for sql in sqls)
        assert any("COMMENT ON COLUMN" in sql for sql in sqls)
        assert any("CREATE INDEX" in sql for sql in sqls)

    def test_create_collection_existing_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)})
        client = _make_client(fake, monkeypatch)
        with pytest.raises(ValueError, match="already exists"):
            client.create_collection(collection_name="docs")

    def test_get_or_create_returns_table_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_missing = FakeCursor(
            fetchone_map={"pg_tables": None, "pgxc_node": (0,)}
        )
        client = _make_client(fake_missing, monkeypatch)
        assert client.get_or_create_collection(collection_name="docs") == (
            "crewai_rag_docs"
        )
        assert any(
            "CREATE TABLE IF NOT EXISTS crewai_rag_docs" in sql
            for sql, _ in fake_missing.executed
        )

        fake_existing = FakeCursor(fetchone_map={"pg_tables": (1,)})
        client = _make_client(fake_existing, monkeypatch)
        assert client.get_or_create_collection(collection_name="docs") == (
            "crewai_rag_docs"
        )
        assert not any(
            "CREATE TABLE" in sql for sql, _ in fake_existing.executed
        )

    def test_delete_collection_drop_if_exists(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor()
        client = _make_client(fake, monkeypatch)
        client.delete_collection(collection_name="docs")
        sql, _ = fake.executed[-1]
        assert sql == "DROP TABLE IF EXISTS crewai_rag_docs"

    def test_reset_drops_only_rag_tables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(
            fetchall_map={
                "pg_tables": [
                    ("crewai_rag_a",),
                    ("memories",),
                    ("crewai_rag_b",),
                ]
            }
        )
        client = _make_client(fake, monkeypatch)
        client.reset()
        drops = [sql for sql, _ in fake.executed if "DROP TABLE" in sql]
        assert drops == [
            "DROP TABLE IF EXISTS crewai_rag_a",
            "DROP TABLE IF EXISTS crewai_rag_b",
        ]

    def test_protocol_conformance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor()
        client = _make_client(fake, monkeypatch)
        assert isinstance(client, BaseClient)


# ---- async wrappers ----


@pytest.mark.block_network(allowed_hosts=[r"127\.0\.0\.1", r"localhost", r"::1"])
def test_async_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeCursor(
        fetchone_map={"pg_tables": (1,)},
        fetchall_map={"embedding <+>": [_search_row("r1", "c", {}, 0.0)]},
    )
    client = _make_client(fake, monkeypatch)

    assert asyncio.run(
        client.aget_or_create_collection(collection_name="docs")
    ) == "crewai_rag_docs"
    asyncio.run(
        client.aadd_documents(collection_name="docs", documents=[{"content": "c"}])
    )
    results = asyncio.run(client.asearch(collection_name="docs", query="q"))
    assert [r["id"] for r in results] == ["r1"]
    asyncio.run(client.adelete_collection(collection_name="docs"))
    assert any(
        "DROP TABLE IF EXISTS crewai_rag_docs" in sql for sql, _ in fake.executed
    )


# ---- config ----


class TestGaussDBRagConfig:
    def test_provider_literal(self) -> None:
        assert GaussDBRagConfig().provider == "gaussdb"

    def test_password_excluded_from_serialization_and_repr(self) -> None:
        cfg = GaussDBRagConfig(user="u", password="SECRET-PW")
        assert "SECRET-PW" not in repr(cfg)
        dumped = TypeAdapter(GaussDBRagConfig).dump_python(cfg)
        assert "password" not in dumped
        assert "SECRET-PW" not in json.dumps(dumped, default=str)
        assert b"SECRET-PW" not in TypeAdapter(GaussDBRagConfig).dump_json(cfg)

    def test_defaults(self) -> None:
        cfg = GaussDBRagConfig()
        assert cfg.host == "localhost"
        assert cfg.port == 5432
        assert cfg.database == "crewai"
        assert cfg.min_connections == 1
        assert cfg.max_connections == 10
        assert cfg.embedding_function is None


# ---- rag factory branch ----


def test_rag_factory_gaussdb_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import Mock, patch

    from crewai.rag.factory import create_client

    mock_config = Mock()
    mock_config.provider = "gaussdb"

    with patch("crewai.rag.factory.require") as mock_require:
        mock_module = Mock()
        mock_client = Mock()
        mock_module.create_client.return_value = mock_client
        mock_require.return_value = mock_module

        result = create_client(mock_config)

        assert result == mock_client
        mock_require.assert_called_once_with(
            "crewai.rag.gaussdb.factory", purpose="The 'gaussdb' provider"
        )
        mock_module.create_client.assert_called_once_with(mock_config)


# ---- integration (real GaussDB) ----

requires_gaussdb = pytest.mark.skipif(
    os.environ.get("GAUSSDB_TEST", "").lower() != "1",
    reason="requires GAUSSDB_TEST=1 and a reachable GaussDB instance",
)

_BASIS = {
    "alpha": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "beta": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "gamma": [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "delta": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
}


def _orthogonal_embedding(texts: list[str]) -> list[list[float]]:
    """Deterministic orthogonal unit vectors keyed on content tokens.

    The dim probe text ("__dim__") and unknown texts fall back to a uniform
    vector, keeping the embedding dimension at 8 for every input.
    """
    out: list[list[float]] = []
    for text in texts:
        for token, vec in _BASIS.items():
            if token in text:
                out.append(vec)
                break
        else:
            out.append([0.1] * 8)
    return out


def _env_rag_config() -> GaussDBRagConfig:
    return GaussDBRagConfig(
        host=os.environ.get("GAUSSDB_HOST", "localhost"),
        port=int(os.environ.get("GAUSSDB_PORT", "5432")),
        user=os.environ.get("GAUSSDB_USER", ""),
        password=os.environ.get("GAUSSDB_PASSWORD", ""),
        database=os.environ.get("GAUSSDB_DATABASE", "crewai"),
        embedding_function=_orthogonal_embedding,
    )


def _env_db_config() -> GaussDBConfig:
    return GaussDBConfig(
        host=os.environ.get("GAUSSDB_HOST", "localhost"),
        port=int(os.environ.get("GAUSSDB_PORT", "5432")),
        user=os.environ.get("GAUSSDB_USER", ""),
        password=os.environ.get("GAUSSDB_PASSWORD", ""),
        database=os.environ.get("GAUSSDB_DATABASE", "crewai"),
    )


@requires_gaussdb
@pytest.mark.block_network(allowed_hosts=[r"127\.0\.0\.1", r"localhost", r"::1"])
class TestGaussDBClientIntegration:
    def test_full_roundtrip(self) -> None:
        from crewai.gaussdb.connection import cursor as db_cursor, reset_pool

        cfg = _env_rag_config()
        client = GaussDBClient(config=cfg)
        client.delete_collection(collection_name="t3_roundtrip")
        try:
            # create / get_or_create
            client.create_collection(collection_name="t3_roundtrip")
            with pytest.raises(ValueError, match="already exists"):
                client.create_collection(collection_name="t3_roundtrip")
            assert (
                client.get_or_create_collection(collection_name="t3_roundtrip")
                == "crewai_rag_t3_roundtrip"
            )

            # add documents (5 rows; doc_id re-add overwrites)
            docs: list[BaseRecord] = [
                {"content": "alpha document", "metadata": {"env": "prod", "rank": 1}},
                {"content": "beta document", "metadata": {"env": "dev"}},
                {
                    "content": "gamma document",
                    "metadata": {"env": "prod", "flag": True},
                },
                {"doc_id": "custom-1", "content": "delta document"},
                {"content": ""},  # empty content is a legal document
            ]
            client.add_documents(collection_name="t3_roundtrip", documents=docs)
            client.add_documents(
                collection_name="t3_roundtrip",
                documents=[{"doc_id": "custom-1", "content": "delta updated"}],
            )

            with db_cursor(_env_db_config()) as cur:
                cur.execute("SELECT count(*) FROM crewai_rag_t3_roundtrip")
                assert cur.fetchone()[0] == 5  # 5 docs, re-add overwritten
                cur.execute(
                    "SELECT content FROM crewai_rag_t3_roundtrip "
                    "WHERE id = 'custom-1'"
                )
                assert cur.fetchone()[0] == "delta updated"

            # search: exact top1 (orthogonal basis -> distance 0 -> score 1.0)
            results = client.search(
                collection_name="t3_roundtrip", query="alpha question", limit=10
            )
            assert results[0]["content"] == "alpha document"
            assert results[0]["score"] == pytest.approx(1.0)

            # limit
            assert len(client.search(
                collection_name="t3_roundtrip", query="alpha question", limit=2
            )) == 2

            # score_threshold keeps only the exact match
            exact = client.search(
                collection_name="t3_roundtrip",
                query="alpha question",
                score_threshold=0.99,
                limit=10,
            )
            assert [r["content"] for r in exact] == ["alpha document"]

            # metadata_filter (equality pushdown)
            prod = client.search(
                collection_name="t3_roundtrip",
                query="alpha question",
                metadata_filter={"env": "prod"},
                limit=10,
            )
            assert {r["content"] for r in prod} == {"alpha document", "gamma document"}

            # bool metadata filter normalization end-to-end
            flagged = client.search(
                collection_name="t3_roundtrip",
                query="alpha question",
                metadata_filter={"flag": True},
                limit=10,
            )
            assert [r["content"] for r in flagged] == ["gamma document"]

            # empty content comes back as "" (nullable column, O-mode driver)
            empty = [
                r
                for r in client.search(
                    collection_name="t3_roundtrip", query="beta question", limit=10
                )
                if r["content"] == ""
            ]
            assert len(empty) == 1

            # dim persisted in the column comment
            with db_cursor(_env_db_config()) as cur:
                cur.execute(
                    "SELECT col_description(a.attrelid, a.attnum) FROM pg_attribute a "
                    "WHERE a.attrelid = '\"crewai_rag_t3_roundtrip\"'::regclass "
                    "AND a.attname = 'embedding'"
                )
                assert cur.fetchone()[0] == "dim=8"

            # async roundtrip on the same table
            async_docs = [{"doc_id": "async-1", "content": "delta async"}]
            asyncio.run(
                client.aadd_documents(
                    collection_name="t3_roundtrip", documents=async_docs
                )
            )
            found = asyncio.run(
                client.asearch(collection_name="t3_roundtrip", query="delta async")
            )
            # "delta updated" shares the "delta" basis vector, so the exact
            # match ties with it — assert membership + exact score instead.
            assert any(
                r["id"] == "async-1" and r["score"] == pytest.approx(1.0)
                for r in found
            )

            # missing collection search raises
            with pytest.raises(ValueError, match="does not exist"):
                client.search(collection_name="t3_never_created", query="q")

            client.delete_collection(collection_name="t3_roundtrip")
            with db_cursor(_env_db_config()) as cur:
                cur.execute(
                    "SELECT 1 FROM pg_tables WHERE tablename = 'crewai_rag_t3_roundtrip'"
                )
                assert cur.fetchone() is None
        finally:
            client.delete_collection(collection_name="t3_roundtrip")
            reset_pool()

    def test_reset_cleans_rag_tables(self) -> None:
        from crewai.gaussdb.connection import cursor as db_cursor, reset_pool

        client = GaussDBClient(config=_env_rag_config())
        try:
            client.get_or_create_collection(collection_name="t3_reset_a")
            client.get_or_create_collection(collection_name="t3_reset_b")
            client.reset()
            with db_cursor(GaussDBConfig.from_env()) as cur:
                cur.execute(
                    "SELECT tablename FROM pg_tables WHERE tablename LIKE %s",
                    ("crewai_rag_t3_reset%",),
                )
                assert cur.fetchall() == []
        finally:
            client.reset()
            reset_pool()
