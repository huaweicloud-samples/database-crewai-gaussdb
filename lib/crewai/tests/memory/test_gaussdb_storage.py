"""Tests for the GaussDB memory storage backend.

Unit tests run against a fake cursor (no DB connection). Integration tests
are gated behind GAUSSDB_TEST=1 and hit a real GaussDB instance.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from datetime import datetime
from typing import Any

import pytest

from crewai.gaussdb.config import GaussDBConfig
from crewai.memory.storage import gaussdb_storage as gs
from crewai.memory.storage.backend import EmbeddingDimensionMismatchError
from crewai.memory.types import MemoryRecord, ScopeInfo

# Deterministic, mutually orthogonal unit vectors (cosine distance 0 to self,
# 1.0 to each other) so integration search results are exact.
V1 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
V2 = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
V3 = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class FakeCursor:
    """Cursor double dispatching fetch results by SQL substring (no DB)."""

    def __init__(
        self,
        fetchone_map: dict[str, tuple | None] | None = None,
        fetchall_map: dict[str, list[tuple]] | None = None,
        rowcount: int = 0,
    ) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self.fetchone_map = fetchone_map or {}
        self.fetchall_map = fetchall_map or {}
        self.rowcount = rowcount

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


def _make_storage(
    fake: FakeCursor, monkeypatch: pytest.MonkeyPatch
) -> gs.GaussDBStorage:
    """Build a GaussDBStorage whose cursor/lock are the given fake."""

    @contextlib.contextmanager
    def fake_cursor(config):  # noqa: ANN001
        yield fake

    monkeypatch.setattr(gs, "cursor", fake_cursor)
    monkeypatch.setattr(gs, "store_lock", lambda name: contextlib.nullcontext())
    return gs.GaussDBStorage(
        config=GaussDBConfig(host="db", port=1, database="unit", user="u")
    )


def _row(
    rid: str = "r1",
    content: str = "c",
    scope: str = "/test",
    categories: str = '["c1"]',
    metadata: str = '{"k": "v"}',
    importance: float = 0.5,
    created: str = "2026-01-01T00:00:00",
    accessed: str = "2026-01-01T00:00:00",
    source: str | None = None,
    private: int = 0,
    distance: float | None = None,
) -> tuple:
    row = (
        rid,
        content,
        scope,
        categories,
        metadata,
        importance,
        created,
        accessed,
        source,
        private,
    )
    if distance is not None:
        return row + (distance,)
    return row


def _merge_row(fake: FakeCursor) -> dict:
    """Decode the single payload row of the recorded MERGE statement."""
    merge = next(
        (sql, params) for sql, params in fake.executed if "MERGE INTO memories" in sql
    )
    return json.loads(merge[1][0])[0]


class TestSave:
    def test_first_save_creates_table_and_merges(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor()  # pg_tables probe -> None: table missing
        storage = _make_storage(fake, monkeypatch)
        rec = MemoryRecord(
            content="hello",
            scope="/test",
            categories=["x"],
            metadata={"env": "prod"},
            embedding=V1,
            created_at=datetime(2026, 1, 1),
            last_accessed=datetime(2026, 1, 2),
            private=True,
        )
        storage.save([rec])

        sqls = [sql for sql, _ in fake.executed]
        create_table = next(sql for sql in sqls if "CREATE TABLE" in sql)
        assert "floatvector(8)" in create_table
        assert "CREATE INDEX IF NOT EXISTS idx_memories_scope" in " ".join(sqls)
        assert "idx_memories_embedding" in " ".join(sqls)

        row = _merge_row(fake)
        assert row["id"] == rec.id
        assert row["content"] == "hello"
        assert row["scope"] == "/test"
        assert row["categories"] == '["x"]'
        assert row["metadata"] == '{"env": "prod"}'
        assert row["importance"] == 0.5
        assert row["created_at"] == "2026-01-01T00:00:00"
        assert row["last_accessed"] == "2026-01-02T00:00:00"
        assert row["private"] == 1
        assert row["embedding"] == "[" + ",".join(repr(v) for v in V1) + "]"

    def test_empty_content_becomes_single_space(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor()
        storage = _make_storage(fake, monkeypatch)
        storage.save([MemoryRecord(content="", embedding=V1)])
        assert _merge_row(fake)["content"] == " "

    def test_dim_mismatch_against_existing_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,), "vector_dims": (8,)})
        storage = _make_storage(fake, monkeypatch)
        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            storage.save([MemoryRecord(content="x", embedding=[0.5] * 16)])
        assert exc_info.value.stored_dim == 8
        assert exc_info.value.new_dim == 16
        assert not any("MERGE" in sql for sql, _ in fake.executed)

    def test_batch_internal_dim_conflict_raises_before_sql(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor()
        storage = _make_storage(fake, monkeypatch)
        with pytest.raises(EmbeddingDimensionMismatchError):
            storage.save(
                [
                    MemoryRecord(content="a", embedding=V1),
                    MemoryRecord(content="b", embedding=[0.5] * 16),
                ]
            )
        assert fake.executed == []

    def test_record_without_embedding_is_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor()
        storage = _make_storage(fake, monkeypatch)
        storage.save([MemoryRecord(content="no embedding")])
        assert fake.executed == []

    def test_mixed_batch_saves_only_embedded_records(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor()
        storage = _make_storage(fake, monkeypatch)
        with_emb = MemoryRecord(content="a", embedding=V1)
        without = MemoryRecord(content="b")
        storage.save([with_emb, without])
        assert len(json.loads(
            next(p for sql, p in fake.executed if "MERGE" in sql)[0]
        )) == 1

    def test_empty_records_is_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor()
        storage = _make_storage(fake, monkeypatch)
        storage.save([])
        assert fake.executed == []

    def test_update_merges_single_record(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor()
        storage = _make_storage(fake, monkeypatch)
        rec = MemoryRecord(content="updated", embedding=V1)
        storage.update(rec)
        row = _merge_row(fake)
        assert row["id"] == rec.id
        assert row["content"] == "updated"


class TestTouchRecords:
    def test_sql_shape_and_rowcount(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)}, rowcount=2)
        storage = _make_storage(fake, monkeypatch)
        assert storage.touch_records(["a", "b"], accessed_at=datetime(2026, 3, 1)) == 2
        sql, params = fake.executed[-1]
        assert sql == "UPDATE memories SET last_accessed = %s WHERE id = ANY(%s)"
        assert params[0] == "2026-03-01T00:00:00"
        assert params[1] == ["a", "b"]

    def test_default_timestamp_is_naive_now(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)}, rowcount=1)
        storage = _make_storage(fake, monkeypatch)
        storage.touch_records(["a"])
        parsed = datetime.fromisoformat(fake.executed[-1][1][0])
        assert parsed.tzinfo is None  # naive, keeps ISO lexicographic order
        assert parsed > datetime(2026, 1, 1)

    def test_empty_ids_returns_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)}, rowcount=2)
        storage = _make_storage(fake, monkeypatch)
        assert storage.touch_records([]) == 0
        assert fake.executed == []

    def test_table_missing_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor()
        storage = _make_storage(fake, monkeypatch)
        assert storage.touch_records(["a"]) == 0
        assert not any("UPDATE" in sql for sql, _ in fake.executed)


class TestSearch:
    def test_sql_shape_and_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={"FROM memories": [_row(distance=0.0)]},
        )
        storage = _make_storage(fake, monkeypatch)
        results = storage.search(
            V1,
            scope_prefix="/team",
            categories=["c1"],
            metadata_filter={"k": "v"},
            limit=5,
            min_score=0.0,
        )

        sql, params = next(
            (sql, p) for sql, p in fake.executed if "FROM memories" in sql
        )
        assert "embedding <+> %s" in sql
        assert "ORDER BY distance" in sql
        assert "WHERE scope LIKE %s" in sql
        assert params[0].startswith("[")  # vector literal
        assert params[1] == "/team%"
        assert params[-1] == 15  # limit * 3 prefetch when Python-side filters exist

        assert len(results) == 1
        rec, score = results[0]
        assert rec.id == "r1"
        assert score == 1.0
        assert rec.embedding is None  # embedding is not backfilled
        assert rec.categories == ["c1"]
        assert rec.metadata == {"k": "v"}
        assert rec.created_at == datetime(2026, 1, 1)
        assert rec.source is None
        assert rec.private is False

    def test_no_filters_keeps_limit_unmultiplied(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={"FROM memories": [_row(distance=0.0)]},
        )
        storage = _make_storage(fake, monkeypatch)
        storage.search(V1, limit=7)
        sql, params = next(
            (sql, p) for sql, p in fake.executed if "FROM memories" in sql
        )
        assert "WHERE" not in sql
        assert params[-1] == 7

    def test_score_is_clamped_cosine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={
                "FROM memories": [
                    _row(rid="a", categories="[]", metadata="{}", distance=0.0),
                    _row(rid="b", categories="[]", metadata="{}", distance=0.5),
                    _row(rid="c", categories="[]", metadata="{}", distance=1.7),
                ]
            },
        )
        storage = _make_storage(fake, monkeypatch)
        results = storage.search(V1, limit=10)
        assert [s for _, s in results] == [1.0, 0.5, 0.0]

    def test_python_side_filters_and_min_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={
                "FROM memories": [
                    _row(rid="ok", distance=0.2),  # score 0.8, matches filters
                    _row(rid="cat_miss", categories='["c2"]', distance=0.0),
                    _row(rid="meta_miss", metadata='{"k": "other"}', distance=0.0),
                    _row(rid="low_score", distance=1.0),  # score 0.0 < min_score
                ]
            },
        )
        storage = _make_storage(fake, monkeypatch)
        results = storage.search(
            V1, categories=["c1"], metadata_filter={"k": "v"}, min_score=0.5, limit=10
        )
        assert [r.id for r, _ in results] == ["ok"]

    def test_dim_mismatch_when_dim_known(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor()
        storage = _make_storage(fake, monkeypatch)
        storage._dim = 8
        with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
            storage.search([0.1] * 16)
        assert exc_info.value.stored_dim == 8
        assert exc_info.value.new_dim == 16
        assert fake.executed == []

    def test_table_missing_returns_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor()
        storage = _make_storage(fake, monkeypatch)
        assert storage.search(V1) == []


class TestDelete:
    def test_dynamic_where_and_rowcount(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)}, rowcount=3)
        storage = _make_storage(fake, monkeypatch)
        n = storage.delete(scope_prefix="/team", older_than=datetime(2026, 1, 1))
        assert n == 3
        sql, params = fake.executed[-1]
        assert sql.startswith("DELETE FROM memories WHERE")
        assert "scope LIKE %s" in sql
        assert "created_at < %s" in sql
        assert " AND " in sql
        assert params == ("/team%", "2026-01-01T00:00:00")

    def test_unconditional_delete_all(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)}, rowcount=9)
        storage = _make_storage(fake, monkeypatch)
        assert storage.delete() == 9
        sql, params = fake.executed[-1]
        assert sql == "DELETE FROM memories"
        assert params == ()

    def test_delete_by_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)}, rowcount=2)
        storage = _make_storage(fake, monkeypatch)
        assert storage.delete(record_ids=["a", "b"]) == 2
        sql, params = fake.executed[-1]
        assert "id IN %s" in sql
        assert params == (("a", "b"),)

    def test_empty_record_ids_delete_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)}, rowcount=9)
        storage = _make_storage(fake, monkeypatch)
        assert storage.delete(record_ids=[]) == 0
        assert not any("DELETE" in sql for sql, _ in fake.executed)

    def test_categories_metadata_filtered_python_side(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={
                "FROM memories": [
                    _row(rid="hit", scope="/team", distance=None),
                    _row(rid="cat_miss", scope="/team", categories='["c2"]'),
                    _row(rid="meta_miss", scope="/team", metadata='{"k": "other"}'),
                ]
            },
            rowcount=1,
        )
        storage = _make_storage(fake, monkeypatch)
        n = storage.delete(
            scope_prefix="/team", categories=["c1"], metadata_filter={"k": "v"}
        )
        assert n == 1
        scan_sql, scan_params = fake.executed[-2]
        assert scan_sql.startswith("SELECT")
        assert "scope LIKE %s" in scan_sql
        assert scan_params == ("/team%",)
        delete_sql, delete_params = fake.executed[-1]
        assert delete_sql.startswith("DELETE FROM memories WHERE id IN %s")
        assert delete_params == (("hit",),)

    def test_scan_with_no_match_deletes_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={"FROM memories": [_row(rid="x", categories='["c2"]')]},
        )
        storage = _make_storage(fake, monkeypatch)
        assert storage.delete(categories=["c1"]) == 0
        assert not any("DELETE" in sql for sql, _ in fake.executed)

    def test_table_missing_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor()
        storage = _make_storage(fake, monkeypatch)
        assert storage.delete() == 0
        assert not any("DELETE" in sql for sql, _ in fake.executed)


class TestGetRecord:
    def test_found_maps_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,), "WHERE id = %s": _row(private=1)}
        )
        storage = _make_storage(fake, monkeypatch)
        got = storage.get_record("r1")
        assert got is not None
        assert got.id == "r1"
        assert got.content == "c"
        assert got.scope == "/test"
        assert got.categories == ["c1"]
        assert got.metadata == {"k": "v"}
        assert got.importance == 0.5
        assert got.created_at == datetime(2026, 1, 1)
        assert got.embedding is None
        assert got.source is None
        assert got.private is True
        sql, params = fake.executed[-1]
        select_part = sql.split("FROM")[0]
        assert select_part.startswith("SELECT id, content, scope,")
        assert "embedding" not in select_part
        assert params == ("r1",)

    def test_missing_row_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)})
        storage = _make_storage(fake, monkeypatch)
        assert storage.get_record("nope") is None

    def test_table_missing_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor()
        storage = _make_storage(fake, monkeypatch)
        assert storage.get_record("r1") is None


class TestListRecords:
    def test_sql_shape_and_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={
                "FROM memories": [
                    _row(rid="r_new", created="2026-01-02T00:00:00"),
                    _row(rid="r_old", created="2026-01-01T00:00:00"),
                ]
            },
        )
        storage = _make_storage(fake, monkeypatch)
        listed = storage.list_records(scope_prefix="/test", limit=1, offset=2)
        sql, params = fake.executed[-1]
        assert "ORDER BY created_at DESC, id" in sql
        assert "OFFSET %s" in sql
        assert "LIMIT %s" in sql
        assert params == ("/test%", 2, 1)
        # The fake cursor ignores LIMIT/OFFSET; mapping preserves the row order.
        assert [r.id for r in listed] == ["r_new", "r_old"]
        assert listed[0].created_at == datetime(2026, 1, 2)

    def test_default_no_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)}, fetchall_map={})
        storage = _make_storage(fake, monkeypatch)
        assert storage.list_records() == []
        sql, params = fake.executed[-1]
        assert "WHERE" not in sql
        assert params == (0, 200)


class TestGetScopeInfo:
    def test_aggregates_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={
                "FROM memories": [
                    ("/test/a", '["x"]', "2026-01-01T00:00:00"),
                    ("/test/b", '["x", "y"]', "2026-01-03T00:00:00"),
                    ("/test/a", '["x"]', "2026-01-02T00:00:00"),
                ]
            },
        )
        storage = _make_storage(fake, monkeypatch)
        info = storage.get_scope_info("/test")
        sql, params = fake.executed[-1]
        assert "WHERE scope LIKE %s" in sql
        assert params == ("/test%",)
        assert isinstance(info, ScopeInfo)
        assert info.path == "/test"
        assert info.record_count == 3
        assert info.categories == ["x", "y"]
        assert info.oldest_record == datetime(2026, 1, 1)
        assert info.newest_record == datetime(2026, 1, 3)
        assert info.child_scopes == ["/test/a", "/test/b"]

    def test_root_scope_scans_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={
                "FROM memories": [
                    ("/", "[]", "2026-01-01T00:00:00"),
                    ("/test", "[]", "2026-01-02T00:00:00"),
                ]
            },
        )
        storage = _make_storage(fake, monkeypatch)
        info = storage.get_scope_info("/")
        sql, _ = fake.executed[-1]
        assert "WHERE" not in sql
        assert info.record_count == 2
        assert info.child_scopes == ["/test"]

    def test_table_missing_empty_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor()
        storage = _make_storage(fake, monkeypatch)
        info = storage.get_scope_info("/test")
        assert info.record_count == 0
        assert info.categories == []
        assert info.oldest_record is None
        assert info.newest_record is None
        assert info.child_scopes == []


class TestListScopes:
    def test_immediate_children_under_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={
                "DISTINCT": [("/test",), ("/test/deep/x",), ("/other",), ("/",)]
            },
        )
        storage = _make_storage(fake, monkeypatch)
        assert storage.list_scopes("/") == ["/other", "/test"]
        sql, _ = fake.executed[-1]
        assert sql.startswith("SELECT DISTINCT scope FROM memories")

    def test_immediate_children_under_parent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={"DISTINCT": [("/test/a",), ("/test/deep/x",)]},
        )
        storage = _make_storage(fake, monkeypatch)
        assert storage.list_scopes("/test") == ["/test/a", "/test/deep"]
        sql, params = fake.executed[-1]
        assert "WHERE scope LIKE %s" in sql
        assert params == ("/test/%",)

    def test_table_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor()
        storage = _make_storage(fake, monkeypatch)
        assert storage.list_scopes() == []


class TestListCategories:
    def test_counts_json_expansion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={
                "FROM memories": [('["x"]',), ('["x", "y"]',), ("[]",)]
            },
        )
        storage = _make_storage(fake, monkeypatch)
        assert storage.list_categories() == {"x": 2, "y": 1}

    def test_scope_prefix_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(
            fetchone_map={"pg_tables": (1,)},
            fetchall_map={"FROM memories": [('["x"]',)]},
        )
        storage = _make_storage(fake, monkeypatch)
        assert storage.list_categories(scope_prefix="/test") == {"x": 1}
        sql, params = fake.executed[-1]
        assert "WHERE scope LIKE %s" in sql
        assert params == ("/test%",)


class TestCount:
    def test_count_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,), "count(*)": (3,)})
        storage = _make_storage(fake, monkeypatch)
        assert storage.count() == 3

    def test_count_with_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,), "count(*)": (2,)})
        storage = _make_storage(fake, monkeypatch)
        assert storage.count("/test") == 2
        sql, params = fake.executed[-1]
        assert "WHERE scope LIKE %s" in sql
        assert params == ("/test%",)

    def test_table_missing_returns_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor()
        storage = _make_storage(fake, monkeypatch)
        assert storage.count() == 0


class TestReset:
    def test_reset_full_drops_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)})
        storage = _make_storage(fake, monkeypatch)
        storage._dim = 8
        storage.reset()
        sql, _ = fake.executed[-1]
        assert sql == "DROP TABLE IF EXISTS memories"
        assert storage._dim is None

    def test_reset_scoped_deletes_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor(fetchone_map={"pg_tables": (1,)}, rowcount=1)
        storage = _make_storage(fake, monkeypatch)
        storage._dim = 8
        storage.reset(scope_prefix="/test/a")
        sql, params = fake.executed[-1]
        assert sql == "DELETE FROM memories WHERE scope LIKE %s"
        assert params == ("/test/a%",)
        assert storage._dim == 8  # scoped reset keeps the known dim

    def test_reset_missing_table_is_noop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakeCursor()
        storage = _make_storage(fake, monkeypatch)
        storage.reset()
        assert not any(
            "DROP" in sql or "DELETE" in sql for sql, _ in fake.executed
        )


class TestScopePattern:
    def test_escapes_like_wildcards(self) -> None:
        assert gs.GaussDBStorage._scope_pattern("/a\\b%c_d") == "/a\\\\b\\%c\\_d%"
        assert gs.GaussDBStorage._scope_pattern("/test") == "/test%"
        assert gs.GaussDBStorage._scope_pattern("/test/") == "/test/%"


# asyncio.run() needs a loopback socketpair for ProactorEventLoop._make_self_pipe,
# which the global --block-network plugin otherwise disables (no real network is
# touched — the storage is faked).
@pytest.mark.block_network(allowed_hosts=[r"127\.0\.0\.1", r"localhost", r"::1"])
def test_async_wrappers(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeCursor(
        fetchone_map={"pg_tables": (1,), "WHERE id = %s": None},
        fetchall_map={"FROM memories": [_row(rid="hit", distance=0.0)]},
        rowcount=1,
    )
    storage = _make_storage(fake, monkeypatch)
    rec = MemoryRecord(content="async", embedding=V1)

    asyncio.run(storage.asave([rec]))
    assert any("MERGE INTO memories" in sql for sql, _ in fake.executed)

    results = asyncio.run(storage.asearch(V1, limit=5))
    assert [r.id for r, _ in results] == ["hit"]

    assert asyncio.run(storage.adelete(record_ids=["x"])) == 1
    asyncio.run(storage.asave([]))  # empty batch is a no-op


def test_memory_string_storage_gaussdb(monkeypatch: pytest.MonkeyPatch) -> None:
    """storage='gaussdb' string spec builds a GaussDBStorage from env config."""
    monkeypatch.setenv("GAUSSDB_HOST", "unit-host")
    from crewai.memory.storage.gaussdb_storage import GaussDBStorage
    from crewai.memory.unified_memory import Memory

    m = Memory(storage="gaussdb")
    try:
        assert isinstance(m._storage, GaussDBStorage)
        assert m._storage.config.host == "unit-host"
    finally:
        m.close()


requires_gaussdb = pytest.mark.skipif(
    os.environ.get("GAUSSDB_TEST", "").lower() != "1",
    reason="requires GAUSSDB_TEST=1 and a reachable GaussDB instance",
)


@requires_gaussdb
# pytest-recording's --block-network patches socket.socket.connect, which breaks
# Windows asyncio.run(): ProactorEventLoop._make_self_pipe needs a loopback
# socketpair. Allow loopback only (psycopg2 connects from C and is unaffected).
@pytest.mark.block_network(allowed_hosts=[r"127\.0\.0\.1", r"localhost", r"::1"])
class TestGaussDBStorageIntegration:
    def test_full_roundtrip(self) -> None:
        from crewai.gaussdb.connection import cursor as db_cursor, reset_pool

        cfg = GaussDBConfig.from_env()
        try:
            with db_cursor(cfg) as cur:
                cur.execute("DROP TABLE IF EXISTS memories")

            storage = gs.GaussDBStorage(config=cfg)
            recs = [
                MemoryRecord(
                    content="alpha",
                    scope="/test/a",
                    categories=["x"],
                    metadata={"env": "prod"},
                    importance=0.8,
                    embedding=V1,
                    created_at=datetime(2026, 1, 1),
                    last_accessed=datetime(2026, 1, 1),
                ),
                MemoryRecord(
                    content="beta",
                    scope="/test/b",
                    categories=["x", "y"],
                    embedding=V2,
                    created_at=datetime(2026, 1, 2),
                    last_accessed=datetime(2026, 1, 2),
                ),
                MemoryRecord(
                    content="gamma",
                    scope="/test/a",
                    metadata={"env": "dev"},
                    embedding=V3,
                    created_at=datetime(2026, 1, 3),
                    last_accessed=datetime(2026, 1, 3),
                ),
            ]
            storage.save(recs)
            assert storage.count() == 3

            results = storage.search(V1, limit=10)
            assert results[0][0].id == recs[0].id
            assert abs(results[0][1] - 1.0) < 1e-6
            assert [r.id for r, _ in storage.search(V1, min_score=0.99, limit=10)] == [
                recs[0].id
            ]
            assert [
                r.id for r, _ in storage.search(V1, scope_prefix="/test/b", limit=10)
            ] == [recs[1].id]
            assert [r.id for r, _ in storage.search(V1, categories=["y"], limit=10)] == [
                recs[1].id
            ]
            assert [
                r.id
                for r, _ in storage.search(V1, metadata_filter={"env": "dev"}, limit=10)
            ] == [recs[2].id]

            got = storage.get_record(recs[0].id)
            assert got is not None
            assert got.content == "alpha"
            assert got.categories == ["x"]
            assert got.metadata == {"env": "prod"}
            assert got.importance == 0.8
            assert got.created_at == datetime(2026, 1, 1)
            assert got.embedding is None

            listed = storage.list_records(scope_prefix="/test")
            assert [r.id for r in listed] == [recs[2].id, recs[1].id, recs[0].id]
            page = storage.list_records(scope_prefix="/test", limit=1, offset=1)
            assert [r.id for r in page] == [recs[1].id]

            recs[0].content = "alpha2"
            recs[0].last_accessed = datetime(2026, 2, 1)
            storage.update(recs[0])
            got2 = storage.get_record(recs[0].id)
            assert got2 is not None
            assert got2.content == "alpha2"
            assert got2.last_accessed == datetime(2026, 2, 1)
            assert storage.count() == 3  # update replaced, not added

            # touch_records: recall side-effect bumps last_accessed (=ANY list
            # adaptation is exercised for real here).
            touched = storage.touch_records([recs[1].id])
            assert touched == 1
            after_touch = storage.get_record(recs[1].id)
            assert after_touch is not None
            assert after_touch.last_accessed > datetime(2026, 1, 2)
            assert storage.touch_records([]) == 0

            assert storage.count("/test") == 3
            assert storage.list_categories() == {"x": 2, "y": 1}
            assert storage.list_categories(scope_prefix="/test/b") == {"x": 1, "y": 1}

            info = storage.get_scope_info("/test")
            assert info.record_count == 3
            assert info.categories == ["x", "y"]
            assert info.oldest_record == datetime(2026, 1, 1)
            assert info.newest_record == datetime(2026, 1, 3)
            assert info.child_scopes == ["/test/a", "/test/b"]

            assert storage.list_scopes("/") == ["/test"]
            assert storage.list_scopes("/test") == ["/test/a", "/test/b"]

            assert storage.delete(record_ids=[recs[2].id]) == 1
            assert storage.count() == 2

            storage.reset(scope_prefix="/test/a")
            assert storage.count() == 1

            storage.reset()
            assert storage.count() == 0
            assert storage._dim is None
        finally:
            reset_pool()

    def test_dim_mismatch_on_existing_table(self) -> None:
        from crewai.gaussdb.connection import cursor as db_cursor, reset_pool

        cfg = GaussDBConfig.from_env()
        try:
            with db_cursor(cfg) as cur:
                cur.execute("DROP TABLE IF EXISTS memories")
            storage = gs.GaussDBStorage(config=cfg)
            storage.save([MemoryRecord(content="8d", embedding=V1)])
            with pytest.raises(EmbeddingDimensionMismatchError) as exc_info:
                storage.save([MemoryRecord(content="16d", embedding=[0.5] * 16)])
            assert exc_info.value.stored_dim == 8
            assert exc_info.value.new_dim == 16
        finally:
            with db_cursor(cfg) as cur:
                cur.execute("DROP TABLE IF EXISTS memories")
            reset_pool()

    def test_async_roundtrip(self) -> None:
        from crewai.gaussdb.connection import cursor as db_cursor, reset_pool

        cfg = GaussDBConfig.from_env()
        try:
            with db_cursor(cfg) as cur:
                cur.execute("DROP TABLE IF EXISTS memories")
            storage = gs.GaussDBStorage(config=cfg)
            rec = MemoryRecord(content="via-asave", scope="/async", embedding=V1)
            asyncio.run(storage.asave([rec]))
            results = asyncio.run(storage.asearch(V1, limit=5))
            assert [r.id for r, _ in results] == [rec.id]
            assert asyncio.run(storage.adelete(record_ids=[rec.id])) == 1
            assert storage.count() == 0
            storage.reset()
        finally:
            reset_pool()

    def test_high_dim_diskann_end_to_end(self) -> None:
        """1536 dims walk the full chain: save -> auto-selected GsDiskANN+PQ
        index -> search. Centralized only: distributed instances reject
        >1024-dim tables at CREATE TABLE, so skip there per the gate."""
        from crewai.gaussdb.connection import cursor as db_cursor, reset_pool
        from crewai.gaussdb.vector import is_distributed

        cfg = GaussDBConfig.from_env()
        with db_cursor(cfg) as cur:
            if is_distributed(cur):
                pytest.skip("distributed instances cap dims at 1024")
            cur.execute("DROP TABLE IF EXISTS memories")
        try:
            storage = gs.GaussDBStorage(config=cfg)
            records = [
                MemoryRecord(
                    content=f"mem {i}",
                    scope="/hd/test",
                    embedding=[
                        0.001 * ((i % 7) - 3) + 0.01 * j for j in range(1536)
                    ],
                )
                for i in range(20)
            ]
            storage.save(records)  # first insert pins dim 1536 -> DiskANN+PQ

            with db_cursor(cfg) as cur:
                cur.execute(
                    "SELECT indexdef FROM pg_indexes WHERE tablename = 'memories' "
                    "AND lower(indexdef) LIKE '%gsdiskann%'"
                )
                defs = [r[0] for r in cur.fetchall()]
            assert defs, "expected a GSDISKANN index on memories"
            assert "pq_nseg=96" in defs[0].lower(), defs[0]

            # Search exercises the DiskANN probe GUC branch (_dim=1536 > 1024).
            hits = storage.search(
                [0.01 * j for j in range(1536)], scope_prefix="/hd/test", limit=3
            )
            assert len(hits) == 3
            assert hits[0][0].content.startswith("mem ")

            storage.reset()  # cleanup: DROP memories
        finally:
            reset_pool()
