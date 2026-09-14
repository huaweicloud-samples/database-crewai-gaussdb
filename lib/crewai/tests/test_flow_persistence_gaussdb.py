"""Tests for the GaussDB flow persistence backend."""

from __future__ import annotations

import contextlib
import gc
import json
import os
from unittest.mock import MagicMock

import pytest


def _install_fake_cursor(monkeypatch: pytest.MonkeyPatch):
    """Replace gaussdb cursor() with a recorder; return (executed, cursor_mock)."""
    from crewai.flow.persistence import gaussdb as g

    executed: list[tuple[str, tuple]] = []

    class FakeCursor(MagicMock):
        def execute(self, sql, params=None):  # type: ignore[no-untyped-def]
            executed.append((sql, params or ()))
            return MagicMock()

    fake = FakeCursor()

    @staticmethod
    def _fake_cursor_ctx(config):  # noqa: ANN001
        @contextlib.contextmanager
        def ctx():
            yield fake

        return ctx()

    monkeypatch.setattr(g, "cursor", _fake_cursor_ctx)
    return executed, fake


def _make(monkeypatch: pytest.MonkeyPatch):
    from crewai.flow.persistence import gaussdb as g

    monkeypatch.setenv("GAUSSDB_DATABASE", "unit_test_db")
    executed, fake = _install_fake_cursor(monkeypatch)
    inst = g.GaussDBFlowPersistence.model_construct(
        persistence_type="GaussDBFlowPersistence",
        config=g.GaussDBConfig.from_env(),
    )
    inst._lock_name = "gaussdb:unit_test_db"  # model_construct leaves PrivateAttr unset
    return inst, executed, fake


class TestGaussDBFlowPersistenceSQL:
    def test_registered_in_persistence_registry(self) -> None:
        from crewai.flow.persistence.base import _persistence_registry
        from crewai.flow.persistence.gaussdb import GaussDBFlowPersistence

        assert _persistence_registry["GaussDBFlowPersistence"] is GaussDBFlowPersistence

    def test_save_state_uses_percent_placeholders(self, monkeypatch) -> None:
        inst, executed, _ = _make(monkeypatch)
        inst.save_state("f-1", "step1", {"a": 1})
        sql, params = executed[0]
        assert "INSERT INTO flow_states" in sql
        assert "%s" in sql and "?" not in sql
        assert params[0] == "f-1" and params[1] == "step1"
        assert json.loads(params[3]) == {"a": 1}

    def test_save_pending_feedback_uses_merge(self, monkeypatch) -> None:
        inst, executed, _ = _make(monkeypatch)
        ctx = MagicMock()
        ctx.method_name = "ask"
        ctx.to_dict.return_value = {"method": "ask"}
        inst.save_pending_feedback("f-1", ctx, {"a": 1})
        # 两语句（INSERT flow_states + MERGE pending_feedback）同事务
        assert len(executed) == 2
        insert_sql, insert_params = executed[0]
        assert "INSERT INTO flow_states" in insert_sql
        assert insert_params[0] == "f-1"
        assert insert_params[1] == "ask"
        assert json.loads(insert_params[3]) == {"a": 1}
        merge_sql, merge_params = executed[1]
        assert "MERGE INTO pending_feedback" in merge_sql
        assert "ON CONFLICT" not in merge_sql
        assert "WHEN MATCHED THEN UPDATE" in merge_sql
        assert "WHEN NOT MATCHED THEN INSERT" in merge_sql
        # MERGE 参数顺序：flow_uuid, context_json, state_json, created_at
        assert merge_params[0] == "f-1"
        assert json.loads(merge_params[1]) == {"method": "ask"}
        assert json.loads(merge_params[2]) == {"a": 1}

    def test_clear_pending_feedback(self, monkeypatch) -> None:
        inst, executed, _ = _make(monkeypatch)
        inst.clear_pending_feedback("f-1")
        sql, params = executed[0]
        assert "DELETE FROM pending_feedback" in sql
        assert params == ("f-1",)

    def test_load_pending_feedback_parses_row(self, monkeypatch) -> None:
        import json as json_mod
        from crewai.flow.persistence import gaussdb as g

        ctx_dict = {
            "flow_id": "f-1", "flow_class": "tests.TinyFlow",
            "method_name": "ask", "method_output": {"t": 1}, "message": "m",
        }
        _executed, fake = _install_fake_cursor(monkeypatch)
        fake.fetchone.return_value = (
            json_mod.dumps({"a": 1}), json_mod.dumps(ctx_dict),
        )
        inst = g.GaussDBFlowPersistence.model_construct(
            persistence_type="GaussDBFlowPersistence",
            config=g.GaussDBConfig.from_env(),
        )
        loaded = inst.load_pending_feedback("f-1")
        assert loaded is not None
        state, ctx = loaded
        assert state == {"a": 1}
        assert ctx.method_name == "ask"

    def test_load_state_selects_latest(self, monkeypatch) -> None:
        from crewai.flow.persistence import gaussdb as g

        executed, fake = _install_fake_cursor(monkeypatch)
        fake.fetchone.return_value = ('{"a": 1}',)
        inst = g.GaussDBFlowPersistence.model_construct(
            persistence_type="GaussDBFlowPersistence",
            config=g.GaussDBConfig.from_env(),
        )
        assert inst.load_state("f-1") == {"a": 1}
        sql = executed[0][0]
        assert "ORDER BY id DESC LIMIT 1" in sql

    def test_load_state_none_when_missing(self, monkeypatch) -> None:
        from crewai.flow.persistence import gaussdb as g

        _executed, fake = _install_fake_cursor(monkeypatch)
        fake.fetchone.return_value = None
        inst = g.GaussDBFlowPersistence.model_construct(
            persistence_type="GaussDBFlowPersistence",
            config=g.GaussDBConfig.from_env(),
        )
        assert inst.load_state("nope") is None


@pytest.mark.skipif(
    os.environ.get("GAUSSDB_TEST", "").lower() != "1",
    reason="requires GAUSSDB_TEST=1 and a reachable GaussDB instance",
)
class TestGaussDBFlowPersistenceIntegration:
    @pytest.fixture(autouse=True)
    def _clean(self):
        from crewai.gaussdb.config import GaussDBConfig
        from crewai.gaussdb.connection import cursor

        cfg = GaussDBConfig.from_env()
        with cursor(cfg) as cur:
            for stmt in (
                "DROP TABLE IF EXISTS pending_feedback",
                "DROP TABLE IF EXISTS flow_states",
                "DROP SEQUENCE IF EXISTS flow_states_id_seq",
            ):
                cur.execute(stmt)
        yield

    def _make(self):
        from crewai.flow.persistence.gaussdb import GaussDBFlowPersistence

        return GaussDBFlowPersistence()

    def test_crud_roundtrip(self) -> None:
        inst = self._make()
        inst.save_state("f-1", "step1", {"a": 1})
        inst.save_state("f-1", "step2", {"a": 2})
        assert inst.load_state("f-1") == {"a": 2}
        assert inst.load_state("f-2") is None

    def test_pending_feedback_roundtrip(self) -> None:
        from crewai.flow.async_feedback.types import PendingFeedbackContext

        inst = self._make()
        ctx = PendingFeedbackContext(
            flow_id="f-1",
            flow_class="tests.TinyFlow",
            method_name="ask",
            method_output={"title": "Draft"},
            message="Please review",
        )
        inst.save_pending_feedback("f-1", ctx, {"a": 1})
        inst.save_pending_feedback("f-1", ctx, {"a": 2})  # MERGE 覆盖
        loaded = inst.load_pending_feedback("f-1")
        assert loaded is not None
        state, loaded_ctx = loaded
        assert state == {"a": 2}
        assert loaded_ctx.method_name == "ask"
        inst.clear_pending_feedback("f-1")
        assert inst.load_pending_feedback("f-1") is None

    def test_pydantic_model_state(self) -> None:
        from pydantic import BaseModel

        class S(BaseModel):
            x: int

        inst = self._make()
        inst.save_state("f-3", "m", S(x=5))
        assert inst.load_state("f-3") == {"x": 5}


class TestDefaultFactory:
    def test_default_flow_persistence_returns_gaussdb_when_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import crewai.flow.persistence.factory as factory_mod
        from crewai.flow.persistence import gaussdb as g

        monkeypatch.setenv("CREWAI_STORAGE_BACKEND", "gaussdb")
        monkeypatch.setattr(factory_mod, "_factory", None)

        # Stub __init__ so construction never touches the database (the real
        # model_validator runs init_db and would require a live GaussDB).
        def fake_init(self, **kwargs):  # noqa: ANN001
            pass

        monkeypatch.setattr(g.GaussDBFlowPersistence, "__init__", fake_init)
        backend = factory_mod.default_flow_persistence()
        assert isinstance(backend, g.GaussDBFlowPersistence)

    def test_default_flow_persistence_explicit_factory_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import crewai.flow.persistence.factory as factory_mod

        monkeypatch.setenv("CREWAI_STORAGE_BACKEND", "gaussdb")
        sentinel = object()
        monkeypatch.setattr(factory_mod, "_factory", lambda: sentinel)
        assert factory_mod.default_flow_persistence() is sentinel

    def test_default_flow_persistence_sqlite_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import crewai.flow.persistence.factory as factory_mod
        from crewai.flow.persistence.sqlite import SQLiteFlowPersistence

        monkeypatch.delenv("CREWAI_STORAGE_BACKEND", raising=False)
        monkeypatch.setattr(factory_mod, "_factory", None)
        backend = factory_mod.default_flow_persistence()
        assert isinstance(backend, SQLiteFlowPersistence)
        # The per-call SQLite connections land in reference cycles (cursor ->
        # connection), keeping flow_states.db locked on Windows; a GC pass
        # closes them so the temp CREWAI_STORAGE_DIR can be removed in teardown.
        gc.collect()

    def test_lazy_export_gaussdb_flow_persistence(self) -> None:
        import crewai.flow.persistence as fp
        from crewai.flow.persistence.gaussdb import (
            GaussDBFlowPersistence as Direct,
        )

        cls = fp.GaussDBFlowPersistence
        assert cls is Direct


@pytest.mark.skipif(
    os.environ.get("GAUSSDB_TEST", "").lower() != "1",
    reason="requires GAUSSDB_TEST=1 and a reachable GaussDB instance",
)
# The suite runs with --block-network, which also blocks the loopback
# socketpair that asyncio's ProactorEventLoop needs on Windows when the event
# bus lazily creates its loop on first Flow construction (Linux uses the
# socketpair syscall and is unaffected). Allow only loopback so that in-process
# socketpair works; the GaussDB connection itself goes through libpq and is
# not intercepted by this guard.
@pytest.mark.block_network(allowed_hosts=[r"127\.0\.0\.1"])
class TestFlowEndToEnd:
    def test_persist_decorator_writes_gaussdb(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from uuid import uuid4

        from pydantic import BaseModel

        # @persist() resolves the default backend at decoration time, so the
        # env var must be set before the flow class below is defined.
        monkeypatch.setenv("CREWAI_STORAGE_BACKEND", "gaussdb")

        from crewai.flow import Flow, persist, start

        class TinyState(BaseModel):
            counter: int = 0

        class TinyFlow(Flow[TinyState]):
            @start()
            @persist()
            def begin(self):
                self.state.counter += 1

        flow_id = str(uuid4())
        flow = TinyFlow()
        flow.kickoff(inputs={"id": flow_id})
        assert flow.state.counter == 1

        # The state reached GaussDB under this flow id and reads back.
        from crewai.flow.persistence.gaussdb import GaussDBFlowPersistence

        inst = GaussDBFlowPersistence()
        loaded = inst.load_state(flow_id)
        assert loaded is not None
        assert loaded["counter"] == 1
