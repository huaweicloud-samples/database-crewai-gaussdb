"""Tests for the GaussDB flow persistence backend."""

from __future__ import annotations

import contextlib
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
        sqls = [sql for sql, _ in executed]
        assert any("MERGE INTO pending_feedback" in sql for sql in sqls)
        assert not any("ON CONFLICT" in sql for sql in sqls)
        merge = next(sql for sql in sqls if "MERGE INTO pending_feedback" in sql)
        assert "WHEN MATCHED THEN UPDATE" in merge
        assert "WHEN NOT MATCHED THEN INSERT" in merge

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
