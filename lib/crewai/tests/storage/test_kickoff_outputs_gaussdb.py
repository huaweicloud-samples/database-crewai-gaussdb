"""Tests for the GaussDB kickoff task outputs storage."""

from __future__ import annotations

import contextlib
import gc
import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crewai.memory.storage import kickoff_task_outputs_gaussdb as g


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fetchall: list[tuple] | None = None,
    rowcount: int = 1,
):
    """Patch the module's cursor()/store_lock with a recorder; build a bare storage.

    The instance is created via __new__ so _initialize_db (and its CREATE TABLE)
    stays out of the recorded statements.
    """
    executed: list[tuple[str, tuple]] = []

    class FakeCursor(MagicMock):
        def execute(self, sql, params=None):  # type: ignore[no-untyped-def]
            executed.append((sql, params or ()))
            return self

        def fetchall(self):  # type: ignore[no-untyped-def]
            return fetchall if fetchall is not None else []

    fake = FakeCursor()
    fake.rowcount = rowcount

    def _fake_cursor_ctx(config):  # noqa: ANN001
        @contextlib.contextmanager
        def ctx():
            yield fake

        return ctx()

    monkeypatch.setattr(g, "cursor", _fake_cursor_ctx)
    monkeypatch.setattr(g, "store_lock", lambda name: contextlib.nullcontext())
    storage = g.GaussDBKickoffTaskOutputsStorage.__new__(
        g.GaussDBKickoffTaskOutputsStorage
    )
    storage.config = g.GaussDBConfig.from_env()
    storage._lock_name = "fake"
    return storage, executed, fake


class TestGaussDBKickoffBehavior:
    def test_add_uses_merge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        storage, executed, _ = _install_fakes(monkeypatch)
        task = MagicMock()
        task.id = "t-1"
        task.expected_output = "out"
        storage.add(task, {"raw": "x"}, task_index=0)
        assert len(executed) == 1
        sql, params = executed[0]
        assert "MERGE INTO latest_kickoff_task_outputs" in sql
        assert "ON CONFLICT" not in sql
        assert params[0] == "t-1"
        assert "raw" in params[2]

    def test_update_builds_dynamic_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        storage, executed, _ = _install_fakes(monkeypatch)
        storage.update(0, output={"raw": "y"}, was_replayed=True)
        sql, params = executed[0]
        assert sql.startswith("UPDATE latest_kickoff_task_outputs SET")
        assert "WHERE task_index = %s" in sql
        assert params[-1] == 0

    def test_update_warns_on_zero_rows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        storage, executed, _ = _install_fakes(monkeypatch, rowcount=0)
        storage.update(7, output={"raw": "y"})
        # 不抛异常即通过（告警路径）；确认语句确实执行过
        assert executed and executed[0][1][-1] == 7

    def test_load_maps_columns_by_position(self, monkeypatch: pytest.MonkeyPatch) -> None:
        storage, executed, _ = _install_fakes(
            monkeypatch,
            fetchall=[
                (
                    "t-1",
                    "exp",
                    '{"raw": "x"}',
                    0,
                    '{"k": "v"}',
                    True,
                    datetime(2026, 9, 13, 1, 2, 3),
                )
            ],
        )
        rows = storage.load()
        assert rows[0]["task_id"] == "t-1"
        assert rows[0]["output"] == {"raw": "x"}
        assert rows[0]["was_replayed"] is True
        assert rows[0]["timestamp"] == "2026-09-13 01:02:03"
        # 列序固定：SELECT 显式列出列，不依赖 SELECT *
        assert "SELECT task_id, expected_output, output, task_index, inputs, was_replayed, timestamp" in " ".join(
            executed[0][0].split()
        )

    def test_delete_all_executes_delete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        storage, executed, _ = _install_fakes(monkeypatch)
        storage.delete_all()
        sql, _ = executed[0]
        assert "DELETE FROM latest_kickoff_task_outputs" in sql


class TestHandlerBranch:
    def test_handler_selects_gaussdb_when_env_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CREWAI_STORAGE_BACKEND", "gaussdb")
        monkeypatch.setenv("GAUSSDB_DATABASE", "unit_test_db")
        from crewai.utilities.task_output_storage_handler import TaskOutputStorageHandler

        # 打桩 _initialize_db 防真连库（__init__ 只读 env，不建连接）
        monkeypatch.setattr(
            g.GaussDBKickoffTaskOutputsStorage, "_initialize_db", lambda self: None
        )
        handler = TaskOutputStorageHandler()
        assert isinstance(
            handler.storage, g.GaussDBKickoffTaskOutputsStorage
        )

    def test_handler_selects_sqlite_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Land the SQLite db in this test's tmp_path instead of the shared user
        # storage dir (Task 4 lesson: Windows keeps file handles locked).
        monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path))
        monkeypatch.delenv("CREWAI_STORAGE_BACKEND", raising=False)
        from crewai.memory.storage.kickoff_task_outputs_storage import (
            KickoffTaskOutputsSQLiteStorage,
        )
        from crewai.utilities.task_output_storage_handler import TaskOutputStorageHandler

        handler = TaskOutputStorageHandler()
        assert isinstance(handler.storage, KickoffTaskOutputsSQLiteStorage)
        gc.collect()


@pytest.mark.skipif(
    os.environ.get("GAUSSDB_TEST", "").lower() != "1",
    reason="requires GAUSSDB_TEST=1 and a reachable GaussDB instance",
)
class TestGaussDBKickoffIntegration:
    def test_full_cycle(self) -> None:
        from crewai.gaussdb.config import GaussDBConfig
        from crewai.gaussdb.connection import cursor

        cfg = GaussDBConfig.from_env()
        with cursor(cfg) as cur:
            cur.execute("DROP TABLE IF EXISTS latest_kickoff_task_outputs")

        storage = g.GaussDBKickoffTaskOutputsStorage()
        task = MagicMock()
        task.id = "t-1"
        task.expected_output = "expected"
        storage.add(task, {"raw": "first"}, task_index=0, inputs={"q": "hi"})
        # 同 task_id MERGE 覆盖（与 SQLite INSERT OR REPLACE 一致：整行覆盖，
        # 所以第二个 add 也带 inputs，否则会清空该列）
        storage.add(task, {"raw": "second"}, task_index=0, inputs={"q": "hi"})
        storage.update(0, output={"raw": "updated"}, was_replayed=True)

        rows = storage.load()
        assert len(rows) == 1
        assert rows[0]["output"] == {"raw": "updated"}
        assert rows[0]["was_replayed"] is True
        assert rows[0]["inputs"] == {"q": "hi"}
        assert rows[0]["timestamp"] is not None  # timestamp 列 + DEFAULT 生效

        storage.delete_all()
        assert storage.load() == []
