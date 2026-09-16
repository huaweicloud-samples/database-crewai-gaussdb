"""CLI log-tasks-outputs against a GaussDB backend."""

from __future__ import annotations

from collections.abc import Generator
import os
from unittest.mock import MagicMock

import pytest

requires_gaussdb = pytest.mark.skipif(
    os.environ.get("GAUSSDB_TEST", "").lower() != "1",
    reason="requires GAUSSDB_TEST=1 and a reachable GaussDB instance",
)


@requires_gaussdb
# pytest-recording's --block-network patches socket.socket.connect, which breaks
# Windows asyncio.run(): ProactorEventLoop._make_self_pipe needs a loopback
# socketpair. Allow loopback only (psycopg2 connects from C and is unaffected);
# the tests are skipped unless GAUSSDB_TEST=1 anyway.
@pytest.mark.block_network(allowed_hosts=[r"127\.0\.0\.1", r"localhost", r"::1"])
class TestTaskOutputsGaussDB:
    @pytest.fixture(autouse=True)
    def _reset_pool(self) -> Generator[None, None, None]:
        # Leftover pooled connections slow down pytest exit.
        yield
        from crewai.gaussdb.connection import reset_pool

        reset_pool()

    def test_load_task_outputs_roundtrip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CREWAI_STORAGE_BACKEND", "gaussdb")
        from crewai.gaussdb.config import GaussDBConfig
        from crewai.gaussdb.connection import cursor
        from crewai.memory.storage.kickoff_task_outputs_gaussdb import (
            GaussDBKickoffTaskOutputsStorage,
        )
        from crewai_cli.task_outputs import load_task_outputs

        with cursor(GaussDBConfig.from_env()) as cur:
            cur.execute("DROP TABLE IF EXISTS latest_kickoff_task_outputs")

        storage = GaussDBKickoffTaskOutputsStorage()
        task = MagicMock()
        task.id = "t-9"
        task.expected_output = "exp"
        storage.add(task, {"raw": "hello"}, task_index=0)

        rows = load_task_outputs()
        assert len(rows) == 1
        assert rows[0]["task_id"] == "t-9"
        assert rows[0]["output"] == {"raw": "hello"}
        assert isinstance(rows[0]["timestamp"], str)

    def test_missing_table_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("CREWAI_STORAGE_BACKEND", "gaussdb")
        from crewai.gaussdb.config import GaussDBConfig
        from crewai.gaussdb.connection import cursor
        from crewai_cli.task_outputs import load_task_outputs

        with cursor(GaussDBConfig.from_env()) as cur:
            cur.execute("DROP TABLE IF EXISTS latest_kickoff_task_outputs")
        # Missing table (first use) degrades to [] like a missing SQLite file.
        assert load_task_outputs() == []


class TestBackendSwitch:
    def test_env_switch_reads_gaussdb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CREWAI_STORAGE_BACKEND=gaussdb routes load_task_outputs to GaussDB."""
        monkeypatch.setenv("CREWAI_STORAGE_BACKEND", "gaussdb")
        monkeypatch.setenv("GAUSSDB_DATABASE", "unit_test_db")
        from crewai_cli import task_outputs as to

        sentinel = [{"task_id": "stub"}]
        monkeypatch.setattr(to, "_load_task_outputs_gaussdb", lambda: sentinel)
        assert to.load_task_outputs() == sentinel

    def test_default_backend_reads_sqlite_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Without the env switch the SQLite file path is read as before."""
        monkeypatch.delenv("CREWAI_STORAGE_BACKEND", raising=False)
        from crewai_cli import task_outputs as to

        monkeypatch.setattr(to, "_db_storage_path", lambda: str(tmp_path))
        assert to.load_task_outputs() == []

    def test_psycopg2_missing_degrades_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """is_gaussdb_backend importable but crewai.gaussdb.connection not → []."""
        monkeypatch.setenv("CREWAI_STORAGE_BACKEND", "gaussdb")
        import sys

        from crewai_cli import task_outputs as to

        # A None value in sys.modules makes the from-import raise ImportError
        # ("import halted") even when the module is already cached — narrow,
        # deterministic, and restored by monkeypatch.
        monkeypatch.setitem(sys.modules, "crewai.gaussdb.connection", None)
        assert to.load_task_outputs() == []
