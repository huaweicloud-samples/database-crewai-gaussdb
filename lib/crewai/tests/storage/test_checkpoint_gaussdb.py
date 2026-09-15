"""Tests for the GaussDB checkpoint provider."""

from __future__ import annotations

import contextlib
import os
from unittest.mock import MagicMock

import pytest

from crewai.state.provider import gaussdb_provider as g


def _install_fakes(monkeypatch: pytest.MonkeyPatch, *, fetchone=None, rowcount=0):
    executed: list[tuple[str, tuple]] = []

    class FakeCursor(MagicMock):
        def execute(self, sql, params=None):  # type: ignore[no-untyped-def]
            executed.append((sql, params or ()))
            return self

        def fetchone(self):  # type: ignore[no-untyped-def]
            return fetchone

    fake = FakeCursor()
    fake.rowcount = rowcount

    def _fake_cursor_ctx(config):  # noqa: ANN001
        @contextlib.contextmanager
        def ctx():
            yield fake

        return ctx()

    monkeypatch.setattr(g, "cursor", _fake_cursor_ctx)
    provider = g.GaussDBProvider.model_construct(provider_type="gaussdb")
    return provider, executed, fake


class TestGaussDBProviderUnit:
    def test_checkpoint_inserts_and_returns_gaussdb_location(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, executed, _ = _install_fakes(monkeypatch)
        location = provider.checkpoint('{"a": 1}', "./whatever")
        assert location.startswith("gaussdb#")
        assert len(executed) == 3  # 2 DDL + 1 INSERT
        insert_sql, insert_params = executed[-1]
        assert "INSERT INTO checkpoints" in insert_sql
        assert "%s::jsonb" in insert_sql
        assert insert_params[3] == "main"
        assert insert_params[4] == '{"a": 1}'

    def test_checkpoint_with_parent_and_branch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, executed, _ = _install_fakes(monkeypatch)
        provider.checkpoint("{}", "./x", parent_id="p-1", branch="feature")
        _, params = executed[-1]
        assert params[2] == "p-1"
        assert params[3] == "feature"

    def test_from_checkpoint_reads_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider, executed, _ = _install_fakes(monkeypatch, fetchone=('{"a": 1}',))
        result = provider.from_checkpoint("gaussdb#abc_12345678")
        assert result == '{"a": 1}'
        sql, params = executed[-1]
        assert "data::text" in sql
        assert params == ("abc_12345678",)

    def test_from_checkpoint_missing_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        provider, _, _ = _install_fakes(monkeypatch)
        with pytest.raises(ValueError, match="Checkpoint not found"):
            provider.from_checkpoint("gaussdb#nope")

    def test_extract_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider, _, _ = _install_fakes(monkeypatch)
        assert provider.extract_id("gaussdb#20260913T120000_abcd1234") == (
            "20260913T120000_abcd1234"
        )

    def test_prune_keeps_latest_n(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider, executed, _ = _install_fakes(monkeypatch, rowcount=3)
        removed = provider.prune("gaussdb", 5, branch="main")
        assert removed == 3
        sql, params = executed[-1]
        assert "ORDER BY seq DESC" in sql
        assert params[2] == 5

    def test_prune_negative_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        provider, _, _ = _install_fakes(monkeypatch, rowcount=-1)
        assert provider.prune("gaussdb", 5, branch="main") == 0


@pytest.mark.skipif(
    os.environ.get("GAUSSDB_TEST", "").lower() != "1",
    reason="requires GAUSSDB_TEST=1 and a reachable GaussDB instance",
)
# pytest-recording's --block-network patches socket.socket.connect, which breaks
# Windows asyncio.run(): ProactorEventLoop._make_self_pipe needs a loopback
# socketpair. Allow loopback only (psycopg2 connects from C and is unaffected);
# the tests are skipped unless GAUSSDB_TEST=1 anyway.
@pytest.mark.block_network(allowed_hosts=[r"127\.0\.0\.1", r"localhost", r"::1"])
class TestGaussDBProviderIntegration:
    def test_checkpoint_roundtrip_and_prune(self) -> None:
        from crewai.gaussdb.config import GaussDBConfig
        from crewai.gaussdb.connection import cursor
        from crewai.state.provider.gaussdb_provider import GaussDBProvider

        cfg = GaussDBConfig.from_env()
        with cursor(cfg) as cur:
            for stmt in (
                "DROP TABLE IF EXISTS checkpoints",
                "DROP SEQUENCE IF EXISTS checkpoints_seq",
            ):
                cur.execute(stmt)

        provider = GaussDBProvider()
        loc1 = provider.checkpoint('{"n": 1}', "gaussdb")
        loc2 = provider.checkpoint('{"n": 2}', "gaussdb")
        loc3 = provider.checkpoint('{"n": 3}', "gaussdb")

        assert provider.from_checkpoint(loc1) == '{"n": 1}'
        assert provider.extract_id(loc2).startswith("2026")

        removed = provider.prune("gaussdb", 2, branch="main")
        assert removed == 1
        with pytest.raises(ValueError, match="Checkpoint not found"):
            provider.from_checkpoint(loc1)
        assert provider.from_checkpoint(loc3) == '{"n": 3}'

    def test_async_roundtrip(self) -> None:
        import asyncio

        from crewai.state.provider.gaussdb_provider import GaussDBProvider

        provider = GaussDBProvider()
        loc = asyncio.run(provider.acheckpoint('{"a": true}', "gaussdb"))
        assert asyncio.run(provider.afrom_checkpoint(loc)) == '{"a": true}'
