"""CLI checkpoint commands against a GaussDB backend."""

from __future__ import annotations

import os

import pytest

requires_gaussdb = pytest.mark.skipif(
    os.environ.get("GAUSSDB_TEST", "").lower() != "1",
    reason="requires GAUSSDB_TEST=1 and a reachable GaussDB instance",
)


@pytest.fixture()
def seeded_gaussdb_checkpoints():
    from crewai.gaussdb.config import GaussDBConfig
    from crewai.gaussdb.connection import cursor, reset_pool
    from crewai.state.provider.gaussdb_provider import GaussDBProvider

    cfg = GaussDBConfig.from_env()
    with cursor(cfg) as cur:
        for stmt in (
            "DROP TABLE IF EXISTS checkpoints",
            "DROP SEQUENCE IF EXISTS checkpoints_seq",
        ):
            cur.execute(stmt)

    try:
        provider = GaussDBProvider()
        payload = (
            '{"entities": [{"entity_type": "flow", "name": "f1", "tasks": []}], '
            '"trigger": "kickoff", "branch": "main", "parent_id": null}'
        )
        yield [provider.checkpoint(payload, "gaussdb") for _ in range(3)]
    finally:
        # Leftover pooled connections slow down pytest exit.
        reset_pool()


@requires_gaussdb
# pytest-recording's --block-network patches socket.socket.connect, which breaks
# Windows asyncio.run(): ProactorEventLoop._make_self_pipe needs a loopback
# socketpair. Allow loopback only (psycopg2 connects from C and is unaffected);
# the tests are skipped unless GAUSSDB_TEST=1 anyway.
@pytest.mark.block_network(allowed_hosts=[r"127\.0\.0\.1", r"localhost", r"::1"])
class TestCheckpointCliGaussDB:
    def test_list_checkpoints(self, seeded_gaussdb_checkpoints, capsys) -> None:
        from crewai_cli.checkpoint_cli import list_checkpoints

        list_checkpoints("gaussdb")
        out = capsys.readouterr().out
        assert "Found 3 checkpoint(s)" in out
        assert "GaussDB" in out

    def test_info_checkpoint_by_location_id(
        self, seeded_gaussdb_checkpoints, capsys
    ) -> None:
        from crewai_cli.checkpoint_cli import info_checkpoint

        info_checkpoint(seeded_gaussdb_checkpoints[0])
        out = capsys.readouterr().out
        assert "Branch:  main" in out
        assert "Trigger: kickoff" in out

    def test_info_checkpoint_by_short_id_fallback(
        self, seeded_gaussdb_checkpoints, capsys
    ) -> None:
        """A short (suffix) id resolves through the LIKE fallback."""
        from crewai_cli.checkpoint_cli import info_checkpoint

        location = seeded_gaussdb_checkpoints[1]
        short_id = location.rsplit("#", 1)[1][-8:]
        info_checkpoint(f"gaussdb#{short_id}")
        out = capsys.readouterr().out
        assert "Checkpoint not found in GaussDB" not in out
        full_id = location.rsplit("#", 1)[1]
        assert f"Name:    {full_id}" in out

    def test_info_checkpoint_latest(self, seeded_gaussdb_checkpoints, capsys) -> None:
        from crewai_cli.checkpoint_cli import info_checkpoint

        info_checkpoint("gaussdb")
        out = capsys.readouterr().out
        assert "Latest checkpoint:" in out

    def test_prune_keep_n(self, seeded_gaussdb_checkpoints, capsys) -> None:
        from crewai_cli.checkpoint_cli import prune_checkpoints

        prune_checkpoints("gaussdb", keep=2, older_than=None)
        out = capsys.readouterr().out
        assert "Pruned 1 checkpoint(s)" in out

    def test_prune_dry_run(self, seeded_gaussdb_checkpoints, capsys) -> None:
        from crewai_cli.checkpoint_cli import prune_checkpoints

        prune_checkpoints("gaussdb", keep=2, older_than=None, dry_run=True)
        out = capsys.readouterr().out
        assert "Would prune from 3 checkpoint(s)" in out
