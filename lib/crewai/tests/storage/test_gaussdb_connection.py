"""Unit tests for crewai.gaussdb config and connection helpers."""

from __future__ import annotations

import os

import pytest

from crewai.gaussdb.config import GaussDBConfig, is_gaussdb_backend


class TestGaussDBConfig:
    def test_from_env_reads_variables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GAUSSDB_HOST", "10.0.0.1")
        monkeypatch.setenv("GAUSSDB_PORT", "19995")
        monkeypatch.setenv("GAUSSDB_USER", "u1")
        monkeypatch.setenv("GAUSSDB_PASSWORD", "p1")
        monkeypatch.setenv("GAUSSDB_DATABASE", "db1")
        cfg = GaussDBConfig.from_env()
        assert cfg.host == "10.0.0.1"
        assert cfg.port == 19995
        assert cfg.user == "u1"
        assert cfg.password == "p1"
        assert cfg.database == "db1"
        assert cfg.min_connections == 1
        assert cfg.max_connections == 10

    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "GAUSSDB_HOST", "GAUSSDB_PORT", "GAUSSDB_USER",
            "GAUSSDB_PASSWORD", "GAUSSDB_DATABASE",
            "GAUSSDB_MIN_CONNECTIONS", "GAUSSDB_MAX_CONNECTIONS",
        ):
            monkeypatch.delenv(var, raising=False)
        cfg = GaussDBConfig.from_env()
        assert cfg.host == "localhost"
        assert cfg.port == 5432
        assert cfg.database == "crewai"
        assert cfg.min_connections == 1
        assert cfg.max_connections == 10

    def test_is_gaussdb_backend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("CREWAI_STORAGE_BACKEND", raising=False)
        assert is_gaussdb_backend() is False
        monkeypatch.setenv("CREWAI_STORAGE_BACKEND", "gaussdb")
        assert is_gaussdb_backend() is True
        monkeypatch.setenv("CREWAI_STORAGE_BACKEND", "GaussDB ")
        assert is_gaussdb_backend() is True  # 大小写与空白不敏感
        monkeypatch.setenv("CREWAI_STORAGE_BACKEND", "sqlite")
        assert is_gaussdb_backend() is False


class TestPool:
    def test_pool_reuse_and_reset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """同一配置复用同一池，配置变化或 reset 后重建（用假池避免真连接）。"""
        import crewai.gaussdb.connection as conn_mod
        from crewai.gaussdb.config import GaussDBConfig

        created: list[str] = []

        class FakePool:
            def __init__(self, tag: str) -> None:
                self.tag = tag
                created.append(tag)

            def closeall(self) -> None:
                pass

        monkeypatch.setattr(
            "crewai.gaussdb.connection._create_pool_instance",
            lambda cfg, minc, maxc: FakePool(cfg.database),
        )

        cfg_a = GaussDBConfig(database="db_a")
        pool1 = conn_mod.get_pool(cfg_a)
        pool2 = conn_mod.get_pool(GaussDBConfig(database="db_a"))
        assert pool1 is pool2

        cfg_b = GaussDBConfig(database="db_b")
        pool3 = conn_mod.get_pool(cfg_b)
        assert pool3 is not pool1

        conn_mod.reset_pool()
        pool4 = conn_mod.get_pool(cfg_b)
        assert pool4 is not pool3
        assert len(created) == 3


requires_gaussdb = pytest.mark.skipif(
    os.environ.get("GAUSSDB_TEST", "").lower() != "1",
    reason="requires GAUSSDB_TEST=1 and a reachable GaussDB instance",
)


@requires_gaussdb
class TestGaussDBIntegration:
    def test_connect_and_query(self) -> None:
        from crewai.gaussdb.config import GaussDBConfig
        from crewai.gaussdb.connection import cursor

        cfg = GaussDBConfig.from_env()
        with cursor(cfg) as cur:
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
        assert "GaussDB Kernel 507" in version

    def test_transaction_rollback_on_error(self) -> None:
        from crewai.gaussdb.config import GaussDBConfig
        from crewai.gaussdb.connection import cursor

        cfg = GaussDBConfig.from_env()
        with pytest.raises(Exception):
            with cursor(cfg) as cur:
                cur.execute("SELECT 1")
                cur.execute("SELECT * FROM __definitely_not_a_table__")
