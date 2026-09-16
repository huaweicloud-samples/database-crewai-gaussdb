"""Unit tests for the shared GaussDB vector helpers."""

from __future__ import annotations

import pytest

from crewai.gaussdb.vector import (
    calc_pq_nseg,
    is_distributed,
    upsert_via_merge,
    validate_dimension,
    vector_index_ddl,
)


class TestCalcPqNseg:
    def test_small_dimensions_return_dim(self) -> None:
        assert calc_pq_nseg(8) == 8
        assert calc_pq_nseg(512) == 512

    def test_mid_dimensions_halve(self) -> None:
        assert calc_pq_nseg(1024) == 512

    def test_large_dimensions_use_known_factors(self) -> None:
        assert calc_pq_nseg(1536) == 96
        assert calc_pq_nseg(2048) == 128
        assert calc_pq_nseg(3072) == 96
        assert calc_pq_nseg(4096) == 128

    def test_prime_dimension_fallback(self) -> None:
        # 1031 是质数：>1024 且无 (96,128,192,256,384,512) 因子 → 兜底取自身
        assert calc_pq_nseg(1031) == 1031


class TestValidateDimension:
    def test_centralized_allows_up_to_4096(self) -> None:
        validate_dimension(1024, distributed=False)
        validate_dimension(3072, distributed=False)
        validate_dimension(4096, distributed=False)

    def test_centralized_rejects_above_4096(self) -> None:
        with pytest.raises(ValueError, match="4096"):
            validate_dimension(4097, distributed=False)

    def test_distributed_rejects_above_1024(self) -> None:
        with pytest.raises(ValueError, match="1024"):
            validate_dimension(1025, distributed=True)

    def test_distributed_allows_1024(self) -> None:
        validate_dimension(1024, distributed=True)


class TestVectorIndexDdl:
    def test_ivfflat_for_small_dim(self) -> None:
        ddl = vector_index_ddl("my_idx", "my_table", "embedding", 1024, distributed=False)
        assert "USING GSIVFFLAT(embedding cosine)" in ddl
        assert "IVF_NLIST = 256" in ddl
        assert "GSDISKANN" not in ddl

    def test_diskann_pq_for_large_dim_centralized(self) -> None:
        ddl = vector_index_ddl("my_idx", "my_table", "embedding", 1536, distributed=False)
        assert "USING GSDISKANN(embedding cosine)" in ddl
        assert "pq_nseg=96" in ddl
        assert "enable_pq=true" in ddl
        assert "subgraph_count=1" in ddl

    def test_large_dim_distributed_raises(self) -> None:
        with pytest.raises(ValueError, match="1024"):
            vector_index_ddl("my_idx", "my_table", "embedding", 1536, distributed=True)


class _RecordingCursor:
    """Minimal cursor double recording execute calls (no DB connection).

    Same recorder idea as _install_fakes in test_kickoff_outputs_gaussdb.py,
    but the helpers under test take a cursor directly — no monkeypatch needed.
    """

    def __init__(self, fetchone_result: tuple | None = None) -> None:
        self.executed: list[tuple[str, tuple]] = []
        self._fetchone_result = fetchone_result

    def execute(self, sql: str, params: tuple | None = None) -> None:
        self.executed.append((sql, params or ()))

    def fetchone(self) -> tuple | None:
        return self._fetchone_result


class TestIsDistributedFakeCursor:
    def test_pgxc_node_count_decides_topology(self) -> None:
        cur = _RecordingCursor(fetchone_result=(3,))
        assert is_distributed(cur) is True
        cur0 = _RecordingCursor(fetchone_result=(0,))  # 集中式：空表 count=0
        assert is_distributed(cur0) is False
        assert "SELECT count(*) FROM pgxc_node" in cur.executed[0][0]


class TestUpsertViaMergeFakeCursor:
    def test_merge_statement_shape(self) -> None:
        cur = _RecordingCursor()
        upsert_via_merge(
            cur,
            "probe_t",
            ["id"],
            [{"id": "r0", "content": "c", "embedding": [0.1, 0.2]}],
            "embedding",
        )
        assert len(cur.executed) == 1
        sql, params = cur.executed[0]
        assert "MERGE INTO probe_t" in sql
        assert "jsonb_array_elements" in sql
        assert "%s::jsonb" in sql
        assert "(e->>'embedding')::floatvector" in sql
        assert "t.id = s.id" in sql
        # 向量在 payload 里编码为 '[...]' 文本，随 jsonb 参数传递
        assert "embedding" in params[0]

    def test_empty_payload_executes_nothing(self) -> None:
        cur = _RecordingCursor()
        upsert_via_merge(cur, "probe_t", ["id"], [], "embedding")
        assert cur.executed == []

    def test_cast_columns_emit_server_cast(self) -> None:
        cur = _RecordingCursor()
        upsert_via_merge(
            cur,
            "probe_t",
            ["id"],
            [{"id": "r0", "content": "c", "metadata": '{"k": "v"}'}],
            "embedding",
            cast_columns={"metadata": "jsonb"},
        )
        sql, _ = cur.executed[0]
        # JSONB 列无隐式 text→jsonb 赋值 cast（实测 507 O-mode），必须显式
        assert "(e->>'metadata')::jsonb AS metadata" in sql
        assert "(e->>'content')::jsonb" not in sql

    def test_rejects_cast_column_not_in_payload(self) -> None:
        cur = _RecordingCursor()
        with pytest.raises(ValueError, match="cast_columns"):
            upsert_via_merge(
                cur,
                "probe_t",
                ["id"],
                [{"id": "r0", "content": "c"}],
                "embedding",
                cast_columns={"missing": "jsonb"},
            )
        assert cur.executed == []

    def test_rejects_key_not_in_payload(self) -> None:
        cur = _RecordingCursor()
        with pytest.raises(ValueError, match="key_columns"):
            upsert_via_merge(
                cur, "probe_t", ["missing"], [{"id": "r0", "content": "c"}], "embedding"
            )
        assert cur.executed == []

    def test_rejects_all_key_payload(self) -> None:
        cur = _RecordingCursor()
        with pytest.raises(ValueError, match="non-key"):
            upsert_via_merge(cur, "probe_t", ["id"], [{"id": "r0"}], "embedding")
        assert cur.executed == []


import os

requires_gaussdb = pytest.mark.skipif(
    os.environ.get("GAUSSDB_TEST", "").lower() != "1",
    reason="requires GAUSSDB_TEST=1 and a reachable GaussDB instance",
)


@requires_gaussdb
class TestVectorHelpersIntegration:
    def test_ensure_vector_index_and_merge_upsert(self) -> None:
        import random

        from crewai.gaussdb.config import GaussDBConfig
        from crewai.gaussdb.connection import cursor, reset_pool
        from crewai.gaussdb.vector import ensure_vector_index, upsert_via_merge

        random.seed(7)
        cfg = GaussDBConfig.from_env()
        try:
            with cursor(cfg) as cur:
                cur.execute("DROP TABLE IF EXISTS probe_gv")
                cur.execute(
                    "CREATE TABLE probe_gv (id VARCHAR(64) PRIMARY KEY, "
                    "content TEXT, embedding floatvector(8) NOT NULL)"
                )
                vec = lambda: [random.uniform(-1, 1) for _ in range(8)]  # noqa: E731
                upsert_via_merge(
                    cur,
                    "probe_gv",
                    ["id"],
                    [
                        {"id": f"r{i}", "content": f"c{i}", "embedding": vec()}
                        for i in range(60)
                    ],
                    "embedding",
                )
                ensure_vector_index(cur, "probe_gv_idx", "probe_gv", "embedding", 8)
                qv = vec()
                cur.execute(
                    "SELECT id FROM probe_gv ORDER BY embedding <+> %s LIMIT 3",
                    ("[" + ",".join(str(x) for x in qv) + "]",),
                )
                assert len(cur.fetchall()) == 3
                # MERGE upsert 覆盖语义
                upsert_via_merge(
                    cur, "probe_gv", ["id"],
                    [{"id": "r0", "content": "updated", "embedding": vec()}],
                    "embedding",
                )
                cur.execute("SELECT content FROM probe_gv WHERE id = 'r0'")
                assert cur.fetchone()[0] == "updated"
                # 极小值/科学计数法向量字面量（repr 产生 1e-05 形式）
                tiny = [1e-05, -2.5e-08, 0.0, 1.0, 3.3e-07, -0.5, 2e-09, 7.5]
                upsert_via_merge(
                    cur, "probe_gv", ["id"],
                    [{"id": "r_tiny", "content": "t", "embedding": tiny}],
                    "embedding",
                )
                cur.execute(
                    "SELECT embedding <+> %s FROM probe_gv WHERE id = 'r_tiny'",
                    ("[" + ",".join(str(x) for x in tiny) + "]",),
                )
                assert abs(cur.fetchone()[0]) < 1e-09  # 自身距离≈0 → 字面量往返无损
                cur.execute("DROP TABLE IF EXISTS probe_gv")
        finally:
            reset_pool()

    def test_large_dim_gate_raises_on_distributed(self) -> None:
        """分布式实例上 >1024 建表被 DB 拒绝（集中式实例上 skip）。"""
        from crewai.gaussdb.config import GaussDBConfig
        from crewai.gaussdb.connection import cursor, reset_pool
        from crewai.gaussdb.vector import is_distributed

        cfg = GaussDBConfig.from_env()
        try:
            with cursor(cfg) as cur:
                if not is_distributed(cur):
                    pytest.skip("not a distributed instance")
                with pytest.raises(Exception, match="1024"):
                    cur.execute(
                        "CREATE TABLE probe_gv_big (id INT, v floatvector(1536))"
                    )
        finally:
            reset_pool()
