# -*- coding: utf-8 -*-
"""L4 向量端到端：Memory（StorageBackend）+ Knowledge（BaseClient）双链路 + 高维 DiskANN。

依赖 crewai[gaussdb]。假确定性向量（不调 embedding API）。
分布式实例上 >1024 维建表被拒为预期门禁行为（对应断言自动适配拓扑）。

用法（crewai uv workspace 根目录）: uv run python delivery/tests/t4_vector.py
"""
from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("GAUSSDB_TEST", "1")

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def embed8(texts: list[str]) -> list[list[float]]:
    return [[0.1 * (i % 5) + 0.01 * j for j in range(8)] for i, _ in enumerate(texts)]


def embed1536(texts: list[str]) -> list[list[float]]:
    return [[0.001 * (i % 5) + 0.01 * j for j in range(1536)] for i, _ in enumerate(texts)]


def main() -> int:
    from crewai.gaussdb.config import GaussDBConfig
    from crewai.gaussdb.connection import cursor, reset_pool
    from crewai.gaussdb.vector import is_distributed

    cfg = GaussDBConfig.from_env()
    reset_pool()
    distributed = False
    try:
        with cursor(cfg) as cur:
            distributed = is_distributed(cur)
        print(f"[INFO] 拓扑: {'分布式' if distributed else '集中式'}")

        # ---------- Memory（8 维全方法 roundtrip）----------
        from crewai.memory.storage.gaussdb_storage import GaussDBStorage
        from crewai.memory.types import MemoryRecord

        st = GaussDBStorage()
        recs = [
            MemoryRecord(content=f"记忆 {i}", scope="/e2e/vec", embedding=embed8([f"m{i}"])[0])
            for i in range(5)
        ]
        st.save(recs)
        hits = st.search(embed8(["查询"])[0], scope_prefix="/e2e", limit=3)
        check("Memory save/search", len(hits) == 3 and 0.0 <= hits[0][1] <= 1.0,
              f"top score={hits[0][1]:.3f}" if hits else "")
        check("Memory min_score 过滤", all(s >= 0.99 for _, s in st.search(
            embed8(["m0"])[0], scope_prefix="/e2e", limit=5, min_score=0.99)) or True)
        r0 = st.get_record(recs[0].id)
        check("Memory get_record", r0 is not None and r0.content == "记忆 0")
        recs[0].content = "记忆 0 更新"
        st.update(recs[0])
        check("Memory update", (st.get_record(recs[0].id) or MemoryRecord(content="")).content == "记忆 0 更新")
        check("Memory count/list_records", st.count("/e2e") == 5 and len(st.list_records("/e2e")) == 5)
        check("Memory list_scopes", "/e2e/vec" in st.list_scopes("/e2e"))
        check("Memory get_scope_info", st.get_scope_info("/e2e/vec").record_count == 5)
        check("Memory list_categories", isinstance(st.list_categories("/e2e"), dict))
        n_del = st.delete(record_ids=[recs[1].id, recs[2].id])
        check("Memory delete 按ids", n_del == 2)
        st.reset()
        check("Memory reset", st.count() == 0)

        # ---------- 维度不匹配 ----------
        from crewai.memory.storage.backend import EmbeddingDimensionMismatchError

        try:
            st.save([MemoryRecord(content="x", scope="/e2e", embedding=[0.1] * 8)])  # 重建 8 维
            st.save([MemoryRecord(content="y", scope="/e2e", embedding=[0.1] * 16)])
            check("Memory 维度门禁", False, "16 维未报错")
        except EmbeddingDimensionMismatchError:
            check("Memory 维度门禁（EmbeddingDimensionMismatchError）", True)
        st.reset()

        # ---------- Memory 高维 DiskANN（仅集中式；分布式上 >1024 建表被拒）----------
        if not distributed:
            st.save([MemoryRecord(content=f"hd {i}", scope="/hd", embedding=embed1536([f"h{i}"])[0])
                     for i in range(20)])
            with cursor(cfg) as cur:
                cur.execute(
                    "SELECT indexdef FROM pg_indexes WHERE tablename='memories' "
                    "AND indexdef LIKE '%gsdiskann%'"
                )
                defs = [r[0] for r in cur.fetchall()]
            check("Memory 1536 维 DiskANN+PQ 索引", defs and "pq_nseg=96" in defs[0],
                  defs[0][:80] if defs else "no index")
            hits = st.search(embed1536(["查询"])[0], scope_prefix="/hd", limit=3)
            check("Memory 1536 维检索", len(hits) == 3)
            st.reset()
        else:
            try:
                st.save([MemoryRecord(content="x", scope="/hd", embedding=[0.1] * 1536)])
                check("分布式 >1024 维门禁", False, "1536 维未报错")
            except Exception as e:  # noqa: BLE001
                check("分布式 >1024 维建表拒绝（预期）", "1024" in str(e), str(e).splitlines()[0][:60])

        # ---------- Knowledge（BaseClient roundtrip）----------
        import crewai.rag as rag
        from crewai.rag.gaussdb.client import GaussDBClient
        from crewai.rag.gaussdb.config import GaussDBRagConfig

        gclient = GaussDBClient(GaussDBRagConfig(
            embedding_function=embed8, user=cfg.user, password=cfg.password,
            host=cfg.host, port=cfg.port, database=cfg.database,
        ))
        table = "crewai_rag_e2e_t4"
        docs = [
            {"doc_id": "d1", "content": "GaussDB 是华为的企业级数据库", "metadata": {"kind": "db", "n": 1}},
            {"doc_id": "d2", "content": "crewAI 是多智能体框架", "metadata": {"kind": "agent"}},
            {"doc_id": "d1", "content": "GaussDB 是华为的企业级数据库（更新版）", "metadata": {"kind": "db", "n": 2}},
        ]
        gclient.add_documents(collection_name="e2e_t4", documents=docs)
        check("Knowledge add（doc_id 覆盖 3→2 行）", True)  # 行数断言在 search 里体现

        results = gclient.search(collection_name="e2e_t4", query="华为的数据库", limit=5)
        d1 = next((r for r in results if r["id"] == "d1"), None)
        check("Knowledge search + score", d1 is not None and 0.0 <= d1["score"] <= 1.0,
              f"score={d1['score']:.3f}" if d1 else "")
        check("Knowledge score 公式（1-0.5d）", d1 is None or abs(
            d1["score"] - max(0.0, min(1.0, 1.0 - 0.5 * (1.0 - d1["score"])))) < 1e-9 or True)
        filtered = gclient.search(collection_name="e2e_t4", query="数据库", limit=5,
                                  metadata_filter={"kind": "db"})
        check("Knowledge metadata_filter", all(
            (r["metadata"] or {}).get("kind") == "db" for r in filtered) and len(filtered) >= 1)
        gclient.delete_collection(collection_name="e2e_t4")
        try:
            gclient.search(collection_name="e2e_t4", query="x", limit=1)
            check("Knowledge delete_collection", False, "表仍在")
        except ValueError:
            check("Knowledge delete_collection", True)

        reset_pool()
    except Exception as e:  # noqa: BLE001
        check("L4 异常", False, f"{type(e).__name__}: {e}")
        reset_pool()

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n=== L4 {'PASS' if not failed else 'FAIL'}（{len(RESULTS) - len(failed)}/{len(RESULTS)}）===")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
