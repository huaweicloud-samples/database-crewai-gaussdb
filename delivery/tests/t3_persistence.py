# -*- coding: utf-8 -*-
"""L3 持久化端到端：Flow 状态 / Kickoff 输出 / Checkpoint / 接线。依赖 crewai[gaussdb]。

用法（crewai uv workspace 根目录）: uv run python delivery/tests/t3_persistence.py
"""
from __future__ import annotations

import asyncio
import os
import sys
from unittest.mock import MagicMock

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("GAUSSDB_TEST", "1")
os.environ.setdefault("CREWAI_STORAGE_BACKEND", "gaussdb")

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def cleanup(cur) -> None:
    for stmt in (
        "DROP TABLE IF EXISTS pending_feedback",
        "DROP TABLE IF EXISTS flow_states",
        "DROP SEQUENCE IF EXISTS flow_states_id_seq",
        "DROP TABLE IF EXISTS latest_kickoff_task_outputs",
        "DROP TABLE IF EXISTS checkpoints",
        "DROP SEQUENCE IF EXISTS checkpoints_seq",
    ):
        cur.execute(stmt)


def main() -> int:
    from crewai.gaussdb.config import GaussDBConfig
    from crewai.gaussdb.connection import cursor, reset_pool

    cfg = GaussDBConfig.from_env()
    reset_pool()
    try:
        with cursor(cfg) as cur:
            cleanup(cur)

        # --- Flow 持久化 ---
        from crewai.flow.persistence.gaussdb import GaussDBFlowPersistence

        fp = GaussDBFlowPersistence()
        fp.save_state("e2e-flow-1", "step1", {"n": 1})
        fp.save_state("e2e-flow-1", "step2", {"n": 2})
        check("Flow save/load（最新覆盖）", fp.load_state("e2e-flow-1") == {"n": 2})
        check("Flow load 不存在", fp.load_state("e2e-nope") is None)

        from crewai.flow.async_feedback.types import PendingFeedbackContext

        ctx = PendingFeedbackContext(
            flow_id="e2e-flow-1", flow_class="tests.T", method_name="ask",
            method_output={"t": 1}, message="please review",
        )
        fp.save_pending_feedback("e2e-flow-1", ctx, {"a": 1})
        fp.save_pending_feedback("e2e-flow-1", ctx, {"a": 2})  # MERGE 覆盖
        loaded = fp.load_pending_feedback("e2e-flow-1")
        check("Flow pending feedback（MERGE 覆盖+读回）",
              loaded is not None and loaded[0] == {"a": 2} and loaded[1].method_name == "ask")
        fp.clear_pending_feedback("e2e-flow-1")
        check("Flow clear pending", fp.load_pending_feedback("e2e-flow-1") is None)

        # --- Kickoff 输出 ---
        from crewai.memory.storage.kickoff_task_outputs_gaussdb import (
            GaussDBKickoffTaskOutputsStorage,
        )

        ko = GaussDBKickoffTaskOutputsStorage()
        task = MagicMock()
        task.id = "e2e-t-1"
        task.expected_output = "expected"
        ko.add(task, {"raw": "first"}, task_index=0, inputs={"q": "hi"})
        ko.add(task, {"raw": "second"}, task_index=0, inputs={"q": "hi"})  # 同 task_id 覆盖
        ko.update(0, output={"raw": "updated"}, was_replayed=True)
        rows = ko.load()
        check("Kickoff add(MERGE 覆盖)/update/load",
              len(rows) == 1 and rows[0]["output"] == {"raw": "updated"}
              and rows[0]["was_replayed"] is True and rows[0]["inputs"] == {"q": "hi"})
        ko.delete_all()
        check("Kickoff delete_all", ko.load() == [])

        # --- Checkpoint ---
        from crewai.state.provider.gaussdb_provider import GaussDBProvider

        prov = GaussDBProvider()
        loc1 = prov.checkpoint('{"n": 1}', "gaussdb")
        loc2 = prov.checkpoint('{"n": 2}', "gaussdb")
        loc3 = prov.checkpoint('{"n": 3}', "gaussdb")
        check("Checkpoint roundtrip", prov.from_checkpoint(loc1) == '{"n": 1}')
        check("Checkpoint prune（seq 保序剪最旧）",
              prov.prune("gaussdb", 2, branch="main") == 1)
        try:
            prov.from_checkpoint(loc1)
            check("Checkpoint 剪掉最旧", False, "loc1 still readable")
        except ValueError:
            check("Checkpoint 剪掉最旧", True)
        check("Checkpoint async", asyncio.run(
            prov.afrom_checkpoint(asyncio.run(prov.acheckpoint('{"a": true}', "gaussdb")))
        ) == '{"a": true}')

        # --- 接线（env 驱动切换）---
        from crewai.flow.persistence.factory import default_flow_persistence
        from crewai.state.provider.utils import detect_provider
        from crewai.utilities.task_output_storage_handler import TaskOutputStorageHandler

        check("接线：flow 工厂", type(default_flow_persistence()).__name__ == "GaussDBFlowPersistence")
        check("接线：kickoff handler",
              "GaussDB" in type(TaskOutputStorageHandler().storage).__name__)
        check("接线：detect_provider",
              type(detect_provider("gaussdb#abc_123")).__name__ == "GaussDBProvider")

        with cursor(cfg) as cur:
            cleanup(cur)
        reset_pool()
    except Exception as e:  # noqa: BLE001
        check("L3 异常", False, f"{type(e).__name__}: {e}")
        reset_pool()

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n=== L3 {'PASS' if not failed else 'FAIL'}（{len(RESULTS) - len(failed)}/{len(RESULTS)}）===")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
