# -*- coding: utf-8 -*-
"""GaussDB 适配分层测试执行器：L1 连通 → L2 SQL 冒烟 → L3 持久化 → L4 向量 → L5 Ollama 端到端。

用法（crewai uv workspace 根目录）:
    uv run python delivery/tests/run_all.py                 # L1~L4
    uv run python delivery/tests/run_all.py --with-ollama   # L1~L5
    uv run python delivery/tests/run_all.py --only t2_sql_smoke

退出码：0 = 所选层全过；1 = 有失败。
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent

LAYERS = [
    ("L1 连通性", "t1_connectivity.py"),
    ("L2 SQL 方言冒烟", "t2_sql_smoke.py"),
    ("L3 持久化端到端", "t3_persistence.py"),
    ("L4 向量端到端", "t4_vector.py"),
    ("L5 Ollama 端到端", "t5_e2e_ollama.py"),
]


def main() -> int:
    args = sys.argv[1:]
    with_ollama = "--with-ollama" in args
    only = None
    if "--only" in args:
        only = args[args.index("--only") + 1]

    selected = []
    for label, script in LAYERS:
        if script == "t5_e2e_ollama.py" and not with_ollama:
            continue
        if only and only not in script:
            continue
        selected.append((label, script))

    if not selected:
        print("no layers selected")
        return 1

    overall_ok = True
    summary: list[tuple[str, bool, float]] = []
    for label, script in selected:
        print(f"\n{'=' * 60}\n>>> {label}（{script}）\n{'=' * 60}")
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(HERE / script)],
            cwd=str(HERE.parent.parent),   # crewAI workspace 根（uv 环境/相对路径一致）
        )
        elapsed = time.time() - t0
        ok = proc.returncode == 0
        summary.append((label, ok, elapsed))
        if not ok:
            overall_ok = False

    print(f"\n{'=' * 60}\n=== SUMMARY ===")
    for label, ok, elapsed in summary:
        print(f"{'PASS' if ok else 'FAIL'}  {label}  ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
