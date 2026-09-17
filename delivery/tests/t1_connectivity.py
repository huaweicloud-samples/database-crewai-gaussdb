# -*- coding: utf-8 -*-
"""L1 连通性测试：版本/兼容模式/向量开关/拓扑/权限。纯 psycopg2，不依赖 crewai。

用法: python t1_connectivity.py   （凭据从环境变量 / .env 读取）
"""
from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import psycopg2

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def main() -> int:
    host = os.environ.get("GAUSSDB_HOST", "localhost")
    port = int(os.environ.get("GAUSSDB_PORT", "5432"))
    user = os.environ.get("GAUSSDB_USER", "")
    password = os.environ.get("GAUSSDB_PASSWORD", "")
    database = os.environ.get("GAUSSDB_DATABASE", "crewai")

    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=database, user=user, password=password,
            connect_timeout=15,
        )
    except Exception as e:  # noqa: BLE001
        check("连接", False, f"{type(e).__name__}: {e}")
        print("\n=== L1 FAIL ===")
        return 1
    check("连接", True, f"{host}:{port}/{database} as {user}")

    cur = conn.cursor()

    cur.execute("SELECT version()")
    version = cur.fetchone()[0]
    check("版本 ≥ 507", "507" in version, version[:60])

    cur.execute("SELECT datcompatibility FROM pg_database WHERE datname = current_database()")
    compat = cur.fetchone()[0]
    check("Oracle 兼容模式", compat in ("A", "ORA"), f"datcompatibility={compat}")

    cur.execute("SHOW enable_vectordb")
    vectordb = cur.fetchone()[0]
    if vectordb == "on":
        check("enable_vectordb", True, "on")
    else:
        check("enable_vectordb", False,
              f"{vectordb}（向量功能需开启：gs_guc set ... enable_vectordb=on + 重启集群）")

    cur.execute("SELECT count(*) FROM pgxc_node")
    n_nodes = cur.fetchone()[0]
    topology = "分布式" if n_nodes else "集中式"
    print(f"[INFO] 拓扑: {topology}（pgxc_node={n_nodes}）")

    try:
        cur.execute("CREATE TABLE __t1_probe (id INT)")
        cur.execute("DROP TABLE __t1_probe")
        check("建表权限", True)
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        check("建表权限", False, str(e).splitlines()[0][:80])

    conn.close()

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n=== L1 {'PASS' if not failed else 'FAIL'}（{len(RESULTS) - len(failed)}/{len(RESULTS)}）===")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
