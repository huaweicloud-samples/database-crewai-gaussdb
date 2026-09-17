# -*- coding: utf-8 -*-
"""L2 SQL 方言契约冒烟：验证目标实例与适配基线的 O 模式方言一致。纯 psycopg2。

每条断言对应适配层依赖的一个方言事实；任何 FAIL 说明实例行为与预期不符
（参考 delivery/注意事项.md §2）。

用法: python t2_sql_smoke.py
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


def expect_error(cur, sql: str, keyword: str) -> bool:
    try:
        cur.execute(sql)
    except Exception as e:  # noqa: BLE001
        return keyword.lower() in str(e).lower()
    return False


def main() -> int:
    conn = psycopg2.connect(
        host=os.environ.get("GAUSSDB_HOST", "localhost"),
        port=int(os.environ.get("GAUSSDB_PORT", "5432")),
        dbname=os.environ.get("GAUSSDB_DATABASE", "crewai"),
        user=os.environ.get("GAUSSDB_USER", ""),
        password=os.environ.get("GAUSSDB_PASSWORD", ""),
        connect_timeout=15,
    )
    conn.autocommit = True
    cur = conn.cursor()

    # --- MERGE INTO（单行 + jsonb 批量）---
    cur.execute("CREATE TABLE __t2_m (id INT PRIMARY KEY, v TEXT)")
    cur.execute("INSERT INTO __t2_m VALUES (1, 'a')")
    cur.execute(
        "MERGE INTO __t2_m t USING (SELECT 1 AS id, 'b' AS v) s ON t.id = s.id "
        "WHEN MATCHED THEN UPDATE SET v = s.v "
        "WHEN NOT MATCHED THEN INSERT (id, v) VALUES (s.id, s.v)"
    )
    cur.execute("SELECT v FROM __t2_m WHERE id = 1")
    check("MERGE INTO 单行 upsert", cur.fetchone()[0] == "b")

    cur.execute(
        "MERGE INTO __t2_m t USING ("
        "  SELECT (e->>'id')::int AS id, e->>'v' AS v"
        "  FROM jsonb_array_elements(%s::jsonb) e) s ON t.id = s.id "
        "WHEN MATCHED THEN UPDATE SET v = s.v "
        "WHEN NOT MATCHED THEN INSERT (id, v) VALUES (s.id, s.v)",
        ('[{"id":1,"v":"c"},{"id":2,"v":"d"}]',),
    )
    cur.execute("SELECT count(*) FROM __t2_m")
    check("MERGE INTO jsonb 批量 upsert", cur.fetchone()[0] == 2)

    check("ON CONFLICT 不可用（预期报错）", expect_error(
        cur, "INSERT INTO __t2_m VALUES (3, 'x') ON CONFLICT (id) DO UPDATE SET v = 'x'", "syntax error"))

    # --- SEQUENCE + RETURNING ---
    cur.execute("CREATE SEQUENCE IF NOT EXISTS __t2_seq START 1")
    cur.execute("CREATE TABLE __t2_s (id BIGINT PRIMARY KEY DEFAULT nextval('__t2_seq'), v TEXT)")
    cur.execute("INSERT INTO __t2_s (v) VALUES ('x') RETURNING id")
    rid = cur.fetchone()[0]
    check("SEQUENCE 自增 + RETURNING", rid >= 1, f"id={rid}")

    # --- JSONB 过滤 ---
    cur.execute("CREATE TABLE __t2_j (id INT, meta JSONB)")
    cur.execute("INSERT INTO __t2_j VALUES (1, '{\"kind\": \"db\"}')")
    cur.execute("SELECT id FROM __t2_j WHERE meta->>'kind' = 'db'")
    check("JSONB ->> 过滤", cur.fetchone() is not None)
    cur.execute("SELECT id FROM __t2_j WHERE meta @> '{\"kind\": \"db\"}'::jsonb")
    check("JSONB @> 包含", cur.fetchone() is not None)

    # --- 分页 / ILIKE / BOOLEAN ---
    cur.execute("SELECT 1 LIMIT 1 OFFSET 0")
    check("LIMIT/OFFSET", cur.fetchone()[0] == 1)
    cur.execute("SELECT 'ABC' ILIKE 'abc'")
    check("ILIKE", cur.fetchone()[0])
    cur.execute("CREATE TABLE __t2_b (id INT, flag BOOLEAN)")
    cur.execute("INSERT INTO __t2_b VALUES (1, %s)", (True,))
    cur.execute("SELECT flag FROM __t2_b WHERE id = 1")
    check("BOOLEAN 直接参数", cur.fetchone()[0] is True)

    # --- 向量（floatvector / <+> / 索引）---
    try:
        cur.execute("CREATE TABLE __t2_v (id INT PRIMARY KEY, v floatvector(4) NOT NULL)")
        cur.execute("INSERT INTO __t2_v VALUES (1, %s), (2, %s)", ("[1,0,0,0]", "[0,1,0,0]"))
        cur.execute("SELECT v <+> %s FROM __t2_v WHERE id = 1", ("[1,0,0,0]",))
        d_same = float(cur.fetchone()[0])
        check("floatvector cast + <+> 余弦距离", d_same == 0.0, f"self-distance={d_same}")
        # 科学计数法字面量往返
        cur.execute("INSERT INTO __t2_v VALUES (3, %s)", ("[1e-05,-2.5e-08,0.0,1.0]",))
        cur.execute("SELECT v <+> %s FROM __t2_v WHERE id = 3", ("[1e-05,-2.5e-08,0.0,1.0]",))
        check("科学计数法字面量往返", abs(float(cur.fetchone()[0])) < 1e-9)
        cur.execute("SET maintenance_work_mem = '512MB'")
        cur.execute("CREATE INDEX __t2_v_idx ON __t2_v USING GSIVFFLAT(v cosine) WITH (IVF_NLIST = 2)")
        check("GsIVFFLAT 索引", True)
        cur.execute("SET gsivfflat_probes = 2")
        cur.execute("SELECT id FROM __t2_v ORDER BY v <+> %s LIMIT 2", ("[1,0,0,0]",))
        check("向量检索走索引形态", len(cur.fetchall()) == 2)
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        check("向量能力（enable_vectordb）", False, str(e).splitlines()[0][:100])

    # --- 清理 ---
    for ddl in (
        "DROP TABLE IF EXISTS __t2_v", "DROP TABLE IF EXISTS __t2_b", "DROP TABLE IF EXISTS __t2_j",
        "DROP TABLE IF EXISTS __t2_s", "DROP SEQUENCE IF EXISTS __t2_seq", "DROP TABLE IF EXISTS __t2_m",
    ):
        cur.execute(ddl)
    conn.close()

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n=== L2 {'PASS' if not failed else 'FAIL'}（{len(RESULTS) - len(failed)}/{len(RESULTS)}）===")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
