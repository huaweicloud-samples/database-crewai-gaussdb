"""Shared vector-SQL helpers for crewAI GaussDB backends.

All dialect facts here are verified against GaussDB 507 (see
survey/crewai-gaussdb-调研报告.md): floatvector columns are NOT NULL and
capped at 4096 dims (centralized) / 1024 dims (distributed, a CREATE TABLE
limit independent of the index); only ``ORDER BY col <+> %s LIMIT k`` uses
the vector index; GsDiskANN+PQ is required above 1024 dims (centralized
only); pq_nseg must divide the dimension.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    # N812: aliasing the lowercase builtin-ish name `cursor` to PascalCase.
    from psycopg2.extensions import cursor as Psycopg2Cursor  # noqa: N812

_CREATE_IVFFLAT = (
    "CREATE INDEX IF NOT EXISTS {name} ON {table} "
    "USING GSIVFFLAT({column} cosine) WITH (IVF_NLIST = {nlist})"
)
_CREATE_DISKANN = (
    "CREATE INDEX IF NOT EXISTS {name} ON {table} "
    "USING GSDISKANN({column} cosine) WITH ("
    "pq_nseg={pq_nseg}, pq_nclus=16, enable_pq=true, "
    "subgraph_count=1, enable_vector_copy=false)"
)


def calc_pq_nseg(dim: int) -> int:
    """pq_nseg must divide the dimension; values verified on GaussDB 507."""
    if dim <= 512:
        return dim
    if dim <= 1024:
        return dim // 2
    for candidate in (96, 128, 192, 256, 384, 512):
        if dim % candidate == 0:
            return candidate
    return dim


def is_distributed(cursor: Psycopg2Cursor) -> bool:
    """Detect deployment topology (pgxc_node only exists on distributed)."""
    cursor.execute("SELECT count(*) FROM pgxc_node")
    row = cursor.fetchone()
    return bool(row and row[0])


def validate_dimension(dim: int, distributed: bool) -> None:
    """Raise with actionable guidance when *dim* exceeds the topology limit."""
    if distributed:
        if dim > 1024:
            raise ValueError(
                f"Embedding dimension {dim} exceeds the distributed GaussDB "
                f"CREATE TABLE limit of 1024. Use an embedding model with "
                f"<= 1024 dimensions, or deploy on a centralized instance."
            )
    elif dim > 4096:
        raise ValueError(
            f"Embedding dimension {dim} exceeds the GaussDB floatvector "
            f"limit of 4096. Use a smaller embedding model."
        )


def vector_index_ddl(
    index_name: str,
    table_name: str,
    column_name: str,
    dim: int,
    distributed: bool,
    ivf_nlist: int = 256,
) -> str:
    """Return the CREATE INDEX statement for *dim* on the given topology."""
    validate_dimension(dim, distributed)
    if dim <= 1024:
        return _CREATE_IVFFLAT.format(
            name=index_name, table=table_name, column=column_name, nlist=ivf_nlist
        )
    return _CREATE_DISKANN.format(
        name=index_name,
        table=table_name,
        column=column_name,
        pq_nseg=calc_pq_nseg(dim),
    )


def ensure_vector_index(
    cursor: Psycopg2Cursor,
    index_name: str,
    table_name: str,
    column_name: str,
    dim: int,
    ivf_nlist: int = 256,
) -> None:
    """Create the vector index if missing, tuning session GUCs for searches.

    Must run inside a transaction on one pooled connection: DiskANN needs
    maintenance_work_mem >= 512MB (quoted value), and the probe GUCs are set
    for subsequent searches on the same session.
    """
    distributed = is_distributed(cursor)
    cursor.execute("SET maintenance_work_mem = '512MB'")
    cursor.execute(
        vector_index_ddl(
            index_name, table_name, column_name, dim, distributed, ivf_nlist
        )
    )
    if dim <= 1024:
        cursor.execute("SET gsivfflat_probes = 25")
    else:
        cursor.execute("SET diskann_probe_ncandidates = 200")


def _vector_literal(values: Any) -> str:
    """Encode a float sequence as a floatvector text literal ('[x,y,...]').

    repr(float) may produce scientific notation (1e-05); the integration
    test below anchors that floatvector parses it back losslessly.
    """
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def upsert_via_merge(
    cursor: Psycopg2Cursor,
    table_name: str,
    key_columns: list[str],
    payload: list[dict[str, Any]],
    vector_column: str,
) -> None:
    """Batch upsert via a single MERGE + jsonb_array_elements payload.

    One statement per batch (distributed GaussDB cannot handle pipelined
    batches). Vectors travel as '[x,y,...]' text inside the jsonb payload
    and are cast server-side via (e->>'col')::floatvector.

    Args:
        key_columns: Merge keys (must cover the distribution key on
            distributed deployments).
        payload: Rows as dicts. The vector column holds a float sequence;
            other values must be JSON-serializable scalars/strings (callers
            pre-encode non-scalar values like JSON text).
        vector_column: The floatvector column name.
    """
    if not payload:
        return
    all_columns = list(payload[0].keys())
    using = ", ".join(
        f"(e->>'{col}')::floatvector AS {col}"
        if col == vector_column
        else f"e->>'{col}' AS {col}"
        for col in all_columns
    )
    update_set = ", ".join(
        f"{col} = s.{col}" for col in all_columns if col not in key_columns
    )
    insert_cols = ", ".join(all_columns)
    insert_vals = ", ".join(f"s.{col}" for col in all_columns)
    on_clause = " AND ".join(f"t.{k} = s.{k}" for k in key_columns)
    encoded = json.dumps(
        [
            {
                col: _vector_literal(row[col]) if col == vector_column else row[col]
                for col in all_columns
            }
            for row in payload
        ],
        ensure_ascii=False,
    )
    cursor.execute(
        f"""
        MERGE INTO {table_name} t
        USING (SELECT {using} FROM jsonb_array_elements(%s::jsonb) e) s
        ON {on_clause}
        WHEN MATCHED THEN UPDATE SET {update_set}
        WHEN NOT MATCHED THEN INSERT ({insert_cols}) VALUES ({insert_vals})
        """,  # nosec # noqa: S608 -- identifiers come from schema names; the payload is bound
        (encoded,),
    )
