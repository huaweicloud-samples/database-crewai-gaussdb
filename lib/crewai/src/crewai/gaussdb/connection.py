"""Thread-safe psycopg2 connection pool for crewAI GaussDB backends.

GaussDB has no official Python pool component; the libpq docs recommend
per-thread connections or an external pool. ``ThreadedConnectionPool`` is the
standard psycopg2 answer and was validated against GaussDB 507 (dify survey
experiment G11 used ``SimpleConnectionPool`` successfully; the threaded
variant only adds locking).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import threading
from typing import TYPE_CHECKING, Any

import psycopg2.extensions
from psycopg2.pool import ThreadedConnectionPool


if TYPE_CHECKING:
    from crewai.gaussdb.config import GaussDBConfig

_pool_lock = threading.Lock()
_pool: ThreadedConnectionPool | None = None
_pool_key: dict[str, Any] | None = None


def _create_pool_instance(
    config: GaussDBConfig, minconn: int, maxconn: int
) -> ThreadedConnectionPool:
    """Indirection for testability (tests monkeypatch this)."""
    return ThreadedConnectionPool(
        minconn,
        maxconn,
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
    )


def get_pool(config: GaussDBConfig) -> ThreadedConnectionPool:
    """Return the process-wide pool for *config*, rebuilding on change.

    All callers must pass an equivalent config for the process lifetime:
    a rebuild closes every connection of the old pool, including ones
    currently checked out.
    """
    global _pool, _pool_key
    # dict(config), not model_dump(): model_dump() honors field exclusion, so
    # after password became exclude=True two configs differing only by
    # password would collide on the same pool. dict(config) iterates the raw
    # (name, value) pairs and keeps the password in the key.
    key = dict(config)
    with _pool_lock:
        if _pool is not None and _pool_key == key:
            return _pool
        if _pool is not None:
            _pool.closeall()
        _pool = _create_pool_instance(
            config, config.min_connections, config.max_connections
        )
        _pool_key = key
        return _pool


def reset_pool() -> None:
    """Close and forget the current pool (used between tests / on teardown)."""
    global _pool, _pool_key
    with _pool_lock:
        if _pool is not None:
            _pool.closeall()
        _pool = None
        _pool_key = None


@contextmanager
def connection(config: GaussDBConfig) -> Iterator[psycopg2.extensions.connection]:
    """Yield a pooled connection with commit-on-success / rollback-on-error."""
    pool = get_pool(config)
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


@contextmanager
def cursor(config: GaussDBConfig) -> Iterator[psycopg2.extensions.cursor]:
    """Yield a cursor on a pooled connection (transaction per with-block)."""
    with connection(config) as conn:
        cur = conn.cursor()
        try:
            yield cur
        finally:
            cur.close()
