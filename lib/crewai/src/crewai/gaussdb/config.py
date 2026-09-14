"""GaussDB connection configuration."""

from __future__ import annotations

import os

from pydantic import BaseModel


class GaussDBConfig(BaseModel):
    """Connection settings for one GaussDB database.

    Mirrors the factory/env conventions of the built-in backends: connection
    details come from ``GAUSSDB_*`` environment variables (dify/LightRAG/n8n
    adapters use the same names, easing ops runbooks).
    """

    host: str = "localhost"
    port: int = 5432
    user: str = ""
    password: str = ""
    database: str = "crewai"
    min_connections: int = 1
    max_connections: int = 10

    @classmethod
    def from_env(cls) -> GaussDBConfig:
        return cls(
            host=os.environ.get("GAUSSDB_HOST", "localhost"),
            port=int(os.environ.get("GAUSSDB_PORT", "5432")),
            user=os.environ.get("GAUSSDB_USER", ""),
            password=os.environ.get("GAUSSDB_PASSWORD", ""),
            database=os.environ.get("GAUSSDB_DATABASE", "crewai"),
            min_connections=int(os.environ.get("GAUSSDB_MIN_CONNECTIONS", "1")),
            max_connections=int(os.environ.get("GAUSSDB_MAX_CONNECTIONS", "10")),
        )


def is_gaussdb_backend() -> bool:
    """True when ``CREWAI_STORAGE_BACKEND=gaussdb``.

    The single opt-in switch for swapping built-in SQLite defaults to the
    GaussDB implementations (flow persistence, kickoff outputs, checkpoint
    provider, and the CLI readers).
    """
    return os.environ.get("CREWAI_STORAGE_BACKEND", "").strip().lower() == "gaussdb"
