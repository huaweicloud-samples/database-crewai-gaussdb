"""GaussDB connection configuration."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field


class GaussDBConfig(BaseModel):
    """Connection settings for one GaussDB database.

    Mirrors the factory/env conventions of the built-in backends: connection
    details come from ``GAUSSDB_*`` environment variables (dify/LightRAG/n8n
    adapters use the same names, easing ops runbooks).
    """

    host: str = "localhost"
    port: int = 5432
    user: str = ""
    # Field(repr=False, exclude=True): the password must never appear in
    # repr(), model_dump(), or checkpoint payloads (entities serialize their
    # whole CheckpointConfig, and this provider is the first one that carries
    # a secret). SecretStr alone is NOT enough — pydantic v2 model_dump
    # (mode="json") would still emit the plaintext.
    password: str = Field(default="", repr=False, exclude=True)
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
