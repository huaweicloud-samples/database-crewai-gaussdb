"""GaussDB RAG configuration model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import field
from typing import Literal

from pydantic import Field
from pydantic.dataclasses import dataclass as pyd_dataclass

from crewai.rag.config.base import BaseRagConfig


@pyd_dataclass(frozen=True)
class GaussDBRagConfig(BaseRagConfig):
    """Configuration for the GaussDB RAG client.

    Connection defaults mirror ``crewai.gaussdb.config.GaussDBConfig``;
    ``password`` is excluded from serialization (never enters checkpoint
    payloads or repr — Plan 1 security contract). Pydantic dataclasses have
    no ``model_dump``; serialize via ``TypeAdapter(GaussDBRagConfig)``, which
    honors the exclusion.
    """

    provider: Literal["gaussdb"] = field(default="gaussdb", init=False)
    embedding_function: Callable[[list[str]], list[list[float]]] | None = field(
        default=None
    )
    host: str = "localhost"
    port: int = 5432
    user: str = ""
    password: str = Field(default="", repr=False, exclude=True)
    database: str = "crewai"
    min_connections: int = 1
    max_connections: int = 10
