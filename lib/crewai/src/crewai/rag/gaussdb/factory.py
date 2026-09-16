"""Factory for the GaussDB RAG client."""

from __future__ import annotations

from typing import TYPE_CHECKING

from crewai.rag.gaussdb.config import GaussDBRagConfig


if TYPE_CHECKING:
    from crewai.rag.core.base_client import BaseClient


def create_client(config: GaussDBRagConfig) -> BaseClient:
    """Create a GaussDB RAG client from configuration.

    The client module is imported lazily so that ``psycopg2`` stays an
    optional dependency (the factory module itself is import-safe).

    Args:
        config: The GaussDB RAG configuration.

    Returns:
        A configured GaussDBClient instance.
    """

    from crewai.rag.gaussdb.client import GaussDBClient

    return GaussDBClient(config)
