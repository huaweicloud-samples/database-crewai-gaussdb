"""
CrewAI Flow Persistence.

This module provides interfaces and implementations for persisting flow states.
"""

from typing import Any

from crewai.flow.persistence.base import FlowPersistence
from crewai.flow.persistence.decorators import persist
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence


__all__ = ["FlowPersistence", "SQLiteFlowPersistence", "persist"]


def __getattr__(name: str) -> Any:
    """Lazily expose the GaussDB backend so importing this package never
    requires psycopg2 to be installed."""
    if name == "GaussDBFlowPersistence":
        from crewai.flow.persistence.gaussdb import GaussDBFlowPersistence

        return GaussDBFlowPersistence
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
