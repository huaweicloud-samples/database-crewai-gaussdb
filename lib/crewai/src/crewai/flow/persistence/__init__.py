"""
CrewAI Flow Persistence.

This module provides interfaces and implementations for persisting flow states.
"""

from typing import TYPE_CHECKING, Any

from crewai.flow.persistence.base import FlowPersistence
from crewai.flow.persistence.decorators import persist
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence


if TYPE_CHECKING:
    from crewai.flow.persistence.gaussdb import (
        GaussDBFlowPersistence as GaussDBFlowPersistence,
    )


__all__ = ["FlowPersistence", "SQLiteFlowPersistence", "persist"]
# GaussDBFlowPersistence intentionally absent from __all__: import * must not require psycopg2.


def __getattr__(name: str) -> Any:
    """Lazily expose the GaussDB backend so importing this package never
    requires psycopg2 to be installed."""
    if name == "GaussDBFlowPersistence":
        from crewai.flow.persistence.gaussdb import GaussDBFlowPersistence

        return GaussDBFlowPersistence
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
