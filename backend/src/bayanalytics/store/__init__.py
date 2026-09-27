"""Persistence: ``AnalysisStore`` protocol plus the in-memory and Postgres implementations.

``build_store(settings)`` picks ``PostgresStore`` when ``settings.database_url`` is set and
``InMemoryStore`` otherwise. The orchestrator calls ``await store.start()`` in the FastAPI
lifespan, then ``await store.mark_interrupted()`` before accepting jobs (AGENT.md section 38),
and ``await store.close()`` at shutdown.
"""

from __future__ import annotations

from bayanalytics.config import Settings
from bayanalytics.store.base import AnalysisStore
from bayanalytics.store.memory import InMemoryStore, interrupted_payload
from bayanalytics.store.postgres import (
    SCHEMA_VERSION,
    PostgresStore,
    SchemaVersionError,
    build_ssl,
    parse_ssl_mode,
    redact_dsn,
)

__all__ = [
    "SCHEMA_VERSION",
    "AnalysisStore",
    "InMemoryStore",
    "PostgresStore",
    "SchemaVersionError",
    "build_ssl",
    "build_store",
    "interrupted_payload",
    "parse_ssl_mode",
    "redact_dsn",
]


def build_store(settings: Settings) -> AnalysisStore:
    """Postgres when ``database_url`` is configured, otherwise the in-memory store."""
    if settings.database_url:
        return PostgresStore(
            settings.database_url,
            pool_min=settings.database_pool_min,
            pool_max=settings.database_pool_max,
        )
    return InMemoryStore()
