"""Laya layer: bounded, non-generative decisions behind a persistent Node worker.

Public surface:

- ``LayaClient`` protocol and ``LayaLoadInfo`` / ``LayaHealth`` (``base``)
- ``LayaWorkerClient`` (``client``): NDJSON client for ``worker/worker.mjs``
- ``MockLaya`` (``mock``): deterministic rule-based double
- ``LayaFinanceWrapper`` (``wrapper``): question sets -> ``LayaDecision`` records
- finance question builders and ``LAYA_SCHEMA_VERSION`` (``schemas``)
- ``compact_state`` / ``validate_questions`` / ``state_digest`` (``compaction``)
- ``create_laya_client(settings)``: picks the implementation from ``settings.laya_mode``
"""

from __future__ import annotations

from collections.abc import Mapping

from bayanalytics.config import Settings
from bayanalytics.laya.base import LayaClient, LayaHealth, LayaLoadInfo
from bayanalytics.laya.client import LayaWorkerClient
from bayanalytics.laya.compaction import (
    compact_state,
    estimate_tokens,
    state_digest,
    validate_questions,
)
from bayanalytics.laya.mock import MockLaya
from bayanalytics.laya.schemas import LAYA_SCHEMA_VERSION
from bayanalytics.laya.wrapper import LayaFinanceWrapper


def create_laya_client(settings: Settings, env: Mapping[str, str] | None = None) -> LayaClient:
    """``MockLaya`` when ``settings.laya_mode == "mock"``, else a ``LayaWorkerClient``."""
    if settings.laya_mode == "mock":
        return MockLaya()
    return LayaWorkerClient(settings, env)


__all__ = [
    "LAYA_SCHEMA_VERSION",
    "LayaClient",
    "LayaFinanceWrapper",
    "LayaHealth",
    "LayaLoadInfo",
    "LayaWorkerClient",
    "MockLaya",
    "compact_state",
    "create_laya_client",
    "estimate_tokens",
    "state_digest",
    "validate_questions",
]
