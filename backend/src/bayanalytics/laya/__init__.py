"""Laya layer: bounded, non-generative decisions behind a persistent Node worker.

Public surface:

- ``LayaClient`` protocol and ``LayaLoadInfo`` / ``LayaHealth`` (``base``)
- ``LayaWorkerClient`` (``client``): NDJSON client for ``worker/worker.mjs``
- ``LayaFinanceWrapper`` (``wrapper``): question sets -> ``LayaDecision`` records
- finance question builders and ``LAYA_SCHEMA_VERSION`` (``schemas``)
- ``compact_state`` / ``validate_questions`` / ``state_digest`` (``compaction``)
- ``create_laya_client(settings)``: the worker client for these settings
"""

from __future__ import annotations

from collections.abc import Mapping

from bayanalytics.config import Settings
from bayanalytics.laya.base import LayaClient, LayaHealth, LayaLoadInfo
from bayanalytics.laya.client import LayaWorkerClient
from bayanalytics.laya.compaction import (
    compact_state,
    state_digest,
    validate_questions,
)
from bayanalytics.laya.schemas import LAYA_SCHEMA_VERSION
from bayanalytics.laya.wrapper import LayaFinanceWrapper


def create_laya_client(settings: Settings, env: Mapping[str, str] | None = None) -> LayaClient:
    return LayaWorkerClient(settings, env)


__all__ = [
    "LAYA_SCHEMA_VERSION",
    "LayaClient",
    "LayaFinanceWrapper",
    "LayaHealth",
    "LayaLoadInfo",
    "LayaWorkerClient",
    "compact_state",
    "create_laya_client",
    "state_digest",
    "validate_questions",
]
