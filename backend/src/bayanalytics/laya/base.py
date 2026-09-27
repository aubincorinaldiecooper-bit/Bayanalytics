"""Laya client boundary. Implementations: ``LayaWorkerClient`` (Node worker) and ``MockLaya``."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel

from bayanalytics.schemas.decisions import LayaQuestion, LayaResult

# Limits of the English checkpoint shipped by @receptron/laya 0.1.2.
LAYA_MAX_LEN = 512
LAYA_HEAD_MAX_LEN = 192
LAYA_MAX_CHOICE_OPTIONS = 20


class LayaLoadInfo(BaseModel):
    load_ms: float
    resident_rss_mb: float | None = None
    package_version: str | None = None
    model_dir: str | None = None
    max_len: int = LAYA_MAX_LEN
    head_max_len: int = LAYA_HEAD_MAX_LEN


class LayaHealth(BaseModel):
    ok: bool
    loaded: bool
    pid: int | None = None
    resident_rss_mb: float | None = None
    restarts: int = 0
    detail: str | None = None


class LayaClient(Protocol):
    async def load(self) -> LayaLoadInfo: ...

    async def system_one(
        self, state: dict[str, Any] | str, questions: dict[str, LayaQuestion]
    ) -> LayaResult: ...

    async def health(self) -> LayaHealth: ...

    async def close(self) -> None: ...
