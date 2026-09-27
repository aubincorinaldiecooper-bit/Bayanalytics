"""Shared per-analysis context passed through every stage.

The orchestrator owns the ``AnalysisContext``; stages use it to emit observable events, to
check for cancellation at safe boundaries and to record stage timings. Nothing here touches
models, the network or the database.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from bayanalytics.errors import AnalysisError
from bayanalytics.schemas.common import ErrorCode

EventSink = Callable[[str, dict[str, Any]], Awaitable[None]]


async def null_sink(_event: str, _data: dict[str, Any]) -> None:
    return None


class CancelToken:
    """Best-effort cooperative cancellation. Stages call ``check()`` at safe boundaries.

    ``reason`` distinguishes a user cancel (CANCELLED) from a backend shutdown (INTERRUPTED),
    so a job stopped by a restart is never reported as cancelled by the user.
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self.reason: ErrorCode = ErrorCode.CANCELLED

    def cancel(self, reason: ErrorCode = ErrorCode.CANCELLED) -> None:
        if not self._event.is_set():
            self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise AnalysisError(self.reason)

    async def wait(self) -> None:
        await self._event.wait()


class StageTimers:
    """Monotonic stage timers recorded in milliseconds."""

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self.elapsed_ms: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        started = self._starts.pop(name, None)
        if started is None:
            return self.elapsed_ms.get(name, 0.0)
        ms = (time.perf_counter() - started) * 1000.0
        self.elapsed_ms[name] = self.elapsed_ms.get(name, 0.0) + ms
        return ms

    def mark(self, name: str, ms: float) -> None:
        self.elapsed_ms[name] = self.elapsed_ms.get(name, 0.0) + ms

    class _Span:
        def __init__(self, timers: StageTimers, name: str) -> None:
            self._timers = timers
            self._name = name

        def __enter__(self) -> StageTimers._Span:
            self._timers.start(self._name)
            return self

        def __exit__(self, *_exc: object) -> None:
            self._timers.stop(self._name)

    def span(self, name: str) -> StageTimers._Span:
        return StageTimers._Span(self, name)


@dataclass
class AnalysisContext:
    analysis_id: str
    emit: EventSink = null_sink
    cancel: CancelToken = field(default_factory=CancelToken)
    timers: StageTimers = field(default_factory=StageTimers)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    async def event(self, event_name: str, /, **data: Any) -> None:
        await self.emit(event_name, data)

    def check_cancelled(self) -> None:
        self.cancel.check()
