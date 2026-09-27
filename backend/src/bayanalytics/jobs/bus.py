"""Per-analysis event bus.

Every observable state change is published once: it gets a monotonic sequence number, is
persisted through the store (so late subscribers and reconnects can replay) and fanned out to
live subscribers. Terminal events close every subscriber queue.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from bayanalytics.schemas.events import EVENT_NAMES, AnalysisEvent
from bayanalytics.store.base import AnalysisStore

log = logging.getLogger(__name__)

_CLOSE = object()


class AnalysisEventBus:
    def __init__(self, store: AnalysisStore) -> None:
        self._store = store
        self._subscribers: dict[str, set[asyncio.Queue[Any]]] = {}
        self._seq: dict[str, int] = {}
        self._terminal: set[str] = set()
        self._lock = asyncio.Lock()

    def register(self, analysis_id: str, last_seq: int = 0) -> None:
        self._seq.setdefault(analysis_id, last_seq)

    def is_terminal(self, analysis_id: str) -> bool:
        return analysis_id in self._terminal

    async def publish(self, analysis_id: str, event: str, data: dict[str, Any]) -> AnalysisEvent:
        if event not in EVENT_NAMES:
            raise ValueError(f"unknown event name: {event}")
        async with self._lock:
            if analysis_id in self._terminal:
                raise RuntimeError(f"analysis {analysis_id} already reached a terminal event")
            seq = self._seq.get(analysis_id, 0) + 1
            self._seq[analysis_id] = seq
            record = AnalysisEvent(event=event, analysis_id=analysis_id, seq=seq, data=data)  # type: ignore[arg-type]
            await self._store.append_event(record)
            if record.terminal:
                self._terminal.add(analysis_id)
            for queue in self._subscribers.get(analysis_id, ()):
                queue.put_nowait(record)
                if record.terminal:
                    queue.put_nowait(_CLOSE)
        return record

    def last_seq(self, analysis_id: str) -> int:
        return self._seq.get(analysis_id, 0)

    def close(self, analysis_id: str) -> None:
        """Release live subscribers (used when a terminal event could not be published)."""
        for queue in self._subscribers.get(analysis_id, ()):
            queue.put_nowait(_CLOSE)

    def forget(self, analysis_id: str) -> None:
        """Drop in-memory bookkeeping for a finished analysis (its events stay in the store).

        Only terminal analyses are forgotten: a later ``stream()`` replays the persisted
        events and stops at the persisted terminal event, so nothing is lost.
        """
        if analysis_id not in self._terminal:
            self._subscribers.pop(analysis_id, None)
            return
        self._subscribers.pop(analysis_id, None)
        self._seq.pop(analysis_id, None)
        self._terminal.discard(analysis_id)

    async def stream(self, analysis_id: str, after_seq: int = 0) -> AsyncIterator[AnalysisEvent]:
        """Replay persisted events after ``after_seq`` then follow live events until terminal."""
        queue: asyncio.Queue[Any] = asyncio.Queue()
        async with self._lock:
            self._subscribers.setdefault(analysis_id, set()).add(queue)
            terminal_already = analysis_id in self._terminal
        try:
            last = after_seq
            # Replay from one event before ``after_seq`` so a client that reconnects with the
            # terminal event's own id (what EventSource does after the server closes) is told
            # the stream is over instead of waiting on a queue that will never fill.
            replay = await self._store.list_events(analysis_id, after_seq=max(after_seq - 1, 0))
            if replay and replay[0].seq == after_seq and replay[0].terminal:
                return
            for record in replay:
                if record.seq > last:
                    last = record.seq
                    yield record
                    if record.terminal:
                        return
            if terminal_already:
                return
            while True:
                item = await queue.get()
                if item is _CLOSE:
                    return
                record = item
                if record.seq <= last:
                    continue
                last = record.seq
                yield record
                if record.terminal:
                    return
        finally:
            subscribers = self._subscribers.get(analysis_id)
            if subscribers is not None:
                subscribers.discard(queue)
