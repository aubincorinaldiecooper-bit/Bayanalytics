"""Server-Sent Events streaming for analysis progress (AGENT.md section 37.2)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from starlette.responses import StreamingResponse

from bayanalytics.jobs.bus import AnalysisEventBus
from bayanalytics.schemas.events import sse_comment

log = logging.getLogger(__name__)

KEEPALIVE_S = 15.0


async def event_stream(
    bus: AnalysisEventBus, analysis_id: str, after_seq: int, keepalive_s: float = KEEPALIVE_S
) -> AsyncIterator[str]:
    """Replay + live events as SSE frames with keepalive comments during quiet stretches.

    The pending read on the bus iterator is kept alive across keepalive intervals: cancelling
    it (as ``asyncio.wait_for`` would) closes the async generator and would end the stream
    after the first quiet period.
    """
    yield sse_comment("connected")
    iterator = bus.stream(analysis_id, after_seq=after_seq).__aiter__()
    pending: asyncio.Task[object] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            done, _ = await asyncio.wait({pending}, timeout=keepalive_s)
            if not done:
                yield sse_comment("keepalive")
                continue
            task, pending = pending, None
            try:
                record = task.result()
            except StopAsyncIteration:
                return
            yield record.to_sse()  # type: ignore[attr-defined]
            if record.terminal:  # type: ignore[attr-defined]
                return
    finally:
        if pending is not None and not pending.done():
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending
        with contextlib.suppress(BaseException):
            await iterator.aclose()  # type: ignore[attr-defined]


def sse_response(stream: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def parse_after_seq(last_event_id: str | None, after: int | None) -> int:
    if after is not None and after >= 0:
        return after
    if last_event_id:
        try:
            return max(0, int(last_event_id.strip()))
        except ValueError:
            return 0
    return 0
