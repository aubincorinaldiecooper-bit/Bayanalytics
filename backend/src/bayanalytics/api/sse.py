"""Server-Sent Events streaming for analysis progress (AGENT.md section 37.2)."""

from __future__ import annotations

import asyncio
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
    yield sse_comment("connected")
    iterator = bus.stream(analysis_id, after_seq=after_seq).__aiter__()
    while True:
        try:
            record = await asyncio.wait_for(iterator.__anext__(), timeout=keepalive_s)
        except TimeoutError:
            yield sse_comment("keepalive")
            continue
        except StopAsyncIteration:
            return
        yield record.to_sse()
        if record.terminal:
            return


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
