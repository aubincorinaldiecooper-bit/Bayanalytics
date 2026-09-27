"""SSE event contract (AGENT.md section 37.2).

Events describe observable system state only. No prompts, no hidden reasoning, no raw
model traces ever go through this channel.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from bayanalytics.schemas.common import utcnow

EventName = Literal[
    "analysis.started",
    "instrument.resolved",
    "research.started",
    "research.query",
    "research.source_found",
    "research.source_rejected",
    "research.completed",
    "normalization.completed",
    "laya.started",
    "laya.decision",
    "laya.completed",
    "calculation.started",
    "calculation.completed",
    "spark.queued",
    "spark.loading",
    "spark.started",
    "spark.token",
    "spark.completed",
    "analysis.completed",
    "analysis.failed",
]

EVENT_NAMES: tuple[str, ...] = EventName.__args__  # type: ignore[attr-defined]
TERMINAL_EVENTS: frozenset[str] = frozenset({"analysis.completed", "analysis.failed"})


class AnalysisEvent(BaseModel):
    event: EventName
    analysis_id: str
    seq: int
    ts: datetime = Field(default_factory=utcnow)
    data: dict[str, Any] = Field(default_factory=dict)

    def to_sse(self) -> str:
        payload = {"analysis_id": self.analysis_id, "seq": self.seq, "ts": self.ts.isoformat()}
        payload.update(self.data)
        body = json.dumps(payload, ensure_ascii=False, default=_json_default)
        return f"id: {self.seq}\nevent: {self.event}\ndata: {body}\n\n"

    @property
    def terminal(self) -> bool:
        return self.event in TERMINAL_EVENTS


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return str(value)


def sse_comment(text: str = "keepalive") -> str:
    return f": {text}\n\n"
