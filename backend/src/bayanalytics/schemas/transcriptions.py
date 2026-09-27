"""Voice transcription response (AGENT.md section 37.5)."""

from __future__ import annotations

from pydantic import BaseModel


class Transcription(BaseModel):
    text: str
    duration_ms: int
    transcription_ms: int
