"""Speech-to-text boundary (listening only, no TTS)."""

from __future__ import annotations

from typing import Protocol

from bayanalytics.schemas.transcriptions import Transcription


class Transcriber(Protocol):
    def available(self) -> bool: ...

    async def transcribe(
        self, audio: bytes, filename: str, content_type: str | None = None
    ) -> Transcription: ...

    async def close(self) -> None: ...
