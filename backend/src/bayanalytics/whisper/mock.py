"""Transcribers that need no whisper.cpp: a fixed-text mock and the disabled placeholder."""

from __future__ import annotations

import time
from typing import Any

from bayanalytics.errors import AnalysisError
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.transcriptions import Transcription
from bayanalytics.whisper.audio import inspect_wav_bytes


class MockTranscriber:
    """Returns a fixed transcript. ``duration_ms`` still comes from the WAV header."""

    def __init__(self, text: str = "Assess Apple.") -> None:
        self.text = text
        self.calls = 0
        self.stats: dict[str, Any] = {}

    def available(self) -> bool:
        return True

    async def transcribe(
        self, audio: bytes, filename: str, content_type: str | None = None
    ) -> Transcription:
        started = time.perf_counter()
        self.calls += 1
        info = inspect_wav_bytes(audio)
        duration_ms = info.duration_ms if info is not None else 0
        transcription_ms = int((time.perf_counter() - started) * 1000)
        self.stats = {
            "input_bytes": len(audio),
            "duration_ms": duration_ms,
            "transcription_ms": transcription_ms,
        }
        return Transcription(
            text=self.text, duration_ms=duration_ms, transcription_ms=transcription_ms
        )

    async def close(self) -> None:
        return None


class DisabledTranscriber:
    """Voice is off (``whisper_mode=disabled``): ``available()`` is False and calls fail."""

    stats: dict[str, Any] = {}

    def available(self) -> bool:
        return False

    async def transcribe(
        self, audio: bytes, filename: str, content_type: str | None = None
    ) -> Transcription:
        raise AnalysisError(
            ErrorCode.WHISPER_FAILED,
            "Voice input is disabled on this backend.",
            retryable=False,
            details={"reason": "voice_disabled"},
            http_status=503,
        )

    async def close(self) -> None:
        return None
