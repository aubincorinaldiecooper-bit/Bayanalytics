"""``DisabledTranscriber``: voice input is off (``whisper_mode=disabled``)."""

from __future__ import annotations

from typing import Any

from bayanalytics.errors import AnalysisError
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.transcriptions import Transcription


class DisabledTranscriber:
    """``available()`` is False and every call fails with a structured 503."""

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
