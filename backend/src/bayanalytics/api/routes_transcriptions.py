"""Voice transcription (AGENT.md 37.5). Transcribes only; never starts an analysis."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, UploadFile

from bayanalytics.api.deps import RuntimeDep
from bayanalytics.errors import AnalysisError
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.transcriptions import Transcription

router = APIRouter(prefix="/transcriptions", tags=["transcriptions"])

MAX_AUDIO_BYTES = 25 * 1024 * 1024


@router.post("", response_model=Transcription)
async def create_transcription(
    audio: Annotated[UploadFile, File(...)], rt: RuntimeDep
) -> Transcription:
    if not rt.transcriber.available():
        raise AnalysisError(
            ErrorCode.WHISPER_FAILED,
            "Voice input is not available on this backend.",
            retryable=False,
            details={"reason": "voice_unavailable"},
            http_status=503,
        )
    data = await audio.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise AnalysisError(
            ErrorCode.INVALID_REQUEST,
            "Audio exceeds the 25 MB limit.",
            details={"limit": MAX_AUDIO_BYTES},
        )
    if not data:
        raise AnalysisError(ErrorCode.INVALID_REQUEST, "Audio upload was empty.")
    return await rt.transcriber.transcribe(data, audio.filename or "audio", audio.content_type)
