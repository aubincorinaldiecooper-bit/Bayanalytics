"""Speech-to-text (listening only). ``build_transcriber(settings)`` picks by ``whisper_mode``.

- ``cli``: ``WhisperCliTranscriber`` (whisper.cpp ``whisper-cli``; ``available()`` is False
  until the binary and model file are found, so ``/capabilities.voice`` stays truthful),
- ``disabled`` (default): ``DisabledTranscriber``.
"""

from __future__ import annotations

from bayanalytics.config import Settings
from bayanalytics.whisper.audio import WavInfo, inspect_wav_bytes, inspect_wav_file
from bayanalytics.whisper.base import Transcriber
from bayanalytics.whisper.client import WhisperCliTranscriber
from bayanalytics.whisper.disabled import DisabledTranscriber

__all__ = [
    "DisabledTranscriber",
    "Transcriber",
    "WavInfo",
    "WhisperCliTranscriber",
    "build_transcriber",
    "inspect_wav_bytes",
    "inspect_wav_file",
]


def build_transcriber(settings: Settings) -> Transcriber:
    if settings.whisper_mode == "cli":
        return WhisperCliTranscriber(settings)
    return DisabledTranscriber()
