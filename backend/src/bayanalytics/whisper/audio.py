"""WAV header inspection with the standard ``wave`` module. Pure, no subprocesses."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass
from pathlib import Path

WHISPER_SAMPLE_RATE = 16_000
WHISPER_CHANNELS = 1
WHISPER_SAMPLE_WIDTH = 2  # 16-bit PCM


@dataclass(frozen=True, slots=True)
class WavInfo:
    channels: int
    sample_rate: int
    sample_width: int  # bytes per sample
    frames: int

    @property
    def duration_ms(self) -> int:
        if self.sample_rate <= 0:
            return 0
        return round(self.frames * 1000 / self.sample_rate)

    @property
    def whisper_ready(self) -> bool:
        """True for the 16 kHz mono 16-bit PCM layout whisper.cpp consumes directly."""
        return (
            self.channels == WHISPER_CHANNELS
            and self.sample_rate == WHISPER_SAMPLE_RATE
            and self.sample_width == WHISPER_SAMPLE_WIDTH
        )


def _read(handle: wave.Wave_read) -> WavInfo:
    return WavInfo(
        channels=handle.getnchannels(),
        sample_rate=handle.getframerate(),
        sample_width=handle.getsampwidth(),
        frames=handle.getnframes(),
    )


def inspect_wav_bytes(data: bytes) -> WavInfo | None:
    """Header info for a PCM WAV, or ``None`` when the bytes are not a WAV ``wave`` can read."""
    try:
        with wave.open(io.BytesIO(data), "rb") as handle:
            return _read(handle)
    except (wave.Error, EOFError, OSError, ValueError):
        return None


def inspect_wav_file(path: Path) -> WavInfo | None:
    try:
        with wave.open(str(path), "rb") as handle:
            return _read(handle)
    except (wave.Error, EOFError, OSError, ValueError):
        return None


def write_silence_wav(
    path: Path,
    duration_ms: int,
    sample_rate: int = WHISPER_SAMPLE_RATE,
    channels: int = WHISPER_CHANNELS,
    sample_width: int = WHISPER_SAMPLE_WIDTH,
) -> WavInfo:
    """Write a silent PCM WAV (used by tests and the fake ffmpeg)."""
    frames = max(0, round(sample_rate * duration_ms / 1000))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00" * (frames * channels * sample_width))
    return WavInfo(
        channels=channels, sample_rate=sample_rate, sample_width=sample_width, frames=frames
    )
