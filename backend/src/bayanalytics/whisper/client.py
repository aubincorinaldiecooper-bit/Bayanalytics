"""whisper.cpp command-line transcriber (AGENT.md sections 1.4, 13, 29, 37.5).

Flow per request::

    bytes -> temp dir -> (ffmpeg -> 16 kHz mono s16 WAV when needed)
          -> whisper-cli -m MODEL -f in.wav -t N -nt -oj -of out
          -> out.json -> concatenated segment text

Guarantees:

- temp files never outlive the call (``finally: rmtree``),
- every failure is an ``AnalysisError(WHISPER_FAILED)`` with a short ``reason`` and no audio
  path, model path or long stderr body in ``details`` or logs,
- the child's peak RSS and timings are measured (``PeakTracker`` + wall clock) and exposed in
  ``stats`` so the API can report ``whisper_peak_rss_mb`` / ``whisper_load_ms`` truthfully,
- one transcription at a time (``asyncio.Lock``): whisper.cpp is CPU bound and the reference
  machine has 8 GB,
- the upload's file suffix only steers ffmpeg's demuxer when it is a known audio suffix
  (anything else is written as ``.bin`` and probed), and ffmpeg runs with
  ``-protocol_whitelist file,crypto,data`` so a container can never make it open a network or
  device protocol,
- both children (ffmpeg and whisper-cli) run with the allow-listed environment from
  :mod:`bayanalytics.procenv`, never the backend's own settings or secrets.

The legacy whisper.cpp binary name ``main`` is supported by setting ``BAY_WHISPER_BIN=main``
(or a path to it); the argument syntax is identical.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from bayanalytics.config import Settings
from bayanalytics.errors import AnalysisError
from bayanalytics.procenv import child_env
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.transcriptions import Transcription
from bayanalytics.telemetry.memory import forget_process
from bayanalytics.telemetry.tracker import PeakTracker
from bayanalytics.whisper.audio import WavInfo, inspect_wav_file

logger = logging.getLogger(__name__)

_STDERR_LIMIT = 200
_LOAD_TIME = re.compile(r"load time\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*ms")

AUDIO_SUFFIXES: frozenset[str] = frozenset(
    {
        ".wav",
        ".webm",
        ".ogg",
        ".oga",
        ".opus",
        ".mp3",
        ".m4a",
        ".aac",
        ".flac",
        ".mp4",
        ".caf",
        ".aiff",
        ".aif",
    }
)
"""Upload suffixes ffmpeg may see. Anything else becomes ``.bin`` (ffmpeg probes the content)."""

FFMPEG_PROTOCOL_WHITELIST = "file,crypto,data"
"""ffmpeg protocols allowed while demuxing an upload: local files only, no network or devices."""

_CONTENT_TYPE_SUFFIX = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/vnd.wave": ".wav",
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/opus": ".ogg",
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/mp4": ".m4a",
    "video/mp4": ".mp4",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
    "audio/x-aac": ".aac",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/x-caf": ".caf",
    "audio/aiff": ".aiff",
    "audio/x-aiff": ".aiff",
}


def resolve_binary(name: str | None) -> str | None:
    """Absolute path for an executable: PATH lookup for bare names, direct check for paths."""
    if not name:
        return None
    candidate = Path(name).expanduser()
    if candidate.is_absolute() or os.sep in name:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
        return None
    return shutil.which(name)


def input_suffix(filename: str | None, content_type: str | None) -> str:
    """The suffix the upload is written with, from an allow list of audio suffixes only.

    The suffix is what ffmpeg uses to pick a demuxer before probing, so a client must not be
    able to choose an arbitrary one: the upload filename counts only when its suffix is in
    :data:`AUDIO_SUFFIXES`, then the declared content type is mapped, and anything else is
    ``.bin`` (ffmpeg then probes the bytes).
    """
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix in AUDIO_SUFFIXES:
            return suffix
    if content_type:
        mapped = _CONTENT_TYPE_SUFFIX.get(content_type.split(";", 1)[0].strip().lower())
        if mapped:
            return mapped
    return ".bin"


def ffmpeg_argv(ffmpeg: str, source: Path, target: Path) -> list[str]:
    """ffmpeg command line converting ``source`` to a 16 kHz mono s16 WAV at ``target``.

    ``-protocol_whitelist`` precedes ``-i`` so it applies to the input (and to anything the
    container references): only local files, the crypto wrapper and data URLs are allowed.
    """
    return [
        ffmpeg,
        "-y",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-protocol_whitelist",
        FFMPEG_PROTOCOL_WHITELIST,
        "-i",
        str(source),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(target),
    ]


def _failed(reason: str, **extra: Any) -> AnalysisError:
    return AnalysisError(ErrorCode.WHISPER_FAILED, details={"reason": reason, **extra})


class WhisperCliTranscriber:
    """``Transcriber`` backed by the ``whisper-cli`` executable from whisper.cpp."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self.stats: dict[str, Any] = {}

    # --- capability -------------------------------------------------------------------

    @property
    def binary(self) -> str | None:
        return resolve_binary(self._settings.whisper_bin)

    @property
    def model_path(self) -> Path | None:
        model = self._settings.whisper_model_path
        if model is None:
            return None
        model = Path(model).expanduser()
        return model if model.is_file() else None

    @property
    def threads(self) -> int:
        configured = self._settings.whisper_threads
        if configured and configured > 0:
            return int(configured)
        return max(1, min(4, os.cpu_count() or 1))

    def available(self) -> bool:
        return (
            self._settings.whisper_mode == "cli"
            and self.binary is not None
            and self.model_path is not None
        )

    def unavailable_reason(self) -> str | None:
        """Why ``available()`` is False, for the health endpoint (no paths)."""
        if self._settings.whisper_mode != "cli":
            return "whisper_mode is not cli"
        if self.binary is None:
            return "whisper binary not found"
        if self.model_path is None:
            return "whisper model file not found"
        return None

    # --- transcription ----------------------------------------------------------------

    async def transcribe(
        self, audio: bytes, filename: str, content_type: str | None = None
    ) -> Transcription:
        if not audio:
            raise _failed("empty_audio")
        if not self.available():
            raise _failed("whisper_unavailable", detail=self.unavailable_reason())
        async with self._lock:
            tmp = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="bay-whisper-"))
            try:
                return await self._transcribe_in(tmp, audio, filename, content_type)
            finally:
                await asyncio.to_thread(shutil.rmtree, tmp, True)

    async def _transcribe_in(
        self, tmp: Path, audio: bytes, filename: str, content_type: str | None
    ) -> Transcription:
        started = time.perf_counter()
        stats: dict[str, Any] = {"input_bytes": len(audio)}
        source = tmp / f"input{input_suffix(filename, content_type)}"
        await asyncio.to_thread(source.write_bytes, audio)

        info = await asyncio.to_thread(inspect_wav_file, source)
        if info is not None and info.whisper_ready:
            wav = source
            stats["converted"] = False
        else:
            wav = tmp / "input16k.wav"
            ffmpeg_started = time.perf_counter()
            await self._convert(source, wav, tmp)
            stats["ffmpeg_ms"] = round((time.perf_counter() - ffmpeg_started) * 1000)
            stats["converted"] = True
            info = await asyncio.to_thread(inspect_wav_file, wav)
            if info is None or not info.whisper_ready:
                raise _failed("ffmpeg_bad_output")
        stats["duration_ms"] = info.duration_ms

        text, run_stats = await self._run_whisper(wav, tmp)
        stats.update(run_stats)
        stats["total_ms"] = round((time.perf_counter() - started) * 1000)
        self.stats = stats
        return Transcription(
            text=text,
            duration_ms=info.duration_ms,
            transcription_ms=int(run_stats["transcription_ms"]),
        )

    async def _convert(self, source: Path, target: Path, tmp: Path) -> None:
        ffmpeg = resolve_binary(self._settings.ffmpeg_bin)
        if ffmpeg is None:
            raise _failed("ffmpeg_missing")
        argv = ffmpeg_argv(ffmpeg, source, target)
        code, stderr, _tracker = await self._run(argv, tmp, "ffmpeg", "ffmpeg_timeout")
        if code != 0:
            self._log_stderr("ffmpeg", code, stderr, tmp)
            raise _failed("ffmpeg_failed", exit_code=code)

    async def _run_whisper(self, wav: Path, tmp: Path) -> tuple[str, dict[str, Any]]:
        outbase = tmp / "out"
        argv = [
            str(self.binary),
            "-m",
            str(self.model_path),
            "-f",
            str(wav),
            "-t",
            str(self.threads),
            "-nt",
            "-oj",
            "-of",
            str(outbase),
        ]
        started = time.perf_counter()
        code, stderr, tracker = await self._run(argv, tmp, "whisper", "timeout")
        transcription_ms = round((time.perf_counter() - started) * 1000)
        if code != 0:
            self._log_stderr("whisper", code, stderr, tmp)
            raise _failed("nonzero_exit", exit_code=code)
        text = await asyncio.to_thread(self._read_output, outbase.with_suffix(".json"))
        child = tracker.children.get("whisper") or {}
        load_ms = _parse_load_ms(stderr)
        run_stats: dict[str, Any] = {
            "transcription_ms": transcription_ms,
            "whisper_runtime_ms": transcription_ms,
            "whisper_load_ms": load_ms,
            "whisper_peak_rss_mb": child.get("peak_rss_mb"),
            "threads": self.threads,
        }
        return text, run_stats

    async def _run(
        self, argv: list[str], cwd: Path, track_as: str, timeout_reason: str
    ) -> tuple[int, str, PeakTracker]:
        timeout = float(self._settings.whisper_timeout_s)
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(cwd),
                env=child_env(),
            )
        except OSError as exc:
            raise _failed(f"{track_as}_spawn_failed", error=type(exc).__name__) from exc
        tracker = PeakTracker(interval_s=0.05, child_pids=lambda: {track_as: proc.pid})
        async with tracker:
            try:
                _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                await _terminate(proc)
                raise _failed(timeout_reason, timeout_s=timeout) from None
            except asyncio.CancelledError:
                await _terminate(proc)
                raise
        forget_process(proc.pid)
        return int(proc.returncode or 0), stderr.decode("utf-8", "replace"), tracker

    @staticmethod
    def _read_output(path: Path) -> str:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _failed("bad_output", error=type(exc).__name__) from exc
        segments = payload.get("transcription") if isinstance(payload, dict) else None
        if not isinstance(segments, list):
            raise _failed("bad_output", error="missing_transcription")
        pieces = [str(seg.get("text", "")) for seg in segments if isinstance(seg, dict)]
        return " ".join("".join(pieces).split())

    def _log_stderr(self, tool: str, code: int, stderr: str, tmp: Path) -> None:
        scrubbed = stderr.replace(str(tmp), "<tmp>")
        model = self._settings.whisper_model_path
        if model is not None:
            scrubbed = scrubbed.replace(str(model), "<model>")
        tail = " ".join(scrubbed.split())[-_STDERR_LIMIT:]
        logger.warning("%s exited with code %s: %s", tool, code, tail)

    async def close(self) -> None:
        return None


def _parse_load_ms(stderr: str) -> float | None:
    match = _LOAD_TIME.search(stderr)
    return float(match.group(1)) if match else None


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        proc.kill()
    except ProcessLookupError:
        return
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except TimeoutError:  # pragma: no cover - the kernel did not reap a killed child
        logger.warning("whisper child did not exit after kill")


__all__ = [
    "AUDIO_SUFFIXES",
    "FFMPEG_PROTOCOL_WHITELIST",
    "WavInfo",
    "WhisperCliTranscriber",
    "ffmpeg_argv",
    "input_suffix",
    "resolve_binary",
]
