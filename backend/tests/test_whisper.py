"""whisper.cpp CLI wrapper (with a fake ``whisper-cli``), the disabled/fixed transcribers and
the builder."""

from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from bayanalytics.config import Settings
from bayanalytics.errors import AnalysisError
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.whisper import (
    DisabledTranscriber,
    WhisperCliTranscriber,
    build_transcriber,
    inspect_wav_bytes,
)
from bayanalytics.whisper.audio import write_silence_wav
from bayanalytics.whisper.client import input_suffix, resolve_binary
from bayanalytics.whisper.disabled import DisabledTranscriber as DisabledFromModule
from doubles import FixedTranscriber

FAKE_WHISPER_OK = """#!/bin/sh
# fake whisper-cli: writes <outbase>.json, echoes whisper.cpp-style timings on stderr
out=""
model=""
while [ $# -gt 0 ]; do
  case "$1" in
    -of) out="$2"; shift 2 ;;
    -m) model="$2"; shift 2 ;;
    *) shift ;;
  esac
done
echo "whisper_init_from_file_with_params_no_state: loading model from '$model'" >&2
echo "whisper_print_timings:     load time =    12.50 ms" >&2
json='{"result":{"language":"en"},'
json="$json"'"transcription":[{"text":" Assess"},{"text":" Apple."}]}'
printf '%s' "$json" > "$out.json"
exit 0
"""

FAKE_WHISPER_FAIL = """#!/bin/sh
echo "error: failed to initialize whisper context from '/secret/model/path.bin'" >&2
exit 3
"""

FAKE_WHISPER_SLOW = """#!/bin/sh
exec sleep 30
"""

FAKE_WHISPER_GARBAGE = """#!/bin/sh
out=""
while [ $# -gt 0 ]; do case "$1" in -of) out="$2"; shift 2 ;; *) shift ;; esac; done
printf 'not json' > "$out.json"
exit 0
"""

FAKE_FFMPEG = f"""#!/bin/sh
# fake ffmpeg: ignores the input and writes a 700 ms 16 kHz mono WAV to the last argument
for last; do :; done
exec {sys.executable} -c "import sys; from pathlib import Path; \
from bayanalytics.whisper.audio import write_silence_wav; \
write_silence_wav(Path(sys.argv[1]), 700)" "$last"
"""


def leftovers(directory: Path) -> list[str]:
    """Entries left behind in the redirected temp dir (must be empty after every call)."""
    return sorted(entry.name for entry in directory.iterdir())


def write_script(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture
def scratch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect tempfile to a fresh directory so leftover temp dirs are detectable."""
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    monkeypatch.setenv("TMPDIR", str(tmpdir))
    monkeypatch.setattr(tempfile, "tempdir", None)
    return tmpdir


@pytest.fixture
def model_file(tmp_path: Path) -> Path:
    model = tmp_path / "ggml-tiny.bin"
    model.write_bytes(b"ggml")
    return model


@pytest.fixture
def wav16k(tmp_path: Path) -> bytes:
    path = tmp_path / "in16k.wav"
    write_silence_wav(path, 1500)
    return path.read_bytes()


@pytest.fixture
def wav44k(tmp_path: Path) -> bytes:
    path = tmp_path / "in44k.wav"
    write_silence_wav(path, 900, sample_rate=44_100, channels=2)
    return path.read_bytes()


def make_settings(tmp_path: Path, script: str, model: Path, **overrides: object) -> Settings:
    binary = write_script(tmp_path / "whisper-cli", script)
    payload: dict[str, object] = {
        "whisper_mode": "cli",
        "whisper_bin": str(binary),
        "whisper_model_path": model,
        "whisper_threads": 2,
        "ffmpeg_bin": str(tmp_path / "missing" / "ffmpeg"),
        "whisper_timeout_s": 10.0,
    }
    payload.update(overrides)
    return Settings(**payload)  # type: ignore[arg-type]


def assert_clean_details(exc: AnalysisError, *forbidden: str) -> None:
    text = json.dumps(exc.details) + str(exc)
    for token in forbidden:
        assert token not in text
    for value in exc.details.values():
        assert len(str(value)) <= 200


# --- WhisperCliTranscriber ------------------------------------------------------------------


async def test_transcribe_16k_wav_via_fake_cli(
    tmp_path: Path, model_file: Path, wav16k: bytes, scratch: Path
) -> None:
    settings = make_settings(tmp_path, FAKE_WHISPER_OK, model_file)
    transcriber = WhisperCliTranscriber(settings)
    assert transcriber.available()
    result = await transcriber.transcribe(wav16k, "clip.wav", "audio/wav")
    assert result.text == "Assess Apple."
    assert result.duration_ms == 1500
    assert 0 <= result.transcription_ms < 10_000
    stats = transcriber.stats
    assert stats["converted"] is False
    assert stats["duration_ms"] == 1500
    assert stats["whisper_load_ms"] == 12.5
    assert stats["transcription_ms"] == result.transcription_ms
    assert stats["whisper_runtime_ms"] >= 0
    assert "whisper_peak_rss_mb" in stats
    assert stats["threads"] == 2
    assert leftovers(scratch) == []  # temp dir removed
    await transcriber.close()


async def test_non_16k_wav_without_ffmpeg_fails_with_ffmpeg_missing(
    tmp_path: Path, model_file: Path, wav44k: bytes, scratch: Path
) -> None:
    transcriber = WhisperCliTranscriber(make_settings(tmp_path, FAKE_WHISPER_OK, model_file))
    with pytest.raises(AnalysisError) as info:
        await transcriber.transcribe(wav44k, "clip.wav", "audio/wav")
    exc = info.value
    assert exc.code is ErrorCode.WHISPER_FAILED
    assert exc.details["reason"] == "ffmpeg_missing"
    assert_clean_details(exc, str(scratch), str(tmp_path))
    assert leftovers(scratch) == []


async def test_non_wav_input_is_converted_with_ffmpeg(
    tmp_path: Path, model_file: Path, scratch: Path
) -> None:
    ffmpeg = write_script(tmp_path / "ffmpeg", FAKE_FFMPEG)
    settings = make_settings(tmp_path, FAKE_WHISPER_OK, model_file, ffmpeg_bin=str(ffmpeg))
    transcriber = WhisperCliTranscriber(settings)
    result = await transcriber.transcribe(
        b"\x1aE\xdf\xa3 not really webm", "voice.webm", "audio/webm"
    )
    assert result.text == "Assess Apple."
    assert result.duration_ms == 700  # from the converted WAV header
    assert transcriber.stats["converted"] is True
    assert transcriber.stats["ffmpeg_ms"] >= 0
    assert leftovers(scratch) == []


async def test_nonzero_exit_maps_to_whisper_failed(
    tmp_path: Path, model_file: Path, wav16k: bytes, scratch: Path, caplog: pytest.LogCaptureFixture
) -> None:
    transcriber = WhisperCliTranscriber(make_settings(tmp_path, FAKE_WHISPER_FAIL, model_file))
    with (
        caplog.at_level("WARNING", logger="bayanalytics.whisper.client"),
        pytest.raises(AnalysisError) as info,
    ):
        await transcriber.transcribe(wav16k, "clip.wav")
    exc = info.value
    assert exc.code is ErrorCode.WHISPER_FAILED
    assert exc.retryable is True
    assert exc.details == {"reason": "nonzero_exit", "exit_code": 3}
    assert_clean_details(exc, str(scratch), str(tmp_path), "/secret/model/path.bin")
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "code 3" in logged
    assert str(scratch) not in logged
    assert len(logged) < 400
    assert leftovers(scratch) == []


async def test_timeout_kills_child_and_maps_to_whisper_failed(
    tmp_path: Path, model_file: Path, wav16k: bytes, scratch: Path
) -> None:
    settings = make_settings(tmp_path, FAKE_WHISPER_SLOW, model_file, whisper_timeout_s=0.3)
    transcriber = WhisperCliTranscriber(settings)
    started = time.perf_counter()
    with pytest.raises(AnalysisError) as info:
        await transcriber.transcribe(wav16k, "clip.wav")
    assert time.perf_counter() - started < 5.0
    exc = info.value
    assert exc.code is ErrorCode.WHISPER_FAILED
    assert exc.details["reason"] == "timeout"
    assert exc.details["timeout_s"] == 0.3
    assert_clean_details(exc, str(scratch))
    assert leftovers(scratch) == []


async def test_bad_json_output_maps_to_whisper_failed(
    tmp_path: Path, model_file: Path, wav16k: bytes, scratch: Path
) -> None:
    transcriber = WhisperCliTranscriber(make_settings(tmp_path, FAKE_WHISPER_GARBAGE, model_file))
    with pytest.raises(AnalysisError) as info:
        await transcriber.transcribe(wav16k, "clip.wav")
    assert info.value.details["reason"] == "bad_output"
    assert_clean_details(info.value, str(scratch))
    assert leftovers(scratch) == []


async def test_empty_audio_and_unavailable(tmp_path: Path, model_file: Path, wav16k: bytes) -> None:
    transcriber = WhisperCliTranscriber(make_settings(tmp_path, FAKE_WHISPER_OK, model_file))
    with pytest.raises(AnalysisError) as info:
        await transcriber.transcribe(b"", "clip.wav")
    assert info.value.details["reason"] == "empty_audio"

    missing_model = make_settings(tmp_path, FAKE_WHISPER_OK, tmp_path / "nope.bin")
    unavailable = WhisperCliTranscriber(missing_model)
    assert not unavailable.available()
    with pytest.raises(AnalysisError) as info:
        await unavailable.transcribe(wav16k, "clip.wav")
    assert info.value.details["reason"] == "whisper_unavailable"
    assert str(tmp_path) not in json.dumps(info.value.details)


def test_available_logic(tmp_path: Path, model_file: Path) -> None:
    ok = WhisperCliTranscriber(make_settings(tmp_path, FAKE_WHISPER_OK, model_file))
    assert ok.available() and ok.unavailable_reason() is None

    no_model = WhisperCliTranscriber(make_settings(tmp_path, FAKE_WHISPER_OK, tmp_path / "x.bin"))
    assert not no_model.available()
    assert no_model.unavailable_reason() == "whisper model file not found"

    none_model = WhisperCliTranscriber(
        make_settings(tmp_path, FAKE_WHISPER_OK, model_file, whisper_model_path=None)
    )
    assert not none_model.available()

    wrong_mode = WhisperCliTranscriber(
        make_settings(tmp_path, FAKE_WHISPER_OK, model_file, whisper_mode="disabled")
    )
    assert not wrong_mode.available()
    assert wrong_mode.unavailable_reason() == "whisper_mode is not cli"

    no_binary = WhisperCliTranscriber(
        make_settings(tmp_path, FAKE_WHISPER_OK, model_file, whisper_bin=str(tmp_path / "absent"))
    )
    assert not no_binary.available()
    assert no_binary.unavailable_reason() == "whisper binary not found"

    not_on_path = WhisperCliTranscriber(
        make_settings(tmp_path, FAKE_WHISPER_OK, model_file, whisper_bin="surely-not-a-real-binary")
    )
    assert not not_on_path.available()

    # legacy `main` binary name resolves through PATH like any other name
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    write_script(legacy_dir / "main", FAKE_WHISPER_OK)
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{legacy_dir}{os.pathsep}{old_path}"
    try:
        legacy = WhisperCliTranscriber(
            make_settings(tmp_path, FAKE_WHISPER_OK, model_file, whisper_bin="main")
        )
        assert legacy.available()
        assert legacy.binary == str(legacy_dir / "main")
    finally:
        os.environ["PATH"] = old_path

    assert resolve_binary(None) is None
    assert resolve_binary("") is None
    assert resolve_binary(str(model_file)) is None  # exists but not executable


def test_threads_default(tmp_path: Path, model_file: Path) -> None:
    auto = WhisperCliTranscriber(
        make_settings(tmp_path, FAKE_WHISPER_OK, model_file, whisper_threads=None)
    )
    assert 1 <= auto.threads <= 4


def test_input_suffix() -> None:
    assert input_suffix("clip.WAV", None) == ".wav"
    assert input_suffix("voice.webm", "audio/webm") == ".webm"
    assert input_suffix("../../etc/passwd", "audio/ogg; codecs=opus") == ".ogg"
    assert input_suffix("noext", "audio/mpeg") == ".mp3"
    assert input_suffix("weird.toolongext", None) == ".bin"
    assert input_suffix("", None) == ".bin"
    assert input_suffix(None, "text/plain") == ".bin"


# --- doubles and builder --------------------------------------------------------------------


async def test_fixed_transcriber_double(wav16k: bytes, wav44k: bytes) -> None:
    fixed = FixedTranscriber()
    assert fixed.available()
    result = await fixed.transcribe(wav16k, "clip.wav", "audio/wav")
    assert result.text == "Assess Apple."
    assert result.duration_ms == 1500  # measured from the WAV header, not configured
    assert result.transcription_ms >= 0
    assert fixed.stats["duration_ms"] == 1500 and fixed.stats["input_bytes"] == len(wav16k)
    result = await fixed.transcribe(wav44k, "clip.wav")
    assert result.duration_ms == 900
    result = await fixed.transcribe(b"definitely not audio", "clip.webm", "audio/webm")
    assert result.duration_ms == 0
    assert fixed.calls == 3
    custom = FixedTranscriber(text="Assess Microsoft over the next twelve months.")
    assert (await custom.transcribe(b"", "x.wav")).text.startswith("Assess Microsoft")
    await fixed.close()


async def test_disabled_transcriber(wav16k: bytes) -> None:
    disabled = DisabledTranscriber()
    assert not disabled.available()
    with pytest.raises(AnalysisError) as info:
        await disabled.transcribe(wav16k, "clip.wav")
    assert info.value.code is ErrorCode.WHISPER_FAILED
    assert info.value.details == {"reason": "voice_disabled"}
    assert info.value.retryable is False
    payload = info.value.payload()
    assert payload.code is ErrorCode.WHISPER_FAILED and payload.details == {
        "reason": "voice_disabled"
    }
    await disabled.close()


def test_build_transcriber(tmp_path: Path, model_file: Path) -> None:
    assert isinstance(build_transcriber(Settings()), DisabledTranscriber)
    assert isinstance(build_transcriber(Settings(whisper_mode="disabled")), DisabledTranscriber)
    assert DisabledTranscriber is DisabledFromModule
    cli = build_transcriber(make_settings(tmp_path, FAKE_WHISPER_OK, model_file))
    assert isinstance(cli, WhisperCliTranscriber) and cli.available()
    assert not build_transcriber(Settings(whisper_mode="cli")).available()  # nothing installed
    # The product knows only cli and disabled; a double is never selectable by configuration.
    with pytest.raises(ValidationError):
        Settings(whisper_mode="mock")
    import bayanalytics.whisper as whisper_pkg

    assert not any("mock" in name.lower() for name in whisper_pkg.__all__)


def test_inspect_wav_bytes(wav16k: bytes, wav44k: bytes) -> None:
    info = inspect_wav_bytes(wav16k)
    assert info is not None and info.whisper_ready
    assert (info.channels, info.sample_rate, info.sample_width) == (1, 16_000, 2)
    assert info.duration_ms == 1500
    info = inspect_wav_bytes(wav44k)
    assert info is not None and not info.whisper_ready
    assert (info.channels, info.sample_rate, info.duration_ms) == (2, 44_100, 900)
    assert inspect_wav_bytes(b"") is None
    assert inspect_wav_bytes(b"RIFF....WAVEfmt garbage") is None
