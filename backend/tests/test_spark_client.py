"""LlamaSparkClient / LlamaServerManager against a fake llama-server (httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.spark import manager as manager_module
from bayanalytics.spark.base import SparkMessage, SparkRunOptions
from bayanalytics.spark.client import LlamaSparkClient
from bayanalytics.spark.manager import LlamaServerManager
from bayanalytics.spark.profiles import (
    MemorySnapshot,
    assert_can_allocate,
    check_availability,
    profile_specs,
    read_lockfile,
    static_probe,
    version_fields,
)

SENTINEL = "PROMPT-CONTENT-MUST-NOT-LEAK"
OK_PROBE = static_probe(available_mb=6000.0, total_mb=8192.0)


# --- fakes ------------------------------------------------------------------------------------


class FakeProcess:
    def __init__(self, server: FakeLlamaServer, argv: list[str]) -> None:
        self.server = server
        self.argv = argv
        self.pid = os.getpid()  # a live pid so psutil RSS sampling works
        self.returncode: int | None = None
        self._exited = asyncio.Event()

    def terminate(self) -> None:
        self._exit(-15)

    def kill(self) -> None:
        self._exit(-9)

    def _exit(self, code: int) -> None:
        if self.returncode is None:
            self.returncode = code
        self.server.healthy = False
        self._exited.set()

    async def wait(self) -> int:
        await self._exited.wait()
        return self.returncode or 0


class StubbornProcess(FakeProcess):
    """Ignores SIGTERM; only kill() ends it."""

    def terminate(self) -> None:
        return None


class FakeLlamaServer:
    def __init__(self) -> None:
        self.healthy = False
        self.auto_healthy = True  # become healthy as soon as a process is spawned
        self.exit_on_spawn = False
        self.n_ctx = 32768
        self.build_info = "b10900-fake"
        self.status = 200
        self.transport_error: type[httpx.TransportError] | None = None
        self.deltas = ["## Summary\n", "Evidence suggests ", "signals are mixed ", "[src_a]."]
        self.finish_reason = "stop"
        self.include_usage = True
        self.include_timings = True
        self.raw_lines: list[str] | None = None
        self.process_cls: type[FakeProcess] = FakeProcess
        self.processes: list[FakeProcess] = []
        self.requests: list[dict[str, Any]] = []
        self.active = 0
        self.overlap = False

    async def spawn(self, argv: list[str]) -> FakeProcess:
        process = self.process_cls(self, argv)
        self.processes.append(process)
        if self.exit_on_spawn:
            process._exit(1)
        else:
            self.healthy = self.auto_healthy
        return process

    async def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            if self.healthy:
                return httpx.Response(200, json={"status": "ok"})
            return httpx.Response(503, json={"error": {"message": "Loading model"}})
        if path == "/props":
            if not self.healthy:
                return httpx.Response(503)
            return httpx.Response(
                200,
                json={
                    "default_generation_settings": {"n_ctx": self.n_ctx},
                    "total_slots": 1,
                    "build_info": self.build_info,
                },
            )
        if path == "/v1/chat/completions":
            self.requests.append(json.loads(request.content))
            if self.transport_error is not None:
                raise self.transport_error("simulated", request=request)
            if self.status != 200:
                return httpx.Response(self.status, json={"error": {"message": "boom"}})
            return httpx.Response(
                200, stream=FakeSSEStream(self), headers={"content-type": "text/event-stream"}
            )
        return httpx.Response(404)

    def sse_lines(self) -> list[str]:
        if self.raw_lines is not None:
            return self.raw_lines
        lines = []
        for index, delta in enumerate(self.deltas):
            chunk = {
                "id": "chatcmpl-1",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": delta}
                        if index
                        else {"role": "assistant", "content": delta},
                        "finish_reason": None,
                    }
                ],
            }
            lines.append("data: " + json.dumps(chunk))
        lines.append(
            "data: "
            + json.dumps(
                {"choices": [{"index": 0, "delta": {}, "finish_reason": self.finish_reason}]}
            )
        )
        final: dict[str, Any] = {"choices": []}
        if self.include_usage:
            final["usage"] = {
                "prompt_tokens": 900,
                "completion_tokens": len(self.deltas),
                "total_tokens": 900 + len(self.deltas),
            }
        if self.include_timings:
            final["timings"] = {
                "prompt_n": 900,
                "prompt_ms": 1500.0,
                "predicted_n": len(self.deltas),
                "predicted_ms": 200.0,
                "predicted_per_second": 20.0,
            }
        if self.include_usage or self.include_timings:
            lines.append("data: " + json.dumps(final))
        lines.append("data: [DONE]")
        return lines

    async def _stream(self) -> AsyncIterator[bytes]:
        self.active += 1
        if self.active > 1:
            self.overlap = True
        try:
            for line in self.sse_lines():
                yield (line + "\n\n").encode()
                await asyncio.sleep(0)
        finally:
            self.active -= 1


class FakeSSEStream(httpx.AsyncByteStream):
    """Forwards ``aclose`` to the generator so the fake sees the client close the response."""

    def __init__(self, server: FakeLlamaServer) -> None:
        self._gen = server._stream()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._gen:
            yield chunk

    async def aclose(self) -> None:
        await self._gen.aclose()


class Harness:
    def __init__(self, settings: Settings, server: FakeLlamaServer, probe=OK_PROBE) -> None:
        self.server = server
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.http = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
        self.manager = LlamaServerManager(
            settings,
            probe=probe,
            spawn=server.spawn,
            http=self.http,
            version_probe=self.version_probe,
            health_poll_interval_s=0.005,
        )
        self.client = LlamaSparkClient(settings, manager=self.manager, http=self.http, probe=probe)
        self.tokens: list[str] = []

    async def version_probe(self) -> str | None:
        return "b10900"

    def ctx(self, analysis_id: str = "an_1") -> AnalysisContext:
        async def emit(name: str, data: dict[str, Any]) -> None:
            self.events.append((name, data))

        return AnalysisContext(analysis_id=analysis_id, emit=emit)

    async def on_token(self, text: str) -> None:
        self.tokens.append(text)

    def loading_events(self) -> list[dict[str, Any]]:
        return [data for name, data in self.events if name == "spark.loading"]

    async def aclose(self) -> None:
        await self.client.close()
        await self.http.aclose()


def messages() -> list[SparkMessage]:
    return [
        SparkMessage(role="system", content="rules"),
        SparkMessage(role="user", content=f"evidence {SENTINEL} [src_a]"),
    ]


# --- fixtures ---------------------------------------------------------------------------------


@pytest.fixture
def server() -> FakeLlamaServer:
    return FakeLlamaServer()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    model = tmp_path / "spark.gguf"
    model.write_bytes(b"GGUF")
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    return Settings(
        spark_mode="managed",
        spark_model_path=model,
        spark_llama_server_bin=str(binary),
        spark_server_url="http://127.0.0.1:8081",
        spark_start_timeout_s=2.0,
        spark_request_timeout_s=5.0,
        spark_threads=4,
    )


@pytest.fixture
async def harness(settings: Settings, server: FakeLlamaServer) -> AsyncIterator[Harness]:
    h = Harness(settings, server)
    yield h
    await h.aclose()


# --- streaming --------------------------------------------------------------------------------


async def test_stream_tokens_stats_and_request_shape(harness: Harness, server: FakeLlamaServer):
    ctx = harness.ctx()
    gen = await harness.client.run(
        "fast", messages(), harness.on_token, ctx, SparkRunOptions(max_tokens=333, stop=["</s>"])
    )

    assert harness.tokens == server.deltas
    assert gen.text == "".join(server.deltas)
    assert gen.truncated is False
    stats = gen.stats
    assert stats.profile == "fast"
    assert stats.context_ceiling == 32768
    assert stats.kv_cache_type == "f16"
    assert stats.finish_reason == "stop"
    assert stats.prompt_tokens == 900
    assert stats.output_tokens == len(server.deltas)
    assert stats.tokens_per_second == 20.0
    assert stats.time_to_first_token_ms is not None
    assert 0 <= stats.time_to_first_token_ms <= stats.total_ms
    assert stats.load_ms is not None and stats.load_ms >= 0
    assert stats.runtime_version == "b10900"
    assert stats.resident_rss_mb is not None and stats.resident_rss_mb > 0
    assert stats.peak_rss_mb is not None and stats.peak_rss_mb >= stats.resident_rss_mb
    assert ctx.diagnostics["spark_ttft_ms"] == stats.time_to_first_token_ms
    assert ctx.diagnostics["spark_load_ms"] == stats.load_ms
    assert ctx.timers.elapsed_ms["spark"] > 0

    body = server.requests[0]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["cache_prompt"] is True
    assert body["max_tokens"] == 333
    assert body["stop"] == ["</s>"]
    assert body["messages"][1]["content"].endswith("[src_a]")
    assert harness.loading_events() == [
        {"profile": "fast", "context_ceiling": 32768, "kv_cache_type": "f16"}
    ]


async def test_truncated_when_finish_reason_is_length(harness: Harness, server: FakeLlamaServer):
    server.finish_reason = "length"
    gen = await harness.client.run("fast", messages(), harness.on_token, harness.ctx())
    assert gen.truncated is True
    assert gen.stats.finish_reason == "length"


async def test_token_counts_estimated_without_usage(harness: Harness, server: FakeLlamaServer):
    server.include_usage = False
    server.include_timings = False
    gen = await harness.client.run("fast", messages(), harness.on_token, harness.ctx())
    assert gen.stats.output_tokens == len(server.deltas)
    assert gen.stats.prompt_tokens is None
    assert gen.stats.tokens_per_second is None


async def test_runtime_version_falls_back_to_props(settings: Settings, server: FakeLlamaServer):
    h = Harness(settings, server)

    async def no_version() -> str | None:
        return None

    h.manager._version_probe = no_version
    try:
        gen = await h.client.run("fast", messages(), h.on_token, h.ctx())
        assert gen.stats.runtime_version == "b10900-fake"
    finally:
        await h.aclose()


# --- error mapping ----------------------------------------------------------------------------


async def test_http_500_maps_to_inference_failed(harness: Harness, server: FakeLlamaServer):
    server.status = 500
    with pytest.raises(AnalysisError) as info:
        await harness.client.run("fast", messages(), harness.on_token, harness.ctx())
    err = info.value
    assert err.code == ErrorCode.SPARK_INFERENCE_FAILED
    assert err.details == {"reason": "http_status", "status": 500}
    assert SENTINEL not in json.dumps(err.details) + err.message
    assert harness.tokens == []


async def test_malformed_sse_maps_to_inference_failed(harness: Harness, server: FakeLlamaServer):
    server.raw_lines = ["data: {not json", "data: [DONE]"]
    with pytest.raises(AnalysisError) as info:
        await harness.client.run("fast", messages(), harness.on_token, harness.ctx())
    assert info.value.code == ErrorCode.SPARK_INFERENCE_FAILED
    assert info.value.details == {"reason": "malformed_sse"}


async def test_stream_ending_without_done_is_a_failure(harness: Harness, server: FakeLlamaServer):
    chunk = {"choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}]}
    server.raw_lines = ["data: " + json.dumps(chunk)]
    with pytest.raises(AnalysisError) as info:
        await harness.client.run("fast", messages(), harness.on_token, harness.ctx())
    assert info.value.code == ErrorCode.SPARK_INFERENCE_FAILED
    assert info.value.details == {"reason": "stream_ended_early"}
    assert harness.tokens == ["partial"]  # streamed, but never returned as a generation


async def test_server_error_chunk_maps_to_inference_failed(
    harness: Harness, server: FakeLlamaServer
):
    server.raw_lines = ['data: {"error": {"message": "context shift disabled"}}']
    with pytest.raises(AnalysisError) as info:
        await harness.client.run("fast", messages(), harness.on_token, harness.ctx())
    assert info.value.details == {"reason": "server_error"}


async def test_transport_error_maps_to_inference_failed(harness: Harness, server: FakeLlamaServer):
    server.transport_error = httpx.ReadTimeout
    with pytest.raises(AnalysisError) as info:
        await harness.client.run("fast", messages(), harness.on_token, harness.ctx())
    assert info.value.code == ErrorCode.SPARK_INFERENCE_FAILED
    assert info.value.details == {"reason": "ReadTimeout"}
    assert harness.manager.in_flight is False


# --- cancellation -----------------------------------------------------------------------------


async def test_cancel_mid_stream_raises_and_stops_tokens(harness: Harness, server: FakeLlamaServer):
    ctx = harness.ctx()
    received: list[str] = []

    async def on_token(text: str) -> None:
        received.append(text)
        if len(received) == 2:
            ctx.cancel.cancel()

    with pytest.raises(AnalysisError) as info:
        await harness.client.run("fast", messages(), on_token, ctx)
    assert info.value.code == ErrorCode.CANCELLED
    assert received == server.deltas[:2]
    assert server.active == 0  # response closed
    assert harness.manager.in_flight is False
    assert not harness.client.busy


async def test_cancelled_before_start_makes_no_request(harness: Harness, server: FakeLlamaServer):
    ctx = harness.ctx()
    ctx.cancel.cancel()
    with pytest.raises(AnalysisError) as info:
        await harness.client.run("fast", messages(), harness.on_token, ctx)
    assert info.value.code == ErrorCode.CANCELLED
    assert server.requests == []
    assert server.processes == []


# --- profile management -----------------------------------------------------------------------


async def test_profile_switch_restarts_with_new_flags(harness: Harness, server: FakeLlamaServer):
    ctx = harness.ctx()
    first = await harness.manager.ensure("fast", ctx)
    assert first.loaded_now is True
    argv = server.processes[0].argv
    assert argv[0].endswith("llama-server")
    assert argv[1:3] == ["-m", str(harness.client._settings.spark_model_path)]
    assert "-c" in argv and argv[argv.index("-c") + 1] == "32768"
    assert argv[argv.index("-ctk") + 1] == "f16"
    assert argv[argv.index("-ctv") + 1] == "f16"
    assert "-fa" not in argv
    assert argv[argv.index("-t") + 1] == "4"
    assert argv[argv.index("--host") + 1] == "127.0.0.1"
    assert argv[argv.index("--port") + 1] == "8081"
    assert "--no-webui" in argv and "--jinja" in argv

    second = await harness.manager.ensure("deep", ctx)
    assert second.loaded_now is True
    assert server.processes[0].returncode is not None  # old server stopped
    argv = server.processes[1].argv
    assert argv[argv.index("-c") + 1] == "131072"
    assert argv[argv.index("-ctk") + 1] == "q4_0"
    assert argv[argv.index("-ctv") + 1] == "q4_0"
    assert argv[argv.index("-fa") + 1] == "on"
    assert harness.manager.loaded_profile == "deep"
    assert harness.manager.n_ctx == 32768  # whatever /props reports

    third = await harness.manager.ensure("fast", ctx)
    assert third.loaded_now is True
    assert len(server.processes) == 3
    assert [e["profile"] for e in harness.loading_events()] == ["fast", "deep", "fast"]
    assert [e["context_ceiling"] for e in harness.loading_events()] == [32768, 131072, 32768]


async def test_ensure_is_noop_when_profile_already_loaded(
    harness: Harness, server: FakeLlamaServer
):
    ctx = harness.ctx()
    await harness.manager.ensure("fast", ctx)
    outcome = await harness.manager.ensure("fast", ctx)
    assert outcome.loaded_now is False
    assert outcome.load_ms == harness.manager.load_ms
    assert len(server.processes) == 1
    assert len(harness.loading_events()) == 1
    assert harness.manager.alive is True
    assert harness.manager.rss_mb() is not None


async def test_ensure_restarts_when_process_died(harness: Harness, server: FakeLlamaServer):
    ctx = harness.ctx()
    await harness.manager.ensure("fast", ctx)
    server.processes[0]._exit(137)
    outcome = await harness.manager.ensure("fast", ctx)
    assert outcome.loaded_now is True
    assert len(server.processes) == 2


async def test_health_wait_timeout_stops_process(settings: Settings, server: FakeLlamaServer):
    settings = settings.model_copy(update={"spark_start_timeout_s": 0.05})
    server.auto_healthy = False
    h = Harness(settings, server)
    try:
        with pytest.raises(AnalysisError) as info:
            await h.manager.ensure("deep", h.ctx())
        assert info.value.code == ErrorCode.SPARK_START_FAILED
        assert info.value.details["stage"] == "health_wait"
        assert server.processes[0].returncode is not None
        assert h.manager.alive is False
        assert h.manager.loaded_profile is None
        assert h.manager.pid is None
    finally:
        await h.aclose()


async def test_process_exit_during_startup(harness: Harness, server: FakeLlamaServer):
    server.exit_on_spawn = True
    with pytest.raises(AnalysisError) as info:
        await harness.manager.ensure("fast", harness.ctx())
    assert info.value.code == ErrorCode.SPARK_START_FAILED
    assert info.value.details == {"stage": "process_exit", "returncode": 1}


async def test_missing_binary_maps_to_start_failed(settings: Settings, server: FakeLlamaServer):
    h = Harness(settings, server)

    async def spawn(argv: list[str]) -> FakeProcess:
        raise FileNotFoundError(argv[0])

    h.manager._spawn = spawn
    try:
        with pytest.raises(AnalysisError) as info:
            await h.manager.ensure("fast", h.ctx())
        assert info.value.code == ErrorCode.SPARK_START_FAILED
        assert info.value.details == {"stage": "spawn", "reason": "llama-server not found"}
        assert str(settings.spark_llama_server_bin) not in info.value.message
    finally:
        await h.aclose()


async def test_missing_model_fails_preflight(settings: Settings, server: FakeLlamaServer, tmp_path):
    settings = settings.model_copy(update={"spark_model_path": tmp_path / "missing.gguf"})
    h = Harness(settings, server)
    try:
        with pytest.raises(AnalysisError) as info:
            await h.manager.ensure("fast", h.ctx())
        assert info.value.code == ErrorCode.SPARK_START_FAILED
        assert info.value.details["stage"] == "preflight"
        assert "missing.gguf" not in info.value.message
        assert server.processes == []
        assert h.client.availability("fast").reason == "model artifact not found"
    finally:
        await h.aclose()


async def test_deep_memory_gate_is_a_structured_error(settings: Settings, server: FakeLlamaServer):
    low = static_probe(available_mb=2048.0, total_mb=8192.0)
    h = Harness(settings, server, probe=low)
    try:
        with pytest.raises(AnalysisError) as info:
            await h.client.run("deep", messages(), h.on_token, h.ctx())
        err = info.value
        assert err.code == ErrorCode.DEEP_PROFILE_UNAVAILABLE
        assert "2048 MB" in err.message and "4096 MB" in err.message
        assert "gguf" not in err.message
        assert err.details["available_mb"] == 2048
        assert server.processes == []  # no silent fallback to fast, no spawn
        assert h.loading_events() == []
        # Fast still fits and runs.
        gen = await h.client.run("fast", messages(), h.on_token, h.ctx())
        assert gen.stats.profile == "fast"
    finally:
        await h.aclose()


async def test_swap_pressure_is_memory_pressure(settings: Settings, server: FakeLlamaServer):
    swapping = static_probe(available_mb=6000.0, swap_used_mb=3000.0, swap_percent=75.0)
    h = Harness(settings, server, probe=swapping)
    try:
        with pytest.raises(AnalysisError) as info:
            await h.manager.ensure("fast", h.ctx())
        assert info.value.code == ErrorCode.MEMORY_PRESSURE
        assert "75%" in info.value.message
    finally:
        await h.aclose()


async def test_restart_guard_while_in_flight(harness: Harness):
    ctx = harness.ctx()
    await harness.manager.ensure("fast", ctx)
    with harness.manager.request_scope():
        with pytest.raises(AnalysisError) as info:
            await harness.manager.ensure("deep", ctx)
        assert info.value.code == ErrorCode.INTERNAL_ERROR
    assert harness.manager.loaded_profile == "fast"


async def test_stop_kills_after_grace(
    settings: Settings, server: FakeLlamaServer, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(manager_module, "PROCESS_STOP_GRACE_S", 0.01)
    server.process_cls = StubbornProcess
    h = Harness(settings, server)
    try:
        await h.manager.ensure("fast", h.ctx())
        await h.manager.stop()
        assert server.processes[0].returncode == -9
        assert h.manager.alive is False
    finally:
        await h.aclose()


# --- concurrency ------------------------------------------------------------------------------


async def test_lock_serialises_concurrent_runs(harness: Harness, server: FakeLlamaServer):
    async def one(analysis_id: str) -> str:
        chunks: list[str] = []

        async def on_token(text: str) -> None:
            chunks.append(text)
            await asyncio.sleep(0)

        gen = await harness.client.run("fast", messages(), on_token, harness.ctx(analysis_id))
        assert "".join(chunks) == gen.text
        return gen.text

    texts = await asyncio.gather(one("an_a"), one("an_b"))
    assert texts == ["".join(server.deltas)] * 2
    assert server.overlap is False
    assert len(server.requests) == 2
    assert len(server.processes) == 1


# --- external mode ----------------------------------------------------------------------------


@pytest.fixture
def external_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={"spark_mode": "external", "spark_server_url": "http://127.0.0.1:9099/"}
    )


async def test_external_mode_uses_health_and_props(
    external_settings: Settings, server: FakeLlamaServer
):
    server.healthy = True
    server.n_ctx = 32768
    h = Harness(external_settings, server)
    try:
        assert h.manager.base_url == "http://127.0.0.1:9099"
        await h.client.start()
        assert h.client.availability("fast").available is True
        deep = h.client.availability("deep")
        assert deep.available is False
        assert deep.code == ErrorCode.DEEP_PROFILE_UNAVAILABLE
        assert "32768" in (deep.reason or "")

        outcome = await h.manager.ensure("fast", h.ctx())
        assert outcome.external is True
        assert outcome.n_ctx == 32768
        assert server.processes == []
        assert h.loading_events() == []

        with pytest.raises(AnalysisError) as info:
            await h.manager.ensure("deep", h.ctx())
        assert info.value.code == ErrorCode.DEEP_PROFILE_UNAVAILABLE
        assert "32768" in info.value.message

        gen = await h.client.run("fast", messages(), h.on_token, h.ctx())
        assert gen.text == "".join(server.deltas)
        assert gen.stats.load_ms is None
    finally:
        await h.aclose()


async def test_external_mode_unhealthy_is_start_failed(
    external_settings: Settings, server: FakeLlamaServer
):
    server.healthy = False
    h = Harness(external_settings, server)
    try:
        await h.client.start()
        cap = h.client.availability("fast")
        assert cap.available is False and cap.code == ErrorCode.SPARK_START_FAILED
        with pytest.raises(AnalysisError) as info:
            await h.client.run("fast", messages(), h.on_token, h.ctx())
        assert info.value.code == ErrorCode.SPARK_START_FAILED
        assert info.value.details == {"stage": "external_health"}
    finally:
        await h.aclose()


# --- availability rules (pure) ---------------------------------------------------------------


def test_check_availability_rules(settings: Settings):
    ok = OK_PROBE
    assert check_availability("fast", settings, ok, True, True).available is True
    assert check_availability("deep", settings, ok, True, True).context_ceiling == 131072

    low = static_probe(available_mb=3000.0, total_mb=8192.0)
    deep = check_availability("deep", settings, low, True, True)
    assert deep.available is False
    assert deep.code == ErrorCode.DEEP_PROFILE_UNAVAILABLE
    assert "3000 MB" in (deep.reason or "") and "4096" in (deep.reason or "")
    assert "/" not in (deep.reason or "")
    assert check_availability("fast", settings, low, True, True).available is True

    swap = static_probe(available_mb=6000.0, swap_used_mb=2000.0, swap_percent=60.0)
    fast = check_availability("fast", settings, swap, True, True)
    assert fast.code == ErrorCode.FAST_PROFILE_UNAVAILABLE
    assert "60%" in (fast.reason or "")

    no_model = check_availability("fast", settings, ok, False, True)
    assert no_model.code == ErrorCode.SPARK_START_FAILED
    assert no_model.reason == "model artifact not found"
    no_runtime = check_availability("fast", settings, ok, True, False)
    assert no_runtime.reason == "llama-server not found"

    external = settings.model_copy(update={"spark_mode": "external"})
    never_probed = static_probe(available_mb=0.0)  # memory arithmetic must not be consulted
    assert check_availability("deep", external, never_probed, False, False).available is True
    down = check_availability("fast", external, never_probed, False, False, external_healthy=False)
    assert down.code == ErrorCode.SPARK_START_FAILED
    small = check_availability(
        "deep", external, never_probed, False, False, external_healthy=True, external_n_ctx=8192
    )
    assert small.code == ErrorCode.DEEP_PROFILE_UNAVAILABLE and "8192" in (small.reason or "")

    mock = settings.model_copy(update={"spark_mode": "mock"})
    assert check_availability("deep", mock, never_probed, False, False).available is True


def test_assert_can_allocate_codes(settings: Settings):
    snapshot = assert_can_allocate("deep", settings, OK_PROBE)
    assert isinstance(snapshot, MemorySnapshot)
    with pytest.raises(AnalysisError) as info:
        assert_can_allocate("fast", settings, static_probe(available_mb=100.0))
    assert info.value.code == ErrorCode.MEMORY_PRESSURE
    with pytest.raises(AnalysisError) as info:
        assert_can_allocate("deep", settings, static_probe(available_mb=3000.0))
    assert info.value.code == ErrorCode.DEEP_PROFILE_UNAVAILABLE


def test_profile_specs_follow_settings(settings: Settings):
    specs = profile_specs(settings)
    assert specs["fast"].context_ceiling == 32768 and specs["fast"].kv_cache_type == "f16"
    assert specs["deep"].context_ceiling == 131072 and specs["deep"].kv_cache_type == "q4_0"
    assert specs["deep"].min_available_mb == settings.spark_deep_min_available_mb


def test_read_lockfile_and_version_fields(settings: Settings, tmp_path: Path):
    lock_path = tmp_path / "spark.lock.json"
    lock_path.write_text(
        json.dumps(
            {
                "hf_repo": "XHToken/Spark-X2.5-1.7B-GGUF",
                "hf_revision": "abc123",
                "gguf_quantization": "Q4_K_M",
                "gguf_sha256": "deadbeef",
                "gguf_file": "spark-x2.5-1.7b-q4_k_m.gguf",
                "llama_cpp_version": "b10901",
                "chat_template_source": "gguf",
            }
        )
    )
    lock = read_lockfile(lock_path)
    assert lock is not None and lock["gguf_sha256"] == "deadbeef"
    fields = version_fields(lock, settings)
    assert fields == {
        "spark_artifact": "XHToken/Spark-X2.5-1.7B-GGUF:Q4_K_M",
        "spark_runtime": "b10901",
        "spark_gguf_sha256": "deadbeef",
        "spark_hf_revision": "abc123",
    }
    assert version_fields(lock, settings, "b10950")["spark_runtime"] == "b10950"
    assert read_lockfile(tmp_path / "nope.json") is None
    assert read_lockfile(None) is None
    (tmp_path / "bad.json").write_text("[]")
    assert read_lockfile(tmp_path / "bad.json") is None
    assert version_fields(None, settings)["spark_artifact"] == settings.spark_artifact


async def test_client_reads_lockfile_for_version_info(
    settings: Settings, server: FakeLlamaServer, tmp_path: Path
):
    lock_path = tmp_path / "spark.lock.json"
    lock_path.write_text(json.dumps({"gguf_sha256": "cafe", "hf_revision": "rev1"}))
    h = Harness(settings.model_copy(update={"spark_lockfile": lock_path}), server)
    try:
        info = await h.client.version_info()
        assert info["spark_gguf_sha256"] == "cafe"
        assert info["spark_hf_revision"] == "rev1"
        assert info["spark_runtime"] == "b10900"
    finally:
        await h.aclose()
