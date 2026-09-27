"""LlamaSparkClient / LlamaServerManager against a fake llama-server (httpx.MockTransport).

The fake implements ``/health``, ``/props``, ``/apply-template``, ``/tokenize`` and the
streaming ``/v1/chat/completions``, records every request with its headers, and can demand
an API key. Its tokenizer is the deterministic word/punctuation rule shared by the doubles.
A request carrying ``response_format`` (Spark pass 1, query understanding) is answered with
``structured_deltas`` / ``structured_finish_reason``, every other one with ``deltas`` /
``finish_reason``; the fake applies no grammar, it only replays what the test scripted.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.base import InstrumentIdentity
from bayanalytics.pipeline.understanding import FALLBACK_NOTE as UNDERSTANDING_FALLBACK_NOTE
from bayanalytics.pipeline.understanding import SYSTEM_PROMPT as UNDERSTANDING_SYSTEM_PROMPT
from bayanalytics.pipeline.understanding import understand_question
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.questions import QueryUnderstanding, query_understanding_schema
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
from bayanalytics.wiring import build_runtime
from doubles import FixedTranscriber, RuleLaya, fixture_research_stack
from test_vertical_slice import FIXTURES, _run_to_completion

SENTINEL = "PROMPT-CONTENT-MUST-NOT-LEAK"
STRUCTURED_REST = (
    '"requirements": [], "comparison_focus": "none", "needs_benchmark": false, '
    '"needs_prior_assessment": false, "recent_period_focus": false}'
)
OK_PROBE = static_probe(available_mb=6000.0, total_mb=8192.0)
_TOKEN = re.compile(r"\w+|[^\w\s]")


def render_prompt(messages: list[dict[str, Any]]) -> str:
    """The fake server's chat template: role-tagged turns plus the assistant cue."""
    return "".join(f"<|{m['role']}|>\n{m['content']}\n" for m in messages) + "<|assistant|>\n"


def tokenize(text: str) -> list[int]:
    """The fake server's tokenizer: one id per word or punctuation mark."""
    return [index + 1 for index, _ in enumerate(_TOKEN.findall(text))]


# --- fakes ------------------------------------------------------------------------------------


class FakeProcess:
    def __init__(self, server: FakeLlamaServer, argv: list[str], env: dict[str, str]) -> None:
        self.server = server
        self.argv = argv
        self.env = dict(env)
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
        self.structured_deltas = ['{"intent": "general_assessment", ', STRUCTURED_REST]
        self.structured_finish_reason = "stop"
        self.include_usage = True
        self.include_timings = True
        self.raw_lines: list[str] | None = None
        self.process_cls: type[FakeProcess] = FakeProcess
        self.processes: list[FakeProcess] = []
        self.requests: list[dict[str, Any]] = []
        self.active = 0
        self.overlap = False
        # authentication: when set, every protected path needs "Authorization: Bearer <key>"
        self.api_key_required: str | None = None
        self.protected_paths: set[str] | None = None  # None = every path
        self.transport_error_paths: set[str] = {"/v1/chat/completions"}
        # measured prompt size endpoints
        self.template_status = 200
        self.tokenize_status = 200
        self.template_shape_ok = True
        self.tokenize_shape_ok = True
        self.template_requests: list[dict[str, Any]] = []
        self.tokenize_requests: list[dict[str, Any]] = []
        self.seen: list[tuple[str, dict[str, str]]] = []  # (path, lower-cased headers)

    async def spawn(self, argv: list[str], env: dict[str, str]) -> FakeProcess:
        process = self.process_cls(self, argv, env)
        self.processes.append(process)
        if self.exit_on_spawn:
            process._exit(1)
        else:
            self.healthy = self.auto_healthy
        return process

    async def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.seen.append((path, {k.lower(): v for k, v in request.headers.items()}))
        if self.api_key_required is not None and (
            self.protected_paths is None or path in self.protected_paths
        ):
            if request.headers.get("authorization") != f"Bearer {self.api_key_required}":
                return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})
        if self.transport_error is not None and path in self.transport_error_paths:
            raise self.transport_error("simulated", request=request)
        if path == "/apply-template":
            body = json.loads(request.content)
            self.template_requests.append(body)
            if self.template_status != 200:
                return httpx.Response(self.template_status, json={"error": {"message": "boom"}})
            if not self.template_shape_ok:
                return httpx.Response(200, json={"rendered": "nope"})
            return httpx.Response(200, json={"prompt": render_prompt(body["messages"])})
        if path == "/tokenize":
            body = json.loads(request.content)
            self.tokenize_requests.append(body)
            if self.tokenize_status != 200:
                return httpx.Response(self.tokenize_status, json={"error": {"message": "boom"}})
            if not self.tokenize_shape_ok:
                return httpx.Response(200, json={"tokens": "not a list"})
            return httpx.Response(200, json={"tokens": tokenize(body["content"])})
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
            if self.status != 200:
                return httpx.Response(self.status, json={"error": {"message": "boom"}})
            return httpx.Response(
                200,
                stream=FakeSSEStream(self, self.requests[-1]),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404)

    def sse_lines(self, body: dict[str, Any] | None = None) -> list[str]:
        if self.raw_lines is not None:
            return self.raw_lines
        structured = bool(body and "response_format" in body)
        deltas = self.structured_deltas if structured else self.deltas
        finish_reason = self.structured_finish_reason if structured else self.finish_reason
        lines = []
        for index, delta in enumerate(deltas):
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
            + json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]})
        )
        final: dict[str, Any] = {"choices": []}
        if self.include_usage:
            final["usage"] = {
                "prompt_tokens": 900,
                "completion_tokens": len(deltas),
                "total_tokens": 900 + len(deltas),
            }
        if self.include_timings:
            final["timings"] = {
                "prompt_n": 900,
                "prompt_ms": 1500.0,
                "predicted_n": len(deltas),
                "predicted_ms": 200.0,
                "predicted_per_second": 20.0,
            }
        if self.include_usage or self.include_timings:
            lines.append("data: " + json.dumps(final))
        lines.append("data: [DONE]")
        return lines

    async def _stream(self, body: dict[str, Any] | None = None) -> AsyncIterator[bytes]:
        self.active += 1
        if self.active > 1:
            self.overlap = True
        try:
            for line in self.sse_lines(body):
                yield (line + "\n\n").encode()
                await asyncio.sleep(0)
        finally:
            self.active -= 1


class FakeSSEStream(httpx.AsyncByteStream):
    """Forwards ``aclose`` to the generator so the fake sees the client close the response."""

    def __init__(self, server: FakeLlamaServer, body: dict[str, Any] | None = None) -> None:
        self._gen = server._stream(body)

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
    # A managed server always runs behind a key: it reaches the child through its environment
    # (never argv) and every request to it carries the matching bearer token.
    key = harness.manager.api_key
    assert isinstance(key, str) and len(key) >= 24
    assert server.processes[0].env == {"LLAMA_API_KEY": key}
    assert key not in " ".join(server.processes[0].argv)
    assert {path for path, _ in server.seen} >= {"/health", "/props", "/v1/chat/completions"}
    assert all(headers.get("authorization") == f"Bearer {key}" for _, headers in server.seen)


async def test_truncated_when_finish_reason_is_length(harness: Harness, server: FakeLlamaServer):
    server.finish_reason = "length"
    gen = await harness.client.run("fast", messages(), harness.on_token, harness.ctx())
    assert gen.truncated is True
    assert gen.stats.finish_reason == "length"


async def test_token_counts_are_none_without_usage(harness: Harness, server: FakeLlamaServer):
    server.include_usage = False
    server.include_timings = False
    ctx = harness.ctx()
    gen = await harness.client.run("fast", messages(), harness.on_token, ctx)
    # Without usage/timings the count is unknown: never report the delta count as measured.
    assert gen.stats.output_tokens is None
    assert ctx.diagnostics["spark_streamed_deltas"] == len(server.deltas)
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


# --- measured prompt size: the session -------------------------------------------------------


async def test_session_measures_prompt_with_the_server_template_and_tokenizer(
    harness: Harness, server: FakeLlamaServer
):
    ctx = harness.ctx()
    msgs = messages()
    payload = [m.model_dump() for m in msgs]
    async with harness.client.session("fast", ctx) as session:
        assert harness.client.busy  # the lane is held for the whole session
        assert session.spec.context_ceiling == 32768 and session.spec.name == "fast"
        count = await session.count_prompt_tokens(msgs)
        # /apply-template renders with the server's own template, /tokenize counts it with the
        # server's own tokenizer and special-token handling: the count is what a request sees.
        rendered = render_prompt(payload)
        assert count == len(tokenize(rendered))
        assert server.template_requests == [{"messages": payload}]
        assert server.tokenize_requests == [
            {"content": rendered, "add_special": True, "parse_special": True}
        ]
        assert server.requests == []  # measuring sends no completion
        shorter = await session.count_prompt_tokens(msgs[:1])
        assert shorter == len(tokenize(render_prompt(payload[:1]))) < count
        gen = await session.generate(msgs, harness.on_token)
        assert gen.text == "".join(server.deltas)
        with pytest.raises(AnalysisError) as info:
            await session.generate(msgs, harness.on_token)
        assert info.value.code == ErrorCode.INTERNAL_ERROR
    assert not harness.client.busy
    assert len(server.requests) == 1 and len(server.processes) == 1
    assert [name for name, _ in harness.events] == ["spark.loading"]  # loaded once, up front
    assert all(
        headers["authorization"] == f"Bearer {harness.manager.api_key}"
        for path, headers in server.seen
        if path in ("/apply-template", "/tokenize")
    )
    # cancelled analyses measure nothing
    cancelled = harness.ctx("an_c")
    async with harness.client.session("fast", cancelled) as session:
        cancelled.cancel.cancel()
        with pytest.raises(AnalysisError) as info:
            await session.count_prompt_tokens(msgs)
        assert info.value.code == ErrorCode.CANCELLED
    assert len(server.template_requests) == 2


@pytest.mark.parametrize(
    ("field", "status", "reason"),
    [
        ("template_status", 500, "apply_template_status"),
        ("template_status", 401, "apply_template_status"),
        ("tokenize_status", 503, "tokenize_status"),
    ],
)
async def test_apply_template_and_tokenize_statuses_map_to_inference_failed(
    harness: Harness, server: FakeLlamaServer, field: str, status: int, reason: str
):
    setattr(server, field, status)
    async with harness.client.session("fast", harness.ctx()) as session:
        with pytest.raises(AnalysisError) as info:
            await session.count_prompt_tokens(messages())
    err = info.value
    assert err.code == ErrorCode.SPARK_INFERENCE_FAILED
    assert err.details == {"reason": reason, "status": status}
    assert SENTINEL not in json.dumps(err.details) + err.message
    assert server.requests == [] and harness.manager.in_flight is False
    assert not harness.client.busy


async def test_apply_template_and_tokenize_shape_and_transport_errors(
    harness: Harness, server: FakeLlamaServer
):
    server.template_shape_ok = False
    async with harness.client.session("fast", harness.ctx()) as session:
        with pytest.raises(AnalysisError) as info:
            await session.count_prompt_tokens(messages())
    assert info.value.details == {"reason": "apply_template_shape"}
    server.template_shape_ok = True
    server.tokenize_shape_ok = False
    async with harness.client.session("fast", harness.ctx()) as session:
        with pytest.raises(AnalysisError) as info:
            await session.count_prompt_tokens(messages())
    assert info.value.details == {"reason": "tokenize_shape"}
    server.tokenize_shape_ok = True
    server.transport_error = httpx.ConnectError
    server.transport_error_paths = {"/apply-template"}
    async with harness.client.session("fast", harness.ctx()) as session:
        with pytest.raises(AnalysisError) as info:
            await session.count_prompt_tokens(messages())
    assert info.value.code == ErrorCode.SPARK_INFERENCE_FAILED
    assert info.value.details == {"reason": "ConnectError"}
    assert SENTINEL not in json.dumps(info.value.details)


# --- Spark pass 1: query understanding --------------------------------------------------------

UNDERSTANDING_IDENTITY = InstrumentIdentity(symbol="AAPL", name="Apple Inc.", cik="320193")
VALUATION_JSON = json.dumps(
    {
        "intent": "valuation",
        "requirements": ["valuation_multiples", "valuation_history"],
        "comparison_focus": "own_history",
        "needs_benchmark": False,
        "needs_prior_assessment": False,
        "recent_period_focus": False,
    }
)


def _split(text: str, parts: int = 4) -> list[str]:
    size = max(1, len(text) // parts)
    return [text[i : i + size] for i in range(0, len(text), size)]


async def _understand(harness: Harness, ctx: AnalysisContext, settings: Settings) -> Any:
    return await understand_question(
        "Assess Apple's valuation",
        UNDERSTANDING_IDENTITY,
        "multi_horizon",
        harness.client,
        "fast",
        ctx,
        settings,
    )


async def test_pass_one_request_is_schema_constrained_short_and_internal(
    harness: Harness, server: FakeLlamaServer, settings: Settings
):
    server.structured_deltas = _split(VALUATION_JSON)
    ctx = harness.ctx()
    understood = await _understand(harness, ctx, settings)
    assert understood.source == "spark"
    assert understood.understanding.requirements == ["valuation_multiples", "valuation_history"]
    (body,) = server.requests
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "query_understanding", "schema": query_understanding_schema()},
    }
    assert body["max_tokens"] == 192 and body["temperature"] == 0.0 and body["stream"] is True
    assert body["messages"][0] == {"role": "system", "content": UNDERSTANDING_SYSTEM_PROMPT}
    assert body["messages"][1]["content"].startswith("Question: Assess Apple's valuation\n")
    # internal: the model load is the only event, nothing reaches the token callback
    assert [name for name, _ in harness.events] == ["spark.loading"]
    assert harness.tokens == [] and harness.client.busy is False
    assert "spark" not in ctx.timers.elapsed_ms and ctx.timers.elapsed_ms["understanding"] > 0
    assert "spark_ttft_ms" not in ctx.diagnostics
    assert "spark_streamed_deltas" not in ctx.diagnostics
    # measured: llama-server usage for tokens, the load only when this pass loaded the profile
    stats = understood.stats
    assert stats.prompt_tokens == 900 and stats.output_tokens == len(server.structured_deltas)
    assert stats.load_ms is not None and stats.load_ms >= 0 and stats.wall_ms >= stats.load_ms
    # the wall clock splits into the wait for the lane, the load and the generation itself
    assert stats.wait_ms is not None and stats.wait_ms >= 0
    assert stats.generation_ms is not None and stats.generation_ms > 0
    assert stats.wait_ms + stats.load_ms + stats.generation_ms <= stats.wall_ms + 0.01
    again = await _understand(harness, harness.ctx("an_2"), settings)
    assert again.stats.load_ms is None  # already loaded: no load to report
    assert again.stats.generation_ms is not None and again.stats.generation_ms > 0
    # the synthesis request never carries a response format
    await harness.client.run("fast", messages(), harness.on_token, harness.ctx("an_3"))
    assert "response_format" not in server.requests[-1]


@pytest.mark.parametrize(
    ("deltas", "finish_reason"),
    [
        (['{"intent": "valuation", "requirements": ["valuation_mult'], "stop"),
        (["not json at all"], "stop"),
        ([], "stop"),
        (_split(VALUATION_JSON), "length"),
    ],
)
async def test_pass_one_falls_back_on_unusable_output(
    harness: Harness, server: FakeLlamaServer, settings: Settings, deltas, finish_reason
):
    server.structured_deltas = deltas
    server.structured_finish_reason = finish_reason
    understood = await _understand(harness, harness.ctx(), settings)
    assert understood.source == "fallback"
    assert understood.understanding == QueryUnderstanding.broad()
    assert understood.notes == [UNDERSTANDING_FALLBACK_NOTE]
    assert [name for name, _ in harness.events] == ["spark.loading"]


async def test_pass_one_token_counts_are_none_without_usage(
    harness: Harness, server: FakeLlamaServer, settings: Settings
):
    server.include_usage = False
    server.include_timings = False
    understood = await _understand(harness, harness.ctx(), settings)
    assert understood.stats.prompt_tokens is None and understood.stats.output_tokens is None
    assert understood.stats.wall_ms > 0


async def test_pass_one_runtime_errors_propagate(
    harness: Harness, server: FakeLlamaServer, settings: Settings
):
    server.status = 500
    with pytest.raises(AnalysisError) as info:
        await _understand(harness, harness.ctx(), settings)
    assert info.value.code == ErrorCode.SPARK_INFERENCE_FAILED
    assert harness.client.busy is False


async def test_pass_one_runs_between_resolution_and_research_on_its_own_session(
    harness: Harness, server: FakeLlamaServer, settings: Settings
):
    """The real client inside the orchestrator: pass 1 after instrument.resolved and before
    research.started, the Spark lock released in between, pass 2 on a second session."""
    server.structured_deltas = _split(VALUATION_JSON)
    rt = build_runtime(
        settings,
        laya=RuleLaya(),
        spark=harness.client,
        transcriber=FixedTranscriber(),
        research=fixture_research_stack(settings, FIXTURES),
    )
    lane: dict[str, bool] = {}
    publish = rt.bus.publish

    async def recording_publish(analysis_id: str, event: str, data: dict[str, Any]) -> Any:
        if event in {"instrument.resolved", "research.started"}:
            lane.setdefault(event, harness.client.busy)
        return await publish(analysis_id, event, data)

    rt.bus.publish = recording_publish  # type: ignore[method-assign]
    _id, events, result = await _run_to_completion(rt, {"query": "Assess Apple's valuation"})
    assert result["status"] == "completed", result["error"]
    names = [e["event"] for e in events]
    # pass 1: a load is visible before research, nothing else of it is streamed
    assert names.index("instrument.resolved") < names.index("spark.loading")
    assert names.index("spark.loading") < names.index("research.started")
    assert names.count("spark.started") == names.count("spark.completed") == 1
    assert names.index("research.completed") < names.index("spark.started")
    tokens = [e["data"]["text"] for e in events if e["event"] == "spark.token"]
    assert tokens == server.deltas and result["streamed_text"] == "".join(server.deltas)
    # the lane was free when research started: pass 1 held the lock only for itself
    assert lane == {"instrument.resolved": False, "research.started": False}
    assert server.overlap is False
    chats = server.requests
    assert len(chats) == 2
    assert "response_format" in chats[0] and "response_format" not in chats[1]
    assert chats[0]["max_tokens"] == 192
    assert chats[1]["max_tokens"] == settings.spark_max_output_tokens
    started = next(e["data"] for e in events if e["event"] == "research.started")
    assert started["question_intent"] == "Valuation"
    assert started["requirements"] == ["Valuation multiples", "Valuation history"]
    telemetry = result["telemetry"]
    assert telemetry["query_understanding_prompt_tokens"] == 900
    assert telemetry["query_understanding_output_tokens"] == len(server.structured_deltas)
    assert telemetry["query_understanding_load_ms"] is not None  # pass 1 loaded the profile
    assert telemetry["spark_load_ms"] is None  # so pass 2 did not
    assert telemetry["query_understanding_ms"] >= telemetry["query_understanding_load_ms"]
    assert telemetry["query_understanding_wait_ms"] is not None
    assert telemetry["query_understanding_generation_ms"] > 0
    assert telemetry["spark_prompt_tokens"] == 900  # pass 2's own usage, not pass 1's
    # without usage from the server the pass-1 counts stay None
    server.include_usage = False
    server.include_timings = False
    _id, _events, second = await _run_to_completion(rt, {"query": "Assess Apple's valuation"})
    telemetry = second["telemetry"]
    assert telemetry["query_understanding_prompt_tokens"] is None
    assert telemetry["query_understanding_output_tokens"] is None
    # (a new app lifespan stopped the server, so pass 1 loaded it again: measured, not usage)
    assert telemetry["query_understanding_load_ms"] is not None
    assert telemetry["query_understanding_ms"] is not None


# --- authentication ---------------------------------------------------------------------------


async def test_managed_server_runs_behind_a_key(settings: Settings, server: FakeLlamaServer):
    keyed = settings.model_copy(update={"spark_api_key": "configured-spark-key"})
    server.api_key_required = "configured-spark-key"
    h = Harness(keyed, server)
    try:
        assert h.manager.api_key == "configured-spark-key"
        assert h.manager.auth_headers == {"Authorization": "Bearer configured-spark-key"}
        assert h.manager.child_env_extra() == {"LLAMA_API_KEY": "configured-spark-key"}
        gen = await h.client.run("fast", messages(), h.on_token, h.ctx())
        assert gen.text == "".join(server.deltas)
        assert server.processes[0].env == {"LLAMA_API_KEY": "configured-spark-key"}
        assert "configured-spark-key" not in " ".join(server.processes[0].argv)
        assert "configured-spark-key" not in " ".join(h.manager.build_argv("deep"))
        assert {path for path, _ in server.seen} == {"/health", "/props", "/v1/chat/completions"}
        assert all(
            headers["authorization"] == "Bearer configured-spark-key" for _, headers in server.seen
        )
    finally:
        await h.aclose()
    assert Settings(spark_api_key="configured-spark-key").redacted()["spark_api_key"] == "***"
    # Without a configured key a managed server still gets one: random and per process.
    first = LlamaServerManager(settings)
    second = LlamaServerManager(settings)
    try:
        assert first.api_key and second.api_key and first.api_key != second.api_key
        assert len(first.api_key) >= 24
        assert first.auth_headers == {"Authorization": f"Bearer {first.api_key}"}
        assert first.child_env_extra() == {"LLAMA_API_KEY": first.api_key}
    finally:
        await first.aclose()
        await second.aclose()


async def test_wrong_or_missing_key_is_refused(settings: Settings, server: FakeLlamaServer):
    # Managed: the server does not accept our key, so it never passes its health check.
    server.api_key_required = "some-other-key"
    quick = settings.model_copy(update={"spark_start_timeout_s": 0.05})
    h = Harness(quick, server)
    try:
        with pytest.raises(AnalysisError) as info:
            await h.client.run("fast", messages(), h.on_token, h.ctx())
        assert info.value.code == ErrorCode.SPARK_START_FAILED
        assert info.value.details["stage"] == "health_wait"
        assert server.requests == [] and server.processes[0].returncode is not None
        assert "some-other-key" not in info.value.message
    finally:
        await h.aclose()
    # A server that is healthy but rejects the authenticated calls: structured failures that
    # never leak the prompt.
    strict = FakeLlamaServer()
    strict.api_key_required = "right-key"
    strict.protected_paths = {"/v1/chat/completions", "/apply-template", "/tokenize"}
    h2 = Harness(settings.model_copy(update={"spark_api_key": "wrong-key"}), strict)
    try:
        with pytest.raises(AnalysisError) as info:
            await h2.client.run("fast", messages(), h2.on_token, h2.ctx())
        assert info.value.code == ErrorCode.SPARK_INFERENCE_FAILED
        assert info.value.details == {"reason": "http_status", "status": 401}
        assert SENTINEL not in json.dumps(info.value.details) + info.value.message
        async with h2.client.session("fast", h2.ctx()) as session:
            with pytest.raises(AnalysisError) as info:
                await session.count_prompt_tokens(messages())
        assert info.value.details == {"reason": "apply_template_status", "status": 401}
    finally:
        await h2.aclose()


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

    async def spawn(argv: list[str], env: dict[str, str]) -> FakeProcess:
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


async def test_external_mode_uses_only_the_configured_key(
    external_settings: Settings, server: FakeLlamaServer
):
    server.healthy = True
    # No key configured: none is invented for a server this backend does not own.
    unkeyed = Harness(external_settings, server)
    try:
        assert unkeyed.manager.api_key is None
        assert unkeyed.manager.auth_headers == {} and unkeyed.manager.child_env_extra() == {}
        await unkeyed.client.start()
        assert unkeyed.client.availability("fast").available is True
        assert server.seen and all("authorization" not in h for _, h in server.seen)
        # ... so a server that demands one is simply not healthy for us.
        server.api_key_required = "external-key"
        assert await unkeyed.manager.probe_external() is False
        assert unkeyed.client.availability("fast").code == ErrorCode.SPARK_START_FAILED
    finally:
        await unkeyed.aclose()
    server.seen.clear()
    keyed = Harness(external_settings.model_copy(update={"spark_api_key": "external-key"}), server)
    try:
        assert keyed.manager.child_env_extra() == {"LLAMA_API_KEY": "external-key"}
        await keyed.client.start()
        assert keyed.client.availability("fast").available is True
        gen = await keyed.client.run("fast", messages(), keyed.on_token, keyed.ctx())
        assert gen.text == "".join(server.deltas)
        assert server.processes == []
        assert all(h["authorization"] == "Bearer external-key" for _, h in server.seen)
    finally:
        await keyed.aclose()


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
    # managed and external are the only modes the product knows
    with pytest.raises(ValueError):
        settings.model_copy(update={"spark_mode": "mock"}).model_validate(
            settings.model_copy(update={"spark_mode": "mock"}).model_dump()
        )


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
    # The artifact is named only by a verified download record, never by configuration.
    assert version_fields(None, settings) == {
        "spark_artifact": None,
        "spark_runtime": None,
        "spark_gguf_sha256": None,
        "spark_hf_revision": None,
    }
    assert version_fields({"hf_repo": "x/y"}, settings)["spark_artifact"] is None
    assert version_fields(None, settings, "b1")["spark_runtime"] == "b1"
    assert "spark_artifact" not in Settings.model_fields


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
