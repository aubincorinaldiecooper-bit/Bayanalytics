"""Event bus, runner, horizon resolution and API routes with an in-test fake store and fake
runtimes. These tests do not depend on the store/laya/spark implementations."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.jobs.bus import AnalysisEventBus
from bayanalytics.jobs.models import AnalysisJob
from bayanalytics.jobs.runner import AnalysisRunner
from bayanalytics.laya.base import LayaHealth, LayaLoadInfo
from bayanalytics.main import create_app
from bayanalytics.pipeline.horizon import detect_horizons, horizons_for, resolve_horizon
from bayanalytics.runtime import Runtime
from bayanalytics.schemas.calculations import CalculationResult
from bayanalytics.schemas.capabilities import ProfileCapability
from bayanalytics.schemas.common import ErrorCode, utcnow
from bayanalytics.schemas.decisions import LayaDecision, LayaResult
from bayanalytics.schemas.events import AnalysisEvent
from bayanalytics.schemas.evidence import NormalizedFact, SourceRecord
from bayanalytics.schemas.requests import CreateAnalysisRequest
from bayanalytics.schemas.results import AnalysisResult
from bayanalytics.schemas.transcriptions import Transcription
from bayanalytics.spark.base import ProfileSpec, SparkGeneration, SparkStreamStats


class FakeStore:
    def __init__(self) -> None:
        self.jobs: dict[str, AnalysisJob] = {}
        self.events: dict[str, list[AnalysisEvent]] = {}
        self.results: dict[str, AnalysisResult] = {}

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def create_job(self, job: AnalysisJob) -> None:
        self.jobs[job.analysis_id] = job.model_copy(deep=True)

    async def update_job(self, job: AnalysisJob) -> None:
        self.jobs[job.analysis_id] = job.model_copy(deep=True)

    async def get_job(self, analysis_id: str) -> AnalysisJob | None:
        job = self.jobs.get(analysis_id)
        return job.model_copy(deep=True) if job else None

    async def append_event(self, event: AnalysisEvent) -> None:
        self.events.setdefault(event.analysis_id, []).append(event)

    async def list_events(self, analysis_id: str, after_seq: int = 0) -> list[AnalysisEvent]:
        return [e for e in self.events.get(analysis_id, []) if e.seq > after_seq]

    async def save_sources(self, analysis_id: str, sources: list[SourceRecord]) -> None: ...

    async def save_facts(self, analysis_id: str, facts: list[NormalizedFact]) -> None: ...

    async def save_decisions(self, analysis_id: str, decisions: list[LayaDecision]) -> None: ...

    async def save_calculations(
        self, analysis_id: str, calculations: list[CalculationResult]
    ) -> None: ...

    async def save_result(self, result: AnalysisResult) -> None:
        self.results[result.analysis_id] = result.model_copy(deep=True)

    async def get_result(self, analysis_id: str) -> AnalysisResult | None:
        return self.results.get(analysis_id)

    async def mark_interrupted(self) -> list[str]:
        ids = []
        for job in self.jobs.values():
            if not job.terminal:
                job.status = "failed"
                job.error = AnalysisError(ErrorCode.INTERRUPTED).payload()
                ids.append(job.analysis_id)
        return ids


class FakeLaya:
    async def load(self) -> LayaLoadInfo:
        return LayaLoadInfo(load_ms=1.0)

    async def system_one(self, state: Any, questions: Any) -> LayaResult:
        return LayaResult(answers={})

    async def health(self) -> LayaHealth:
        return LayaHealth(ok=True, loaded=True)

    async def close(self) -> None: ...


class FakeSpark:
    def __init__(self, deep: bool = True) -> None:
        self.deep = deep

    async def start(self) -> None: ...

    async def close(self) -> None: ...

    def availability(self, profile: str) -> ProfileCapability:
        if profile == "deep" and not self.deep:
            return ProfileCapability(
                available=False,
                context_ceiling=131072,
                reason="not enough memory",
                code=ErrorCode.DEEP_PROFILE_UNAVAILABLE,
            )
        return ProfileCapability(
            available=True, context_ceiling=32768 if profile == "fast" else 131072
        )

    def profile_spec(self, profile: str) -> ProfileSpec:
        return ProfileSpec(
            name=profile, context_ceiling=32768, kv_cache_type="f16", min_available_mb=0
        )  # type: ignore[arg-type]

    async def run(self, profile, messages, on_token, ctx, options=None) -> SparkGeneration:
        await on_token("hello")
        return SparkGeneration(
            text="hello",
            stats=SparkStreamStats(
                profile=profile, context_ceiling=32768, kv_cache_type="f16", total_ms=1
            ),
        )


class FakeTranscriber:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    def available(self) -> bool:
        return self.ok

    async def transcribe(self, audio: bytes, filename: str, content_type=None) -> Transcription:
        return Transcription(text="Assess Apple.", duration_ms=1000, transcription_ms=10)

    async def close(self) -> None: ...


def _result(job: AnalysisJob, status: str = "completed", **kw: Any) -> AnalysisResult:
    return AnalysisResult(
        analysis_id=job.analysis_id,
        status=status,  # type: ignore[arg-type]
        query=job.query,
        profile=job.profile,
        horizon=job.resolved_horizon,
        as_of=job.as_of,
        created_at=job.created_at,
        completed_at=utcnow(),
        **kw,
    )


async def _pipeline_ok(job: AnalysisJob, ctx: AnalysisContext) -> AnalysisResult:
    await ctx.event("analysis.started", query=job.query, profile=job.profile)
    await ctx.event("spark.token", text="hel")
    await ctx.event("spark.token", text="lo")
    return _result(job, streamed_text="hello")


async def _pipeline_slow(job: AnalysisJob, ctx: AnalysisContext) -> AnalysisResult:
    await ctx.event("analysis.started", query=job.query, profile=job.profile)
    try:
        for _ in range(200):
            ctx.check_cancelled()
            await asyncio.sleep(0.01)
    except AnalysisError as exc:
        return _result(job, "cancelled", error=exc.payload(), partial=True)
    return _result(job)


async def _pipeline_raises(job: AnalysisJob, ctx: AnalysisContext) -> AnalysisResult:
    raise RuntimeError("boom")


def _runtime(pipeline, *, deep: bool = True, voice: bool = True) -> Runtime:
    settings = Settings(laya_mode="mock", spark_mode="mock", whisper_mode="mock")
    store = FakeStore()
    bus = AnalysisEventBus(store)
    runner = AnalysisRunner(store, bus, pipeline, versions={"normalization_version": "t"})
    return Runtime(
        settings=settings,
        store=store,
        bus=bus,
        runner=runner,
        laya=FakeLaya(),
        spark=FakeSpark(deep=deep),
        transcriber=FakeTranscriber(voice),
    )


# ---------------------------------------------------------------- horizon


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Assess Apple.", "multi_horizon"),
        ("What might happen to Shopify over the next 12 months?", "multi_horizon"),
        ("Is Microsoft expensive relative to its history?", "multi_horizon"),
        ("How will NVIDIA trade this week?", "near_term"),
        ("What should I expect from Apple's next earnings?", "next_cycle"),
        ("Where could Tesla be in 12 months?", "medium_term"),
        ("Is Costco a good long-term compounder?", "long_term"),
        ("What's next for Meta?", "multi_horizon"),
    ],
)
def test_resolve_horizon(query: str, expected: str) -> None:
    assert resolve_horizon(query, "auto") == expected


def test_explicit_horizon_wins() -> None:
    assert resolve_horizon("this week", "long_term") == "long_term"
    assert detect_horizons("next quarter and next decade") == ["next_cycle", "long_term"]
    assert resolve_horizon("next quarter and next decade") == "multi_horizon"
    assert horizons_for("multi_horizon") == ["near_term", "next_cycle", "medium_term", "long_term"]
    assert horizons_for("near_term") == ["near_term"]


# ---------------------------------------------------------------- bus + runner


async def test_bus_replay_then_live() -> None:
    store = FakeStore()
    bus = AnalysisEventBus(store)
    bus.register("an_x")
    await bus.publish("an_x", "analysis.started", {"query": "q"})
    await bus.publish("an_x", "spark.token", {"text": "a"})

    async def collect(after: int) -> list[AnalysisEvent]:
        out = []
        async for ev in bus.stream("an_x", after_seq=after):
            out.append(ev)
        return out

    task = asyncio.create_task(collect(1))
    await asyncio.sleep(0.01)
    await bus.publish("an_x", "spark.token", {"text": "b"})
    await bus.publish("an_x", "analysis.completed", {"status": "completed"})
    events = await task
    assert [e.seq for e in events] == [2, 3, 4]
    assert events[-1].terminal
    with pytest.raises(RuntimeError):
        await bus.publish("an_x", "spark.token", {"text": "late"})
    # Replay after terminal returns the persisted tail and closes.
    assert [e.seq for e in await collect(0)] == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        await bus.publish("an_y", "not.an.event", {})


async def test_runner_completes_and_persists() -> None:
    rt = _runtime(_pipeline_ok)
    await rt.runner.start()
    job = await rt.runner.submit(
        CreateAnalysisRequest(query="Assess Apple."), resolved_horizon="multi_horizon", budget=None
    )
    assert job.status == "queued"
    events = [e async for e in rt.bus.stream(job.analysis_id)]
    assert [e.event for e in events] == [
        "analysis.started",
        "spark.token",
        "spark.token",
        "analysis.completed",
    ]
    await asyncio.sleep(0.01)
    stored = await rt.store.get_job(job.analysis_id)
    assert stored is not None and stored.status == "completed" and stored.last_seq == 4
    result = await rt.store.get_result(job.analysis_id)
    assert result is not None and result.streamed_text == "hello"
    assert rt.runner.active_count == 0


async def test_runner_cancel_marks_cancelled_and_emits_failed() -> None:
    rt = _runtime(_pipeline_slow)
    await rt.runner.start()
    job = await rt.runner.submit(
        CreateAnalysisRequest(query="Assess Apple."), resolved_horizon="multi_horizon", budget=None
    )
    await asyncio.sleep(0.05)
    cancelled = await rt.runner.cancel(job.analysis_id)
    assert cancelled is not None and cancelled.cancel_requested
    events = [e async for e in rt.bus.stream(job.analysis_id)]
    assert events[-1].event == "analysis.failed"
    assert events[-1].data["error"]["code"] == "CANCELLED"
    assert events[-1].data["status"] == "cancelled"
    await asyncio.sleep(0.01)
    result = await rt.store.get_result(job.analysis_id)
    assert result is not None and result.status == "cancelled" and result.partial
    assert await rt.runner.cancel("an_missing") is None


async def test_runner_contract_violation_becomes_internal_error() -> None:
    rt = _runtime(_pipeline_raises)
    await rt.runner.start()
    job = await rt.runner.submit(
        CreateAnalysisRequest(query="Assess Apple."), resolved_horizon="multi_horizon", budget=None
    )
    events = [e async for e in rt.bus.stream(job.analysis_id)]
    assert events[-1].event == "analysis.failed"
    assert events[-1].data["error"]["code"] == "INTERNAL_ERROR"


async def test_runner_marks_interrupted_on_start() -> None:
    rt = _runtime(_pipeline_ok)
    stale = AnalysisJob(
        analysis_id="an_stale",
        query="q",
        profile="fast",
        requested_horizon="auto",
        resolved_horizon="multi_horizon",
        status="researching",
    )
    await rt.store.create_job(stale)
    assert await rt.runner.start() == ["an_stale"]
    job = await rt.store.get_job("an_stale")
    assert (
        job is not None and job.status == "failed" and job.error and job.error.code == "INTERRUPTED"
    )


# ---------------------------------------------------------------- API


async def _client(rt: Runtime) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(rt.settings, runtime=rt)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def test_api_create_stream_get_cancel() -> None:
    rt = _runtime(_pipeline_ok)
    async for client in _client(rt):
        health = await client.get("/api/v1/health")
        assert health.status_code == 200 and health.json()["status"] == "ok"
        caps = await client.get("/api/v1/capabilities")
        assert caps.json()["profiles"]["deep"]["available"] is True
        assert caps.json()["voice"] is True

        created = await client.post(
            "/api/v1/analyses", json={"query": "Assess Apple.", "profile": "fast"}
        )
        assert created.status_code == 202, created.text
        body = created.json()
        assert body["status"] == "queued" and body["resolved_horizon"] == "multi_horizon"
        analysis_id = body["analysis_id"]

        async with client.stream("GET", f"/api/v1/analyses/{analysis_id}/events") as resp:
            assert resp.headers["content-type"].startswith("text/event-stream")
            raw = "".join([chunk async for chunk in resp.aiter_text()])
        assert "event: analysis.started" in raw
        assert raw.count("event: spark.token") == 2
        assert "event: analysis.completed" in raw
        assert "id: 4" in raw

        # Reconnect with Last-Event-ID replays only the tail.
        async with client.stream(
            "GET", f"/api/v1/analyses/{analysis_id}/events", headers={"Last-Event-ID": "3"}
        ) as resp:
            raw2 = "".join([chunk async for chunk in resp.aiter_text()])
        assert "event: spark.token" not in raw2 and "event: analysis.completed" in raw2

        got = await client.get(f"/api/v1/analyses/{analysis_id}")
        assert got.status_code == 200 and got.json()["status"] == "completed"
        assert got.json()["streamed_text"] == "hello"

        cancel = await client.post(f"/api/v1/analyses/{analysis_id}/cancel")
        assert cancel.status_code == 200 and cancel.json()["cancel_requested"] is False

        missing = await client.get("/api/v1/analyses/an_missing")
        assert missing.status_code == 404 and missing.json()["error"]["code"] == "NOT_FOUND"
        missing_ev = await client.get("/api/v1/analyses/an_missing/events")
        assert missing_ev.status_code == 404


async def test_api_deep_unavailable_is_structured() -> None:
    rt = _runtime(_pipeline_ok, deep=False)
    async for client in _client(rt):
        caps = await client.get("/api/v1/capabilities")
        deep = caps.json()["profiles"]["deep"]
        assert deep["available"] is False and deep["code"] == "DEEP_PROFILE_UNAVAILABLE"
        created = await client.post(
            "/api/v1/analyses", json={"query": "Assess Apple.", "profile": "deep"}
        )
        assert created.status_code == 503
        assert created.json()["error"]["code"] == "DEEP_PROFILE_UNAVAILABLE"
        assert created.json()["error"]["retryable"] is True


async def test_api_validation_and_transcription() -> None:
    rt = _runtime(_pipeline_ok, voice=False)
    async for client in _client(rt):
        bad = await client.post("/api/v1/analyses", json={"query": "   "})
        assert bad.status_code == 422 and bad.json()["error"]["code"] == "INVALID_REQUEST"
        bad2 = await client.post("/api/v1/analyses", json={"query": "x", "profile": "turbo"})
        assert bad2.status_code == 422
        tr = await client.post(
            "/api/v1/transcriptions", files={"audio": ("a.wav", b"RIFF....", "audio/wav")}
        )
        assert tr.status_code == 503 and tr.json()["error"]["code"] == "WHISPER_FAILED"
    rt2 = _runtime(_pipeline_ok, voice=True)
    async for client in _client(rt2):
        tr = await client.post(
            "/api/v1/transcriptions", files={"audio": ("a.wav", b"RIFF....", "audio/wav")}
        )
        assert tr.status_code == 200 and tr.json()["text"] == "Assess Apple."
        empty = await client.post(
            "/api/v1/transcriptions", files={"audio": ("a.wav", b"", "audio/wav")}
        )
        assert empty.status_code == 422


# ---------------------------------------------------------------- regressions (PR review)


async def test_sse_keepalive_does_not_close_the_stream() -> None:
    """A quiet stretch longer than the keepalive interval must not end the stream."""
    from bayanalytics.api.sse import event_stream

    store = FakeStore()
    bus = AnalysisEventBus(store)
    bus.register("an_slow")

    async def publisher() -> None:
        await bus.publish("an_slow", "analysis.started", {"query": "q"})
        await asyncio.sleep(0.12)  # longer than two keepalive intervals
        await bus.publish("an_slow", "spark.token", {"text": "late"})
        await asyncio.sleep(0.07)
        await bus.publish("an_slow", "analysis.completed", {"status": "completed"})

    task = asyncio.create_task(publisher())
    frames = [frame async for frame in event_stream(bus, "an_slow", 0, keepalive_s=0.05)]
    await task
    assert frames.count(": keepalive\n\n") >= 2
    events = [f for f in frames if f.startswith("id: ")]
    assert "event: spark.token" in events[1]
    assert "event: analysis.completed" in events[-1]
    assert len(events) == 3


async def test_interrupted_jobs_get_a_terminal_event_on_startup() -> None:
    rt = _runtime(_pipeline_ok)
    stale = AnalysisJob(
        analysis_id="an_stale2",
        query="q",
        profile="fast",
        requested_horizon="auto",
        resolved_horizon="multi_horizon",
        status="synthesizing",
    )
    await rt.store.create_job(stale)
    await rt.store.append_event(
        AnalysisEvent(event="analysis.started", analysis_id="an_stale2", seq=1, data={})
    )
    await rt.store.append_event(
        AnalysisEvent(event="spark.token", analysis_id="an_stale2", seq=2, data={"text": "x"})
    )
    assert await rt.runner.start() == ["an_stale2"]
    events = await rt.store.list_events("an_stale2")
    assert [e.seq for e in events] == [1, 2, 3]
    assert events[-1].event == "analysis.failed"
    assert events[-1].data["error"]["code"] == "INTERRUPTED"
    job = await rt.store.get_job("an_stale2")
    assert job is not None and job.status == "failed" and job.last_seq == 3
    # A reconnecting client replays and terminates instead of hanging.
    replayed = [e async for e in rt.bus.stream("an_stale2", after_seq=2)]
    assert [e.event for e in replayed] == ["analysis.failed"]
    # Starting again is idempotent: no second terminal event.
    await rt.runner.start()
    assert len(await rt.store.list_events("an_stale2")) == 3
