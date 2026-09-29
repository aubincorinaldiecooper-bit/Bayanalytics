"""Event bus, runner, horizon resolution and API routes with an in-test fake store and fake
runtimes. These tests do not depend on the store/laya/spark implementations."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import datetime, timedelta
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
from bayanalytics.schemas.results import AnalysisResult, ExecutionInfo
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

    async def list_jobs(
        self, limit: int = 50, *, before: tuple[datetime, str] | None = None
    ) -> list[AnalysisJob]:
        jobs = sorted(self.jobs.values(), key=lambda j: (j.created_at, j.analysis_id), reverse=True)
        if before is not None:
            jobs = [j for j in jobs if (j.created_at, j.analysis_id) < before]
        return [job.model_copy(deep=True) for job in jobs[:limit]]

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

    async def count_tokens(self, texts: Any) -> list[int]:
        return [len(str(t).split()) for t in texts]

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

    @contextlib.asynccontextmanager
    async def session(self, profile, ctx):
        yield _FakeSparkSession(self, profile)

    async def run(self, profile, messages, on_token, ctx, options=None) -> SparkGeneration:
        await on_token("hello")
        return SparkGeneration(
            text="hello",
            stats=SparkStreamStats(
                profile=profile, context_ceiling=32768, kv_cache_type="f16", total_ms=1
            ),
        )


class _FakeSparkSession:
    def __init__(self, spark: FakeSpark, profile: str) -> None:
        self.spec = spark.profile_spec(profile)
        self._spark = spark
        self._profile = profile

    async def count_prompt_tokens(self, messages) -> int:
        return sum(len(m.content.split()) for m in messages)

    async def generate(self, messages, on_token, options=None) -> SparkGeneration:
        return await self._spark.run(self._profile, messages, on_token, None, options)


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
    settings = Settings()
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
        ("What might happen to Shopify over the next 12 months?", "medium_term"),
        ("What might happen to NVDA next quarter?", "next_cycle"),
        ("long-term outlook for Apple", "long_term"),
        ("5-year view on Costco", "long_term"),
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


async def test_runner_reads_versions_at_submit_time() -> None:
    calls: list[int] = []

    def versions() -> dict[str, Any]:
        calls.append(1)
        return {
            "normalization_version": "n-1",
            "laya_schema_version": "l-1",
            "spark_artifact": None,  # not measured yet: never a configured label
            "spark_runtime": "b1234",
        }

    rt = _runtime(_pipeline_ok)
    rt.runner = AnalysisRunner(rt.store, rt.bus, _pipeline_ok, versions=versions)
    await rt.runner.start()
    assert calls == []  # nothing is read at construction
    job = await rt.runner.submit(
        CreateAnalysisRequest(query="Assess Apple."), resolved_horizon="multi_horizon", budget=None
    )
    assert calls == [1]
    assert job.normalization_version == "n-1" and job.laya_schema_version == "l-1"
    assert job.spark_artifact is None and job.spark_runtime == "b1234"
    [_ async for _ in rt.bus.stream(job.analysis_id)]
    stored = await rt.store.get_job(job.analysis_id)
    assert stored is not None and stored.spark_artifact is None
    # A plain dict still works, and an empty one leaves the fields empty/None.
    plain = AnalysisRunner(rt.store, rt.bus, _pipeline_ok, versions={})
    job2 = await plain.submit(
        CreateAnalysisRequest(query="Assess Apple."), resolved_horizon="near_term", budget=None
    )
    assert job2.spark_artifact is None and job2.normalization_version == ""
    [_ async for _ in rt.bus.stream(job2.analysis_id)]


async def test_admission_reserves_the_slot_before_the_store_write() -> None:
    class SuspendingStore(FakeStore):
        """``create_job`` suspends, as a real database round trip would."""

        def __init__(self) -> None:
            super().__init__()
            self.gate = asyncio.Event()

        async def create_job(self, job: AnalysisJob) -> None:
            await self.gate.wait()
            await super().create_job(job)

    store = SuspendingStore()
    bus = AnalysisEventBus(store)
    runner = AnalysisRunner(store, bus, _pipeline_ok, versions={}, max_active=4)
    await runner.start()

    async def submit() -> AnalysisJob | AnalysisError:
        try:
            return await runner.submit(
                CreateAnalysisRequest(query="Assess Apple."),
                resolved_horizon="multi_horizon",
                budget=None,
            )
        except AnalysisError as exc:
            return exc

    tasks = [asyncio.create_task(submit()) for _ in range(5)]
    await asyncio.sleep(0)  # every submit has reached the store write (or been refused)
    assert runner.active_count == 4  # admissions awaiting the write already count
    store.gate.set()
    outcomes = await asyncio.gather(*tasks)
    refused = [o for o in outcomes if isinstance(o, AnalysisError)]
    admitted = [o for o in outcomes if isinstance(o, AnalysisJob)]
    assert len(refused) == 1 and len(admitted) == 4
    assert refused[0].code == ErrorCode.TOO_MANY_ANALYSES
    assert refused[0].details == {"active": 4, "limit": 4}
    for job in admitted:
        [_ async for _ in bus.stream(job.analysis_id)]
    await asyncio.sleep(0.01)
    assert runner.active_count == 0


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


@contextlib.asynccontextmanager
async def _client(rt: Runtime) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(rt.settings, runtime=rt)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def test_api_create_stream_get_cancel() -> None:
    rt = _runtime(_pipeline_ok)
    async with _client(rt) as client:
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


async def test_api_list_analyses_paginates_newest_first() -> None:
    rt = _runtime(_pipeline_ok)
    base = utcnow()
    for i in range(3):
        await rt.store.create_job(
            AnalysisJob(
                analysis_id=f"an_{i}",
                created_at=base + timedelta(seconds=i),
                updated_at=base + timedelta(seconds=i),
                finished_at=base + timedelta(seconds=i) if i == 0 else None,
                query=f"query {i}",
                profile="fast",
                requested_horizon="auto",
                resolved_horizon="multi_horizon",
                status="completed" if i == 0 else "researching",
            )
        )
    async with _client(rt) as client:
        first = await client.get("/api/v1/analyses", params={"limit": 2})
        assert first.status_code == 200, first.text
        page = first.json()
        assert [a["analysis_id"] for a in page["analyses"]] == ["an_2", "an_1"]
        assert page["analyses"][0]["query"] == "query 2"
        assert page["analyses"][0]["profile"] == "fast"
        assert page["analyses"][0]["horizon"] == "multi_horizon"
        assert page["analyses"][0]["instrument"] is None
        assert page["analyses"][0]["completed_at"] is None
        assert page["next_cursor"]

        second = await client.get(
            "/api/v1/analyses", params={"limit": 2, "cursor": page["next_cursor"]}
        )
        tail = second.json()
        assert [a["analysis_id"] for a in tail["analyses"]] == ["an_0"]
        assert tail["analyses"][0]["status"] == "completed"
        assert tail["analyses"][0]["completed_at"] is not None
        # Last page ends the walk instead of looping on an empty page.
        assert tail["next_cursor"] is None

        bad = await client.get("/api/v1/analyses", params={"cursor": "!!not-base64!!"})
        assert bad.status_code == 422 and bad.json()["error"]["code"] == "INVALID_REQUEST"


async def test_api_list_analyses_empty_and_limit_bounds() -> None:
    rt = _runtime(_pipeline_ok)
    async with _client(rt) as client:
        empty = await client.get("/api/v1/analyses")
        assert empty.status_code == 200
        assert empty.json() == {"analyses": [], "next_cursor": None}
        assert (await client.get("/api/v1/analyses", params={"limit": 0})).status_code == 422
        assert (await client.get("/api/v1/analyses", params={"limit": 101})).status_code == 422


async def test_api_deep_unavailable_is_structured() -> None:
    rt = _runtime(_pipeline_ok, deep=False)
    async with _client(rt) as client:
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
    async with _client(rt) as client:
        bad = await client.post("/api/v1/analyses", json={"query": "   "})
        assert bad.status_code == 422 and bad.json()["error"]["code"] == "INVALID_REQUEST"
        bad2 = await client.post("/api/v1/analyses", json={"query": "x", "profile": "turbo"})
        assert bad2.status_code == 422
        tr = await client.post(
            "/api/v1/transcriptions", files={"audio": ("a.wav", b"RIFF....", "audio/wav")}
        )
        assert tr.status_code == 503 and tr.json()["error"]["code"] == "WHISPER_FAILED"
    rt2 = _runtime(_pipeline_ok, voice=True)
    async with _client(rt2) as client:
        tr = await client.post(
            "/api/v1/transcriptions", files={"audio": ("a.wav", b"RIFF....", "audio/wav")}
        )
        assert tr.status_code == 200 and tr.json()["text"] == "Assess Apple."
        empty = await client.post(
            "/api/v1/transcriptions", files={"audio": ("a.wav", b"", "audio/wav")}
        )
        assert empty.status_code == 422


async def test_result_is_readable_the_moment_the_terminal_event_arrives() -> None:
    class SlowStore(FakeStore):
        """Writes suspend, as a real database would."""

        async def save_result(self, result: AnalysisResult) -> None:
            await asyncio.sleep(0)
            await super().save_result(result)

        async def update_job(self, job: AnalysisJob) -> None:
            await asyncio.sleep(0)
            await super().update_job(job)

    store = SlowStore()
    bus = AnalysisEventBus(store)
    runner = AnalysisRunner(store, bus, _pipeline_ok, versions={})
    await runner.start()
    job = await runner.submit(
        CreateAnalysisRequest(query="Assess Apple."), resolved_horizon="multi_horizon", budget=None
    )
    seen_terminal = False
    async for event in bus.stream(job.analysis_id):
        if event.terminal:
            seen_terminal = True
            result = await store.get_result(job.analysis_id)
            assert result is not None and result.status == "completed"  # GET must not 404
            stored = await store.get_job(job.analysis_id)
            assert stored is not None and stored.status == "completed"
    assert seen_terminal


async def test_event_published_during_replay_is_delivered_exactly_once() -> None:
    class InterleavingStore(FakeStore):
        """``list_events`` suspends (a real query would), so a publish can land in between."""

        async def list_events(self, analysis_id: str, after_seq: int = 0) -> list[AnalysisEvent]:
            await asyncio.sleep(0)
            return await super().list_events(analysis_id, after_seq)

    store = InterleavingStore()
    bus = AnalysisEventBus(store)
    bus.register("an_x")
    await bus.publish("an_x", "analysis.started", {"query": "q"})

    async def collect() -> list[int]:
        return [e.seq async for e in bus.stream("an_x")]

    task = asyncio.create_task(collect())
    await asyncio.sleep(0)  # subscriber registered, replay query in flight
    await bus.publish("an_x", "spark.token", {"text": "a"})  # queued live AND in the replay
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await bus.publish("an_x", "analysis.completed", {"status": "completed"})
    assert await task == [1, 2, 3]


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


# ---------------------------------------------------------------- hardening (security review)


async def test_api_key_gate_when_configured() -> None:
    rt = _runtime(_pipeline_ok)
    rt.settings = rt.settings.model_copy(update={"api_key": "s3cret"})
    async with _client(rt) as client:
        assert (await client.get("/api/v1/health")).status_code == 401
        denied = await client.post("/api/v1/analyses", json={"query": "Assess Apple."})
        assert denied.status_code == 401 and denied.json()["error"]["code"] == "UNAUTHORIZED"
        ok = await client.get("/api/v1/health", headers={"Authorization": "Bearer s3cret"})
        assert ok.status_code == 200
        ok2 = await client.get("/api/v1/capabilities", headers={"X-API-Key": "s3cret"})
        assert ok2.status_code == 200
        wrong = await client.get("/api/v1/health", headers={"X-API-Key": "nope"})
        assert wrong.status_code == 401


async def test_refuses_non_loopback_without_api_key() -> None:
    rt = _runtime(_pipeline_ok)
    rt.settings = rt.settings.model_copy(update={"host": "0.0.0.0"})
    app = create_app(rt.settings, runtime=rt)
    with pytest.raises(RuntimeError, match="BAY_API_KEY"):
        async with app.router.lifespan_context(app):
            pass
    rt2 = _runtime(_pipeline_ok)
    rt2.settings = rt2.settings.model_copy(update={"host": "0.0.0.0", "api_key": "k"})
    app2 = create_app(rt2.settings, runtime=rt2)
    async with app2.router.lifespan_context(app2):
        pass


async def test_body_limits() -> None:
    rt = _runtime(_pipeline_ok)
    rt.settings = rt.settings.model_copy(
        update={"max_request_body_bytes": 200, "max_upload_bytes": 300}
    )
    async with _client(rt) as client:
        big = await client.post("/api/v1/analyses", json={"query": "x" * 500})
        assert big.status_code == 413 and big.json()["error"]["code"] == "INVALID_REQUEST"
        small = await client.post("/api/v1/analyses", json={"query": "Assess Apple."})
        assert small.status_code == 202
        upload = await client.post(
            "/api/v1/transcriptions", files={"audio": ("a.wav", b"x" * 1000, "audio/wav")}
        )
        assert upload.status_code == 413


async def test_admission_control_limits_concurrent_analyses() -> None:
    rt = _runtime(_pipeline_slow)
    rt.runner._max_active = 2
    async with _client(rt) as client:
        first = await client.post("/api/v1/analyses", json={"query": "Assess Apple."})
        second = await client.post("/api/v1/analyses", json={"query": "Assess Apple."})
        third = await client.post("/api/v1/analyses", json={"query": "Assess Apple."})
        assert first.status_code == 202 and second.status_code == 202
        assert third.status_code == 429
        assert third.json()["error"]["code"] == "TOO_MANY_ANALYSES"
        assert third.json()["error"]["retryable"] is True
        for resp in (first, second):
            await client.post(f"/api/v1/analyses/{resp.json()['analysis_id']}/cancel")
        await asyncio.sleep(0.1)
        again = await client.post("/api/v1/analyses", json={"query": "Assess Apple."})
        assert again.status_code == 202


async def test_bus_forgets_terminal_bookkeeping_but_replays_from_store() -> None:
    rt = _runtime(_pipeline_ok)
    await rt.runner.start()
    job = await rt.runner.submit(
        CreateAnalysisRequest(query="Assess Apple."), resolved_horizon="multi_horizon", budget=None
    )
    events = [e async for e in rt.bus.stream(job.analysis_id)]
    await asyncio.sleep(0.02)
    assert job.analysis_id not in rt.bus._seq
    assert not rt.bus.is_terminal(job.analysis_id)
    replay = [e async for e in rt.bus.stream(job.analysis_id, after_seq=0)]
    assert [e.seq for e in replay] == [e.seq for e in events]
    assert replay[-1].terminal


async def test_unhandled_exception_envelope_never_leaks_the_exception_text() -> None:
    rt = _runtime(_pipeline_ok)

    def boom(profile: str) -> ProfileCapability:
        raise RuntimeError("secret detail: /Users/me/models/spark.gguf")

    rt.spark.availability = boom  # type: ignore[method-assign]
    app = create_app(rt.settings, runtime=rt)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/analyses", json={"query": "Assess Apple."})
    assert resp.status_code == 500
    error = resp.json()["error"]
    assert error["code"] == "INTERNAL_ERROR" and error["retryable"] is True
    assert "secret" not in resp.text and "spark.gguf" not in resp.text
    assert "RuntimeError" not in error["message"]
    assert error["details"] == {"exception": "RuntimeError"}


async def test_health_overall_follows_the_worst_component() -> None:
    rt = _runtime(_pipeline_ok, voice=False)
    async with _client(rt) as client:
        health = (await client.get("/api/v1/health")).json()
        # "disabled" is a truthful state, not a degradation
        assert health["status"] == "ok"
        assert {c["name"]: c["status"] for c in health["components"]} == {
            "laya": "ok",
            "spark": "ok",
            "whisper": "disabled",
            "store": "ok",
        }
        assert health["execution"] == {
            "spark_mode": "managed",
            "whisper_mode": "disabled",
            "deployment": "local",
            "search_configured": False,
        }
        assert set(ExecutionInfo.model_fields) == set(health["execution"])
        caps = (await client.get("/api/v1/capabilities")).json()
        assert caps["execution"] == health["execution"] and caps["voice"] is False

    degraded = _runtime(_pipeline_ok)

    async def loading() -> LayaHealth:
        return LayaHealth(ok=True, loaded=False, detail="loading")

    degraded.laya.health = loading  # type: ignore[method-assign]
    async with _client(degraded) as client:
        health = (await client.get("/api/v1/health")).json()
        assert health["status"] == "degraded"
        assert next(c for c in health["components"] if c["name"] == "laya")["status"] == "degraded"

    class DownStore(FakeStore):
        async def count_active(self) -> int:
            raise ConnectionError("database gone")

    down = _runtime(_pipeline_ok)
    down.store = DownStore()
    down.bus = AnalysisEventBus(down.store)
    down.runner = AnalysisRunner(down.store, down.bus, _pipeline_ok, versions={})
    down.laya.health = loading  # type: ignore[method-assign]  # degraded AND down -> down
    async with _client(down) as client:
        health = (await client.get("/api/v1/health")).json()
        assert health["status"] == "down"
        store = next(c for c in health["components"] if c["name"] == "store")
        assert store["status"] == "down" and store["detail"] == "ConnectionError"
        assert "database gone" not in (await client.get("/api/v1/health")).text


async def test_health_is_cached_briefly() -> None:
    rt = _runtime(_pipeline_ok)
    rt.settings = rt.settings.model_copy(update={"health_cache_s": 10.0})
    calls = {"n": 0}
    original = rt.laya.health

    async def counting() -> Any:
        calls["n"] += 1
        return await original()

    rt.laya.health = counting  # type: ignore[method-assign]
    async with _client(rt) as client:
        await client.get("/api/v1/health")
        await client.get("/api/v1/health")
        await client.get("/api/v1/health")
    assert calls["n"] == 1


def test_redacted_settings_hide_paths_and_email() -> None:
    from pathlib import Path

    s = Settings(
        database_url="postgres://u:p@h/db",
        api_key="k",
        spark_model_path=Path("/Users/me/models/spark/model-q4.gguf"),
        research_contact_email="me@example.com",
    )
    red = s.redacted()
    assert red["database_url"] == "***" and red["api_key"] == "***"
    assert red["spark_model_path"] == ".../model-q4.gguf"
    assert red["research_contact_email"] == "***"


async def test_healthz_is_unauthed_but_versioned_health_stays_protected() -> None:
    rt = _runtime(_pipeline_ok)
    rt.settings = rt.settings.model_copy(update={"api_key": "s3cret"})
    async with _client(rt) as client:
        ready = await client.get("/healthz")
        assert ready.status_code == 200
        assert ready.json() == {"status": "ok"}
        assert ready.headers["cache-control"] == "no-store"
        assert (await client.get("/api/v1/health")).status_code == 401


async def test_healthz_returns_503_when_runtime_is_degraded() -> None:
    rt = _runtime(_pipeline_ok, deep=False)

    # Fast availability remains healthy in the normal test fake, so make the real readiness
    # path degraded by exposing a Laya health result that is alive but not loaded.
    async def degraded_health() -> LayaHealth:
        return LayaHealth(ok=True, loaded=False)

    rt.laya.health = degraded_health  # type: ignore[method-assign]
    async with _client(rt) as client:
        ready = await client.get("/healthz")
        assert ready.status_code == 503
        assert ready.json() == {"status": "unavailable"}
