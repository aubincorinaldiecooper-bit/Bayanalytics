"""Regression tests for the adversarial-review findings (correctness, security, completeness)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext, CancelToken
from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.base import LayaDecisions
from bayanalytics.laya.mock import MockLaya
from bayanalytics.laya.wrapper import LayaFinanceWrapper
from bayanalytics.main import create_app
from bayanalytics.pipeline.assemble import merge_horizons
from bayanalytics.runtime import Runtime
from bayanalytics.schemas.common import ErrorCode, utcnow
from bayanalytics.schemas.decisions import (
    ChoiceAnswer,
    LayaDecision,
    LayaQuestion,
    LayaQuestionSet,
    LayaResult,
    LayaUsage,
)
from bayanalytics.schemas.requests import CreateAnalysisRequest
from bayanalytics.schemas.results import HorizonAssessment
from bayanalytics.spark.bundle import OVERFLOW_TRIM
from bayanalytics.spark.mock import MockSpark
from bayanalytics.wiring import build_runtime
from test_spark_client import FakeLlamaServer, Harness, messages

FIXTURES = Path(__file__).parent / "fixtures" / "research" / "apple"


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "research_provider": "fixture",
        "research_fixture_dir": FIXTURES,
        "laya_mode": "mock",
        "spark_mode": "mock",
        "whisper_mode": "mock",
        "log_level": "WARNING",
    }
    base.update(overrides)
    return Settings(**base)


async def _client(rt: Runtime) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(rt.settings, runtime=rt)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=60) as c:
            yield c


async def _run(rt: Runtime, body: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    async for client in _client(rt):
        created = await client.post("/api/v1/analyses", json=body)
        assert created.status_code == 202, created.text
        analysis_id = created.json()["analysis_id"]
        events = [e async for e in rt.bus.stream(analysis_id)]
        await asyncio.sleep(0.02)
        return events, (await client.get(f"/api/v1/analyses/{analysis_id}")).json()
    raise AssertionError("no client")


# --------------------------------------------------------------- synthesis honesty


async def test_truncated_synthesis_is_marked_partial() -> None:
    def summary_only(_messages: Any) -> str:
        return "## Summary\nEvidence suggests signals are mixed [src_x].\n"

    rt = build_runtime(_settings(), spark=MockSpark(text_factory=summary_only))
    _events, result = await _run(rt, {"query": "Assess Apple."})
    assert result["status"] == "completed"
    assert result["partial"] is True
    for horizon in result["horizon_assessments"].values():
        assert horizon["synthesized"] is False and horizon["summary"] == ""
    assert any("no synthesis was produced" in u for u in result["assessment"]["uncertainties"])


async def test_context_overflow_is_a_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import bayanalytics.pipeline.orchestrator as orch

    def overflow(bundle: Any, *_args: Any, **_kw: Any) -> tuple[Any, list[str]]:
        return bundle, ["dropped 3 non-primary excerpts", OVERFLOW_TRIM]

    monkeypatch.setattr(orch, "fit_bundle", overflow)
    spark = MockSpark()
    rt = build_runtime(_settings(), spark=spark)
    events, result = await _run(rt, {"query": "Assess Apple."})
    assert events[-1].event == "analysis.failed"
    error = events[-1].data["error"]
    assert (
        error["code"] == "SPARK_INFERENCE_FAILED"
        and error["details"]["reason"] == "context_overflow"
    )
    assert "Deep" in error["message"]
    assert result["status"] == "failed" and not result["streamed_text"]
    assert spark.runs == []  # the over-budget prompt was never sent


async def test_result_exposes_freshness_and_serialized_source_flags() -> None:
    rt = build_runtime(_settings())
    _events, result = await _run(rt, {"query": "Assess Apple."})
    assert result["freshness_summary"]["facts"]["total"] > 0
    assert "warnings" in result["freshness_summary"]
    assert all("is_primary" in s and "rank" in s for s in result["sources"])
    assert any(s["is_primary"] for s in result["sources"])
    assert result["partial"] is False


# --------------------------------------------------------------- retrieval loop


async def test_research_loop_honours_laya_stop_and_never_repeats_an_intent() -> None:
    laya = MockLaya(force={"research_intent": "stop_research", "evidence_sufficient": 0.5})
    rt = build_runtime(_settings(), laya=laya)
    events, result = await _run(rt, {"query": "Assess Apple."})
    assert result["status"] == "completed"
    research = result["telemetry"]["research"]
    assert research["termination_reason"] in {"laya_stop", "no_new_evidence"}
    assert len(research["intents"]) == len(set(research["intents"]))
    queries = [e.data.get("label") for e in events if e.event == "research.query"]
    assert len(queries) == len(set(queries))


# --------------------------------------------------------------- shutdown and cancel semantics


async def _slow_pipeline(job: Any, ctx: AnalysisContext) -> Any:
    from bayanalytics.schemas.results import AnalysisResult

    await ctx.event("analysis.started", query=job.query)
    try:
        for _ in range(500):
            ctx.check_cancelled()
            await asyncio.sleep(0.01)
    except AnalysisError as exc:
        status = "cancelled" if exc.code == ErrorCode.CANCELLED else "failed"
        return AnalysisResult(
            analysis_id=job.analysis_id,
            status=status,
            query=job.query,
            profile=job.profile,
            horizon=job.resolved_horizon,
            as_of=job.as_of,
            created_at=job.created_at,
            completed_at=utcnow(),
            error=exc.payload(),
            partial=True,
        )
    raise AssertionError("unreachable")


async def test_shutdown_marks_running_jobs_interrupted_not_cancelled() -> None:
    from test_core_jobs_api import _runtime

    rt = _runtime(_slow_pipeline)
    await rt.runner.start()
    job = await rt.runner.submit(
        CreateAnalysisRequest(query="Assess Apple."), resolved_horizon="multi_horizon", budget=None
    )
    await asyncio.sleep(0.05)
    await rt.runner.shutdown(timeout_s=2.0)
    stored = await rt.store.get_job(job.analysis_id)
    assert stored is not None and stored.status == "failed"
    assert stored.error is not None and stored.error.code == "INTERRUPTED"
    assert stored.cancel_requested is False
    events = await rt.store.list_events(job.analysis_id)
    assert events[-1].event == "analysis.failed"
    assert events[-1].data["error"]["code"] == "INTERRUPTED"


async def test_user_cancel_keeps_the_flag_on_the_final_row() -> None:
    from test_core_jobs_api import _runtime

    rt = _runtime(_slow_pipeline)
    await rt.runner.start()
    job = await rt.runner.submit(
        CreateAnalysisRequest(query="Assess Apple."), resolved_horizon="multi_horizon", budget=None
    )
    await asyncio.sleep(0.03)
    await rt.runner.cancel(job.analysis_id)
    events = [e async for e in rt.bus.stream(job.analysis_id)]
    await asyncio.sleep(0.02)
    stored = await rt.store.get_job(job.analysis_id)
    assert stored is not None and stored.status == "cancelled" and stored.cancel_requested is True
    assert events[-1].data["error"]["code"] == "CANCELLED"


def test_cancel_token_carries_a_reason() -> None:
    token = CancelToken()
    token.cancel(ErrorCode.INTERRUPTED)
    token.cancel(ErrorCode.CANCELLED)  # first reason wins
    with pytest.raises(AnalysisError) as exc:
        token.check()
    assert exc.value.code == "INTERRUPTED"


# --------------------------------------------------------------- assembly


def _decision(horizon: str, choice: str, confidence: float) -> LayaDecision:
    probs = {k: (1 - confidence) / 3 for k in ("bullish", "neutral", "bearish", "mixed")}
    probs[choice] = confidence
    return LayaDecision(
        decision_id=f"dec_{horizon}",
        stage="horizon",
        decision_type=f"horizon_stance_{horizon}",
        question=LayaQuestion(type="choice", instructions="x", criteria=dict.fromkeys(probs, "d")),
        answer=ChoiceAnswer(choice=choice, probabilities=probs),
        confidence=confidence,
        state_digest="d",
        created_at=utcnow(),
    )


def test_merge_horizons_filters_uncited_and_reports_low_confidence() -> None:
    from bayanalytics.schemas.results import EvidenceItem

    decisions = LayaDecisions(
        decisions=[_decision("near_term", "bullish", 0.9), _decision("long_term", "bearish", 0.3)]
    )
    parsed = {
        "near_term": HorizonAssessment(
            horizon="near_term",
            summary="Near-term explanation",
            key_evidence=[
                EvidenceItem(text="cited", source_ids=["src_ok"]),
                EvidenceItem(text="uncited", source_ids=[]),
                EvidenceItem(text="unknown", source_ids=["src_ghost"]),
            ],
        )
    }
    bull = [EvidenceItem(text="b", source_ids=["src_ok"], stance="bullish")]
    out, notes = merge_horizons(["near_term", "long_term"], parsed, decisions, bull, [], {"src_ok"})
    assert [i.text for i in out["near_term"].key_evidence] == ["cited"]
    assert out["near_term"].stance == "bullish" and out["near_term"].synthesized
    assert out["long_term"].stance == "mixed" and out["long_term"].low_confidence
    assert out["long_term"].synthesized is False and out["long_term"].summary == ""
    assert any("not confident" in n for n in notes) and any("no synthesis" in n for n in notes)


# --------------------------------------------------------------- laya truncation detection


def test_wrapper_flags_truncated_laya_state() -> None:
    ctx = AnalysisContext(analysis_id="an_t")
    question_set = LayaQuestionSet(
        stage="evidence_scan",
        state={"k": "v"},
        questions={
            "a": LayaQuestion(type="noul", instructions="a?"),
            "b": LayaQuestion(type="noul", instructions="b?"),
        },
    )
    LayaFinanceWrapper._check_truncation(
        question_set, LayaResult(answers={}, usage=LayaUsage(input_tokens=2 * 512)), ctx
    )
    assert ctx.diagnostics["laya_truncated"][0]["stage"] == "evidence_scan"
    ctx2 = AnalysisContext(analysis_id="an_ok")
    LayaFinanceWrapper._check_truncation(
        question_set, LayaResult(answers={}, usage=LayaUsage(input_tokens=2 * 300)), ctx2
    )
    assert "laya_truncated" not in ctx2.diagnostics


# --------------------------------------------------------------- spark: cancel while silent


class SilentServer(FakeLlamaServer):
    """llama-server that has accepted the request but produces nothing (prompt processing)."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.closed = False

    async def _stream(self) -> AsyncIterator[bytes]:
        try:
            await self.release.wait()
            yield b"data: [DONE]\n\n"
        finally:
            self.closed = True


async def test_cancel_is_honoured_while_llama_server_is_silent(tmp_path: Path) -> None:
    model = tmp_path / "spark.gguf"
    model.write_bytes(b"GGUF")
    binary = tmp_path / "llama-server"
    binary.write_text("#!/bin/sh\n")
    settings = Settings(
        spark_mode="managed",
        spark_model_path=model,
        spark_llama_server_bin=str(binary),
        spark_start_timeout_s=2.0,
        spark_request_timeout_s=30.0,
    )
    server = SilentServer()
    harness = Harness(settings, server)
    try:
        ctx = harness.ctx()

        async def cancel_soon() -> None:
            await asyncio.sleep(0.1)
            ctx.cancel.cancel()

        canceller = asyncio.create_task(cancel_soon())
        started = time.perf_counter()
        with pytest.raises(AnalysisError) as exc:
            await harness.client.run("fast", messages(), harness.on_token, ctx)
        assert exc.value.code == "CANCELLED"
        assert time.perf_counter() - started < 2.0  # not the 30 s request timeout
        assert server.closed  # the response was closed so llama-server aborts
        assert harness.tokens == []
        # The Spark lane is free again immediately.
        assert not harness.client._lock.locked()
        await canceller
    finally:
        server.release.set()
        await harness.aclose()


# --------------------------------------------------------------- frontend-audit fixes


async def test_reconnect_with_terminal_id_closes_immediately() -> None:
    rt = build_runtime(_settings())
    async for client in _client(rt):
        created = await client.post("/api/v1/analyses", json={"query": "Assess Apple."})
        analysis_id = created.json()["analysis_id"]
        events = [e async for e in rt.bus.stream(analysis_id)]
        terminal_seq = events[-1].seq
        await asyncio.sleep(0.02)
        # What EventSource sends after the server closes at the terminal frame.
        started = time.perf_counter()
        async with client.stream(
            "GET",
            f"/api/v1/analyses/{analysis_id}/events",
            headers={"Last-Event-ID": str(terminal_seq)},
        ) as resp:
            raw = "".join([chunk async for chunk in resp.aiter_text()])
        assert time.perf_counter() - started < 1.0
        assert "event:" not in raw and "stream complete" in raw
        # Beyond the end (bogus id) also closes.
        async with client.stream(
            "GET", f"/api/v1/analyses/{analysis_id}/events", params={"after": terminal_seq + 50}
        ) as resp:
            raw2 = "".join([chunk async for chunk in resp.aiter_text()])
        assert "event:" not in raw2
        # The bus itself also refuses to wait when handed the terminal id.
        again = [e async for e in rt.bus.stream(analysis_id, after_seq=terminal_seq)]
        assert again == []


async def test_spark_queued_is_emitted_while_the_lane_is_busy() -> None:
    rt = build_runtime(_settings(), spark=MockSpark(delay_s=0.02))
    async for client in _client(rt):
        first = (await client.post("/api/v1/analyses", json={"query": "Assess Apple."})).json()
        second = (await client.post("/api/v1/analyses", json={"query": "Assess Apple."})).json()
        ev1 = [e async for e in rt.bus.stream(first["analysis_id"])]
        ev2 = [e async for e in rt.bus.stream(second["analysis_id"])]
        names2 = [e.event for e in ev2]
        assert "spark.queued" in names2
        assert names2.index("spark.queued") < names2.index("spark.started")
        assert ev1[-1].event == "analysis.completed" and ev2[-1].event == "analysis.completed"


async def test_too_many_analyses_has_retry_after_and_keepalive_setting_is_used() -> None:
    from test_core_jobs_api import _pipeline_slow, _runtime

    rt = _runtime(_pipeline_slow)
    rt.runner._max_active = 1
    rt.settings = rt.settings.model_copy(update={"sse_keepalive_s": 0.05})
    async for client in _client(rt):
        first = await client.post("/api/v1/analyses", json={"query": "Assess Apple."})
        second = await client.post("/api/v1/analyses", json={"query": "Assess Apple."})
        assert second.status_code == 429 and second.headers["retry-after"] == "5"
        analysis_id = first.json()["analysis_id"]
        await asyncio.sleep(0.2)
        await client.post(f"/api/v1/analyses/{analysis_id}/cancel")
        async with client.stream("GET", f"/api/v1/analyses/{analysis_id}/events") as resp:
            raw = "".join([chunk async for chunk in resp.aiter_text()])
        # A 0.05 s keepalive against a pipeline that ticks every 10 ms yields keepalives
        # only when nothing is published; the terminal event must still close the stream.
        assert "event: analysis.failed" in raw
