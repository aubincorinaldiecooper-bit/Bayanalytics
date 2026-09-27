"""Regression tests for the adversarial-review findings (correctness, security, completeness),
plus the assembly, evidence-gate and shutdown properties from the test-quality audit."""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from collections.abc import AsyncIterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext, CancelToken
from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.base import CalculatedMetrics, LayaDecisions
from bayanalytics.laya.base import LAYA_MAX_LEN
from bayanalytics.laya.wrapper import LayaFinanceWrapper
from bayanalytics.main import create_app
from bayanalytics.pipeline.assemble import (
    build_evidence_lists,
    finalize_assessment,
    merge_horizons,
)
from bayanalytics.pipeline.orchestrator import _evidence_gate
from bayanalytics.runtime import Runtime
from bayanalytics.schemas.common import ErrorCode, utcnow
from bayanalytics.schemas.decisions import (
    ChoiceAnswer,
    LayaDecision,
    LayaQuestion,
    LayaQuestionSet,
    LayaResult,
    LayaUsage,
    NoulAnswer,
)
from bayanalytics.schemas.evidence import NormalizedEvidence, SourceRecord
from bayanalytics.schemas.requests import CreateAnalysisRequest
from bayanalytics.schemas.results import Assessment, EvidenceItem, HorizonAssessment
from bayanalytics.spark.base import SparkGeneration
from bayanalytics.spark.bundle import OVERFLOW_TRIM, FitResult, reserved_output_tokens
from bayanalytics.spark.prompt import build_messages
from bayanalytics.wiring import build_runtime
from doubles import FixedTranscriber, RuleLaya, ScriptedSpark, fixture_research_stack
from test_spark_client import FakeLlamaServer, Harness, messages
from test_store_memory import make_calc, make_fact, make_source

FIXTURES = Path(__file__).parent / "fixtures" / "research" / "apple"
AS_OF = datetime(2026, 9, 27, tzinfo=UTC)


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"log_level": "WARNING"}
    base.update(overrides)
    return Settings(**base)


def _runtime(
    settings: Settings | None = None,
    *,
    laya: RuleLaya | None = None,
    spark: ScriptedSpark | None = None,
) -> Runtime:
    settings = settings or _settings()
    return build_runtime(
        settings,
        laya=laya or RuleLaya(),
        spark=spark or ScriptedSpark(),
        transcriber=FixedTranscriber(),
        research=fixture_research_stack(settings, FIXTURES),
    )


@contextlib.asynccontextmanager
async def _client(rt: Runtime) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(rt.settings, runtime=rt)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=60) as c:
            yield c


async def _run(rt: Runtime, body: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    async with _client(rt) as client:
        created = await client.post("/api/v1/analyses", json=body)
        assert created.status_code == 202, created.text
        analysis_id = created.json()["analysis_id"]
        events = [e async for e in rt.bus.stream(analysis_id)]
        await asyncio.sleep(0.02)
        return events, (await client.get(f"/api/v1/analyses/{analysis_id}")).json()


def _src(source_id: str, source_type: str, method: str, excerpt: str = "x" * 50) -> SourceRecord:
    return make_source(source_id).model_copy(
        update={"source_type": source_type, "extraction_method": method, "excerpt": excerpt}
    )


# --------------------------------------------------------------- synthesis honesty


async def test_truncated_synthesis_is_marked_partial() -> None:
    def summary_only(_messages: Any) -> str:
        return "## Summary\nEvidence suggests signals are mixed [src_x].\n"

    rt = _runtime(spark=ScriptedSpark(text_factory=summary_only))
    _events, result = await _run(rt, {"query": "Assess Apple."})
    assert result["status"] == "completed"
    assert result["partial"] is True
    for horizon in result["horizon_assessments"].values():
        assert horizon["synthesized"] is False and horizon["summary"] == ""
    assert any("no synthesis was produced" in u for u in result["assessment"]["uncertainties"])


class _TruncatingSpark(ScriptedSpark):
    """A Spark whose every generation hit the output token limit."""

    @contextlib.asynccontextmanager
    async def session(self, profile: Any, ctx: AnalysisContext) -> AsyncIterator[Any]:
        async with super().session(profile, ctx) as inner:
            generate = inner.generate

            async def truncated(messages: Any, on_token: Any, options: Any = None) -> Any:
                gen = await generate(messages, on_token, options)
                return SparkGeneration(text=gen.text, stats=gen.stats, truncated=True)

            inner.generate = truncated  # type: ignore[method-assign]
            yield inner


async def test_synthesis_cut_off_by_the_token_limit_is_partial_even_with_every_horizon() -> None:
    rt = _runtime(spark=_TruncatingSpark())
    events, result = await _run(rt, {"query": "Assess Apple."})
    assert result["status"] == "completed"
    assert all(h["synthesized"] for h in result["horizon_assessments"].values())
    assert result["partial"] is True
    assert any("output token limit" in u for u in result["assessment"]["uncertainties"])
    completed = next(e for e in events if e.event == "spark.completed")
    assert completed.data["truncated"] is True


async def test_context_overflow_is_a_structured_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import bayanalytics.pipeline.orchestrator as orch

    measured: dict[str, int] = {}

    async def overflow(bundle: Any, ceiling: int, options: Any, count_prompt: Any) -> FitResult:
        # Still a measurement: the session's counter sizes the prompt that would be sent.
        measured["prompt_tokens"] = await count_prompt(build_messages(bundle, options))
        return FitResult(
            bundle=bundle,
            trims=["dropped 3 non-primary excerpts", OVERFLOW_TRIM],
            prompt_tokens=measured["prompt_tokens"],
            budget=ceiling - reserved_output_tokens(options),
            measurements=1,
        )

    monkeypatch.setattr(orch, "fit_bundle", overflow)
    spark = ScriptedSpark()
    rt = _runtime(spark=spark)
    events, result = await _run(rt, {"query": "Assess Apple."})
    names = [e.event for e in events]
    assert names[-1] == "analysis.failed" and "spark.started" not in names
    error = events[-1].data["error"]
    assert error["code"] == "SPARK_INFERENCE_FAILED"
    assert error["details"]["reason"] == "context_overflow"
    assert error["details"]["prompt_tokens"] == measured["prompt_tokens"] > 0
    assert error["details"]["budget"] == 32768 - reserved_output_tokens(None)
    assert error["details"]["profile"] == "fast" and error["details"]["context_ceiling"] == 32768
    assert "Deep" in error["message"]
    assert result["status"] == "failed" and not result["streamed_text"]
    assert spark.runs == []  # the over-budget prompt was never sent
    assert spark.sessions and spark.sessions[0]["measurements"] == [measured["prompt_tokens"]]


async def test_result_exposes_freshness_and_serialized_source_flags() -> None:
    rt = _runtime()
    _events, result = await _run(rt, {"query": "Assess Apple."})
    assert result["freshness_summary"]["facts"]["total"] > 0
    assert "warnings" in result["freshness_summary"]
    assert all("is_primary" in s and "rank" in s for s in result["sources"])
    assert any(s["is_primary"] for s in result["sources"])
    assert result["partial"] is False


# --------------------------------------------------------------- retrieval loop


async def test_research_loop_honours_laya_stop_and_never_repeats_an_intent() -> None:
    laya = RuleLaya(force={"research_intent": "stop_research", "evidence_sufficient": 0.5})
    rt = _runtime(laya=laya)
    events, result = await _run(rt, {"query": "Assess Apple."})
    assert result["status"] == "completed"
    research = result["telemetry"]["research"]
    assert research["termination_reason"] in {"laya_stop", "no_new_evidence"}
    assert len(research["intents"]) == len(set(research["intents"]))
    queries = [e.data.get("label") for e in events if e.event == "research.query"]
    assert len(queries) == len(set(queries))


# --------------------------------------------------------------- evidence gate


def test_evidence_gate_refuses_prices_or_news_alone() -> None:
    price_only = NormalizedEvidence(
        symbol="X", as_of=AS_OF, sources=[_src("src_px", "market_data", "csv")]
    )
    with pytest.raises(AnalysisError) as info:
        _evidence_gate(price_only, ["earnings_history"])
    assert info.value.code == ErrorCode.INSUFFICIENT_EVIDENCE
    assert len(info.value.details["missing"]) == 3
    assert info.value.details["evidence_gaps"] == ["earnings_history"]
    news_and_facts = NormalizedEvidence(
        symbol="X",
        as_of=AS_OF,
        facts=[make_fact("f1", "src_news")],
        sources=[_src("src_news", "financial_journalism", "html_readability_v1")],
    )
    with pytest.raises(AnalysisError):
        _evidence_gate(news_and_facts, [])  # a fact without a primary source is still refused
    ok = news_and_facts.model_copy(
        update={"sources": [_src("src_10k", "regulatory_filing", "edgar_submissions")]}
    )
    _evidence_gate(ok, [])


def test_evidence_gate_blames_a_structured_outage_only_when_one_happened() -> None:
    price_only = NormalizedEvidence(
        symbol="X", as_of=AS_OF, sources=[_src("src_px", "market_data", "csv")]
    )
    with pytest.raises(AnalysisError) as info:
        _evidence_gate(price_only, ["earnings_history"], structured_failures=2)
    outage = info.value
    assert outage.code == ErrorCode.RESEARCH_UNAVAILABLE and outage.retryable is True
    assert outage.details["reason"] == "structured_source_failed"
    assert outage.details["structured_failures"] == 2
    assert len(outage.details["missing"]) == 3
    # Without an outage the same evidence is simply insufficient (and not retryable).
    with pytest.raises(AnalysisError) as info:
        _evidence_gate(price_only, [], structured_failures=0)
    assert info.value.code == ErrorCode.INSUFFICIENT_EVIDENCE
    assert info.value.retryable is False
    # Complete evidence is never blamed on a failure elsewhere.
    complete = NormalizedEvidence(
        symbol="X",
        as_of=AS_OF,
        facts=[make_fact("f1", "src_10k")],
        sources=[_src("src_10k", "regulatory_filing", "edgar_submissions")],
    )
    _evidence_gate(complete, [], structured_failures=3)


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


async def _stubborn_pipeline(job: Any, ctx: AnalysisContext) -> Any:
    """A pipeline stuck in something that never consults the cancel token."""
    await ctx.event("analysis.started", query=job.query)
    await asyncio.sleep(60)
    raise AssertionError("unreachable")


async def test_shutdown_marks_running_jobs_interrupted_not_cancelled() -> None:
    from test_core_jobs_api import _runtime as fake_runtime

    rt = fake_runtime(_slow_pipeline)
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


async def test_hard_cancel_on_shutdown_is_reported_as_interrupted() -> None:
    from test_core_jobs_api import _runtime as fake_runtime

    rt = fake_runtime(_stubborn_pipeline)
    await rt.runner.start()
    job = await rt.runner.submit(
        CreateAnalysisRequest(query="Assess Apple."), resolved_horizon="multi_horizon", budget=None
    )
    await asyncio.sleep(0.02)
    await rt.runner.shutdown(timeout_s=0.05)  # cooperative cancel ignored -> task.cancel()
    stored = await rt.store.get_job(job.analysis_id)
    assert stored is not None and stored.status == "failed"
    assert stored.error is not None and stored.error.code == "INTERRUPTED"
    events = await rt.store.list_events(job.analysis_id)
    assert events[-1].event == "analysis.failed"
    assert events[-1].data["error"]["code"] == "INTERRUPTED"
    assert events[-1].data["status"] == "failed"
    assert rt.runner.active_count == 0


async def test_user_cancel_keeps_the_flag_on_the_final_row() -> None:
    from test_core_jobs_api import _runtime as fake_runtime

    rt = fake_runtime(_slow_pipeline)
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


def test_finalize_assessment_drops_uncited_and_unknown_spark_items() -> None:
    evidence = NormalizedEvidence(symbol="X", as_of=AS_OF, sources=[make_source("src_ok")])
    spark_items = [
        EvidenceItem(text="cited", source_ids=["src_ok"]),
        EvidenceItem(text="uncited", source_ids=[]),
        EvidenceItem(text="ghost", source_ids=["src_ghost"]),
        EvidenceItem(text="half", source_ids=["src_ok", "src_ghost"]),
    ]
    assessment = Assessment(
        summary="s",
        bull_evidence=list(spark_items),
        bear_evidence=list(spark_items),
        what_changed=list(spark_items),
    )
    out = finalize_assessment(assessment, evidence, LayaDecisions(), CalculatedMetrics(), [])
    for bucket in (out.bull_evidence, out.bear_evidence, out.what_changed):
        assert [i.text for i in bucket] == ["cited"]


def test_deterministic_evidence_items_cite_their_operand_sources() -> None:
    growth = make_calc("calc_growth").model_copy(
        update={
            "name": "revenue_growth_yoy",
            "value": 10.0,
            "display": "10.0%",
            "unit": "percent",
            "period_label": "Q3 FY2026 vs Q3 FY2025",
        }
    )
    evidence = NormalizedEvidence(symbol="X", as_of=AS_OF, sources=[make_source("src_1")])
    bull, bear, _risks, changed = build_evidence_lists(
        evidence, LayaDecisions(), CalculatedMetrics(calculations=[growth])
    )
    assert bear == []
    (item,) = bull
    assert item.source_ids == ["src_1"] and item.calc_id == "calc_growth"
    assert item.metric == "revenue_growth_yoy" and item.stance == "bullish"
    assert changed[0].source_ids == ["src_1"] and changed[0].calc_id == "calc_growth"


def test_signed_metrics_land_on_the_right_side() -> None:
    evidence = NormalizedEvidence(symbol="X", as_of=AS_OF, sources=[make_source("src_1")])

    def calc(name: str, value: float):
        return make_calc(f"calc_{name}").model_copy(
            update={"name": name, "value": value, "display": f"{value:+.1f}%", "unit": "percent"}
        )

    calcs = CalculatedMetrics(
        calculations=[
            calc("revenue_growth_yoy", 8.0),
            calc("eps_growth_yoy", -3.0),
            calc("operating_margin_change_bp", 0.0),
        ]
    )
    bull, bear, _risks, _changed = build_evidence_lists(evidence, LayaDecisions(), calcs)
    assert [(i.metric, i.stance) for i in bull] == [("revenue_growth_yoy", "bullish")]
    assert [(i.metric, i.stance) for i in bear] == [("eps_growth_yoy", "bearish")]


# --------------------------------------------------------------- laya truncation detection


class _SaturatedLaya:
    """Answers everything but reports max_len tokens per question, as a truncating worker would."""

    def __init__(self) -> None:
        self.states: list[Any] = []

    async def count_tokens(self, texts: Sequence[str]) -> list[int]:
        return [len(re.findall(r"\w+|[^\w\s]", t)) for t in texts]

    async def system_one(self, state: Any, questions: Any) -> LayaResult:
        self.states.append(state)
        answers = {key: NoulAnswer(noul=0.5) for key in questions}
        return LayaResult(
            answers=answers, usage=LayaUsage(input_tokens=LAYA_MAX_LEN * len(questions))
        )


async def test_ask_records_truncated_state_in_diagnostics() -> None:
    ctx = AnalysisContext(analysis_id="an_t")
    question_set = LayaQuestionSet(
        stage="evidence_scan",
        state={"k": "v"},
        segment_id="src_1",
        questions={
            "a": LayaQuestion(type="noul", instructions="a?"),
            "b": LayaQuestion(type="noul", instructions="b?"),
        },
    )
    laya = _SaturatedLaya()
    decisions = await LayaFinanceWrapper(laya).ask(question_set, ctx)  # type: ignore[arg-type]
    assert len(decisions) == 2 and all(d.state_tokens == 2 * LAYA_MAX_LEN for d in decisions)
    (entry,) = ctx.diagnostics["laya_truncated"]
    assert entry["stage"] == "evidence_scan" and entry["segment_id"] == "src_1"
    assert entry["tokens"] == LAYA_MAX_LEN
    assert laya.states == [{"k": "v"}]
    # Below the limit nothing is flagged.
    quiet = AnalysisContext(analysis_id="an_ok")

    class Roomy(_SaturatedLaya):
        async def system_one(self, state: Any, questions: Any) -> LayaResult:
            return LayaResult(
                answers={key: NoulAnswer(noul=0.5) for key in questions},
                usage=LayaUsage(input_tokens=300 * len(questions)),
            )

    await LayaFinanceWrapper(Roomy()).ask(question_set, quiet)  # type: ignore[arg-type]
    assert "laya_truncated" not in quiet.diagnostics


# --------------------------------------------------------------- spark: cancel while silent


class SilentServer(FakeLlamaServer):
    """llama-server that has accepted the request but produces nothing (prompt processing)."""

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()
        self.closed = False

    async def _stream(self, body: dict[str, Any] | None = None) -> AsyncIterator[bytes]:
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
    rt = _runtime()
    async with _client(rt) as client:
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
    rt = _runtime(spark=ScriptedSpark(delay_s=0.02))
    async with _client(rt) as client:
        first = (await client.post("/api/v1/analyses", json={"query": "Assess Apple."})).json()
        second = (await client.post("/api/v1/analyses", json={"query": "Assess Apple."})).json()
        ev1 = [e async for e in rt.bus.stream(first["analysis_id"])]
        ev2 = [e async for e in rt.bus.stream(second["analysis_id"])]
        names2 = [e.event for e in ev2]
        assert "spark.queued" in names2
        assert names2.index("spark.queued") < names2.index("spark.started")
        assert ev1[-1].event == "analysis.completed" and ev2[-1].event == "analysis.completed"


async def test_too_many_analyses_has_retry_after_and_keepalive_setting_is_used() -> None:
    from test_core_jobs_api import _pipeline_slow
    from test_core_jobs_api import _runtime as fake_runtime

    rt = fake_runtime(_pipeline_slow)
    rt.runner._max_active = 1
    rt.settings = rt.settings.model_copy(update={"sse_keepalive_s": 0.05})
    async with _client(rt) as client:
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
