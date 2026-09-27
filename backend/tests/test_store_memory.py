"""InMemoryStore behaviour. The ``check_*`` coroutines are backend-agnostic and are reused by
``test_store_postgres.py`` against a real database when ``BAY_TEST_DATABASE_URL`` is set."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from bayanalytics.errors import default_message
from bayanalytics.jobs.models import AnalysisJob
from bayanalytics.schemas.calculations import CalculationInput, CalculationResult
from bayanalytics.schemas.common import ErrorCode, new_id, utcnow
from bayanalytics.schemas.decisions import ChoiceAnswer, LayaDecision, LayaQuestion
from bayanalytics.schemas.events import AnalysisEvent
from bayanalytics.schemas.evidence import NormalizedFact, Period, SourceRecord
from bayanalytics.schemas.results import AnalysisResult, Telemetry
from bayanalytics.store import InMemoryStore, interrupted_payload

# --- model factories ------------------------------------------------------------------------


def make_job(
    analysis_id: str | None = None, status: str = "queued", **overrides: Any
) -> AnalysisJob:
    payload: dict[str, Any] = {
        "analysis_id": analysis_id or new_id("an"),
        "query": "Assess Apple.",
        "profile": "fast",
        "requested_horizon": "auto",
        "resolved_horizon": "near_term",
        "status": status,
        "normalization_version": "2026.09-1",
        "laya_schema_version": "finance-v1",
        "spark_artifact": "XHToken/Spark-X2.5-1.7B-GGUF:Q4_K_M",
    }
    payload.update(overrides)
    return AnalysisJob(**payload)


def make_event(
    analysis_id: str, seq: int, event: str = "research.query", **data: Any
) -> AnalysisEvent:
    return AnalysisEvent(event=event, analysis_id=analysis_id, seq=seq, data=data)


def make_source(source_id: str, content_hash: str | None = "abc123") -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        url=f"https://www.sec.gov/{source_id}",
        title=f"Filing {source_id}",
        publisher="SEC",
        source_type="regulatory_filing",
        published_at=datetime(2026, 8, 1, tzinfo=UTC),
        retrieved_at=utcnow(),
        symbol="AAPL",
        fiscal_period="Q3 FY2026",
        excerpt="Net sales were $94.0 billion.",
        content_hash=content_hash,
        extraction_method="xbrl",
        freshness="current",
    )


def make_fact(fact_id: str, source_id: str, value: float = 94.0e9) -> NormalizedFact:
    return NormalizedFact(
        fact_id=fact_id,
        metric="revenue",
        value=value,
        unit="USD",
        currency="USD",
        period=Period(kind="fiscal_quarter", fiscal_year=2026, fiscal_period="Q3"),
        basis="gaap",
        source_id=source_id,
        raw_value="94,036",
        extraction_method="xbrl",
        published_at=datetime(2026, 8, 1, tzinfo=UTC),
    )


def make_decision(decision_id: str, stage: str = "horizon") -> LayaDecision:
    return LayaDecision(
        decision_id=decision_id,
        stage=stage,
        decision_type="stance",
        question=LayaQuestion(
            type="choice",
            instructions="Pick the stance.",
            criteria={"bullish": "up", "bearish": "down"},
        ),
        answer=ChoiceAnswer(choice="bullish", probabilities={"bullish": 0.7, "bearish": 0.3}),
        confidence=0.7,
        state_digest="sha256:deadbeef",
        state_tokens=120,
        created_at=utcnow(),
        schema_version="finance-v1",
    )


def make_calc(calc_id: str, value: float | None = 28.5) -> CalculationResult:
    return CalculationResult(
        calc_id=calc_id,
        name="pe_ttm",
        formula="price / eps_ttm",
        inputs=[CalculationInput(name="price", value=228.0, unit="USD", source_id="src_1")],
        value=value,
        unit="ratio",
        period_label="TTM",
        status="computed" if value is not None else "unavailable",
        missing_inputs=[] if value is not None else ["eps_ttm"],
        display=f"{value:.1f}x" if value is not None else "n/a",
    )


def make_result(job: AnalysisJob, status: str = "completed") -> AnalysisResult:
    return AnalysisResult(
        analysis_id=job.analysis_id,
        status=status,
        query=job.query,
        profile=job.profile,
        horizon=job.resolved_horizon,
        as_of=job.as_of,
        created_at=job.created_at,
        completed_at=utcnow(),
        sources=[make_source("src_1")],
        calculations=[make_calc("calc_1")],
        laya_decisions=[make_decision("dec_1")],
        streamed_text="Apple looks ...",
        telemetry=Telemetry(profile="fast", retrieval_ms=1200.5, process_peak_rss_mb=512.25),
    )


# --- backend-agnostic checks -----------------------------------------------------------------


async def check_job_round_trip(store: Any) -> None:
    job = make_job()
    await store.create_job(job)
    stored = await store.get_job(job.analysis_id)
    assert stored is not None
    assert stored == job
    assert stored.created_at.tzinfo is not None
    assert stored.as_of.tzinfo is not None

    job.status = "researching"
    job.started_at = utcnow()
    job.source_ids = ["src_1", "src_2"]
    job.last_seq = 4
    job.telemetry = {"retrieval_ms": 10.5}
    job.touch()
    await store.update_job(job)
    updated = await store.get_job(job.analysis_id)
    assert updated is not None
    assert updated.status == "researching"
    assert updated.source_ids == ["src_1", "src_2"]
    assert updated.last_seq == 4
    assert updated.telemetry == {"retrieval_ms": 10.5}
    assert updated.started_at is not None and updated.started_at.tzinfo is not None
    assert await store.get_job("an_does_not_exist") is None


async def check_list_jobs_keyset_paging(store: Any) -> None:
    # Back-dated, and every page here is read through a cursor that sits just above them, so
    # the check is unaffected by rows other tests left in a shared database.
    base = datetime(2026, 9, 1, 12, tzinfo=UTC)
    prefix = f"an_page_{new_id('x')[-6:]}_"
    # The two middle jobs share a timestamp: the analysis_id half of the key is what keeps
    # paging from skipping or repeating one of them.
    ids = [f"{prefix}{suffix}" for suffix in ("a", "b", "c", "d")]
    stamps = (
        base,
        base + timedelta(minutes=1),
        base + timedelta(minutes=1),
        base + timedelta(minutes=2),
    )
    for analysis_id, created_at in zip(ids, stamps, strict=True):
        await store.create_job(make_job(analysis_id, created_at=created_at, updated_at=created_at))

    # Walk pages of two, following the cursor the way the API does, starting just above the
    # newest of these jobs. Other rows may be interleaved, so the assertion is about these
    # four: each is seen exactly once, newest first, and no row is repeated across pages.
    cursor: tuple[datetime, str] | None = (base + timedelta(minutes=3), "an_")
    seen: list[str] = []
    for _ in range(20):
        page = await store.list_jobs(2, before=cursor)
        if not page:
            break
        seen.extend(job.analysis_id for job in page)
        last = page[-1]
        cursor = (last.created_at, last.analysis_id)
        if set(ids) <= set(seen):
            break

    assert len(seen) == len(set(seen))
    assert [analysis_id for analysis_id in seen if analysis_id in ids] == list(reversed(ids))


async def check_deep_copy_isolation(store: Any) -> None:
    job = make_job()
    await store.create_job(job)
    job.status = "failed"
    job.source_ids.append("mutated")
    job.telemetry["mutated"] = True
    stored = await store.get_job(job.analysis_id)
    assert stored is not None
    assert stored.status == "queued"
    assert stored.source_ids == []
    assert stored.telemetry == {}
    stored.source_ids.append("again")
    stored.telemetry["x"] = 1
    again = await store.get_job(job.analysis_id)
    assert again is not None
    assert again.source_ids == []
    assert again.telemetry == {}


async def check_events(store: Any) -> None:
    job = make_job()
    await store.create_job(job)
    aid = job.analysis_id
    await store.append_event(make_event(aid, 1, "analysis.started", profile="fast"))
    await store.append_event(make_event(aid, 3, "research.started"))
    await store.append_event(make_event(aid, 2, "instrument.resolved", symbol="AAPL"))
    events = await store.list_events(aid)
    assert [e.seq for e in events] == [1, 2, 3]
    assert [e.event for e in events] == [
        "analysis.started",
        "instrument.resolved",
        "research.started",
    ]
    assert events[1].data == {"symbol": "AAPL"}
    assert all(e.ts.tzinfo is not None for e in events)
    assert [e.seq for e in await store.list_events(aid, after_seq=1)] == [2, 3]
    assert await store.list_events(aid, after_seq=3) == []
    assert await store.list_events("an_unknown") == []


async def check_duplicate_seq_rejected(store: Any) -> None:
    job = make_job()
    await store.create_job(job)
    await store.append_event(make_event(job.analysis_id, 1))
    with pytest.raises(ValueError, match="duplicate"):
        await store.append_event(make_event(job.analysis_id, 1, "research.completed"))
    events = await store.list_events(job.analysis_id)
    assert len(events) == 1 and events[0].event == "research.query"


async def check_result_round_trip(store: Any) -> None:
    job = make_job()
    await store.create_job(job)
    assert await store.get_result(job.analysis_id) is None
    result = make_result(job)
    await store.save_result(result)
    stored = await store.get_result(job.analysis_id)
    assert stored == result
    assert stored is not None and stored.completed_at is not None
    assert stored.completed_at.tzinfo is not None
    assert stored.telemetry.process_peak_rss_mb == 512.25
    # overwrite by id
    result.status = "failed"
    result.partial = True
    await store.save_result(result)
    again = await store.get_result(job.analysis_id)
    assert again is not None and again.status == "failed" and again.partial is True


async def check_records_round_trip(store: Any) -> None:
    job = make_job()
    await store.create_job(job)
    aid = job.analysis_id
    await store.save_sources(aid, [make_source("src_1"), make_source("src_2", None)])
    await store.save_sources(aid, [make_source("src_1", "newhash")])  # upsert
    sources = {s.source_id: s for s in await store.get_sources(aid)}
    assert set(sources) == {"src_1", "src_2"}
    assert sources["src_1"].content_hash == "newhash"
    assert sources["src_2"].content_hash is None
    assert sources["src_1"].retrieved_at.tzinfo is not None

    await store.save_facts(aid, [make_fact("fact_1", "src_1"), make_fact("fact_2", "src_2", 1.5)])
    facts = {f.fact_id: f for f in await store.get_facts(aid)}
    assert facts["fact_1"].period.fiscal_period == "Q3"
    assert facts["fact_2"].value == 1.5

    await store.save_decisions(aid, [make_decision("dec_1")])
    decisions = await store.get_decisions(aid)
    assert len(decisions) == 1 and decisions[0].decision == "bullish"
    assert decisions[0].created_at.tzinfo is not None

    await store.save_calculations(aid, [make_calc("calc_1"), make_calc("calc_2", None)])
    calcs = {c.calc_id: c for c in await store.get_calculations(aid)}
    assert calcs["calc_1"].value == 28.5
    assert calcs["calc_2"].status == "unavailable" and calcs["calc_2"].value is None

    # empty batches are no-ops
    await store.save_sources(aid, [])
    await store.save_facts(aid, [])
    await store.save_decisions(aid, [])
    await store.save_calculations(aid, [])
    assert len(await store.get_sources(aid)) == 2


async def check_mark_interrupted(store: Any) -> None:
    queued = make_job(status="queued")
    running = make_job(status="synthesizing")
    completed = make_job(status="completed", finished_at=utcnow())
    failed = make_job(status="failed", finished_at=utcnow())
    cancelled = make_job(status="cancelled", finished_at=utcnow())
    for job in (queued, running, completed, failed, cancelled):
        await store.create_job(job)

    affected = await store.mark_interrupted()
    assert {queued.analysis_id, running.analysis_id} <= set(affected)
    assert not {completed.analysis_id, failed.analysis_id, cancelled.analysis_id} & set(affected)

    for job in (queued, running):
        stored = await store.get_job(job.analysis_id)
        assert stored is not None
        assert stored.status == "failed"
        assert stored.error is not None
        assert stored.error.code == ErrorCode.INTERRUPTED
        assert stored.error.message == default_message(ErrorCode.INTERRUPTED)
        assert stored.error.retryable is True
        assert stored.finished_at is not None and stored.finished_at.tzinfo is not None
        assert stored.updated_at >= job.updated_at
        assert stored.query == job.query  # the rest of the snapshot is untouched
    for job in (completed, failed, cancelled):
        stored = await store.get_job(job.analysis_id)
        assert stored is not None
        assert stored.status == job.status
        assert stored.error is None

    # idempotent: a second pass finds nothing of ours
    assert not {queued.analysis_id, running.analysis_id} & set(await store.mark_interrupted())


# --- InMemoryStore -----------------------------------------------------------------------------


@pytest.fixture
async def store() -> InMemoryStore:
    s = InMemoryStore()
    await s.start()
    return s


async def test_job_round_trip(store: InMemoryStore) -> None:
    await check_job_round_trip(store)


async def test_list_jobs_keyset_paging(store: InMemoryStore) -> None:
    await check_list_jobs_keyset_paging(store)


async def test_deep_copy_isolation(store: InMemoryStore) -> None:
    await check_deep_copy_isolation(store)


async def test_events_ordering_and_after_seq(store: InMemoryStore) -> None:
    await check_events(store)


async def test_duplicate_seq_rejected(store: InMemoryStore) -> None:
    await check_duplicate_seq_rejected(store)


async def test_result_round_trip(store: InMemoryStore) -> None:
    await check_result_round_trip(store)


async def test_records_round_trip(store: InMemoryStore) -> None:
    await check_records_round_trip(store)


async def test_mark_interrupted(store: InMemoryStore) -> None:
    await check_mark_interrupted(store)


async def test_count_active(store: InMemoryStore) -> None:
    assert await store.count_active() == 0
    a = make_job(status="queued")
    b = make_job(status="researching")
    c = make_job(status="completed")
    for job in (a, b, c):
        await store.create_job(job)
    assert await store.count_active() == 2
    b.status = "cancelled"
    await store.update_job(b)
    assert await store.count_active() == 1
    await store.mark_interrupted()
    assert await store.count_active() == 0
    await store.close()


def test_interrupted_payload_shape() -> None:
    payload = interrupted_payload()
    assert payload.code is ErrorCode.INTERRUPTED
    assert payload.retryable is True
    assert payload.message == default_message(ErrorCode.INTERRUPTED)
    assert payload.details is None


async def test_terminal_analyses_are_evicted_beyond_the_cap() -> None:
    from bayanalytics.jobs.models import AnalysisJob
    from bayanalytics.schemas.events import AnalysisEvent
    from bayanalytics.store.memory import InMemoryStore

    store = InMemoryStore(max_terminal=2)
    for i in range(4):
        job = AnalysisJob(
            analysis_id=f"an_{i}",
            query="q",
            profile="fast",
            requested_horizon="auto",
            resolved_horizon="multi_horizon",
        )
        await store.create_job(job)
        await store.append_event(
            AnalysisEvent(event="analysis.started", analysis_id=job.analysis_id, seq=1, data={})
        )
        job.status = "completed"
        job.finished_at = job.created_at
        await store.update_job(job)
    assert store.resident_analyses == 2
    assert await store.get_job("an_0") is None and await store.list_events("an_0") == []
    assert await store.get_job("an_3") is not None
    # Active jobs are never evicted.
    active = AnalysisJob(
        analysis_id="an_active",
        query="q",
        profile="fast",
        requested_horizon="auto",
        resolved_horizon="multi_horizon",
        status="researching",
    )
    await store.create_job(active)
    await store.update_job(active)
    assert await store.get_job("an_active") is not None
