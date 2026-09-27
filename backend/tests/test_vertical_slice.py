"""Integration tests for the first engineering task (AGENT.md sections 19, 40).

Fixture research (synthetic Apple data) + MockLaya + MockSpark through the real runtime,
runner, event bus, orchestrator and API. No network, no model weights.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import httpx

from bayanalytics.config import Settings
from bayanalytics.laya.mock import MockLaya
from bayanalytics.main import create_app
from bayanalytics.runtime import Runtime
from bayanalytics.schemas.events import AnalysisEvent
from bayanalytics.spark.mock import MockSpark
from bayanalytics.wiring import build_runtime

FIXTURES = Path(__file__).parent / "fixtures" / "research" / "apple"


def _settings(**overrides: object) -> Settings:
    base = {
        "research_provider": "fixture",
        "research_fixture_dir": FIXTURES,
        "laya_mode": "mock",
        "spark_mode": "mock",
        "whisper_mode": "mock",
        "log_level": "WARNING",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


async def _client(rt: Runtime) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(rt.settings, runtime=rt)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", timeout=60
        ) as client:
            yield client


def _parse_sse(raw: str) -> list[dict]:
    events = []
    for block in raw.split("\n\n"):
        lines = [line for line in block.splitlines() if line and not line.startswith(":")]
        if not lines:
            continue
        record: dict = {}
        for line in lines:
            key, _, value = line.partition(": ")
            record[key] = value
        if "event" in record:
            record["data"] = json.loads(record["data"])
            record["id"] = int(record["id"])
            events.append(record)
    return events


async def _run_to_completion(rt: Runtime, body: dict) -> tuple[str, list[dict], dict]:
    async for client in _client(rt):
        created = await client.post("/api/v1/analyses", json=body)
        assert created.status_code == 202, created.text
        analysis_id = created.json()["analysis_id"]
        async with client.stream("GET", f"/api/v1/analyses/{analysis_id}/events") as resp:
            raw = "".join([chunk async for chunk in resp.aiter_text()])
        events = _parse_sse(raw)
        await asyncio.sleep(0.02)
        result = (await client.get(f"/api/v1/analyses/{analysis_id}")).json()
        return analysis_id, events, result
    raise AssertionError("client context did not yield")


# ----------------------------------------------------------------------------- full flow


async def test_assess_apple_end_to_end() -> None:
    rt = build_runtime(_settings())
    analysis_id, events, result = await _run_to_completion(
        rt, {"query": "Assess Apple.", "profile": "fast"}
    )
    names = [e["event"] for e in events]
    # Sequence numbers are contiguous and the stream closes on the terminal event.
    assert [e["id"] for e in events] == list(range(1, len(events) + 1))
    assert names[0] == "analysis.started" and names[-1] == "analysis.completed"
    assert names.count("analysis.completed") == 1 and "analysis.failed" not in names
    # Observable state sequence (AGENT.md 37.2): every stage is visible, in order.
    order = [
        "analysis.started",
        "instrument.resolved",
        "research.started",
        "research.query",
        "research.source_found",
        "research.completed",
        "normalization.completed",
        "laya.started",
        "laya.decision",
        "laya.completed",
        "calculation.started",
        "calculation.completed",
        "spark.loading",
        "spark.started",
        "spark.token",
        "spark.completed",
        "analysis.completed",
    ]
    # Laya also runs inside the research phase (research_plan), so each stage is searched for
    # after the previous one rather than by first occurrence.
    cursor = -1
    positions = []
    for name in order:
        cursor = names.index(name, cursor + 1)
        positions.append(cursor)
    assert positions == sorted(positions), list(zip(order, positions, strict=True))
    assert names.index("laya.started") < names.index("research.completed")  # Laya-directed loop
    assert names.index("spark.loading") < names.index("spark.started") < names.index("spark.token")
    resolved = next(e["data"] for e in events if e["event"] == "instrument.resolved")
    assert resolved["symbol"] == "AAPL" and resolved["cik"] == "0000320193"
    # Streamed text is exactly the concatenation of the token events.
    streamed = "".join(e["data"]["text"] for e in events if e["event"] == "spark.token")
    assert streamed and result["streamed_text"] == streamed
    # No hidden reasoning: token data only carries text, decisions only carry recorded state.
    for e in events:
        if e["event"] == "spark.token":
            assert set(e["data"]) == {"analysis_id", "seq", "ts", "text"}
        if e["event"] == "laya.decision":
            assert {"decision_type", "decision", "confidence", "stage"} <= set(e["data"])
    # Structured result (37.3).
    assert result["status"] == "completed" and result["error"] is None
    assert result["analysis_id"] == analysis_id and result["horizon"] == "multi_horizon"
    assert result["instrument"]["symbol"] == "AAPL"
    assert len(result["sources"]) >= 10
    assert any(s["source_type"] == "regulatory_filing" for s in result["sources"])
    assert all(len(s["excerpt"]) <= 600 for s in result["sources"])
    calcs = result["calculations"]
    computed = [c for c in calcs if c["status"] == "computed"]
    assert len(computed) >= 20
    for calc in calcs:
        assert calc["formula"] and calc["name"]
        if calc["status"] == "unavailable":
            assert calc["value"] is None and (calc["missing_inputs"] or calc["notes"])
    assert len(result["laya_decisions"]) >= 40
    assessment = result["assessment"]
    assert assessment["summary"]
    assert assessment["bull_evidence"] and all(
        i["source_ids"] or i["calc_id"] or i["decision_id"] for i in assessment["bull_evidence"]
    )
    assert assessment["fundamentals"]["revenue_growth_yoy"]["display"].endswith("%")
    assert "market_cap" in assessment["valuation"]
    assert assessment["benchmark_context"]["benchmarks"]
    assert len(result["horizon_assessments"]) == 4
    for horizon, item in result["horizon_assessments"].items():
        assert item["stance"] in {"bullish", "neutral", "bearish", "mixed"}
        assert item["decision_id"] and 0 < item["confidence"] <= 1
        assert horizon in {"near_term", "next_cycle", "medium_term", "long_term"}
    telemetry = result["telemetry"]
    for key in (
        "retrieval_ms",
        "normalization_ms",
        "laya_ms",
        "math_ms",
        "spark_total_ms",
        "total_request_ms",
    ):
        assert telemetry[key] is not None and telemetry[key] >= 0, key
    research = telemetry["research"]
    assert research["queries_issued"] >= 5 and research["sources_fetched"] >= 10
    assert research["termination_reason"]
    assert telemetry["versions"]["laya_schema_version"] == "finance-v1"
    assert telemetry["versions"]["normalization_version"] == "2026.09-1"
    assert telemetry["process_peak_rss_mb"] and telemetry["system_total_ram_mb"]
    # Every cited source id in the assessment exists in the source list.
    known = {s["source_id"] for s in result["sources"]}
    for bucket in ("bull_evidence", "bear_evidence", "risks", "what_changed"):
        for item in assessment[bucket]:
            assert set(item["source_ids"]) <= known
    # Reconnect replays the tail only.
    async for client in _client(rt):
        async with client.stream(
            "GET",
            f"/api/v1/analyses/{analysis_id}/events",
            headers={"Last-Event-ID": str(len(events) - 1)},
        ) as resp:
            tail = _parse_sse("".join([chunk async for chunk in resp.aiter_text()]))
        assert [e["event"] for e in tail] == ["analysis.completed"]


async def test_leakage_guard_freezes_information_set() -> None:
    as_of = datetime(2026, 6, 30, tzinfo=UTC)
    rt = build_runtime(_settings(eval_as_of=as_of))
    _analysis_id, events, result = await _run_to_completion(rt, {"query": "Assess Apple."})
    assert result["status"] == "completed"
    assert result["as_of"].startswith("2026-06-30")
    for source in result["sources"]:
        if source["published_at"]:
            published = datetime.fromisoformat(source["published_at"].replace("Z", "+00:00"))
            assert published <= as_of, source["title"]
    rejected = [e["data"] for e in events if e["event"] == "research.source_rejected"]
    assert any(r["reason"] == "published_after_as_of" for r in rejected)
    # Prices stop at the cut-off, so every close-based calculation is dated on or before it.
    market = result["assessment"]["market_context"]
    assert market["price"]["session_date"] <= "2026-06-30"


# ----------------------------------------------------------------------------- degraded paths


async def test_cancel_mid_synthesis_preserves_partial_content() -> None:
    # httpx's ASGI transport buffers a streaming body, so the live stream is read in-process
    # through the event bus while the cancel goes through the real API.
    spark = MockSpark(delay_s=0.02)
    rt = build_runtime(_settings(), spark=spark)
    async for client in _client(rt):
        created = await client.post("/api/v1/analyses", json={"query": "Assess Apple."})
        analysis_id = created.json()["analysis_id"]
        events: list[AnalysisEvent] = []
        cancelled_at: int | None = None
        async for event in rt.bus.stream(analysis_id):
            events.append(event)
            if event.event == "spark.token" and cancelled_at is None:
                cancel = await client.post(f"/api/v1/analyses/{analysis_id}/cancel")
                assert cancel.status_code == 200, cancel.text
                assert cancel.json()["cancel_requested"] is True
                cancelled_at = event.seq
        assert cancelled_at is not None
        assert events[-1].event == "analysis.failed"
        assert events[-1].data["error"]["code"] == "CANCELLED"
        assert events[-1].data["status"] == "cancelled"
        assert "spark.completed" not in {e.event for e in events}
        tokens_after_cancel = [
            e for e in events if e.event == "spark.token" and e.seq > cancelled_at
        ]
        assert len(tokens_after_cancel) <= 2  # cancellation is checked between chunks
        await asyncio.sleep(0.05)
        result = (await client.get(f"/api/v1/analyses/{analysis_id}")).json()
        assert result["status"] == "cancelled" and result["partial"] is True
        assert result["streamed_text"]  # preserved, but never presented as complete
        assert result["sources"] and result["calculations"]
        assert result["error"]["code"] == "CANCELLED"
        # Cancelling again is a no-op on a terminal analysis.
        again = await client.post(f"/api/v1/analyses/{analysis_id}/cancel")
        assert again.status_code == 200 and again.json()["status"] == "cancelled"


async def test_deep_profile_unavailable_is_a_structured_error() -> None:
    rt = build_runtime(_settings(), spark=MockSpark(deep_available=False))
    async for client in _client(rt):
        caps = (await client.get("/api/v1/capabilities")).json()
        assert caps["profiles"]["fast"]["available"] is True
        assert caps["profiles"]["deep"]["available"] is False
        assert caps["profiles"]["deep"]["code"] == "DEEP_PROFILE_UNAVAILABLE"
        created = await client.post(
            "/api/v1/analyses", json={"query": "Assess Apple.", "profile": "deep"}
        )
        assert created.status_code == 503
        assert created.json()["error"]["code"] == "DEEP_PROFILE_UNAVAILABLE"
        assert created.json()["error"]["message"].endswith("Try Fast.")


async def test_ambiguous_instrument_fails_with_candidates() -> None:
    rt = build_runtime(_settings())
    _id, events, result = await _run_to_completion(rt, {"query": "Compare Apple and Microsoft."})
    assert events[-1]["event"] == "analysis.failed"
    error = events[-1]["data"]["error"]
    assert error["code"] == "AMBIGUOUS_INSTRUMENT"
    assert {c["symbol"] for c in error["details"]["candidates"]} == {"AAPL", "MSFT"}
    assert result["status"] == "failed" and result["error"]["code"] == "AMBIGUOUS_INSTRUMENT"
    assert [e["event"] for e in events] == ["analysis.started", "analysis.failed"]


async def test_laya_failure_is_structured_and_keeps_sources() -> None:
    from bayanalytics.errors import AnalysisError
    from bayanalytics.schemas.common import ErrorCode

    laya = MockLaya(raise_error=AnalysisError(ErrorCode.LAYA_INFERENCE_FAILED))
    rt = build_runtime(_settings(), laya=laya)
    _id, events, result = await _run_to_completion(rt, {"query": "Assess Apple."})
    assert events[-1]["event"] == "analysis.failed"
    assert events[-1]["data"]["error"]["code"] == "LAYA_INFERENCE_FAILED"
    assert result["status"] == "failed" and result["partial"] is True
    assert result["sources"]  # research already happened and is preserved
    assert not result["streamed_text"]


async def test_explicit_instrument_and_horizon() -> None:
    rt = build_runtime(_settings())
    _id, events, result = await _run_to_completion(
        rt,
        {
            "query": "How does it look over the next earnings?",
            "instrument": {"symbol": "aapl", "exchange": "nasdaq"},
            "horizon": "near_term",
        },
    )
    assert result["status"] == "completed"
    assert result["horizon"] == "near_term"
    assert list(result["horizon_assessments"]) == ["near_term"]
    resolved = next(e["data"] for e in events if e["event"] == "instrument.resolved")
    assert resolved["resolution_method"] == "instrument_ref" and resolved["exchange"] == "NASDAQ"


async def test_stored_artifacts_and_health() -> None:
    rt = build_runtime(_settings())
    analysis_id, _events, _result = await _run_to_completion(rt, {"query": "Assess Apple."})
    job = await rt.store.get_job(analysis_id)
    assert job is not None and job.status == "completed" and job.instrument is not None
    assert job.source_ids and job.last_seq > 100
    assert job.profile == "fast" and job.normalization_version == "2026.09-1"
    stored_events = await rt.store.list_events(analysis_id)
    assert stored_events[-1].event == "analysis.completed"
    assert all(isinstance(e, AnalysisEvent) for e in stored_events)
    async for client in _client(rt):
        health = (await client.get("/api/v1/health")).json()
        assert health["status"] == "ok" and {c["name"] for c in health["components"]} >= {
            "laya",
            "spark",
            "store",
        }
