"""Integration tests for the vertical slice (AGENT.md sections 19, 40).

Fixture research (synthetic Apple data) + ``RuleLaya`` + ``ScriptedSpark`` + ``FixedTranscriber``
through the real runtime, runner, event bus, orchestrator and API. No network, no model weights.
The doubles are injected through ``build_runtime``; nothing in the product can select them.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from bayanalytics.config import Settings
from bayanalytics.errors import AnalysisError
from bayanalytics.main import create_app
from bayanalytics.research.sources import domain_of
from bayanalytics.runtime import Runtime
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.events import AnalysisEvent
from bayanalytics.wiring import build_runtime
from doubles import FixedTranscriber, RuleLaya, ScriptedSpark, fixture_research_stack

FIXTURES = Path(__file__).parent / "fixtures" / "research" / "apple"
EXECUTION = {
    "spark_mode": "managed",
    "whisper_mode": "disabled",
    "deployment": "local",
    "search_configured": True,
}


def _settings(**overrides: object) -> Settings:
    base: dict[str, Any] = {"log_level": "WARNING"}
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _runtime(
    settings: Settings | None = None,
    *,
    laya: RuleLaya | None = None,
    spark: ScriptedSpark | None = None,
    transcriber: FixedTranscriber | None = None,
) -> Runtime:
    settings = settings or _settings()
    return build_runtime(
        settings,
        laya=laya or RuleLaya(),
        spark=spark or ScriptedSpark(),
        transcriber=transcriber or FixedTranscriber(),
        research=fixture_research_stack(settings, FIXTURES),
    )


@contextlib.asynccontextmanager
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
    async with _client(rt) as client:
        created = await client.post("/api/v1/analyses", json=body)
        assert created.status_code == 202, created.text
        analysis_id = created.json()["analysis_id"]
        async with client.stream("GET", f"/api/v1/analyses/{analysis_id}/events") as resp:
            raw = "".join([chunk async for chunk in resp.aiter_text()])
        events = _parse_sse(raw)
        await asyncio.sleep(0.02)
        result = (await client.get(f"/api/v1/analyses/{analysis_id}")).json()
        return analysis_id, events, result


# ----------------------------------------------------------------------------- full flow


async def test_assess_apple_end_to_end() -> None:
    rt = _runtime()
    analysis_id, events, result = await _run_to_completion(
        rt, {"query": "Assess $AAPL.", "profile": "fast"}
    )
    names = [e["event"] for e in events]
    # Sequence numbers are contiguous and the stream closes on the terminal event.
    assert [e["id"] for e in events] == list(range(1, len(events) + 1))
    assert names[0] == "analysis.started" and names[-1] == "analysis.completed"
    assert names.count("analysis.completed") == 1 and "analysis.failed" not in names
    # Observable state sequence (AGENT.md 37.2): every stage is visible, in order.
    # Spark pass 1 (query understanding) runs between instrument.resolved and research: it is
    # internal, so the first model load shows as spark.loading before research and nothing
    # else of it is streamed.
    order = [
        "analysis.started",
        "instrument.resolved",
        "spark.loading",
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
    assert names.count("spark.loading") == 1 and names.count("spark.started") == 1
    assert names.index("research.completed") < names.index("spark.started")
    started = next(e["data"] for e in events if e["event"] == "analysis.started")
    assert started["execution"] == EXECUTION
    resolved = next(e["data"] for e in events if e["event"] == "instrument.resolved")
    # the instrument is the ticker in the question: no directory, no name, no CIK
    assert resolved["symbol"] == "AAPL" and resolved["resolution_method"] == "cashtag"
    assert resolved["name"] == "" and resolved["cik"] is None
    # The Spark prompt size is measured by the session before generation and reported as such
    # on spark.started, spark.completed and in telemetry; nothing is estimated.
    spark_started = next(e["data"] for e in events if e["event"] == "spark.started")
    spark_completed = next(e["data"] for e in events if e["event"] == "spark.completed")
    assert "prompt_tokens_estimate" not in spark_started
    assert isinstance(spark_started["prompt_tokens"], int) and spark_started["prompt_tokens"] > 0
    assert spark_completed["prompt_tokens"] == spark_started["prompt_tokens"]
    assert spark_completed["output_tokens"] is None  # the double produced no model tokens
    assert spark_completed["tokens_per_second"] is None and spark_completed["truncated"] is False
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
    # Web pages from several sites; every page opened was a search hit.
    sources = result["sources"]
    assert len(sources) >= 2 and len({domain_of(s["url"]) for s in sources}) >= 2
    searched = json.loads((FIXTURES / "searches.json").read_text())["searches"]
    hits = {hit["url"] for results in searched.values() for hit in results}
    fetched = [e["data"]["url"] for e in events if e["event"] == "research.fetching"]
    assert fetched and set(fetched) <= hits
    assert all(len(s["excerpt"]) <= 600 for s in sources)
    # No verified figures: every calculation runs and reports its missing operands.
    calcs = result["calculations"]
    assert calcs
    for calc in calcs:
        assert calc["formula"] and calc["name"]
        assert calc["status"] == "unavailable"
        assert calc["value"] is None and (calc["missing_inputs"] or calc["notes"])
    assert result["laya_decisions"]
    assessment = result["assessment"]
    assert assessment["summary"]
    assert assessment["bull_evidence"] and all(
        i["source_ids"] or i["calc_id"] or i["decision_id"] for i in assessment["bull_evidence"]
    )
    assert (
        "Web search returned no page with price history or quarterly figures: this "
        f"assessment is based only on the text of {len(sources)} web pages found by search."
    ) in assessment["uncertainties"]
    assert "benchmarks" not in assessment["benchmark_context"]
    # No search result held a price or figures table: honest empty views, no market events.
    assert result["market"] == {"series": [], "fundamentals": None}
    assert "market.series" not in names and "market.fundamentals" not in names
    assert len(result["horizon_assessments"]) == 4
    for horizon, item in result["horizon_assessments"].items():
        assert item["stance"] in {"bullish", "neutral", "bearish", "mixed"}
        assert item["decision_id"] and 0 < item["confidence"] <= 1
        assert horizon in {"near_term", "next_cycle", "medium_term", "long_term"}
    telemetry = result["telemetry"]
    for key in (
        "query_understanding_ms",
        "retrieval_ms",
        "normalization_ms",
        "laya_ms",
        "math_ms",
        "spark_total_ms",
        "total_request_ms",
    ):
        assert telemetry[key] is not None and telemetry[key] >= 0, key
    research = telemetry["research"]
    assert research["queries_issued"] >= 5 and research["sources_fetched"] >= 2
    assert research["termination_reason"] and research["queries_failed"] == 0
    assert "structured_failures" not in research
    assert telemetry["spark_prompt_tokens"] == spark_started["prompt_tokens"]
    # Measured or None: the doubles measure no model, no memory and no runtime.
    for key in (
        "spark_output_tokens",
        "spark_tokens_per_second",
        "spark_load_ms",
        "query_understanding_output_tokens",
        "query_understanding_load_ms",
        "spark_resident_ram_mb",
        "laya_resident_ram_mb",
        "laya_warm_inference_ms",
    ):
        assert telemetry[key] is None, key
    versions = telemetry["versions"]
    assert versions["laya_schema_version"] == "finance-v1"
    assert versions["normalization_version"] == "2026.09-1"
    assert versions["laya_package_version"] is None
    assert versions["spark_artifact"] is None and versions["spark_runtime"] is None
    assert versions["execution"] == EXECUTION
    assert telemetry["process_peak_rss_mb"] and telemetry["system_total_ram_mb"]
    assert "mock" not in json.dumps(result).lower()
    # Every cited source id in the assessment exists in the source list.
    known = {s["source_id"] for s in result["sources"]}
    for bucket in ("bull_evidence", "bear_evidence", "risks", "what_changed"):
        for item in assessment[bucket]:
            assert set(item["source_ids"]) <= known
    # Reconnect replays the tail only.
    async with _client(rt) as client:
        async with client.stream(
            "GET",
            f"/api/v1/analyses/{analysis_id}/events",
            headers={"Last-Event-ID": str(len(events) - 1)},
        ) as resp:
            tail = _parse_sse("".join([chunk async for chunk in resp.aiter_text()]))
        assert [e["event"] for e in tail] == ["analysis.completed"]


async def test_spark_citations_of_bundle_sources_are_recognised() -> None:
    rt = _runtime()
    _id, _events, result = await _run_to_completion(rt, {"query": "Assess $AAPL."})
    assessment = result["assessment"]
    # ScriptedSpark cites only ids rendered in the bundle, so none may be reported as unknown ...
    assert not any("unknown source" in u for u in assessment["uncertainties"])
    # ... and Spark's own horizon bullets (not the deterministic fallback) must survive.
    for horizon in result["horizon_assessments"].values():
        assert any("in the bundle" in e["text"] for e in horizon["key_evidence"]), horizon
        assert horizon["synthesized"] is True
    assert result["partial"] is False


async def test_provenance_and_decision_ids_are_stable_across_runs() -> None:
    as_of = datetime(2026, 9, 26, tzinfo=UTC)
    body = {"query": "Assess $AAPL."}
    first = (await _run_to_completion(_runtime(_settings(eval_as_of=as_of)), body))[2]
    second = (await _run_to_completion(_runtime(_settings(eval_as_of=as_of)), body))[2]
    assert first["analysis_id"] != second["analysis_id"]
    assert first["status"] == second["status"] == "completed"
    # Same URL -> same source id; same compacted state, stage and question -> same decision id.
    assert [s["source_id"] for s in first["sources"]] == [s["source_id"] for s in second["sources"]]
    assert len({s["source_id"] for s in first["sources"]}) == len(first["sources"])
    assert [d["decision_id"] for d in first["laya_decisions"]] == [
        d["decision_id"] for d in second["laya_decisions"]
    ]
    assert len({d["decision_id"] for d in first["laya_decisions"]}) == len(first["laya_decisions"])
    # no verified figures means no history segments to score
    assert not any(d["stage"] == "history_scan" for d in first["laya_decisions"])


async def test_leakage_guard_freezes_information_set() -> None:
    as_of = datetime(2026, 6, 30, tzinfo=UTC)
    rt = _runtime(_settings(eval_as_of=as_of))
    _analysis_id, events, result = await _run_to_completion(rt, {"query": "Assess $AAPL."})
    assert result["status"] == "completed"
    assert result["as_of"].startswith("2026-06-30")
    for source in result["sources"]:
        if source["published_at"]:
            published = datetime.fromisoformat(source["published_at"].replace("Z", "+00:00"))
            assert published <= as_of, source["title"]
    rejected = [e["data"] for e in events if e["event"] == "research.source_rejected"]
    assert any(r["reason"] == "published_after_as_of" for r in rejected)
    # Only pages dated on or before the cut-off were kept: here the two older ones.
    assert {domain_of(s["url"]) for s in result["sources"]} == {"barrons.com", "zacks.com"}


# ----------------------------------------------------------------------------- degraded paths


_ONE_SITE_PAGE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="bay-fixture" content="true">
<title>[Fixture] {title}</title>
<meta property="article:published_time" content="2026-09-{day}T09:00:00Z"></head>
<body><main><article><h1>[Fixture] {title}</h1>
<p>Synthetic fixture page {n} for the single-website test. It describes no real company and
exists only so the extractor finds enough readable prose on the page to keep it.</p>
<p>The fixture text says the fixture company's fixture quarter was broadly in line with the
fixture expectations, and that fixture margins were steady, without giving any figure.</p>
</article></main></body></html>
"""


def _one_site_fixture(tmp_path: Path) -> Path:
    """Every search returns two pages from one website: kept, but one site is not enough."""
    pages = json.loads((FIXTURES / "pages.json").read_text())["pages"]
    for entry in pages.values():  # keep every apple body reachable from the new directory
        if entry.get("body_file"):
            entry["body_file"] = str((FIXTURES / entry["body_file"]).resolve())
    hits = []
    for n, day in ((1, 21), (2, 22)):
        url = f"https://www.cnbc.com/2026/09/{day}/fixture-single-site-{n}.html"
        body = tmp_path / f"single_site_{n}.html"
        body.write_text(_ONE_SITE_PAGE.format(title=f"Single-site page {n}", day=day, n=n))
        pages[url] = {"status": 200, "content_type": "text/html", "body_file": str(body)}
        hits.append({"url": url, "title": f"[Fixture] Single-site page {n}", "engine": "fixture"})
    (tmp_path / "pages.json").write_text(json.dumps({"fixture": True, "pages": pages}))
    (tmp_path / "searches.json").write_text(json.dumps({"fixture": True, "searches": {"*": hits}}))
    return tmp_path


async def test_pages_from_a_single_website_are_insufficient_evidence(tmp_path: Path) -> None:
    settings = _settings()
    rt = build_runtime(
        settings,
        laya=RuleLaya(),
        spark=ScriptedSpark(),
        transcriber=FixedTranscriber(),
        research=fixture_research_stack(settings, _one_site_fixture(tmp_path)),
    )
    _id, events, result = await _run_to_completion(rt, {"query": "Assess $AAPL."})
    assert result["status"] == "failed" and events[-1]["event"] == "analysis.failed"
    error = result["error"]
    assert error["code"] == "INSUFFICIENT_EVIDENCE"
    assert error["details"]["domains"] == ["cnbc.com"] and error["details"]["web_pages"] == 2
    assert error["details"]["missing"] == ["pages from at least 2 different websites (found 1)"]
    kept = [e["data"]["url"] for e in events if e["event"] == "research.source_found"]
    assert any("fixture-single-site" in url for url in kept)
    assert "spark.started" not in [e["event"] for e in events]  # nothing was synthesised


async def test_cancel_mid_synthesis_preserves_partial_content() -> None:
    # httpx's ASGI transport buffers a streaming body, so the live stream is read in-process
    # through the event bus while the cancel goes through the real API.
    spark = ScriptedSpark(delay_s=0.02)
    rt = _runtime(spark=spark)
    async with _client(rt) as client:
        created = await client.post("/api/v1/analyses", json={"query": "Assess $AAPL."})
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
        assert spark.runs[-1]["completed"] is False
        # Cancelling again is a no-op on a terminal analysis.
        again = await client.post(f"/api/v1/analyses/{analysis_id}/cancel")
        assert again.status_code == 200 and again.json()["status"] == "cancelled"


async def test_deep_profile_unavailable_is_a_structured_error() -> None:
    rt = _runtime(spark=ScriptedSpark(deep_available=False))
    async with _client(rt) as client:
        caps = (await client.get("/api/v1/capabilities")).json()
        assert caps["profiles"]["fast"]["available"] is True
        assert caps["profiles"]["deep"]["available"] is False
        assert caps["profiles"]["deep"]["code"] == "DEEP_PROFILE_UNAVAILABLE"
        created = await client.post(
            "/api/v1/analyses", json={"query": "Assess $AAPL.", "profile": "deep"}
        )
        assert created.status_code == 503
        assert created.json()["error"]["code"] == "DEEP_PROFILE_UNAVAILABLE"
        assert created.json()["error"]["message"].endswith("Try Fast.")


async def test_ambiguous_instrument_is_answered_at_post() -> None:
    rt = _runtime()
    async with _client(rt) as client:
        created = await client.post("/api/v1/analyses", json={"query": "Compare AAPL and MSFT."})
        assert created.status_code == 422
        error = created.json()["error"]
        assert error["code"] == "AMBIGUOUS_INSTRUMENT"
        assert error["message"] == "Which company did you mean?"
        assert {c["symbol"] for c in error["details"]["candidates"]} == {"AAPL", "MSFT"}
        assert rt.runner.active_count == 0  # no orphan analysis was created
        health = (await client.get("/api/v1/health")).json()
        assert health["active_analyses"] == 0


async def test_laya_failure_is_structured_and_keeps_sources() -> None:
    laya = RuleLaya(raise_error=AnalysisError(ErrorCode.LAYA_INFERENCE_FAILED))
    rt = _runtime(laya=laya)
    _id, events, result = await _run_to_completion(rt, {"query": "Assess $AAPL."})
    assert events[-1]["event"] == "analysis.failed"
    assert events[-1]["data"]["error"]["code"] == "LAYA_INFERENCE_FAILED"
    assert result["status"] == "failed" and result["partial"] is True
    assert result["sources"]  # research already happened and is preserved
    assert not result["streamed_text"]


async def test_explicit_instrument_and_horizon() -> None:
    rt = _runtime()
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
    rt = _runtime()
    analysis_id, _events, result = await _run_to_completion(rt, {"query": "Assess $AAPL."})
    job = await rt.store.get_job(analysis_id)
    assert job is not None and job.status == "completed" and job.instrument is not None
    assert job.source_ids and job.last_seq > 20
    assert job.profile == "fast" and job.normalization_version == "2026.09-1"
    assert job.spark_artifact is None and job.spark_runtime is None  # nothing was measured
    stored_events = await rt.store.list_events(analysis_id)
    assert stored_events[-1].event == "analysis.completed"
    assert all(isinstance(e, AnalysisEvent) for e in stored_events)
    # Every artifact of a completed analysis is persisted, not only the result document.
    facts = await rt.store.get_facts(analysis_id)
    assert facts == [] and result["freshness_summary"]["facts"]["total"] == 0
    assert {s.source_id for s in await rt.store.get_sources(analysis_id)} == {
        s["source_id"] for s in result["sources"]
    }
    assert {c.calc_id for c in await rt.store.get_calculations(analysis_id)} == {
        c["calc_id"] for c in result["calculations"]
    }
    assert len(await rt.store.get_decisions(analysis_id)) == len(result["laya_decisions"])
    async with _client(rt) as client:
        health = (await client.get("/api/v1/health")).json()
        assert health["status"] == "ok"
        assert {c["name"]: c["status"] for c in health["components"]} == {
            "laya": "ok",
            "spark": "ok",
            "whisper": "ok",
            "store": "ok",
        }
        assert health["execution"] == EXECUTION and health["active_analyses"] == 0
        assert "mock" not in json.dumps(health).lower()
        caps = (await client.get("/api/v1/capabilities")).json()
        assert caps["execution"] == EXECUTION
        assert caps["voice"] is True and caps["research"] is True
        assert caps["profiles"]["fast"]["available"] and caps["profiles"]["deep"]["available"]


# ----------------------------------------------------------------------- live research views


async def test_live_research_events_end_to_end() -> None:
    """The run announces every request right before it goes out, reports the search hits,
    and never ships the excerpt of a page whose terms do not allow redistribution."""
    _, events, result = await _run_to_completion(
        _runtime(), {"query": "Assess $AAPL.", "profile": "fast"}
    )
    names = [e["event"] for e in events]
    assert result["status"] == "completed"
    assert "research.fetching" in names and "research.search_results" in names
    assert names.index("research.search_results") < names.index("research.fetching")
    assert names.index("research.fetching") < names.index("research.source_found")
    assert not any(n.startswith("market.") for n in names)
    assert all(e["data"]["kind"] == "web" for e in events if e["event"] == "research.fetching")
    for e in events:
        if e["event"] == "research.source_found":
            assert e["data"]["preview"] is None  # no page here holds a price or figures table
            if e["data"]["redistribution"] != "allowed":
                assert e["data"]["excerpt"] is None
    async with _client(_runtime()) as client:
        caps = (await client.get("/api/v1/capabilities")).json()
    assert "market" not in caps and caps["web_search"] is True
