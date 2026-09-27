"""ResearchRunner, intents and the fixture/recording providers, all against the apple fixture."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.base import InstrumentIdentity, ResearchBudget
from bayanalytics.research.extract import EXCERPT_CHARS, MAX_TEXT_CHARS
from bayanalytics.research.fixture_provider import (
    FixtureFetcher,
    FixtureResearchProvider,
    RecordingFetcher,
    RecordingProvider,
)
from bayanalytics.research.http_provider import HttpResearchProvider, build_research_stack
from bayanalytics.research.intents import (
    PlannedQuery,
    ResearchIntent,
    build_queries,
    gap_to_intent,
    seed_plan,
)
from bayanalytics.research.provider import ResearchProvider
from bayanalytics.research.runner import ResearchRunner, RoundResult
from bayanalytics.schemas.common import ErrorCode

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "research" / "apple"
AS_OF = datetime(2026, 9, 26, tzinfo=UTC)
IDENTITY = InstrumentIdentity(
    symbol="AAPL",
    exchange="NASDAQ",
    name="Apple Inc.",
    cik="320193",
    sic="3571",
    fiscal_year_end="0930",
)


class RecordingCtx:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.ctx = AnalysisContext(analysis_id="an_test", emit=self.sink)

    async def sink(self, name: str, data: dict[str, Any]) -> None:
        self.events.append((name, data))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [data for event, data in self.events if event == name]


@pytest.fixture
def settings() -> Settings:
    return Settings(
        research_provider="fixture",
        research_fixture_dir=FIXTURE_DIR,
        research_contact_email="dev@example.com",
        research_min_request_interval_s=0.0,
    )


@pytest.fixture
def stack(settings: Settings):
    return build_research_stack(settings)


def make_runner(stack, settings: Settings, **budget: Any) -> ResearchRunner:
    provider, edgar, prices = stack
    params = {"max_fetch_per_round": 10, "max_sources": 24, "max_rounds": 4}
    params.update(budget)
    return ResearchRunner(provider, edgar, prices, settings, ResearchBudget(**params))


def one(
    intent: ResearchIntent, horizon: str = "multi_horizon", gaps: list[str] | None = None
) -> PlannedQuery:
    planned = build_queries(intent, IDENTITY, horizon, AS_OF, gaps or [])
    assert len(planned) == 1
    return planned[0]


# --------------------------------------------------------------------------------------
# intents
# --------------------------------------------------------------------------------------


def test_seed_plan_per_horizon() -> None:
    assert seed_plan("near_term") == [
        ResearchIntent.retrieve_recent_news,
        ResearchIntent.retrieve_price_history,
        ResearchIntent.retrieve_earnings_history,
        ResearchIntent.retrieve_sector_benchmark,
        ResearchIntent.retrieve_latest_filing,
    ]
    assert seed_plan("next_cycle") == [
        ResearchIntent.retrieve_latest_filing,
        ResearchIntent.retrieve_earnings_history,
        ResearchIntent.retrieve_guidance_history,
        ResearchIntent.retrieve_recent_news,
        ResearchIntent.retrieve_price_history,
    ]
    assert seed_plan("medium_term") == [
        ResearchIntent.retrieve_earnings_history,
        ResearchIntent.retrieve_latest_filing,
        ResearchIntent.retrieve_price_history,
        ResearchIntent.retrieve_sector_benchmark,
        ResearchIntent.retrieve_historical_coverage,
    ]
    assert seed_plan("long_term") == [
        ResearchIntent.retrieve_earnings_history,
        ResearchIntent.retrieve_latest_filing,
        ResearchIntent.retrieve_historical_coverage,
        ResearchIntent.retrieve_management_commentary,
        ResearchIntent.retrieve_price_history,
    ]
    assert seed_plan("multi_horizon") == [
        ResearchIntent.retrieve_latest_filing,
        ResearchIntent.retrieve_earnings_history,
        ResearchIntent.retrieve_price_history,
        ResearchIntent.retrieve_sector_benchmark,
        ResearchIntent.retrieve_recent_news,
        ResearchIntent.retrieve_guidance_history,
    ]
    assert seed_plan("unknown") == seed_plan("multi_horizon")
    assert seed_plan("near_term") is not seed_plan("near_term")  # fresh list each call


def test_research_intents_are_exactly_the_locked_set() -> None:
    assert [i.value for i in ResearchIntent] == [
        "retrieve_latest_filing",
        "retrieve_recent_news",
        "retrieve_historical_coverage",
        "retrieve_price_history",
        "retrieve_sector_benchmark",
        "retrieve_earnings_history",
        "retrieve_guidance_history",
        "retrieve_management_commentary",
        "retrieve_missing_metric",
        "stop_research",
    ]


def test_build_queries_templates_are_deterministic() -> None:
    news = one(ResearchIntent.retrieve_recent_news)
    assert news.kind == "search"
    assert news.query == '"Apple Inc." earnings OR guidance OR outlook'
    assert news.params == {"categories": "news", "time_range": "month"}
    hist = one(ResearchIntent.retrieve_historical_coverage)
    assert hist.query == '"Apple Inc." stock 2025 results' and hist.params["time_range"] == "year"
    guidance = one(ResearchIntent.retrieve_guidance_history)
    assert guidance.query == '"Apple Inc." guidance raised OR lowered OR cut'
    mgmt = one(ResearchIntent.retrieve_management_commentary)
    assert mgmt.query == '"Apple Inc." earnings call transcript CEO'
    gaps = build_queries(
        ResearchIntent.retrieve_missing_metric,
        IDENTITY,
        "long_term",
        AS_OF,
        ["free_cash_flow", "capex", "ignored"],
    )
    assert [q.query for q in gaps] == [
        '"Apple Inc." free cash flow 2026',
        '"Apple Inc." capex 2026',
    ]
    assert gaps[0].params == {"gap": "free_cash_flow"}
    assert (
        build_queries(ResearchIntent.retrieve_missing_metric, IDENTITY, "long_term", AS_OF, [])
        == []
    )
    filing = one(ResearchIntent.retrieve_latest_filing)
    assert filing.kind == "edgar_submissions" and filing.params["cik"] == "320193"
    assert filing.params["forms"] == ["10-K", "10-Q", "8-K"] and filing.params["limit"] == 6
    facts = one(ResearchIntent.retrieve_earnings_history)
    assert facts.kind == "edgar_companyfacts" and facts.params == {"cik": "320193"}
    prices = one(ResearchIntent.retrieve_price_history, "near_term")
    assert (
        prices.kind == "prices"
        and prices.params["symbol"] == "aapl.us"
        and prices.params["days"] == 400
    )
    long_prices = one(ResearchIntent.retrieve_price_history, "long_term")
    assert long_prices.params["days"] == 5 * 366
    bench = one(ResearchIntent.retrieve_sector_benchmark)
    assert bench.kind == "benchmarks" and bench.params["sic"] == "3571"
    assert build_queries(ResearchIntent.stop_research, IDENTITY, "near_term", AS_OF) == []
    assert build_queries("retrieve_recent_news", IDENTITY, "near_term", AS_OF) == build_queries(
        ResearchIntent.retrieve_recent_news, IDENTITY, "near_term", AS_OF
    )
    assert all(q.label for q in [news, hist, guidance, mgmt, filing, facts, prices, bench])


def test_gap_to_intent() -> None:
    assert gap_to_intent("price_history") is ResearchIntent.retrieve_price_history
    assert gap_to_intent("sector benchmark") is ResearchIntent.retrieve_sector_benchmark
    assert gap_to_intent("guidance") is ResearchIntent.retrieve_guidance_history
    assert gap_to_intent("latest 10-K") is ResearchIntent.retrieve_latest_filing
    assert gap_to_intent("management commentary") is ResearchIntent.retrieve_management_commentary
    assert gap_to_intent("recent news") is ResearchIntent.retrieve_recent_news
    assert gap_to_intent("historical coverage") is ResearchIntent.retrieve_historical_coverage
    assert gap_to_intent("revenue") is ResearchIntent.retrieve_earnings_history
    assert gap_to_intent("eps_diluted") is ResearchIntent.retrieve_earnings_history
    assert gap_to_intent("something_else") is ResearchIntent.retrieve_missing_metric


# --------------------------------------------------------------------------------------
# runner: search
# --------------------------------------------------------------------------------------


async def test_execute_search_applies_guards_dedup_and_events(stack, settings: Settings) -> None:
    runner = make_runner(stack, settings)
    rec = RecordingCtx()
    result = await runner.execute(
        one(ResearchIntent.retrieve_recent_news), IDENTITY, AS_OF, rec.ctx
    )
    assert isinstance(result, RoundResult)
    assert result.intent == "retrieve_recent_news" and result.kind == "search"
    kept_types = sorted(s.source_type for s in result.sources)
    assert kept_types == [
        "earnings_release",
        "financial_journalism",
        "financial_journalism",
        "secondary_commentary",
    ]
    assert len(result.evidence) == 4
    reasons = sorted(reason for _, reason in result.rejected)
    assert reasons == ["published_after_as_of", "thin_content"]
    leaked = next(s for s, reason in result.rejected if reason == "published_after_as_of")
    assert "reuters.com" in leaked.url and leaked.rejected_reason == "published_after_as_of"
    thin = next(s for s, reason in result.rejected if reason == "thin_content")
    assert thin.source_type == "investor_relations"
    assert result.duplicates == 2  # utm variant (canonical URL) + syndicated copy (content hash)
    assert all(s.published_at is not None and s.published_at <= AS_OF for s in result.sources)
    assert all(
        s.symbol == "AAPL" and s.research_intent == "retrieve_recent_news" for s in result.sources
    )
    assert all(s.freshness in ("current", "recent", "stale") for s in result.sources)
    assert all(s.redistribution == "metadata_only" for s in result.sources)
    for record in result.evidence:
        assert len(record.excerpt) <= EXCERPT_CHARS
        assert len(record.text) <= MAX_TEXT_CHARS
        assert record.source_type != "unverified_web"
    for source in result.sources:
        assert len(source.excerpt) <= EXCERPT_CHARS
    stats = runner.stats
    assert stats.search_rounds == 1
    assert stats.queries_issued == 1
    assert stats.sources_fetched == 7
    assert stats.sources_rejected == 2
    assert stats.duplicate_sources_removed == 2
    assert stats.intents == ["retrieve_recent_news"]
    assert stats.retrieval_total_ms > 0
    assert rec.ctx.timers.elapsed_ms["retrieval"] > 0
    queries = rec.named("research.query")
    assert queries == [
        {
            "intent": "retrieve_recent_news",
            "kind": "search",
            "query": '"Apple Inc." earnings OR guidance OR outlook',
            "label": "recent news for Apple Inc.",
            "round": 1,
        }
    ]
    found = rec.named("research.source_found")
    assert len(found) == 4
    assert {f["source_id"] for f in found} == {s.source_id for s in result.sources}
    assert all(f["intent"] == "retrieve_recent_news" and f["round"] == 1 for f in found)
    assert set(found[0]) == set(result.sources[0].public_view()) | {"intent", "round"}
    rejected = rec.named("research.source_rejected")
    assert len(rejected) == 2
    assert all(set(r) == {"url", "title", "reason", "intent", "round"} for r in rejected)
    assert rec.events[0][0] == "research.query"


async def test_execute_search_respects_max_fetch_per_round(stack, settings: Settings) -> None:
    runner = make_runner(stack, settings, max_fetch_per_round=2)
    rec = RecordingCtx()
    result = await runner.execute(
        one(ResearchIntent.retrieve_recent_news), IDENTITY, AS_OF, rec.ctx
    )
    assert runner.stats.sources_fetched == 2
    assert len(result.sources) + len(result.rejected) <= 2
    assert result.duplicates == 1  # the utm variant is skipped before any fetch


async def test_execute_search_skips_paywalled_pages(
    stack, settings: Settings, tmp_path: Path
) -> None:
    pages = json.loads((FIXTURE_DIR / "pages.json").read_text())
    searches = {
        "fixture": True,
        "searches": {
            "*": [
                {
                    "url": "https://www.wsj.com/fixture/paywalled-apple-story",
                    "title": "[Fixture] Paywalled",
                }
            ]
        },
    }
    (tmp_path / "pages.json").write_text(json.dumps(pages))
    (tmp_path / "searches.json").write_text(json.dumps(searches))
    fetcher = FixtureFetcher(FIXTURE_DIR)
    provider = FixtureResearchProvider(tmp_path, fetcher=fetcher)
    runner = ResearchRunner(provider, stack[1], stack[2], settings, ResearchBudget())
    rec = RecordingCtx()
    result = await runner.execute(
        one(ResearchIntent.retrieve_recent_news), IDENTITY, AS_OF, rec.ctx
    )
    assert result.sources == []
    assert [reason for _, reason in result.rejected] == ["paywalled"]
    assert result.rejected[0][0].source_type == "financial_journalism"


# --------------------------------------------------------------------------------------
# runner: structured retrieval
# --------------------------------------------------------------------------------------


async def test_execute_edgar_submissions_yields_filing_sources(stack, settings: Settings) -> None:
    runner = make_runner(stack, settings)
    rec = RecordingCtx()
    result = await runner.execute(
        one(ResearchIntent.retrieve_latest_filing), IDENTITY, AS_OF, rec.ctx
    )
    assert result.submissions is not None and result.submissions.fiscal_year_end == "0930"
    assert result.sources[0].title == "SEC EDGAR submissions"
    filings = result.sources[1:]
    assert len(filings) == 6
    assert all(s.source_type == "regulatory_filing" and s.publisher == "SEC EDGAR" for s in filings)
    assert all(s.published_at is not None and s.published_at <= AS_OF for s in filings)
    titles = [s.title for s in filings]
    assert "10-K filed 2025-10-31" in titles
    assert all(t.split(" filed ")[0] in ("10-K", "10-Q", "8-K") for t in titles)
    ten_k = next(s for s in filings if s.title == "10-K filed 2025-10-31")
    assert ten_k.fiscal_period == "FY2025"
    assert ten_k.url.endswith("0000320193-25-000123-index.htm")
    assert 0 < len(ten_k.excerpt) <= EXCERPT_CHARS and "FIXTURE" in ten_k.excerpt
    assert all(len(s.excerpt) <= EXCERPT_CHARS for s in filings)
    for record in result.evidence:
        assert record.extraction_method == "edgar_submissions"
        assert len(record.text) <= EXCERPT_CHARS  # never the whole filing
        assert record.structured["form"] in ("10-K", "10-Q", "8-K")
    assert any(
        "excerpt unavailable" in note for note in result.notes
    )  # missing primary docs degrade gracefully
    assert len(rec.named("research.source_found")) == 7


async def test_execute_edgar_companyfacts_returns_contract_rows(stack, settings: Settings) -> None:
    runner = make_runner(stack, settings)
    rec = RecordingCtx()
    result = await runner.execute(
        one(ResearchIntent.retrieve_earnings_history), IDENTITY, AS_OF, rec.ctx
    )
    assert len(result.facts_rows) > 100
    assert {"concept", "metric", "value", "unit", "end", "filed", "source_id"} <= set(
        result.facts_rows[0]
    )
    assert all(r["filed"] <= "2026-09-26" for r in result.facts_rows)
    assert len(result.sources) == 1
    source = result.sources[0]
    assert source.title == "SEC XBRL company facts" and source.source_type == "regulatory_filing"
    assert all(r["source_id"] == source.source_id for r in result.facts_rows)
    assert result.notes == ["8 XBRL rows filed after as_of were dropped"]
    assert runner.stats.queries_issued == 1 and runner.stats.sources_fetched == 1


async def test_execute_prices_and_benchmarks(stack, settings: Settings) -> None:
    runner = make_runner(stack, settings)
    rec = RecordingCtx()
    prices = await runner.execute(
        one(ResearchIntent.retrieve_price_history, "near_term"), IDENTITY, AS_OF, rec.ctx
    )
    assert prices.price_series is not None
    assert prices.price_series.symbol == "aapl.us" and prices.price_series.label == "AAPL"
    assert prices.price_series.exchange == "NASDAQ"
    assert prices.price_series.price_type == "latest_close"
    assert prices.price_series.session_date.isoformat() == "2026-09-25"
    assert prices.price_series.points[0].date.isoformat() >= "2025-08-22"
    assert prices.sources[0].source_type == "market_data"
    assert prices.sources[0].source_id == prices.price_series.source_id
    bench = await runner.execute(
        one(ResearchIntent.retrieve_sector_benchmark, "near_term"), IDENTITY, AS_OF, rec.ctx
    )
    assert [r.symbol for r in bench.benchmark_refs] == ["^SPX", "XLK"]
    assert set(bench.benchmark_series) == {"^SPX", "XLK"}
    assert bench.benchmark_series["XLK"].symbol == "xlk.us"
    assert len(bench.sources) == 2
    assert {s.metadata["benchmark_role"] for s in bench.sources} == {"broad_market", "sector"}
    assert runner.stats.queries_issued == 3
    assert runner.stats.sources_fetched == 3


async def test_benchmark_failure_is_rejected_not_fatal(stack, settings: Settings) -> None:
    runner = make_runner(stack, settings)
    rec = RecordingCtx()
    odd = InstrumentIdentity(
        symbol="AAPL", name="Apple Inc.", cik="320193", sic="6022"
    )  # XLF not in fixture
    planned = build_queries(ResearchIntent.retrieve_sector_benchmark, odd, "near_term", AS_OF)[0]
    assert planned.params["sic"] == "6022"
    result = await runner.execute(planned, odd, AS_OF, rec.ctx)
    assert set(result.benchmark_series) == {"^SPX"}
    assert [reason.split(":")[0] for _, reason in result.rejected] == ["fetch_failed"]
    assert result.rejected[0][0].url == "https://stooq.com/q/d/l/?s=xlf.us&i=d"
    assert runner.stats.sources_rejected == 1


async def test_prices_failure_is_rejected(stack, settings: Settings) -> None:
    runner = make_runner(stack, settings)
    rec = RecordingCtx()
    other = InstrumentIdentity(symbol="ZZZZ", name="Nowhere Corp", cik="1")
    result = await runner.execute(
        one(ResearchIntent.retrieve_price_history).model_copy(
            update={"params": {"symbol": "zzzz.us", "days": 30}}
        ),
        other,
        AS_OF,
        rec.ctx,
    )
    assert result.price_series is None
    assert [reason.split(":")[0] for _, reason in result.rejected] == ["fetch_failed"]


# --------------------------------------------------------------------------------------
# runner: budget, outage, cancellation
# --------------------------------------------------------------------------------------


async def test_max_sources_terminates_research(stack, settings: Settings) -> None:
    runner = make_runner(stack, settings, max_sources=2)
    rec = RecordingCtx()
    result = await runner.execute(
        one(ResearchIntent.retrieve_latest_filing), IDENTITY, AS_OF, rec.ctx
    )
    assert len(result.sources) == 2
    assert result.budget_exhausted is True
    assert runner.budget_exhausted is True
    assert runner.stats.termination_reason == "max_sources"
    fetched_before = runner.stats.sources_fetched
    more = await runner.execute(one(ResearchIntent.retrieve_recent_news), IDENTITY, AS_OF, rec.ctx)
    assert more.sources == [] and runner.stats.sources_fetched == fetched_before
    assert runner.kept_sources == 2
    assert runner.finish("plan_complete").termination_reason == "max_sources"


async def test_first_structured_outage_raises_research_unavailable(
    stack, settings: Settings
) -> None:
    runner = make_runner(stack, settings)
    rec = RecordingCtx()
    missing = InstrumentIdentity(symbol="NOPE", name="Nowhere Corp", cik="999999")
    with pytest.raises(AnalysisError) as info:
        await runner.execute(
            one(ResearchIntent.retrieve_latest_filing).model_copy(
                update={"params": {"cik": "999999"}}
            ),
            missing,
            AS_OF,
            rec.ctx,
        )
    assert info.value.code is ErrorCode.RESEARCH_UNAVAILABLE
    assert info.value.details["kind"] == "edgar_submissions"
    with pytest.raises(AnalysisError):
        await runner.execute(
            one(ResearchIntent.retrieve_earnings_history).model_copy(
                update={"params": {"cik": "999999"}}
            ),
            missing,
            AS_OF,
            rec.ctx,
        )


async def test_later_structured_outage_is_recorded_not_raised(stack, settings: Settings) -> None:
    runner = make_runner(stack, settings)
    rec = RecordingCtx()
    await runner.execute(one(ResearchIntent.retrieve_earnings_history), IDENTITY, AS_OF, rec.ctx)
    later = await runner.execute(
        one(ResearchIntent.retrieve_latest_filing).model_copy(update={"params": {"cik": "999999"}}),
        IDENTITY,
        AS_OF,
        rec.ctx,
    )
    assert later.sources == [] and later.submissions is None
    assert any("unavailable" in note for note in later.notes)


async def test_missing_cik_is_a_note_not_an_outage(stack, settings: Settings) -> None:
    runner = make_runner(stack, settings)
    rec = RecordingCtx()
    no_cik = InstrumentIdentity(symbol="X", name="X Corp")
    planned = PlannedQuery(
        kind="edgar_companyfacts", intent=ResearchIntent.retrieve_earnings_history, params={}
    )
    result = await runner.execute(planned, no_cik, AS_OF, rec.ctx)
    assert result.facts_rows == [] and "no CIK" in result.notes[0]


async def test_cancellation_is_checked_at_boundaries(stack, settings: Settings) -> None:
    runner = make_runner(stack, settings)
    rec = RecordingCtx()
    rec.ctx.cancel.cancel()
    with pytest.raises(AnalysisError) as info:
        await runner.execute(one(ResearchIntent.retrieve_recent_news), IDENTITY, AS_OF, rec.ctx)
    assert info.value.code is ErrorCode.CANCELLED


async def test_rounds_are_tracked_by_begin_round(stack, settings: Settings) -> None:
    runner = make_runner(stack, settings)
    rec = RecordingCtx()
    assert runner.begin_round() == 1
    await runner.execute(one(ResearchIntent.retrieve_price_history), IDENTITY, AS_OF, rec.ctx)
    assert runner.begin_round() == 2
    await runner.execute(one(ResearchIntent.retrieve_sector_benchmark), IDENTITY, AS_OF, rec.ctx)
    assert runner.stats.search_rounds == 2
    assert [q["round"] for q in rec.named("research.query")] == [1, 2]
    assert runner.stats.intents == ["retrieve_price_history", "retrieve_sector_benchmark"]


async def test_full_seed_plan_runs_on_fixture(stack, settings: Settings) -> None:
    runner = make_runner(stack, settings)
    rec = RecordingCtx()
    sources = []
    for intent in seed_plan("multi_horizon"):
        for planned in build_queries(intent, IDENTITY, "multi_horizon", AS_OF, []):
            result = await runner.execute(planned, IDENTITY, AS_OF, rec.ctx)
            sources.extend(result.sources)
    stats = runner.finish("plan_complete")
    assert stats.termination_reason == "plan_complete"
    assert stats.queries_issued == 7
    assert len(sources) == 7 + 1 + 1 + 2 + 4
    assert len({s.source_id for s in sources}) == len(sources)
    assert all(s.rejected_reason is None for s in sources)
    assert stats.sources_rejected == 2
    assert len(rec.named("research.source_found")) == len(sources)


# --------------------------------------------------------------------------------------
# providers / factory
# --------------------------------------------------------------------------------------


def test_build_research_stack_fixture_and_http(settings: Settings) -> None:
    provider, edgar, prices = build_research_stack(settings)
    assert isinstance(provider, FixtureResearchProvider)
    assert isinstance(provider, ResearchProvider)
    assert edgar is not None and prices is not None
    http_settings = Settings(research_provider="http", research_search_url="https://searx.local")
    http_provider, _, _ = build_research_stack(http_settings)
    assert isinstance(http_provider, HttpResearchProvider)
    assert isinstance(http_provider, ResearchProvider)
    assert http_provider.fetcher.user_agent == "BayAnalytics/0.1"
    with pytest.raises(ValueError, match="research_fixture_dir"):
        build_research_stack(Settings(research_provider="fixture"))


async def test_http_provider_closes_owned_client() -> None:
    provider = HttpResearchProvider(Settings(research_search_url=None))
    await provider.aclose()
    with pytest.raises(Exception, match="not configured"):
        await provider.search("anything")


async def test_fixture_provider_search_matching_and_fallback(settings: Settings) -> None:
    provider = FixtureResearchProvider(FIXTURE_DIR)
    exact = await provider.search('"Apple Inc." earnings OR guidance OR outlook')
    relaxed = await provider.search('  "apple inc."   EARNINGS or guidance OR outlook ')
    assert [r.url for r in exact] == [r.url for r in relaxed]
    assert len(exact) == 8
    fallback = await provider.search_with(
        "something unknown", categories="news", time_range="month"
    )
    assert [r.url for r in fallback] == [r.url for r in exact]
    narrowed = await provider.search('"Apple Inc." stock 2025 results')
    assert len(narrowed) == 3
    assert provider.queries[-1] == '"Apple Inc." stock 2025 results'
    page = await provider.open(
        "https://www.cnbc.com/2026/09/18/fixture-apple-quarter-preview.html?utm_source=x"
    )
    assert page.status == 200 and "bay-fixture" in page.body
    record = await provider.extract(
        "https://www.cnbc.com/2026/09/18/fixture-apple-quarter-preview.html"
    )
    assert record.publisher == "Fixture Wire (synthetic)"
    assert record.published_at == datetime(2026, 9, 18, 12, 30, tzinfo=UTC)


async def test_recording_provider_round_trips_into_fixture_layout(tmp_path: Path) -> None:
    inner = FixtureResearchProvider(FIXTURE_DIR)
    recorder = RecordingProvider(inner, tmp_path)
    results = await recorder.search_with(
        '"Apple Inc." earnings OR guidance OR outlook', categories="news"
    )
    record = await recorder.extract(results[0].url)
    fetcher = RecordingFetcher(FixtureFetcher(FIXTURE_DIR), tmp_path)
    page = await fetcher.open("https://data.sec.gov/submissions/CIK0000320193.json", ttl_s=10)
    assert page.status == 200
    replay = FixtureResearchProvider(tmp_path)
    replayed = await replay.search('"Apple Inc." earnings OR guidance OR outlook')
    assert [r.url for r in replayed] == [r.url for r in results]
    assert replayed[0].published_at == results[0].published_at
    again = await replay.extract(results[0].url)
    assert again.content_hash == record.content_hash and again.title == record.title
    submissions = await replay.open("https://data.sec.gov/submissions/CIK0000320193.json")
    assert json.loads(submissions.body)["name"] == "Apple Inc."
    pages = json.loads((tmp_path / "pages.json").read_text())
    assert pages["fixture"] is True and len(pages["pages"]) == 2
    searches = json.loads((tmp_path / "searches.json").read_text())
    assert searches["fixture"] is True and len(searches["searches"]) == 1


def test_fixture_files_are_marked_synthetic() -> None:
    for name in (
        "pages.json",
        "searches.json",
        "edgar/companyfacts_CIK0000320193.json",
        "edgar/submissions_CIK0000320193.json",
        "edgar/company_tickers.json",
    ):
        assert json.loads((FIXTURE_DIR / name).read_text())["fixture"] is True
    for html in list((FIXTURE_DIR / "news").glob("*.html")) + list(
        (FIXTURE_DIR / "edgar").glob("*.html")
    ):
        assert 'name="bay-fixture"' in html.read_text()


def test_fixture_csvs_regenerate_deterministically(tmp_path: Path) -> None:
    from bayanalytics.research.fixtures_build import build

    build(tmp_path)
    for name in (
        "prices/aapl.us.csv",
        "prices/spx.csv",
        "prices/xlk.us.csv",
        "edgar/companyfacts_CIK0000320193.json",
        "edgar/submissions_CIK0000320193.json",
        "pages.json",
        "searches.json",
    ):
        assert (tmp_path / name).read_bytes() == (FIXTURE_DIR / name).read_bytes(), name
