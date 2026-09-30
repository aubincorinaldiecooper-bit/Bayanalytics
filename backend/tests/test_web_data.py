"""Data retrieval from search results, with Laya's bounded choices.

The synthetic "webdata" fixture (tests/fixtures/research/webdata, invented company NWND) returns
a news page, a daily price-history page, two quarterly-results pages and a CSV response with
S&P 500 values, each on its own website, plus the searches a company-name lookup runs. These
tests cover: Laya ordering the hits and choosing tables, columns and lines (and a declined or
unsure choice skipping the table); the price series and figures becoming the analysis's
evidence with their pages; the market events and result; the name -> ticker lookup; and the
POST contract for a question that names a company.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import json
import sys
from collections.abc import AsyncIterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from bayanalytics.calculations.registry import run_pack
from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.base import AnalysisRequest, InstrumentIdentity, ResearchBudget
from bayanalytics.instruments.equity import EquityAnalyzer
from bayanalytics.instruments.identity import InstrumentResolver, company_phrase, name_query
from bayanalytics.laya.schemas import (
    MAX_DYNAMIC_OPTIONS,
    MAX_OPTION_TEXT_CHARS,
    figure_line_questions,
    instrument_choice_questions,
    option_text,
)
from bayanalytics.laya.wrapper import LayaFinanceWrapper
from bayanalytics.main import create_app
from bayanalytics.research.intents import ResearchIntent, build_queries
from bayanalytics.research.market import fundamentals_view
from bayanalytics.research.runner import ResearchRunner
from bayanalytics.research.sources import canonical_url, domain_of
from bayanalytics.runtime import Runtime
from bayanalytics.schemas.common import ErrorCode, stable_id
from bayanalytics.wiring import build_runtime
from doubles import FixedTranscriber, RuleLaya, ScriptedSpark, fixture_research_stack

WEBDATA = Path(__file__).parent / "fixtures" / "research" / "webdata"
AS_OF = datetime(2026, 9, 26, tzinfo=UTC)
IDENTITY = InstrumentIdentity(symbol="NWND")
NEWS_URL = "https://www.fixturewire.example/markets/northwind-fixture-quarter-review"
PRICES_URL = "https://quotes.fixture-markets.example/nwnd/history"
QUARTERLY_URL = "https://www.fixture-financials.example/stocks/nwnd/financials/quarterly"
EARNINGS_URL = "https://www.fixture-earnings.example/nwnd/results-by-quarter"
SP500_URL = "https://data.fixture-index.example/sp500/daily.csv"


def _sid(url: str) -> str:
    return stable_id("src", canonical_url(url))


class Recorder:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.ctx = AnalysisContext(analysis_id="an_web", emit=self.sink)

    async def sink(self, name: str, data: dict[str, Any]) -> None:
        self.events.append((name, data))

    def names(self) -> list[str]:
        return [name for name, _ in self.events]

    def named(self, name: str) -> list[dict[str, Any]]:
        return [data for event, data in self.events if event == name]


def _settings(**overrides: Any) -> Settings:
    return Settings(research_min_request_interval_s=0.0, log_level="WARNING", **overrides)


def _runner(laya: RuleLaya | None = None) -> ResearchRunner:
    settings = _settings()
    return ResearchRunner(
        fixture_research_stack(settings, WEBDATA),
        settings,
        ResearchBudget(max_fetch_per_round=10, max_sources=24),
        laya=LayaFinanceWrapper(laya) if laya is not None else None,
    )


async def _execute(runner: ResearchRunner, intent: ResearchIntent, rec: Recorder) -> None:
    for planned in build_queries(intent, IDENTITY, "multi_horizon", AS_OF, []):
        await runner.execute(planned, IDENTITY, AS_OF, rec.ctx)


def _analyzer(laya: RuleLaya | None = None) -> EquityAnalyzer:
    settings = _settings()

    async def resolver_factory() -> InstrumentResolver:
        return InstrumentResolver()

    return EquityAnalyzer(
        settings,
        fixture_research_stack(settings, WEBDATA),
        LayaFinanceWrapper(laya or RuleLaya()),
        resolver_factory,
    )


def _request() -> AnalysisRequest:
    return AnalysisRequest(
        analysis_id="an_web",
        query="Assess $NWND.",
        profile="fast",
        requested_horizon="auto",
        resolved_horizon="multi_horizon",
        as_of=AS_OF,
        budget=ResearchBudget(),
    )


# ----------------------------------------------------------------------------- queries


def test_data_queries_are_topic_only_and_by_ticker() -> None:
    def query(intent: ResearchIntent) -> str:
        (planned,) = build_queries(intent, IDENTITY, "near_term", AS_OF, [])
        assert planned.params == {}  # evergreen data pages: no time range
        return planned.query or ""

    assert query(ResearchIntent.retrieve_price_history) == '"NWND" stock historical prices daily'
    assert query(ResearchIntent.retrieve_earnings_history) == (
        '"NWND" quarterly revenue gross profit earnings per share'
    )
    assert query(ResearchIntent.retrieve_sector_benchmark) == (
        "S&P 500 index historical prices daily"
    )
    assert name_query("apple") == '"apple" stock ticker symbol'


# ----------------------------------------------------------------------------- Laya choices


async def test_laya_orders_the_hits_before_they_are_opened() -> None:
    rec = Recorder()
    runner = _runner(RuleLaya())
    await _execute(runner, ResearchIntent.retrieve_price_history, rec)
    fetched = [e["url"] for e in rec.named("research.fetching")]
    assert fetched == [PRICES_URL, NEWS_URL]  # engine order is news first
    (decision,) = [d for d in runner.decisions if d.stage == "source_selection"]
    assert decision.decision_type == "open_order" and decision.decision == "r2"
    assert decision.segment_id == stable_id("search", '"NWND" stock historical prices daily')
    names = rec.names()
    assert names.index("laya.started") > names.index("research.search_results")
    assert names.index("laya.completed") < names.index("research.fetching")
    # Without Laya, or when Laya fails, the engine order stands and no table is read.
    for laya in (None, RuleLaya(raise_error=AnalysisError(ErrorCode.LAYA_INFERENCE_FAILED))):
        rec = Recorder()
        runner = _runner(laya)
        await _execute(runner, ResearchIntent.retrieve_price_history, rec)
        assert [e["url"] for e in rec.named("research.fetching")] == [NEWS_URL, PRICES_URL]
        assert rec.named("market.series") == [] and runner.market.prices() is None
        assert runner.decisions == []


async def test_laya_picks_the_price_table_and_close_column_then_the_series_is_kept() -> None:
    rec = Recorder()
    runner = _runner(RuleLaya())
    await _execute(runner, ResearchIntent.retrieve_price_history, rec)
    found = {e["url"]: e for e in rec.named("research.source_found")}
    preview = found[PRICES_URL]["preview"]
    assert preview["columns"] == ["Date", "Close"] and len(preview["rows"]) == 4
    assert preview["rows"][0][0] == "Sep 25, 2026"  # newest first, nothing after as_of
    assert found[NEWS_URL]["preview"] is None
    (series,) = rec.named("market.series")
    names = rec.names()
    assert names.index("market.series") > names.index("research.source_found")
    assert series["role"] == "company" and series["symbol"] == "NWND"
    assert series["source_id"] == _sid(PRICES_URL) and series["currency"] == "USD"
    points = series["points"]
    assert points[0][0] == "2025-06-02" and points[-1][0] == "2026-09-25"
    assert [p[0] for p in points] == sorted(p[0] for p in points)
    assert len(points[0]) == 6 and all(p[4] > 0 for p in points)
    choices = {d.decision_type: d for d in runner.decisions if d.stage == "data_identification"}
    assert choices["price_table"].decision == "t1"
    assert choices["close_column"].decision == "c5"  # "Close*", not "Adj Close**"
    assert choices["close_column"].segment_id == f"{_sid(PRICES_URL)}:t1"
    # "Download CSV" inside the page is never opened: only search hits are.
    assert all("download" not in e["url"] for e in rec.named("research.fetching"))


@pytest.mark.parametrize(
    "force",
    [
        {"close_column": ("c5", 0.4)},  # unsure: below the confidence floor
        {"price_table": "none"},  # declined
        {"close_column": "none"},
    ],
)
async def test_an_unsure_or_declined_choice_skips_the_table(force: dict[str, Any]) -> None:
    rec = Recorder()
    runner = _runner(RuleLaya(force=force))
    await _execute(runner, ResearchIntent.retrieve_price_history, rec)
    assert rec.named("market.series") == [] and runner.market.prices() is None
    found = {e["url"]: e for e in rec.named("research.source_found")}
    assert found[PRICES_URL]["preview"] is None  # kept for its text, no data read


async def test_a_csv_response_is_kept_for_its_data_and_is_the_market_series() -> None:
    rec = Recorder()
    runner = _runner(RuleLaya())
    await _execute(runner, ResearchIntent.retrieve_sector_benchmark, rec)
    (found,) = rec.named("research.source_found")
    assert found["url"] == SP500_URL and found["text_chars"] == 0
    assert found["preview"]["columns"] == ["Date", "Close"]
    (series,) = rec.named("market.series")
    assert series["role"] == "broad_market" and series["symbol"] == "S&P 500"
    assert series["points"][-1][0] == "2026-09-25"
    # Without Laya nothing is read, and a page with no text and no data stays rejected.
    rec = Recorder()
    await _execute(_runner(None), ResearchIntent.retrieve_sector_benchmark, rec)
    assert [e["reason"] for e in rec.named("research.source_rejected")] == ["thin_content"]


async def test_laya_picks_the_results_table_and_a_line_per_figure() -> None:
    rec = Recorder()
    runner = _runner(RuleLaya())
    await _execute(runner, ResearchIntent.retrieve_earnings_history, rec)
    found = {e["url"]: e for e in rec.named("research.source_found")}
    preview = found[QUARTERLY_URL]["preview"]
    assert preview["columns"] == ["Metric", "Period", "Value"]
    assert preview["rows"][0] == ["Revenue", "Q3 2026", "$22.8B"]
    lines = {
        d.decision_type: d.decision
        for d in runner.decisions
        if d.segment_id == f"{_sid(QUARTERLY_URL)}:t1:lines"
    }
    assert lines["line_revenue"] == "l1"  # "Revenue", not "Cost of Revenue"
    assert set(lines) >= {"line_gross_profit", "line_eps_diluted", "line_free_cash_flow"}
    pages = {page.source_id: page for page in runner.market.pages}
    assert set(pages) == {_sid(QUARTERLY_URL), _sid(EARNINGS_URL)}
    metrics = {f.metric for f in pages[_sid(QUARTERLY_URL)].figures}
    assert metrics == {
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "eps_diluted",
        "eps_basic",
        "operating_cash_flow",
        "capex",
        "free_cash_flow",
        "shares_outstanding",
    }
    # the future quarter and the TTM column are not figures
    assert all(f.end <= AS_OF.date() for page in pages.values() for f in page.figures)
    assert any("after 2026-09-26 dropped" in n for n in runner.market.notes)
    # A line Laya is unsure about is not read.
    runner = _runner(RuleLaya(force={"line_revenue": ("l1", 0.3)}))
    await _execute(runner, ResearchIntent.retrieve_earnings_history, Recorder())
    assert all(f.metric != "revenue" for page in runner.market.pages for f in page.figures)


# ----------------------------------------------------------------------------- the analysis


async def test_figures_and_prices_become_the_evidence_with_their_pages() -> None:
    analyzer = _analyzer()
    rec = Recorder()
    identity = await analyzer.identify("Assess $NWND.", rec.ctx)
    sources = await analyzer.retrieve(identity, _request(), rec.ctx)
    evidence = await analyzer.normalize(sources, rec.ctx)

    by_source = {f.source_id for f in evidence.facts}
    assert by_source == {_sid(QUARTERLY_URL), _sid(EARNINGS_URL)}
    revenue = [f for f in evidence.facts_for("revenue") if f.period.end == date(2026, 6, 27)]
    assert revenue and revenue[0].period.kind == "fiscal_quarter"
    assert revenue[0].period.label == "quarter to 2026-06-27"
    assert revenue[0].extraction_method == "web_html" and revenue[0].raw_value == "22,760"
    # The two pages disagree on one quarter: the existing conflict detection flags it.
    (conflict,) = evidence.conflicts
    assert conflict.metric == "revenue" and conflict.period_label == "quarter to 2026-03-28"
    assert {v.source_id for v in conflict.values} == {_sid(QUARTERLY_URL), _sid(EARNINGS_URL)}
    assert sorted(v.value for v in conflict.values) == [23_810e6, 24_520e6]
    notes = evidence.uncertainties
    assert notes[0] == (
        "Financial figures come from 2 web pages found by search "
        "(fixture-financials.example, fixture-earnings.example)."
    )
    assert notes[1].startswith("Daily prices for NWND come from one web page (fixture-markets")
    assert notes[2] == "S&P 500 daily prices come from another web page (fixture-index.example)."
    single = next(n for n in notes if n.startswith("Found on only one page"))
    assert "gross profit (9 periods, fixture-financials.example)" in single
    assert "revenue (4 periods, fixture-financials.example)" in single
    assert (
        "pages date one period differently (2025-09-27, 2025-09-30); treated as the period "
        "ending 2025-09-27"
    ) in notes
    assert not any("No verified financial figures" in n for n in notes)
    # Prices: the company's page and the S&P 500's page, different sites.
    assert evidence.prices is not None and evidence.prices.source_id == _sid(PRICES_URL)
    assert evidence.benchmarks["broad_market"].source_id == _sid(SP500_URL)

    calcs = {c.name: c for c in run_pack("all_standard", evidence, AS_OF)}
    for name in (
        "revenue_growth_yoy",
        "gross_margin",
        "operating_margin",
        "fcf_margin",
        "eps_growth_yoy",
        "pe_ttm",
        "market_cap",
        "price_return_1y",
        "volatility_1y_annualized",
        "max_drawdown_1y",
        "beta_1y_vs_market",
    ):
        assert calcs[name].status == "computed", (name, calcs[name].missing_inputs)
    close = evidence.prices.points[-1].close
    quarters = sorted(
        (f for f in evidence.facts_for("eps_diluted") if f.period.kind == "fiscal_quarter"),
        key=lambda f: f.period.end,  # type: ignore[arg-type,return-value]
    )
    eps_ttm = sum(f.value for f in quarters[-4:])
    assert calcs["pe_ttm"].value == pytest.approx(close / eps_ttm)
    shares = max(evidence.facts_for("shares_outstanding"), key=lambda f: f.period.end)  # type: ignore[arg-type,return-value]
    assert calcs["market_cap"].value == pytest.approx(close * shares.value)
    assert calcs["revenue_growth_yoy"].value == pytest.approx((22_760 / 21_870 - 1) * 100)
    assert calcs["beta_1y_vs_market"].value == pytest.approx(1.2, abs=0.1)
    inputs = {i.name: i for i in calcs["pe_ttm"].inputs}
    assert inputs["price"].source_id == _sid(PRICES_URL)

    fundamentals = fundamentals_view(evidence, AS_OF, analyzer.market_labels)
    assert fundamentals is not None and len(fundamentals.quarters) == 8
    assert [q.label for q in fundamentals.quarters][-2:] == ["Q2 2026", "Q3 2026"]
    assert fundamentals.quarters[-1].end == "2026-06-27"
    assert fundamentals.quarters[-1].gross_margin_pct == pytest.approx(9_150 / 22_760 * 100, 0.01)
    assert set(fundamentals.source_ids) <= {_sid(QUARTERLY_URL), _sid(EARNINGS_URL)}


def test_evidence_note_without_any_data_page_says_so() -> None:
    from bayanalytics.research.market import MarketData

    assert MarketData().data_notes("AAPL", 1, []) == [
        "Web search returned no page with price history or quarterly figures: this "
        "assessment is based only on the text of 1 web page found by search."
    ]


# ----------------------------------------------------------------------------- vertical slice


def _runtime(laya: RuleLaya | None = None) -> Runtime:
    settings = _settings(eval_as_of=AS_OF)
    return build_runtime(
        settings,
        laya=laya or RuleLaya(),
        spark=ScriptedSpark(),
        transcriber=FixedTranscriber(),
        research=fixture_research_stack(settings, WEBDATA),
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


def _parse_sse(raw: str) -> list[tuple[str, dict[str, Any]]]:
    events = []
    for block in raw.split("\n\n"):
        fields = dict(
            line.split(": ", 1) for line in block.splitlines() if line and not line.startswith(":")
        )
        if "event" in fields:
            events.append((fields["event"], json.loads(fields["data"])))
    return events


async def _run(rt: Runtime, query: str) -> tuple[int, list[tuple[str, dict]], dict]:
    async with _client(rt) as client:
        created = await client.post("/api/v1/analyses", json={"query": query})
        if created.status_code != 202:
            return created.status_code, [], created.json()
        analysis_id = created.json()["analysis_id"]
        async with client.stream("GET", f"/api/v1/analyses/{analysis_id}/events") as resp:
            raw = "".join([chunk async for chunk in resp.aiter_text()])
        await asyncio.sleep(0.02)
        result = (await client.get(f"/api/v1/analyses/{analysis_id}")).json()
        return created.status_code, _parse_sse(raw), result


async def test_vertical_slice_reads_prices_and_figures_from_search_results() -> None:
    status, events, result = await _run(_runtime(), "Assess $NWND.")
    assert status == 202 and result["status"] == "completed", result.get("error")
    names = [name for name, _ in events]
    series = [data for name, data in events if name == "market.series"]
    assert [s["role"] for s in series] == ["company", "broad_market"]
    assert domain_of(PRICES_URL) != domain_of(SP500_URL)
    assert [s["source_id"] for s in series] == [_sid(PRICES_URL), _sid(SP500_URL)]
    (fundamentals,) = [data for name, data in events if name == "market.fundamentals"]
    assert names.index("normalization.completed") < names.index("market.fundamentals")
    assert names.index("market.fundamentals") < names.index("calculation.started")
    assert len(fundamentals["quarters"]) == 8 and fundamentals["currency"] == "USD"
    # the result carries exactly what the events carried
    market = result["market"]
    assert [s["source_id"] for s in market["series"]] == [s["source_id"] for s in series]
    assert market["series"][0]["points"] == series[0]["points"]
    assert market["fundamentals"]["quarters"] == fundamentals["quarters"]
    assert "price_display" not in market
    # every page opened was a search hit, from four different sites plus the news site
    searched = json.loads((WEBDATA / "searches.json").read_text())["searches"]
    hits = {hit["url"] for results in searched.values() for hit in results}
    fetched = [data["url"] for name, data in events if name == "research.fetching"]
    assert set(fetched) <= hits
    assert {domain_of(s["url"]) for s in result["sources"]} == {
        "fixturewire.example",
        "fixture-markets.example",
        "fixture-financials.example",
        "fixture-earnings.example",
        "fixture-index.example",
    }
    previews = {
        data["url"]: data["preview"] for name, data in events if name == "research.source_found"
    }
    assert previews[NEWS_URL] is None and previews[PRICES_URL]["columns"] == ["Date", "Close"]
    assert previews[EARNINGS_URL]["columns"] == ["Metric", "Period", "Value"]
    calcs = {c["name"]: c for c in result["calculations"]}
    for name in ("pe_ttm", "market_cap", "revenue_growth_yoy", "beta_1y_vs_market"):
        assert calcs[name]["status"] == "computed", name
    stages = {d["stage"] for d in result["laya_decisions"]}
    assert {"source_selection", "data_identification", "research_plan"} <= stages
    assert len({d["decision_id"] for d in result["laya_decisions"]}) == len(
        result["laya_decisions"]
    )
    uncertainties = result["assessment"]["uncertainties"]
    assert any(u.startswith("Financial figures come from 2 web pages") for u in uncertainties)
    assert any(u.startswith("Found on only one page") for u in uncertainties)


# ----------------------------------------------------------------------------- name -> ticker


async def test_a_company_name_resolves_to_its_ticker_through_search_and_laya() -> None:
    for laya in (RuleLaya(), RuleLaya(force={"instrument_choice": "AAPL"})):
        analyzer = _analyzer(laya)
        rec = Recorder()
        identity = await analyzer.identify("Assess apple.", rec.ctx)
        assert identity.symbol == "AAPL" and identity.name == "Apple Inc."
        assert identity.resolution_method == "name_search" and identity.exchange == "NASDAQ"
        (query,) = rec.named("research.query")
        assert query["intent"] == "resolve_instrument"
        assert query["query"] == '"apple" stock ticker symbol'
        (results,) = rec.named("research.search_results")
        assert results["total"] == 4 and results["failed"] is False
        assert rec.names()[2:] == ["laya.started", "laya.decision", "laya.completed"]
        (decision,) = analyzer.identity_decisions
        assert decision.stage == "instrument_resolution"
        assert decision.decision_type == "instrument_choice" and decision.decision == "AAPL"
        assert set(decision.question.criteria or {}) == {"AAPL", "APLE", "none"}


async def test_a_ticker_in_the_question_wins_without_a_search() -> None:
    analyzer = _analyzer()
    rec = Recorder()
    identity = await analyzer.identify("Is $APLE cheaper than apple?", rec.ctx)
    assert identity.symbol == "APLE" and rec.events == []
    assert analyzer.provider.queries == []  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("query", "force", "symbols"),
    [
        ("Assess Delta.", {}, ["DFXA", "DFXP"]),  # tie: same number of sites
        ("Assess apple.", {"instrument_choice": ("AAPL", 0.5)}, ["AAPL", "APLE"]),  # unsure
        ("Assess apple.", {"instrument_choice": "none"}, ["AAPL", "APLE"]),  # declined
    ],
)
async def test_an_unclear_name_returns_the_candidates_as_found(
    query: str, force: dict[str, Any], symbols: list[str]
) -> None:
    analyzer = _analyzer(RuleLaya(force=force))
    with pytest.raises(AnalysisError) as info:
        await analyzer.identify(query, Recorder().ctx)
    error = info.value
    assert error.code is ErrorCode.AMBIGUOUS_INSTRUMENT
    assert error.details["reason"] == "multiple_companies"
    candidates = error.details["candidates"]
    assert [c["symbol"] for c in candidates] == symbols
    assert all(c["name"] for c in candidates)  # the names the results wrote


async def test_no_candidate_in_the_results_asks_for_the_ticker() -> None:
    analyzer = _analyzer()
    rec = Recorder()
    with pytest.raises(AnalysisError) as info:
        await analyzer.identify("Assess Zorblax.", rec.ctx)
    assert info.value.details == {"reason": "ticker_required", "candidates": []}
    assert rec.named("research.search_results")[0]["total"] == 2
    assert "laya.started" not in rec.names()  # nothing to choose from


async def test_post_accepts_a_company_name_and_the_analysis_resolves_it() -> None:
    status, events, result = await _run(_runtime(), "Assess Northwind.")
    assert status == 202
    names = [name for name, _ in events]
    assert names.index("research.query") < names.index("instrument.resolved")
    resolved = next(data for name, data in events if name == "instrument.resolved")
    assert resolved["symbol"] == "NWND" and resolved["name"] == "Northwind Fixture Foods"
    assert resolved["resolution_method"] == "name_search"
    assert result["status"] == "completed" and result["instrument"]["symbol"] == "NWND"
    assert result["laya_decisions"][0]["stage"] == "instrument_resolution"
    # nothing names a company: still answered at POST
    status, _events, body = await _run(_runtime(), "How is it doing?")
    assert status == 422 and body["error"]["details"]["reason"] == "ticker_required"
    # a name that cannot be told apart fails on the stream with the candidates
    status, events, result = await _run(_runtime(), "Assess Delta.")
    assert status == 202 and result["status"] == "failed"
    assert result["error"]["code"] == "AMBIGUOUS_INSTRUMENT"
    assert [c["symbol"] for c in result["error"]["details"]["candidates"]] == ["DFXA", "DFXP"]
    assert [d["stage"] for d in result["laya_decisions"]] == ["instrument_resolution"]
    assert events[-1][0] == "analysis.failed"


def test_company_phrase() -> None:
    assert company_phrase("Assess apple.") == "apple"
    assert company_phrase("How is Johnson & Johnson doing this quarter?") == "Johnson & Johnson"
    assert company_phrase("What do you think about Coca-Cola's margins?") == "Coca-Cola"
    assert company_phrase("How is it doing?") is None


# ----------------------------------------------------------------------------- schemas


def test_dynamic_questions_respect_laya_limits() -> None:
    too_many = [(f"S{i}", "Company") for i in range(MAX_DYNAMIC_OPTIONS + 1)]
    with pytest.raises(ValueError, match="at most"):
        instrument_choice_questions(too_many)
    with pytest.raises(ValueError, match="collides"):
        instrument_choice_questions([("none", "x")])
    long = option_text("x" * 200)
    assert len(long) == MAX_OPTION_TEXT_CHARS and long.endswith("...")
    assert option_text("a [MASK] b") == "a b"
    with pytest.raises(ValueError, match="unknown metric"):
        figure_line_questions({"ebitda": [("l1", "EBITDA")]})
    questions = figure_line_questions({"revenue": [("l1", "Total")]}, across=False)
    assert questions["line_revenue"].instructions == "Which column is total revenue?"
    assert list(questions["line_revenue"].criteria or {}) == ["l1", "none"]


# ----------------------------------------------------------------------------- fixture


def test_webdata_fixture_regenerates_deterministically(tmp_path: Path) -> None:
    path = WEBDATA.parent / "build_webdata.py"
    spec = importlib.util.spec_from_file_location("build_webdata", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    written = module.build(tmp_path)
    for path in written:
        name = path.relative_to(tmp_path).as_posix()
        assert path.read_bytes() == (WEBDATA / name).read_bytes(), name
    assert json.loads((WEBDATA / "pages.json").read_text())["fixture"] is True
    for html in (WEBDATA / "pages").glob("*.html"):
        assert 'name="bay-fixture"' in html.read_text()
