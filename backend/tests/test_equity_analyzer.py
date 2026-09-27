"""EquityAnalyzer: retrieval-state bookkeeping, Laya-directed planning, normalization details
(name changes, the dividend caveat, ranked text evidence), history segments with measured
volatility and the Spark bundle's analogues. Runs on the synthetic fixture with ``RuleLaya``.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from bayanalytics.calculations.primitives import annualized_volatility
from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext
from bayanalytics.instruments.base import (
    AnalysisRequest,
    CalculatedMetrics,
    InstrumentIdentity,
    LayaDecisions,
    ResearchBudget,
)
from bayanalytics.instruments.equity import EquityAnalyzer, RetrievalState
from bayanalytics.instruments.identity import InstrumentResolver
from bayanalytics.laya.wrapper import LayaFinanceWrapper
from bayanalytics.normalization.periods import label as period_label
from bayanalytics.research.intents import ResearchIntent
from bayanalytics.research.runner import RoundResult
from bayanalytics.schemas.common import stable_id, utcnow
from bayanalytics.schemas.decisions import LayaDecision, LayaQuestion, NoulAnswer
from bayanalytics.schemas.evidence import (
    EventSegment,
    NormalizedEvidence,
    Period,
    PriceSeries,
    SourceRecord,
)
from bayanalytics.schemas.results import ResearchStats
from doubles import RuleLaya, fixture_research_stack
from test_calculations import quarterly_facts, synthetic_prices
from test_store_memory import make_source

FIXTURES = Path(__file__).parent / "fixtures" / "research" / "apple"
AS_OF = datetime(2026, 9, 26, tzinfo=UTC)
IDENTITY = InstrumentIdentity(
    symbol="AAPL",
    exchange="NASDAQ",
    name="Apple Inc.",
    cik="320193",
    sic="3571",
    fiscal_year_end="0930",
)


def _settings() -> Settings:
    return Settings(research_contact_email="dev@example.com", research_min_request_interval_s=0.0)


def _analyzer(laya: RuleLaya | None = None) -> EquityAnalyzer:
    settings = _settings()
    stack = fixture_research_stack(settings, FIXTURES)
    edgar = stack[1]

    async def resolver_factory() -> InstrumentResolver:
        return InstrumentResolver(await edgar.company_tickers())

    return EquityAnalyzer(settings, stack, LayaFinanceWrapper(laya or RuleLaya()), resolver_factory)


def _bare() -> EquityAnalyzer:
    """An analyzer for the pure helpers: no research stack, no Laya."""
    analyzer = EquityAnalyzer(Settings(), (None, None, None), None, None)  # type: ignore[arg-type]
    analyzer.identity = IDENTITY
    return analyzer


def _request(horizon: str = "multi_horizon", **budget: Any) -> AnalysisRequest:
    return AnalysisRequest(
        analysis_id="an_equity",
        query="Assess Apple.",
        profile="fast",
        requested_horizon="auto",
        resolved_horizon=horizon,  # type: ignore[arg-type]
        as_of=AS_OF,
        budget=ResearchBudget(**budget),
    )


class Recorder:
    """Records events together with how many Laya calls had happened when each was emitted."""

    def __init__(self, laya: RuleLaya | None = None) -> None:
        self.laya = laya
        self.events: list[tuple[str, dict[str, Any], int]] = []
        self.ctx = AnalysisContext(analysis_id="an_equity", emit=self.sink)

    async def sink(self, name: str, data: dict[str, Any]) -> None:
        self.events.append((name, data, len(self.laya.calls) if self.laya else 0))

    def names(self) -> list[str]:
        return [name for name, _, _ in self.events]


def _src(source_id: str, source_type: str, method: str, excerpt: str = "x" * 50) -> SourceRecord:
    return make_source(source_id).model_copy(
        update={"source_type": source_type, "extraction_method": method, "excerpt": excerpt}
    )


# ------------------------------------------------------------------ retrieval state


def test_retrieval_state_absorbs_notes_once_and_counts_failures() -> None:
    state = RetrievalState()
    state.absorb(
        RoundResult(
            intent="retrieve_latest_filing",
            kind="edgar_submissions",
            notes=["EDGAR submissions unavailable (fetch_failed)"],
        )
    )
    state.absorb(
        RoundResult(
            intent="retrieve_earnings_history",
            kind="edgar_companyfacts",
            notes=[
                "EDGAR company facts unavailable (timeout)",
                "8 XBRL rows filed after as_of were dropped",
            ],
        )
    )
    # the same note twice is recorded once but counted every time it happens
    state.absorb(
        RoundResult(
            intent="retrieve_latest_filing",
            kind="edgar_submissions",
            notes=["EDGAR submissions unavailable (fetch_failed)"],
        )
    )
    assert state.notes == [
        "EDGAR submissions unavailable (fetch_failed)",
        "EDGAR company facts unavailable (timeout)",
        "8 XBRL rows filed after as_of were dropped",
    ]
    assert state.structured_failures == 3 and state.queries_failed == 0
    assert set(ResearchStats.model_fields) >= {"queries_failed", "structured_failures"}
    merged = _bare()._merge_stats(ResearchStats(), ["earnings_history"])
    assert merged.queries_failed == 0 and merged.structured_failures == 0
    assert merged.evidence_gaps_remaining == 1


def test_retrieval_state_counts_a_failed_search_as_a_query_failure() -> None:
    state = RetrievalState()
    state.absorb(
        RoundResult(
            intent="retrieve_recent_news",
            kind="search",
            notes=["search_failed: recent news for Apple Inc."],
        )
    )
    assert state.queries_failed == 1
    assert state.structured_failures == 0  # a web search outage is not an EDGAR/Stooq outage


def test_compute_gaps_reports_earnings_history_until_four_quarters_exist() -> None:
    analyzer = _bare()
    assert "earnings_history" in analyzer.compute_gaps("multi_horizon", AS_OF)
    analyzer.state.rows = [
        {"metric": "revenue", "fy": 2026, "fp": fp, "value": 1.0} for fp in ("Q1", "Q2", "Q3")
    ]
    assert "earnings_history" in analyzer.compute_gaps("multi_horizon", AS_OF)
    analyzer.state.rows.append({"metric": "revenue", "fy": 2025, "fp": "Q4", "value": 1.0})
    assert "earnings_history" not in analyzer.compute_gaps("multi_horizon", AS_OF)
    # annual rows never satisfy the quarterly requirement
    analyzer.state.rows = [
        {"metric": "revenue", "fy": fy, "fp": "FY", "value": 1.0} for fy in (2022, 2023, 2024, 2025)
    ]
    assert "earnings_history" in analyzer.compute_gaps("multi_horizon", AS_OF)


# ------------------------------------------------------------------ planning


async def test_plan_next_emits_laya_started_before_asking_and_refreshes_news_once() -> None:
    laya = RuleLaya(
        force={
            "research_intent": "stop_research",
            "evidence_sufficient": 0.9,
            "stale_evidence_matters": 0.9,
        }
    )
    analyzer = _analyzer(laya)
    rec = Recorder(laya)
    intent, sufficient = await analyzer._plan_next(IDENTITY, _request(), ["recent_news"], rec.ctx)
    names = rec.names()
    assert names[0] == "laya.started" and names[-1] == "laya.completed"
    assert names.count("laya.decision") == 3
    _started_name, started_data, calls_before = rec.events[0]
    assert started_data == {"stage": "research_plan", "questions": 3}
    assert calls_before == 0  # emitted before Laya was asked, not after
    assert rec.events[-1][1] == {"stage": "research_plan", "decisions": 3}
    assert len(analyzer.research_decisions) == 3
    # Laya judged the evidence stale: one refresh of recent coverage, and sufficiency is capped
    # so the loop does not stop on the very answer that asked for fresher evidence.
    assert intent is ResearchIntent.retrieve_recent_news and sufficient == 0.5
    assert [n for n in analyzer.state.notes if "stale enough to matter" in n] == [
        "Laya judged the available evidence stale enough to matter; recent coverage was refreshed"
    ]
    # Once recent news was retrieved the staleness is noted with the confidence, not refreshed.
    analyzer.state.executed.append(str(ResearchIntent.retrieve_recent_news))
    intent, sufficient = await analyzer._plan_next(IDENTITY, _request(), [], rec.ctx)
    assert intent is ResearchIntent.stop_research and sufficient == 0.9
    assert any(
        n.endswith("(confidence 0.90); treat the assessment with caution")
        for n in analyzer.state.notes
    )

    # Below the threshold Laya's stop is honoured and nothing is noted ...
    calm = _analyzer(
        RuleLaya(force={"research_intent": "stop_research", "stale_evidence_matters": 0.3})
    )
    intent, _ = await calm._plan_next(IDENTITY, _request(), [], Recorder().ctx)
    assert intent is ResearchIntent.stop_research and calm.state.notes == []
    # ... unless a gap remains whose intent has not been tried yet.
    intent, _ = await calm._plan_next(IDENTITY, _request(), ["price_history"], Recorder().ctx)
    assert intent is ResearchIntent.retrieve_price_history
    calm.state.executed.append(str(ResearchIntent.retrieve_price_history))
    intent, _ = await calm._plan_next(IDENTITY, _request(), ["price_history"], Recorder().ctx)
    assert intent is ResearchIntent.stop_research


# ------------------------------------------------------------------ normalization


def test_text_evidence_excludes_structured_and_market_data_sources() -> None:
    records = [
        _src("src_px", "market_data", "csv"),
        _src("src_xbrl", "regulatory_filing", "json", "312 XBRL facts across 8 metrics"),
        _src("src_subs", "regulatory_filing", "json", "Apple Inc. (CIK 320193)"),
        _src("src_10k", "regulatory_filing", "edgar_submissions"),
        _src("src_news", "financial_journalism", "html_readability_v1"),
        _src("src_empty", "financial_journalism", "html_readability_v1", ""),
    ]
    items = _bare()._text_evidence(records)
    assert [i["source_id"] for i in items] == ["src_10k", "src_news"]
    assert all(set(i) >= {"fact", "rank", "is_primary", "freshness", "published_at"} for i in items)


def test_text_evidence_is_ranked_by_source_then_newest_first() -> None:
    def dated(source_id: str, source_type: str, day: int | None) -> SourceRecord:
        published = datetime(2026, 8, day, tzinfo=UTC) if day else None
        return _src(source_id, source_type, "html_readability_v1").model_copy(
            update={"published_at": published}
        )

    records = [
        dated("src_old_news", "financial_journalism", 1),
        dated("src_blog", "secondary_commentary", 25),
        dated("src_new_news", "financial_journalism", 20),
        dated("src_undated", "financial_journalism", None),
        dated("src_10k", "regulatory_filing", 3),
        dated("src_release", "earnings_release", 2),
    ]
    items = _bare()._text_evidence(records)
    assert [i["source_id"] for i in items] == [
        "src_10k",
        "src_release",
        "src_new_news",
        "src_old_news",
        "src_undated",
        "src_blog",
    ]
    assert [i["rank"] for i in items] == sorted(i["rank"] for i in items)


async def test_normalize_wires_name_changes_and_the_dividend_caveat() -> None:
    analyzer = _analyzer()
    rec = Recorder()
    identity = IDENTITY.model_copy()
    analyzer.identity = identity
    await analyzer.enrich_identity(identity, rec.ctx)
    assert identity.sector == "Electronic Computers" and identity.fiscal_year_end == "0930"
    sources = await analyzer.retrieve(identity, _request(), rec.ctx)
    assert sources and analyzer.stats.queries_failed == 0
    # two filings in the fixture have no primary document: their excerpt fetches fail
    assert analyzer.stats.structured_failures == 2
    assert sum("excerpt unavailable" in n for n in analyzer.state.notes) == 2
    evidence = await analyzer.normalize(sources, rec.ctx)
    # EDGAR's former names become name-change actions (the entity is unchanged) ...
    actions = evidence.corporate_actions
    assert [a.kind for a in actions] == ["name_change", "name_change"]
    assert actions[0].detail == "formerly APPLE COMPUTER INC"
    assert actions[0].effective == date(2007, 1, 4)
    assert actions[1].detail == "formerly APPLE COMPUTER INC/ FA"
    assert actions[1].effective == date(1997, 7, 28)
    # ... with the mild warning, and prices carry the total-return caveat.
    renamed = [u for u in evidence.uncertainties if u.startswith("name change on 2007-01-04")]
    assert len(renamed) == 1 and "the reporting entity is unchanged" in renamed[0]
    assert evidence.prices is not None
    assert "price returns exclude dividends (price return, not total return)" in (
        evidence.uncertainties
    )
    assert len(evidence.uncertainties) == len(set(evidence.uncertainties))
    # segments carry stable ids derived from the symbol and the period label
    assert evidence.segments
    for segment in evidence.segments:
        assert segment.segment_id == stable_id("seg", "AAPL", period_label(segment.period))
    # Without a price series there is no dividend caveat to give.
    analyzer.state.price_series = None
    bare = await analyzer.normalize(sources, rec.ctx)
    assert bare.prices is None
    assert not any("exclude dividends" in u for u in bare.uncertainties)


async def test_enrich_identity_records_former_names_not_dict_reprs() -> None:
    analyzer = _analyzer()
    identity = IDENTITY.model_copy()
    await analyzer.enrich_identity(identity, Recorder().ctx)
    assert identity.ticker_history == ["APPLE COMPUTER INC", "APPLE COMPUTER INC/ FA"]


# ------------------------------------------------------------------ segments


def _quarter_closes(prices: PriceSeries, period: Period) -> list[float]:
    assert period.start is not None and period.end is not None
    return [p.close for p in prices.points if period.start <= p.date <= period.end]


def test_segments_measure_volatility_only_with_twenty_closes() -> None:
    analyzer = _bare()
    facts = quarterly_facts()
    prices = synthetic_prices("AAPL", 200.0, 0.0006, 1.0, points=400)  # ~19 months of closes
    segments = analyzer._segments(facts, prices)
    assert [s.period.label for s in segments] == [
        "Q4 FY2024",
        "Q1 FY2025",
        "Q2 FY2025",
        "Q3 FY2025",
        "Q4 FY2025",
        "Q1 FY2026",
        "Q2 FY2026",
        "Q3 FY2026",
    ]
    with_vol = [s for s in segments if "volatility_annualized_pct" in s.summary]
    without = [s for s in segments if "volatility_annualized_pct" not in s.summary]
    assert with_vol and without
    for segment in segments:
        closes = _quarter_closes(prices, segment.period)
        if len(closes) >= 20:
            expected = annualized_volatility(closes)
            assert expected is not None
            assert segment.summary["volatility_annualized_pct"] == round(expected * 100, 2)
        else:
            assert "volatility_annualized_pct" not in segment.summary
        assert segment.segment_id == stable_id("seg", "AAPL", period_label(segment.period))
        assert segment.summary["period"] == period_label(segment.period)
        assert segment.summary["revenue"] > 0 and "operating_margin_pct" in segment.summary
    # growth needs the same fiscal quarter a year earlier
    assert "revenue_growth_yoy" not in segments[0].summary
    assert segments[-1].summary["revenue_growth_yoy"] == pytest.approx(
        (1 + 0.02 * 7) / (1 + 0.02 * 3) * 100 - 100, abs=0.01
    )
    assert analyzer._segments(facts, None) and not any(
        "volatility_annualized_pct" in s.summary for s in analyzer._segments(facts, None)
    )
    # Laya is asked about the volatility regime only where a measured volatility exists.
    evidence = NormalizedEvidence(symbol="AAPL", as_of=AS_OF, segments=segments)
    history_sets = [
        s for s in analyzer.build_laya_questions(evidence, _request()) if s.stage == "history_scan"
    ]
    assert len(history_sets) == len(segments)
    for question_set, segment in zip(history_sets, segments, strict=True):
        assert question_set.segment_id == segment.segment_id
        assert ("volatility_regime" in question_set.questions) == (
            "volatility_annualized_pct" in segment.summary
        )
        assert question_set.state["instrument"] == "AAPL"


# ------------------------------------------------------------------ spark bundle


def _segment(
    index: int, growth: float, margin: float, volatility: float | None = None
) -> EventSegment:
    period = Period(
        kind="fiscal_quarter",
        fiscal_year=2024 + index // 4,
        fiscal_period=f"Q{index % 4 + 1}",
        end=date(2024 + index // 4, 3 * (index % 4) + 3, 28),
        label=f"Q{index % 4 + 1} FY{2024 + index // 4}",
    )
    summary: dict[str, Any] = {
        "period": period.label,
        "revenue": 100.0 + index,
        "revenue_growth_yoy": growth,
        "operating_margin_pct": margin,
    }
    if volatility is not None:
        summary["volatility_annualized_pct"] = volatility
    return EventSegment(
        segment_id=f"seg_{index}", period=period, summary=summary, source_ids=["src_xbrl"]
    )


def _flag(segment_id: str, decision_type: str, noul: float) -> LayaDecision:
    return LayaDecision(
        decision_id=f"dec_{segment_id}_{decision_type}",
        stage="history_scan",
        decision_type=decision_type,
        question=LayaQuestion(type="noul", instructions="?"),
        answer=NoulAnswer(noul=noul),
        confidence=max(noul, 1 - noul),
        state_digest="d",
        segment_id=segment_id,
        created_at=utcnow(),
    )


def test_spark_bundle_fills_historical_analogues_deterministically() -> None:
    analyzer = _bare()
    segments = [
        _segment(0, 9.0, 31.0),  # distance 2
        _segment(1, 20.0, 30.0),  # distance 10 -> beyond the top three
        _segment(2, 12.0, 28.0),  # distance 4
        _segment(3, 11.0, 40.0),  # distance 11, Laya found it unusual -> 6
        _segment(4, 10.0, 30.0),  # the latest period: never its own analogue
    ]
    decisions = LayaDecisions(decisions=[_flag("seg_3", "historically_unusual", 1.0)])
    evidence = NormalizedEvidence(symbol="AAPL", as_of=AS_OF, segments=segments)
    bundle = analyzer.build_spark_bundle(evidence, decisions, CalculatedMetrics(), _request())
    analogues = bundle.historical_analogues
    assert [a["period"] for a in analogues] == ["Q1 FY2024", "Q3 FY2024", "Q4 FY2024"]
    assert [a["similarity_distance"] for a in analogues] == [2.0, 4.0, 6.0]
    assert [a["laya_material_or_unusual"] for a in analogues] == [0.0, 0.0, 1.0]
    for analogue in analogues:
        assert analogue["method"].startswith("deterministic nearest-neighbour")
        assert analogue["source_ids"] == ["src_xbrl"] and "period" not in analogue["summary"]
        assert analogue["label"] == analogue["period"]
    again = analyzer.build_spark_bundle(evidence, decisions, CalculatedMetrics(), _request())
    assert again.historical_analogues == analogues
    # measured volatility takes part in the distance when both periods have one
    with_vol = [
        _segment(0, 10.0, 30.0, volatility=60.0),
        _segment(1, 10.0, 30.0, volatility=20.0),
        _segment(2, 10.0, 30.0, volatility=22.0),
    ]
    bundle = analyzer.build_spark_bundle(
        NormalizedEvidence(symbol="AAPL", as_of=AS_OF, segments=with_vol),
        LayaDecisions(),
        CalculatedMetrics(),
        _request(),
    )
    assert [a["period"] for a in bundle.historical_analogues] == ["Q2 FY2024", "Q1 FY2024"]
    assert [a["similarity_distance"] for a in bundle.historical_analogues] == [1.0, 19.0]
    # fewer than two periods with growth: nothing to compare against
    single = NormalizedEvidence(symbol="AAPL", as_of=AS_OF, segments=[_segment(0, 5.0, 30.0)])
    bundle = analyzer.build_spark_bundle(single, LayaDecisions(), CalculatedMetrics(), _request())
    assert bundle.historical_analogues == []
    assert bundle.instrument["symbol"] == "AAPL" and bundle.horizons == [
        "near_term",
        "next_cycle",
        "medium_term",
        "long_term",
    ]
