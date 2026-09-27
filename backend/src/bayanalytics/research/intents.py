"""Bounded research intents and deterministic query templates (AGENT.md sections 21, 28).

Laya never invents search strings: it chooses one of ``ResearchIntent`` and this module maps
that choice plus company identity, horizon and evidence gap to ``PlannedQuery`` objects. The
templates are deterministic so retrieval is reproducible for evaluation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from bayanalytics.instruments.base import InstrumentIdentity
from bayanalytics.research.dates import ensure_utc
from bayanalytics.research.prices import to_stooq_symbol
from bayanalytics.schemas.questions import AnalyticalRequirements

QueryKind = Literal["search", "edgar_submissions", "edgar_companyfacts", "prices", "benchmarks"]


class ResearchIntent(StrEnum):
    retrieve_latest_filing = "retrieve_latest_filing"
    retrieve_recent_news = "retrieve_recent_news"
    retrieve_historical_coverage = "retrieve_historical_coverage"
    retrieve_price_history = "retrieve_price_history"
    retrieve_sector_benchmark = "retrieve_sector_benchmark"
    retrieve_earnings_history = "retrieve_earnings_history"
    retrieve_guidance_history = "retrieve_guidance_history"
    retrieve_management_commentary = "retrieve_management_commentary"
    retrieve_missing_metric = "retrieve_missing_metric"
    stop_research = "stop_research"


class PlannedQuery(BaseModel):
    kind: QueryKind
    intent: ResearchIntent
    query: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    label: str = ""


SEED_PLANS: dict[str, list[ResearchIntent]] = {
    "near_term": [
        ResearchIntent.retrieve_recent_news,
        ResearchIntent.retrieve_price_history,
        ResearchIntent.retrieve_earnings_history,
        ResearchIntent.retrieve_sector_benchmark,
        ResearchIntent.retrieve_latest_filing,
    ],
    "next_cycle": [
        ResearchIntent.retrieve_latest_filing,
        ResearchIntent.retrieve_earnings_history,
        ResearchIntent.retrieve_guidance_history,
        ResearchIntent.retrieve_recent_news,
        ResearchIntent.retrieve_price_history,
    ],
    "medium_term": [
        ResearchIntent.retrieve_earnings_history,
        ResearchIntent.retrieve_latest_filing,
        ResearchIntent.retrieve_price_history,
        ResearchIntent.retrieve_sector_benchmark,
        ResearchIntent.retrieve_historical_coverage,
    ],
    "long_term": [
        ResearchIntent.retrieve_earnings_history,
        ResearchIntent.retrieve_latest_filing,
        ResearchIntent.retrieve_historical_coverage,
        ResearchIntent.retrieve_management_commentary,
        ResearchIntent.retrieve_price_history,
    ],
    "multi_horizon": [
        ResearchIntent.retrieve_latest_filing,
        ResearchIntent.retrieve_earnings_history,
        ResearchIntent.retrieve_price_history,
        ResearchIntent.retrieve_sector_benchmark,
        ResearchIntent.retrieve_recent_news,
        ResearchIntent.retrieve_guidance_history,
    ],
}

# Price-history window per horizon (calendar days). Benchmarks use the same window so
# relative returns line up.
PRICE_DAYS: dict[str, int] = {
    "near_term": 400,
    "next_cycle": 400,
    "medium_term": 800,
    "long_term": 5 * 366,
    "multi_horizon": 5 * 366,
}

MAX_GAP_QUERIES = 2

_GAP_KEYWORDS: tuple[tuple[tuple[str, ...], ResearchIntent], ...] = (
    (
        ("price", "return", "volatility", "drawdown", "momentum"),
        ResearchIntent.retrieve_price_history,
    ),
    (("benchmark", "sector", "index", "peer"), ResearchIntent.retrieve_sector_benchmark),
    (("guidance", "outlook", "forecast"), ResearchIntent.retrieve_guidance_history),
    (("filing", "10-k", "10-q", "8-k", "annual report"), ResearchIntent.retrieve_latest_filing),
    (
        ("commentary", "management", "transcript", "call", "ceo", "cfo"),
        ResearchIntent.retrieve_management_commentary,
    ),
    (("news", "event", "recent", "catalyst", "headline"), ResearchIntent.retrieve_recent_news),
    (
        ("history", "historical", "coverage", "prior_year", "analogue"),
        ResearchIntent.retrieve_historical_coverage,
    ),
    (
        (
            "revenue",
            "eps",
            "earnings",
            "margin",
            "net_income",
            "cash",
            "capex",
            "shares",
            "debt",
            "equity",
            "assets",
            "income",
            "profit",
        ),
        ResearchIntent.retrieve_earnings_history,
    ),
)


def gap_to_intent(gap: str) -> ResearchIntent:
    text = gap.strip().lower().replace(" ", "_")
    for keywords, intent in _GAP_KEYWORDS:
        if any(keyword.replace(" ", "_") in text for keyword in keywords):
            return intent
    return ResearchIntent.retrieve_missing_metric


def seed_plan(
    horizon: str, requirements: AnalyticalRequirements | None = None
) -> list[ResearchIntent]:
    """The first round's intents: what the question requires first, then the horizon seed.

    Required intents are deduplicated against the seed and each other and ``stop_research`` is
    never planned, so the plan is bounded by the intent set; execution is bounded by the
    research budget (sources, rounds, time), which the loop checks after every intent. Without
    requirements (or for ``general_assessment``, which requires nothing) the plan is exactly
    the horizon seed.
    """
    base = list(SEED_PLANS.get(horizon, SEED_PLANS["multi_horizon"]))
    if requirements is None or not requirements.required_research_intents:
        return base
    plan: list[ResearchIntent] = []
    for name in [*requirements.required_research_intents, *base]:
        intent = ResearchIntent(name)
        if intent is ResearchIntent.stop_research or intent in plan:
            continue
        plan.append(intent)
    return plan


def _display_name(identity: InstrumentIdentity) -> str:
    return identity.name.strip() or identity.symbol


def build_queries(
    intent: ResearchIntent | str,
    identity: InstrumentIdentity,
    horizon: str,
    as_of: datetime,
    gaps: list[str] | None = None,
) -> list[PlannedQuery]:
    """Deterministic templates: same inputs always yield the same planned queries."""
    intent = ResearchIntent(intent)
    name = _display_name(identity)
    year = ensure_utc(as_of).year
    days = PRICE_DAYS.get(horizon, PRICE_DAYS["multi_horizon"])
    gaps = [g for g in (gaps or []) if g and g.strip()]

    if intent is ResearchIntent.stop_research:
        return []
    if intent is ResearchIntent.retrieve_recent_news:
        return [
            PlannedQuery(
                kind="search",
                intent=intent,
                query=f'"{name}" earnings OR guidance OR outlook',
                params={"categories": "news", "time_range": "month"},
                label=f"recent news for {name}",
            )
        ]
    if intent is ResearchIntent.retrieve_historical_coverage:
        return [
            PlannedQuery(
                kind="search",
                intent=intent,
                query=f'"{name}" stock {year - 1} results',
                params={"time_range": "year"},
                label=f"historical coverage for {name} ({year - 1})",
            )
        ]
    if intent is ResearchIntent.retrieve_guidance_history:
        return [
            PlannedQuery(
                kind="search",
                intent=intent,
                query=f'"{name}" guidance raised OR lowered OR cut',
                params={"categories": "news", "time_range": "year"},
                label=f"guidance history for {name}",
            )
        ]
    if intent is ResearchIntent.retrieve_management_commentary:
        return [
            PlannedQuery(
                kind="search",
                intent=intent,
                query=f'"{name}" earnings call transcript CEO',
                params={"time_range": "year"},
                label=f"management commentary for {name}",
            )
        ]
    if intent is ResearchIntent.retrieve_missing_metric:
        planned = []
        for gap in gaps[:MAX_GAP_QUERIES]:
            gap_text = gap.strip().replace("_", " ")
            planned.append(
                PlannedQuery(
                    kind="search",
                    intent=intent,
                    query=f'"{name}" {gap_text} {year}',
                    params={"gap": gap},
                    label=f"missing metric {gap} for {name}",
                )
            )
        return planned
    if intent is ResearchIntent.retrieve_latest_filing:
        return [
            PlannedQuery(
                kind="edgar_submissions",
                intent=intent,
                params={"cik": identity.cik, "forms": ["10-K", "10-Q", "8-K"], "limit": 6},
                label=f"latest EDGAR filings for {name}",
            )
        ]
    if intent is ResearchIntent.retrieve_earnings_history:
        return [
            PlannedQuery(
                kind="edgar_companyfacts",
                intent=intent,
                params={"cik": identity.cik},
                label=f"XBRL company facts for {name}",
            )
        ]
    if intent is ResearchIntent.retrieve_price_history:
        return [
            PlannedQuery(
                kind="prices",
                intent=intent,
                params={
                    "symbol": to_stooq_symbol(identity.symbol, identity.exchange),
                    "days": days,
                    "exchange": identity.exchange,
                },
                label=f"daily prices for {identity.symbol}",
            )
        ]
    if intent is ResearchIntent.retrieve_sector_benchmark:
        return [
            PlannedQuery(
                kind="benchmarks",
                intent=intent,
                params={"sic": identity.sic, "days": days},
                label=f"benchmarks for {identity.symbol} (SIC {identity.sic or 'unknown'})",
            )
        ]
    return []
