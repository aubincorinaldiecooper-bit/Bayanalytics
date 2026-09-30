"""Bounded research intents and deterministic query templates (AGENT.md sections 21, 28).

Laya never invents search strings: it chooses one of ``ResearchIntent`` and this module maps
that choice plus company identity, horizon and evidence gap to ``PlannedQuery`` objects. The
templates are deterministic so retrieval is reproducible for evaluation.

Every intent is a web search. Templates describe the topic only: no ``site:`` operators and no
provider, site or brand names (``tests/test_source_policy.py`` checks this), so what is read is
whatever the search engine returns. The data intents (price history, earnings history, the
market benchmark) search for pages that carry the numbers as tables, by ticker: naming an
index ("S&P 500") is a topic, not a source. There is no sector without a company directory, so
the benchmark is the broad market only.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from bayanalytics.instruments.base import InstrumentIdentity
from bayanalytics.research.dates import ensure_utc
from bayanalytics.schemas.questions import AnalyticalRequirements

QueryKind = Literal["search"]


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
    """The first round's intents: the question's required intents, then the horizon seed.

    Deduplicated in that order; ``stop_research`` is never planned, so the plan is bounded by
    the intent set and execution by the research budget (sources, rounds, time), which the
    loop checks after every intent. Without required intents (a general assessment) the plan
    is exactly the horizon seed.
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


BROAD_MARKET_QUERY = "S&P 500 index historical prices daily"

# What each intent's pages should hold, in words (Laya's question when it orders the hits).
INTENT_TOPICS: dict[ResearchIntent, str] = {
    ResearchIntent.retrieve_latest_filing: "the company's latest annual or quarterly report",
    ResearchIntent.retrieve_recent_news: "recent news about the company",
    ResearchIntent.retrieve_historical_coverage: "coverage of the company's past results",
    ResearchIntent.retrieve_price_history: "a table of the stock's daily prices",
    ResearchIntent.retrieve_sector_benchmark: "a table of the S&P 500 index's daily prices",
    ResearchIntent.retrieve_earnings_history: "a table of the company's quarterly results",
    ResearchIntent.retrieve_guidance_history: "changes to the company's guidance",
    ResearchIntent.retrieve_management_commentary: "management remarks on an earnings call",
    ResearchIntent.retrieve_missing_metric: "the figure the analysis is missing",
    ResearchIntent.stop_research: "nothing further",
}


def intent_topic(intent: ResearchIntent | str, gap: str | None = None) -> str:
    """The topic the hits of ``intent``'s search should hold (a missing metric names its gap)."""
    resolved = ResearchIntent(intent)
    if resolved is ResearchIntent.retrieve_missing_metric and gap:
        return f"the company's {gap.strip().replace('_', ' ')} figures"
    return INTENT_TOPICS[resolved]


def build_queries(
    intent: ResearchIntent | str,
    identity: InstrumentIdentity,
    horizon: str,
    as_of: datetime,
    gaps: list[str] | None = None,
) -> list[PlannedQuery]:
    """Deterministic templates: same inputs always yield the same planned web searches."""
    intent = ResearchIntent(intent)
    name = _display_name(identity)
    symbol = identity.symbol
    year = ensure_utc(as_of).year
    gaps = [g for g in (gaps or []) if g and g.strip()]

    def search(query: str, label: str, **params: Any) -> list[PlannedQuery]:
        return [PlannedQuery(kind="search", intent=intent, query=query, params=params, label=label)]

    if intent is ResearchIntent.stop_research:
        return []
    if intent is ResearchIntent.retrieve_recent_news:
        return search(
            f'"{name}" earnings OR guidance OR outlook',
            f"recent news for {name}",
            categories="news",
            time_range="month",
        )
    if intent is ResearchIntent.retrieve_historical_coverage:
        return search(
            f'"{name}" stock {year - 1} results',
            f"historical coverage for {name} ({year - 1})",
            time_range="year",
        )
    if intent is ResearchIntent.retrieve_guidance_history:
        return search(
            f'"{name}" guidance raised OR lowered OR cut',
            f"guidance history for {name}",
            categories="news",
            time_range="year",
        )
    if intent is ResearchIntent.retrieve_management_commentary:
        return search(
            f'"{name}" earnings call transcript CEO',
            f"management commentary for {name}",
            time_range="year",
        )
    if intent is ResearchIntent.retrieve_latest_filing:
        return search(
            f'"{name}" annual report 10-K OR quarterly report 10-Q {year}',
            f"latest annual or quarterly report for {name}",
            time_range="year",
        )
    # Data pages are evergreen tables, so these searches carry no time range; rows after
    # ``as_of`` are dropped when the tables are read.
    if intent is ResearchIntent.retrieve_earnings_history:
        return search(
            f'"{symbol}" quarterly revenue gross profit earnings per share',
            f"quarterly figures for {symbol}",
        )
    if intent is ResearchIntent.retrieve_price_history:
        return search(
            f'"{symbol}" stock historical prices daily',
            f"daily price history for {symbol}",
        )
    if intent is ResearchIntent.retrieve_sector_benchmark:
        return search(BROAD_MARKET_QUERY, "daily S&P 500 history (broad-market benchmark)")
    if intent is ResearchIntent.retrieve_missing_metric:
        planned: list[PlannedQuery] = []
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
    return []
