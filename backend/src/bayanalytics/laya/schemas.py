"""Finance decision schemas for Laya (AGENT.md sections 2, 3.2, 3.4, 21 and 28).

Every question here is a bounded, non-generative judgement in the exact shape
``@receptron/laya`` ``systemOne`` accepts. Builders return batches that share one compact state so
the orchestrator can answer a whole stage in a single forward pass. Laya's confidence on these
answers is confidence in the structured decision, never a market-outcome probability
(section 28). The whole module is validated against Laya's head/option limits at import time.
"""

from __future__ import annotations

from collections.abc import Iterable

from bayanalytics.laya.compaction import validate_questions
from bayanalytics.schemas.common import SINGLE_HORIZONS
from bayanalytics.schemas.decisions import LayaQuestion

LAYA_SCHEMA_VERSION = "finance-v1"

# Stage names recorded on LayaDecision.stage.
STAGE_RESEARCH_PLAN = "research_plan"
STAGE_EVIDENCE_SCAN = "evidence_scan"
STAGE_HISTORY_SCAN = "history_scan"
STAGE_CALCULATION = "calculation"
STAGE_HORIZON = "horizon"
STAGE_SYNTHESIS_GATE = "synthesis_gate"

# Bounded research intents (section 21). The research layer maps each to a deterministic query
# template; Laya only ever picks one of these.
RESEARCH_INTENTS: tuple[str, ...] = (
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
)

STANCES: tuple[str, ...] = ("bullish", "neutral", "bearish", "mixed")
CALCULATION_PACKS: tuple[str, ...] = (
    "growth_and_margins",
    "valuation_vs_history",
    "returns_vs_benchmark",
    "volatility_and_drawdown",
    "all_standard",
)

VALUATION_LEVELS: tuple[str, ...] = (
    "well below historical norm",
    "below historical norm",
    "near historical norm",
    "above historical norm",
    "far above historical norm",
)
REVENUE_MOMENTUM_LEVELS: tuple[str, ...] = (
    "contracting sharply",
    "contracting",
    "flat",
    "growing",
    "accelerating strongly",
)
GROWTH_DURABILITY_LEVELS: tuple[str, ...] = (
    "very fragile",
    "fragile",
    "uncertain",
    "durable",
    "very durable",
)

_STANCE_CRITERIA: dict[str, str] = {
    "bullish": "evidence points to improvement",
    "neutral": "no clear direction",
    "bearish": "evidence points to deterioration",
    "mixed": "credible evidence conflicts",
}


def _choice(instructions: str, criteria: dict[str, str]) -> LayaQuestion:
    return LayaQuestion(type="choice", instructions=instructions, criteria=dict(criteria))


def _score(instructions: str, levels: Iterable[str]) -> LayaQuestion:
    return LayaQuestion(type="score", instructions=instructions, criteria=list(levels))


def _noul(instructions: str) -> LayaQuestion:
    return LayaQuestion(type="noul", instructions=instructions)


# ---- research planning (section 21) --------------------------------------------------------

# Kept terse on purpose: ten described options must fit Laya's 192-token head with margin.
RESEARCH_INTENT = _choice(
    "Which evidence should be retrieved next?",
    {
        "retrieve_latest_filing": "newest periodic filing",
        "retrieve_recent_news": "news from recent weeks",
        "retrieve_historical_coverage": "coverage of a past period",
        "retrieve_price_history": "past prices and returns",
        "retrieve_sector_benchmark": "sector or peer data",
        "retrieve_earnings_history": "past quarterly results",
        "retrieve_guidance_history": "past guidance revisions",
        "retrieve_management_commentary": "earnings-call remarks",
        "retrieve_missing_metric": "a metric a calculation needs",
        "stop_research": "evidence is sufficient",
    },
)

EVIDENCE_SUFFICIENT = _noul(
    "Is the collected evidence sufficient, current and sourced well enough to assess this company?"
)
STALE_EVIDENCE_MATTERS = _noul(
    "Would the age of the available evidence materially weaken a current assessment?"
)

# ---- evidence scanning (sections 2, 22, 30) -----------------------------------------------

SOURCE_IS_MATERIAL = _noul(
    "Is this source material to assessing the company, given its type, date and content?"
)
MATERIAL_CHANGE = _noul("Does this new evidence materially change the current company assessment?")
ESCALATE_TO_SPARK = _noul(
    "Does this evidence need reasoning about conflicts or context beyond a bounded judgement?"
)
EVIDENCE_STANCE = _choice(
    "Taken together, is this evidence bullish, neutral, bearish or mixed for the company?",
    _STANCE_CRITERIA,
)
GUIDANCE_TREND = _choice(
    "Is management guidance deteriorating, unchanged or improving versus the prior period?",
    {
        "deteriorating": "guidance cut or outlook weakened",
        "unchanged": "guidance reiterated",
        "improving": "guidance raised or outlook strengthened",
    },
)
SENTIMENT_TREND = _choice(
    "Is sentiment in coverage and commentary weakening, stable or improving?",
    {
        "weakening": "tone turning more negative",
        "stable": "tone broadly unchanged",
        "improving": "tone turning more positive",
    },
)

# ---- historical segments (section 3.4) -----------------------------------------------------

REVENUE_MOMENTUM = _score(
    "Rate revenue momentum in this period from sharp contraction to strong acceleration.",
    REVENUE_MOMENTUM_LEVELS,
)
MARGIN_DIRECTION = _choice(
    "Are operating margins contracting, stable or expanding in this period?",
    {
        "contracting": "margins falling",
        "stable": "margins roughly flat",
        "expanding": "margins rising",
    },
)
VOLATILITY_REGIME = _choice(
    "Is realised price volatility low, normal or elevated relative to the company's own history?",
    {
        "low": "calmer than usual",
        "normal": "in line with history",
        "elevated": "well above usual",
    },
)
HISTORICALLY_UNUSUAL = _noul(
    "Is this period historically unusual for the company compared with its own record?"
)

# ---- calculations (section 3.3) ------------------------------------------------------------

CALCULATION_PACK = _choice(
    "Which deterministic calculation pack should run next given the available inputs?",
    {
        "growth_and_margins": "revenue growth and margin change",
        "valuation_vs_history": "multiples versus own history",
        "returns_vs_benchmark": "returns relative to a benchmark",
        "volatility_and_drawdown": "realised volatility and drawdowns",
        "all_standard": "every standard calculation",
    },
)
VALUATION_EXTREMENESS = _score(
    "Where does the current valuation sit relative to the company's own historical range?",
    VALUATION_LEVELS,
)
BENCHMARK_RELATIVE = _choice(
    "Over the stated period, did the stock underperform, track or outperform its benchmark?",
    {
        "underperforming": "clearly behind the benchmark",
        "in_line": "close to the benchmark",
        "outperforming": "clearly ahead of the benchmark",
    },
)
DRAWDOWN_NATURE = _choice(
    "Is the recent drawdown idiosyncratic to the company, market-wide or a mix of both?",
    {
        "idiosyncratic": "company-specific causes",
        "market_wide": "broad market or sector move",
        "mixed": "both company and market causes",
    },
)
GROWTH_DURABILITY = _score(
    "How durable is the company's growth over the coming years given demand and competition?",
    GROWTH_DURABILITY_LEVELS,
)

# ---- horizon stances (section 28) ---------------------------------------------------------

_HORIZON_INSTRUCTIONS: dict[str, str] = {
    "near_term": (
        "Stance for the next days to several weeks. Weigh volatility, upcoming events, "
        "recent news, current momentum and earnings proximity."
    ),
    "next_cycle": (
        "Stance for the next earnings report or quarter. Weigh guidance, revenue trajectory, "
        "margins, demand indicators and recent operational changes."
    ),
    "medium_term": (
        "Stance for the next 6-12 months. Weigh valuation, growth durability, competitive "
        "pressure, macro exposure and execution risk."
    ),
    "long_term": (
        "Stance for the coming years. Weigh market structure, competitive advantages, "
        "capital intensity, business quality and structural risks."
    ),
}

HORIZON_STANCE: dict[str, LayaQuestion] = {
    horizon: _choice(text, _STANCE_CRITERIA) for horizon, text in _HORIZON_INSTRUCTIONS.items()
}


def horizon_stance_key(horizon: str) -> str:
    return f"horizon_stance_{horizon}"


ALL_QUESTIONS: dict[str, LayaQuestion] = {
    "research_intent": RESEARCH_INTENT,
    "evidence_sufficient": EVIDENCE_SUFFICIENT,
    "stale_evidence_matters": STALE_EVIDENCE_MATTERS,
    "source_is_material": SOURCE_IS_MATERIAL,
    "material_change": MATERIAL_CHANGE,
    "escalate_to_spark": ESCALATE_TO_SPARK,
    "evidence_stance": EVIDENCE_STANCE,
    "guidance_trend": GUIDANCE_TREND,
    "sentiment_trend": SENTIMENT_TREND,
    "revenue_momentum": REVENUE_MOMENTUM,
    "margin_direction": MARGIN_DIRECTION,
    "volatility_regime": VOLATILITY_REGIME,
    "historically_unusual": HISTORICALLY_UNUSUAL,
    "calculation_pack": CALCULATION_PACK,
    "valuation_extremeness": VALUATION_EXTREMENESS,
    "benchmark_relative": BENCHMARK_RELATIVE,
    "drawdown_nature": DRAWDOWN_NATURE,
    "growth_durability": GROWTH_DURABILITY,
    **{horizon_stance_key(h): q for h, q in HORIZON_STANCE.items()},
}


def _batch(*keys: str) -> dict[str, LayaQuestion]:
    return {key: ALL_QUESTIONS[key].model_copy(deep=True) for key in keys}


# ---- builders: one batch per shared state -------------------------------------------------


def research_plan_questions() -> dict[str, LayaQuestion]:
    """Decide the next bounded research intent and whether the loop can stop (section 21).

    State: instrument identity, the question and horizon, evidence gaps, source counts,
    freshness of what has been collected so far.
    """
    return _batch("research_intent", "evidence_sufficient", "stale_evidence_matters")


def evidence_scan_questions() -> dict[str, LayaQuestion]:
    """Judge one normalised evidence item or a small batch of facts from one source."""
    return _batch(
        "source_is_material",
        "material_change",
        "evidence_stance",
        "guidance_trend",
        "sentiment_trend",
        "escalate_to_spark",
    )


def history_segment_questions() -> dict[str, LayaQuestion]:
    """Score one normalised historical period or event (section 3.4)."""
    return _batch(
        "revenue_momentum",
        "margin_direction",
        "guidance_trend",
        "sentiment_trend",
        "volatility_regime",
        "historically_unusual",
        "material_change",
    )


def calculation_questions() -> dict[str, LayaQuestion]:
    """Pick the calculation pack and classify the exact metrics once computed (section 3.3)."""
    return _batch(
        "calculation_pack",
        "valuation_extremeness",
        "benchmark_relative",
        "drawdown_nature",
    )


def horizon_questions(horizons: Iterable[str]) -> dict[str, LayaQuestion]:
    """Stance per requested horizon (section 28). ``multi_horizon`` expands to all four."""
    requested: list[str] = []
    for horizon in horizons:
        if horizon == "multi_horizon":
            requested.extend(SINGLE_HORIZONS)
        elif horizon in HORIZON_STANCE:
            requested.append(horizon)
        else:
            raise ValueError(
                f"unknown horizon {horizon!r}; expected one of "
                f"{', '.join(SINGLE_HORIZONS)} or multi_horizon"
            )
    if not requested:
        raise ValueError("at least one horizon is required")
    ordered = [h for h in SINGLE_HORIZONS if h in requested]
    return _batch(*(horizon_stance_key(h) for h in ordered))


def synthesis_gate_questions() -> dict[str, LayaQuestion]:
    """Final bounded checks before Spark: sufficiency, escalation, overall stance, durability."""
    return _batch(
        "evidence_sufficient",
        "stale_evidence_matters",
        "escalate_to_spark",
        "evidence_stance",
        "historically_unusual",
        "growth_durability",
    )


BUILDERS = (
    research_plan_questions,
    evidence_scan_questions,
    history_segment_questions,
    calculation_questions,
    lambda: horizon_questions(SINGLE_HORIZONS),
    synthesis_gate_questions,
)


def validate_all_builders() -> None:
    """Run the head/option guard on every builder output (also executed at import time)."""
    validate_questions(ALL_QUESTIONS)
    for builder in BUILDERS:
        validate_questions(builder())


validate_all_builders()
