"""Finance decision schemas for Laya (AGENT.md sections 2, 3.2, 3.4, 21 and 28).

Every question here is a bounded, non-generative judgement in the exact shape
``@receptron/laya`` ``systemOne`` accepts. Builders return batches that share one compact state so
the orchestrator can answer a whole stage in a single forward pass. Laya's confidence on these
answers is confidence in the structured decision, never a market-outcome probability
(section 28). The whole module is validated against Laya's head/option limits at import time.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

from bayanalytics.laya.compaction import validate_questions
from bayanalytics.schemas.common import SINGLE_HORIZONS
from bayanalytics.schemas.decisions import LayaQuestion
from bayanalytics.schemas.questions import REQUIREMENT_LABELS, REQUIREMENT_NAMES

LAYA_SCHEMA_VERSION = "finance-v1"

# Stage names recorded on LayaDecision.stage.
STAGE_QUESTION_VALIDATION = "question_validation"
STAGE_RESEARCH_PLAN = "research_plan"
STAGE_EVIDENCE_SCAN = "evidence_scan"
STAGE_HISTORY_SCAN = "history_scan"
STAGE_CALCULATION = "calculation"
STAGE_HORIZON = "horizon"
STAGE_SYNTHESIS_GATE = "synthesis_gate"
# Dynamic choices during research: the options are things that exist in this run (company
# candidates found by a web search, the hits of a search, the tables and lines of a page).
STAGE_INSTRUMENT_RESOLUTION = "instrument_resolution"
STAGE_SOURCE_SELECTION = "source_selection"
STAGE_DATA_IDENTIFICATION = "data_identification"

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


# ---- question validation (Spark pass 1 proposes, Laya confirms or drops) -----------------

REQUIREMENTS_SUPPORTED_KEY = "requirements_supported"
REQUIREMENTS_SUPPORTED = _noul("The proposed requirements fit the question.")


def requirement_key(requirement: str) -> str:
    """The noul key that asks whether the question needs ``requirement``."""
    return f"requirement_{requirement}"


REQUIREMENT_QUESTIONS: dict[str, LayaQuestion] = {
    requirement_key(name): _noul(
        f"Answering the question requires {REQUIREMENT_LABELS[name].lower()}."
    )
    for name in REQUIREMENT_NAMES
}

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
    **REQUIREMENT_QUESTIONS,
    REQUIREMENTS_SUPPORTED_KEY: REQUIREMENTS_SUPPORTED,
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


def requirement_validation_questions(requirements: Iterable[str]) -> dict[str, LayaQuestion]:
    """Bounded validation of Spark's interpretation (stage ``question_validation``): one noul
    per requirement Spark proposed plus ``requirements_supported``.

    State: the question, the instrument, the horizon, the proposed intent's label and the
    proposed requirements' labels. Laya can only confirm or drop what was proposed; it never
    adds a requirement, plans research or computes anything.
    """
    keys: list[str] = []
    for requirement in requirements:
        if requirement not in REQUIREMENT_LABELS:
            raise ValueError(f"unknown requirement {requirement!r}")
        key = requirement_key(requirement)
        if key not in keys:
            keys.append(key)
    if not keys:
        raise ValueError("at least one proposed requirement is required")
    return _batch(*keys, REQUIREMENTS_SUPPORTED_KEY)


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


def history_segment_questions(*, include_volatility: bool = False) -> dict[str, LayaQuestion]:
    """Score one normalised historical period (section 3.4).

    Only questions the segment state can support are asked: the XBRL-derived state carries
    revenue, income, margins and growth, so momentum / margin direction / unusualness /
    materiality are always asked; ``volatility_regime`` only when the caller put a measured
    volatility for the period into the state. Guidance and sentiment per past period would
    need dated commentary for that period, which the vertical slice does not retrieve, so
    they are not asked.
    """
    keys = ["revenue_momentum", "margin_direction", "historically_unusual", "material_change"]
    if include_volatility:
        keys.append("volatility_regime")
    return _batch(*keys)


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


def overall_scan_questions() -> dict[str, LayaQuestion]:
    """One batch over the whole normalised evidence summary (orchestrator "evidence_scan").

    State: latest metrics, source/primary counts, conflicts, freshness warnings, headlines.
    """
    return _batch(
        "material_change",
        "guidance_trend",
        "sentiment_trend",
        "volatility_regime",
        "evidence_stance",
        "stale_evidence_matters",
        "calculation_pack",
    )


def text_evidence_questions() -> dict[str, LayaQuestion]:
    """Judge one retrieved text source: is it material, and which way does it point."""
    return _batch("source_is_material", "evidence_stance")


def horizon_context_questions(horizons: Iterable[str]) -> dict[str, LayaQuestion]:
    """Horizon stances plus the valuation / benchmark / drawdown / durability judgements that
    are only answerable once the deterministic calculations are in the state."""
    batch = horizon_questions(horizons)
    batch.update(
        _batch(
            "valuation_extremeness", "benchmark_relative", "drawdown_nature", "growth_durability"
        )
    )
    return batch


BUILDERS = (
    lambda: requirement_validation_questions(REQUIREMENT_NAMES),
    research_plan_questions,
    evidence_scan_questions,
    history_segment_questions,
    calculation_questions,
    lambda: horizon_questions(SINGLE_HORIZONS),
    synthesis_gate_questions,
    lambda: history_segment_questions(include_volatility=True),
    overall_scan_questions,
    text_evidence_questions,
    lambda: horizon_context_questions(SINGLE_HORIZONS),
)


# ---- dynamic choices (options built at run time from what actually exists) ---------------

NONE_OPTION = "none"
MAX_DYNAMIC_OPTIONS = 8
"""Options a dynamic choice offers besides ``none``: well under Laya's 20-option limit, and
short descriptions keep the head inside its 192 tokens (measured again at call time)."""
MAX_OPTION_TEXT_CHARS = 48
"""Characters of one option description (Laya keeps at most 48 tokens per option)."""

INSTRUMENT_CHOICE_KEY = "instrument_choice"
OPEN_ORDER_KEY = "open_order"
PRICE_TABLE_KEY = "price_table"
FIGURES_TABLE_KEY = "figures_table"
CLOSE_COLUMN_KEY = "close_column"

LINE_SUBJECTS: dict[str, str] = {
    "revenue": "total revenue",
    "gross_profit": "gross profit",
    "operating_income": "operating income",
    "net_income": "net income",
    "eps_diluted": "diluted earnings per share",
    "eps_basic": "basic earnings per share",
    "operating_cash_flow": "cash flow from operating activities",
    "capex": "capital expenditure",
    "free_cash_flow": "free cash flow",
    "shares_outstanding": "the number of shares outstanding",
}

_OPTION_WS = re.compile(r"\s+")


def line_key(metric: str) -> str:
    """The choice key that asks which line of a table reports ``metric``."""
    return f"line_{metric}"


def option_text(text: str) -> str:
    """One option description: whitespace collapsed, Laya's ``[MASK]`` marker removed, cut to
    ``MAX_OPTION_TEXT_CHARS`` characters."""
    flat = _OPTION_WS.sub(" ", str(text).replace("[MASK]", " ")).strip()
    if len(flat) > MAX_OPTION_TEXT_CHARS:
        flat = flat[: MAX_OPTION_TEXT_CHARS - 3].rstrip() + "..."
    return flat or "(blank)"


def dynamic_choice(
    instructions: str, options: Sequence[tuple[str, str]], none_text: str | None = None
) -> LayaQuestion:
    """A choice among ``options`` (``(key, description)``, at most ``MAX_DYNAMIC_OPTIONS``),
    plus ``none`` with ``none_text`` when given. Raises ``ValueError`` when the options break
    Laya's structural limits (too many, colliding keys); token lengths are measured when the
    question is asked."""
    if not options:
        raise ValueError("a dynamic choice needs at least one option")
    if len(options) > MAX_DYNAMIC_OPTIONS:
        raise ValueError(f"at most {MAX_DYNAMIC_OPTIONS} options, got {len(options)}")
    criteria = {str(key): option_text(text) for key, text in options}
    if none_text is not None:
        if NONE_OPTION in criteria:
            raise ValueError("an option key collides with 'none'")
        criteria[NONE_OPTION] = none_text
    question = _choice(instructions, criteria)
    validate_questions({"dynamic_choice": question})
    return question


def instrument_choice_questions(candidates: Sequence[tuple[str, str]]) -> dict[str, LayaQuestion]:
    """Stage ``instrument_resolution``: which listed company the analyst means, among the
    tickers a web search named (key: the symbol, description: the name as found).

    State: the company phrase from the question and each candidate's symbol, name and the
    number of distinct websites that named it.
    """
    return {
        INSTRUMENT_CHOICE_KEY: dynamic_choice(
            "Which listed company does the user mean?", candidates, "none of these companies"
        )
    }


def result_order_questions(topic: str, hits: Sequence[tuple[str, str]]) -> dict[str, LayaQuestion]:
    """Stage ``source_selection``: which search hit most likely contains ``topic``; the answer's
    probabilities order the hits (key ``r<n>``, description: site and title). No snippet is
    used as evidence."""
    return {
        OPEN_ORDER_KEY: dynamic_choice(f"Which search result most likely contains {topic}?", hits)
    }


def table_choice_questions(
    price_tables: Sequence[tuple[str, str]], figure_tables: Sequence[tuple[str, str]]
) -> dict[str, LayaQuestion]:
    """Stage ``data_identification``: which table of a page is the daily price history and
    which reports quarterly results (keys ``t<n>``, described by header row and row count)."""
    questions: dict[str, LayaQuestion] = {}
    if price_tables:
        questions[PRICE_TABLE_KEY] = dynamic_choice(
            "Which of these tables is the daily price history?",
            price_tables,
            "none of these tables",
        )
    if figure_tables:
        questions[FIGURES_TABLE_KEY] = dynamic_choice(
            "Which of these tables reports the company's quarterly results?",
            figure_tables,
            "none of these tables",
        )
    if not questions:
        raise ValueError("at least one kind of table is required")
    return questions


def close_column_questions(columns: Sequence[tuple[str, str]]) -> dict[str, LayaQuestion]:
    """Stage ``data_identification``: which column of the chosen price table is the closing
    price (keys ``c<n>``, described by the header cell as the page wrote it)."""
    return {
        CLOSE_COLUMN_KEY: dynamic_choice(
            "Which column is the closing price?", columns, "no closing price column"
        )
    }


def figure_line_questions(
    options: Mapping[str, Sequence[tuple[str, str]]], *, across: bool = True
) -> dict[str, LayaQuestion]:
    """Stage ``data_identification``: for each metric, which line of the chosen table reports
    it (rows when periods are columns, columns otherwise; keys ``l<n>``, described by the
    label as the page wrote it). Only metrics with a shortlist are asked."""
    noun = "row" if across else "column"
    questions: dict[str, LayaQuestion] = {}
    for metric, lines in options.items():
        if metric not in LINE_SUBJECTS:
            raise ValueError(f"unknown metric {metric!r}")
        if lines:
            questions[line_key(metric)] = dynamic_choice(
                f"Which {noun} is {LINE_SUBJECTS[metric]}?", lines, "not reported here"
            )
    if not questions:
        raise ValueError("at least one metric with options is required")
    return questions


def _full_dynamic_examples() -> list[dict[str, LayaQuestion]]:
    """Every dynamic builder at its largest option count, for the import-time guard."""
    many = [(f"x{i}", "option description of typical length") for i in range(MAX_DYNAMIC_OPTIONS)]
    return [
        instrument_choice_questions(many),
        result_order_questions("a table of the stock's daily prices", many),
        table_choice_questions(many, many),
        close_column_questions(many),
        figure_line_questions(dict.fromkeys(LINE_SUBJECTS, many)),
    ]


def validate_all_builders() -> None:
    """Run the head/option guard on every builder output (also executed at import time)."""
    validate_questions(ALL_QUESTIONS)
    for builder in BUILDERS:
        validate_questions(builder())
    for batch in _full_dynamic_examples():
        validate_questions(batch)


validate_all_builders()
