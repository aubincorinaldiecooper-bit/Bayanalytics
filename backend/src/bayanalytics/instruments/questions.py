"""Deterministic question classification and the per-kind analytical requirements table.

The gap this closes: the analyst's question must shape the research plan and the calculations,
not only the Spark prompt. The flow is

    query -> classify_question (rules) -> [Laya question_scan when unclear] -> requirements
          -> seed_plan / compute_gaps -> calculations -> check_requirements -> Spark focus

``classify_question`` is keyword/regex scoring with a confidence and the matched cues; it says
``unclear`` when nothing fires or two kinds tie, and it is *low-confidence* when the leading kind
beats a competing kind by only one point. Both cases are handed to Laya, which picks one of the
same bounded kinds (:func:`bayanalytics.laya.schemas.question_kind_questions`); Laya never plans
research or computes anything. ``REQUIREMENTS`` maps every kind to the research intents, the
registry calculations and the operands (canonical metric names, ``prices``, ``benchmark``,
``sector_benchmark``) the question needs, the focus sentence Spark receives and the horizons to
emphasise. ``general_assessment`` requires nothing beyond the horizon seed plan and the
Laya-chosen calculation pack, so an unclassified "Assess X." behaves exactly as before.
``check_requirements`` reports, after the calculations, which requirements were met; unmet ones
are uncertainties, never failures.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bayanalytics.calculations.registry import CANONICAL_METRICS, SPECS
from bayanalytics.instruments.base import CalculatedMetrics
from bayanalytics.research.intents import ResearchIntent
from bayanalytics.schemas.decisions import ChoiceAnswer, LayaDecision, NoulAnswer
from bayanalytics.schemas.evidence import NormalizedEvidence, PriceSeries
from bayanalytics.schemas.questions import (
    QUESTION_KIND_LABELS,
    QUESTION_KINDS,
    UNCLEAR,
    AnalyticalRequirements,
    MissingRequirement,
    QuestionClassification,
    RequirementsReport,
)

LOW_CONFIDENCE = 0.6
"""Below this the rules' verdict is handed to Laya (a leading kind with a close competitor)."""
LAYA_CONFIDENCE_FLOOR = 0.4
"""Below this Laya's pick is not trusted to narrow the analysis: a general assessment runs."""
RECENT_PERIOD_THRESHOLD = 0.5

PRICE_OPERAND = "prices"
BENCHMARK_OPERAND = "benchmark"
SECTOR_BENCHMARK_OPERAND = "sector_benchmark"
_SERIES_OPERANDS: dict[str, str] = {
    PRICE_OPERAND: "price_history",
    BENCHMARK_OPERAND: "sector_benchmark",
    SECTOR_BENCHMARK_OPERAND: "sector_benchmark",
}
"""Non-fact operands and the evidence-gap label the research loop already uses for them."""
OPERAND_NAMES: frozenset[str] = frozenset(CANONICAL_METRICS) | frozenset(_SERIES_OPERANDS)


# ---- rules ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cue:
    kind: str
    weight: int
    pattern: re.Pattern[str]


def _cue(kind: str, weight: int, pattern: str) -> _Cue:
    return _Cue(kind, weight, re.compile(pattern, re.IGNORECASE))


# Weight 2 (or 3) for phrasings that name the kind, 1 for words that merely lean towards it.
# A rule counts once per query however often it matches.
_RULES: tuple[_Cue, ...] = (
    # general assessment
    _cue("general_assessment", 2, r"\bassess(?:ment)?\b"),
    _cue("general_assessment", 2, r"\banaly[sz]e\b|\banalysis\b"),
    _cue("general_assessment", 2, r"\boverview\b|\bevaluate\b|\bevaluation\b"),
    _cue("general_assessment", 2, r"\bhow (?:is|are) .{0,40}?\b(?:doing|performing|faring)\b"),
    _cue("general_assessment", 2, r"\bhow (?:does|do) .{0,40}?\blook\b"),
    _cue("general_assessment", 2, r"\bwhat do you think\b|\bthoughts on\b|\btell me about\b"),
    _cue("general_assessment", 2, r"\bshould i (?:buy|sell|hold|invest|own)\b"),
    _cue("general_assessment", 2, r"\bgood (?:investment|stock|company|buy)\b"),
    _cue("general_assessment", 1, r"\bbull(?:ish)? (?:or|and|vs\.?|versus) bear(?:ish)?\b"),
    # thesis change
    _cue("thesis_change", 3, r"\bthesis\b"),
    _cue(
        "thesis_change",
        3,
        r"\bstill (?:a |an )?(?:buy|sell|hold|bullish|bearish|attractive|compelling|intact|"
        r"worth (?:it|owning|holding)|the same story)\b",
    ),
    _cue(
        "thesis_change",
        2,
        r"\bchange[ds]? (?:the |my |our |your |its )?(?:story|picture|view|case|narrative|"
        r"outlook|investment case)\b",
    ),
    _cue("thesis_change", 2, r"\b(?:has|did|does|is) (?:anything|something|the story) chang"),
    _cue("thesis_change", 2, r"\bwhat(?:'s| has| is)? changed\b|\bwhat changed\b"),
    _cue("thesis_change", 2, r"\bupdate (?:the |my |your |our )?(?:view|case|call)\b"),
    _cue("thesis_change", 1, r"\bnew information\b|\bstill (?:hold|holds|stands|stand)\b"),
    # valuation
    _cue("valuation", 2, r"\b(?:over|under)valued\b|\bvaluation\b|\bfair value\b"),
    _cue("valuation", 2, r"\bexpensive\b|\bcheap(?:ly)?\b|\bpricey\b|\bstretched\b"),
    _cue(
        "valuation",
        2,
        r"\bp/?e\b|\bp/s\b|\bev ?/ ?ebitda\b|\bprice[- ]to[- ](?:earnings|sales|book|cash)\b",
    ),
    _cue("valuation", 2, r"\b(?:fcf|free cash flow) yield\b|\bearnings yield\b"),
    _cue("valuation", 1, r"\bmultiples?\b|\bpremium\b|\bdiscount\b"),
    _cue("valuation", 1, r"\bpriced (?:in|for)\b|\bmarket cap\b|\bworth (?:the|its) price\b"),
    # growth
    _cue("growth", 2, r"\bgrow(?:th|ing|s|n)?\b"),
    _cue("growth", 2, r"\bcagr\b|\bsales trend\b|\brevenue (?:trend|trajectory)\b"),
    _cue("growth", 1, r"\btop[- ]line\b|\baccelerat|\bdecelerat|\bscaling\b"),
    # profitability and margins
    _cue("profitability_margins", 2, r"\bmargins?\b|\bprofitab(?:le|ility)\b"),
    _cue("profitability_margins", 2, r"\b(?:gross|operating|net) (?:income|profit|margin)\b"),
    _cue("profitability_margins", 1, r"\bprofits?\b|\bearnings quality\b|\bbottom[- ]line\b"),
    _cue("profitability_margins", 1, r"\bfree cash flow\b(?! yield)|\bfcf\b(?! yield)"),
    _cue("profitability_margins", 1, r"\bcost (?:structure|base|control|cutting|discipline)\b"),
    # relative performance
    _cue(
        "relative_performance",
        2,
        r"\b(?:vs\.?|versus|against|relative to|compared (?:to|with)) (?:the |its )?"
        r"(?:s&p|s&p ?500|sp500|spx|nasdaq|dow|market|index|sector|peers?|benchmark|"
        r"competitors?|rivals?|industry)\b",
    ),
    _cue("relative_performance", 2, r"\b(?:out|under)perform"),
    _cue("relative_performance", 2, r"\bbeta\b|\brelative (?:return|performance|strength)\b"),
    _cue("relative_performance", 1, r"\bs&p ?500\b|\bspx\b|\bnasdaq\b|\bbenchmark\b|\bpeers?\b"),
    _cue("relative_performance", 1, r"\bversus\b|\bvs\.?\b|\bcompared (?:to|with)\b|\bsector\b"),
    # risk and volatility
    _cue("risk_volatility", 2, r"\bvolatil|\bdrawdowns?\b|\bhow (?:safe|risky|volatile)\b"),
    _cue(
        "risk_volatility",
        1,
        r"\brisk(?:y|s|ier|iest)?\b|\bdownside\b|\bsafety\b|"
        r"\bsafe (?:stock|investment|bet|company|to (?:buy|own|hold))\b",
    ),
    _cue("risk_volatility", 1, r"\bsell[- ]?off\b|\bcrash(?:ed|ing)?\b|\bcollapse\b"),
    # guidance and outlook
    _cue("guidance_outlook", 2, r"\bguidance\b|\boutlook\b|\bguided?\b"),
    _cue("guidance_outlook", 1, r"\bforecasts?\b|\bexpect(?:s|ed|ations?)?\b|\bconsensus\b"),
    _cue("guidance_outlook", 1, r"\bmanagement (?:said|says|expects|commentary|tone)\b"),
    _cue("guidance_outlook", 1, r"\bwhat(?:'s| is) (?:next|ahead|coming)\b"),
    # earnings reaction
    _cue(
        "earnings_reaction",
        2,
        r"\bwhy (?:did|has|is|was|were) .{0,50}?\b(?:drop|dropped|fall|fell|fallen|plunge[d]?|"
        r"sink|sank|slide|slid|tank(?:ed)?|crash(?:ed)?|rally|rallied|jump(?:ed)?|surge[d]?|"
        r"soar(?:ed)?|spike[d]?|pop(?:ped)?|move[d]?|down|up)\b",
    ),
    _cue("earnings_reaction", 2, r"\b(?:earnings|results) (?:reaction|beat|miss|surprise)\b"),
    _cue(
        "earnings_reaction",
        2,
        r"\b(?:beat|missed?|topped|exceeded) (?:estimates|expectations|consensus|the street)\b",
    ),
    _cue("earnings_reaction", 2, r"\bpost[- ]earnings\b|\bmarket(?:'s)? reaction\b"),
    _cue("earnings_reaction", 2, r"\bhow did the (?:stock|market|shares) (?:react|take|respond)"),
    _cue(
        "earnings_reaction",
        1,
        r"\bafter (?:the |its |their |last |latest |recent |this )?"
        r"(?:earnings|results|report|quarter|print|call)\b",
    ),
    _cue(
        "earnings_reaction",
        1,
        r"\b(?:latest|last|most recent|recent|this|newest) (?:quarter|earnings|results|print|"
        r"report|numbers)\b",
    ),
    _cue(
        "earnings_reaction", 1, r"\bq[1-4](?:\s*(?:fy)?\s*\d{2,4})? (?:results|earnings|numbers)\b"
    ),
    # balance sheet and liquidity
    _cue("balance_sheet_liquidity", 2, r"\bbalance sheet\b|\bliquidity\b|\bsolven(?:t|cy)\b"),
    _cue("balance_sheet_liquidity", 2, r"\bdebt\b|\bleverage[d]?\b|\bindebted"),
    _cue("balance_sheet_liquidity", 2, r"\bnet (?:cash|debt)\b|\bcash burn\b|\brunway\b"),
    _cue("balance_sheet_liquidity", 2, r"\bcash (?:position|pile|balance|reserves|on hand)\b"),
    _cue("balance_sheet_liquidity", 2, r"\bbankrupt"),
    _cue("balance_sheet_liquidity", 1, r"\bcredit (?:rating|risk|quality)\b|\bcovenants?\b"),
    # dividends and capital return
    _cue("dividends_capital_return", 2, r"\bdividends?\b|\bpayout\b"),
    _cue("dividends_capital_return", 2, r"\bbuy-?backs?\b|\brepurchases?\b"),
    _cue("dividends_capital_return", 2, r"\bcapital (?:return|allocation)\b"),
    _cue("dividends_capital_return", 2, r"\bshareholder (?:returns?|yield)\b"),
    _cue("dividends_capital_return", 1, r"\bincome (?:stock|investor|play)\b"),
    _cue("dividends_capital_return", 1, r"(?<!fcf )(?<!free cash flow )\byield\b"),
)

_RECENT_PERIOD: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:latest|last|most recent|recent|this|newest) (?:quarter|earnings|results|print|"
        r"report|call|filing|10-?[qk]|numbers)\b",
        r"\bafter (?:the |its |their |last |latest |recent |this )?"
        r"(?:earnings|results|report|quarter|print|call)\b",
        r"\bpost[- ]earnings\b",
        r"\bq[1-4]\b",
    )
)

_WS = re.compile(r"\s+")
_FRAMING_KIND = "general_assessment"
"""Its cues frame the request; any specific kind that scored outranks it."""


def _normalise(query: str) -> str:
    return _WS.sub(" ", str(query or "")).strip()


def classify_question(query: str, resolved_horizon: str | None = None) -> QuestionClassification:
    """Deterministic classification of the analyst's question.

    Every rule adds its weight to its kind once. ``general_assessment`` cues ("assess",
    "analyze", "should I buy", ...) are framing, not a topic: they win only when no specific
    kind scored. Otherwise only the specific kinds are ranked, and ``general_assessment`` stays
    last in ``candidates`` (its cues stay in ``cues``) without counting as a competitor.

    The leading kind wins with a confidence that rises with its margin over the runner-up and
    falls when a competing kind scored at all (``0.5 + 0.2 * margin - 0.15 * [competitor]``,
    capped at 0.95): one plain cue gives 0.7, a decisive phrase 0.9, and a lead of a single
    point over a competing kind 0.55, below ``LOW_CONFIDENCE``, so Laya arbitrates. Nothing
    fired, or a tie between two kinds, is ``unclear``. ``resolved_horizon`` does not change the
    kind; it is recorded by the caller.
    """
    text = _normalise(query)
    scores: dict[str, int] = {}
    cues: list[str] = []
    for rule in _RULES:
        match = rule.pattern.search(text)
        if match is None:
            continue
        scores[rule.kind] = scores.get(rule.kind, 0) + rule.weight
        cues.append(f"{rule.kind}:{match.group(0).lower()}")
    recent = any(p.search(text) for p in _RECENT_PERIOD)
    specific = {kind: score for kind, score in scores.items() if kind != _FRAMING_KIND}
    ranked = sorted(
        (specific or scores).items(), key=lambda kv: (-kv[1], QUESTION_KINDS.index(kv[0]))
    )
    candidates = [kind for kind, _ in ranked]
    if specific and _FRAMING_KIND in scores:
        candidates.append(_FRAMING_KIND)
    if not ranked:
        return QuestionClassification(
            kind=UNCLEAR, cues=cues, candidates=candidates, recent_period=recent
        )
    top = ranked[0][1]
    second = ranked[1][1] if len(ranked) > 1 else 0
    if top == second:
        return QuestionClassification(
            kind=UNCLEAR,
            cues=cues,
            candidates=candidates,
            recent_period=recent,
            note=f"rules conflict between {ranked[0][0]} and {ranked[1][0]}",
        )
    confidence = 0.5 + 0.2 * (top - second) - (0.15 if second else 0.0)
    confidence = round(min(0.95, confidence), 2)
    note = None
    if confidence < LOW_CONFIDENCE:
        note = f"rules lean to {ranked[0][0]} over {ranked[1][0]} by one point"
    return QuestionClassification(
        kind=ranked[0][0],
        confidence=confidence,
        cues=cues,
        candidates=candidates,
        recent_period=recent,
        note=note,
    )


def needs_laya(classification: QuestionClassification) -> bool:
    return classification.unclear or classification.confidence < LOW_CONFIDENCE


def classification_from_laya(
    rules: QuestionClassification, decisions: Sequence[LayaDecision]
) -> QuestionClassification:
    """Combine the rules' verdict with Laya's bounded ``question_scan`` answers.

    Laya's ``question_kind`` choice becomes the kind with Laya's confidence in that choice.
    Below ``LAYA_CONFIDENCE_FLOOR`` the pick is not trusted to narrow the analysis: the widest
    kind (``general_assessment``) runs and the result says so. ``recent_period_focus`` adds to
    the rules' own recent-period detection; it never removes it.
    """
    kind = "general_assessment"
    confidence = 0.0
    decision_id: str | None = None
    note: str | None = None
    recent = rules.recent_period
    for decision in decisions:
        if decision.decision_type == "question_kind" and isinstance(decision.answer, ChoiceAnswer):
            decision_id = decision.decision_id
            confidence = round(decision.confidence, 3)
            choice = decision.answer.choice
            if choice not in QUESTION_KINDS:
                note = "Laya chose an unknown question kind; a general assessment was produced"
            elif confidence < LAYA_CONFIDENCE_FLOOR:
                note = (
                    f"Laya was not confident about the question kind ({choice}, "
                    f"{confidence:.2f}); a general assessment was produced"
                )
            else:
                kind = choice
        elif decision.decision_type == "recent_period_focus" and isinstance(
            decision.answer, NoulAnswer
        ):
            recent = recent or decision.answer.noul >= RECENT_PERIOD_THRESHOLD
    if decision_id is None:
        raise ValueError("question_scan decisions carry no question_kind choice")
    return QuestionClassification(
        kind=kind,
        confidence=confidence,
        source="laya",
        cues=list(rules.cues),
        candidates=list(rules.candidates),
        recent_period=recent,
        decision_id=decision_id,
        note=note,
    )


# ---- requirements table ---------------------------------------------------------------------


@dataclass(frozen=True)
class _Row:
    intents: tuple[ResearchIntent, ...]
    calculations: tuple[str, ...]
    operands: tuple[str, ...]
    focus: str
    horizons: tuple[str, ...]
    recent_period: bool = False


_I = ResearchIntent

REQUIREMENTS: dict[str, _Row] = {
    # Nothing beyond the horizon seed plan and the Laya-chosen pack: today's behaviour.
    "general_assessment": _Row(
        intents=(),
        calculations=(),
        operands=(),
        focus="A general assessment of the company across the requested horizons.",
        horizons=(),
    ),
    "thesis_change": _Row(
        intents=(
            _I.retrieve_latest_filing,
            _I.retrieve_earnings_history,
            _I.retrieve_recent_news,
            _I.retrieve_guidance_history,
            _I.retrieve_price_history,
        ),
        calculations=(
            "revenue_growth_yoy",
            "revenue_growth_qoq",
            "eps_growth_yoy",
            "operating_margin_change_bp",
            "price_return_1m",
            "price_return_3m",
        ),
        operands=("revenue", "eps_diluted", "operating_income", PRICE_OPERAND),
        focus=(
            "The analyst asks whether the latest reported period changes the investment thesis. "
            "Contrast the newest quarter with the prior trend, the guidance direction, the "
            "market's reaction and, when one is given, the prior completed assessment, and say "
            "explicitly what changed and what did not."
        ),
        horizons=("next_cycle", "medium_term"),
        recent_period=True,
    ),
    "valuation": _Row(
        intents=(
            _I.retrieve_earnings_history,
            _I.retrieve_price_history,
            _I.retrieve_latest_filing,
            _I.retrieve_historical_coverage,
        ),
        calculations=(
            "market_cap",
            "pe_ttm",
            "ps_ttm",
            "fcf_yield_ttm",
            "pe_5y_percentile",
            "pe_history_percentile",
            "valuation_reconciliation_1y",
        ),
        operands=(PRICE_OPERAND, "shares_outstanding", "eps_diluted", "revenue"),
        focus=(
            "The analyst asks about valuation: whether the shares look expensive or cheap. "
            "Anchor on the computed multiples, where the trailing P/E sits within the "
            "company's own history and how much of the price move the multiple explains (the "
            "reconciliation verdict, restated as given); never state a target price."
        ),
        horizons=("medium_term", "long_term"),
    ),
    "growth": _Row(
        intents=(
            _I.retrieve_earnings_history,
            _I.retrieve_latest_filing,
            _I.retrieve_guidance_history,
            _I.retrieve_historical_coverage,
        ),
        calculations=(
            "revenue_growth_yoy",
            "revenue_growth_qoq",
            "revenue_cagr_3y",
            "eps_growth_yoy",
        ),
        operands=("revenue", "eps_diluted"),
        focus=(
            "The analyst asks about growth: the revenue and earnings trajectory, whether it is "
            "accelerating or decelerating, and how durable it looks."
        ),
        horizons=("next_cycle", "medium_term", "long_term"),
    ),
    "profitability_margins": _Row(
        intents=(_I.retrieve_earnings_history, _I.retrieve_latest_filing),
        calculations=(
            "gross_margin",
            "operating_margin",
            "net_margin",
            "fcf_margin",
            "operating_margin_change_bp",
            "free_cash_flow_ttm",
        ),
        operands=(
            "revenue",
            "gross_profit",
            "operating_income",
            "net_income",
            "operating_cash_flow",
            "capex",
        ),
        focus=(
            "The analyst asks about margins and profitability: the level and direction of "
            "gross, operating, net and free-cash-flow margins."
        ),
        horizons=("next_cycle", "medium_term"),
    ),
    "relative_performance": _Row(
        intents=(
            _I.retrieve_price_history,
            _I.retrieve_sector_benchmark,
            _I.retrieve_recent_news,
        ),
        calculations=(
            "price_return_1y",
            "relative_return_1y_vs_market",
            "relative_return_1y_vs_sector",
            "beta_1y_vs_market",
            "drawdown_vs_market_1y",
        ),
        operands=(PRICE_OPERAND, BENCHMARK_OPERAND, SECTOR_BENCHMARK_OPERAND),
        focus=(
            "The analyst asks about performance relative to the market and the sector: returns "
            "against the benchmarks, beta, and whether drawdowns were company-specific."
        ),
        horizons=("near_term", "medium_term"),
    ),
    "risk_volatility": _Row(
        intents=(
            _I.retrieve_price_history,
            _I.retrieve_sector_benchmark,
            _I.retrieve_recent_news,
            _I.retrieve_latest_filing,
        ),
        calculations=(
            "volatility_30d_annualized",
            "volatility_1y_annualized",
            "max_drawdown_1y",
            "beta_1y_vs_market",
            "drawdown_vs_market_1y",
        ),
        operands=(PRICE_OPERAND, BENCHMARK_OPERAND),
        focus=(
            "The analyst asks about risk and volatility: realised volatility, drawdowns, "
            "sensitivity to the market and the risks the filings and coverage disclose."
        ),
        horizons=("near_term", "medium_term"),
    ),
    "guidance_outlook": _Row(
        intents=(
            _I.retrieve_guidance_history,
            _I.retrieve_recent_news,
            _I.retrieve_management_commentary,
            _I.retrieve_latest_filing,
            _I.retrieve_earnings_history,
        ),
        calculations=("revenue_growth_yoy", "revenue_growth_qoq", "operating_margin_change_bp"),
        operands=("revenue", "operating_income"),
        focus=(
            "The analyst asks about guidance and outlook: the direction of management guidance, "
            "what the latest commentary signals and how it compares with the reported trajectory."
        ),
        horizons=("next_cycle", "medium_term"),
    ),
    "earnings_reaction": _Row(
        intents=(
            _I.retrieve_recent_news,
            _I.retrieve_earnings_history,
            _I.retrieve_latest_filing,
            _I.retrieve_price_history,
            _I.retrieve_sector_benchmark,
        ),
        calculations=(
            "revenue_growth_yoy",
            "revenue_growth_qoq",
            "eps_growth_yoy",
            "operating_margin_change_bp",
            "price_return_1m",
            "volatility_30d_annualized",
        ),
        operands=("revenue", "eps_diluted", "operating_income", PRICE_OPERAND),
        focus=(
            "The analyst asks about the latest results and the market's reaction to them: what "
            "the newest quarter showed against the prior trend and how the shares moved around it."
        ),
        horizons=("near_term", "next_cycle"),
        recent_period=True,
    ),
    "balance_sheet_liquidity": _Row(
        intents=(_I.retrieve_earnings_history, _I.retrieve_latest_filing),
        calculations=("enterprise_value", "free_cash_flow_ttm", "fcf_margin"),
        operands=(
            "cash_and_equivalents",
            "total_debt",
            "stockholders_equity",
            "total_assets",
            "operating_cash_flow",
            "capex",
        ),
        focus=(
            "The analyst asks about the balance sheet and liquidity: cash, debt, equity and the "
            "cash generation available to service them."
        ),
        horizons=("medium_term", "long_term"),
    ),
    "dividends_capital_return": _Row(
        intents=(
            _I.retrieve_earnings_history,
            _I.retrieve_latest_filing,
            _I.retrieve_recent_news,
        ),
        calculations=("free_cash_flow_ttm", "fcf_yield_ttm", "fcf_margin", "market_cap"),
        operands=("operating_cash_flow", "capex", "shares_outstanding", PRICE_OPERAND),
        focus=(
            "The analyst asks about dividends and capital return. Dividend and buyback amounts "
            "are not among the retrieved metrics: reason from free cash flow capacity, the "
            "share count and what the filings and coverage say about the payout, and say what "
            "is missing."
        ),
        horizons=("medium_term", "long_term"),
    ),
}


def _validate_table() -> None:
    assert set(REQUIREMENTS) == set(QUESTION_KINDS), "every question kind needs requirements"
    for kind, row in REQUIREMENTS.items():
        for intent in row.intents:
            assert intent is not ResearchIntent.stop_research, kind
        for name in row.calculations:
            assert name in SPECS, f"{kind}: unknown calculation {name}"
        for operand in row.operands:
            assert operand in OPERAND_NAMES, f"{kind}: unknown operand {operand}"
        assert len(set(row.calculations)) == len(row.calculations), kind
        assert len(set(row.operands)) == len(row.operands), kind


_validate_table()


def requirements_for(classification: QuestionClassification) -> AnalyticalRequirements:
    """The requirements row for a resolved classification (``unclear`` is a caller error)."""
    if classification.unclear:
        raise ValueError("an unclear classification must be resolved by Laya first")
    row = REQUIREMENTS[classification.kind]
    return AnalyticalRequirements(
        question_kind=classification.kind,  # type: ignore[arg-type]
        confidence=classification.confidence,
        source=classification.source,
        cues=list(classification.cues),
        recent_period=classification.recent_period or row.recent_period,
        required_research_intents=[str(i) for i in row.intents],
        required_calculations=list(row.calculations),
        required_operands=list(row.operands),
        focus=row.focus,
        horizons_emphasis=list(row.horizons),
        decision_id=classification.decision_id,
        note=classification.note,
    )


# ---- research gaps ---------------------------------------------------------------------------


def operand_gaps(
    requirements: AnalyticalRequirements | None,
    rows: Iterable[Mapping[str, Any]],
    price_series: PriceSeries | None,
    benchmark_series: Mapping[str, Any],
) -> list[str]:
    """Required operands the retrieval state does not hold yet, as evidence-gap labels.

    Fact operands are their canonical metric name (``gap_to_intent`` maps most of them to
    ``retrieve_earnings_history`` and the rest to ``retrieve_missing_metric``, whose query
    template spells the metric out); series operands reuse the loop's existing labels.
    """
    if requirements is None:
        return []
    present = {str(r.get("metric")) for r in rows if r.get("metric")}
    gaps: list[str] = []
    for operand in requirements.required_operands:
        if operand in _SERIES_OPERANDS:
            missing = (
                (price_series is None or not price_series.points)
                if operand == PRICE_OPERAND
                else not benchmark_series
            )
            label = _SERIES_OPERANDS[operand]
        else:
            missing = operand not in present
            label = operand
        if missing and label not in gaps:
            gaps.append(label)
    return gaps


# ---- validation after the calculations -----------------------------------------------------


def _has_benchmark(evidence: NormalizedEvidence, role: str) -> bool:
    for ref in evidence.benchmark_refs:
        if ref.role != role:
            continue
        series = evidence.benchmarks.get(ref.symbol) or evidence.benchmarks.get(ref.role)
        if series is not None and series.points:
            return True
    series = evidence.benchmarks.get(role)
    return bool(series is not None and series.points)


def _operand_present(evidence: NormalizedEvidence | None, operand: str) -> bool:
    if evidence is None:
        return False
    if operand == PRICE_OPERAND:
        return evidence.prices is not None and bool(evidence.prices.points)
    if operand == BENCHMARK_OPERAND:
        return _has_benchmark(evidence, "broad_market")
    if operand == SECTOR_BENCHMARK_OPERAND:
        return _has_benchmark(evidence, "sector")
    return bool(evidence.facts_for(operand))


def _operand_reason(operand: str) -> str:
    if operand == PRICE_OPERAND:
        return "no price history was retrieved"
    if operand == BENCHMARK_OPERAND:
        return "no broad-market benchmark series was retrieved"
    if operand == SECTOR_BENCHMARK_OPERAND:
        return "no sector benchmark series was retrieved"
    return f"no {operand.replace('_', ' ')} facts were retrieved"


def check_requirements(
    requirements: AnalyticalRequirements,
    evidence: NormalizedEvidence | None,
    calculations: CalculatedMetrics,
    executed_intents: Sequence[str] = (),
) -> RequirementsReport:
    """Which requirements the analysis met; every unmet one is one uncertainty sentence.

    A required calculation is satisfied when a computed result with that name exists; an
    unavailable result names its missing operands (or the reason it has no meaningful value).
    A required operand is satisfied when the normalized evidence holds it. Nothing here fails
    the analysis: the report is additive and the sentences reach the result and Spark.
    """
    label = QUESTION_KIND_LABELS[requirements.question_kind]
    by_name = {c.name: c for c in calculations.calculations}
    satisfied_calcs: list[str] = []
    missing_calcs: list[MissingRequirement] = []
    for name in requirements.required_calculations:
        calc = by_name.get(name)
        if calc is not None and calc.status == "computed":
            satisfied_calcs.append(name)
            continue
        if calc is None:
            missing_calcs.append(MissingRequirement(name=name, reason="not computed"))
        elif calc.missing_inputs:
            missing_calcs.append(
                MissingRequirement(
                    name=name,
                    reason="missing " + ", ".join(calc.missing_inputs),
                    missing_inputs=list(calc.missing_inputs),
                )
            )
        else:
            missing_calcs.append(
                MissingRequirement(name=name, reason=calc.notes[-1] if calc.notes else "no value")
            )
    satisfied_operands: list[str] = []
    missing_operands: list[MissingRequirement] = []
    for operand in requirements.required_operands:
        if _operand_present(evidence, operand):
            satisfied_operands.append(operand)
        else:
            missing_operands.append(
                MissingRequirement(name=operand, reason=_operand_reason(operand))
            )
    executed = [str(i) for i in executed_intents]
    missing_intents = [i for i in requirements.required_research_intents if i not in executed]

    uncertainties: list[str] = []
    for item in missing_operands:
        uncertainties.append(f"the question asks about {label} but {item.reason}")
    for item in missing_calcs:
        uncertainties.append(
            f"the question asks about {label} but {_calc_label(item.name)} could not be "
            f"computed: {item.reason}"
        )
    if missing_intents:
        uncertainties.append(
            f"the question asks about {label} but the research budget ended before "
            + ", ".join(missing_intents)
            + (" was" if len(missing_intents) == 1 else " were")
            + " executed"
        )
    if requirements.note:
        uncertainties.append(requirements.note)
    return RequirementsReport(
        classification=requirements.classification(),
        focus=requirements.focus,
        horizons_emphasis=list(requirements.horizons_emphasis),
        required_research_intents=list(requirements.required_research_intents),
        executed_research_intents=[
            i for i in requirements.required_research_intents if i in executed
        ],
        missing_research_intents=missing_intents,
        required_calculations=list(requirements.required_calculations),
        satisfied_calculations=satisfied_calcs,
        missing_calculations=missing_calcs,
        required_operands=list(requirements.required_operands),
        satisfied_operands=satisfied_operands,
        missing_operands=missing_operands,
        uncertainties=uncertainties,
    )


_CALC_LABELS: dict[str, str] = {
    "pe_ttm": "the trailing P/E",
    "ps_ttm": "the price-to-sales multiple",
    "ev_ebitda_ttm": "EV/EBITDA",
    "fcf_yield_ttm": "the free-cash-flow yield",
    "pe_5y_percentile": "the P/E's five-year percentile",
    "pe_history_percentile": "the P/E's percentile over its full available history",
    "valuation_reconciliation_1y": "the one-year price-versus-earnings reconciliation",
    "valuation_reconciliation_3y": "the three-year price-versus-earnings reconciliation",
    "market_cap": "the market cap",
    "enterprise_value": "the enterprise value",
    "revenue_growth_yoy": "year-over-year revenue growth",
    "revenue_growth_qoq": "sequential revenue growth",
    "revenue_cagr_3y": "the three-year revenue CAGR",
    "eps_growth_yoy": "year-over-year EPS growth",
    "gross_margin": "the gross margin",
    "operating_margin": "the operating margin",
    "net_margin": "the net margin",
    "fcf_margin": "the free-cash-flow margin",
    "operating_margin_change_bp": "the operating-margin change",
    "free_cash_flow_ttm": "trailing free cash flow",
    "price_return_1m": "the one-month price return",
    "price_return_3m": "the three-month price return",
    "price_return_1y": "the one-year price return",
    "relative_return_1y_vs_market": "the one-year return relative to the market",
    "relative_return_1y_vs_sector": "the one-year return relative to the sector",
    "beta_1y_vs_market": "the one-year beta",
    "drawdown_vs_market_1y": "the drawdown relative to the market",
    "volatility_30d_annualized": "the 30-day volatility",
    "volatility_1y_annualized": "the one-year volatility",
    "max_drawdown_1y": "the one-year maximum drawdown",
}


def _calc_label(name: str) -> str:
    return _CALC_LABELS.get(name, name)
