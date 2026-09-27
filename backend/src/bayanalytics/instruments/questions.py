"""Deterministic requirements builder: interpreted question -> research, calculations, checks.

No rule here decides what a question is about. Spark pass 1
(:mod:`bayanalytics.pipeline.understanding`) interprets the question into a bounded
``QueryUnderstanding``; Laya confirms or drops each proposed requirement
(:mod:`bayanalytics.pipeline.questions`); this module only turns what survived into work:

    kept requirements -> REQUIREMENT_TABLE rows (union, deduplicated, stable order)
                      -> research intents (facts first), calculations, operands, price window
    intent            -> INTENT_TABLE row -> focus sentence and horizons to emphasise
    after the math    -> check_requirements -> satisfied / unmet -> uncertainties (never failures)

Both tables are validated at import against ``ResearchIntent``, the calculation registry
(``SPECS``) and the operand names. A general assessment with no requirements builds nothing,
so "Assess X." runs exactly the horizon seed plan and the Laya-chosen calculation pack.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bayanalytics.calculations.registry import CANONICAL_METRICS, PE_HISTORY_YEARS, SPECS
from bayanalytics.instruments.base import CalculatedMetrics
from bayanalytics.laya.schemas import REQUIREMENTS_SUPPORTED_KEY, requirement_key
from bayanalytics.research.intents import ResearchIntent, facts_first
from bayanalytics.schemas.decisions import LayaDecision, NoulAnswer
from bayanalytics.schemas.evidence import NormalizedEvidence, PriceSeries
from bayanalytics.schemas.questions import (
    COMPARISON_FOCUS_LABELS,
    GENERAL_ASSESSMENT,
    QUESTION_INTENTS,
    REQUIREMENT_LABELS,
    REQUIREMENT_NAMES,
    AnalyticalRequirements,
    InterpretationSource,
    MissingRequirement,
    QueryUnderstanding,
    RequirementsReport,
)

REQUIREMENT_REJECT_BELOW = 0.3
"""A proposed requirement is dropped when Laya's noul for it is below this."""
SUPPORT_REJECT_BELOW = 0.3
"""The whole interpretation is rejected (a general assessment runs) when Laya's
``requirements_supported`` noul is below this."""

PRIOR_ASSESSMENT = "prior_assessment"
LATEST_PERIOD = "latest_period"
IMPLIED_BY_FLAGS: tuple[tuple[str, str], ...] = (
    ("needs_benchmark", "benchmark_comparison"),
    ("needs_prior_assessment", PRIOR_ASSESSMENT),
    ("recent_period_focus", LATEST_PERIOD),
)
"""Interpretation flags and the requirement each one implies (part of Spark's proposal)."""

REJECTED_NOTE = (
    "the interpretation of the question was not confirmed; a general assessment was produced"
)

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

LONG_PRICE_WINDOW_DAYS = PE_HISTORY_YEARS * 366
"""Price window for valuation history and the price-versus-earnings reconciliation: the P/E
percentiles need at least eight quarter-end P/E observations and the three-year reconciliation
needs the price three years before the latest trailing period, whatever the horizon."""


# ---- tables ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class RequirementSpec:
    intents: tuple[ResearchIntent, ...] = ()
    calculations: tuple[str, ...] = ()
    operands: tuple[str, ...] = ()
    also_calculated: tuple[str, ...] = ()
    min_price_days: int | None = None


@dataclass(frozen=True)
class IntentSpec:
    focus: str
    horizons: tuple[str, ...] = field(default_factory=tuple)


_I = ResearchIntent

REQUIREMENT_TABLE: dict[str, RequirementSpec] = {
    "valuation_multiples": RequirementSpec(
        intents=(
            _I.retrieve_earnings_history,
            _I.retrieve_price_history,
            _I.retrieve_latest_filing,
        ),
        calculations=("market_cap", "pe_ttm", "ps_ttm", "fcf_yield_ttm"),
        operands=(PRICE_OPERAND, "shares_outstanding", "eps_diluted", "revenue"),
    ),
    "valuation_history": RequirementSpec(
        intents=(
            _I.retrieve_earnings_history,
            _I.retrieve_price_history,
            _I.retrieve_historical_coverage,
        ),
        calculations=("pe_ttm", "pe_5y_percentile", "pe_history_percentile"),
        operands=(PRICE_OPERAND, "eps_diluted"),
        min_price_days=LONG_PRICE_WINDOW_DAYS,
    ),
    "price_vs_earnings": RequirementSpec(
        intents=(_I.retrieve_earnings_history, _I.retrieve_price_history),
        calculations=("valuation_reconciliation_1y",),
        operands=(PRICE_OPERAND, "eps_diluted", "revenue"),
        also_calculated=("valuation_reconciliation_3y",),
        min_price_days=LONG_PRICE_WINDOW_DAYS,
    ),
    "earnings_trajectory": RequirementSpec(
        intents=(
            _I.retrieve_earnings_history,
            _I.retrieve_latest_filing,
            _I.retrieve_guidance_history,
        ),
        calculations=("eps_growth_yoy",),
        operands=("eps_diluted", "net_income"),
    ),
    "revenue_trajectory": RequirementSpec(
        intents=(
            _I.retrieve_earnings_history,
            _I.retrieve_latest_filing,
            _I.retrieve_guidance_history,
        ),
        calculations=("revenue_growth_yoy", "revenue_growth_qoq", "revenue_cagr_3y"),
        operands=("revenue",),
    ),
    "margin_trajectory": RequirementSpec(
        intents=(_I.retrieve_earnings_history, _I.retrieve_latest_filing),
        calculations=(
            "gross_margin",
            "operating_margin",
            "net_margin",
            "operating_margin_change_bp",
        ),
        operands=("revenue", "gross_profit", "operating_income", "net_income"),
    ),
    "cash_flow": RequirementSpec(
        intents=(_I.retrieve_earnings_history, _I.retrieve_latest_filing),
        calculations=("free_cash_flow_ttm", "fcf_margin"),
        operands=("operating_cash_flow", "capex", "revenue"),
    ),
    "price_performance": RequirementSpec(
        intents=(_I.retrieve_price_history, _I.retrieve_recent_news),
        calculations=("price_return_1m", "price_return_3m", "price_return_1y"),
        operands=(PRICE_OPERAND,),
    ),
    "benchmark_comparison": RequirementSpec(
        intents=(_I.retrieve_price_history, _I.retrieve_sector_benchmark),
        calculations=(
            "relative_return_1y_vs_market",
            "relative_return_1y_vs_sector",
            "beta_1y_vs_market",
            "drawdown_vs_market_1y",
        ),
        operands=(PRICE_OPERAND, BENCHMARK_OPERAND, SECTOR_BENCHMARK_OPERAND),
    ),
    "volatility_drawdown": RequirementSpec(
        intents=(_I.retrieve_price_history,),
        calculations=("volatility_30d_annualized", "volatility_1y_annualized", "max_drawdown_1y"),
        operands=(PRICE_OPERAND,),
    ),
    "balance_sheet": RequirementSpec(
        intents=(_I.retrieve_earnings_history, _I.retrieve_latest_filing),
        calculations=("enterprise_value",),
        operands=("cash_and_equivalents", "total_debt", "stockholders_equity", "total_assets"),
    ),
    "capital_return": RequirementSpec(
        intents=(
            _I.retrieve_earnings_history,
            _I.retrieve_latest_filing,
            _I.retrieve_recent_news,
        ),
        calculations=("free_cash_flow_ttm", "fcf_yield_ttm"),
        operands=("operating_cash_flow", "capex", "shares_outstanding", PRICE_OPERAND),
    ),
    "guidance": RequirementSpec(
        intents=(
            _I.retrieve_guidance_history,
            _I.retrieve_management_commentary,
            _I.retrieve_recent_news,
        ),
    ),
    "latest_period": RequirementSpec(
        intents=(
            _I.retrieve_earnings_history,
            _I.retrieve_latest_filing,
            _I.retrieve_recent_news,
            _I.retrieve_management_commentary,
        ),
        calculations=(
            "revenue_growth_yoy",
            "revenue_growth_qoq",
            "eps_growth_yoy",
            "operating_margin_change_bp",
            "price_return_1m",
        ),
        operands=("revenue", "eps_diluted", "operating_income", PRICE_OPERAND),
    ),
    # The prior-assessment thesis diff (pipeline/thesis.py) is question-agnostic and always
    # computed; requiring it only checks that a prior completed assessment exists.
    PRIOR_ASSESSMENT: RequirementSpec(),
    "recent_coverage": RequirementSpec(intents=(_I.retrieve_recent_news,)),
}

INTENT_TABLE: dict[str, IntentSpec] = {
    # Nothing is emphasised: a broad request runs as it always has.
    GENERAL_ASSESSMENT: IntentSpec(
        focus="A general assessment of the company across the requested horizons."
    ),
    "valuation": IntentSpec(
        focus=(
            "The analyst asks about valuation: whether the shares look expensive or cheap. "
            "Anchor on the computed multiples and, where given, where the P/E sits within the "
            "company's own history; never state a target price."
        ),
        horizons=("medium_term", "long_term"),
    ),
    "valuation_vs_fundamentals": IntentSpec(
        focus=(
            "The analyst asks whether the valuation is justified by the fundamentals: set the "
            "multiples against the earnings and revenue trajectory and restate the "
            "price-versus-earnings reconciliation verdict exactly as given."
        ),
        horizons=("medium_term", "long_term"),
    ),
    "growth": IntentSpec(
        focus=(
            "The analyst asks about growth: the revenue and earnings trajectory, whether it is "
            "accelerating or decelerating, and how durable it looks."
        ),
        horizons=("next_cycle", "medium_term", "long_term"),
    ),
    "profitability": IntentSpec(
        focus=(
            "The analyst asks about profitability: the level and direction of gross, "
            "operating, net and free-cash-flow margins."
        ),
        horizons=("next_cycle", "medium_term"),
    ),
    "event_impact": IntentSpec(
        focus=(
            "The analyst asks what the latest results or a specific event changed. Contrast the "
            "newest period and the newest dated evidence with the prior trend and, when one is "
            "given, the prior completed assessment, and say explicitly what changed and what "
            "did not."
        ),
        horizons=("near_term", "next_cycle"),
    ),
    "relative_performance": IntentSpec(
        focus=(
            "The analyst asks how the stock performed relative to a benchmark: returns against "
            "the market and the sector, beta, and whether drawdowns were company-specific."
        ),
        horizons=("near_term", "medium_term"),
    ),
    "risk": IntentSpec(
        focus=(
            "The analyst asks about risk: realised volatility, drawdowns, sensitivity to the "
            "market and the risks the filings and coverage disclose."
        ),
        horizons=("near_term", "medium_term"),
    ),
    "balance_sheet": IntentSpec(
        focus=(
            "The analyst asks about the balance sheet: cash, debt, equity and the cash "
            "generation available to service them."
        ),
        horizons=("medium_term", "long_term"),
    ),
    "capital_return": IntentSpec(
        focus=(
            "The analyst asks about dividends and capital return. Dividend and buyback amounts "
            "are not among the retrieved metrics: reason from free cash flow, the share count "
            "and what the filings and coverage say about the payout, and say what is missing."
        ),
        horizons=("medium_term", "long_term"),
    ),
    "guidance_outlook": IntentSpec(
        focus=(
            "The analyst asks about guidance and outlook: the direction of management "
            "guidance, what the latest commentary signals and how it compares with the "
            "reported trajectory."
        ),
        horizons=("next_cycle", "medium_term"),
    ),
}


def _validate_tables() -> None:
    assert set(REQUIREMENT_TABLE) == set(REQUIREMENT_NAMES), "every requirement needs a row"
    assert set(INTENT_TABLE) == set(QUESTION_INTENTS), "every intent needs a row"
    for name, row in REQUIREMENT_TABLE.items():
        for intent in row.intents:
            assert isinstance(intent, ResearchIntent), name
            assert intent is not ResearchIntent.stop_research, name
        for calc in (*row.calculations, *row.also_calculated):
            assert calc in SPECS, f"{name}: unknown calculation {calc}"
        for operand in row.operands:
            assert operand in OPERAND_NAMES, f"{name}: unknown operand {operand}"
        for values in (row.intents, row.calculations, row.operands, row.also_calculated):
            assert len(set(values)) == len(values), name
        assert not set(row.calculations) & set(row.also_calculated), name
    for intent, spec in INTENT_TABLE.items():
        assert spec.focus, intent


_validate_tables()


# ---- proposal and Laya's validation -------------------------------------------------------


def proposed_requirements(understanding: QueryUnderstanding) -> list[str]:
    """Spark's proposal: its requirements, then those its flags imply (deduplicated)."""
    proposal = list(dict.fromkeys(understanding.requirements))
    for flag, requirement in IMPLIED_BY_FLAGS:
        if getattr(understanding, flag) and requirement not in proposal:
            proposal.append(requirement)
    return proposal


@dataclass(frozen=True)
class ValidationOutcome:
    kept: list[str]
    dropped: list[str]
    rejected: bool
    decision_ids: list[str]


def combine_validation(
    proposal: Sequence[str], decisions: Sequence[LayaDecision]
) -> ValidationOutcome:
    """Laya's ``question_validation`` answers applied to Spark's proposal.

    A requirement is kept unless its noul is below ``REQUIREMENT_REJECT_BELOW``; the whole
    interpretation is rejected (every proposed requirement dropped, a general assessment runs)
    when ``requirements_supported`` is below ``SUPPORT_REJECT_BELOW``. Only proposed
    requirements are read, so Laya can never add one.
    """
    nouls: dict[str, float] = {}
    decision_ids: list[str] = []
    for decision in decisions:
        if isinstance(decision.answer, NoulAnswer):
            nouls[decision.decision_type] = decision.answer.noul
            decision_ids.append(decision.decision_id)
    if REQUIREMENTS_SUPPORTED_KEY not in nouls:
        raise ValueError("question_validation decisions carry no requirements_supported answer")
    kept: list[str] = []
    dropped: list[str] = []
    for requirement in proposal:
        noul = nouls.get(requirement_key(requirement))
        if noul is None:
            raise ValueError(f"question_validation carries no answer for {requirement}")
        (dropped if noul < REQUIREMENT_REJECT_BELOW else kept).append(requirement)
    if nouls[REQUIREMENTS_SUPPORTED_KEY] < SUPPORT_REJECT_BELOW:
        return ValidationOutcome([], list(proposal), True, decision_ids)
    return ValidationOutcome(kept, dropped, False, decision_ids)


# ---- builder --------------------------------------------------------------------------------


def _union(values: Iterable[Iterable[Any]]) -> list[Any]:
    out: list[Any] = []
    for group in values:
        for value in group:
            if value not in out:
                out.append(value)
    return out


def build_requirements(
    understanding: QueryUnderstanding,
    *,
    source: InterpretationSource,
    kept: Sequence[str] | None = None,
    dropped: Sequence[str] = (),
    notes: Sequence[str] = (),
    decision_ids: Sequence[str] = (),
) -> AnalyticalRequirements:
    """Compose the requirements of an interpretation from the tables.

    ``kept`` is what survived Laya's validation (by default Spark's whole proposal, for the
    paths where Laya was not asked). Rows are unioned in requirement order; the research
    intents are then ordered facts first (``research.intents.facts_first``).
    """
    requirements = list(kept) if kept is not None else proposed_requirements(understanding)
    rows = [REQUIREMENT_TABLE[name] for name in requirements]
    calculations = _union(row.calculations for row in rows)
    also = [c for c in _union(row.also_calculated for row in rows) if c not in calculations]
    windows = [row.min_price_days for row in rows if row.min_price_days]
    intent = INTENT_TABLE[understanding.intent]
    focus = intent.focus
    comparison = COMPARISON_FOCUS_LABELS.get(understanding.comparison_focus, "")
    if comparison and understanding.intent != GENERAL_ASSESSMENT:
        focus = f"{focus} The comparison asked for is against {comparison}."
    return AnalyticalRequirements(
        intent=understanding.intent,
        requirements=requirements,  # type: ignore[arg-type]
        comparison_focus=understanding.comparison_focus,
        source=source,
        dropped_by_validation=list(dropped),  # type: ignore[arg-type]
        recent_period=LATEST_PERIOD in requirements,
        required_research_intents=[
            str(i) for i in facts_first(_union(row.intents for row in rows))
        ],
        required_calculations=calculations,
        also_calculated=also,
        required_operands=_union(row.operands for row in rows),
        min_price_days=max(windows) if windows else None,
        focus=focus,
        horizons_emphasis=list(intent.horizons),
        notes=list(dict.fromkeys(notes)),
        decision_ids=list(decision_ids),
    )


def rejected_requirements(
    proposal: Sequence[str], notes: Sequence[str], decision_ids: Sequence[str]
) -> AnalyticalRequirements:
    """Laya rejected the interpretation: a general assessment, with every proposed requirement
    recorded as dropped and the rejection as an uncertainty."""
    return build_requirements(
        QueryUnderstanding.broad(),
        source="fallback",
        kept=[],
        dropped=proposal,
        notes=[*notes, REJECTED_NOTE],
        decision_ids=decision_ids,
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


# ---- acceptance checks after the calculations ----------------------------------------------


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


def _retrieved_intents(evidence: NormalizedEvidence | None) -> set[str]:
    """The research intents that produced at least one kept (not rejected) source."""
    if evidence is None:
        return set()
    return {
        s.research_intent
        for s in evidence.sources
        if s.research_intent and s.rejected_reason is None
    }


def _owner(requirements: Sequence[str], attribute: str, item: str) -> str:
    """The first kept requirement whose row lists ``item`` (for the sentence's subject)."""
    for name in requirements:
        if item in getattr(REQUIREMENT_TABLE[name], attribute):
            return name
    return requirements[0] if requirements else ""


def _needs(requirement: str) -> str:
    return REQUIREMENT_LABELS[requirement].lower() if requirement else "this"


def prior_assessment_reason(prior_available: bool | None, symbol: str | None) -> str:
    if prior_available is None:
        return "the analysis stopped before the prior-assessment lookup"
    return (
        f"no earlier completed assessment of {symbol or 'this company'} exists, so what "
        "changed is judged against the retrieved history only"
    )


def check_requirements(
    requirements: AnalyticalRequirements,
    evidence: NormalizedEvidence | None,
    calculations: CalculatedMetrics,
    executed_intents: Sequence[str] = (),
    *,
    prior_available: bool | None = None,
    symbol: str | None = None,
) -> RequirementsReport:
    """Which requirements the analysis met; every unmet one is one uncertainty sentence.

    A required calculation is satisfied when a computed result with that name exists; an
    unavailable result names its missing operands (or the reason it has no meaningful value).
    A required operand is satisfied when the normalized evidence holds it; a required intent
    when the research loop executed it; ``prior_assessment`` when a prior completed assessment
    exists (``prior_available``: ``None`` means the lookup was not reached). A requirement with
    no calculations or operands of its own (guidance, recent coverage) is met only by evidence:
    at least one kept source retrieved by one of its intents, since an executed search can
    return nothing usable. A requirement is satisfied when all of its own parts are. Nothing
    here fails the analysis.
    """
    kept = list(requirements.requirements)
    by_name = {c.name: c for c in calculations.calculations}
    unmet_parts: dict[str, list[str]] = {name: [] for name in kept}
    uncertainties: list[str] = []

    satisfied_operands: list[str] = []
    missing_operands: list[MissingRequirement] = []
    for operand in requirements.required_operands:
        if _operand_present(evidence, operand):
            satisfied_operands.append(operand)
            continue
        owner = _owner(kept, "operands", operand)
        reason = _operand_reason(operand)
        missing_operands.append(
            MissingRequirement(
                name=operand, reason=reason, requirement=REQUIREMENT_LABELS.get(owner)
            )
        )
        uncertainties.append(f"the question needs {_needs(owner)} but {reason}")
        for name in kept:
            if operand in REQUIREMENT_TABLE[name].operands:
                unmet_parts[name].append(reason)

    satisfied_calcs: list[str] = []
    missing_calcs: list[MissingRequirement] = []
    for calc_name in requirements.required_calculations:
        calc = by_name.get(calc_name)
        if calc is not None and calc.status == "computed":
            satisfied_calcs.append(calc_name)
            continue
        if calc is None:
            reason, inputs = "not computed", []
        elif calc.missing_inputs:
            reason, inputs = "missing " + ", ".join(calc.missing_inputs), list(calc.missing_inputs)
        else:
            reason, inputs = (calc.notes[-1] if calc.notes else "no value"), []
        owner = _owner(kept, "calculations", calc_name)
        missing_calcs.append(
            MissingRequirement(
                name=calc_name,
                reason=reason,
                missing_inputs=inputs,
                requirement=REQUIREMENT_LABELS.get(owner),
            )
        )
        label = calc_label(calc_name)
        uncertainties.append(
            f"the question needs {_needs(owner)} but {label} could not be computed: {reason}"
        )
        for name in kept:
            if calc_name in REQUIREMENT_TABLE[name].calculations:
                unmet_parts[name].append(f"{label} could not be computed ({reason})")

    executed = [str(i) for i in executed_intents]
    missing_intents = [i for i in requirements.required_research_intents if i not in executed]
    if missing_intents:
        uncertainties.append(
            "the question needs "
            + ", ".join(missing_intents)
            + " but the research budget ended before "
            + ("it was" if len(missing_intents) == 1 else "they were")
            + " executed"
        )
        for name in kept:
            for intent in REQUIREMENT_TABLE[name].intents:
                if str(intent) in missing_intents:
                    unmet_parts[name].append(f"{intent} was not executed")

    retrieved = _retrieved_intents(evidence)
    for name in kept:
        row = REQUIREMENT_TABLE[name]
        if row.calculations or row.operands or not row.intents:
            continue  # met through its calculations and operands, or (prior) its own lookup
        if any(str(intent) in retrieved for intent in row.intents):
            continue
        reason = "no usable source was retrieved for it"
        unmet_parts[name].append(reason)
        uncertainties.append(f"the question needs {_needs(name)} but {reason}")

    if PRIOR_ASSESSMENT in kept and not prior_available:
        subject = symbol or (evidence.symbol if evidence is not None else None)
        reason = prior_assessment_reason(prior_available, subject)
        unmet_parts[PRIOR_ASSESSMENT].append(reason)
        uncertainties.append(f"the question asks about the prior assessment but {reason}")

    uncertainties.extend(requirements.notes)
    unmet = [
        MissingRequirement(
            name=REQUIREMENT_LABELS[name],
            reason="; ".join(dict.fromkeys(parts)),
            requirement=REQUIREMENT_LABELS[name],
        )
        for name, parts in unmet_parts.items()
        if parts
    ]
    return RequirementsReport(
        question_intent=requirements.intent_label,
        requirements=requirements.requirement_labels,
        interpretation_source=requirements.source,
        dropped_by_validation=[REQUIREMENT_LABELS[r] for r in requirements.dropped_by_validation],
        focus="" if requirements.broad else requirements.focus,
        horizons_emphasis=list(requirements.horizons_emphasis),
        satisfied_requirements=[
            REQUIREMENT_LABELS[name] for name, parts in unmet_parts.items() if not parts
        ],
        unmet_requirements=unmet,
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
        uncertainties=list(dict.fromkeys(uncertainties)),
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


def calc_label(name: str) -> str:
    return _CALC_LABELS.get(name, name)


def _validate_labels() -> None:
    for row in REQUIREMENT_TABLE.values():
        for calc in row.calculations:
            assert calc in _CALC_LABELS, f"no readable label for {calc}"


_validate_labels()
