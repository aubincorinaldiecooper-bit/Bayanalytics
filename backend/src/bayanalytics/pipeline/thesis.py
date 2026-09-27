"""Prior assessment lookup and the thesis diff (AGENT.md section 12: "what changed").

"Did the latest quarter change the thesis?" is answered against the assessment this backend
produced last time for the same instrument, not inferred from the narrative. The comparison
is deterministic and structural: stances (overall and per horizon), the deterministic
calculation values both runs computed, conflicts and uncertainties that appeared or went
away, and whether the information set moved on (a newer quarter, newer prices). The prior
run's narrative is never reused; Spark receives a bounded, evidence-only block built from
the diff (:func:`prior_assessment_block`) and the result carries the full diff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from bayanalytics.calculations.formatting import format_change
from bayanalytics.instruments.base import CalculatedMetrics, LayaDecisions
from bayanalytics.pipeline.assemble import LOW_CONFIDENCE_FLOOR
from bayanalytics.schemas.calculations import CalculationResult
from bayanalytics.schemas.common import HORIZON_LABELS, SINGLE_HORIZONS, Stance
from bayanalytics.schemas.decisions import ChoiceAnswer
from bayanalytics.schemas.evidence import Conflict, NormalizedEvidence
from bayanalytics.schemas.results import (
    AnalysisResult,
    FreshnessChange,
    HorizonAssessment,
    MetricChange,
    StanceChange,
    ThesisDiff,
)
from bayanalytics.store.base import AnalysisStore

THESIS_METRICS: tuple[str, ...] = (
    "revenue_growth_yoy",
    "eps_growth_yoy",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "fcf_margin",
    "pe_ttm",
    "pe_history_percentile",
    "valuation_reconciliation_1y",
    "price_return_3m",
    "price_return_1y",
)
"""Calculations compared value-then vs value-now when both assessments computed them."""

STANCES: frozenset[str] = frozenset({"bullish", "neutral", "bearish", "mixed"})
PRIOR_BLOCK_MAX_METRICS = 10
PRIOR_BLOCK_MAX_ITEMS = 4  # conflicts / uncertainties per direction in the Spark block
PRIOR_BLOCK_MAX_SUMMARY = 8


def no_prior_assessment_note(symbol: str) -> str:
    return (
        f"no prior completed assessment of {symbol} was found in this store; what changed is "
        "judged against the retrieved history only, not against an earlier assessment"
    )


async def find_prior_assessment(
    store: AnalysisStore, symbol: str, *, before: datetime, owner_id: str | None
) -> AnalysisResult | None:
    """The latest completed assessment of ``symbol`` created strictly before ``before``.

    ``owner_id`` is required so every caller states the scope it asks for; it must be
    ``None`` until analyses record an owner (the store refuses anything else). With
    ``None`` the lookup spans the whole store: correct for a single-user deployment only.
    """
    return await store.latest_completed_result(symbol, before=before, owner_id=owner_id)


# ------------------------------------------------------------------ snapshots --------------


@dataclass
class ThesisSnapshot:
    """The structured fields of one assessment that the diff compares."""

    stances: dict[str, Stance] = field(default_factory=dict)  # by horizon
    calculations: list[CalculationResult] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    freshness_summary: dict[str, Any] = field(default_factory=dict)


def overall_stance(stances: dict[str, Stance]) -> Stance | None:
    """One stance for the whole assessment from the horizon stances: a single direction when
    every horizon agrees or is neutral, ``mixed`` when horizons disagree or any is mixed,
    ``neutral`` when all are neutral, ``None`` without stances."""
    values = [s for s in stances.values() if s in STANCES]
    if not values:
        return None
    directional = {s for s in values if s in ("bullish", "bearish")}
    if "mixed" in values or len(directional) == 2:
        return "mixed"
    if len(directional) == 1:
        return next(iter(directional))  # type: ignore[return-value]
    return "neutral"


def snapshot_of_result(result: AnalysisResult) -> ThesisSnapshot:
    return ThesisSnapshot(
        stances={h: a.stance for h, a in result.horizon_assessments.items()},
        calculations=list(result.calculations),
        conflicts=list(result.assessment.conflicts),
        uncertainties=list(result.assessment.uncertainties),
        freshness_summary=dict(result.freshness_summary),
    )


def stances_from_decisions(decisions: LayaDecisions, horizons: list[str]) -> dict[str, Stance]:
    """Laya's horizon stances under the same rule ``merge_horizons`` applies: below the
    confidence floor the reported stance is ``mixed``."""
    out: dict[str, Stance] = {}
    for horizon in horizons:
        decision = decisions.latest(f"horizon_stance_{horizon}")
        if decision is None or not isinstance(decision.answer, ChoiceAnswer):
            continue
        choice = decision.answer.choice
        if decision.confidence < LOW_CONFIDENCE_FLOOR or choice not in STANCES:
            out[horizon] = "mixed"
        else:
            out[horizon] = choice  # type: ignore[assignment]
    return out


def snapshot_of_draft(
    evidence: NormalizedEvidence,
    decisions: LayaDecisions,
    calculations: CalculatedMetrics,
    horizons: list[str],
    extra_uncertainties: list[str] | None = None,
) -> ThesisSnapshot:
    """The current run before synthesis: Laya's stances, the calculations, the evidence's
    conflicts and uncertainties plus what the pipeline has added so far."""
    return ThesisSnapshot(
        stances=stances_from_decisions(decisions, horizons),
        calculations=list(calculations.calculations),
        conflicts=list(evidence.conflicts),
        uncertainties=[*evidence.uncertainties, *(extra_uncertainties or [])],
        freshness_summary=dict(evidence.freshness_summary),
    )


def snapshot_of_assembled(
    horizon_assessments: dict[str, HorizonAssessment],
    calculations: CalculatedMetrics,
    conflicts: list[Conflict],
    uncertainties: list[str],
    freshness_summary: dict[str, Any],
) -> ThesisSnapshot:
    """The current run once assembled (final stances, final conflicts and uncertainties)."""
    return ThesisSnapshot(
        stances={h: a.stance for h, a in horizon_assessments.items()},
        calculations=list(calculations.calculations),
        conflicts=list(conflicts),
        uncertainties=list(uncertainties),
        freshness_summary=dict(freshness_summary),
    )


# ------------------------------------------------------------------ diff ------------------


def conflict_key(conflict: Conflict) -> str:
    return (
        f"{conflict.metric} {conflict.period_label}" if conflict.period_label else conflict.metric
    )


def _computed(calculations: list[CalculationResult]) -> dict[str, CalculationResult]:
    return {c.name: c for c in calculations if c.status == "computed" and c.value is not None}


def _facts_block(summary: dict[str, Any]) -> dict[str, Any]:
    facts = summary.get("facts")
    return facts if isinstance(facts, dict) else {}


def _prices_block(summary: dict[str, Any]) -> dict[str, Any]:
    prices = summary.get("prices")
    return prices if isinstance(prices, dict) else {}


def diff_assessments(
    previous: AnalysisResult, current: AnalysisResult | ThesisSnapshot
) -> ThesisDiff:
    """Compare ``current`` with the earlier completed ``previous`` assessment.

    Pure and deterministic: the same two inputs always give the same diff. ``current`` is a
    finished result or a :class:`ThesisSnapshot` of a run in progress.
    """
    before = snapshot_of_result(previous)
    now = current if isinstance(current, ThesisSnapshot) else snapshot_of_result(current)

    ordered = [h for h in SINGLE_HORIZONS if h in before.stances or h in now.stances]
    ordered += sorted((set(before.stances) | set(now.stances)) - set(ordered))
    # The overall stance is compared over the horizons both runs assessed: a near-term run
    # after a multi-horizon one covers less, which is a change of scope, not of thesis.
    shared = [h for h in ordered if h in before.stances and h in now.stances]
    overall = StanceChange(
        scope="overall",
        previous=overall_stance({h: before.stances[h] for h in shared}),
        current=overall_stance({h: now.stances[h] for h in shared}),
        compared_horizons=shared,
    )
    overall.changed = (
        overall.previous is not None
        and overall.current is not None
        and overall.previous != overall.current
    )
    horizons: list[StanceChange] = []
    for horizon in ordered:
        item = StanceChange(
            scope=horizon, previous=before.stances.get(horizon), current=now.stances.get(horizon)
        )
        item.changed = (
            item.previous is not None and item.current is not None and item.previous != item.current
        )
        horizons.append(item)

    metrics: list[MetricChange] = []
    prev_calcs, cur_calcs = _computed(before.calculations), _computed(now.calculations)
    for name in THESIS_METRICS:
        prev, cur = prev_calcs.get(name), cur_calcs.get(name)
        if prev is None or cur is None or prev.unit != cur.unit:
            continue
        assert prev.value is not None and cur.value is not None
        metrics.append(
            MetricChange(
                name=name,
                unit=cur.unit,
                previous_value=prev.value,
                current_value=cur.value,
                delta=cur.value - prev.value,
                previous_display=prev.display,
                current_display=cur.display,
                previous_period=prev.period_label,
                current_period=cur.period_label,
                previous_calc_id=prev.calc_id,
                current_calc_id=cur.calc_id,
            )
        )

    prev_conflicts = {conflict_key(c) for c in before.conflicts}
    cur_conflicts = {conflict_key(c) for c in now.conflicts}
    prev_unc, cur_unc = set(before.uncertainties), set(now.uncertainties)

    prev_facts, cur_facts = (
        _facts_block(before.freshness_summary),
        _facts_block(now.freshness_summary),
    )
    prev_prices, cur_prices = (
        _prices_block(before.freshness_summary),
        _prices_block(now.freshness_summary),
    )
    prev_q, cur_q = prev_facts.get("latest_quarter_end"), cur_facts.get("latest_quarter_end")
    prev_px, cur_px = prev_prices.get("latest_date"), cur_prices.get("latest_date")
    freshness = FreshnessChange(
        previous_latest_quarter_end=prev_q,
        current_latest_quarter_end=cur_q,
        new_quarter=bool(prev_q and cur_q and str(cur_q) > str(prev_q)),
        previous_price_date=prev_px,
        current_price_date=cur_px,
        newer_prices=bool(prev_px and cur_px and str(cur_px) > str(prev_px)),
    )

    diff = ThesisDiff(
        previous_analysis_id=previous.analysis_id,
        previous_as_of=previous.as_of,
        previous_created_at=previous.created_at,
        previous_horizon=previous.horizon,
        overall=overall,
        horizons=horizons,
        metrics=metrics,
        new_conflicts=sorted(cur_conflicts - prev_conflicts),
        resolved_conflicts=sorted(prev_conflicts - cur_conflicts),
        new_uncertainties=[u for u in now.uncertainties if u not in prev_unc],
        resolved_uncertainties=[u for u in before.uncertainties if u not in cur_unc],
        freshness=freshness,
        stance_changed=overall.changed or any(h.changed for h in horizons),
        horizon_scope_changed=set(before.stances) != set(now.stances),
    )
    diff.summary = summarize(diff)
    return diff


def summarize(diff: ThesisDiff) -> list[str]:
    """Deterministic one-line statements, evidence-only (no prose from either run)."""
    lines: list[str] = []
    when = diff.previous_as_of.date().isoformat()
    lines.append(f"prior assessment {diff.previous_analysis_id} as of {when}")
    lines.append(_overall_line(diff))
    for item in diff.horizons:
        lines.append(_stance_line(HORIZON_LABELS.get(item.scope, item.scope), item))
    for metric in diff.metrics:
        lines.append(
            f"{metric.name}: {metric.previous_display} -> {metric.current_display} "
            f"({_delta_text(metric)})"
        )
    if diff.new_conflicts:
        lines.append("new conflicts: " + "; ".join(diff.new_conflicts))
    if diff.resolved_conflicts:
        lines.append("conflicts no longer present: " + "; ".join(diff.resolved_conflicts))
    f = diff.freshness
    if f.new_quarter:
        lines.append(
            f"new quarter since the prior assessment: latest quarter end "
            f"{f.current_latest_quarter_end} (was {f.previous_latest_quarter_end})"
        )
    elif f.previous_latest_quarter_end and f.current_latest_quarter_end:
        lines.append(
            f"no new quarter since the prior assessment (latest quarter end still "
            f"{f.current_latest_quarter_end})"
        )
    if f.newer_prices:
        lines.append(f"prices now to {f.current_price_date} (were to {f.previous_price_date})")
    return lines


def _overall_line(diff: ThesisDiff) -> str:
    if not diff.horizon_scope_changed:
        return _stance_line("overall stance", diff.overall)
    if not diff.overall.compared_horizons:
        return "overall stance: not compared (the two assessments share no horizon)"
    shared = ", ".join(diff.overall.compared_horizons)
    return _stance_line(f"overall stance over the horizons both assessed ({shared})", diff.overall)


def _stance_line(label: str, item: StanceChange) -> str:
    if item.previous is None and item.current is None:
        return f"{label}: not assessed in either run"
    if item.previous is None:
        return f"{label}: {item.current} (not assessed previously)"
    if item.current is None:
        return f"{label}: previously {item.previous} (not assessed now)"
    if item.changed:
        return f"{label}: {item.previous} -> {item.current} (changed)"
    return f"{label}: {item.current} (unchanged)"


def _delta_text(metric: MetricChange) -> str:
    if metric.delta is None:
        return "delta unavailable"
    if metric.unit == "percent":
        sign = "+" if metric.delta > 0 else ""
        return f"{sign}{metric.delta:.1f} points"
    if metric.unit == "percentile":
        sign = "+" if metric.delta > 0 else ""
        return f"{sign}{metric.delta:.0f} percentile points"
    return format_change(metric.delta, metric.unit)


# ------------------------------------------------------------------ Spark block -----------


def prior_assessment_block(diff: ThesisDiff) -> dict[str, Any]:
    """The bounded, evidence-only ``prior_assessment`` entry of the Spark bundle: previous
    stances, the deltas and the prior run's ``as_of``. Rendered by ``spark/prompt.py``."""
    return {
        "analysis_id": diff.previous_analysis_id,
        "as_of": diff.previous_as_of.isoformat(),
        "horizon": diff.previous_horizon,
        "stance_changed": diff.stance_changed,
        "horizon_scope_changed": diff.horizon_scope_changed,
        "overall_compared_horizons": list(diff.overall.compared_horizons),
        "stances": [
            {
                "scope": s.scope,
                "previous": s.previous,
                "current": s.current,
                "changed": s.changed,
            }
            for s in (diff.overall, *diff.horizons)
        ],
        "metrics": [
            {
                "name": m.name,
                "previous": m.previous_display,
                "current": m.current_display,
                "delta": _delta_text(m),
                "previous_period": m.previous_period,
                "current_period": m.current_period,
            }
            for m in diff.metrics[:PRIOR_BLOCK_MAX_METRICS]
        ],
        "new_conflicts": diff.new_conflicts[:PRIOR_BLOCK_MAX_ITEMS],
        "resolved_conflicts": diff.resolved_conflicts[:PRIOR_BLOCK_MAX_ITEMS],
        "freshness": diff.freshness.model_dump(),
        "summary": diff.summary[:PRIOR_BLOCK_MAX_SUMMARY],
    }


__all__ = [
    "THESIS_METRICS",
    "ThesisSnapshot",
    "diff_assessments",
    "find_prior_assessment",
    "no_prior_assessment_note",
    "overall_stance",
    "prior_assessment_block",
    "snapshot_of_assembled",
    "snapshot_of_draft",
    "snapshot_of_result",
    "stances_from_decisions",
    "summarize",
]
