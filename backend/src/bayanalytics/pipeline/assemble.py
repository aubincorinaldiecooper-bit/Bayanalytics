"""Deterministic assembly of the structured result (AGENT.md sections 12, 37.3).

Spark writes the narrative; every list and number in the structured sections comes from the
normalized evidence, the Laya decisions and the deterministic calculations, each tied to its
source ids. Nothing here invents a value.
"""

from __future__ import annotations

from typing import Any

from bayanalytics.instruments.base import CalculatedMetrics, LayaDecisions
from bayanalytics.schemas.calculations import CalculationResult
from bayanalytics.schemas.common import HORIZON_LABELS
from bayanalytics.schemas.decisions import ChoiceAnswer, LayaDecision, NoulAnswer, ScoreAnswer
from bayanalytics.schemas.evidence import NormalizedEvidence
from bayanalytics.schemas.results import Assessment, EvidenceItem, HorizonAssessment

_VALUATION_LEVELS = [
    "well below historical norm",
    "below historical norm",
    "near historical norm",
    "above historical norm",
    "far above historical norm",
]


def _calc_view(calc: CalculationResult) -> dict[str, Any]:
    return {
        "value": calc.value,
        "unit": calc.unit,
        "display": calc.display,
        "period": calc.period_label,
        "calc_id": calc.calc_id,
        "source_ids": sorted({i.source_id for i in calc.inputs if i.source_id}),
    }


def _decision_view(decision: LayaDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    view: dict[str, Any] = {
        "decision": decision.decision,
        "confidence": round(decision.confidence, 3),
        "decision_id": decision.decision_id,
    }
    if (
        isinstance(decision.answer, ScoreAnswer)
        and decision.decision_type == "valuation_extremeness"
    ):
        idx = min(max(round(decision.answer.score), 0), len(_VALUATION_LEVELS) - 1)
        view["label"] = _VALUATION_LEVELS[idx]
    return view


def _calc_sources(calc: CalculationResult) -> list[str]:
    return sorted({i.source_id for i in calc.inputs if i.source_id})


def _pct(calc: CalculationResult) -> float | None:
    return calc.value


def build_structured_sections(
    evidence: NormalizedEvidence,
    decisions: LayaDecisions,
    calculations: CalculatedMetrics,
) -> dict[str, Any]:
    by_name = {c.name: c for c in calculations.calculations if c.status == "computed"}
    scan = {d.decision_type: d for d in decisions.decisions if d.stage == "evidence_scan"}
    horizon_stage = {d.decision_type: d for d in decisions.decisions if d.stage == "horizon"}

    fundamentals: dict[str, Any] = {}
    for name in (
        "revenue_growth_yoy",
        "revenue_growth_qoq",
        "revenue_cagr_3y",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "fcf_margin",
        "operating_margin_change_bp",
        "free_cash_flow_ttm",
        "eps_growth_yoy",
    ):
        if name in by_name:
            fundamentals[name] = _calc_view(by_name[name])
    for key in ("guidance_trend", "margin_direction", "revenue_momentum", "growth_durability"):
        decision = scan.get(key) or horizon_stage.get(key)
        if decision is not None:
            fundamentals[key] = _decision_view(decision)

    valuation: dict[str, Any] = {}
    for name in (
        "market_cap",
        "enterprise_value",
        "pe_ttm",
        "ps_ttm",
        "ev_ebitda_ttm",
        "fcf_yield_ttm",
        "pe_5y_percentile",
    ):
        if name in by_name:
            valuation[name] = _calc_view(by_name[name])
    if (
        d := horizon_stage.get("valuation_extremeness") or scan.get("valuation_extremeness")
    ) is not None:
        valuation["valuation_extremeness"] = _decision_view(d)

    benchmark_context: dict[str, Any] = {
        "benchmarks": [r.model_dump() for r in evidence.benchmark_refs],
    }
    for name in (
        "relative_return_1y_vs_market",
        "relative_return_1y_vs_sector",
        "beta_1y_vs_market",
        "drawdown_vs_market_1y",
    ):
        if name in by_name:
            benchmark_context[name] = _calc_view(by_name[name])
    for key in ("benchmark_relative", "drawdown_nature"):
        decision = horizon_stage.get(key) or scan.get(key)
        if decision is not None:
            benchmark_context[key] = _decision_view(decision)

    historical_context: dict[str, Any] = {
        "periods": [seg.summary for seg in evidence.segments[-8:]],
    }
    if "revenue_cagr_3y" in by_name:
        historical_context["revenue_cagr_3y"] = _calc_view(by_name["revenue_cagr_3y"])
    unusual = [
        {
            "period": next(
                (
                    s.summary.get("period")
                    for s in evidence.segments
                    if s.segment_id == d.segment_id
                ),
                None,
            ),
            "probability": round(d.answer.noul, 3),
            "decision_id": d.decision_id,
        }
        for d in decisions.decisions
        if d.decision_type == "historically_unusual"
        and isinstance(d.answer, NoulAnswer)
        and d.answer.noul >= 0.6
    ]
    if unusual:
        historical_context["unusual_periods"] = unusual
    if (d := scan.get("historically_unusual")) is not None:
        historical_context["historically_unusual"] = _decision_view(d)

    market_context: dict[str, Any] = {}
    for name in (
        "price_return_1m",
        "price_return_3m",
        "price_return_6m",
        "price_return_1y",
        "price_return_ytd",
        "volatility_30d_annualized",
        "volatility_1y_annualized",
        "max_drawdown_1y",
        "ma_50",
        "ma_200",
        "price_vs_ma_200_pct",
    ):
        if name in by_name:
            market_context[name] = _calc_view(by_name[name])
    for key in ("volatility_regime", "sentiment_trend"):
        decision = scan.get(key)
        if decision is not None:
            market_context[key] = _decision_view(decision)
    if evidence.prices is not None:
        market_context["price"] = {
            "value": evidence.prices.latest.close if evidence.prices.latest else None,
            "price_type": evidence.prices.price_type,
            "session_date": evidence.prices.session_date.isoformat()
            if evidence.prices.session_date
            else None,
            "exchange_timezone": evidence.prices.exchange_timezone,
            "source_id": evidence.prices.source_id,
        }
    return {
        "fundamentals": fundamentals,
        "valuation": valuation,
        "benchmark_context": benchmark_context,
        "historical_context": historical_context,
        "market_context": market_context,
    }


def build_evidence_lists(
    evidence: NormalizedEvidence,
    decisions: LayaDecisions,
    calculations: CalculatedMetrics,
) -> tuple[list[EvidenceItem], list[EvidenceItem], list[EvidenceItem], list[EvidenceItem]]:
    """Returns (bull, bear, risks, what_changed) built from cited facts and decisions."""
    by_name = {c.name: c for c in calculations.calculations if c.status == "computed"}
    bull: list[EvidenceItem] = []
    bear: list[EvidenceItem] = []
    risks: list[EvidenceItem] = []
    changed: list[EvidenceItem] = []

    def signed(name: str, label: str, positive_is_bull: bool = True) -> None:
        calc = by_name.get(name)
        if calc is None or calc.value is None:
            return
        item = EvidenceItem(
            text=f"{label} {calc.display}"
            + (f" ({calc.period_label})" if calc.period_label else ""),
            source_ids=_calc_sources(calc),
            metric=name,
            period_label=calc.period_label,
            calc_id=calc.calc_id,
        )
        is_positive = calc.value > 0
        if (is_positive and positive_is_bull) or (not is_positive and not positive_is_bull):
            item.stance = "bullish"
            bull.append(item)
        elif calc.value != 0:
            item.stance = "bearish"
            bear.append(item)

    signed("revenue_growth_yoy", "Revenue growth (YoY)")
    signed("eps_growth_yoy", "Diluted EPS growth (YoY)")
    signed("operating_margin_change_bp", "Operating margin change")
    signed("relative_return_1y_vs_market", "1-year return relative to the broad market")
    signed("relative_return_1y_vs_sector", "1-year return relative to the sector benchmark")

    # Text evidence labelled by Laya.
    stance_by_source = {
        d.segment_id: d
        for d in decisions.decisions
        if d.stage == "text_evidence" and d.decision_type == "evidence_stance" and d.segment_id
    }
    material_by_source = {
        d.segment_id: d
        for d in decisions.decisions
        if d.stage == "text_evidence" and d.decision_type == "source_is_material" and d.segment_id
    }
    for item in evidence.text_evidence:
        decision = stance_by_source.get(item["source_id"])
        if decision is None or not isinstance(decision.answer, ChoiceAnswer):
            continue
        material = material_by_source.get(item["source_id"])
        if (
            material is not None
            and isinstance(material.answer, NoulAnswer)
            and material.answer.noul < 0.4
        ):
            continue
        text = item["fact"].split(". ")[0][:220]
        ev = EvidenceItem(
            text=f"{item['title']}: {text}",
            source_ids=[item["source_id"]],
            stance=decision.answer.choice,  # type: ignore[arg-type]
            decision_id=decision.decision_id,
        )
        if decision.answer.choice == "bullish":
            bull.append(ev)
        elif decision.answer.choice == "bearish":
            bear.append(ev)

    # Laya scan decisions that are directional.
    scan = {d.decision_type: d for d in decisions.decisions if d.stage == "evidence_scan"}
    guidance = scan.get("guidance_trend")
    if guidance is not None and isinstance(guidance.answer, ChoiceAnswer):
        if guidance.answer.choice == "improving":
            bull.append(
                EvidenceItem(
                    text="Guidance trend classified as improving",
                    stance="bullish",
                    decision_id=guidance.decision_id,
                )
            )
        elif guidance.answer.choice == "deteriorating":
            bear.append(
                EvidenceItem(
                    text="Guidance trend classified as deteriorating",
                    stance="bearish",
                    decision_id=guidance.decision_id,
                )
            )

    # Risks: deterministic signals plus conflicts and stale evidence.
    vol = scan.get("volatility_regime")
    if vol is not None and isinstance(vol.answer, ChoiceAnswer) and vol.answer.choice == "elevated":
        calc = by_name.get("volatility_30d_annualized")
        risks.append(
            EvidenceItem(
                text="Volatility regime classified as elevated"
                + (f" (30-day annualized {calc.display})" if calc else ""),
                source_ids=_calc_sources(calc) if calc else [],
                decision_id=vol.decision_id,
                calc_id=calc.calc_id if calc else None,
            )
        )
    dd = by_name.get("max_drawdown_1y")
    if dd is not None and dd.value is not None and dd.value <= -0.2:
        risks.append(
            EvidenceItem(
                text=f"Maximum 1-year drawdown {dd.display}",
                source_ids=_calc_sources(dd),
                calc_id=dd.calc_id,
                metric=dd.name,
            )
        )
    valuation = next(
        (d for d in decisions.decisions if d.decision_type == "valuation_extremeness"), None
    )
    if (
        valuation is not None
        and isinstance(valuation.answer, ScoreAnswer)
        and valuation.answer.score >= 3.0
    ):
        risks.append(
            EvidenceItem(
                text="Valuation scored above its historical norm", decision_id=valuation.decision_id
            )
        )
    for conflict in evidence.conflicts:
        if conflict.material:
            risks.append(
                EvidenceItem(
                    text=f"Sources disagree on {conflict.metric}"
                    + (f" for {conflict.period_label}" if conflict.period_label else ""),
                    source_ids=[v.source_id for v in conflict.values],
                    metric=conflict.metric,
                    period_label=conflict.period_label,
                )
            )
    for warning in evidence.freshness_summary.get("warnings", [])[:2]:
        risks.append(EvidenceItem(text=str(warning)))

    # What changed: material history segments (latest first) and the latest quarter movement.
    material_segments = [
        d
        for d in decisions.decisions
        if d.stage == "history_scan"
        and d.decision_type == "material_change"
        and isinstance(d.answer, NoulAnswer)
        and d.answer.noul >= 0.6
    ]
    for d in material_segments[-3:][::-1]:
        seg = next((s for s in evidence.segments if s.segment_id == d.segment_id), None)
        if seg is None:
            continue
        parts = [f"{seg.summary.get('period')}"]
        if "revenue_growth_yoy" in seg.summary:
            parts.append(f"revenue {seg.summary['revenue_growth_yoy']:+.1f}% YoY")
        if "operating_margin_pct" in seg.summary:
            parts.append(f"operating margin {seg.summary['operating_margin_pct']:.1f}%")
        changed.append(
            EvidenceItem(
                text=": ".join([parts[0], ", ".join(parts[1:])]) if len(parts) > 1 else parts[0],
                source_ids=seg.source_ids,
                period_label=seg.summary.get("period"),
                decision_id=d.decision_id,
            )
        )
    if not changed and "revenue_growth_yoy" in by_name:
        calc = by_name["revenue_growth_yoy"]
        changed.append(
            EvidenceItem(
                text=f"Latest reported revenue growth {calc.display} YoY",
                source_ids=_calc_sources(calc),
                calc_id=calc.calc_id,
                period_label=calc.period_label,
            )
        )
    return bull, bear, risks, changed


def merge_horizons(
    horizons: list[str],
    parsed: dict[str, HorizonAssessment],
    decisions: LayaDecisions,
    bull: list[EvidenceItem],
    bear: list[EvidenceItem],
) -> dict[str, HorizonAssessment]:
    """Laya owns stance + confidence; Spark supplies the explanation."""
    out: dict[str, HorizonAssessment] = {}
    for horizon in horizons:
        decision = decisions.latest(f"horizon_stance_{horizon}")
        spark = parsed.get(horizon) or parsed.get(f"horizon_{horizon}")
        item = HorizonAssessment(
            horizon=horizon,
            summary=(spark.summary if spark else ""),
            key_evidence=list(spark.key_evidence) if spark else [],
        )
        if decision is not None and isinstance(decision.answer, ChoiceAnswer):
            item.stance = decision.answer.choice  # type: ignore[assignment]
            item.confidence = round(decision.confidence, 3)
            item.decision_id = decision.decision_id
        if not item.key_evidence:
            item.key_evidence = (bull[:2] + bear[:2])[:4]
        if not item.summary:
            item.summary = (
                f"{HORIZON_LABELS.get(horizon, horizon)}: Laya stance {item.stance}; see evidence."
            )
        out[horizon] = item
    return out


def finalize_assessment(
    assessment: Assessment,
    evidence: NormalizedEvidence,
    decisions: LayaDecisions,
    calculations: CalculatedMetrics,
    extra_uncertainties: list[str],
) -> Assessment:
    sections = build_structured_sections(evidence, decisions, calculations)
    bull, bear, risks, changed = build_evidence_lists(evidence, decisions, calculations)
    # Spark-parsed items are kept only when they cite known sources; deterministic items come first.
    known = {s.source_id for s in evidence.sources}

    def cited(items: list[EvidenceItem]) -> list[EvidenceItem]:
        return [i for i in items if i.source_ids and all(s in known for s in i.source_ids)]

    for key in (
        "fundamentals",
        "valuation",
        "benchmark_context",
        "historical_context",
        "market_context",
    ):
        structured = sections[key]
        narrative = getattr(assessment, key)
        if isinstance(narrative, dict) and (narrative.get("text") or narrative.get("items")):
            structured["narrative"] = {
                "text": narrative.get("text", ""),
                "items": narrative.get("items", []),
                "source_ids": [s for s in narrative.get("source_ids", []) if s in known],
            }
        setattr(assessment, key, structured)
    assessment.bull_evidence = bull + cited(assessment.bull_evidence)
    assessment.bear_evidence = bear + cited(assessment.bear_evidence)
    assessment.risks = risks + [r for r in assessment.risks if r.text]
    assessment.what_changed = changed + cited(assessment.what_changed)
    assessment.conflicts = list(evidence.conflicts)
    seen: set[str] = set()
    merged: list[str] = []
    for text in (
        list(evidence.uncertainties) + list(extra_uncertainties) + list(assessment.uncertainties)
    ):
        if text and text not in seen:
            seen.add(text)
            merged.append(text)
    assessment.uncertainties = merged
    return assessment
