"""Prior-assessment lookup and the thesis diff.

Unit tests compare two synthetic results (changed stance, unchanged stance, new and resolved
conflicts, metric deltas, freshness); the end-to-end tests run "Assess Apple." twice through
the real runtime with the doubles and check that the second result carries a ``thesis_diff``
referencing the first and that the Spark prompt received the prior-assessment block.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from bayanalytics.config import Settings
from bayanalytics.instruments.base import LayaDecisions
from bayanalytics.pipeline.thesis import (
    THESIS_METRICS,
    ThesisSnapshot,
    diff_assessments,
    find_prior_assessment,
    no_prior_assessment_note,
    overall_stance,
    prior_assessment_block,
    stances_from_decisions,
)
from bayanalytics.schemas.calculations import CalculationInput, CalculationResult
from bayanalytics.schemas.common import utcnow
from bayanalytics.schemas.decisions import ChoiceAnswer, LayaDecision, LayaQuestion
from bayanalytics.schemas.evidence import Conflict, ConflictValue
from bayanalytics.schemas.results import (
    AnalysisResult,
    Assessment,
    HorizonAssessment,
    InstrumentView,
)
from bayanalytics.spark.prompt import PRIOR_MAX_LINES, render_prior_assessment
from bayanalytics.store import InMemoryStore
from bayanalytics.wiring import build_runtime
from doubles import FixedTranscriber, RuleLaya, ScriptedSpark, fixture_research_stack
from test_vertical_slice import FIXTURES, _run_to_completion, _runtime, _settings

# --- synthetic results -----------------------------------------------------------------------


def _calc(name: str, value: float, unit: str, display: str, period: str) -> CalculationResult:
    return CalculationResult(
        calc_id=f"calc_{name}",
        name=name,
        formula="f",
        inputs=[CalculationInput(name="x", value=value, source_id="src_1")],
        value=value,
        unit=unit,
        period_label=period,
        display=display,
    )


def _conflict(metric: str, period: str | None = "Q3 FY2026") -> Conflict:
    return Conflict(
        metric=metric,
        period_label=period,
        values=[
            ConflictValue(value=1.0, source_id="src_1"),
            ConflictValue(value=2.0, source_id="src_2"),
        ],
        material=True,
    )


def make_result(
    analysis_id: str,
    created_at: datetime,
    stances: dict[str, str],
    calcs: list[CalculationResult],
    *,
    conflicts: list[Conflict] | None = None,
    uncertainties: list[str] | None = None,
    quarter_end: str | None = "2026-06-27",
    price_date: str | None = "2026-09-25",
    horizon: str = "multi_horizon",
) -> AnalysisResult:
    return AnalysisResult(
        analysis_id=analysis_id,
        status="completed",
        query="Assess Apple.",
        instrument=InstrumentView(symbol="AAPL", name="Apple Inc."),
        profile="fast",
        horizon=horizon,  # type: ignore[arg-type]
        as_of=created_at,
        created_at=created_at,
        completed_at=created_at + timedelta(seconds=30),
        assessment=Assessment(conflicts=conflicts or [], uncertainties=uncertainties or []),
        horizon_assessments={
            h: HorizonAssessment(horizon=h, stance=s)  # type: ignore[arg-type]
            for h, s in stances.items()
        },
        calculations=calcs,
        freshness_summary={
            "facts": {"latest_quarter_end": quarter_end},
            "prices": {"latest_date": price_date},
        },
    )


T0 = datetime(2026, 6, 30, 12, tzinfo=UTC)
T1 = datetime(2026, 9, 27, 12, tzinfo=UTC)
FOUR_BULLISH = {h: "bullish" for h in ("near_term", "next_cycle", "medium_term", "long_term")}


def _baseline_calcs(revenue_growth: float = 12.0, pe: float = 30.0) -> list[CalculationResult]:
    return [
        _calc(
            "revenue_growth_yoy", revenue_growth, "percent", f"{revenue_growth:.1f}%", "Q2 vs Q2"
        ),
        _calc("eps_growth_yoy", 10.0, "percent", "10.0%", "Q2 vs Q2"),
        _calc("operating_margin", 30.0, "percent", "30.0%", "TTM"),
        _calc("pe_ttm", pe, "ratio", f"{pe:.1f}\u00d7", "close; EPS TTM"),
        _calc("price_return_1y", 20.0, "percent", "20.0%", "1y"),
        _calc("market_cap", 3.0e12, "USD", "$3.0T", "close"),  # not a thesis metric
    ]


# --- unit: stances ----------------------------------------------------------------------------


def test_overall_stance_rules() -> None:
    assert overall_stance({}) is None
    assert overall_stance({"near_term": "bullish", "long_term": "bullish"}) == "bullish"
    assert overall_stance({"near_term": "bearish", "long_term": "bearish"}) == "bearish"
    assert overall_stance({"near_term": "bullish", "long_term": "bearish"}) == "mixed"
    assert overall_stance({"near_term": "neutral", "long_term": "neutral"}) == "neutral"
    assert overall_stance({"near_term": "neutral", "long_term": "bullish"}) == "bullish"
    assert overall_stance({"near_term": "mixed", "long_term": "bullish"}) == "mixed"
    assert overall_stance({"near_term": "mixed"}) == "mixed"


def test_stances_from_decisions_apply_the_confidence_floor() -> None:
    def decision(horizon: str, choice: str, confidence: float) -> LayaDecision:
        return LayaDecision(
            decision_id=f"dec_{horizon}",
            stage="horizon",
            decision_type=f"horizon_stance_{horizon}",
            question=LayaQuestion(
                type="choice", instructions="Stance?", criteria={"bullish": "b", "bearish": "s"}
            ),
            answer=ChoiceAnswer(
                choice=choice, probabilities={"bullish": confidence, "bearish": 1 - confidence}
            ),
            confidence=confidence,
            state_digest="sha256:x",
            created_at=utcnow(),
        )

    decisions = LayaDecisions(
        decisions=[decision("near_term", "bullish", 0.8), decision("long_term", "bullish", 0.3)]
    )
    stances = stances_from_decisions(decisions, ["near_term", "next_cycle", "long_term"])
    assert stances == {"near_term": "bullish", "long_term": "mixed"}  # next_cycle: no decision


# --- unit: the diff ---------------------------------------------------------------------------


def test_changed_stance_is_reported_with_previous_values() -> None:
    previous = make_result("an_prev", T0, FOUR_BULLISH, _baseline_calcs())
    current = make_result(
        "an_cur", T1, {**FOUR_BULLISH, "medium_term": "mixed"}, _baseline_calcs(revenue_growth=8.0)
    )
    diff = diff_assessments(previous, current)
    assert diff.previous_analysis_id == "an_prev"
    assert diff.previous_as_of == T0 and diff.previous_created_at == T0
    assert diff.previous_horizon == "multi_horizon"
    assert diff.overall.model_dump() == {
        "scope": "overall",
        "previous": "bullish",
        "current": "mixed",
        "changed": True,
        "compared_horizons": ["near_term", "next_cycle", "medium_term", "long_term"],
    }
    assert diff.horizon_scope_changed is False
    by_scope = {h.scope: h for h in diff.horizons}
    assert list(by_scope) == ["near_term", "next_cycle", "medium_term", "long_term"]
    assert by_scope["medium_term"].previous == "bullish"
    assert by_scope["medium_term"].current == "mixed"
    assert by_scope["medium_term"].changed is True
    assert by_scope["near_term"].changed is False and by_scope["near_term"].current == "bullish"
    assert diff.stance_changed is True
    revenue = next(m for m in diff.metrics if m.name == "revenue_growth_yoy")
    assert revenue.previous_value == 12.0 and revenue.current_value == 8.0
    assert revenue.delta == -4.0 and revenue.previous_display == "12.0%"
    assert revenue.previous_calc_id == "calc_revenue_growth_yoy"
    assert "overall stance: bullish -> mixed (changed)" in diff.summary
    assert "Medium term (6-12 months): bullish -> mixed (changed)" in diff.summary
    assert "Near term (days to several weeks): bullish (unchanged)" in diff.summary
    assert "revenue_growth_yoy: 12.0% -> 8.0% (-4.0 points)" in diff.summary
    assert diff.summary[0] == "prior assessment an_prev as of 2026-06-30"
    assert diff.new_conflicts == [] and diff.resolved_conflicts == []


def test_unchanged_assessment_has_no_changes() -> None:
    previous = make_result("an_prev", T0, FOUR_BULLISH, _baseline_calcs())
    current = make_result("an_cur", T1, FOUR_BULLISH, _baseline_calcs())
    diff = diff_assessments(previous, current)
    assert diff.stance_changed is False and diff.overall.changed is False
    assert all(h.changed is False for h in diff.horizons)
    assert [m.name for m in diff.metrics] == [
        "revenue_growth_yoy",
        "eps_growth_yoy",
        "operating_margin",
        "pe_ttm",
        "price_return_1y",
    ]  # in THESIS_METRICS order; market_cap is not compared
    assert all(m.delta == 0.0 for m in diff.metrics)
    assert all(name in THESIS_METRICS for name in (m.name for m in diff.metrics))
    assert "overall stance: bullish (unchanged)" in diff.summary
    assert "pe_ttm: 30.0\u00d7 -> 30.0\u00d7 (0.0\u00d7)" in diff.summary
    assert diff.freshness.new_quarter is False and diff.freshness.newer_prices is False
    assert "no new quarter since the prior assessment (latest quarter end still 2026-06-27)" in (
        diff.summary
    )
    # Pure: the same inputs give the same diff.
    assert diff_assessments(previous, current).model_dump() == diff.model_dump()


def test_new_and_resolved_conflicts_uncertainties_and_freshness() -> None:
    previous = make_result(
        "an_prev",
        T0,
        {"near_term": "neutral"},
        _baseline_calcs(),
        conflicts=[_conflict("revenue", "Q2 FY2026")],
        uncertainties=["guidance not found", "old warning"],
        quarter_end="2026-03-28",
        price_date="2026-06-30",
        horizon="near_term",
    )
    current = make_result(
        "an_cur",
        T1,
        {"near_term": "neutral", "next_cycle": "bearish"},
        _baseline_calcs(),
        conflicts=[_conflict("net_income", "Q3 FY2026"), _conflict("eps_diluted", None)],
        uncertainties=["guidance not found", "new warning"],
        quarter_end="2026-06-27",
        price_date="2026-09-25",
    )
    diff = diff_assessments(previous, current)
    assert diff.new_conflicts == ["eps_diluted", "net_income Q3 FY2026"]
    assert diff.resolved_conflicts == ["revenue Q2 FY2026"]
    assert diff.new_uncertainties == ["new warning"]
    assert diff.resolved_uncertainties == ["old warning"]
    assert diff.freshness.model_dump() == {
        "previous_latest_quarter_end": "2026-03-28",
        "current_latest_quarter_end": "2026-06-27",
        "new_quarter": True,
        "previous_price_date": "2026-06-30",
        "current_price_date": "2026-09-25",
        "newer_prices": True,
    }
    # A horizon assessed only now is recorded, not counted as a change, and the overall stance
    # compares the one horizon both runs assessed: a wider scope is not a change of thesis.
    by_scope = {h.scope: h for h in diff.horizons}
    assert by_scope["next_cycle"].previous is None and by_scope["next_cycle"].changed is False
    assert diff.overall.previous == "neutral" and diff.overall.current == "neutral"
    assert diff.overall.compared_horizons == ["near_term"] and diff.overall.changed is False
    assert diff.horizon_scope_changed is True and diff.stance_changed is False
    assert (
        "overall stance over the horizons both assessed (near_term): neutral (unchanged)"
        in diff.summary
    )
    assert "new conflicts: eps_diluted; net_income Q3 FY2026" in diff.summary
    assert "conflicts no longer present: revenue Q2 FY2026" in diff.summary
    assert (
        "new quarter since the prior assessment: latest quarter end 2026-06-27 (was 2026-03-28)"
        in diff.summary
    )
    assert "prices now to 2026-09-25 (were to 2026-06-30)" in diff.summary
    assert "Next cycle (next earnings / quarter): bearish (not assessed previously)" in diff.summary


def test_narrower_horizon_scope_is_not_a_change_of_overall_stance() -> None:
    # A multi-horizon run that aggregates to mixed, then a near-term run that is still
    # bullish on the near term: the scope narrowed, the thesis did not change.
    previous = make_result(
        "an_prev", T0, {"near_term": "bullish", "long_term": "bearish"}, _baseline_calcs()
    )
    current = make_result(
        "an_cur", T1, {"near_term": "bullish"}, _baseline_calcs(), horizon="near_term"
    )
    diff = diff_assessments(previous, current)
    assert diff.overall.previous == "bullish" and diff.overall.current == "bullish"
    assert diff.overall.compared_horizons == ["near_term"] and diff.overall.changed is False
    assert diff.horizon_scope_changed is True and diff.stance_changed is False
    by_scope = {h.scope: h for h in diff.horizons}
    assert by_scope["long_term"].current is None and by_scope["long_term"].changed is False
    assert (
        "overall stance over the horizons both assessed (near_term): bullish (unchanged)"
        in diff.summary
    )
    lines = render_prior_assessment(prior_assessment_block(diff))
    assert (
        "- stance overall over the horizons both runs assessed (near_term): then bullish, "
        "now bullish (unchanged)"
    ) in lines
    assert not any("mixed" in line and "overall" in line for line in lines)
    # A real change on the shared horizon is still reported.
    moved = make_result("an_cur", T1, {"near_term": "bearish"}, _baseline_calcs())
    diff = diff_assessments(previous, moved)
    assert diff.overall.previous == "bullish" and diff.overall.current == "bearish"
    assert diff.overall.changed is True and diff.stance_changed is True


def test_disjoint_horizon_scopes_are_not_compared_overall() -> None:
    previous = make_result("an_prev", T0, {"long_term": "bearish"}, _baseline_calcs())
    current = make_result("an_cur", T1, {"near_term": "bullish"}, _baseline_calcs())
    diff = diff_assessments(previous, current)
    assert diff.overall.previous is None and diff.overall.current is None
    assert diff.overall.compared_horizons == [] and diff.overall.changed is False
    assert diff.horizon_scope_changed is True and diff.stance_changed is False
    assert "overall stance: not compared (the two assessments share no horizon)" in diff.summary
    lines = render_prior_assessment(prior_assessment_block(diff))
    assert "- stance overall: not compared (the runs share no horizon)" in lines
    assert "- stance long_term: then bearish, now not assessed (unchanged)" in lines


def test_metrics_are_compared_only_when_computed_in_both_with_one_unit() -> None:
    previous = make_result(
        "an_prev",
        T0,
        FOUR_BULLISH,
        [
            _calc("revenue_growth_yoy", 12.0, "percent", "12.0%", "Q2"),
            _calc("pe_ttm", 30.0, "ratio", "30.0x", "close"),
        ],
    )
    unavailable = _calc("revenue_growth_yoy", 0.0, "percent", "unavailable", "Q3")
    unavailable.status = "unavailable"
    unavailable.value = None
    current = make_result(
        "an_cur",
        T1,
        FOUR_BULLISH,
        [unavailable, _calc("pe_ttm", 31.0, "percent", "31.0%", "close")],
    )
    diff = diff_assessments(previous, current)
    assert diff.metrics == []  # one unavailable now, the other changed unit


def test_snapshot_input_and_prior_block_are_bounded() -> None:
    previous = make_result(
        "an_prev",
        T0,
        FOUR_BULLISH,
        [_calc(name, 1.0, "percent", "1.0%", "p") for name in THESIS_METRICS],
        conflicts=[_conflict(f"m{i}") for i in range(6)],
    )
    snapshot = ThesisSnapshot(
        stances={"near_term": "bearish"},
        calculations=[_calc(name, 2.0, "percent", "2.0%", "q") for name in THESIS_METRICS],
        conflicts=[_conflict(f"n{i}") for i in range(6)],
        uncertainties=["u"],
        freshness_summary={},
    )
    diff = diff_assessments(previous, snapshot)
    assert len(diff.metrics) == len(THESIS_METRICS) and len(diff.new_conflicts) == 6
    block = prior_assessment_block(diff)
    assert block["analysis_id"] == "an_prev" and block["as_of"] == T0.isoformat()
    assert block["horizon"] == "multi_horizon" and block["stance_changed"] is True
    assert len(block["metrics"]) == 10 and len(block["new_conflicts"]) == 4
    assert len(block["resolved_conflicts"]) == 4 and len(block["summary"]) == 8
    assert block["stances"][0] == {
        "scope": "overall",
        "previous": "bullish",
        "current": "bearish",
        "changed": True,
    }
    assert block["metrics"][0] == {
        "name": "revenue_growth_yoy",
        "previous": "1.0%",
        "current": "2.0%",
        "delta": "+1.0 points",
        "previous_period": "p",
        "current_period": "q",
    }
    assert block["freshness"]["new_quarter"] is False
    lines = render_prior_assessment(block)
    assert 1 < len(lines) <= PRIOR_MAX_LINES
    assert lines[0].startswith(
        "Prior assessment (deterministic comparison with analysis an_prev as of 2026-06-30"
    )
    assert (
        "- stance overall over the horizons both runs assessed (near_term): then bullish, "
        "now bearish (changed)"
    ) in lines
    assert block["horizon_scope_changed"] is True
    assert block["overall_compared_horizons"] == ["near_term"]
    assert "- stance near_term: then bullish, now bearish (changed)" in lines
    assert "- revenue_growth_yoy: then 1.0%, now 2.0% (+1.0 points) [p -> q]" in lines
    assert "- new conflict: n0 Q3 FY2026" in lines and "- conflict gone: m0 Q3 FY2026" in lines
    # The narrative of the prior run is never part of the block.
    assert "summary" not in " ".join(lines).lower().replace("summary", "", 0)
    assert not any("Evidence suggests" in line for line in lines)


# --- end to end ------------------------------------------------------------------------------


class _RecordingStore(InMemoryStore):
    """The real in-memory store, recording how the pipeline asks for the prior assessment."""

    def __init__(self) -> None:
        super().__init__()
        self.lookups: list[dict[str, Any]] = []

    async def latest_completed_result(self, symbol: str, **kwargs: Any) -> AnalysisResult | None:
        self.lookups.append({"symbol": symbol, **kwargs})
        return await super().latest_completed_result(symbol, **kwargs)


async def test_orchestrator_asks_for_the_prior_assessment_unscoped_and_says_so() -> None:
    store = _RecordingStore()
    settings = _settings()
    rt = build_runtime(
        settings,
        store=store,
        laya=RuleLaya(),
        spark=ScriptedSpark(),
        transcriber=FixedTranscriber(),
        research=fixture_research_stack(settings, FIXTURES),
    )
    analysis_id, _events, result = await _run_to_completion(rt, {"query": "Assess Apple."})
    assert result["status"] == "completed"
    job = await store.get_job(analysis_id)
    assert job is not None
    # owner_id is passed explicitly (present in the call, and None: single-tenant seam).
    assert store.lookups == [{"symbol": "AAPL", "before": job.created_at, "owner_id": None}]


async def test_find_prior_assessment_requires_an_explicit_owner_scope() -> None:
    store = InMemoryStore()
    with pytest.raises(TypeError):
        await find_prior_assessment(store, "AAPL", before=T1)  # type: ignore[call-arg]
    assert await find_prior_assessment(store, "AAPL", before=T1, owner_id=None) is None
    with pytest.raises(NotImplementedError, match="analyses do not carry ownership yet"):
        await find_prior_assessment(store, "AAPL", before=T1, owner_id="user_1")


def _user_prompt(spark: ScriptedSpark, index: int) -> str:
    return next(m.content for m in reversed(spark.runs[index]["messages"]) if m.role == "user")


async def test_second_assessment_carries_the_thesis_diff_against_the_first() -> None:
    spark = ScriptedSpark()
    rt = _runtime(spark=spark)
    first_id, _events, first = await _run_to_completion(rt, {"query": "Assess Apple."})
    assert first["status"] == "completed" and first["thesis_diff"] is None
    assert no_prior_assessment_note("AAPL") in first["assessment"]["uncertainties"]
    first_prompt = _user_prompt(spark, 0)
    assert "Prior assessment: (none found for this instrument)" in first_prompt
    assert no_prior_assessment_note("AAPL") in first_prompt
    assert "A prior assessment block is included" not in first_prompt

    second_id, _events, second = await _run_to_completion(rt, {"query": "Assess Apple."})
    assert second["status"] == "completed" and second_id != first_id
    diff = second["thesis_diff"]
    assert diff is not None
    assert diff["previous_analysis_id"] == first_id
    assert diff["previous_as_of"] == first["as_of"]
    assert diff["previous_created_at"] == first["created_at"]
    assert diff["previous_horizon"] == "multi_horizon"
    assert no_prior_assessment_note("AAPL") not in second["assessment"]["uncertainties"]
    # Same fixture, same information set: nothing changed, and the diff says so explicitly.
    assert diff["stance_changed"] is False
    assert {h["scope"]: h for h in diff["horizons"]}.keys() == set(second["horizon_assessments"])
    for item in diff["horizons"]:
        assert item["previous"] == first["horizon_assessments"][item["scope"]]["stance"]
        assert item["current"] == second["horizon_assessments"][item["scope"]]["stance"]
        assert item["changed"] is False
    assert diff["overall"]["previous"] == diff["overall"]["current"]
    assert diff["metrics"] and all(m["delta"] == 0.0 for m in diff["metrics"])
    assert {m["name"] for m in diff["metrics"]} <= set(THESIS_METRICS)
    assert "valuation_reconciliation_1y" in {m["name"] for m in diff["metrics"]}
    assert diff["new_conflicts"] == [] and diff["resolved_conflicts"] == []
    assert diff["freshness"]["new_quarter"] is False
    assert diff["freshness"]["current_latest_quarter_end"] == "2026-06-27"
    assert diff["summary"][0] == f"prior assessment {first_id} as of {first['as_of'][:10]}"
    # Spark received the bounded prior block and the instruction to use it.
    second_prompt = _user_prompt(spark, 1)
    header = f"Prior assessment (deterministic comparison with analysis {first_id} as of "
    assert header in second_prompt
    block = second_prompt[second_prompt.index(header) : second_prompt.index("</EVIDENCE>")]
    assert 1 < len(block.strip().splitlines()) <= PRIOR_MAX_LINES
    overall = diff["overall"]
    assert f"- stance overall: then {overall['previous']}, now {overall['current']}" in block
    assert "- revenue_growth_yoy: then " in block
    assert "- latest quarter end: then 2026-06-27, now 2026-06-27 (no new quarter)" in block
    assert "A prior assessment block is included" in second_prompt
    # Reconciliation verdicts reach Spark as recorded values, with the restate-only rule.
    headline = next(c for c in second["calculations"] if c["name"] == "valuation_reconciliation_1y")
    verdict = headline["meta"]["reconciliation"]["verdict"]
    assert f'"verdict":"{verdict}"' in second_prompt
    assert '"verdict_rule":"' in second_prompt
    assert "restate each verdict and its numbers exactly as given" in second_prompt
    # The stored result and the store's own lookup agree with the API view.
    stored = await rt.store.get_result(second_id)
    assert stored is not None and stored.thesis_diff is not None
    assert stored.thesis_diff.previous_analysis_id == first_id
    latest = await rt.store.latest_completed_result("AAPL")
    assert latest is not None and latest.analysis_id == second_id
    earlier = await rt.store.latest_completed_result("AAPL", before=stored.created_at)
    assert earlier is not None and earlier.analysis_id == first_id


async def test_a_new_quarter_between_assessments_is_visible_in_the_diff() -> None:
    # Two runtimes over one store: the first frozen at 2026-06-30 (Q2 FY2026 is the latest
    # quarter), the second at 2026-09-26 after the Q3 10-Q; the diff reports the new quarter
    # and the moved metrics instead of leaving that to inference.
    store = InMemoryStore()
    spark = ScriptedSpark()

    def runtime(settings: Settings):
        return build_runtime(
            settings,
            store=store,
            laya=RuleLaya(),
            spark=spark,
            transcriber=FixedTranscriber(),
            research=fixture_research_stack(settings, FIXTURES),
        )

    first_id, _e, first = await _run_to_completion(
        runtime(_settings(eval_as_of=datetime(2026, 6, 30, tzinfo=UTC))),
        {"query": "Assess Apple."},
    )
    _second_id, _e, second = await _run_to_completion(
        runtime(_settings(eval_as_of=datetime(2026, 9, 26, tzinfo=UTC))),
        {"query": "Assess Apple."},
    )
    assert first["freshness_summary"]["facts"]["latest_quarter_end"] == "2026-03-28"
    diff = second["thesis_diff"]
    assert diff is not None and diff["previous_analysis_id"] == first_id
    assert diff["previous_as_of"].startswith("2026-06-30")
    assert diff["freshness"] == {
        "previous_latest_quarter_end": "2026-03-28",
        "current_latest_quarter_end": "2026-06-27",
        "new_quarter": True,
        "previous_price_date": "2026-06-30",
        "current_price_date": "2026-09-25",
        "newer_prices": True,
    }
    assert (
        "new quarter since the prior assessment: latest quarter end 2026-06-27 (was 2026-03-28)"
        in diff["summary"]
    )
    metrics: dict[str, dict[str, Any]] = {m["name"]: m for m in diff["metrics"]}
    assert metrics["price_return_1y"]["delta"] != 0.0
    assert metrics["revenue_growth_yoy"]["previous_period"] == "Q2 FY2026 vs Q2 FY2025"
    assert metrics["revenue_growth_yoy"]["current_period"] == "Q3 FY2026 vs Q3 FY2025"
    prompt = _user_prompt(spark, 1)
    assert "- latest quarter end: then 2026-03-28, now 2026-06-27 (new quarter)" in prompt
    assert "- latest close: then 2026-06-30, now 2026-09-25" in prompt
