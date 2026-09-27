"""Valuation-vs-fundamentals reconciliation and the historical P/E percentile window.

The arithmetic is checked directly (the product owner's example, negative growth, missing
and non-positive EPS, sign conventions), then through ``compute`` (recorded operands, the
decomposition in ``meta``, the verdict note) and through ``run_pack`` on synthetic evidence
(TTM pairs, the fiscal-year fallback, named missing operands, the attached percentile).
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from bayanalytics.calculations import primitives as prim
from bayanalytics.calculations import reconciliation as rec
from bayanalytics.calculations.operands import OperandResolver
from bayanalytics.calculations.registry import (
    CALCULATION_PACKS,
    PE_HISTORY_MIN_POINTS,
    SPECS,
    compute,
    run_pack,
)
from bayanalytics.schemas.calculations import CalculationInput
from bayanalytics.schemas.evidence import NormalizedFact, Period
from test_calculations import (
    AS_OF,
    _fact,
    annual_facts,
    balance_facts,
    build_evidence,
    quarterly_facts,
    synthetic_prices,
)

# ======================================================================================
# arithmetic
# ======================================================================================


class TestDecomposition:
    def test_owner_example(self):
        # price +35 %, EPS +10 %: ~23 % of the move is multiple expansion.
        parts = rec.decompose(0.35, 0.10)
        assert parts is not None
        assert parts.multiple_change == pytest.approx(1.35 / 1.10 - 1)
        assert round(parts.multiple_change * 100, 1) == 22.7
        assert parts.earnings_contribution == pytest.approx(0.10)
        assert parts.interaction == pytest.approx(0.10 * parts.multiple_change)
        assert parts.earnings_contribution + parts.multiple_contribution + parts.interaction == (
            pytest.approx(0.35)
        )
        assert parts.multiple_share == pytest.approx(parts.multiple_change / 0.35)
        assert parts.verdict.label == "mostly multiple expansion"
        assert parts.verdict.rule == "g >= +0.5, m >= +0.5, |m| - |g| > 1.0"
        record = parts.as_dict()
        assert record["verdict"] == "mostly multiple expansion"
        assert record["verdict_rule"] == "g >= +0.5, m >= +0.5, |m| - |g| > 1.0"
        assert record["thresholds"] == {"negligible_points": 0.5, "equal_points": 1.0}
        assert record["multiple_change_pct"] == pytest.approx(22.727272727)
        assert record["contributions_pct"]["earnings"] == pytest.approx(10.0)
        assert record["contributions_pct"]["interaction"] == pytest.approx(2.2727272727)
        assert record["shares_of_move"]["multiple"] == pytest.approx(0.6493506494)
        assert record["identity"] == "(1 + price_return) = (1 + eps_growth) x (1 + multiple_change)"

    @pytest.mark.parametrize(
        ("price_return", "eps_growth"),
        [(0.35, 0.10), (-0.10, 0.10), (0.10, -0.20), (-0.40, -0.10), (0.0, 0.05), (2.0, 0.5)],
    )
    def test_identity_holds(self, price_return: float, eps_growth: float):
        m = rec.implied_multiple_change(price_return, eps_growth)
        assert m is not None
        assert (1 + eps_growth) * (1 + m) == pytest.approx(1 + price_return)
        parts = rec.decompose(price_return, eps_growth)
        assert parts is not None
        total = parts.earnings_contribution + parts.multiple_contribution + parts.interaction
        assert total == pytest.approx(price_return)

    @pytest.mark.parametrize(
        ("price_return", "eps_growth", "expected"),
        [
            (0.0, 0.0, "no material change"),
            (0.10, 0.0, "multiple expansion with flat earnings"),
            (-0.10, 0.0, "multiple contraction with flat earnings"),
            (0.10, 0.10, "earnings growth at a steady multiple"),
            (-0.10, -0.10, "earnings decline at a steady multiple"),
            (0.35, 0.10, "mostly multiple expansion"),
            (0.35, 0.30, "mostly earnings growth"),
            (0.21, 0.10, "earnings and multiple contributed equally"),
            (-0.40, -0.10, "mostly multiple contraction"),
            (-0.20, -0.15, "mostly earnings decline"),
            (-0.19, -0.10, "earnings and multiple contracted equally"),
            (-0.10, 0.10, "de-rating despite growth"),
            (0.05, 0.10, "earnings growth offset by de-rating"),
            (0.10, -0.20, "re-rating despite earnings decline"),
            (-0.10, -0.30, "earnings decline offset by re-rating"),
        ],
    )
    def test_every_verdict_rule(self, price_return: float, eps_growth: float, expected: str):
        parts = rec.decompose(price_return, eps_growth)
        assert parts is not None
        assert parts.verdict.label == expected
        rule = next(rule for label, rule, _test in rec.VERDICT_RULES if label == expected)
        assert parts.verdict.rule == rule

    def test_the_rule_table_is_the_fixed_vocabulary(self):
        assert len(rec.VERDICT_RULES) == 15
        assert rec.VERDICTS == tuple(label for label, _r, _t in rec.VERDICT_RULES)
        assert len(set(rec.VERDICTS)) == len(rec.VERDICTS)
        assert rec.THRESHOLDS == {"negligible_points": 0.5, "equal_points": 1.0}
        # Every documented condition is printed in the module docstring, word for word.
        for label, rule, _test in rec.VERDICT_RULES:
            assert rule in rec.__doc__ and label in rec.__doc__, label

    def test_thresholds_at_their_boundaries(self):
        # negligible: |x| < 0.5 percentage points counts as flat
        assert rec.verdict(0.104, 0.004, 0.0996).label == "multiple expansion with flat earnings"
        assert rec.verdict(0.107, 0.006, 0.1004).label == "mostly multiple expansion"
        assert rec.verdict(0.004, 0.0, 0.004).label == "no material change"
        assert rec.verdict(0.1, 0.1, -0.004).label == "earnings growth at a steady multiple"
        # equal: magnitudes within 1.0 point contributed equally, beyond it "mostly"
        assert rec.verdict(0.2, 0.10, 0.109).label == "earnings and multiple contributed equally"
        assert rec.verdict(0.2, 0.10, 0.112).label == "mostly multiple expansion"
        assert rec.verdict(0.2, 0.109, 0.10).label == "earnings and multiple contributed equally"
        assert rec.verdict(0.2, 0.112, 0.10).label == "mostly earnings growth"
        assert rec.verdict(-0.2, -0.10, -0.109).label == "earnings and multiple contracted equally"
        # the price return decides "despite" vs "offset" at the same 0.5-point threshold
        assert rec.verdict(-0.006, 0.10, -0.096).label == "de-rating despite growth"
        assert rec.verdict(-0.004, 0.10, -0.094).label == "earnings growth offset by de-rating"
        assert rec.verdict(0.006, -0.10, 0.118).label == "re-rating despite earnings decline"
        assert rec.verdict(0.004, -0.10, 0.116).label == "earnings decline offset by re-rating"

    def test_rules_are_exhaustive(self):
        # Every (R, g) the identity can produce maps to exactly one documented verdict.
        steps = [-0.6, -0.3, -0.1, -0.012, -0.006, -0.004, 0.0, 0.004, 0.006, 0.012, 0.1, 0.5, 2.0]
        seen: set[str] = set()
        for r in steps:
            for g in [x for x in steps if x > -1.0]:
                parts = rec.decompose(r, g)
                assert parts is not None
                assert parts.verdict.label in rec.VERDICTS
                seen.add(parts.verdict.label)
        assert len(seen) >= 12

    def test_negative_growth(self):
        # EPS 1.00 -> 0.80 with the price 100 -> 110: the multiple re-rated by 37.5 %.
        g = rec.eps_ratio_growth(0.80, 1.00)
        assert g == pytest.approx(-0.20)
        parts = rec.decompose(0.10, g)
        assert parts is not None
        assert parts.multiple_change == pytest.approx(0.375)
        assert parts.earnings_contribution == pytest.approx(-0.20)
        assert parts.interaction == pytest.approx(-0.075)
        assert parts.verdict.label == "re-rating despite earnings decline"

    def test_non_positive_eps_has_no_multiple(self):
        assert rec.eps_ratio_growth(1.0, -2.0) is None
        assert rec.eps_ratio_growth(-1.0, 2.0) is None
        assert rec.eps_ratio_growth(1.0, 0.0) is None
        assert rec.eps_ratio_growth(2.0, 1.0) == pytest.approx(1.0)
        assert rec.implied_multiple_change(0.2, -1.5) is None  # 1 + g <= 0
        assert rec.implied_multiple_change(None, 0.1) is None
        assert rec.decompose(None, 0.1) is None
        assert rec.decompose(0.1, float("nan")) is None

    def test_sign_convention_differs_from_growth_rate_on_purpose(self):
        # growth_rate keeps the direction meaningful through a loss (a shrinking loss is
        # +50 %); the multiplicative identity needs the plain ratio, which has no meaning
        # through non-positive earnings, so the reconciliation refuses rather than approximates.
        assert prim.growth_rate(-50.0, -100.0) == pytest.approx(0.5)
        assert rec.eps_ratio_growth(-50.0, -100.0) is None


# ======================================================================================
# compute: recorded operands, decomposition in meta, verdict note
# ======================================================================================


def _ci(name: str, value: float | None, **kw) -> CalculationInput:
    return CalculationInput(name=name, value=value, **kw)


def owner_inputs(**overrides: CalculationInput | None) -> dict[str, CalculationInput | None]:
    inputs: dict[str, CalculationInput | None] = {
        "start_close": _ci("start_close", 100.0, period_label="close 2025-09-25", source_id="px"),
        "end_close": _ci("end_close", 135.0, period_label="close 2026-09-25", source_id="px"),
        "eps_current": _ci("eps_current", 1.10, period_label="TTM to 2026-06-27", source_id="x"),
        "eps_previous": _ci("eps_previous", 1.00, period_label="TTM to 2025-06-28", source_id="x"),
        "revenue_current": _ci("revenue_current", 108.0, period_label="TTM to 2026-06-27"),
        "revenue_previous": _ci("revenue_previous", 100.0, period_label="TTM to 2025-06-28"),
        "pe_percentile": _ci(
            "pe_percentile", 92.0, unit="percentile", period_label="over 19 available quarters"
        ),
    }
    inputs.update(overrides)
    return inputs


class TestReconciliationCalculation:
    def test_component_records_are_registered_per_window(self):
        pack = CALCULATION_PACKS["valuation_vs_history"]
        for years in (1, 3):
            names = [
                f"reconciliation_price_return_{years}y",
                f"reconciliation_eps_growth_{years}y",
                f"reconciliation_revenue_growth_{years}y",
                f"valuation_reconciliation_{years}y",
            ]
            positions = [pack.index(name) for name in names]
            assert positions == sorted(positions)  # components precede their headline
            for name in names:
                assert SPECS[name].unit == "percent" and name in CALCULATION_PACKS["all_standard"]
        assert SPECS["reconciliation_price_return_1y"].required_inputs == (
            "start_close",
            "end_close",
        )
        assert SPECS["reconciliation_eps_growth_1y"].required_inputs == (
            "eps_current",
            "eps_previous",
        )
        assert SPECS["reconciliation_revenue_growth_3y"].required_inputs == (
            "revenue_current",
            "revenue_previous",
        )

    def test_component_records_compute_from_the_same_operands(self):
        inputs = owner_inputs()
        price = compute("reconciliation_price_return_1y", inputs)
        eps = compute("reconciliation_eps_growth_1y", inputs)
        revenue = compute("reconciliation_revenue_growth_1y", inputs)
        headline = compute("valuation_reconciliation_1y", inputs)
        assert price.value == pytest.approx(35.0) and price.display == "35.0%"
        assert price.formula == "(close[close 2026-09-25] / close[close 2025-09-25] - 1) x 100"
        assert eps.value == pytest.approx(10.0)
        assert eps.formula == (
            "(eps_diluted[TTM to 2026-06-27] / eps_diluted[TTM to 2025-06-28] - 1) x 100"
        )
        assert revenue.value == pytest.approx(8.0)
        assert [i.name for i in eps.inputs] == ["eps_current", "eps_previous"]
        assert [i.source_id for i in eps.inputs] == ["x", "x"]
        # The headline is exactly the identity over the component records' values ...
        expected = ((1 + price.value / 100) / (1 + eps.value / 100) - 1) * 100
        assert headline.value == pytest.approx(expected)
        # ... and its verdict is the rule table applied to those values.
        v = rec.verdict(price.value / 100, eps.value / 100, headline.value / 100)
        assert headline.meta["reconciliation"]["verdict"] == v.label
        assert headline.meta["reconciliation"]["verdict_rule"] == v.rule
        missing = compute("reconciliation_eps_growth_1y", owner_inputs(eps_current=None))
        assert missing.status == "unavailable" and missing.missing_inputs == ["eps_current"]
        negative = compute(
            "reconciliation_eps_growth_1y", owner_inputs(eps_previous=_ci("eps_previous", -0.5))
        )
        assert negative.status == "unavailable" and negative.meta["reason"] == "not_meaningful"
        assert negative.notes == [
            "trailing P/E not meaningful: eps_previous = -0.5 (non-positive earnings)"
        ]
        zero_base = compute(
            "reconciliation_revenue_growth_1y",
            owner_inputs(revenue_previous=_ci("revenue_previous", 0.0)),
        )
        assert zero_base.notes == ["growth from a zero revenue base is undefined"]

    def test_bundle_view_carries_verdicts_as_recorded(self):
        computed = compute("valuation_reconciliation_1y", owner_inputs(), calc_id="c1")
        unavailable = compute("valuation_reconciliation_3y", {}, calc_id="c3")
        component = compute("reconciliation_eps_growth_1y", owner_inputs())
        view = rec.bundle_view([computed, unavailable, component])
        assert list(view) == ["valuation_reconciliation_1y"]
        entry = view["valuation_reconciliation_1y"]
        assert entry["verdict"] == "mostly multiple expansion"
        assert entry["verdict_rule"] == "g >= +0.5, m >= +0.5, |m| - |g| > 1.0"
        assert entry["multiple_change"] == "22.7%" and entry["calc_id"] == "c1"
        assert entry["price_return_pct"] == pytest.approx(35.0)
        assert entry["pe_percentile"] == 92.0
        assert entry["statement"] == computed.notes[-1]
        assert rec.bundle_view([unavailable]) == {}

    def test_specs_registered_in_the_valuation_pack(self):
        for name in ("valuation_reconciliation_1y", "valuation_reconciliation_3y"):
            assert name in SPECS and name in CALCULATION_PACKS["valuation_vs_history"]
            assert name in CALCULATION_PACKS["all_standard"]
            spec = SPECS[name]
            assert spec.required_inputs == (
                "start_close",
                "end_close",
                "eps_current",
                "eps_previous",
            )
            assert spec.optional_inputs == ("revenue_current", "revenue_previous", "pe_percentile")
            assert spec.unit == "percent" and spec.detail is not None
        assert "pe_history_percentile" in CALCULATION_PACKS["valuation_vs_history"]
        assert "pe_5y_percentile" in CALCULATION_PACKS["valuation_vs_history"]  # kept

    def test_owner_example_through_compute(self):
        result = compute("valuation_reconciliation_1y", owner_inputs(), calc_id="calc_r")
        assert result.status == "computed"
        assert result.value == pytest.approx(22.72727272727)
        assert result.display == "22.7%" and result.unit == "percent"
        assert result.formula == (
            "(1 + price_return) / (1 + eps_growth) - 1, x 100, with price_return = "
            "close[close 2026-09-25] / close[close 2025-09-25] - 1 and eps_growth = "
            "eps_diluted[TTM to 2026-06-27] / eps_diluted[TTM to 2025-06-28] - 1"
        )
        assert [i.name for i in result.inputs] == [
            "start_close",
            "end_close",
            "eps_current",
            "eps_previous",
            "revenue_current",
            "revenue_previous",
            "pe_percentile",
        ]
        record = result.meta["reconciliation"]
        assert record["price_return_pct"] == pytest.approx(35.0)
        assert record["eps_growth_pct"] == pytest.approx(10.0)
        assert record["multiple_change_pct"] == pytest.approx(22.72727272727)
        assert record["contributions_pct"]["multiple"] == pytest.approx(22.72727272727)
        assert record["contributions_pct"]["interaction"] == pytest.approx(2.2727272727)
        assert record["revenue_growth_pct"] == pytest.approx(8.0)
        assert record["pe_percentile"] == 92.0
        assert record["verdict"] == "mostly multiple expansion"
        assert record["verdict_rule"] == "g >= +0.5, m >= +0.5, |m| - |g| > 1.0"
        assert record["thresholds"] == {"negligible_points": 0.5, "equal_points": 1.0}
        assert record["component_calcs"] == {
            "price_return": "reconciliation_price_return_1y",
            "eps_growth": "reconciliation_eps_growth_1y",
            "revenue_growth": "reconciliation_revenue_growth_1y",
        }
        assert result.notes == [
            "mostly multiple expansion: of the +35.0% price change, +10.0 points came from "
            "EPS growth, +22.7 points from the change in the multiple and +2.3 points from "
            "their interaction; revenue grew 8.0%; trailing P/E sits at its 92nd percentile "
            "of its available history"
        ]
        assert result.event_view()["display"] == "22.7%"

    def test_reproducible(self):
        first = compute("valuation_reconciliation_1y", owner_inputs()).model_dump()
        second = compute("valuation_reconciliation_1y", owner_inputs()).model_dump()
        assert first == second

    def test_missing_eps_is_unavailable_with_named_operands(self):
        result = compute("valuation_reconciliation_1y", owner_inputs(eps_previous=None))
        assert result.status == "unavailable"
        assert result.missing_inputs == ["eps_previous"]
        assert result.value is None and result.display == "unavailable"
        assert result.meta["reason"] == "missing_inputs"
        assert "reconciliation" not in result.meta and result.notes == []
        closes_only = compute(
            "valuation_reconciliation_3y",
            {"start_close": _ci("start_close", 100.0), "end_close": _ci("end_close", 120.0)},
        )
        assert closes_only.missing_inputs == ["eps_current", "eps_previous"]
        nothing = compute("valuation_reconciliation_3y", {})
        assert nothing.missing_inputs == ["start_close", "end_close", "eps_current", "eps_previous"]

    def test_non_positive_eps_is_not_meaningful(self):
        result = compute(
            "valuation_reconciliation_1y",
            owner_inputs(eps_previous=_ci("eps_previous", -1.0)),
        )
        assert result.status == "unavailable" and result.missing_inputs == []
        assert result.meta["reason"] == "not_meaningful"
        assert result.notes == [
            "trailing P/E not meaningful: eps_previous = -1.0 (non-positive earnings)"
        ]
        assert "reconciliation" not in result.meta
        zero = compute(
            "valuation_reconciliation_1y", owner_inputs(eps_current=_ci("eps_current", 0.0))
        )
        assert zero.status == "unavailable"
        assert any("eps_current = 0.0" in note for note in zero.notes)

    def test_optional_context_never_changes_the_value(self):
        full = compute("valuation_reconciliation_1y", owner_inputs())
        bare = compute(
            "valuation_reconciliation_1y",
            owner_inputs(revenue_current=None, revenue_previous=None, pe_percentile=None),
        )
        assert bare.status == "computed" and bare.value == full.value
        record = bare.meta["reconciliation"]
        assert record["revenue_growth_pct"] is None and record["pe_percentile"] is None
        assert bare.notes[0].endswith(
            "revenue growth over the window is unavailable; the trailing P/E's historical "
            "percentile is unavailable"
        )
        assert [i.value for i in bare.inputs][-3:] == [None, None, None]  # recorded as absent

    def test_negative_growth_through_compute(self):
        result = compute(
            "valuation_reconciliation_1y",
            owner_inputs(
                end_close=_ci("end_close", 110.0),
                eps_current=_ci("eps_current", 0.80),
                revenue_current=_ci("revenue_current", 95.0),
            ),
        )
        assert result.value == pytest.approx(37.5)
        record = result.meta["reconciliation"]
        assert record["verdict"] == "re-rating despite earnings decline"
        assert record["contributions_pct"]["earnings"] == pytest.approx(-20.0)
        assert record["contributions_pct"]["interaction"] == pytest.approx(-7.5)
        assert result.notes[0].startswith(
            "re-rating despite earnings decline: of the +10.0% price change, -20.0 points "
            "came from EPS growth, +37.5 points from the change in the multiple and -7.5 "
            "points from their interaction; revenue declined 5.0%"
        )


# ======================================================================================
# run_pack: operand selection on synthetic evidence
# ======================================================================================


def long_eps_history(quarters: int, eps_start: float = 1.0, step: float = 0.02) -> list:
    """``quarters`` consecutive fiscal quarters of diluted EPS ending Q3 FY2026 (2026-06-27),
    walking back 91 days per quarter, labelled with fiscal year and quarter."""
    fps = ["Q1", "Q2", "Q3", "Q4"]
    end, fy, index = date(2026, 6, 27), 2026, 2
    periods: list[Period] = []
    for _ in range(quarters):
        period = Period(
            kind="fiscal_quarter",
            fiscal_year=fy,
            fiscal_period=fps[index],
            start=end - timedelta(days=90),
            end=end,
        )
        period.label = f"{fps[index]} FY{fy}"
        periods.append(period)
        end -= timedelta(days=91)
        index -= 1
        if index < 0:
            index, fy = 3, fy - 1
    periods.reverse()
    return [
        _fact("eps_diluted", eps_start + step * i, p, unit="USD_per_share")
        for i, p in enumerate(periods)
    ]


def _by_name(pack: str, evidence) -> dict:
    return {r.name: r for r in run_pack(pack, evidence, AS_OF)}


class TestReconciliationPack:
    def test_one_year_uses_ttm_pairs_and_records_the_lag(self):
        evidence = build_evidence()
        result = _by_name("valuation_vs_history", evidence)["valuation_reconciliation_1y"]
        assert result.status == "computed", result.notes
        inputs = {i.name: i for i in result.inputs}
        assert inputs["eps_current"].value == pytest.approx(7.1)
        assert inputs["eps_current"].period_label == "TTM to 2026-06-27"
        assert inputs["eps_previous"].value == pytest.approx(6.3)
        assert inputs["eps_previous"].period_label == "TTM to 2025-06-28"
        assert inputs["revenue_current"].period_label == "TTM to 2026-06-27"
        assert inputs["revenue_previous"].period_label == "TTM to 2025-06-28"
        assert inputs["start_close"].period_label == "close 2025-09-25"
        assert inputs["end_close"].period_label == "close 2026-09-25"
        assert inputs["pe_percentile"].value is None  # only 4 history points on this evidence
        assert result.period_label == (
            "2025-09-25 to 2026-09-25; EPS TTM to 2026-06-27 vs TTM to 2025-06-28"
        )
        assert result.meta["eps_basis"] == "ttm" and result.meta["revenue_basis"] == "ttm"
        assert result.meta["window"]["name"] == "1y" and result.meta["window"]["available"]
        assert result.meta["pe_percentile"]["available"] is False
        assert result.meta["pe_percentile"]["points"] == 4
        assert any(
            "price window ends 2026-09-25, EPS window ends 2026-06-27 (90 days earlier" in n
            for n in result.notes
        )
        # Every fact behind the TTM operands is recorded, so the value is reproducible.
        assert len(result.meta["operands"]["eps_current"]["fact_ids"]) == 4
        assert len(result.meta["operands"]["eps_previous"]["fact_ids"]) == 4
        expected = (inputs["end_close"].value / inputs["start_close"].value) / (7.1 / 6.3) - 1
        assert result.value == pytest.approx(expected * 100)
        record = result.meta["reconciliation"]
        assert record["verdict"] in rec.VERDICTS
        assert record["revenue_growth_pct"] == pytest.approx(
            (inputs["revenue_current"].value / inputs["revenue_previous"].value - 1) * 100
        )
        again = _by_name("valuation_vs_history", evidence)["valuation_reconciliation_1y"]
        assert again.model_dump() == result.model_dump()

    def test_component_records_on_evidence_match_the_headline(self):
        results = _by_name("valuation_vs_history", build_evidence())
        headline = results["valuation_reconciliation_1y"]
        price = results["reconciliation_price_return_1y"]
        eps = results["reconciliation_eps_growth_1y"]
        revenue = results["reconciliation_revenue_growth_1y"]
        for record in (price, eps, revenue):
            assert record.status == "computed", record.notes
            assert record.meta["reconciliation"] == "valuation_reconciliation_1y"
        # Same window as the standalone 1y price return, same operands as the headline.
        standalone = _by_name("returns_vs_benchmark", build_evidence())["price_return_1y"]
        assert price.value == pytest.approx(standalone.value)
        assert price.period_label == "2025-09-25 to 2026-09-25"
        assert eps.value == pytest.approx((7.1 / 6.3 - 1) * 100)
        assert eps.period_label == "TTM to 2026-06-27 vs TTM to 2025-06-28"
        assert eps.meta["basis"] == "ttm"
        assert len(eps.meta["operands"]["current"]["fact_ids"]) == 4
        assert revenue.period_label == "TTM to 2026-06-27 vs TTM to 2025-06-28"
        record = headline.meta["reconciliation"]
        assert record["price_return_pct"] == pytest.approx(price.value)
        assert record["eps_growth_pct"] == pytest.approx(eps.value)
        assert record["revenue_growth_pct"] == pytest.approx(revenue.value)
        assert headline.value == pytest.approx(
            ((1 + price.value / 100) / (1 + eps.value / 100) - 1) * 100
        )
        derived = rec.verdict(price.value / 100, eps.value / 100, headline.value / 100)
        assert (record["verdict"], record["verdict_rule"]) == (derived.label, derived.rule)

    def test_three_year_components_name_their_own_missing_operands(self):
        results = _by_name("valuation_vs_history", build_evidence())
        price = results["reconciliation_price_return_3y"]
        assert price.status == "unavailable"
        assert price.missing_inputs == ["start_close", "end_close"]
        assert price.notes == ["price series does not reach back 3 year(s)"]
        eps = results["reconciliation_eps_growth_3y"]
        assert eps.status == "computed"  # the fiscal-year pair exists even without 3y prices
        assert eps.period_label == "FY2025 vs FY2022" and eps.meta["basis"] == "fiscal_year"
        assert eps.value == pytest.approx((7.5 / 6.0 - 1) * 100)
        facts = [f for f in quarterly_facts() + annual_facts() if f.metric != "eps_diluted"]
        bare = _by_name("valuation_vs_history", build_evidence(facts=facts + balance_facts()))
        assert bare["reconciliation_eps_growth_1y"].missing_inputs == [
            "eps_current",
            "eps_previous",
        ]
        assert bare["reconciliation_revenue_growth_1y"].status == "computed"

    def test_three_years_unavailable_names_missing_operands(self):
        result = _by_name("valuation_vs_history", build_evidence())["valuation_reconciliation_3y"]
        assert result.status == "unavailable"
        assert result.missing_inputs == ["start_close", "end_close"]  # prices reach back ~1 year
        inputs = {i.name: i for i in result.inputs}
        assert inputs["eps_current"].period_label == "FY2025"  # fiscal-year fallback, labelled
        assert inputs["eps_previous"].period_label == "FY2022"
        assert result.meta["eps_basis"] == "fiscal_year"
        assert result.meta["window"] == {"available": False, "name": "3y"}
        assert "price series does not reach back 3 year(s)" in result.notes
        assert any("fiscal-year pair used (FY2025 vs FY2022)" in n for n in result.notes)
        assert "reconciliation" not in result.meta

    def test_fiscal_year_fallback_when_a_quarter_is_missing(self):
        # Dropping Q2 FY2025 breaks the earlier TTM window; the pair falls back to fiscal years
        # at both ends (never one TTM against one fiscal year).
        facts = quarterly_facts(skip_index=2) + annual_facts() + balance_facts()
        result = _by_name("valuation_vs_history", build_evidence(facts=facts))[
            "valuation_reconciliation_1y"
        ]
        assert result.status == "computed", result.notes
        inputs = {i.name: i for i in result.inputs}
        assert inputs["eps_current"].period_label == "FY2025"
        assert inputs["eps_previous"].period_label == "FY2024"
        assert result.meta["eps_basis"] == "fiscal_year"
        assert any(
            "trailing-twelve-month eps_diluted pair unavailable; fiscal-year pair used "
            "(FY2025 vs FY2024)" in n
            for n in result.notes
        )
        assert result.period_label == "2025-09-25 to 2026-09-25; EPS FY2025 vs FY2024"

    def test_no_eps_at_all_is_unavailable_with_both_named(self):
        facts = [f for f in quarterly_facts() + annual_facts() if f.metric != "eps_diluted"]
        result = _by_name("valuation_vs_history", build_evidence(facts=facts + balance_facts()))[
            "valuation_reconciliation_1y"
        ]
        assert result.status == "unavailable"
        assert result.missing_inputs == ["eps_current", "eps_previous"]
        assert "no trailing-twelve-month eps_diluted (four consecutive quarters missing)" in (
            result.notes
        )
        assert "no fiscal-year eps_diluted" in result.notes

    def test_long_history_attaches_the_percentile(self):
        facts = long_eps_history(12) + [
            f for f in quarterly_facts() + annual_facts() if f.metric != "eps_diluted"
        ]
        evidence = build_evidence(
            facts=facts + balance_facts(),
            prices=synthetic_prices("AAPL", 150.0, 0.0006, 1.0, points=800),
        )
        results = _by_name("valuation_vs_history", evidence)
        percentile = results["pe_history_percentile"]
        assert percentile.status == "computed", percentile.notes
        result = results["valuation_reconciliation_1y"]
        assert result.status == "computed", result.notes
        attached = next(i for i in result.inputs if i.name == "pe_percentile")
        assert attached.value == pytest.approx(percentile.value)
        assert attached.unit == "percentile"
        assert attached.period_label == percentile.meta["window"]["label"]
        assert result.meta["pe_percentile"]["available"] is True
        assert result.meta["reconciliation"]["pe_percentile"] == pytest.approx(percentile.value)
        assert result.notes[-1].endswith(
            f"trailing P/E sits at its {percentile.display} of its available history"
        )


# ======================================================================================
# historical P/E percentile: the window actually used
# ======================================================================================


class TestPercentileWindow:
    def test_short_history_is_unavailable_and_reports_its_window(self):
        results = _by_name("valuation_vs_history", build_evidence())
        for name, years in (("pe_5y_percentile", 5), ("pe_history_percentile", None)):
            result = results[name]
            assert result.status == "unavailable", name
            assert result.missing_inputs == ["pe_history"]
            assert result.period_label == (
                "close 2026-09-25; EPS TTM to 2026-06-27; "
                "over 4 available quarters, Q4 FY2025 to Q3 FY2026"
            )
            window = result.meta["window"]
            assert window["years"] == years and window["points"] == 4
            assert window["label"] == "over 4 available quarters, Q4 FY2025 to Q3 FY2026"
            assert window["first_quarter_end"] == "2025-09-27"
            assert window["last_quarter_end"] == "2026-06-27"
            assert (window["cutoff"] is None) is (years is None)
            assert f"P/E history has 4 points; at least {PE_HISTORY_MIN_POINTS} are required" in (
                result.notes
            )
            assert any("no close within 10 days before 2025-06-28" in n for n in result.notes)

    def test_long_history_reports_the_actual_window(self):
        facts = long_eps_history(12) + [
            f for f in quarterly_facts() + annual_facts() if f.metric != "eps_diluted"
        ]
        evidence = build_evidence(
            facts=facts + balance_facts(),
            prices=synthetic_prices("AAPL", 150.0, 0.0006, 1.0, points=800),
        )
        results = _by_name("valuation_vs_history", evidence)
        result = results["pe_history_percentile"]
        assert result.status == "computed", result.notes
        assert result.unit == "percentile" and 0.0 <= result.value <= 100.0
        history = result.meta["history"]
        assert len(history) == 9  # 12 quarters -> 9 trailing windows, all with a close
        assert result.period_label == (
            "close 2026-09-25; EPS TTM to 2026-06-27; "
            "over 9 available quarters, Q3 FY2024 to Q3 FY2026"
        )
        window = result.meta["window"]
        assert window == {
            "years": None,
            "points": 9,
            "label": "over 9 available quarters, Q3 FY2024 to Q3 FY2026",
            "first_quarter_end": history[0]["quarter_end"],
            "last_quarter_end": "2026-06-27",
            "cutoff": None,
        }
        series_input = next(i for i in result.inputs if i.name == "pe_history")
        assert series_input.value == 9.0 and series_input.unit == "observations"
        price = next(i.value for i in result.inputs if i.name == "price")
        eps = next(i.value for i in result.inputs if i.name == "eps_ttm")
        assert result.value == pytest.approx(
            prim.percentile_rank(price / eps, [h["pe"] for h in history])
        )
        # Within five years the capped calculation sees the same window and agrees exactly.
        capped = results["pe_5y_percentile"]
        assert capped.status == "computed" and capped.value == result.value
        assert capped.meta["window"]["years"] == 5 and capped.meta["window"]["points"] == 9
        assert capped.meta["window"]["cutoff"] == "2021-09-27"
        assert capped.period_label == result.period_label

    def test_history_beyond_five_years_widens_only_the_full_window(self):
        facts = long_eps_history(30) + [
            f for f in quarterly_facts() + annual_facts() if f.metric != "eps_diluted"
        ]
        evidence = build_evidence(
            facts=facts + balance_facts(),
            prices=synthetic_prices("AAPL", 60.0, 0.0006, 1.0, points=2000),
        )
        results = _by_name("valuation_vs_history", evidence)
        full, capped = results["pe_history_percentile"], results["pe_5y_percentile"]
        assert full.status == "computed" and capped.status == "computed"
        assert full.meta["window"]["points"] == 27  # 30 quarters -> 27 trailing windows
        assert capped.meta["window"]["points"] < full.meta["window"]["points"]
        assert capped.meta["window"]["first_quarter_end"] >= "2021-09-27"
        assert full.meta["window"]["first_quarter_end"] < "2021-09-27"
        assert full.period_label.endswith("over 27 available quarters, Q1 FY2020 to Q3 FY2026")
        assert capped.period_label != full.period_label
        # Different windows, different ranks: the record says which window each rank is in.
        resolver = OperandResolver(evidence, AS_OF)
        assert len(resolver.pe_history(None)[0]) == 27
        assert len(resolver.pe_history(5)[0]) == capped.meta["window"]["points"]

    def test_pe_history_window_label_uses_period_labels(self):
        facts = long_eps_history(8)
        evidence = build_evidence(
            facts=facts + balance_facts(),
            prices=synthetic_prices("AAPL", 150.0, 0.0006, 1.0, points=800),
        )
        result = _by_name("valuation_vs_history", evidence)["pe_history_percentile"]
        assert result.status == "unavailable"  # 8 quarters -> 5 trailing windows
        assert result.meta["window"]["label"] == "over 5 available quarters, Q3 FY2025 to Q3 FY2026"
        assert result.period_label.endswith("over 5 available quarters, Q3 FY2025 to Q3 FY2026")


def test_ttm_ending_near_locates_the_matching_window():
    evidence = build_evidence()
    resolver = OperandResolver(evidence, AS_OF)
    current = resolver.ttm("eps_diluted")
    assert current is not None and current.period_end == date(2026, 6, 27)
    previous = resolver.ttm_ending_near("eps_diluted", date(2025, 6, 27))
    assert previous is not None
    assert previous.period_label == "TTM to 2025-06-28" and previous.value == pytest.approx(6.3)
    assert resolver.ttm_ending_near("eps_diluted", date(2024, 6, 27)) is None  # before history
    assert resolver.ttm_ending_near("eps_diluted", date(2025, 8, 15)) is None  # no quarter near
    facts: list[NormalizedFact] = quarterly_facts(skip_index=1)  # gap inside the earlier window
    gapped = OperandResolver(build_evidence(facts=facts + annual_facts()), AS_OF)
    assert gapped.ttm_ending_near("eps_diluted", date(2025, 6, 27)) is None
