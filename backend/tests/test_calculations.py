"""Tests for the deterministic calculation layer: primitives, formatting, registry, packs."""

from __future__ import annotations

import math
from datetime import UTC, date, datetime, timedelta

import pytest

from bayanalytics.calculations import formatting
from bayanalytics.calculations import primitives as prim
from bayanalytics.calculations.operands import OperandResolver, shift_months
from bayanalytics.calculations.registry import (
    CALCULATION_PACKS,
    CANONICAL_METRICS,
    METRIC_UNITS,
    PE_HISTORY_MIN_POINTS,
    SPECS,
    compute,
    require,
    run_pack,
    run_packs,
    spec_for,
)
from bayanalytics.errors import AnalysisError
from bayanalytics.schemas.calculations import CalculationInput
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.evidence import (
    BenchmarkRef,
    NormalizedEvidence,
    NormalizedFact,
    Period,
    PricePoint,
    PriceSeries,
)

AS_OF = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)  # a Sunday; latest completed close is Fri 25th


# ======================================================================================
# primitives
# ======================================================================================


class TestGrowthAndMargins:
    def test_growth_rate_basic(self):
        assert prim.growth_rate(110.0, 100.0) == pytest.approx(0.10)
        assert prim.growth_rate(90.0, 100.0) == pytest.approx(-0.10)

    def test_growth_rate_negative_base_is_sign_correct(self):
        # loss shrinking from -100 to -50 is an improvement
        assert prim.growth_rate(-50.0, -100.0) == pytest.approx(0.5)
        # loss growing from -100 to -150 is a deterioration
        assert prim.growth_rate(-150.0, -100.0) == pytest.approx(-0.5)
        # swing from loss to profit is positive
        assert prim.growth_rate(50.0, -100.0) == pytest.approx(1.5)

    def test_growth_rate_invalid(self):
        assert prim.growth_rate(100.0, 0.0) is None
        assert prim.growth_rate(100.0, None) is None
        assert prim.growth_rate(None, 100.0) is None
        assert prim.growth_rate(float("nan"), 100.0) is None
        assert prim.growth_rate(True, 100.0) is None

    def test_cagr(self):
        assert prim.cagr(100.0, 121.0, 2) == pytest.approx(0.10)
        assert prim.cagr(100.0, 0.0, 2) == pytest.approx(-1.0)
        assert prim.cagr(0.0, 100.0, 2) is None
        assert prim.cagr(-100.0, 100.0, 2) is None
        assert prim.cagr(100.0, -1.0, 2) is None
        assert prim.cagr(100.0, 121.0, 0) is None
        assert prim.cagr(None, 121.0, 2) is None

    def test_margin(self):
        assert prim.margin(45.0, 100.0) == pytest.approx(0.45)
        assert prim.margin(-5.0, 100.0) == pytest.approx(-0.05)
        assert prim.margin(45.0, 0.0) is None
        assert prim.margin(45.0, -100.0) is None
        assert prim.margin(None, 100.0) is None

    def test_margin_change_bp(self):
        assert prim.margin_change_bp(0.302, 0.29) == pytest.approx(120.0)
        assert prim.margin_change_bp(0.25, 0.30) == pytest.approx(-500.0)
        assert prim.margin_change_bp(None, 0.3) is None

    def test_ttm_sum_requires_exactly_four(self):
        assert prim.ttm_sum([1.0, 2.0, 3.0, 4.0]) == 10.0
        assert prim.ttm_sum([1.0, 2.0, 3.0]) is None
        assert prim.ttm_sum([1.0, 2.0, 3.0, 4.0, 5.0]) is None
        assert prim.ttm_sum([1.0, None, 3.0, 4.0]) is None
        assert prim.ttm_sum(None) is None
        assert prim.ttm_sum([]) is None


class TestValuation:
    def test_price_to_earnings(self):
        assert prim.price_to_earnings(100.0, 4.0) == pytest.approx(25.0)
        assert prim.price_to_earnings(100.0, -1.0) is None  # negative earnings
        assert prim.price_to_earnings(100.0, 0.0) is None
        assert prim.price_to_earnings(None, 4.0) is None

    def test_price_to_sales(self):
        assert prim.price_to_sales(1_000.0, 250.0) == pytest.approx(4.0)
        assert prim.price_to_sales(1_000.0, 0.0) is None
        assert prim.price_to_sales(1_000.0, None) is None

    def test_enterprise_value_requires_all_operands(self):
        assert prim.enterprise_value(1_000.0, 200.0, 50.0) == pytest.approx(1_150.0)
        assert prim.enterprise_value(1_000.0, None, 50.0) is None
        assert prim.enterprise_value(1_000.0, 200.0, None) is None

    def test_ev_to_ebitda(self):
        assert prim.ev_to_ebitda(1_150.0, 100.0) == pytest.approx(11.5)
        assert prim.ev_to_ebitda(1_150.0, -100.0) is None
        assert prim.ev_to_ebitda(1_150.0, 0.0) is None

    def test_free_cash_flow_treats_capex_as_outflow_magnitude(self):
        assert prim.free_cash_flow(100.0, 30.0) == pytest.approx(70.0)
        assert prim.free_cash_flow(100.0, -30.0) == pytest.approx(70.0)
        assert prim.free_cash_flow(100.0, None) is None

    def test_fcf_yield(self):
        assert prim.fcf_yield(50.0, 1_000.0) == pytest.approx(0.05)
        assert prim.fcf_yield(50.0, 0.0) is None
        assert prim.fcf_yield(None, 1_000.0) is None


class TestPricesReturnsRisk:
    def test_period_return(self):
        assert prim.period_return([100.0, 110.0, 120.0]) == pytest.approx(0.20)
        assert prim.period_return([]) is None
        assert prim.period_return([100.0]) is None
        assert prim.period_return([0.0, 100.0]) is None
        assert prim.period_return([100.0, None]) is None

    def test_returns_series(self):
        assert prim.returns_series([100.0, 110.0, 99.0]) == pytest.approx([0.10, -0.10])
        assert prim.returns_series([]) == []
        assert prim.returns_series([100.0]) == []
        assert prim.returns_series([100.0, 0.0]) is None
        assert prim.returns_series(None) is None

    def test_relative_return_in_percentage_points(self):
        assert prim.relative_return(0.08, 0.04) == pytest.approx(4.0)
        assert prim.relative_return(0.08, 0.25) == pytest.approx(-17.0)
        assert prim.relative_return(None, 0.04) is None

    def test_annualized_volatility(self):
        assert prim.annualized_volatility([100.0, 100.0, 100.0, 100.0]) == pytest.approx(0.0)
        closes = [100.0, 101.0, 99.0, 102.0, 98.0]
        rets = prim.returns_series(closes)
        assert rets is not None
        import statistics

        expected = statistics.stdev(rets) * math.sqrt(252)
        assert prim.annualized_volatility(closes) == pytest.approx(expected)
        assert prim.annualized_volatility([100.0, 101.0]) is None
        assert prim.annualized_volatility([]) is None
        assert prim.annualized_volatility(closes, trading_days=0) is None

    def test_max_drawdown(self):
        closes = [100.0, 120.0, 90.0, 110.0, 80.0, 130.0]
        dd, peak, trough = prim.max_drawdown(closes)
        assert dd == pytest.approx(80.0 / 120.0 - 1.0)
        assert (peak, trough) == (1, 4)
        assert prim.max_drawdown([1.0, 2.0, 3.0]) == (0.0, 0, 0)
        assert prim.max_drawdown([]) is None
        assert prim.max_drawdown([100.0, 0.0]) is None

    def test_moving_average(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert prim.moving_average(values, 3) == pytest.approx(4.0)
        assert prim.moving_average(values, 5) == pytest.approx(3.0)
        assert prim.moving_average(values, 6) is None
        assert prim.moving_average(values, 0) is None
        assert prim.moving_average([1.0, None, 3.0], 3) is None

    def test_zscore(self):
        assert prim.zscore(4.0, [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(
            (4.0 - 5.0) / 2.138089935
        )
        assert prim.zscore(4.0, [5.0, 5.0, 5.0]) is None
        assert prim.zscore(4.0, [5.0]) is None
        assert prim.zscore(None, [1.0, 2.0]) is None

    def test_percentile_rank_handles_ties(self):
        history = [10.0, 20.0, 20.0, 30.0]
        assert prim.percentile_rank(20.0, history) == pytest.approx((1 + 0.5 * 2) / 4 * 100)
        assert prim.percentile_rank(5.0, history) == 0.0
        assert prim.percentile_rank(40.0, history) == 100.0
        assert prim.percentile_rank(7.0, [7.0, 7.0]) == 50.0
        assert prim.percentile_rank(7.0, []) is None
        assert prim.percentile_rank(None, history) is None

    def test_correlation(self):
        assert prim.correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
        assert prim.correlation([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
        assert prim.correlation([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) is None
        assert prim.correlation([1.0, 2.0], [1.0, 2.0, 3.0]) is None
        assert prim.correlation([1.0], [1.0]) is None

    def test_beta(self):
        bench = [0.01, -0.02, 0.015, 0.005, -0.01]
        asset = [2 * r for r in bench]
        assert prim.beta(asset, bench) == pytest.approx(2.0)
        assert prim.beta(asset, [0.0] * 5) is None
        assert prim.beta(asset[:3], bench) is None
        assert prim.beta(None, bench) is None

    def test_scenario_table(self):
        rows = prim.scenario_table(100.0, [-0.1, 0.0, 0.2])
        assert rows == [
            {"delta": -0.1, "value": pytest.approx(90.0), "change": pytest.approx(-10.0)},
            {"delta": 0.0, "value": 100.0, "change": 0.0},
            {"delta": 0.2, "value": pytest.approx(120.0), "change": pytest.approx(20.0)},
        ]
        assert prim.scenario_table(100.0, []) == []
        assert prim.scenario_table(None, [0.1]) is None
        assert prim.scenario_table(100.0, [0.1, None]) is None

    def test_helpers(self):
        assert prim.is_number(1) and prim.is_number(1.5)
        assert not prim.is_number(True)
        assert not prim.is_number(float("inf"))
        assert not prim.is_number("1")
        assert prim.safe_div(1.0, 4.0) == 0.25
        assert prim.safe_div(1.0, 0.0) is None


# ======================================================================================
# formatting
# ======================================================================================


class TestFormatting:
    @pytest.mark.parametrize(
        ("value", "unit", "expected"),
        [
            (42.8e9, "USD", "$42.8B"),
            (950_000.0, "USD", "$950.0K"),
            (12.34, "USD", "$12.34"),
            (-1.2e9, "USD", "-$1.2B"),
            (2.5e12, "USD", "$2.5T"),
            (999.96e6, "USD", "$1.0B"),
            (18.2, "percent", "18.2%"),
            (-7.4, "percent", "-7.4%"),
            (34.1, "ratio", "34.1\u00d7"),
            (6.13, "USD_per_share", "$6.13"),
            (-0.52, "USD_per_share", "-$0.52"),
            (15.3e9, "shares", "15.3B shares"),
            (42, "days", "42 days"),
            (120, "bp", "+120 bp"),
            (-35, "bp", "-35 bp"),
            (0, "bp", "0 bp"),
            (72, "percentile", "72nd percentile"),
            (11, "percentile", "11th percentile"),
            (3.4e6, "EUR", "€3.4M"),
            (None, "USD", "unavailable"),
            (float("nan"), "percent", "unavailable"),
        ],
    )
    def test_format_value(self, value, unit, expected):
        assert formatting.format_value(value, unit) == expected

    def test_decimals_override(self):
        assert formatting.format_value(1.23e6, "USD", decimals=2) == "$1.23M"
        assert formatting.format_value(1.23e6, "USD") == "$1.2M"

    def test_format_pair_shares_one_scale(self):
        assert formatting.format_pair(42.8e9, 950e6, "USD") == ("$42.8B", "$0.9B")
        assert formatting.format_pair(950e6, 42.8e9, "USD") == ("$0.9B", "$42.8B")
        assert formatting.format_pair(18.2, -7.4, "percent") == ("18.2%", "-7.4%")
        assert formatting.format_pair(None, 42.8e9, "USD") == ("unavailable", "$42.8B")

    def test_format_series(self):
        assert formatting.format_series([1.5e9, 2.0e9, 0.4e9], "shares") == [
            "1.5B shares",
            "2.0B shares",
            "0.4B shares",
        ]

    def test_format_change(self):
        assert formatting.format_change(4.0, "percent") == "+4.0%"
        assert formatting.format_change(-4.0, "percent") == "-4.0%"
        assert formatting.format_change(120, "bp") == "+120 bp"

    def test_unknown_unit_is_not_hidden(self):
        assert formatting.format_value(3.0, "widgets") == "3 widgets"


# ======================================================================================
# registry: compute / require / packs
# ======================================================================================


def _ci(name: str, value: float | None, **kw) -> CalculationInput:
    return CalculationInput(name=name, value=value, **kw)


class TestRegistryCompute:
    def test_canonical_metrics_and_packs(self):
        assert "revenue" in CANONICAL_METRICS and METRIC_UNITS["eps_diluted"] == "USD_per_share"
        assert set(CALCULATION_PACKS) == {
            "growth_and_margins",
            "valuation_vs_history",
            "returns_vs_benchmark",
            "volatility_and_drawdown",
            "all_standard",
        }
        union = [
            n
            for p in (
                "growth_and_margins",
                "valuation_vs_history",
                "returns_vs_benchmark",
                "volatility_and_drawdown",
            )
            for n in CALCULATION_PACKS[p]
        ]
        assert CALCULATION_PACKS["all_standard"] == union
        assert set(union) == set(SPECS)
        assert "pe_5y_percentile" in CALCULATION_PACKS["valuation_vs_history"]
        assert "drawdown_vs_market_1y" in CALCULATION_PACKS["returns_vs_benchmark"]
        with pytest.raises(KeyError):
            spec_for("nope")

    def test_compute_records_inputs_formula_and_percent_conversion(self):
        result = compute(
            "revenue_growth_yoy",
            {
                "revenue_current": _ci(
                    "revenue_current",
                    110.0,
                    unit="USD",
                    source_id="s1",
                    fact_id="f1",
                    period_label="Q3 FY2026",
                ),
                "revenue_previous": _ci(
                    "revenue_previous",
                    100.0,
                    unit="USD",
                    source_id="s2",
                    fact_id="f2",
                    period_label="Q3 FY2025",
                ),
            },
            calc_id="calc_1",
            period_label="Q3 FY2026 vs Q3 FY2025",
        )
        assert result.status == "computed"
        assert result.value == pytest.approx(10.0)
        assert result.unit == "percent"
        assert result.display == "10.0%"
        assert (
            result.formula
            == "(revenue[Q3 FY2026] - revenue[Q3 FY2025]) / |revenue[Q3 FY2025]| x 100"
        )
        assert [i.fact_id for i in result.inputs] == ["f1", "f2"]
        assert [i.source_id for i in result.inputs] == ["s1", "s2"]
        assert result.calc_id == "calc_1"
        assert result.event_view()["display"] == "10.0%"

    def test_compute_unavailable_lists_missing_inputs_and_never_infers(self):
        result = compute("pe_ttm", {"price": _ci("price", 100.0)})
        assert result.status == "unavailable"
        assert result.missing_inputs == ["eps_ttm"]
        assert result.value is None
        assert result.display == "unavailable"
        assert result.meta["reason"] == "missing_inputs"
        # a None-valued input counts as missing too
        result = compute("pe_ttm", {"price": _ci("price", 100.0), "eps_ttm": _ci("eps_ttm", None)})
        assert result.missing_inputs == ["eps_ttm"]
        result = compute("enterprise_value", {})
        assert result.missing_inputs == [
            "price",
            "shares_outstanding",
            "total_debt",
            "cash_and_equivalents",
        ]

    def test_compute_not_meaningful_is_unavailable_with_note(self):
        result = compute("pe_ttm", {"price": _ci("price", 100.0), "eps_ttm": _ci("eps_ttm", -2.0)})
        assert result.status == "unavailable"
        assert result.missing_inputs == []
        assert result.meta["reason"] == "not_meaningful"
        assert any("negative earnings" in note for note in result.notes)

    def test_require_raises_only_for_unavailable(self):
        ok = compute("pe_ttm", {"price": _ci("price", 100.0), "eps_ttm": _ci("eps_ttm", 4.0)})
        assert require(ok) is ok
        assert ok.display == "25.0\u00d7"
        bad = compute("pe_ttm", {"price": _ci("price", 100.0)})
        with pytest.raises(AnalysisError) as excinfo:
            require(bad)
        assert excinfo.value.code == ErrorCode.MISSING_CALCULATION_INPUT
        assert excinfo.value.details["missing_inputs"] == ["eps_ttm"]
        assert excinfo.value.details["calculation"] == "pe_ttm"

    def test_bp_and_usd_displays(self):
        result = compute(
            "operating_margin_change_bp",
            {
                "operating_income_current": _ci("operating_income_current", 30.2),
                "revenue_current": _ci("revenue_current", 100.0),
                "operating_income_previous": _ci("operating_income_previous", 29.0),
                "revenue_previous": _ci("revenue_previous", 100.0),
            },
        )
        assert result.value == pytest.approx(120.0)
        assert result.display == "+120 bp"
        fcf = compute(
            "free_cash_flow_ttm",
            {"operating_cash_flow_ttm": _ci("a", 120e9), "capex_ttm": _ci("b", 13.4e9)},
        )
        assert fcf.display == "$106.6B"


# ======================================================================================
# packs on hand-built evidence
# ======================================================================================

QUARTER_ENDS = [
    date(2024, 9, 28),
    date(2024, 12, 28),
    date(2025, 3, 29),
    date(2025, 6, 28),
    date(2025, 9, 27),
    date(2025, 12, 27),
    date(2026, 3, 28),
    date(2026, 6, 27),
]
QUARTER_FPS = ["Q4", "Q1", "Q2", "Q3", "Q4", "Q1", "Q2", "Q3"]
QUARTER_FYS = [2024, 2025, 2025, 2025, 2025, 2026, 2026, 2026]
BASE_REVENUE = 80e9
SHARES = 15.0e9


def _fact(metric: str, value: float, period: Period, unit: str = "USD", **kw) -> NormalizedFact:
    published = kw.pop(
        "published_at",
        datetime.combine(period.end + timedelta(days=30), datetime.min.time(), tzinfo=UTC),
    )
    return NormalizedFact(
        fact_id=f"f_{metric}_{period.end.isoformat()}",
        metric=metric,
        value=value,
        unit=unit,
        currency="USD",
        period=period,
        basis="gaap",
        source_id="src_xbrl",
        published_at=published,
        **kw,
    )


def quarterly_facts(skip_index: int | None = None) -> list[NormalizedFact]:
    facts: list[NormalizedFact] = []
    for i, (end, fp, fy) in enumerate(zip(QUARTER_ENDS, QUARTER_FPS, QUARTER_FYS, strict=True)):
        if i == skip_index:
            continue
        start = QUARTER_ENDS[i - 1] + timedelta(days=1) if i > 0 else end - timedelta(days=90)
        period = Period(
            kind="fiscal_quarter", fiscal_year=fy, fiscal_period=fp, start=start, end=end
        )
        period.label = f"{fp} FY{fy}"
        revenue = BASE_REVENUE * (1 + 0.02 * i)
        facts += [
            _fact("revenue", revenue, period),
            _fact("gross_profit", revenue * 0.45, period),
            _fact("operating_income", revenue * 0.30, period),
            _fact("net_income", revenue * 0.25, period),
            _fact("eps_diluted", 1.5 + 0.05 * i, period, unit="USD_per_share"),
            _fact("operating_cash_flow", revenue * 0.32, period),
            _fact("capex", revenue * 0.04, period),
            _fact("depreciation_amortization", revenue * 0.03, period),
        ]
    return facts


def annual_facts() -> list[NormalizedFact]:
    facts = []
    for fy, end, revenue in [
        (2022, date(2022, 9, 24), 300e9),
        (2023, date(2023, 9, 30), 310e9),
        (2024, date(2024, 9, 28), 330e9),
        (2025, date(2025, 9, 27), 360e9),
    ]:
        period = Period(
            kind="fiscal_year",
            fiscal_year=fy,
            fiscal_period="FY",
            start=end - timedelta(days=363),
            end=end,
        )
        period.label = f"FY{fy}"
        facts.append(_fact("revenue", revenue, period))
        facts.append(_fact("gross_profit", revenue * 0.44, period))
        facts.append(_fact("eps_diluted", 6.0 + 0.5 * (fy - 2022), period, unit="USD_per_share"))
    return facts


def balance_facts() -> list[NormalizedFact]:
    shares_period = Period(kind="instant", end=date(2026, 7, 17), label="as of 2026-07-17")
    bs_period = Period(kind="instant", end=date(2026, 6, 27), label="as of 2026-06-27")
    published = datetime(2026, 7, 31, tzinfo=UTC)
    return [
        _fact("shares_outstanding", SHARES, shares_period, unit="shares", published_at=published),
        _fact("total_debt", 100e9, bs_period, published_at=published),
        _fact("cash_and_equivalents", 60e9, bs_period, published_at=published),
    ]


def synthetic_prices(
    symbol: str,
    start_price: float,
    drift: float,
    seed: float,
    points: int = 300,
    end: date = date(2026, 9, 25),
) -> PriceSeries:
    """Deterministic weekday closes ending on ``end`` (inclusive), oldest first."""
    closes: list[PricePoint] = []
    d = end
    while len(closes) < points:
        if d.weekday() < 5:
            closes.append(PricePoint(date=d, close=0.0))
        d -= timedelta(days=1)
    closes.reverse()
    price = start_price
    for i, point in enumerate(closes):
        noise = math.sin(i * 0.7 + seed) * 0.01 + math.cos(i * 0.13 + seed) * 0.005
        price = price * (1 + drift + noise)
        point.close = round(price, 4)
        point.open = round(price * 0.999, 4)
        point.high = round(price * 1.01, 4)
        point.low = round(price * 0.99, 4)
        point.volume = 1_000_000.0
    return PriceSeries(
        symbol=symbol,
        source_id=f"src_px_{symbol}",
        points=closes,
        retrieved_at=AS_OF,
        exchange="NASDAQ",
    )


def build_evidence(
    *,
    facts: list[NormalizedFact] | None = None,
    prices: PriceSeries | None = None,
    with_benchmarks: bool = True,
    benchmark_key: str = "role",
) -> NormalizedEvidence:
    facts = quarterly_facts() + annual_facts() + balance_facts() if facts is None else facts
    prices = synthetic_prices("AAPL", 200.0, 0.0006, 1.0) if prices is None else prices
    benchmarks: dict[str, PriceSeries] = {}
    refs: list[BenchmarkRef] = []
    if with_benchmarks:
        spy = synthetic_prices("SPY", 500.0, 0.0004, 2.0)
        xlk = synthetic_prices("XLK", 200.0, 0.0008, 3.0)
        refs = [
            BenchmarkRef(role="broad_market", symbol="SPY", name="S&P 500 ETF"),
            BenchmarkRef(role="sector", symbol="XLK", name="Technology Select Sector"),
        ]
        if benchmark_key == "role":
            benchmarks = {"broad_market": spy, "sector": xlk}
        else:
            benchmarks = {"SPY": spy, "XLK": xlk}
    return NormalizedEvidence(
        symbol="AAPL",
        as_of=AS_OF,
        facts=facts,
        prices=prices,
        benchmarks=benchmarks,
        benchmark_refs=refs,
    )


@pytest.fixture(scope="module")
def evidence() -> NormalizedEvidence:
    return build_evidence()


@pytest.fixture(scope="module")
def all_results(evidence: NormalizedEvidence):
    return {r.name: r for r in run_pack("all_standard", evidence, AS_OF)}


class TestRunPack:
    def test_every_calculation_present_in_order(self, evidence):
        results = run_pack("all_standard", evidence, AS_OF)
        assert [r.name for r in results] == CALCULATION_PACKS["all_standard"]
        for result in results:
            assert result.meta["pack"] == "all_standard"
            assert result.meta["as_of"] == AS_OF.isoformat()
            assert result.calc_id == f"calc_{result.name}"
        with pytest.raises(KeyError):
            run_pack("nope", evidence, AS_OF)

    def test_reproducible(self, evidence):
        first = [r.model_dump() for r in run_pack("all_standard", evidence, AS_OF)]
        second = [r.model_dump() for r in run_pack("all_standard", evidence, AS_OF)]
        assert first == second
        assert first == [r.model_dump() for r in run_packs(["all_standard"], evidence, AS_OF)]

    def test_run_packs_dedupes(self, evidence):
        results = run_packs(["growth_and_margins", "all_standard"], evidence, AS_OF)
        assert [r.name for r in results] == CALCULATION_PACKS["all_standard"]

    def test_calc_id_factory(self, evidence):
        results = run_pack(
            "growth_and_margins", evidence, AS_OF, calc_id_factory=lambda n: f"an1:{n}"
        )
        assert results[0].calc_id == "an1:revenue_growth_yoy"

    def test_revenue_growth_yoy_uses_prior_year_quarter(self, all_results):
        result = all_results["revenue_growth_yoy"]
        assert result.status == "computed"
        current = BASE_REVENUE * (1 + 0.02 * 7)
        previous = BASE_REVENUE * (1 + 0.02 * 3)
        assert result.value == pytest.approx((current - previous) / previous * 100)
        assert result.period_label == "Q3 FY2026 vs Q3 FY2025"
        assert (
            result.formula
            == "(revenue[Q3 FY2026] - revenue[Q3 FY2025]) / |revenue[Q3 FY2025]| x 100"
        )
        assert [i.fact_id for i in result.inputs] == [
            "f_revenue_2026-06-27",
            "f_revenue_2025-06-28",
        ]

    def test_revenue_growth_qoq_and_cagr(self, all_results):
        qoq = all_results["revenue_growth_qoq"]
        assert qoq.period_label == "Q3 FY2026 vs Q2 FY2026"
        assert qoq.value == pytest.approx((1 + 0.14) / (1 + 0.12) * 100 - 100)
        cagr = all_results["revenue_cagr_3y"]
        assert cagr.period_label == "FY2022 to FY2025"
        assert cagr.value == pytest.approx(((360e9 / 300e9) ** (1 / 3) - 1) * 100)

    def test_margins_use_ttm_with_label(self, all_results):
        for name, expected in (
            ("gross_margin", 45.0),
            ("operating_margin", 30.0),
            ("net_margin", 25.0),
            ("fcf_margin", 28.0),
        ):
            result = all_results[name]
            assert result.status == "computed", name
            assert result.value == pytest.approx(expected)
            assert result.period_label == "TTM to 2026-06-27"
            assert result.meta["period_basis"] == "ttm"
            assert result.display == f"{expected:.1f}%"
        fcf = all_results["free_cash_flow_ttm"]
        ttm_revenue = sum(BASE_REVENUE * (1 + 0.02 * i) for i in range(4, 8))
        assert fcf.value == pytest.approx(ttm_revenue * 0.28)
        assert fcf.period_label == "TTM to 2026-06-27"
        assert all(i.fact_id is None for i in fcf.inputs)  # TTM operands carry fact_ids in meta
        assert len(fcf.meta["operands"]["operating_cash_flow"]["fact_ids"]) == 4

    def test_margin_change_and_eps_growth(self, all_results):
        assert all_results["operating_margin_change_bp"].value == pytest.approx(0.0)
        assert all_results["operating_margin_change_bp"].display == "0 bp"
        eps = all_results["eps_growth_yoy"]
        assert eps.value == pytest.approx(((1.5 + 0.35) - (1.5 + 0.15)) / (1.5 + 0.15) * 100)

    def test_market_cap_records_operand_dates_and_note(self, all_results, evidence):
        result = all_results["market_cap"]
        latest_close = evidence.prices.points[-1]
        assert latest_close.date == date(2026, 9, 25)
        assert result.value == pytest.approx(latest_close.close * SHARES)
        price_input = next(i for i in result.inputs if i.name == "price")
        shares_input = next(i for i in result.inputs if i.name == "shares_outstanding")
        assert price_input.period_label == "close 2026-09-25"
        assert price_input.source_id == "src_px_AAPL"
        assert shares_input.period_label == "as of 2026-07-17"
        assert shares_input.fact_id == "f_shares_outstanding_2026-07-17"
        assert any("2026-07-17" in n and "70 days earlier" in n for n in result.notes)
        assert result.display.startswith("$") and result.display.endswith("T")

    def test_valuation_multiples(self, all_results, evidence):
        close = evidence.prices.points[-1].close
        eps_ttm = sum(1.5 + 0.05 * i for i in range(4, 8))
        pe = all_results["pe_ttm"]
        assert pe.value == pytest.approx(close / eps_ttm)
        assert pe.unit == "ratio" and pe.display.endswith("\u00d7")
        ev = all_results["enterprise_value"]
        assert ev.value == pytest.approx(close * SHARES + 100e9 - 60e9)
        ttm_revenue = sum(BASE_REVENUE * (1 + 0.02 * i) for i in range(4, 8))
        assert all_results["ps_ttm"].value == pytest.approx(close * SHARES / ttm_revenue)
        assert all_results["ev_ebitda_ttm"].value == pytest.approx(ev.value / (ttm_revenue * 0.33))
        assert all_results["fcf_yield_ttm"].value == pytest.approx(
            ttm_revenue * 0.28 / (close * SHARES) * 100
        )

    def test_pe_percentile_unavailable_with_short_history(self, all_results):
        result = all_results["pe_5y_percentile"]
        assert result.status == "unavailable"
        assert result.missing_inputs == ["pe_history"]
        assert any(f"at least {PE_HISTORY_MIN_POINTS}" in n for n in result.notes)

    def test_pe_percentile_computed_with_long_history(self):
        # 12 quarters + prices back to 2023 -> 9 quarter-end trailing P/E points.
        extra_ends = [date(2023, 9, 30), date(2023, 12, 30), date(2024, 3, 30), date(2024, 6, 29)]
        extra = []
        for i, end in enumerate(extra_ends):
            start = extra_ends[i - 1] + timedelta(days=1) if i > 0 else end - timedelta(days=90)
            fy = 2023 if end.month == 9 else 2024
            fp = ["Q4", "Q1", "Q2", "Q3"][i]
            period = Period(
                kind="fiscal_quarter", fiscal_year=fy, fiscal_period=fp, start=start, end=end
            )
            period.label = f"{fp} FY{fy}"
            extra.append(_fact("eps_diluted", 1.2 + 0.05 * i, period, unit="USD_per_share"))
        facts = extra + quarterly_facts() + annual_facts() + balance_facts()
        prices = synthetic_prices("AAPL", 150.0, 0.0006, 1.0, points=800)
        evidence = build_evidence(facts=facts, prices=prices)
        result = next(
            r
            for r in run_pack("valuation_vs_history", evidence, AS_OF)
            if r.name == "pe_5y_percentile"
        )
        assert result.status == "computed", result.notes
        assert 0.0 <= result.value <= 100.0
        assert result.unit == "percentile"
        assert result.display.endswith("percentile")
        history = result.meta["history"]
        assert len(history) >= PE_HISTORY_MIN_POINTS
        assert history[0]["quarter_end"] == "2024-06-29"
        pes = [h["pe"] for h in history]
        current_pe = next(i.value for i in result.inputs if i.name == "price") / next(
            i.value for i in result.inputs if i.name == "eps_ttm"
        )
        assert result.value == pytest.approx(prim.percentile_rank(current_pe, pes))

    def test_price_returns_windows(self, all_results, evidence):
        resolver = OperandResolver(evidence, AS_OF)
        series = evidence.prices
        for name, months in (
            ("price_return_1m", 1),
            ("price_return_3m", 3),
            ("price_return_6m", 6),
            ("price_return_1y", 12),
        ):
            result = all_results[name]
            assert result.status == "computed", name
            start = resolver.close_on_or_before(series, shift_months(date(2026, 9, 25), -months))
            assert result.value == pytest.approx((series.points[-1].close / start.close - 1) * 100)
            assert result.meta["window"]["end"] == "2026-09-25"
            assert result.inputs[0].period_label == f"close {start.date.isoformat()}"
        ytd = all_results["price_return_ytd"]
        start = resolver.close_on_or_before(series, date(2025, 12, 31))
        assert ytd.value == pytest.approx((series.points[-1].close / start.close - 1) * 100)
        assert ytd.meta["window"]["requested_start"] == "2025-12-31"

    def test_relative_returns_record_benchmark(self, all_results, evidence):
        market = all_results["relative_return_1y_vs_market"]
        sector = all_results["relative_return_1y_vs_sector"]
        assert market.status == "computed" and sector.status == "computed"
        assert market.meta["benchmark"] == {
            "role": "broad_market",
            "symbol": "SPY",
            "name": "S&P 500 ETF",
            "reason": "",
        }
        assert sector.meta["benchmark"]["symbol"] == "XLK"
        asset = all_results["price_return_1y"].value
        spy = evidence.benchmarks["broad_market"]
        resolver = OperandResolver(evidence, AS_OF)
        start = resolver.close_on_or_before(spy, shift_months(date(2026, 9, 25), -12))
        bench_return = (spy.points[-1].close / start.close - 1) * 100
        assert market.value == pytest.approx(asset - bench_return)

    def test_beta_and_drawdown_vs_market(self, all_results, evidence):
        beta = all_results["beta_1y_vs_market"]
        assert beta.status == "computed"
        assert beta.unit == "ratio"
        assert any("common sessions" in n for n in beta.notes)
        assert beta.meta["window"]["points"] >= 200
        dd = all_results["drawdown_vs_market_1y"]
        assert dd.status == "computed"
        asset_dd = all_results["max_drawdown_1y"].value
        spy_window = OperandResolver(evidence, AS_OF).window(
            "b", evidence.benchmarks["broad_market"], months=12
        )
        spy_dd = prim.max_drawdown(list(spy_window.closes))[0] * 100
        assert dd.value == pytest.approx(asset_dd - spy_dd)

    def test_volatility_and_moving_averages(self, all_results, evidence):
        closes = [p.close for p in evidence.prices.points]
        vol30 = all_results["volatility_30d_annualized"]
        assert vol30.value == pytest.approx(prim.annualized_volatility(closes[-31:]) * 100)
        assert vol30.meta["window"]["points"] == 31
        vol1y = all_results["volatility_1y_annualized"]
        assert vol1y.status == "computed"
        assert vol1y.meta["window"]["points"] >= 200
        ma50 = all_results["ma_50"]
        assert ma50.value == pytest.approx(prim.moving_average(closes, 50))
        assert ma50.unit == "USD_per_share" and ma50.display.startswith("$")
        ma200 = all_results["ma_200"]
        assert ma200.value == pytest.approx(prim.moving_average(closes, 200))
        pvm = all_results["price_vs_ma_200_pct"]
        assert pvm.value == pytest.approx((closes[-1] / ma200.value - 1) * 100)
        mdd = all_results["max_drawdown_1y"]
        assert mdd.value <= 0
        assert "peak_date" in mdd.meta and "trough_date" in mdd.meta

    def test_latest_close_never_uses_partial_session(self):
        # as_of Friday 14:00 New York (18:00 UTC): Friday's session is still open.
        evidence = build_evidence()
        friday_intraday = datetime(2026, 9, 25, 18, 0, tzinfo=UTC)
        result = next(
            r
            for r in run_pack("valuation_vs_history", evidence, friday_intraday)
            if r.name == "market_cap"
        )
        price_input = next(i for i in result.inputs if i.name == "price")
        assert price_input.period_label == "close 2026-09-24"

    def test_prices_after_as_of_are_ignored(self):
        evidence = build_evidence()
        earlier = datetime(2026, 8, 15, 12, 0, tzinfo=UTC)  # a Saturday
        results = {r.name: r for r in run_pack("returns_vs_benchmark", evidence, earlier)}
        assert results["price_return_1y"].meta["window"]["end"] == "2026-08-14"


class TestRunPackDegradation:
    def test_ttm_requires_four_consecutive_quarters(self):
        # Drop Q1 FY2026 (index 5): the latest four quarters are no longer consecutive.
        evidence = build_evidence(
            facts=quarterly_facts(skip_index=5) + annual_facts() + balance_facts()
        )
        results = {r.name: r for r in run_pack("growth_and_margins", evidence, AS_OF)}
        gross = results["gross_margin"]
        assert gross.status == "computed"
        assert gross.period_label == "FY2025"  # fiscal-year fallback, labelled
        assert gross.meta["period_basis"] == "fiscal_year"
        assert gross.value == pytest.approx(44.0)
        assert any("four consecutive fiscal quarters unavailable" in n for n in gross.notes)
        # no fiscal-year operating income exists -> unavailable, never a 3-quarter sum
        op = results["operating_margin"]
        assert op.status == "unavailable"
        assert set(op.missing_inputs) == {"operating_income", "revenue"}
        assert op.value is None

    def test_ttm_never_mixes_quarters_and_years(self):
        evidence = build_evidence(
            facts=quarterly_facts()[: 8 * 3] + annual_facts() + balance_facts()
        )
        results = {r.name: r for r in run_pack("valuation_vs_history", evidence, AS_OF)}
        pe = results["pe_ttm"]
        assert pe.status == "computed"
        eps_input = next(i for i in pe.inputs if i.name == "eps_ttm")
        assert eps_input.period_label == "FY2025"
        assert eps_input.fact_id == "f_eps_diluted_2025-09-27"

    def test_missing_benchmark_is_unavailable_not_guessed(self):
        evidence = build_evidence(with_benchmarks=False)
        results = {r.name: r for r in run_pack("returns_vs_benchmark", evidence, AS_OF)}
        rel = results["relative_return_1y_vs_market"]
        assert rel.status == "unavailable"
        assert rel.missing_inputs == ["benchmark_start_close", "benchmark_end_close"]
        assert rel.meta["benchmark"]["symbol"] is None
        assert results["beta_1y_vs_market"].missing_inputs == ["benchmark_closes"]
        assert results["price_return_1y"].status == "computed"

    def test_benchmarks_keyed_by_symbol_are_found(self):
        evidence = build_evidence(benchmark_key="symbol")
        results = {r.name: r for r in run_pack("returns_vs_benchmark", evidence, AS_OF)}
        assert results["relative_return_1y_vs_sector"].status == "computed"
        assert results["relative_return_1y_vs_sector"].meta["benchmark"]["symbol"] == "XLK"

    def test_no_prices_makes_price_calculations_unavailable(self):
        evidence = build_evidence(prices=None, with_benchmarks=False)
        evidence = evidence.model_copy(update={"prices": None})
        results = run_pack("all_standard", evidence, AS_OF)
        by_name = {r.name: r for r in results}
        assert by_name["market_cap"].missing_inputs == ["price"]
        assert by_name["ma_200"].missing_inputs == ["closes"]
        assert by_name["gross_margin"].status == "computed"
        # every result still carries the pack meta and a display
        assert all(r.display for r in results)

    def test_short_price_history_is_unavailable(self):
        evidence = build_evidence(prices=synthetic_prices("AAPL", 200.0, 0.0006, 1.0, points=40))
        results = {r.name: r for r in run_pack("volatility_and_drawdown", evidence, AS_OF)}
        assert results["volatility_30d_annualized"].status == "computed"
        assert results["ma_50"].status == "unavailable"
        assert results["ma_200"].status == "unavailable"
        assert results["volatility_1y_annualized"].status == "unavailable"
        assert results["max_drawdown_1y"].status == "unavailable"

    def test_empty_evidence(self):
        evidence = NormalizedEvidence(symbol="X", as_of=AS_OF)
        results = run_pack("all_standard", evidence, AS_OF)
        assert all(r.status == "unavailable" for r in results)
        assert all(r.display == "unavailable" for r in results)
        assert all(r.missing_inputs for r in results)


class TestOperandResolver:
    def test_shift_months_clamps_day(self):
        assert shift_months(date(2026, 3, 31), -1) == date(2026, 2, 28)
        assert shift_months(date(2026, 1, 15), -12) == date(2025, 1, 15)
        assert shift_months(date(2024, 2, 29), 12) == date(2025, 2, 28)

    def test_gaap_preferred_over_adjusted_with_note(self):
        facts = quarterly_facts()
        period = facts[-8].period  # latest quarter's revenue period
        adjusted = NormalizedFact(
            fact_id="f_adj",
            metric="revenue",
            value=1.0,
            unit="USD",
            currency="USD",
            period=period,
            basis="adjusted",
            source_id="src_pr",
        )
        resolver = OperandResolver(build_evidence(facts=[*facts, adjusted]), AS_OF)
        operand = resolver.latest_fq("revenue")
        assert operand.fact_id == "f_revenue_2026-06-27"
        assert any("adjusted value 1.0" in n for n in operand.notes)

    def test_facts_after_as_of_are_invisible(self):
        resolver = OperandResolver(build_evidence(), datetime(2026, 1, 15, tzinfo=UTC))
        assert resolver.latest_fq("revenue").period_label == "Q4 FY2025"
        assert resolver.ttm("revenue").period_label == "TTM to 2025-09-27"

    def test_ttm_and_prior_quarter_lookups(self):
        resolver = OperandResolver(build_evidence(), AS_OF)
        ttm = resolver.ttm("revenue")
        assert ttm.period_kind == "ttm" and len(ttm.fact_ids) == 4
        latest = resolver.latest_fq("revenue")
        assert resolver.prior_year_quarter("revenue", latest).period_label == "Q3 FY2025"
        assert resolver.previous_quarter("revenue", latest).period_label == "Q2 FY2026"
        assert resolver.latest_balance("shares_outstanding").period_label == "as of 2026-07-17"
        assert (
            resolver.fy_years_before("revenue", resolver.latest_fy("revenue"), 3).period_label
            == "FY2022"
        )
        assert resolver.ttm("total_debt") is None
