"""Reproducible calculation records (AGENT.md sections 3.3, 12, 25).

Canonical metric names used across the backend (``NormalizedFact.metric``)::

    revenue                     USD     top-line revenue for the period
    gross_profit                USD
    operating_income            USD
    net_income                  USD
    eps_diluted                 USD_per_share
    eps_basic                   USD_per_share
    operating_cash_flow         USD
    capex                       USD     capital expenditure (positive outflow magnitude)
    free_cash_flow              USD     derived: operating_cash_flow - |capex|
    shares_outstanding          shares  instant (cover-page or balance-sheet date)
    cash_and_equivalents        USD     instant
    total_debt                  USD     instant
    stockholders_equity         USD     instant
    total_assets                USD     instant
    research_and_development    USD
    depreciation_amortization   USD
    ebitda                      USD     derived: operating_income + depreciation_amortization

Units: ``USD`` (or another 3-letter currency code), ``USD_per_share``, ``shares``,
``percent`` (a value of 18.2 means 18.2 %), ``ratio`` (a multiple); calculations may also
produce ``bp`` (basis points), ``days`` and ``percentile`` (a 0-100 rank).

Every ``CalculationResult`` records the formula with the operands' period labels filled in,
each operand with its fact / source / period, the full-precision value, the unit and the
display string. When a required operand is missing the result is ``unavailable`` with the
missing operands listed: nothing is inferred, estimated or defaulted. ``compute`` never
raises for missing inputs; ``require`` turns an unavailable result into
``AnalysisError(MISSING_CALCULATION_INPUT)`` for callers that need to fail loudly.

Values in ``percent`` units are stored as percent numbers (``18.2``); the primitives return
fractions and the conversion (``x 100``) is part of the recorded formula.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from bayanalytics.calculations import primitives as prim
from bayanalytics.calculations.formatting import UNAVAILABLE, format_value
from bayanalytics.calculations.operands import (
    Aligned,
    Operand,
    OperandResolver,
    SeriesOperand,
)
from bayanalytics.errors import AnalysisError
from bayanalytics.schemas.calculations import CalculationInput, CalculationResult
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.evidence import NormalizedEvidence

METRIC_UNITS: dict[str, str] = {
    "revenue": "USD",
    "gross_profit": "USD",
    "operating_income": "USD",
    "net_income": "USD",
    "eps_diluted": "USD_per_share",
    "eps_basic": "USD_per_share",
    "operating_cash_flow": "USD",
    "capex": "USD",
    "free_cash_flow": "USD",
    "shares_outstanding": "shares",
    "cash_and_equivalents": "USD",
    "total_debt": "USD",
    "stockholders_equity": "USD",
    "total_assets": "USD",
    "research_and_development": "USD",
    "depreciation_amortization": "USD",
    "ebitda": "USD",
}
CANONICAL_METRICS: tuple[str, ...] = tuple(METRIC_UNITS)
UNITS: tuple[str, ...] = ("USD", "USD_per_share", "shares", "percent", "ratio")

# Window rules (documented here, applied in run_pack).
VOLATILITY_30D_RETURNS = 30  # 30 daily returns = 31 closes
VOLATILITY_1Y_MIN_OBSERVATIONS = 200  # of ~252 sessions in a calendar year
BETA_MIN_OBSERVATIONS = 60
PE_HISTORY_MIN_POINTS = 8
PE_HISTORY_YEARS = 5

InputValue = CalculationInput | SeriesOperand | None


@dataclass(frozen=True)
class CalculationSpec:
    """What a named calculation is: its formula, operands, unit and the primitive it runs."""

    name: str
    formula: str
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    unit: str
    description: str
    fn: Callable[..., Any]
    percent_from_fraction: bool = False
    explain_none: Callable[[dict[str, Any]], str | None] | None = None


class _LabelMap(dict):
    """format_map helper: unknown placeholders render as their own name."""

    def __missing__(self, key: str) -> str:
        return key


def _pct(value: float | None) -> float | None:
    return None if value is None else value * 100.0


def _closes(series: SeriesOperand | None) -> list[float] | None:
    return None if series is None else list(series.closes)


def _drawdown_fraction(series: SeriesOperand | None) -> float | None:
    result = prim.max_drawdown(_closes(series))
    return None if result is None else result[0]


def _beta_from_closes(asset_closes: SeriesOperand, benchmark_closes: SeriesOperand) -> float | None:
    asset_returns = prim.returns_series(_closes(asset_closes))
    bench_returns = prim.returns_series(_closes(benchmark_closes))
    return prim.beta(asset_returns, bench_returns)


def _not_positive(name: str, label: str) -> Callable[[dict[str, Any]], str | None]:
    def explain(values: dict[str, Any]) -> str | None:
        value = values.get(name)
        if isinstance(value, int | float) and value <= 0:
            return f"{label}: multiple not meaningful ({name} = {value!r})"
        return None

    return explain


def _explain_growth(values: dict[str, Any]) -> str | None:
    if values.get("previous") == 0 or values.get("revenue_previous") == 0:
        return "growth from a zero base is undefined"
    return None


SPECS: dict[str, CalculationSpec] = {}


def _spec(spec: CalculationSpec) -> CalculationSpec:
    SPECS[spec.name] = spec
    return spec


# ------------------------------------------------------------------ growth and margins ----

_spec(
    CalculationSpec(
        name="revenue_growth_yoy",
        formula="(revenue[{revenue_current}] - revenue[{revenue_previous}]) "
        "/ |revenue[{revenue_previous}]| x 100",
        required_inputs=("revenue_current", "revenue_previous"),
        optional_inputs=(),
        unit="percent",
        description="Year-over-year revenue growth: latest fiscal quarter vs the same quarter a "
        "year earlier (falls back to latest fiscal year vs prior fiscal year, labelled).",
        fn=lambda revenue_current, revenue_previous: prim.growth_rate(
            revenue_current, revenue_previous
        ),
        percent_from_fraction=True,
        explain_none=_explain_growth,
    )
)
_spec(
    CalculationSpec(
        name="revenue_growth_qoq",
        formula="(revenue[{revenue_current}] - revenue[{revenue_previous}]) "
        "/ |revenue[{revenue_previous}]| x 100",
        required_inputs=("revenue_current", "revenue_previous"),
        optional_inputs=(),
        unit="percent",
        description="Sequential revenue growth: latest fiscal quarter vs the quarter before it.",
        fn=lambda revenue_current, revenue_previous: prim.growth_rate(
            revenue_current, revenue_previous
        ),
        percent_from_fraction=True,
        explain_none=_explain_growth,
    )
)
_spec(
    CalculationSpec(
        name="revenue_cagr_3y",
        formula="(revenue[{revenue_end}] / revenue[{revenue_begin}]) ^ (1/3) - 1, x 100",
        required_inputs=("revenue_begin", "revenue_end"),
        optional_inputs=(),
        unit="percent",
        description="Three-year compound annual revenue growth between fiscal years.",
        fn=lambda revenue_begin, revenue_end: prim.cagr(revenue_begin, revenue_end, 3),
        percent_from_fraction=True,
        explain_none=lambda v: (
            "CAGR needs a positive starting revenue"
            if isinstance(v.get("revenue_begin"), int | float) and v["revenue_begin"] <= 0
            else None
        ),
    )
)
for _margin_name, _numerator in (
    ("gross_margin", "gross_profit"),
    ("operating_margin", "operating_income"),
    ("net_margin", "net_income"),
):
    _spec(
        CalculationSpec(
            name=_margin_name,
            formula=f"{_numerator}[{{{_numerator}}}] / revenue[{{revenue}}] x 100",
            required_inputs=(_numerator, "revenue"),
            optional_inputs=(),
            unit="percent",
            description=f"{_numerator.replace('_', ' ')} as a percentage of revenue over the same "
            "period (TTM when four consecutive quarters exist, else the latest fiscal year).",
            fn=(lambda num: lambda **kw: prim.margin(kw[num], kw["revenue"]))(_numerator),
            percent_from_fraction=True,
            explain_none=lambda v: "revenue not positive" if v.get("revenue", 1) <= 0 else None,
        )
    )
_spec(
    CalculationSpec(
        name="fcf_margin",
        formula="(operating_cash_flow[{operating_cash_flow}] - |capex[{capex}]|) "
        "/ revenue[{revenue}] x 100",
        required_inputs=("operating_cash_flow", "capex", "revenue"),
        optional_inputs=(),
        unit="percent",
        description="Free cash flow (OCF - capex) as a percentage of revenue over the same period.",
        fn=lambda operating_cash_flow, capex, revenue: prim.margin(
            prim.free_cash_flow(operating_cash_flow, capex), revenue
        ),
        percent_from_fraction=True,
        explain_none=lambda v: "revenue not positive" if v.get("revenue", 1) <= 0 else None,
    )
)
_spec(
    CalculationSpec(
        name="operating_margin_change_bp",
        formula="(operating_income[{operating_income_current}] / revenue[{revenue_current}] - "
        "operating_income[{operating_income_previous}] / revenue[{revenue_previous}]) x 10000",
        required_inputs=(
            "operating_income_current",
            "revenue_current",
            "operating_income_previous",
            "revenue_previous",
        ),
        optional_inputs=(),
        unit="bp",
        description="Operating margin expansion (+) or contraction (-) in basis points: latest "
        "fiscal quarter vs the same quarter a year earlier.",
        fn=lambda **kw: prim.margin_change_bp(
            prim.margin(kw["operating_income_current"], kw["revenue_current"]),
            prim.margin(kw["operating_income_previous"], kw["revenue_previous"]),
        ),
    )
)
_spec(
    CalculationSpec(
        name="free_cash_flow_ttm",
        formula="operating_cash_flow[{operating_cash_flow_ttm}] - |capex[{capex_ttm}]|",
        required_inputs=("operating_cash_flow_ttm", "capex_ttm"),
        optional_inputs=(),
        unit="USD",
        description="Trailing-twelve-month free cash flow from the latest four consecutive "
        "quarters (fiscal year when quarters are incomplete, labelled).",
        fn=lambda operating_cash_flow_ttm, capex_ttm: prim.free_cash_flow(
            operating_cash_flow_ttm, capex_ttm
        ),
    )
)
_spec(
    CalculationSpec(
        name="eps_growth_yoy",
        formula="(eps_diluted[{eps_current}] - eps_diluted[{eps_previous}]) "
        "/ |eps_diluted[{eps_previous}]| x 100",
        required_inputs=("eps_current", "eps_previous"),
        optional_inputs=(),
        unit="percent",
        description="Year-over-year diluted EPS growth (|previous| denominator keeps the sign "
        "meaningful when the base is a loss).",
        fn=lambda eps_current, eps_previous: prim.growth_rate(eps_current, eps_previous),
        percent_from_fraction=True,
        explain_none=lambda v: (
            "growth from zero EPS is undefined" if v.get("eps_previous") == 0 else None
        ),
    )
)

# ------------------------------------------------------------------ valuation --------------

_spec(
    CalculationSpec(
        name="market_cap",
        formula="price[{price}] x shares_outstanding[{shares_outstanding}]",
        required_inputs=("price", "shares_outstanding"),
        optional_inputs=(),
        unit="USD",
        description="Latest completed close times the latest reported shares outstanding.",
        fn=lambda price, shares_outstanding: price * shares_outstanding,
    )
)
_spec(
    CalculationSpec(
        name="enterprise_value",
        formula="price[{price}] x shares_outstanding[{shares_outstanding}] "
        "+ total_debt[{total_debt}] - cash_and_equivalents[{cash_and_equivalents}]",
        required_inputs=("price", "shares_outstanding", "total_debt", "cash_and_equivalents"),
        optional_inputs=(),
        unit="USD",
        description="Market cap plus total debt minus cash and equivalents (latest balances).",
        fn=lambda price, shares_outstanding, total_debt, cash_and_equivalents: (
            prim.enterprise_value(price * shares_outstanding, total_debt, cash_and_equivalents)
        ),
    )
)
_spec(
    CalculationSpec(
        name="pe_ttm",
        formula="price[{price}] / eps_diluted[{eps_ttm}]",
        required_inputs=("price", "eps_ttm"),
        optional_inputs=(),
        unit="ratio",
        description="Trailing P/E: latest close over the sum of the latest four consecutive "
        "quarterly diluted EPS (an approximation when the share count changed within the year).",
        fn=lambda price, eps_ttm: prim.price_to_earnings(price, eps_ttm),
        explain_none=_not_positive("eps_ttm", "negative earnings"),
    )
)
_spec(
    CalculationSpec(
        name="ps_ttm",
        formula="price[{price}] x shares_outstanding[{shares_outstanding}] "
        "/ revenue[{revenue_ttm}]",
        required_inputs=("price", "shares_outstanding", "revenue_ttm"),
        optional_inputs=(),
        unit="ratio",
        description="Price-to-sales: market cap over trailing-twelve-month revenue.",
        fn=lambda price, shares_outstanding, revenue_ttm: prim.price_to_sales(
            price * shares_outstanding, revenue_ttm
        ),
        explain_none=_not_positive("revenue_ttm", "revenue not positive"),
    )
)
_spec(
    CalculationSpec(
        name="ev_ebitda_ttm",
        formula="(price[{price}] x shares_outstanding[{shares_outstanding}] "
        "+ total_debt[{total_debt}] - cash_and_equivalents[{cash_and_equivalents}]) "
        "/ (operating_income[{operating_income_ttm}] "
        "+ depreciation_amortization[{depreciation_amortization_ttm}])",
        required_inputs=(
            "price",
            "shares_outstanding",
            "total_debt",
            "cash_and_equivalents",
            "operating_income_ttm",
            "depreciation_amortization_ttm",
        ),
        optional_inputs=(),
        unit="ratio",
        description="Enterprise value over trailing EBITDA (operating income + D&A).",
        fn=lambda **kw: prim.ev_to_ebitda(
            prim.enterprise_value(
                kw["price"] * kw["shares_outstanding"],
                kw["total_debt"],
                kw["cash_and_equivalents"],
            ),
            kw["operating_income_ttm"] + kw["depreciation_amortization_ttm"],
        ),
        explain_none=lambda v: (
            "negative EBITDA: multiple not meaningful"
            if all(
                isinstance(v.get(k), int | float)
                for k in ("operating_income_ttm", "depreciation_amortization_ttm")
            )
            and v["operating_income_ttm"] + v["depreciation_amortization_ttm"] <= 0
            else None
        ),
    )
)
_spec(
    CalculationSpec(
        name="fcf_yield_ttm",
        formula="(operating_cash_flow[{operating_cash_flow_ttm}] - |capex[{capex_ttm}]|) / "
        "(price[{price}] x shares_outstanding[{shares_outstanding}]) x 100",
        required_inputs=("operating_cash_flow_ttm", "capex_ttm", "price", "shares_outstanding"),
        optional_inputs=(),
        unit="percent",
        description="Trailing free cash flow as a percentage of market cap.",
        fn=lambda operating_cash_flow_ttm, capex_ttm, price, shares_outstanding: prim.fcf_yield(
            prim.free_cash_flow(operating_cash_flow_ttm, capex_ttm), price * shares_outstanding
        ),
        percent_from_fraction=True,
    )
)
_spec(
    CalculationSpec(
        name="pe_5y_percentile",
        formula="percentile_rank(price[{price}] / eps_diluted[{eps_ttm}], trailing P/E at each "
        "fiscal-quarter end over the last 5 years)",
        required_inputs=("price", "eps_ttm", "pe_history"),
        optional_inputs=(),
        unit="percentile",
        description="Where today's trailing P/E sits within its own five-year history of "
        "quarter-end trailing P/Es (mid-rank percentile, 0-100). Unavailable with fewer than "
        f"{PE_HISTORY_MIN_POINTS} history points.",
        fn=lambda price, eps_ttm, pe_history: prim.percentile_rank(
            prim.price_to_earnings(price, eps_ttm), list(pe_history.closes)
        ),
        explain_none=_not_positive("eps_ttm", "negative earnings"),
    )
)

# ------------------------------------------------------------------ returns ----------------

for _window_name in ("1m", "3m", "6m", "1y", "ytd"):
    _spec(
        CalculationSpec(
            name=f"price_return_{_window_name}",
            formula="(close[{end_close}] / close[{start_close}] - 1) x 100",
            required_inputs=("start_close", "end_close"),
            optional_inputs=(),
            unit="percent",
            description=f"Price return over {_window_name.upper()}"
            " from the last close on or before the window start to the latest completed close "
            "(price return: dividends excluded).",
            fn=lambda start_close, end_close: prim.period_return([start_close, end_close]),
            percent_from_fraction=True,
        )
    )
for _role in ("market", "sector"):
    _spec(
        CalculationSpec(
            name=f"relative_return_1y_vs_{_role}",
            formula="(asset[{asset_end_close}] / asset[{asset_start_close}] - 1) - "
            "(benchmark[{benchmark_end_close}] / benchmark[{benchmark_start_close}] - 1), "
            "in percentage points",
            required_inputs=(
                "asset_start_close",
                "asset_end_close",
                "benchmark_start_close",
                "benchmark_end_close",
            ),
            optional_inputs=(),
            unit="percent",
            description=f"One-year price return minus the {_role} benchmark's, "
            "in percentage points.",
            fn=lambda **kw: prim.relative_return(
                prim.period_return([kw["asset_start_close"], kw["asset_end_close"]]),
                prim.period_return([kw["benchmark_start_close"], kw["benchmark_end_close"]]),
            ),
        )
    )
_spec(
    CalculationSpec(
        name="beta_1y_vs_market",
        formula="cov(daily returns[{asset_closes}], benchmark daily returns[{benchmark_closes}]) "
        "/ var(benchmark daily returns)",
        required_inputs=("asset_closes", "benchmark_closes"),
        optional_inputs=(),
        unit="coefficient",
        description="Beta to the broad-market benchmark over one year of daily simple returns "
        f"aligned on common sessions (at least {BETA_MIN_OBSERVATIONS} observations).",
        fn=_beta_from_closes,
    )
)
_spec(
    CalculationSpec(
        name="drawdown_vs_market_1y",
        formula="max_drawdown(asset closes[{asset_closes}]) "
        "- max_drawdown(benchmark closes[{benchmark_closes}]), in percentage points",
        required_inputs=("asset_closes", "benchmark_closes"),
        optional_inputs=(),
        unit="percent",
        description="Is the one-year drawdown idiosyncratic: the asset's maximum drawdown minus "
        "the broad market's over the same window (negative = the asset fell further).",
        fn=lambda asset_closes, benchmark_closes: prim.relative_return(
            _drawdown_fraction(asset_closes), _drawdown_fraction(benchmark_closes)
        ),
    )
)

# ------------------------------------------------------------------ volatility -------------

_spec(
    CalculationSpec(
        name="volatility_30d_annualized",
        formula="stdev(last 30 daily simple returns[{closes}]) x sqrt(252) x 100",
        required_inputs=("closes",),
        optional_inputs=(),
        unit="percent",
        description="Annualised volatility of the last 30 daily returns (31 closes).",
        fn=lambda closes: prim.annualized_volatility(_closes(closes)),
        percent_from_fraction=True,
    )
)
_spec(
    CalculationSpec(
        name="volatility_1y_annualized",
        formula="stdev(daily simple returns over one year[{closes}]) x sqrt(252) x 100",
        required_inputs=("closes",),
        optional_inputs=(),
        unit="percent",
        description="Annualised volatility of daily returns over the last year "
        f"(at least {VOLATILITY_1Y_MIN_OBSERVATIONS} closes).",
        fn=lambda closes: prim.annualized_volatility(_closes(closes)),
        percent_from_fraction=True,
    )
)
_spec(
    CalculationSpec(
        name="max_drawdown_1y",
        formula="min over the window of (close / running peak close - 1)[{closes}] x 100",
        required_inputs=("closes",),
        optional_inputs=(),
        unit="percent",
        description="Largest peak-to-trough decline in closes over the last year (negative).",
        fn=lambda closes: _drawdown_fraction(closes),
        percent_from_fraction=True,
    )
)
for _window in (50, 200):
    _spec(
        CalculationSpec(
            name=f"ma_{_window}",
            formula=f"mean(last {_window} closes[{{closes}}])",
            required_inputs=("closes",),
            optional_inputs=(),
            unit="USD_per_share",
            description=f"Simple {_window}-session moving average of closes.",
            fn=(lambda w: lambda closes: prim.moving_average(_closes(closes), w))(_window),
        )
    )
_spec(
    CalculationSpec(
        name="price_vs_ma_200_pct",
        formula="(price[{price}] / mean(last 200 closes[{closes}]) - 1) x 100",
        required_inputs=("price", "closes"),
        optional_inputs=(),
        unit="percent",
        description="Latest close relative to its 200-session moving average.",
        fn=lambda price, closes: prim.growth_rate(price, prim.moving_average(_closes(closes), 200)),
        percent_from_fraction=True,
    )
)

# ------------------------------------------------------------------ packs ------------------

CALCULATION_PACKS: dict[str, list[str]] = {
    "growth_and_margins": [
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
    ],
    "valuation_vs_history": [
        "market_cap",
        "enterprise_value",
        "pe_ttm",
        "ps_ttm",
        "ev_ebitda_ttm",
        "fcf_yield_ttm",
        "pe_5y_percentile",
    ],
    "returns_vs_benchmark": [
        "price_return_1m",
        "price_return_3m",
        "price_return_6m",
        "price_return_1y",
        "price_return_ytd",
        "relative_return_1y_vs_market",
        "relative_return_1y_vs_sector",
        "beta_1y_vs_market",
        "drawdown_vs_market_1y",
    ],
    "volatility_and_drawdown": [
        "volatility_30d_annualized",
        "volatility_1y_annualized",
        "max_drawdown_1y",
        "ma_50",
        "ma_200",
        "price_vs_ma_200_pct",
    ],
}
CALCULATION_PACKS["all_standard"] = [
    name
    for pack in (
        "growth_and_margins",
        "valuation_vs_history",
        "returns_vs_benchmark",
        "volatility_and_drawdown",
    )
    for name in CALCULATION_PACKS[pack]
]


def spec_for(name: str) -> CalculationSpec:
    try:
        return SPECS[name]
    except KeyError as exc:
        raise KeyError(f"unknown calculation {name!r}") from exc


# ------------------------------------------------------------------ compute ----------------


def compute(
    name: str,
    inputs: dict[str, InputValue],
    calc_id: str | None = None,
    period_label: str | None = None,
    *,
    notes: Iterable[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> CalculationResult:
    """Run the named calculation on the given operands and record everything.

    ``inputs`` maps each operand name of the spec to a ``CalculationInput`` (scalar, with its
    provenance passed through) or a ``SeriesOperand`` (a window of closes, recorded as an
    observation count with the window in ``meta["series"]``). A missing key, a ``None`` entry,
    a ``None`` value or an empty series makes the result ``unavailable`` with that operand in
    ``missing_inputs`` and the display ``"unavailable"``. A primitive that returns ``None`` for
    present operands (P/E on negative earnings) is also ``unavailable``, with the reason in
    ``notes`` and ``meta["reason"] = "not_meaningful"``.
    """
    spec = spec_for(name)
    result_notes = list(notes or [])
    result_meta: dict[str, Any] = dict(meta or {})
    recorded: list[CalculationInput] = []
    labels: dict[str, str] = {}
    values: dict[str, Any] = {}
    missing: list[str] = []
    series_meta: dict[str, Any] = {}

    for input_name in (*spec.required_inputs, *spec.optional_inputs):
        item = inputs.get(input_name)
        required = input_name in spec.required_inputs
        if isinstance(item, SeriesOperand):
            recorded.append(item.to_input())
            series_meta[input_name] = item.provenance()
            labels[input_name] = item.period_label
            if item.closes:
                values[input_name] = item
            elif required:
                missing.append(input_name)
            for note in item.notes:
                if note not in result_notes:
                    result_notes.append(note)
        elif isinstance(item, CalculationInput):
            recorded.append(item)
            labels[input_name] = item.period_label or input_name
            if prim.is_number(item.value):
                values[input_name] = item.value
            elif required:
                missing.append(input_name)
        else:
            recorded.append(CalculationInput(name=input_name, value=None))
            labels[input_name] = input_name
            if required:
                missing.append(input_name)

    formula = spec.formula.format_map(_LabelMap(labels))
    if series_meta:
        result_meta["series"] = series_meta
    result = CalculationResult(
        calc_id=calc_id or f"calc_{name}",
        name=name,
        formula=formula,
        inputs=recorded,
        unit=spec.unit,
        period_label=period_label,
        notes=result_notes,
        meta=result_meta,
    )
    if missing:
        result.status = "unavailable"
        result.missing_inputs = missing
        result.value = None
        result.display = UNAVAILABLE
        result.meta["reason"] = "missing_inputs"
        return result

    kwargs = {k: values[k] for k in spec.required_inputs}
    for optional in spec.optional_inputs:
        kwargs[optional] = values.get(optional)
    raw = spec.fn(**kwargs)
    if raw is None:
        result.status = "unavailable"
        result.value = None
        result.display = UNAVAILABLE
        result.meta["reason"] = "not_meaningful"
        explanation = spec.explain_none(values) if spec.explain_none else None
        result.notes.append(explanation or "the formula has no meaningful value for these operands")
        return result
    value = float(raw)
    if spec.percent_from_fraction:
        value = value * 100.0
    result.value = value
    result.display = format_value(value, spec.unit)
    return result


def require(result: CalculationResult) -> CalculationResult:
    """Return ``result`` when it is computed, else raise ``MISSING_CALCULATION_INPUT``.

    Opt-in: ``compute`` and ``run_pack`` never raise for missing operands. Use this when a
    calculation is a hard requirement of the caller's own contract.
    """
    if result.status == "computed":
        return result
    raise AnalysisError(
        ErrorCode.MISSING_CALCULATION_INPUT,
        f"Calculation {result.name!r} is unavailable: "
        + (
            "missing " + ", ".join(result.missing_inputs)
            if result.missing_inputs
            else "; ".join(result.notes) or "no meaningful value"
        ),
        details={
            "calculation": result.name,
            "calc_id": result.calc_id,
            "missing_inputs": list(result.missing_inputs),
            "notes": list(result.notes),
        },
    )


def require_all(results: Sequence[CalculationResult]) -> list[CalculationResult]:
    return [require(result) for result in results]


# ------------------------------------------------------------------ pack resolution --------


@dataclass
class _Resolved:
    inputs: dict[str, InputValue]
    period_label: str | None = None
    notes: list[str] | None = None
    meta: dict[str, Any] | None = None


def _op_input(name: str, operand: Operand | None) -> InputValue:
    return None if operand is None else operand.to_input(name)


def _provenance(**operands: Operand | None) -> dict[str, Any]:
    return {name: (op.provenance() if op else None) for name, op in operands.items()}


def _notes_of(*operands: Operand | None) -> list[str]:
    notes: list[str] = []
    for op in operands:
        if op is None:
            continue
        for note in op.notes:
            if note not in notes:
                notes.append(note)
    return notes


def _yoy_pair(
    resolver: OperandResolver, metric: str
) -> tuple[Operand | None, Operand | None, list[str]]:
    """Latest fiscal quarter and the same quarter a year earlier; fiscal-year pair as the
    labelled fallback. Both operands always share one period kind."""
    current = resolver.latest_fq(metric)
    previous = resolver.prior_year_quarter(metric, current) if current else None
    if current is not None and previous is not None:
        return current, previous, []
    fy_current = resolver.latest_fy(metric)
    fy_previous = resolver.fy_years_before(metric, fy_current, 1) if fy_current else None
    if fy_current is not None and fy_previous is not None:
        return (
            fy_current,
            fy_previous,
            ["quarterly year-over-year pair unavailable; fiscal-year pair used"],
        )
    return current, previous, []


def _resolve_growth_yoy(resolver: OperandResolver, metric: str, prefix: str) -> _Resolved:
    current, previous, notes = _yoy_pair(resolver, metric)
    return _Resolved(
        inputs={
            f"{prefix}_current": _op_input(f"{prefix}_current", current),
            f"{prefix}_previous": _op_input(f"{prefix}_previous", previous),
        },
        period_label=f"{current.period_label} vs {previous.period_label}"
        if current and previous
        else None,
        notes=[*notes, *_notes_of(current, previous)],
        meta={"operands": _provenance(current=current, previous=previous)},
    )


def _resolve_revenue_qoq(resolver: OperandResolver) -> _Resolved:
    current = resolver.latest_fq("revenue")
    previous = resolver.previous_quarter("revenue", current) if current else None
    return _Resolved(
        inputs={
            "revenue_current": _op_input("revenue_current", current),
            "revenue_previous": _op_input("revenue_previous", previous),
        },
        period_label=f"{current.period_label} vs {previous.period_label}"
        if current and previous
        else None,
        notes=_notes_of(current, previous),
        meta={"operands": _provenance(current=current, previous=previous)},
    )


def _resolve_revenue_cagr(resolver: OperandResolver) -> _Resolved:
    end = resolver.latest_fy("revenue")
    begin = resolver.fy_years_before("revenue", end, 3) if end else None
    return _Resolved(
        inputs={
            "revenue_begin": _op_input("revenue_begin", begin),
            "revenue_end": _op_input("revenue_end", end),
        },
        period_label=f"{begin.period_label} to {end.period_label}" if begin and end else None,
        notes=_notes_of(begin, end),
        meta={"operands": _provenance(begin=begin, end=end), "window": {"years": 3}},
    )


def _resolve_aligned(resolver: OperandResolver, metrics: Sequence[str]) -> _Resolved:
    aligned: Aligned = resolver.aligned(metrics)
    inputs: dict[str, InputValue] = {m: _op_input(m, aligned.operands.get(m)) for m in metrics}
    return _Resolved(
        inputs=inputs,
        period_label=aligned.period_label,
        notes=[*aligned.notes, *_notes_of(*aligned.operands.values())],
        meta={
            "operands": _provenance(**aligned.operands),
            "period_basis": aligned.basis,
        },
    )


def _resolve_margin_change(resolver: OperandResolver) -> _Resolved:
    current_q = resolver.latest_fq("revenue")
    inputs: dict[str, InputValue] = {}
    ops: dict[str, Operand | None] = {
        "revenue_current": current_q,
        "operating_income_current": None,
        "revenue_previous": None,
        "operating_income_previous": None,
    }
    if current_q is not None:
        period = resolver.period_of(current_q)
        if period is not None:
            ops["operating_income_current"] = resolver.for_period("operating_income", period)
        ops["revenue_previous"] = resolver.prior_year_quarter("revenue", current_q)
        if ops["operating_income_current"] is not None:
            ops["operating_income_previous"] = resolver.prior_year_quarter(
                "operating_income", ops["operating_income_current"]
            )
    for name, op in ops.items():
        inputs[name] = _op_input(name, op)
    cur, prev = ops["revenue_current"], ops["revenue_previous"]
    return _Resolved(
        inputs=inputs,
        period_label=f"{cur.period_label} vs {prev.period_label}" if cur and prev else None,
        notes=_notes_of(*ops.values()),
        meta={"operands": _provenance(**ops)},
    )


def _resolve_ttm_metrics(resolver: OperandResolver, metrics: Sequence[str]) -> _Resolved:
    """TTM (or labelled fiscal-year) operands named ``<metric>_ttm``, aligned on one period."""
    aligned = resolver.aligned(metrics)
    inputs: dict[str, InputValue] = {
        f"{m}_ttm": _op_input(f"{m}_ttm", aligned.operands.get(m)) for m in metrics
    }
    return _Resolved(
        inputs=inputs,
        period_label=aligned.period_label,
        notes=[*aligned.notes, *_notes_of(*aligned.operands.values())],
        meta={"operands": _provenance(**aligned.operands), "period_basis": aligned.basis},
    )


def _resolve_eps_growth(resolver: OperandResolver) -> _Resolved:
    return _resolve_growth_yoy(resolver, "eps_diluted", "eps")


def _market_cap_inputs(
    resolver: OperandResolver,
) -> tuple[dict[str, InputValue], list[str], dict[str, Any]]:
    price, shares, notes = resolver.market_cap_operands()
    inputs: dict[str, InputValue] = {
        "price": _op_input("price", price),
        "shares_outstanding": _op_input("shares_outstanding", shares),
    }
    return (
        inputs,
        [*notes, *_notes_of(price, shares)],
        _provenance(price=price, shares_outstanding=shares),
    )


def _resolve_market_cap(resolver: OperandResolver) -> _Resolved:
    inputs, notes, prov = _market_cap_inputs(resolver)
    price = inputs["price"]
    return _Resolved(
        inputs=inputs,
        period_label=price.period_label if isinstance(price, CalculationInput) else None,
        notes=notes,
        meta={"operands": prov},
    )


def _resolve_enterprise_value(resolver: OperandResolver) -> _Resolved:
    inputs, notes, prov = _market_cap_inputs(resolver)
    debt = resolver.latest_balance("total_debt")
    cash = resolver.latest_balance("cash_and_equivalents")
    inputs["total_debt"] = _op_input("total_debt", debt)
    inputs["cash_and_equivalents"] = _op_input("cash_and_equivalents", cash)
    prov.update(_provenance(total_debt=debt, cash_and_equivalents=cash))
    price = inputs["price"]
    return _Resolved(
        inputs=inputs,
        period_label=price.period_label if isinstance(price, CalculationInput) else None,
        notes=[*notes, *_notes_of(debt, cash)],
        meta={"operands": prov},
    )


def _resolve_pe(resolver: OperandResolver) -> _Resolved:
    price = resolver.latest_close()
    eps = resolver.ttm_or_fy("eps_diluted")
    return _Resolved(
        inputs={"price": _op_input("price", price), "eps_ttm": _op_input("eps_ttm", eps)},
        period_label=f"{price.period_label}; EPS {eps.period_label}" if price and eps else None,
        notes=_notes_of(price, eps),
        meta={"operands": _provenance(price=price, eps_ttm=eps)},
    )


def _resolve_ps(resolver: OperandResolver) -> _Resolved:
    inputs, notes, prov = _market_cap_inputs(resolver)
    revenue = resolver.ttm_or_fy("revenue")
    inputs["revenue_ttm"] = _op_input("revenue_ttm", revenue)
    prov.update(_provenance(revenue_ttm=revenue))
    return _Resolved(
        inputs=inputs,
        period_label=revenue.period_label if revenue else None,
        notes=[*notes, *_notes_of(revenue)],
        meta={"operands": prov},
    )


def _resolve_ev_ebitda(resolver: OperandResolver) -> _Resolved:
    ev = _resolve_enterprise_value(resolver)
    aligned = resolver.aligned(["operating_income", "depreciation_amortization"])
    inputs = dict(ev.inputs)
    for metric in ("operating_income", "depreciation_amortization"):
        inputs[f"{metric}_ttm"] = _op_input(f"{metric}_ttm", aligned.operands.get(metric))
    prov = dict(ev.meta["operands"]) if ev.meta else {}
    prov.update(_provenance(**{f"{m}_ttm": op for m, op in aligned.operands.items()}))
    return _Resolved(
        inputs=inputs,
        period_label=aligned.period_label,
        notes=[*(ev.notes or []), *aligned.notes, *_notes_of(*aligned.operands.values())],
        meta={"operands": prov, "period_basis": aligned.basis},
    )


def _resolve_fcf_yield(resolver: OperandResolver) -> _Resolved:
    inputs, notes, prov = _market_cap_inputs(resolver)
    aligned = resolver.aligned(["operating_cash_flow", "capex"])
    for metric in ("operating_cash_flow", "capex"):
        inputs[f"{metric}_ttm"] = _op_input(f"{metric}_ttm", aligned.operands.get(metric))
    prov.update(_provenance(**{f"{m}_ttm": op for m, op in aligned.operands.items()}))
    return _Resolved(
        inputs=inputs,
        period_label=aligned.period_label,
        notes=[*notes, *aligned.notes, *_notes_of(*aligned.operands.values())],
        meta={"operands": prov, "period_basis": aligned.basis},
    )


def _resolve_pe_percentile(resolver: OperandResolver) -> _Resolved:
    price = resolver.latest_close()
    eps = resolver.ttm("eps_diluted")
    history, history_notes = resolver.pe_history(PE_HISTORY_YEARS)
    notes = [*_notes_of(price, eps), *history_notes]
    series: SeriesOperand | None = None
    if len(history) >= PE_HISTORY_MIN_POINTS:
        series = SeriesOperand(
            name="pe_history",
            closes=tuple(point["pe"] for point in history),
            dates=tuple(datetime.fromisoformat(point["quarter_end"]).date() for point in history),
            source_id=resolver.evidence.prices.source_id if resolver.evidence.prices else None,
            symbol=resolver.evidence.symbol,
            notes=(f"{len(history)} quarter-end trailing P/E points",),
        )
    else:
        notes.append(
            f"P/E history has {len(history)} points; at least {PE_HISTORY_MIN_POINTS} are required"
        )
    return _Resolved(
        inputs={
            "price": _op_input("price", price),
            "eps_ttm": _op_input("eps_ttm", eps),
            "pe_history": series,
        },
        period_label=f"{price.period_label}; EPS {eps.period_label}" if price and eps else None,
        notes=notes,
        meta={
            "operands": _provenance(price=price, eps_ttm=eps),
            "history": history,
            "window": {"years": PE_HISTORY_YEARS, "points": len(history)},
        },
    )


def _return_inputs(
    resolver: OperandResolver, series, prefix: str, **window_kwargs: Any
) -> tuple[dict[str, InputValue], SeriesOperand | None, list[str]]:
    window = resolver.window(f"{prefix}closes", series, **window_kwargs)
    if window is None or len(window.closes) < 2:
        return {f"{prefix}start_close": None, f"{prefix}end_close": None}, window, []
    currency = f"{series.currency}_per_share"
    start = CalculationInput(
        name=f"{prefix}start_close",
        value=window.closes[0],
        unit=currency,
        source_id=window.source_id,
        period_label=f"close {window.start.isoformat()}",  # type: ignore[union-attr]
    )
    end = CalculationInput(
        name=f"{prefix}end_close",
        value=window.closes[-1],
        unit=currency,
        source_id=window.source_id,
        period_label=f"close {window.end.isoformat()}",  # type: ignore[union-attr]
    )
    return {f"{prefix}start_close": start, f"{prefix}end_close": end}, window, list(window.notes)


def _window_meta(window: SeriesOperand | None, **extra: Any) -> dict[str, Any]:
    if window is None:
        return {"window": {"available": False, **extra}}
    return {
        "window": {
            "available": True,
            "start": window.start.isoformat() if window.start else None,
            "end": window.end.isoformat() if window.end else None,
            "requested_start": window.requested_start.isoformat()
            if window.requested_start
            else None,
            "points": len(window.closes),
            **extra,
        }
    }


def _resolve_price_return(resolver: OperandResolver, window_name: str) -> _Resolved:
    series = resolver.price_series()
    kwargs: dict[str, Any] = (
        {"ytd": True}
        if window_name == "ytd"
        else {"months": {"1m": 1, "3m": 3, "6m": 6, "1y": 12}[window_name]}
    )
    inputs, window, notes = _return_inputs(resolver, series, "", **kwargs)
    if series is None or not series.points:
        notes.append("no price series")
    elif window is None:
        notes.append(f"price series does not reach back {window_name}")
    return _Resolved(
        inputs=inputs,
        period_label=window.period_label if window else None,
        notes=notes,
        meta=_window_meta(window, name=window_name),
    )


def _resolve_relative_return(resolver: OperandResolver, role: str) -> _Resolved:
    series = resolver.price_series()
    bench, record = resolver.benchmark_series(role)
    asset_inputs, asset_window, notes = _return_inputs(resolver, series, "asset_", months=12)
    inputs: dict[str, InputValue] = dict(asset_inputs)
    bench_window: SeriesOperand | None = None
    if bench is None:
        notes.append(f"no {role.replace('_', ' ')} benchmark series available")
        inputs["benchmark_start_close"] = None
        inputs["benchmark_end_close"] = None
    else:
        end_date = asset_window.end if asset_window else None
        bench_inputs, bench_window, bench_notes = _return_inputs(
            resolver, bench, "benchmark_", months=12, end_date=end_date
        )
        inputs.update(bench_inputs)
        notes.extend(bench_notes)
        if (
            asset_window
            and bench_window
            and (asset_window.start != bench_window.start or asset_window.end != bench_window.end)
        ):
            notes.append(
                f"asset window {asset_window.period_label} vs benchmark window "
                f"{bench_window.period_label}: session dates differ"
            )
    return _Resolved(
        inputs=inputs,
        period_label=asset_window.period_label if asset_window else None,
        notes=notes,
        meta={
            **_window_meta(asset_window, name="1y"),
            "benchmark": record,
            "benchmark_window": _window_meta(bench_window)["window"],
        },
    )


def _resolve_benchmark_series_pair(resolver: OperandResolver, role: str, minimum: int) -> _Resolved:
    series = resolver.price_series()
    bench, record = resolver.benchmark_series(role)
    asset_window = resolver.window("asset_closes", series, months=12)
    bench_window = resolver.window("benchmark_closes", bench, months=12) if bench else None
    notes: list[str] = []
    if asset_window is None:
        notes.append("price series does not cover one year")
    if bench is None:
        notes.append(f"no {role.replace('_', ' ')} benchmark series available")
    elif bench_window is None:
        notes.append("benchmark series does not cover one year")
    if asset_window is not None and bench_window is not None:
        asset_window, bench_window = OperandResolver.align(asset_window, bench_window)
        if len(asset_window.closes) < minimum:
            notes.append(
                f"only {len(asset_window.closes)} common sessions; at least {minimum} required"
            )
            asset_window = bench_window = None
    return _Resolved(
        inputs={"asset_closes": asset_window, "benchmark_closes": bench_window},
        period_label=asset_window.period_label if asset_window else None,
        notes=notes,
        meta={**_window_meta(asset_window, name="1y"), "benchmark": record},
    )


def _resolve_closes(resolver: OperandResolver, **window_kwargs: Any) -> _Resolved:
    series = resolver.price_series()
    window = resolver.window("closes", series, **window_kwargs)
    notes: list[str] = []
    if series is None or not series.points:
        notes.append("no price series")
    elif window is None:
        notes.append("price series is shorter than the requested window")
    return _Resolved(
        inputs={"closes": window},
        period_label=window.period_label if window else None,
        notes=notes,
        meta=_window_meta(window, **{k: v for k, v in window_kwargs.items()}),
    )


def _resolve_volatility_1y(resolver: OperandResolver) -> _Resolved:
    resolved = _resolve_closes(resolver, months=12)
    window = resolved.inputs.get("closes")
    if isinstance(window, SeriesOperand) and len(window.closes) < VOLATILITY_1Y_MIN_OBSERVATIONS:
        resolved.notes = [
            *(resolved.notes or []),
            f"only {len(window.closes)} closes in the year; at least "
            f"{VOLATILITY_1Y_MIN_OBSERVATIONS} required",
        ]
        resolved.inputs["closes"] = None
    return resolved


def _resolve_max_drawdown(resolver: OperandResolver) -> _Resolved:
    resolved = _resolve_closes(resolver, months=12)
    window = resolved.inputs.get("closes")
    if isinstance(window, SeriesOperand):
        result = prim.max_drawdown(list(window.closes))
        if result is not None:
            _fraction, peak_index, trough_index = result
            resolved.meta = {
                **(resolved.meta or {}),
                "peak_date": window.dates[peak_index].isoformat(),
                "trough_date": window.dates[trough_index].isoformat(),
                "peak_close": window.closes[peak_index],
                "trough_close": window.closes[trough_index],
            }
    return resolved


def _resolve_price_vs_ma(resolver: OperandResolver) -> _Resolved:
    resolved = _resolve_closes(resolver, trading_days=200)
    price = resolver.latest_close()
    resolved.inputs["price"] = _op_input("price", price)
    resolved.notes = [*(resolved.notes or []), *_notes_of(price)]
    resolved.meta = {**(resolved.meta or {}), "operands": _provenance(price=price)}
    return resolved


_RESOLVERS: dict[str, Callable[[OperandResolver], _Resolved]] = {
    "revenue_growth_yoy": lambda r: _resolve_growth_yoy(r, "revenue", "revenue"),
    "revenue_growth_qoq": _resolve_revenue_qoq,
    "revenue_cagr_3y": _resolve_revenue_cagr,
    "gross_margin": lambda r: _resolve_aligned(r, ["gross_profit", "revenue"]),
    "operating_margin": lambda r: _resolve_aligned(r, ["operating_income", "revenue"]),
    "net_margin": lambda r: _resolve_aligned(r, ["net_income", "revenue"]),
    "fcf_margin": lambda r: _resolve_aligned(r, ["operating_cash_flow", "capex", "revenue"]),
    "operating_margin_change_bp": _resolve_margin_change,
    "free_cash_flow_ttm": lambda r: _resolve_ttm_metrics(r, ["operating_cash_flow", "capex"]),
    "eps_growth_yoy": _resolve_eps_growth,
    "market_cap": _resolve_market_cap,
    "enterprise_value": _resolve_enterprise_value,
    "pe_ttm": _resolve_pe,
    "ps_ttm": _resolve_ps,
    "ev_ebitda_ttm": _resolve_ev_ebitda,
    "fcf_yield_ttm": _resolve_fcf_yield,
    "pe_5y_percentile": _resolve_pe_percentile,
    "price_return_1m": lambda r: _resolve_price_return(r, "1m"),
    "price_return_3m": lambda r: _resolve_price_return(r, "3m"),
    "price_return_6m": lambda r: _resolve_price_return(r, "6m"),
    "price_return_1y": lambda r: _resolve_price_return(r, "1y"),
    "price_return_ytd": lambda r: _resolve_price_return(r, "ytd"),
    "relative_return_1y_vs_market": lambda r: _resolve_relative_return(r, "broad_market"),
    "relative_return_1y_vs_sector": lambda r: _resolve_relative_return(r, "sector"),
    "beta_1y_vs_market": lambda r: _resolve_benchmark_series_pair(
        r, "broad_market", BETA_MIN_OBSERVATIONS
    ),
    "drawdown_vs_market_1y": lambda r: _resolve_benchmark_series_pair(r, "broad_market", 2),
    "volatility_30d_annualized": lambda r: _resolve_closes(
        r, trading_days=VOLATILITY_30D_RETURNS + 1
    ),
    "volatility_1y_annualized": _resolve_volatility_1y,
    "max_drawdown_1y": _resolve_max_drawdown,
    "ma_50": lambda r: _resolve_closes(r, trading_days=50),
    "ma_200": lambda r: _resolve_closes(r, trading_days=200),
    "price_vs_ma_200_pct": _resolve_price_vs_ma,
}

assert set(_RESOLVERS) == set(SPECS), "every spec needs a resolver"
assert set(CALCULATION_PACKS["all_standard"]) == set(SPECS), "every spec belongs to a pack"


def run_pack(
    pack: str,
    evidence: NormalizedEvidence,
    as_of: datetime,
    calc_id_factory: Callable[[str], str] | None = None,
    *,
    holidays: Sequence[date] | None = None,
) -> list[CalculationResult]:
    """Run every calculation in ``pack`` against ``evidence`` as of ``as_of``.

    Operands are pulled deterministically (see :mod:`bayanalytics.calculations.operands`), so
    the same evidence and ``as_of`` always produce identical results. ``calc_id_factory`` maps
    a calculation name to an id; the default is ``calc_<name>``. Results come back in pack
    order, one per calculation, computed or unavailable; nothing raises for missing data.
    Every result's ``meta`` carries ``pack``, ``as_of`` and, where relevant, ``window``,
    ``operands`` (fact ids, periods, sources) and ``benchmark`` (which benchmark was used).
    """
    if pack not in CALCULATION_PACKS:
        raise KeyError(f"unknown calculation pack {pack!r}")
    resolver = OperandResolver(evidence, as_of, holidays=holidays)
    factory = calc_id_factory or (lambda name: f"calc_{name}")
    results: list[CalculationResult] = []
    for name in CALCULATION_PACKS[pack]:
        resolved = _RESOLVERS[name](resolver)
        meta = {"pack": pack, "as_of": as_of.isoformat(), "symbol": evidence.symbol}
        meta.update(resolved.meta or {})
        results.append(
            compute(
                name,
                resolved.inputs,
                calc_id=factory(name),
                period_label=resolved.period_label,
                notes=resolved.notes,
                meta=meta,
            )
        )
    return results


def run_packs(
    packs: Iterable[str],
    evidence: NormalizedEvidence,
    as_of: datetime,
    calc_id_factory: Callable[[str], str] | None = None,
) -> list[CalculationResult]:
    """Run several packs, de-duplicating calculations that appear in more than one."""
    seen: set[str] = set()
    results: list[CalculationResult] = []
    for pack in packs:
        for result in run_pack(pack, evidence, as_of, calc_id_factory):
            if result.name in seen:
                continue
            seen.add(result.name)
            results.append(result)
    return results
