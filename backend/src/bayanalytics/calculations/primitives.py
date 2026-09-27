"""Pure, deterministic arithmetic primitives (AGENT.md sections 3.3 and 25).

Every function here is ordinary software: no model, no network, no randomness, no hidden
state. The contract shared by all of them:

* Inputs are plain floats (or lists of floats). ``bool`` is rejected as a number.
* When an input is missing (``None``), non-finite (NaN/inf) or mathematically invalid for
  the formula (a zero denominator, a negative base for a root, an empty series), the function
  returns ``None``. It never raises for missing data and it never silently substitutes a
  value (no "treat missing as zero", no clamping).
* Results are returned at full float precision. Ratios are fractions (``0.182`` means
  18.2 %), never pre-rounded: rounding happens only in :mod:`bayanalytics.calculations.formatting`.
* Series are ordered oldest -> newest. Nothing here sorts a series for the caller.

The registry (:mod:`bayanalytics.calculations.registry`) wraps these primitives with operand
provenance so every value can be reproduced from the recorded inputs.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

Number = int | float


def is_number(value: Any) -> bool:
    """True when ``value`` is a finite int/float (``bool`` is excluded on purpose)."""
    if value is None or isinstance(value, bool):
        return False
    if not isinstance(value, int | float):
        return False
    return math.isfinite(value)


def _all_numbers(values: Sequence[Any] | None, minimum: int = 1) -> bool:
    if values is None:
        return False
    if len(values) < minimum:
        return False
    return all(is_number(v) for v in values)


def safe_div(numerator: Number | None, denominator: Number | None) -> float | None:
    """``numerator / denominator`` or ``None`` when either is missing or the denominator is 0."""
    if not is_number(numerator) or not is_number(denominator):
        return None
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


# --------------------------------------------------------------------------------------
# Growth and margins
# --------------------------------------------------------------------------------------


def growth_rate(current: Number | None, previous: Number | None) -> float | None:
    """Period-over-period growth as a fraction: ``(current - previous) / |previous|``.

    The denominator uses the absolute value of the base so that the sign of the result is
    always the direction of the change: moving from a loss of -100 to a loss of -50 is an
    improvement and reports ``+0.5``; the naive ``(-50 - -100) / -100 = -0.5`` would call it a
    deterioration. Returns ``None`` when ``previous`` is ``None`` or ``0`` (growth from a zero
    base is undefined) or when ``current`` is missing.
    """
    if not is_number(current) or not is_number(previous):
        return None
    if previous == 0:
        return None
    return (float(current) - float(previous)) / abs(float(previous))


def cagr(begin: Number | None, end: Number | None, years: Number | None) -> float | None:
    """Compound annual growth rate as a fraction: ``(end / begin) ** (1 / years) - 1``.

    Requires ``begin > 0``, ``end >= 0`` and ``years > 0``; a CAGR over a negative or zero
    base has no real root and returns ``None``. ``end == 0`` is allowed and yields ``-1.0``.
    """
    if not is_number(begin) or not is_number(end) or not is_number(years):
        return None
    if begin <= 0 or end < 0 or years <= 0:
        return None
    return (float(end) / float(begin)) ** (1.0 / float(years)) - 1.0


def margin(numerator: Number | None, revenue: Number | None) -> float | None:
    """A margin as a fraction of revenue: ``numerator / revenue``.

    Returns ``None`` when revenue is missing or not strictly positive (a margin on zero or
    negative revenue is not meaningful).
    """
    if not is_number(numerator) or not is_number(revenue):
        return None
    if revenue <= 0:
        return None
    return float(numerator) / float(revenue)


def margin_change_bp(current_margin: Number | None, previous_margin: Number | None) -> float | None:
    """Margin expansion (+) or contraction (-) in basis points.

    Both margins are fractions (``0.182`` for 18.2 %); ``1 bp = 0.0001``, so the result is
    ``(current - previous) * 10_000``.
    """
    if not is_number(current_margin) or not is_number(previous_margin):
        return None
    return (float(current_margin) - float(previous_margin)) * 10_000.0


def ttm_sum(quarterly_values: Sequence[Number | None] | None) -> float | None:
    """Trailing-twelve-month sum of exactly four quarterly values.

    Returns ``None`` unless exactly four finite values are supplied: three quarters plus an
    estimate is not a TTM figure, and five quarters is a different window.
    """
    if quarterly_values is None or len(quarterly_values) != 4:
        return None
    if not _all_numbers(quarterly_values, minimum=4):
        return None
    return float(sum(float(v) for v in quarterly_values))


# --------------------------------------------------------------------------------------
# Valuation
# --------------------------------------------------------------------------------------


def price_to_earnings(price: Number | None, eps_ttm: Number | None) -> float | None:
    """Trailing P/E: ``price / eps_ttm``.

    Returns ``None`` when ``eps_ttm <= 0``: a P/E on negative (or zero) earnings is not a
    meaningful multiple. Callers should attach the note "negative earnings" rather than
    reporting a negative multiple.
    """
    if not is_number(price) or not is_number(eps_ttm):
        return None
    if eps_ttm <= 0 or price < 0:
        return None
    return float(price) / float(eps_ttm)


def price_to_sales(market_cap: Number | None, revenue_ttm: Number | None) -> float | None:
    """Price-to-sales: ``market_cap / revenue_ttm``; ``None`` unless revenue is positive."""
    if not is_number(market_cap) or not is_number(revenue_ttm):
        return None
    if revenue_ttm <= 0 or market_cap < 0:
        return None
    return float(market_cap) / float(revenue_ttm)


def enterprise_value(
    market_cap: Number | None, total_debt: Number | None, cash: Number | None
) -> float | None:
    """Enterprise value: ``market_cap + total_debt - cash``. All three operands are required;
    a missing debt or cash figure is not treated as zero."""
    if not is_number(market_cap) or not is_number(total_debt) or not is_number(cash):
        return None
    return float(market_cap) + float(total_debt) - float(cash)


def ev_to_ebitda(ev: Number | None, ebitda: Number | None) -> float | None:
    """EV / EBITDA; ``None`` when EBITDA is not positive (negative EBITDA multiples are not
    meaningful)."""
    if not is_number(ev) or not is_number(ebitda):
        return None
    if ebitda <= 0:
        return None
    return float(ev) / float(ebitda)


def free_cash_flow(operating_cash_flow: Number | None, capex: Number | None) -> float | None:
    """Free cash flow: ``operating_cash_flow - |capex|``.

    XBRL reports capital expenditure as ``PaymentsToAcquirePropertyPlantAndEquipment``, a
    positive outflow magnitude; some statements present the same number negative under the
    cash-flow sign convention. Capex is therefore always treated as a positive outflow
    magnitude (``abs``) so the same economic quantity produces the same FCF regardless of the
    sign convention of the source.
    """
    if not is_number(operating_cash_flow) or not is_number(capex):
        return None
    return float(operating_cash_flow) - abs(float(capex))


def fcf_yield(fcf_ttm: Number | None, market_cap: Number | None) -> float | None:
    """FCF yield as a fraction: ``fcf_ttm / market_cap``; ``None`` unless market cap > 0."""
    if not is_number(fcf_ttm) or not is_number(market_cap):
        return None
    if market_cap <= 0:
        return None
    return float(fcf_ttm) / float(market_cap)


# --------------------------------------------------------------------------------------
# Prices, returns, risk
# --------------------------------------------------------------------------------------


def period_return(closes: Sequence[Number | None] | None) -> float | None:
    """Simple return over a series of closes: ``closes[-1] / closes[0] - 1``.

    Requires at least two finite closes and a strictly positive first close.
    """
    if closes is None or len(closes) < 2:
        return None
    first, last = closes[0], closes[-1]
    if not is_number(first) or not is_number(last) or first <= 0:
        return None
    return float(last) / float(first) - 1.0


def returns_series(closes: Sequence[Number | None] | None) -> list[float] | None:
    """Simple period-over-period returns ``closes[i] / closes[i-1] - 1``.

    Returns an empty list for fewer than two closes and ``None`` when any close is missing,
    non-finite or non-positive (a return through a zero price is undefined).
    """
    if closes is None:
        return None
    if len(closes) < 2:
        return [] if all(is_number(c) for c in closes) else None
    if not all(is_number(c) and c > 0 for c in closes):
        return None
    return [float(closes[i]) / float(closes[i - 1]) - 1.0 for i in range(1, len(closes))]


def relative_return(asset_return: Number | None, benchmark_return: Number | None) -> float | None:
    """Asset return minus benchmark return, in percentage points.

    Both inputs are fractions (``0.08`` for +8 %); the result is ``(asset - benchmark) * 100``
    so ``relative_return(0.08, 0.04) == 4.0`` (four percentage points of outperformance).
    """
    if not is_number(asset_return) or not is_number(benchmark_return):
        return None
    return (float(asset_return) - float(benchmark_return)) * 100.0


def annualized_volatility(
    closes: Sequence[Number | None] | None, trading_days: int = 252
) -> float | None:
    """Annualised volatility of simple daily returns as a fraction.

    ``stdev(returns) * sqrt(trading_days)`` using the sample standard deviation (n - 1).
    Requires at least three closes (two returns). Uses simple, not log, returns.
    """
    if not is_number(trading_days) or trading_days <= 0:
        return None
    rets = returns_series(closes)
    if rets is None or len(rets) < 2:
        return None
    return statistics.stdev(rets) * math.sqrt(float(trading_days))


def max_drawdown(
    closes: Sequence[Number | None] | None,
) -> tuple[float, int, int] | None:
    """Maximum peak-to-trough drawdown of a close series.

    Returns ``(drawdown_fraction, peak_index, trough_index)`` where ``drawdown_fraction`` is
    ``trough / peak - 1`` (a non-positive number, ``-0.234`` for a 23.4 % drawdown) and the
    indices point at the running peak and the trough that realised it. A monotonically rising
    series returns ``(0.0, 0, 0)``. Requires at least one finite, positive close.
    """
    if closes is None or len(closes) == 0:
        return None
    if not all(is_number(c) and c > 0 for c in closes):
        return None
    peak = float(closes[0])
    peak_index = 0
    best = 0.0
    best_peak = 0
    best_trough = 0
    for index, raw in enumerate(closes):
        close = float(raw)
        if close > peak:
            peak = close
            peak_index = index
        drawdown = close / peak - 1.0
        if drawdown < best:
            best = drawdown
            best_peak = peak_index
            best_trough = index
    return best, best_peak, best_trough


def moving_average(values: Sequence[Number | None] | None, window: int) -> float | None:
    """Simple moving average of the most recent ``window`` values (the latest SMA value).

    Requires at least ``window`` finite values; the average is over exactly the last
    ``window`` entries of the series.
    """
    if values is None or not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        return None
    if len(values) < window:
        return None
    tail = values[-window:]
    if not _all_numbers(tail, minimum=window):
        return None
    return float(sum(float(v) for v in tail)) / float(window)


def zscore(value: Number | None, history: Sequence[Number | None] | None) -> float | None:
    """``(value - mean(history)) / stdev(history)`` with the sample standard deviation.

    Requires at least two history points with non-zero dispersion.
    """
    if not is_number(value) or not _all_numbers(history, minimum=2):
        return None
    assert history is not None
    values = [float(v) for v in history]
    spread = statistics.stdev(values)
    if spread == 0:
        return None
    return (float(value) - statistics.fmean(values)) / spread


def percentile_rank(value: Number | None, history: Sequence[Number | None] | None) -> float | None:
    """Percentile rank of ``value`` within ``history`` on a 0..100 scale.

    Uses the mid-rank convention for ties: ``(count_below + 0.5 * count_equal) / n * 100``, so a
    value equal to every history point sits at 50 and a value above all of them at 100.
    """
    if not is_number(value) or not _all_numbers(history, minimum=1):
        return None
    assert history is not None
    target = float(value)
    below = sum(1 for v in history if float(v) < target)
    equal = sum(1 for v in history if float(v) == target)
    return (below + 0.5 * equal) / float(len(history)) * 100.0


def correlation(
    a: Sequence[Number | None] | None, b: Sequence[Number | None] | None
) -> float | None:
    """Pearson correlation of two equally long series; ``None`` for fewer than two points,
    length mismatch or a constant series."""
    if a is None or b is None or len(a) != len(b):
        return None
    if not _all_numbers(a, minimum=2) or not _all_numbers(b, minimum=2):
        return None
    xs = [float(v) for v in a]
    ys = [float(v) for v in b]
    try:
        return statistics.correlation(xs, ys)
    except statistics.StatisticsError:
        return None


def beta(
    asset_returns: Sequence[Number | None] | None,
    bench_returns: Sequence[Number | None] | None,
) -> float | None:
    """Beta of the asset to the benchmark: ``cov(asset, bench) / var(bench)``.

    Both series must be the same length and already aligned by observation date. Uses the
    sample covariance and variance (both n - 1, so the ratio equals the population ratio).
    ``None`` for fewer than two observations or a constant benchmark.
    """
    if asset_returns is None or bench_returns is None:
        return None
    if len(asset_returns) != len(bench_returns):
        return None
    if not _all_numbers(asset_returns, minimum=2) or not _all_numbers(bench_returns, minimum=2):
        return None
    xs = [float(v) for v in asset_returns]
    ys = [float(v) for v in bench_returns]
    variance = statistics.variance(ys)
    if variance == 0:
        return None
    return statistics.covariance(xs, ys) / variance


def scenario_table(
    base_value: Number | None, deltas: Sequence[Number | None] | None
) -> list[dict[str, float]] | None:
    """Simple what-if table: for each fractional ``delta`` return ``base * (1 + delta)``.

    Each row is ``{"delta": delta, "value": base * (1 + delta), "change": base * delta}``.
    Returns ``None`` when the base or any delta is missing; an empty ``deltas`` gives ``[]``.
    """
    if not is_number(base_value) or deltas is None:
        return None
    if not all(is_number(d) for d in deltas):
        return None
    base = float(base_value)
    return [
        {"delta": float(d), "value": base * (1.0 + float(d)), "change": base * float(d)}
        for d in deltas
    ]
