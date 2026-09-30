"""Client-facing market views: price series, small previews of structured sources and the
quarterly fundamentals series. Pure functions over data the pipeline already holds.

Price points only leave the backend when ``Settings.price_display`` is on (see config): Stooq's
terms are personal use, so its data is ``metadata_only`` by default.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from urllib.parse import urlsplit

from bayanalytics.calculations.operands import OperandResolver
from bayanalytics.research.edgar import EdgarSubmissions
from bayanalytics.research.extract import registrable_domain
from bayanalytics.schemas.evidence import NormalizedEvidence, PriceSeries
from bayanalytics.schemas.results import FundamentalQuarter, MarketFundamentals, MarketSeries

MAX_SERIES_POINTS = 1300
PREVIEW_ROWS = 4
FUNDAMENTAL_QUARTERS = 8
_FACT_PREVIEW = (
    ("revenue", "Revenue"),
    ("gross_profit", "Gross profit"),
    ("operating_cash_flow", "Operating cash flow"),
    ("eps_diluted", "Diluted EPS"),
)


def domain_of(url: str) -> str:
    return registrable_domain(urlsplit(url).netloc) if url else ""


def short_date(d: date | datetime | str | None) -> str:
    if d is None:
        return "—"
    if isinstance(d, str):
        try:
            d = date.fromisoformat(d[:10])
        except ValueError:
            return d
    return f"{d:%b} {d.day}, {d.year}"


def money(value: float, unit: str | None = "USD") -> str:
    if unit == "USD/shares":
        return f"${value:,.2f}"
    sign = "-" if value < 0 else ""
    v = abs(value)
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if v >= size:
            return f"{sign}${v / size:,.1f}{suffix}"
    return f"{sign}${v:,.0f}"


def series_payload(series: PriceSeries, role: str, symbol: str, name: str) -> MarketSeries:
    points = [
        (p.date.isoformat(), p.open, p.high, p.low, float(p.close), p.volume)
        for p in series.points[-MAX_SERIES_POINTS:]
    ]
    return MarketSeries(
        role=role,  # type: ignore[arg-type]
        symbol=symbol,
        name=name,
        source_id=series.source_id,
        currency=series.currency,
        points=points,
    )


def price_preview(series: PriceSeries) -> dict[str, Any]:
    rows = [
        [short_date(p.date), f"{p.close:,.2f}"] for p in reversed(series.points[-PREVIEW_ROWS:])
    ]
    return {"columns": ["Date", "Close"], "rows": rows}


def submissions_preview(subs: EdgarSubmissions, as_of: datetime) -> dict[str, Any] | None:
    filings = subs.latest_filings(forms=(), limit=PREVIEW_ROWS, as_of=as_of)
    if not filings:
        return None
    rows = [
        [f.form, short_date(f.filing_date), short_date(f.report_date) if f.report_date else "—"]
        for f in filings
    ]
    return {"columns": ["Form", "Filed", "Period"], "rows": rows}


def facts_preview(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    out: list[list[str]] = []
    for metric, label in _FACT_PREVIEW:
        candidates = [
            r
            for r in rows
            if r.get("metric") == metric and str(r.get("form", "")).startswith("10-")
        ]
        if not candidates:
            continue
        latest = max(candidates, key=lambda r: (r["end"], r["filed"]))
        fp, fy = latest.get("fp"), latest.get("fy")
        period = f"{fp} {fy}" if fp and fy else short_date(latest["end"])
        out.append([label, period, money(float(latest["value"]), latest.get("unit"))])
    return {"columns": ["Metric", "Period", "Value"], "rows": out} if out else None


def fundamentals_view(evidence: NormalizedEvidence, as_of: datetime) -> MarketFundamentals | None:
    resolver = OperandResolver(evidence, as_of)
    revenue = resolver.quarterly("revenue")[-FUNDAMENTAL_QUARTERS:]
    if not revenue:
        return None
    gross = {f.period.end: f for f in resolver.quarterly("gross_profit")}
    quarters: list[FundamentalQuarter] = []
    source_ids: list[str] = []
    for fact in revenue:
        period = fact.period
        label = period.label or (
            f"{period.fiscal_period} {period.fiscal_year}"
            if period.fiscal_period and period.fiscal_year
            else short_date(period.end)
        )
        margin = None
        gp = gross.get(period.end)
        if gp is not None and fact.value > 0:
            margin = round(gp.value / fact.value * 100, 2)
            if gp.source_id not in source_ids:
                source_ids.append(gp.source_id)
        if fact.source_id not in source_ids:
            source_ids.append(fact.source_id)
        quarters.append(
            FundamentalQuarter(
                label=label,
                end=period.end.isoformat() if period.end else "",
                revenue=fact.value,
                gross_margin_pct=margin,
            )
        )
    return MarketFundamentals(currency="USD", quarters=quarters, source_ids=source_ids)


def market_series(evidence: NormalizedEvidence, symbol: str, name: str) -> list[MarketSeries]:
    """Company plus benchmark series for the final result (only called with price display on)."""
    out: list[MarketSeries] = []
    if evidence.prices is not None and evidence.prices.points:
        out.append(series_payload(evidence.prices, "company", symbol, name))
    for ref in evidence.benchmark_refs:
        series = evidence.benchmarks.get(ref.role) or evidence.benchmarks.get(ref.symbol)
        if series is not None and series.points:
            out.append(series_payload(series, ref.role, ref.symbol, ref.name))
    return out
