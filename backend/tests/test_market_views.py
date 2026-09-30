"""Market views: quarterly fundamentals from normalized facts, result series and previews."""

from __future__ import annotations

from datetime import UTC, date, datetime

from bayanalytics.research.market import (
    facts_preview,
    fundamentals_view,
    market_series,
    money,
    short_date,
)
from bayanalytics.schemas.evidence import (
    BenchmarkRef,
    NormalizedEvidence,
    NormalizedFact,
    Period,
    PricePoint,
    PriceSeries,
)

AS_OF = datetime(2026, 9, 29, tzinfo=UTC)


def _quarter(
    metric: str, fy: int, fp: str, end: date, value: float, fact_id: str
) -> NormalizedFact:
    return NormalizedFact(
        fact_id=fact_id,
        metric=metric,
        value=value,
        unit="USD",
        currency="USD",
        period=Period(kind="fiscal_quarter", fiscal_year=fy, fiscal_period=fp, end=end),
        basis="gaap",
        source_id="src_facts",
    )


def _series(symbol: str, closes: list[float]) -> PriceSeries:
    points = [
        PricePoint(
            date=date(2026, 9, 21 + i), open=c - 1, high=c + 1, low=c - 2, close=c, volume=100.0
        )
        for i, c in enumerate(closes)
    ]
    return PriceSeries(symbol=symbol, source_id=f"src_{symbol}", points=points, retrieved_at=AS_OF)


def test_fundamentals_view_pairs_revenue_and_gross_margin_by_quarter() -> None:
    facts = [
        _quarter("revenue", 2026, "Q1", date(2025, 12, 27), 800.0e6, "f1"),
        _quarter("revenue", 2026, "Q2", date(2026, 3, 28), 900.0e6, "f2"),
        _quarter("gross_profit", 2026, "Q2", date(2026, 3, 28), 270.0e6, "f3"),
        _quarter("revenue", 2026, "Q3", date(2026, 6, 27), 1000.0e6, "f4"),
        # published after as_of by period end: must not appear
        _quarter("revenue", 2026, "Q4", date(2026, 10, 3), 1.2e9, "f5"),
    ]
    evidence = NormalizedEvidence(symbol="NWND", as_of=AS_OF, facts=facts)
    view = fundamentals_view(evidence, AS_OF)
    assert view is not None
    assert [q.label for q in view.quarters] == ["Q1 2026", "Q2 2026", "Q3 2026"]
    assert [q.revenue for q in view.quarters] == [800.0e6, 900.0e6, 1000.0e6]
    assert [q.gross_margin_pct for q in view.quarters] == [None, 30.0, None]
    assert view.quarters[1].end == "2026-03-28"
    assert view.source_ids == ["src_facts"]


def test_fundamentals_view_is_none_without_quarterly_revenue() -> None:
    evidence = NormalizedEvidence(symbol="NWND", as_of=AS_OF, facts=[])
    assert fundamentals_view(evidence, AS_OF) is None


def test_market_series_orders_company_then_benchmarks() -> None:
    evidence = NormalizedEvidence(
        symbol="NWND",
        as_of=AS_OF,
        prices=_series("nwnd.us", [40.0, 41.0]),
        benchmarks={"^SPX": _series("^spx", [6000.0, 6010.0]), "XLP": _series("xlp.us", [80.0])},
        benchmark_refs=[
            BenchmarkRef(role="broad_market", symbol="^SPX", name="S&P 500"),
            BenchmarkRef(role="sector", symbol="XLP", name="Consumer staples (XLP)"),
        ],
    )
    series = market_series(evidence, "NWND", "Northwind Foods, Inc.")
    assert [(s.role, s.symbol) for s in series] == [
        ("company", "NWND"),
        ("broad_market", "^SPX"),
        ("sector", "XLP"),
    ]
    assert series[0].points[-1] == ("2026-09-22", 40.0, 42.0, 39.0, 41.0, 100.0)


def test_facts_preview_uses_latest_periodic_report_values() -> None:
    rows = [
        {
            "metric": "revenue",
            "value": 8.0e8,
            "unit": "USD",
            "end": "2026-03-28",
            "filed": "2026-05-07",
            "fy": 2026,
            "fp": "Q2",
            "form": "10-Q",
        },
        {
            "metric": "revenue",
            "value": 9.012e8,
            "unit": "USD",
            "end": "2026-06-27",
            "filed": "2026-08-06",
            "fy": 2026,
            "fp": "Q3",
            "form": "10-Q",
        },
        {
            "metric": "revenue",
            "value": 9.9e8,
            "unit": "USD",
            "end": "2026-06-27",
            "filed": "2026-08-07",
            "fy": 2026,
            "fp": "Q3",
            "form": "8-K",
        },
        {
            "metric": "eps_diluted",
            "value": 0.62,
            "unit": "USD/shares",
            "end": "2026-06-27",
            "filed": "2026-08-06",
            "fy": 2026,
            "fp": "Q3",
            "form": "10-Q",
        },
    ]
    preview = facts_preview(rows)
    assert preview == {
        "columns": ["Metric", "Period", "Value"],
        "rows": [["Revenue", "Q3 2026", "$901.2M"], ["Diluted EPS", "Q3 2026", "$0.62"]],
    }
    assert facts_preview([]) is None


def test_formatting_helpers() -> None:
    assert money(2.41e9) == "$2.4B" and money(-3.5e6) == "-$3.5M" and money(950.0) == "$950"
    assert short_date(date(2026, 8, 6)) == "Aug 6, 2026" and short_date(None) == "—"
    assert short_date("2025-11-14") == "Nov 14, 2025"
