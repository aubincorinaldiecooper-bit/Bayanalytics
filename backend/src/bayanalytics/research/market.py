"""Price series and reported figures read from the pages a web search returned.

``MarketData`` accumulates, for one analysis, what the research runner read out of search
results (``research.tables`` parses, Laya chooses among the tables and lines that exist,
``research.selection``): the company's daily price series and the broad-market series, each
from its own page, and the figures each page reported. It turns them into what the rest of the
pipeline consumes, always keeping the page a number came from:

* the ``market.series`` payloads and the result's ``market.series``;
* rows in the ``normalization.facts.build_facts`` contract (one per page, metric and period,
  ``source_id`` = the page), so two pages that disagree become a ``Conflict`` there;
* the uncertainty lines that say what was found (figures from how many pages, which figures
  appear on only one page) or that search returned no page with the data;
* the small ``preview`` tables on ``research.source_found`` and the quarterly fundamentals view.

Nothing here reaches the network, and nothing is filled in: a missing series stays missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from bayanalytics.calculations.operands import OperandResolver
from bayanalytics.research.provider import EvidenceRecord
from bayanalytics.research.sources import domain_of
from bayanalytics.research.tables import (
    INSTANT,
    PER_SHARE_METRICS,
    QUARTER,
    Figure,
    FigureSet,
    PriceTable,
    RawTable,
)
from bayanalytics.schemas.evidence import NormalizedEvidence, PriceSeries, SourceRecord
from bayanalytics.schemas.results import FundamentalQuarter, MarketFundamentals, MarketSeries

COMPANY = "company"
BROAD_MARKET = "broad_market"
BROAD_MARKET_SYMBOL = "S&P 500"
BROAD_MARKET_NAME = "S&P 500 index"
MAX_SERIES_POINTS = 1300
PREVIEW_ROWS = 4
FUNDAMENTAL_QUARTERS = 8
PERIOD_ALIGN_DAYS = 7
"""Two pages dating the same period within this many days report the same period (one writes
the fiscal quarter end, another the calendar month end)."""
MAX_SINGLE_PAGE_METRICS = 10

METRIC_LABELS: dict[str, str] = {
    "revenue": "revenue",
    "gross_profit": "gross profit",
    "operating_income": "operating income",
    "net_income": "net income",
    "eps_diluted": "diluted EPS",
    "eps_basic": "basic EPS",
    "operating_cash_flow": "operating cash flow",
    "capex": "capital expenditure",
    "free_cash_flow": "free cash flow",
    "shares_outstanding": "shares outstanding",
}
_PREVIEW_METRICS = (
    "revenue",
    "gross_profit",
    "net_income",
    "eps_diluted",
    "operating_cash_flow",
    "free_cash_flow",
    "operating_income",
    "shares_outstanding",
)
_CURRENCY_SIGNS = {"USD": "$", "EUR": "\u20ac", "GBP": "\u00a3", "JPY": "\u00a5"}
_BROAD_MARKET = re.compile(
    r"s\s*&\s*p\s*500|\bsp\s?500\b|\bspx\b|\bgspc\b|standard\s*(?:&|and)\s*poor'?s\s*500", re.I
)


# --------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------


def short_date(value: date | datetime | str | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        try:
            value = date.fromisoformat(value[:10])
        except ValueError:
            return value
    return f"{value:%b} {value.day}, {value.year}"


def money(value: float, currency: str | None, *, per_share: bool = False) -> str:
    """``$87.4B``, ``$1.48``, ``EUR 3.2M``; a share count without a sign (``15.1B``)."""
    sign = "-" if value < 0 else ""
    prefix = ""
    if currency:
        symbol = _CURRENCY_SIGNS.get(currency)
        prefix = symbol if symbol else f"{currency} "
    magnitude = abs(value)
    if per_share:
        return f"{sign}{prefix}{magnitude:,.2f}"
    for size, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= size:
            return f"{sign}{prefix}{magnitude / size:,.1f}{suffix}"
    return f"{sign}{prefix}{magnitude:,.0f}"


def figure_value(figure: Figure) -> str:
    if figure.metric in PER_SHARE_METRICS:
        return money(figure.value, figure.currency, per_share=True)
    return money(figure.value, figure.currency)


def price_preview(table: PriceTable) -> dict[str, Any]:
    """``["Date", "Close"]``, the last ``PREVIEW_ROWS`` sessions, newest first."""
    rows = [[short_date(p.date), f"{p.close:,.2f}"] for p in reversed(table.points[-PREVIEW_ROWS:])]
    return {"columns": ["Date", "Close"], "rows": rows}


def figures_preview(figures: FigureSet) -> dict[str, Any] | None:
    """``["Metric", "Period", "Value"]``: the latest period of up to ``PREVIEW_ROWS`` metrics."""
    rows: list[list[str]] = []
    for metric in _PREVIEW_METRICS:
        found = [f for f in figures.figures if f.metric == metric]
        if not found:
            continue
        quarterly = [f for f in found if f.kind in (QUARTER, INSTANT)]
        latest = max(quarterly or found, key=lambda f: f.end)
        period = latest.period_label or short_date(latest.end)
        rows.append([METRIC_LABELS[metric].capitalize(), period, figure_value(latest)])
        if len(rows) >= PREVIEW_ROWS:
            break
    return {"columns": ["Metric", "Period", "Value"], "rows": rows} if rows else None


# --------------------------------------------------------------------------------------
# which page is about what
# --------------------------------------------------------------------------------------


def _page_texts(record: EvidenceRecord, tables: list[RawTable]) -> list[str]:
    texts = [record.title, record.text[:6000]]
    for table in tables[:8]:
        texts.append(table.context)
        texts.append(" ".join(table.columns))
    return texts


def mentions_symbol(symbol: str, record: EvidenceRecord, tables: list[RawTable]) -> bool:
    """The page names the ticker (as a whole token, any class-separator spelling)."""
    variants = {symbol, symbol.replace(".", "-"), symbol.replace("-", "."), symbol.replace(".", "")}
    body = "|".join(re.escape(v) for v in sorted(variants, key=len, reverse=True) if v)
    pattern = re.compile(rf"(?<![A-Za-z0-9])(?:{body})(?![A-Za-z0-9])")
    if any(pattern.search(text or "") for text in _page_texts(record, tables)):
        return True
    url = f"{record.final_url or record.url} {record.url}"
    return re.search(rf"(?<![A-Za-z0-9])(?:{body})(?![A-Za-z0-9])", url, re.I) is not None


def mentions_broad_market(record: EvidenceRecord, tables: list[RawTable]) -> bool:
    """The page is about the S&P 500 (its name, ``SPX``, ``GSPC`` or ``SP500``)."""
    texts = [*_page_texts(record, tables), record.final_url or record.url, record.url]
    return any(_BROAD_MARKET.search(text or "") for text in texts)


# --------------------------------------------------------------------------------------
# accumulation
# --------------------------------------------------------------------------------------


@dataclass
class KeptSeries:
    role: str
    symbol: str
    name: str
    domain: str
    series: PriceSeries
    currency_stated: bool = True

    def payload(self) -> MarketSeries:
        points = [
            (p.date.isoformat(), p.open, p.high, p.low, float(p.close), p.volume)
            for p in self.series.points[-MAX_SERIES_POINTS:]
        ]
        return MarketSeries(
            role=self.role,  # type: ignore[arg-type]
            symbol=self.symbol,
            name=self.name,
            source_id=self.series.source_id,
            currency=self.series.currency,
            points=points,
        )


@dataclass
class PageFigures:
    source_id: str
    domain: str
    figures: list[Figure]


@dataclass
class MarketData:
    """What the research of one analysis read out of search results (see the module doc)."""

    series: dict[str, KeptSeries] = field(default_factory=dict)
    pages: list[PageFigures] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def note(self, text: str) -> None:
        if text not in self.notes:
            self.notes.append(text)

    # -- price series ------------------------------------------------------------------

    def offer_series(
        self,
        role: str,
        symbol: str,
        name: str,
        source: SourceRecord,
        table: PriceTable,
        exchange: str | None = None,
    ) -> MarketSeries | None:
        """Keep ``table`` as the ``role`` series when no series is kept for the role yet or it
        has more sessions than the kept one; the company and the market series never come from
        the same page. Returns the ``market.series`` payload when kept, else ``None``."""
        held = self.series.get(role)
        if held is not None and len(held.series.points) >= len(table.points):
            return None
        if any(
            kept.series.source_id == source.source_id and kept.role != role
            for kept in self.series.values()
        ):
            return None
        series = PriceSeries(
            symbol=symbol,
            source_id=source.source_id,
            points=list(table.points),
            price_type="historical_close",
            exchange=exchange,
            retrieved_at=source.retrieved_at,
            currency=table.currency,
            label=name,
        )
        kept = KeptSeries(role, symbol, name, domain_of(source.url), series, table.currency_stated)
        self.series[role] = kept
        return kept.payload()

    def prices(self) -> PriceSeries | None:
        kept = self.series.get(COMPANY)
        return kept.series if kept is not None else None

    def benchmarks(self) -> dict[str, PriceSeries]:
        return {role: kept.series for role, kept in self.series.items() if role != COMPANY}

    def series_payloads(self) -> list[MarketSeries]:
        order = (COMPANY, BROAD_MARKET)
        return [self.series[role].payload() for role in order if role in self.series]

    # -- figures -----------------------------------------------------------------------

    def add_figures(self, source: SourceRecord, figures: FigureSet, as_of: date) -> None:
        domain = domain_of(source.url)
        for note in figures.notes(as_of):
            self.note(f"{domain}: {note}")
        if figures.figures:
            self.pages.append(PageFigures(source.source_id, domain, list(figures.figures)))

    def metric_rows(self) -> list[dict[str, Any]]:
        """``{"metric": ...}`` per figure, for the research loop's operand gaps."""
        return [{"metric": f.metric} for page in self.pages for f in page.figures]

    def _canonical_ends(self) -> tuple[dict[tuple[str, date], date], list[str]]:
        """Each (kind, end) mapped to one end per period: ends within ``PERIOD_ALIGN_DAYS`` of
        each other are one period, dated as most pages date it (the earliest on a tie)."""
        reported: dict[tuple[str, date], set[str]] = {}
        for page in self.pages:
            for figure in page.figures:
                reported.setdefault((figure.kind, figure.end), set()).add(page.source_id)
        canonical: dict[tuple[str, date], date] = {}
        notes: list[str] = []
        kinds = sorted({kind for kind, _ in reported})
        for kind in kinds:
            clusters: list[list[date]] = []
            for end in sorted(end for k, end in reported if k == kind):
                if clusters and (end - clusters[-1][0]).days <= PERIOD_ALIGN_DAYS:
                    clusters[-1].append(end)
                else:
                    clusters.append([end])
            for cluster in clusters:
                chosen = min(cluster, key=lambda d: (-len(reported[(kind, d)]), d))
                for end in cluster:
                    canonical[(kind, end)] = chosen
                if len(cluster) > 1:
                    listed = ", ".join(d.isoformat() for d in cluster)
                    notes.append(
                        f"pages date one period differently ({listed}); treated as the period "
                        f"ending {chosen.isoformat()}"
                    )
        return canonical, notes

    def fact_rows(self) -> tuple[list[dict[str, Any]], list[str], dict[date, str]]:
        """Rows for ``build_facts`` (one per page, metric and period), the alignment notes, and
        the pages' own quarter labels by period end when every page that labels it agrees."""
        canonical, notes = self._canonical_ends()
        rows: list[dict[str, Any]] = []
        labels: dict[date, set[str]] = {}
        seen: set[tuple[str, str, str, date]] = set()
        for page in self.pages:
            for figure in page.figures:
                end = canonical[(figure.kind, figure.end)]
                key = (page.source_id, figure.metric, figure.kind, end)
                if key in seen:
                    continue
                seen.add(key)
                if figure.kind == QUARTER and figure.period_label:
                    labels.setdefault(end, set()).add(figure.period_label)
                rows.append(
                    {
                        "concept": f"web:{figure.label}",
                        "metric": figure.metric,
                        "value": figure.value,
                        "raw_value": figure.raw_value,
                        "unit": figure.unit,
                        "start": None,
                        "end": end.isoformat(),
                        "fy": None,
                        "fp": None,
                        "form": None,
                        "filed": None,
                        "accn": None,
                        "frame": None,
                        "source_id": page.source_id,
                        "basis": "unknown",
                        "currency": figure.currency or "USD",
                        "period_kind": None if figure.kind == INSTANT else figure.kind,
                        "extraction_method": f"web_{figure.origin}",
                    }
                )
        agreed = {end: next(iter(found)) for end, found in labels.items() if len(found) == 1}
        return rows, notes, agreed

    # -- what the result says about it -------------------------------------------------

    def _domains(self, source_ids: list[str]) -> list[str]:
        by_source = {page.source_id: page.domain for page in self.pages}
        out: list[str] = []
        for source_id in source_ids:
            domain = by_source.get(source_id, "")
            if domain and domain not in out:
                out.append(domain)
        return out

    def data_notes(self, symbol: str, text_pages: int, rows: list[dict[str, Any]]) -> list[str]:
        """What the analysis rests on, in plain words: where the figures and prices came from,
        which figures only one page reported, or that search returned no page with them."""
        company = self.series.get(COMPANY)
        market = self.series.get(BROAD_MARKET)
        noun = "page" if text_pages == 1 else "pages"
        if not rows and company is None and market is None:
            return [
                "Web search returned no page with price history or quarterly figures: this "
                f"assessment is based only on the text of {text_pages} web {noun} found by "
                "search."
            ]
        notes: list[str] = []
        sources = list(dict.fromkeys(str(row["source_id"]) for row in rows))
        if sources:
            count = "one web page" if len(sources) == 1 else f"{len(sources)} web pages"
            notes.append(
                f"Financial figures come from {count} found by search "
                f"({', '.join(self._domains(sources))})."
            )
        else:
            notes.append(
                f"Web search returned no page with quarterly figures for {symbol}: calculations "
                "that need them report the missing inputs."
            )
        if company is not None:
            points = company.series.points
            notes.append(
                f"Daily prices for {symbol} come from one web page ({company.domain}): "
                f"{len(points)} sessions from {points[0].date.isoformat()} to "
                f"{points[-1].date.isoformat()}; returns are price returns (no dividends) as "
                "the page's closes give them."
            )
            if not company.currency_stated:
                notes.append(f"{company.domain} states no currency for its prices; USD assumed.")
        else:
            notes.append(
                f"Web search returned no page with daily prices for {symbol}: price-based "
                "calculations report the missing inputs."
            )
        if market is not None:
            notes.append(f"S&P 500 daily prices come from another web page ({market.domain}).")
        else:
            notes.append(
                "Web search returned no page with S&P 500 daily prices: comparisons with the "
                "market are unavailable."
            )
        single = self._single_page(rows)
        if single:
            notes.append(
                "Found on only one page, not corroborated by a second: " + "; ".join(single) + "."
            )
        return notes

    def _single_page(self, rows: list[dict[str, Any]]) -> list[str]:
        pages: dict[tuple[str, str, str], set[str]] = {}
        for row in rows:
            key = (str(row["metric"]), str(row["period_kind"]), str(row["end"]))
            pages.setdefault(key, set()).add(str(row["source_id"]))
        counts: dict[tuple[str, str], int] = {}
        for (metric, _kind, _end), sources in pages.items():
            if len(sources) == 1:
                (source_id,) = sources
                counts[(metric, source_id)] = counts.get((metric, source_id), 0) + 1
        by_source = {page.source_id: page.domain for page in self.pages}
        order = list(METRIC_LABELS)
        lines = []
        for (metric, source_id), n in sorted(
            counts.items(), key=lambda item: (order.index(item[0][0]), item[0][1])
        ):
            periods = "period" if n == 1 else "periods"
            lines.append(f"{METRIC_LABELS[metric]} ({n} {periods}, {by_source.get(source_id, '')})")
        return lines[:MAX_SINGLE_PAGE_METRICS]


# --------------------------------------------------------------------------------------
# quarterly fundamentals view
# --------------------------------------------------------------------------------------


def fundamentals_view(
    evidence: NormalizedEvidence, as_of: datetime, labels: dict[date, str] | None = None
) -> MarketFundamentals | None:
    """Up to ``FUNDAMENTAL_QUARTERS`` quarters of revenue and gross margin (%) from the
    normalized facts, oldest first, with the pages they came from; ``None`` without quarterly
    revenue. A quarter is labelled as the pages labelled it ("Q3 2026") when they agree, else
    by its end date."""
    labels = labels or {}
    resolver = OperandResolver(evidence, as_of)
    revenue = resolver.quarterly("revenue")[-FUNDAMENTAL_QUARTERS:]
    if not revenue:
        return None
    gross = {f.period.end: f for f in resolver.quarterly("gross_profit")}
    quarters: list[FundamentalQuarter] = []
    source_ids: list[str] = []
    for fact in revenue:
        end = fact.period.end
        margin = None
        gross_fact = gross.get(end)
        if gross_fact is not None and fact.value > 0 and gross_fact.currency == fact.currency:
            margin = round(gross_fact.value / fact.value * 100, 2)
            if gross_fact.source_id not in source_ids:
                source_ids.append(gross_fact.source_id)
        if fact.source_id not in source_ids:
            source_ids.append(fact.source_id)
        quarters.append(
            FundamentalQuarter(
                label=labels.get(end) if end is not None and end in labels else short_date(end),
                end=end.isoformat() if end else "",
                revenue=fact.value,
                gross_margin_pct=margin,
            )
        )
    return MarketFundamentals(
        currency=revenue[-1].currency or "USD", quarters=quarters, source_ids=source_ids
    )
