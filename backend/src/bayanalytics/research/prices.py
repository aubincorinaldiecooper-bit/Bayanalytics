"""Stooq daily price history and benchmark selection (AGENT.md sections 33 and 34).

Stooq serves split-adjusted (not dividend-adjusted) daily bars as CSV. The series is labelled
``latest_close`` with the last session date preserved so the normalization layer never
confuses it with an intraday quote. Redistribution is metadata-only (Stooq terms).
"""

from __future__ import annotations

import csv
import io
from datetime import UTC, date, datetime, timedelta

from bayanalytics.research.dates import ensure_utc
from bayanalytics.research.fetch import PRICE_CSV_TTL_S, PageFetcher
from bayanalytics.research.provider import ResearchProviderError
from bayanalytics.research.sources import canonical_url
from bayanalytics.schemas.common import stable_id
from bayanalytics.schemas.evidence import BenchmarkRef, PricePoint, PriceSeries, SourceRecord

STOOQ_DAILY_URL = "https://stooq.com/q/d/l/?s={symbol}&i=d"
STOOQ_TERMS_NOTE = "Stooq terms: personal use, verify before redistribution"
_EXCHANGE_SUFFIX: dict[str, str] = {
    "LSE": "uk",
    "LON": "uk",
    "TSX": "ca",
    "TSE": "ca",
    "XETRA": "de",
    "FWB": "de",
    "HKEX": "hk",
    "TYO": "jp",
}

BROAD_MARKET = BenchmarkRef(role="broad_market", symbol="^SPX", name="S&P 500")

# (sic_low, sic_high, etf, name) - first matching row wins, so the narrower carve-outs come
# before the wide ranges they are cut out of (3711/3714 autos -> XLY before 3700-3799 -> XLI,
# 6500-6599/6798 real estate -> XLRE before 6000-6799 -> XLF).
SECTOR_SIC_TABLE: tuple[tuple[int, int, str, str], ...] = (
    (3570, 3579, "XLK", "Technology Select Sector SPDR"),
    (3670, 3679, "XLK", "Technology Select Sector SPDR"),
    (7370, 7379, "XLK", "Technology Select Sector SPDR"),
    (2830, 2836, "XLV", "Health Care Select Sector SPDR"),
    (8000, 8099, "XLV", "Health Care Select Sector SPDR"),
    (3840, 3851, "XLV", "Health Care Select Sector SPDR"),
    (6500, 6599, "XLRE", "Real Estate Select Sector SPDR"),
    (6798, 6798, "XLRE", "Real Estate Select Sector SPDR"),
    (6000, 6799, "XLF", "Financial Select Sector SPDR"),
    (1300, 1389, "XLE", "Energy Select Sector SPDR"),
    (2900, 2999, "XLE", "Energy Select Sector SPDR"),
    (3711, 3711, "XLY", "Consumer Discretionary Select Sector SPDR"),
    (3714, 3714, "XLY", "Consumer Discretionary Select Sector SPDR"),
    (5200, 5999, "XLY", "Consumer Discretionary Select Sector SPDR"),
    (2000, 2099, "XLP", "Consumer Staples Select Sector SPDR"),
    (2100, 2100, "XLP", "Consumer Staples Select Sector SPDR"),
    (5140, 5149, "XLP", "Consumer Staples Select Sector SPDR"),
    (3500, 3569, "XLI", "Industrial Select Sector SPDR"),
    (3700, 3799, "XLI", "Industrial Select Sector SPDR"),
    (4210, 4231, "XLI", "Industrial Select Sector SPDR"),
    (8700, 8748, "XLI", "Industrial Select Sector SPDR"),
    (2800, 2829, "XLB", "Materials Select Sector SPDR"),
    (3300, 3399, "XLB", "Materials Select Sector SPDR"),
    (1000, 1099, "XLB", "Materials Select Sector SPDR"),
    (4900, 4991, "XLU", "Utilities Select Sector SPDR"),
    (4800, 4899, "XLC", "Communication Services Select Sector SPDR"),
    (7810, 7841, "XLC", "Communication Services Select Sector SPDR"),
)

BENCHMARKS: dict[str, str] = {"^SPX": "S&P 500"}
for _low, _high, _etf, _name in SECTOR_SIC_TABLE:
    BENCHMARKS.setdefault(_etf, f"{_name} ({_etf})")


def to_stooq_symbol(ticker: str, exchange: str | None = None) -> str:
    """AAPL -> aapl.us, BRK-B / BRK.B -> brk-b.us, ^SPX -> ^spx."""
    symbol = ticker.strip().lower().replace(".", "-")
    if symbol.startswith("^"):
        return symbol
    if "." in ticker and ticker.rsplit(".", 1)[-1].lower() in {"us", "uk", "ca", "de", "hk", "jp"}:
        return ticker.strip().lower()
    suffix = _EXCHANGE_SUFFIX.get((exchange or "").upper(), "us")
    return f"{symbol}.{suffix}"


def benchmark_stooq_symbol(benchmark_symbol: str) -> str:
    return to_stooq_symbol(benchmark_symbol)


def select_benchmarks(sic: str | None) -> list[BenchmarkRef]:
    """Always the broad market; plus one sector ETF when the SIC maps (reason records the SIC)."""
    refs = [BROAD_MARKET.model_copy(update={"reason": "default broad-market benchmark"})]
    code: int | None = None
    if sic and str(sic).strip().isdigit():
        code = int(str(sic).strip())
    if code is not None:
        for low, high, etf, name in SECTOR_SIC_TABLE:
            if low <= code <= high:
                refs.append(
                    BenchmarkRef(
                        role="sector",
                        symbol=etf,
                        name=f"{name} ({etf})",
                        reason=f"SIC {code} maps to {etf} via range {low}-{high}",
                    )
                )
                break
    return refs


def parse_stooq_csv(body: str) -> list[PricePoint]:
    text = body.strip()
    if not text:
        raise ResearchProviderError("Stooq returned an empty body")
    lowered = text[:200].lower()
    if lowered.startswith("no data") or "no data" in lowered.splitlines()[0]:
        raise ResearchProviderError("Stooq: no data for symbol")
    if "exceeded the daily hits limit" in lowered:
        raise ResearchProviderError("Stooq: daily hits limit exceeded")
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if not header or header[0].strip().lower() != "date":
        raise ResearchProviderError("Stooq: unexpected CSV header")
    columns = [h.strip().lower() for h in header]
    points: list[PricePoint] = []
    for row in reader:
        if not row or len(row) < len(columns):
            continue
        record = dict(zip(columns, (cell.strip() for cell in row), strict=False))
        try:
            day = date.fromisoformat(record["date"])
            close = float(record["close"])
        except (KeyError, ValueError):
            continue
        points.append(
            PricePoint(
                date=day,
                open=_float_or_none(record.get("open")),
                high=_float_or_none(record.get("high")),
                low=_float_or_none(record.get("low")),
                close=close,
                volume=_float_or_none(record.get("volume")),
            )
        )
    points.sort(key=lambda p: p.date)
    return points


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


class StooqPrices:
    def __init__(self, fetcher: PageFetcher) -> None:
        self._fetcher = fetcher

    def url_for(self, symbol_stooq: str) -> str:
        return STOOQ_DAILY_URL.format(symbol=symbol_stooq)

    async def daily(
        self,
        symbol_stooq: str,
        as_of: datetime | date,
        days: int,
        *,
        label: str = "",
        exchange: str | None = None,
    ) -> PriceSeries:
        """Daily bars ending at the last session on or before ``as_of`` (leakage guard)."""
        series, _ = await self.daily_with_source(
            symbol_stooq, as_of, days, label=label, exchange=exchange
        )
        return series

    async def daily_with_source(
        self,
        symbol_stooq: str,
        as_of: datetime | date,
        days: int,
        *,
        label: str = "",
        exchange: str | None = None,
        intent: str = "retrieve_price_history",
    ) -> tuple[PriceSeries, SourceRecord]:
        """Like ``daily`` but also returns the Stooq ``SourceRecord`` (market_data)."""
        cutoff = ensure_utc(as_of).date() if isinstance(as_of, datetime) else as_of
        start = cutoff - timedelta(days=max(1, days))
        url = self.url_for(symbol_stooq)
        page = await self._fetcher.open(url, ttl_s=PRICE_CSV_TTL_S, accept="text/csv,text/plain")
        points = [p for p in parse_stooq_csv(page.body) if start <= p.date <= cutoff]
        if not points:
            raise ResearchProviderError(f"Stooq: no rows for {symbol_stooq} up to {cutoff}")
        session_date = points[-1].date
        published = datetime(session_date.year, session_date.month, session_date.day, tzinfo=UTC)
        source = SourceRecord(
            source_id=stable_id("src", canonical_url(url)),
            url=url,
            title=f"Stooq daily prices {symbol_stooq}",
            publisher="Stooq",
            source_type="market_data",
            published_at=published,
            retrieved_at=page.fetched_at,
            symbol=symbol_stooq,
            excerpt=(
                f"{len(points)} daily bars {points[0].date.isoformat()}.."
                f"{session_date.isoformat()}, last close {points[-1].close}"
            ),
            extraction_method="csv",
            freshness="current" if (cutoff - session_date).days <= 4 else "stale",
            redistribution="metadata_only",
            terms_note=STOOQ_TERMS_NOTE,
            research_intent=intent,
            metadata={"rows": len(points), "from_cache": page.from_cache, "split_adjusted": True},
        )
        series = PriceSeries(
            symbol=symbol_stooq,
            source_id=source.source_id,
            points=points,
            price_type="latest_close",
            exchange=exchange,
            session_date=session_date,
            retrieved_at=page.fetched_at,
            split_adjusted=True,
            dividend_adjusted=False,
            currency=_currency_for(symbol_stooq),
            label=label or symbol_stooq,
        )
        return series, source


def _currency_for(symbol_stooq: str) -> str:
    suffix = symbol_stooq.rsplit(".", 1)[-1] if "." in symbol_stooq else "us"
    return {"us": "USD", "uk": "GBP", "ca": "CAD", "de": "EUR", "hk": "HKD", "jp": "JPY"}.get(
        suffix, "USD"
    )
