"""Regenerate the synthetic "apple" research fixture set deterministically.

Usage::

    .venv/bin/python -m bayanalytics.research.fixtures_build [tests/fixtures/research/apple]

Everything written here is FIXTURE DATA: the company identity (Apple Inc., CIK 320193, AAPL)
is real because identity resolution is real, but every number, filing accession, price bar
and article is invented. Files carry ``"fixture": true`` / ``<meta name="bay-fixture">``.

Generated: prices/*.csv (seeded random walks), edgar/companyfacts_CIK0000320193.json,
edgar/submissions_CIK0000320193.json, edgar/company_tickers.json, pages.json, searches.json.
The HTML bodies under news/ and edgar/ are hand-written and only referenced from pages.json.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

CIK = 320193
FIXTURE_END = date(2026, 9, 25)  # last business day in the price CSVs
BUSINESS_DAYS = 520


# --------------------------------------------------------------------------------------
# prices
# --------------------------------------------------------------------------------------


def business_days_ending(end: date, count: int) -> list[date]:
    days: list[date] = []
    cursor = end
    while len(days) < count:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    days.reverse()
    return days


def random_walk_csv(
    path: Path, *, seed: int, start: float, drift: float, vol: float, base_volume: int
) -> None:
    rng = random.Random(seed)
    price = start
    rows: list[list[str]] = []
    for day in business_days_ending(FIXTURE_END, BUSINESS_DAYS):
        daily_return = drift + vol * rng.gauss(0.0, 1.0)
        close = price * math.exp(daily_return)
        open_ = price * (1.0 + 0.25 * vol * rng.gauss(0.0, 1.0))
        high = max(open_, close) * (1.0 + abs(rng.gauss(0.0, 1.0)) * 0.4 * vol)
        low = min(open_, close) * (1.0 - abs(rng.gauss(0.0, 1.0)) * 0.4 * vol)
        volume = int(base_volume * (0.7 + 0.6 * rng.random()))
        rows.append(
            [
                day.isoformat(),
                f"{open_:.2f}",
                f"{high:.2f}",
                f"{low:.2f}",
                f"{close:.2f}",
                str(volume),
            ]
        )
        price = close
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Date", "Open", "High", "Low", "Close", "Volume"])
        writer.writerows(rows)


# --------------------------------------------------------------------------------------
# EDGAR fiscal calendar (fiscal year end 09-30; quarters end late Dec / Mar / Jun / Sep)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Period:
    fy: int
    fp: str  # Q1..Q3 or FY
    start: date
    end: date
    filed: date
    accession: str
    form: str
    primary_document: str


FISCAL_PERIODS: tuple[Period, ...] = (
    Period(
        2024,
        "Q1",
        date(2023, 10, 1),
        date(2023, 12, 30),
        date(2024, 2, 2),
        "0000320193-24-000006",
        "10-Q",
        "aapl-20231230.htm",
    ),
    Period(
        2024,
        "Q2",
        date(2023, 12, 31),
        date(2024, 3, 30),
        date(2024, 5, 3),
        "0000320193-24-000069",
        "10-Q",
        "aapl-20240330.htm",
    ),
    Period(
        2024,
        "Q3",
        date(2024, 3, 31),
        date(2024, 6, 29),
        date(2024, 8, 2),
        "0000320193-24-000081",
        "10-Q",
        "aapl-20240629.htm",
    ),
    Period(
        2024,
        "FY",
        date(2023, 10, 1),
        date(2024, 9, 28),
        date(2024, 11, 1),
        "0000320193-24-000123",
        "10-K",
        "aapl-20240928.htm",
    ),
    Period(
        2025,
        "Q1",
        date(2024, 9, 29),
        date(2024, 12, 28),
        date(2025, 1, 31),
        "0000320193-25-000008",
        "10-Q",
        "aapl-20241228.htm",
    ),
    Period(
        2025,
        "Q2",
        date(2024, 12, 29),
        date(2025, 3, 29),
        date(2025, 5, 2),
        "0000320193-25-000057",
        "10-Q",
        "aapl-20250329.htm",
    ),
    Period(
        2025,
        "Q3",
        date(2025, 3, 30),
        date(2025, 6, 28),
        date(2025, 8, 1),
        "0000320193-25-000073",
        "10-Q",
        "aapl-20250628.htm",
    ),
    Period(
        2025,
        "FY",
        date(2024, 9, 29),
        date(2025, 9, 27),
        date(2025, 10, 31),
        "0000320193-25-000123",
        "10-K",
        "aapl-20250927.htm",
    ),
    Period(
        2026,
        "Q1",
        date(2025, 9, 28),
        date(2025, 12, 27),
        date(2026, 1, 30),
        "0000320193-26-000010",
        "10-Q",
        "aapl-20251227.htm",
    ),
    Period(
        2026,
        "Q2",
        date(2025, 12, 28),
        date(2026, 3, 28),
        date(2026, 5, 1),
        "0000320193-26-000052",
        "10-Q",
        "aapl-20260328.htm",
    ),
    Period(
        2026,
        "Q3",
        date(2026, 3, 29),
        date(2026, 6, 27),
        date(2026, 7, 31),
        "0000320193-26-000079",
        "10-Q",
        "aapl-20260627.htm",
    ),
    # Filed after the fixture as_of (2026-09-26): exercises the leakage guard.
    Period(
        2026,
        "FY",
        date(2025, 9, 28),
        date(2026, 9, 26),
        date(2026, 10, 30),
        "0000320193-26-000121",
        "10-K",
        "aapl-20260926.htm",
    ),
)

EIGHT_KS: tuple[tuple[date, str, str], ...] = (
    (date(2026, 7, 30), "0000320193-26-000077", "aapl-20260730.htm"),
    (date(2026, 4, 30), "0000320193-26-000050", "aapl-20260430.htm"),
    (date(2026, 1, 29), "0000320193-26-000008", "aapl-20260129.htm"),
    (date(2025, 10, 30), "0000320193-25-000121", "aapl-20251030.htm"),
)

# Fixture quarterly magnitudes (USD billions) for FY2024 by fiscal quarter; +4.5 % per year.
BASE_REVENUE = {"Q1": 95.0, "Q2": 85.0, "Q3": 80.0, "Q4": 88.0}
GROWTH = 1.045
GROSS_MARGIN = 0.45
OPERATING_MARGIN = 0.305
NET_MARGIN = 0.25
OCF_RATIO = 0.29
CAPEX_B = {"Q1": 2.4, "Q2": 2.7, "Q3": 2.6, "Q4": 3.0}
DILUTED_SHARES_START = 15.40e9  # declines 0.7 % per quarter (buybacks)
SHARES_OUT_START = 15.30e9


def quarter_metrics(fy: int, quarter: str, q_index: int) -> dict[str, float]:
    scale = GROWTH ** (fy - 2024)
    revenue = BASE_REVENUE[quarter] * 1e9 * scale
    diluted = DILUTED_SHARES_START * (0.993**q_index)
    net_income = revenue * NET_MARGIN
    return {
        "Revenues": round(revenue),
        "GrossProfit": round(revenue * GROSS_MARGIN),
        "OperatingIncomeLoss": round(revenue * OPERATING_MARGIN),
        "NetIncomeLoss": round(net_income),
        "NetCashProvidedByUsedInOperatingActivities": round(revenue * OCF_RATIO),
        "PaymentsToAcquirePropertyPlantAndEquipment": round(CAPEX_B[quarter] * 1e9 * scale),
        "EarningsPerShareDiluted": round(net_income / diluted, 2),
        "_diluted_shares": diluted,
    }


def calendar_frame(day: date, instant: bool = False) -> str:
    quarter = (day.month - 1) // 3 + 1
    return f"CY{day.year}Q{quarter}{'I' if instant else ''}"


def build_companyfacts() -> dict:
    units: dict[str, list[dict]] = {
        name: [] for name in quarter_metrics(2024, "Q1", 0) if not name.startswith("_")
    }
    shares_rows: list[dict] = []
    quarters_by_fy: dict[int, list[tuple[str, dict[str, float], Period | None]]] = {}
    q_index = 0
    for period in FISCAL_PERIODS:
        if period.fp == "FY":
            continue
        metrics = quarter_metrics(period.fy, period.fp, q_index)
        quarters_by_fy.setdefault(period.fy, []).append((period.fp, metrics, period))
        q_index += 1
        # 3-month rows
        for name, value in metrics.items():
            if name.startswith("_"):
                continue
            units[name].append(
                _row(value, period.start, period.end, period, frame=calendar_frame(period.end))
            )
        # year-to-date rows for Q2 / Q3
        if period.fp in ("Q2", "Q3"):
            ytd = quarters_by_fy[period.fy]
            fy_start = ytd[0][2].start
            for name in units:
                if name == "EarningsPerShareDiluted":
                    ni = sum(m["NetIncomeLoss"] for _, m, _ in ytd)
                    sh = sum(m["_diluted_shares"] for _, m, _ in ytd) / len(ytd)
                    value = round(ni / sh, 2)
                else:
                    value = sum(m[name] for _, m, _ in ytd)
                units[name].append(_row(value, fy_start, period.end, period, frame=None))
    # FY rows: Q1..Q3 explicit + implied Q4 (Q4 rows themselves are not tagged, as on EDGAR)
    q4_index = 3
    for period in FISCAL_PERIODS:
        if period.fp != "FY":
            continue
        q4 = quarter_metrics(period.fy, "Q4", q4_index)
        q4_index += 4
        year = [m for _, m, _ in quarters_by_fy[period.fy]] + [q4]
        for name in units:
            if name == "EarningsPerShareDiluted":
                ni = sum(m["NetIncomeLoss"] for m in year)
                sh = sum(m["_diluted_shares"] for m in year) / len(year)
                value = round(ni / sh, 2)
            else:
                value = sum(m[name] for m in year)
            units[name].append(
                _row(value, period.start, period.end, period, frame=f"CY{period.fy}")
            )
    # dei shares outstanding instants (one per filing, dated ~2 weeks before filing)
    for index, period in enumerate(FISCAL_PERIODS):
        as_of_date = period.filed - timedelta(days=14)
        shares = round(SHARES_OUT_START * (0.993**index))
        shares_rows.append(
            {
                "end": as_of_date.isoformat(),
                "val": shares,
                "accn": period.accession,
                "fy": period.fy,
                "fp": period.fp,
                "form": period.form,
                "filed": period.filed.isoformat(),
                "frame": calendar_frame(as_of_date, instant=True),
            }
        )
    us_gaap: dict[str, dict] = {}
    for name, rows in units.items():
        unit = "USD/shares" if name == "EarningsPerShareDiluted" else "USD"
        us_gaap[name] = {
            "label": f"Fixture {name}",
            "description": f"Synthetic fixture values for {name}; not real company data.",
            "units": {unit: rows},
        }
    # Duplicate revenue concept with identical values: parse_company_facts must keep only the
    # first concept in CONCEPT_MAP order (us-gaap:Revenues).
    us_gaap["RevenueFromContractWithCustomerExcludingAssessedTax"] = {
        "label": "Fixture RevenueFromContractWithCustomerExcludingAssessedTax",
        "description": "Same values as Revenues; exercises first-match-wins.",
        "units": {"USD": list(units["Revenues"])},
    }
    return {
        "fixture": True,
        "cik": CIK,
        "entityName": "Apple Inc.",
        "facts": {
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "label": "Fixture Entity Common Stock, Shares Outstanding",
                    "description": "Synthetic fixture values; not real company data.",
                    "units": {"shares": shares_rows},
                }
            },
            "us-gaap": us_gaap,
        },
    }


def _row(value: float, start: date, end: date, period: Period, frame: str | None) -> dict:
    row = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "val": value,
        "accn": period.accession,
        "fy": period.fy,
        "fp": period.fp,
        "form": period.form,
        "filed": period.filed.isoformat(),
    }
    if frame:
        row["frame"] = frame
    return row


def build_submissions() -> dict:
    filings: list[tuple[date, date, str, str, str, str]] = []
    for period in FISCAL_PERIODS:
        desc = "10-K" if period.form == "10-K" else "10-Q"
        filings.append(
            (period.filed, period.end, period.form, period.accession, period.primary_document, desc)
        )
    for filed, accession, doc in EIGHT_KS:
        filings.append((filed, filed, "8-K", accession, doc, "8-K"))
    filings.append(
        (
            date(2026, 1, 9),
            date(2026, 2, 24),
            "DEF 14A",
            "0001308179-26-000005",
            "aapl-proxy2026.htm",
            "DEF 14A",
        )
    )
    filings.append(
        (
            date(2026, 8, 4),
            date(2026, 8, 3),
            "4",
            "0000320193-26-000082",
            "xslF345X05/wk-form4_1.xml",
            "FORM 4",
        )
    )
    filings.sort(key=lambda f: f[0], reverse=True)
    recent = {
        "accessionNumber": [f[3] for f in filings],
        "filingDate": [f[0].isoformat() for f in filings],
        "reportDate": [f[1].isoformat() for f in filings],
        "acceptanceDateTime": [f"{f[0].isoformat()}T21:30:00.000Z" for f in filings],
        "act": ["34" for _ in filings],
        "form": [f[2] for f in filings],
        "fileNumber": ["001-36743" for _ in filings],
        "items": ["" for _ in filings],
        "size": [1000000 for _ in filings],
        "isXBRL": [1 if f[2] in ("10-K", "10-Q") else 0 for f in filings],
        "isInlineXBRL": [1 if f[2] in ("10-K", "10-Q") else 0 for f in filings],
        "primaryDocument": [f[4] for f in filings],
        "primaryDocDescription": [f[5] for f in filings],
    }
    return {
        "fixture": True,
        "cik": str(CIK),
        "entityType": "operating",
        "sic": "3571",
        "sicDescription": "Electronic Computers",
        "name": "Apple Inc.",
        "tickers": ["AAPL"],
        "exchanges": ["Nasdaq"],
        "ein": "000000000",
        "fiscalYearEnd": "0930",
        "stateOfIncorporation": "CA",
        "formerNames": [
            {
                "name": "APPLE COMPUTER INC",
                "from": "1994-01-26T00:00:00.000Z",
                "to": "2007-01-04T00:00:00.000Z",
            },
            {
                "name": "APPLE COMPUTER INC/ FA",
                "from": "1997-07-28T00:00:00.000Z",
                "to": "1997-07-28T00:00:00.000Z",
            },
        ],
        "filings": {"recent": recent, "files": []},
    }


def build_company_tickers() -> dict:
    return {
        "fixture": True,
        "companies": [
            {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc.", "exchange": "NASDAQ"},
            {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP", "exchange": "NASDAQ"},
            {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP", "exchange": "NASDAQ"},
            {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc.", "exchange": "NASDAQ"},
            {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc.", "exchange": "NASDAQ"},
            {
                "cik_str": 6201,
                "ticker": "AAL",
                "title": "American Airlines Group Inc.",
                "exchange": "NASDAQ",
            },
            {"cik_str": 4962, "ticker": "AXP", "title": "AMERICAN EXPRESS CO", "exchange": "NYSE"},
        ],
    }


# --------------------------------------------------------------------------------------
# pages.json / searches.json
# --------------------------------------------------------------------------------------

ARCHIVE = "https://www.sec.gov/Archives/edgar/data/320193"
NEWS = [
    # url, body file, search publishedDate (None -> only the page carries the date), snippet
    (
        "https://www.cnbc.com/2026/09/18/fixture-apple-quarter-preview.html",
        "news/cnbc_quarter_preview.html",
        "2026-09-18T12:30:00Z",
        "Fixture preview of the September quarter: what the synthetic numbers imply.",
    ),
    (
        "https://www.cnbc.com/2026/09/18/fixture-apple-quarter-preview.html?utm_source=newsletter&utm_medium=email",
        "news/cnbc_quarter_preview.html",
        "2026-09-18T12:30:00Z",
        "Same fixture preview reached through a tracking link (canonical-URL duplicate).",
    ),
    (
        "https://www.marketwatch.com/story/fixture-apple-outlook-2026-09-12",
        "news/marketwatch_outlook.html",
        None,
        "Fixture outlook column: guidance framing for the next cycle.",
    ),
    (
        "https://www.businesswire.com/news/home/20260730001234/en/Fixture-Apple-Reports-Third-Quarter-Results",
        "news/businesswire_q3_release.html",
        "2026-07-30T20:05:00Z",
        "Fixture earnings release for the June quarter (synthetic figures).",
    ),
    (
        "https://www.fool.com/investing/2026/09/20/fixture-apple-what-to-watch/",
        "news/fool_what_to_watch.html",
        "2026-09-20",
        "Fixture commentary: three things to watch in the synthetic quarter.",
    ),
    (
        "https://www.reuters.com/technology/fixture-apple-october-event-2026-10-02/",
        "news/reuters_after_as_of.html",
        None,
        "Fixture article dated after the evaluation as_of; must be rejected by the leakage guard.",
    ),
    (
        "https://investor.example-fixture.com/news/short-notice",
        "news/ir_thin_notice.html",
        "2026-09-15",
        "Fixture investor-relations notice that is too short to be evidence.",
    ),
    (
        "https://www.prnewswire.com/news-releases/fixture-apple-reports-third-quarter-results-301234567.html",
        "news/prnewswire_q3_syndicated.html",
        "2026-07-30T20:07:00Z",
        "Syndicated copy of the fixture earnings release (content-hash duplicate).",
    ),
]


def build_pages() -> dict:
    pages: dict[str, dict] = {
        "https://www.sec.gov/files/company_tickers.json": {
            "status": 200,
            "content_type": "application/json",
            "body_file": "edgar/company_tickers.json",
        },
        f"https://data.sec.gov/submissions/CIK{CIK:010d}.json": {
            "status": 200,
            "content_type": "application/json",
            "body_file": f"edgar/submissions_CIK{CIK:010d}.json",
        },
        f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK:010d}.json": {
            "status": 200,
            "content_type": "application/json",
            "body_file": f"edgar/companyfacts_CIK{CIK:010d}.json",
        },
        f"{ARCHIVE}/000032019325000123/aapl-20250927.htm": {
            "status": 200,
            "content_type": "text/html",
            "body_file": "edgar/aapl-20250927_10k_primary.html",
        },
        f"{ARCHIVE}/000032019326000079/aapl-20260627.htm": {
            "status": 200,
            "content_type": "text/html",
            "body_file": "edgar/aapl-20260627_10q_primary.html",
        },
        f"{ARCHIVE}/000032019326000052/aapl-20260328.htm": {
            "status": 200,
            "content_type": "text/html",
            "body_file": "edgar/aapl-20260328_10q_primary.html",
        },
        f"{ARCHIVE}/000032019326000077/aapl-20260730.htm": {
            "status": 200,
            "content_type": "text/html",
            "body_file": "edgar/aapl-20260730_8k_primary.html",
        },
        "https://stooq.com/q/d/l/?s=aapl.us&i=d": {
            "status": 200,
            "content_type": "text/csv",
            "body_file": "prices/aapl.us.csv",
        },
        "https://stooq.com/q/d/l/?s=^spx&i=d": {
            "status": 200,
            "content_type": "text/csv",
            "body_file": "prices/spx.csv",
        },
        "https://stooq.com/q/d/l/?s=xlk.us&i=d": {
            "status": 200,
            "content_type": "text/csv",
            "body_file": "prices/xlk.us.csv",
        },
        "https://www.wsj.com/fixture/paywalled-apple-story": {
            "status": 403,
            "content_type": "text/html",
            "paywalled": True,
        },
    }
    seen: set[str] = set()
    for url, body_file, _, _ in NEWS:
        if body_file in seen and "utm_" in url:
            continue
        seen.add(body_file)
        pages[url] = {
            "status": 200,
            "content_type": "text/html; charset=utf-8",
            "body_file": body_file,
        }
    return {"fixture": True, "pages": pages}


def build_searches() -> dict:
    results = []
    for url, _, published, snippet in NEWS:
        item = {
            "url": url,
            "title": _title_for(url),
            "content": snippet,
            "engine": "fixture",
            "score": 1.0,
        }
        if published:
            item["publishedDate"] = published
        results.append(item)
    return {
        "fixture": True,
        "searches": {
            '"Apple Inc." earnings OR guidance OR outlook': results,
            '"Apple Inc." stock 2025 results': results[2:5],
            "*": results,
        },
    }


def _title_for(url: str) -> str:
    if "cnbc" in url:
        return "[Fixture] Apple quarter preview: what the synthetic numbers imply"
    if "marketwatch" in url:
        return "[Fixture] Opinion: the outlook for the next cycle"
    if "businesswire" in url or "prnewswire" in url:
        return "[Fixture] Apple Reports Third Quarter Results"
    if "fool" in url:
        return "[Fixture] 3 things to watch this quarter"
    if "reuters" in url:
        return "[Fixture] October event recap"
    return "[Fixture] Investor notice"


def build(out_dir: Path) -> list[Path]:
    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    random_walk_csv(
        out_dir / "prices" / "aapl.us.csv",
        seed=20260925,
        start=172.0,
        drift=0.00035,
        vol=0.016,
        base_volume=55_000_000,
    )
    random_walk_csv(
        out_dir / "prices" / "spx.csv",
        seed=5001,
        start=4400.0,
        drift=0.00030,
        vol=0.009,
        base_volume=3_500_000_000,
    )
    random_walk_csv(
        out_dir / "prices" / "xlk.us.csv",
        seed=7003,
        start=168.0,
        drift=0.00040,
        vol=0.013,
        base_volume=7_000_000,
    )
    written += [out_dir / "prices" / name for name in ("aapl.us.csv", "spx.csv", "xlk.us.csv")]
    for name, payload in (
        (f"edgar/companyfacts_CIK{CIK:010d}.json", build_companyfacts()),
        (f"edgar/submissions_CIK{CIK:010d}.json", build_submissions()),
        ("edgar/company_tickers.json", build_company_tickers()),
        ("pages.json", build_pages()),
        ("searches.json", build_searches()),
    ):
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    default = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "research" / "apple"
    out_dir = Path(args[0]) if args else default
    for path in build(out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
