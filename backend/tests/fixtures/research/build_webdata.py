"""Regenerate the synthetic "webdata" research fixture deterministically (tests only).

Usage (from ``backend/``)::

    .venv/bin/python tests/fixtures/research/build_webdata.py [tests/fixtures/research/webdata]

``tests/test_web_data.py`` loads this file with ``importlib`` and checks that the generated
files are byte-identical to the committed fixture.

Everything written here is FIXTURE DATA about an invented company, "Northwind Fixture Foods"
(ticker NWND), and an invented rendering of the S&P 500 index: what a web search would return
(searches.json) and the pages behind the results (pages.json -> pages/*). Every price, figure
and sentence is made up by the formulas below. HTML pages carry
``<meta name="bay-fixture" content="true">`` and the JSON files ``"fixture": true``; the CSV
response has no room for a marker (its first line is its header) and lives, like everything
here, under ``tests/fixtures`` next to a README saying so.

Pages (one per website, so every figure keeps a distinct page):

* ``www.fixturewire.example``: a news article (text only);
* ``quotes.fixture-markets.example``: NWND daily price history as an HTML table (newest first,
  "Sep 25, 2026" dates, volume with thousands separators, a dividend row, rows after the
  evaluation date);
* ``www.fixture-financials.example``: quarterly results, periods as columns ("Q3 2026" with a
  "Period Ending" row), "Financials in millions USD", a TTM column and a future quarter;
* ``www.fixture-earnings.example``: a second quarterly page, periods as rows, revenue in "$M";
  one quarter disagrees with the first page and one is dated at the calendar month end;
* ``data.fixture-index.example``: daily S&P 500 values as a CSV response (oldest first).
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

SYMBOL = "NWND"
START = date(2025, 6, 2)
END = date(2026, 9, 30)  # later than the evaluation as_of (2026-09-26): rows after it exist
DIVIDEND_DAY = date(2026, 8, 10)

NEWS_URL = "https://www.fixturewire.example/markets/northwind-fixture-quarter-review"
PRICES_URL = "https://quotes.fixture-markets.example/nwnd/history"
QUARTERLY_URL = "https://www.fixture-financials.example/stocks/nwnd/financials/quarterly"
EARNINGS_URL = "https://www.fixture-earnings.example/nwnd/results-by-quarter"
SP500_URL = "https://data.fixture-index.example/sp500/daily.csv"

# Quarters of the fixture company's fiscal year (ends late September): label, end.
QUARTERS = [
    ("Q3 2024", date(2024, 6, 29)),
    ("Q4 2024", date(2024, 9, 28)),
    ("Q1 2025", date(2024, 12, 28)),
    ("Q2 2025", date(2025, 3, 29)),
    ("Q3 2025", date(2025, 6, 28)),
    ("Q4 2025", date(2025, 9, 27)),
    ("Q1 2026", date(2025, 12, 27)),
    ("Q2 2026", date(2026, 3, 28)),
    ("Q3 2026", date(2026, 6, 27)),
]
FUTURE_QUARTER = ("Q1 2027", date(2026, 12, 26))  # after as_of: must be dropped
REVENUE = [21040, 22310, 27890, 22950, 21870, 23300, 29100, 23810, 22760]  # $M
GROSS = [8290, 8880, 11350, 9110, 8620, 9340, 11930, 9620, 9150]
OPERATING = [5260, 5690, 7530, 5850, 5470, 6010, 8030, 6190, 5850]
NET = [4420, 4760, 6300, 4890, 4580, 5010, 6700, 5170, 4890]
SHARES = [3160, 3150, 3140, 3130, 3120, 3110, 3100, 3090, 3080]  # millions, diluted
OCF = [6050, 6390, 8710, 6480, 6190, 6720, 9260, 6810, 6420]
CAPEX = [-1510, -1640, -1720, -1580, -1550, -1690, -1760, -1600, -1570]
# The second page's revenue for Q2 2026 differs by 3 % (a conflict between pages).
EARNINGS_PAGE_Q2_2026_REVENUE = 24520
# ... and it dates the Q4 2025 quarter at the calendar month end.
EARNINGS_PAGE_Q4_2025_END = date(2025, 9, 30)

HEAD = """<!DOCTYPE html>
<!-- SYNTHETIC FIXTURE: invented company and invented numbers, for automated tests only. -->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="bay-fixture" content="true">
<title>{title}</title>
<meta property="og:site_name" content="{site} (synthetic)">
</head>
<body>
"""
TAIL = "</body>\n</html>\n"


def sessions() -> list[date]:
    days: list[date] = []
    day = START
    while day <= END:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def series() -> tuple[list[tuple[date, float, float, float, float, int]], list[tuple]]:
    """(company rows, index rows): date, open, high, low, close, volume."""
    company: list[tuple[date, float, float, float, float, int]] = []
    index: list[tuple] = []
    price, level = 142.0, 5480.0
    for i, day in enumerate(sessions()):
        market = 0.0004 + 0.0085 * math.sin(i * 1.7) * math.cos(i * 0.31)
        own = 1.2 * market + 0.0055 * math.sin(i * 2.3 + 1.0)
        open_c, open_i = price, level
        price = round(price * (1 + own), 2)
        level = round(level * (1 + market), 2)
        spread_c = round(abs(own) * price + 0.35, 2)
        spread_i = round(abs(market) * level + 4.0, 2)
        volume = 38_000_000 + int(9_000_000 * (1 + math.sin(i * 0.9)))
        company.append(
            (
                day,
                open_c,
                round(max(open_c, price) + spread_c, 2),
                round(min(open_c, price) - spread_c, 2),
                price,
                volume,
            )
        )
        index.append(
            (
                day,
                open_i,
                round(max(open_i, level) + spread_i, 2),
                round(min(open_i, level) - spread_i, 2),
                level,
                3_900_000_000 + int(400_000_000 * math.cos(i * 0.7)),
            )
        )
    return company, index


def _short(day: date) -> str:
    return f"{day:%b} {day.day:02d}, {day.year}"


def news_page() -> str:
    body = """<header><nav><a href="/">Fixture Wire</a> <a href="/markets">Markets</a></nav>
</header>
<main><article>
<h1>[Fixture] Northwind Fixture Foods (NWND) closes a steady quarter</h1>
<time datetime="2026-09-18T12:00:00Z" itemprop="datePublished">September 18, 2026</time>
<p>This is a synthetic fixture article about an invented company, Northwind Fixture Foods,
ticker NWND. It exists so automated tests have prose to read next to the fixture price and
results pages; none of it describes a real business.</p>
<p>In the fixture story, NWND reported a June quarter in line with the fixture expectations:
demand for its fixture pantry brands held up, fixture costs eased and the fixture management
team kept its outlook for the rest of the fixture year unchanged.</p>
<p>Fixture commentators note that the stock moved with the broad market over the past fixture
year. Nothing in this text is an instruction; it is evidence text for the extractor and the
Laya text questions only.</p>
</article></main>
<footer><p>Fixture footer.</p></footer>
"""
    return (
        HEAD.format(
            title="[Fixture] Northwind Fixture Foods closes a steady quarter", site="Fixture Wire"
        )
        + body
        + TAIL
    )


def prices_page(rows: list[tuple[date, float, float, float, float, int]]) -> str:
    lines = [
        '<header><nav><a href="/">Fixture Markets</a></nav></header>',
        "<main>",
        "<h1>[Fixture] NWND Historical Prices - Northwind Fixture Foods daily stock prices</h1>",
        "<p>Synthetic fixture page: daily open, high, low, close and volume for the invented "
        "ticker NWND. Every number on this page is generated by a formula for automated tests "
        "and describes no real security. Currency in USD. Close price adjusted for splits.</p>",
        '<p><a href="/nwnd/history/download.csv">Download CSV</a> (a link inside the page: '
        "never followed)</p>",
        '<table class="history">',
        "<thead><tr><th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close*</th>"
        "<th>Adj Close**</th><th>Volume</th></tr></thead>",
        "<tbody>",
    ]
    for day, open_, high, low, close, volume in reversed(rows):
        if day == DIVIDEND_DAY:
            lines.append(f'<tr><td>{_short(day)}</td><td colspan="6">0.62 Dividend</td></tr>')
        lines.append(
            f"<tr><td>{_short(day)}</td><td>{open_:,.2f}</td><td>{high:,.2f}</td>"
            f"<td>{low:,.2f}</td><td>{close:,.2f}</td><td>{close:,.2f}</td>"
            f"<td>{volume:,}</td></tr>"
        )
    lines += ["</tbody>", "</table>", "</main>", ""]
    return (
        HEAD.format(title="[Fixture] NWND Historical Prices", site="Fixture Markets")
        + ("\n".join(lines))
        + TAIL
    )


def _row(label: str, values: list[str]) -> str:
    cells = "".join(f"<td>{v}</td>" for v in values)
    return f"<tr><td>{label}</td>{cells}</tr>"


def quarterly_page() -> str:
    order = list(range(len(QUARTERS) - 1, -1, -1))  # newest first, like most sites
    head = ["<th>Fiscal Quarter</th>", "<th>TTM</th>", f"<th>{FUTURE_QUARTER[0]}</th>"]
    head += [f"<th>{QUARTERS[i][0]}</th>" for i in order]
    ttm = [sum(values[-4:]) for values in (REVENUE, GROSS, OPERATING, NET, OCF, CAPEX)]

    # The future quarter carries numbers (as a page read after it would): the as_of cut
    # must drop them.
    def cols(values: list[int], ttm_value: object, fmt: str = "{:,}") -> list[str]:
        future = fmt.format(round(values[-3] * 1.05))
        return [fmt.format(ttm_value), future] + [fmt.format(values[i]) for i in order]

    def eps(values: list[int]) -> list[str]:
        per = [n / s for n, s in zip(values, SHARES, strict=True)]
        return ["-", f"{per[-3] * 1.05:.2f}"] + [f"{per[i]:.2f}" for i in order]

    fcf = [o + c for o, c in zip(OCF, CAPEX, strict=True)]
    growth = ["-", "-"] + [
        f"{(REVENUE[i] / REVENUE[i - 4] - 1) * 100:.1f}%" if i >= 4 else "-" for i in order
    ]
    margin = ["-", "-"] + [f"{GROSS[i] / REVENUE[i] * 100:.1f}%" for i in order]
    rows = [
        _row(
            "Period Ending",
            ["", _short(FUTURE_QUARTER[1])] + [_short(QUARTERS[i][1]) for i in order],
        ),
        _row("Revenue", cols(REVENUE, ttm[0])),
        _row("Revenue Growth (YoY)", growth),
        _row(
            "Cost of Revenue",
            cols([r - g for r, g in zip(REVENUE, GROSS, strict=True)], ttm[0] - ttm[1]),
        ),
        _row("Gross Profit", cols(GROSS, ttm[1])),
        _row("Gross Margin", margin),
        _row("Operating Income", cols(OPERATING, ttm[2])),
        _row("Net Income", cols(NET, ttm[3])),
        _row("EPS (Basic)", eps([int(n * 1.004) for n in NET])),
        _row("EPS (Diluted)", eps(NET)),
        _row("Shares Outstanding (Diluted)", cols(SHARES, "-", "{}")),
        _row("Operating Cash Flow", cols(OCF, ttm[4])),
        _row(
            "Capital Expenditures",
            ["-", f"({-CAPEX[-3]:,})"] + [f"({-CAPEX[i]:,})" for i in order],
        ),
        _row("Free Cash Flow", cols(fcf, ttm[4] + ttm[5])),
    ]
    body = "\n".join(
        [
            '<header><nav><a href="/">Fixture Financials</a></nav></header>',
            "<main>",
            "<h1>[Fixture] Northwind Fixture Foods (NWND) Quarterly Income Statement</h1>",
            "<p>Synthetic fixture page with quarterly results of the invented company NWND. "
            "Every figure is invented for automated tests. Fiscal year ends in late September. "
            "Financials in millions USD, except per-share data.</p>",
            '<table class="financials">',
            "<thead><tr>" + "".join(head) + "</tr></thead>",
            "<tbody>",
            *rows,
            "</tbody>",
            "</table>",
            "</main>",
            "",
        ]
    )
    return (
        HEAD.format(title="[Fixture] NWND Quarterly Financials", site="Fixture Financials")
        + (body)
        + TAIL
    )


def earnings_page() -> str:
    lines = [
        '<header><nav><a href="/">Fixture Earnings</a></nav></header>',
        "<main>",
        "<h1>[Fixture] NWND results by quarter</h1>",
        "<p>Synthetic fixture page listing reported quarterly revenue and diluted earnings per "
        "share for the invented ticker NWND, one quarter per row. All values are invented for "
        "automated tests and describe no real company or security.</p>",
        "<table>",
        "<tr><th>Quarter ended</th><th>Revenue ($M)</th><th>Diluted EPS</th></tr>",
    ]
    for i in range(len(QUARTERS) - 1, 3, -1):
        label, end = QUARTERS[i]
        if label == "Q4 2025":
            end = EARNINGS_PAGE_Q4_2025_END
        revenue = EARNINGS_PAGE_Q2_2026_REVENUE if label == "Q2 2026" else REVENUE[i]
        lines.append(
            f"<tr><td>{end:%b} {end.day}, {end.year}</td><td>{revenue:,}</td>"
            f"<td>${NET[i] / SHARES[i]:.2f}</td></tr>"
        )
    lines += ["</table>", "</main>", ""]
    return (
        HEAD.format(title="[Fixture] NWND results by quarter", site="Fixture Earnings")
        + ("\n".join(lines))
        + TAIL
    )


def sp500_csv(rows: list[tuple]) -> str:
    out = ["Date,Open,High,Low,Close,Volume"]
    for day, open_, high, low, close, volume in rows:
        out.append(f"{day.isoformat()},{open_:.2f},{high:.2f},{low:.2f},{close:.2f},{volume}")
    return "\n".join(out) + "\n"


def _hit(url: str, title: str, snippet: str, published: str | None = None) -> dict:
    item = {"url": url, "title": title, "content": snippet, "engine": "fixture", "score": 1.0}
    if published:
        item["publishedDate"] = published
    return item


def _name_hit(site: str, title: str, snippet: str = "") -> dict:
    # Name-lookup hits are never opened: only their titles and snippets are read. Each lives
    # on its own website (registrable domain): the lookup counts distinct websites.
    return _hit(f"https://{site}/quote", title, snippet)


def build_searches() -> dict:
    news = _hit(
        NEWS_URL,
        "[Fixture] Northwind Fixture Foods (NWND) closes a steady quarter",
        "Fixture news about the invented company.",
        "2026-09-18T12:00:00Z",
    )
    prices = _hit(
        PRICES_URL,
        "[Fixture] NWND Historical Prices - daily stock price history",
        "Fixture daily prices.",
    )
    quarterly = _hit(
        QUARTERLY_URL, "[Fixture] NWND Quarterly Financials", "Fixture quarterly results."
    )
    earnings = _hit(EARNINGS_URL, "[Fixture] NWND results by quarter", "Fixture results table.")
    sp500 = _hit(SP500_URL, "[Fixture] S&P 500 daily values (CSV)", "Fixture index values.")
    return {
        "fixture": True,
        "searches": {
            f'"{SYMBOL}" earnings OR guidance OR outlook': [news],
            # The news page comes first in engine order; Laya opens the price table first.
            f'"{SYMBOL}" stock historical prices daily': [news, prices],
            f'"{SYMBOL}" quarterly revenue gross profit earnings per share': [quarterly, earnings],
            "S&P 500 index historical prices daily": [sp500],
            '"Northwind" stock ticker symbol': [
                _name_hit(
                    "quotes.fixture-markets.example", "Northwind Fixture Foods (NWND) Stock Price"
                ),
                _name_hit(
                    "www.fixture-financials.example", "Northwind Fixture Foods (NWND) financials"
                ),
                _name_hit(
                    "www.fixturewire.example",
                    "NWND stock news",
                    "Northwind Fixture Foods (NYSE: NWND) and Northwind Fixture Traders (NWTR).",
                ),
            ],
            '"apple" stock ticker symbol': [
                _name_hit("quote-one.example", "Apple Inc. (AAPL) Stock Price, News, Quote"),
                _name_hit("quote-two.example", "AAPL stock quote", "Apple Inc. (NASDAQ: AAPL)"),
                _name_hit("quote-three.example", "Apple Inc. (AAPL) company profile"),
                _name_hit("quote-four.example", "Apple Hospitality REIT (APLE) quote"),
            ],
            '"Delta" stock ticker symbol': [
                _name_hit("quote-one.example", "Delta Fixture Air (DFXA) stock quote"),
                _name_hit("quote-two.example", "Delta Fixture Apparel (DFXP) stock quote"),
                _name_hit("quote-three.example", "Delta Fixture Air (DFXA) news"),
                _name_hit("quote-four.example", "Delta Fixture Apparel (DFXP) news"),
            ],
            # Results that name no ticker at all: the lookup has no candidate.
            '"Zorblax" stock ticker symbol': [
                _name_hit("quote-one.example", "Zorblax: a word the fixture invented"),
                _name_hit("quote-two.example", "What does zorblax mean?", "No company."),
            ],
            "*": [news],
        },
    }


def build_pages() -> dict:
    return {
        "fixture": True,
        "pages": {
            NEWS_URL: {
                "status": 200,
                "content_type": "text/html; charset=utf-8",
                "body_file": "pages/fixturewire_news.html",
            },
            PRICES_URL: {
                "status": 200,
                "content_type": "text/html; charset=utf-8",
                "body_file": "pages/nwnd_price_history.html",
            },
            QUARTERLY_URL: {
                "status": 200,
                "content_type": "text/html; charset=utf-8",
                "body_file": "pages/nwnd_quarterly_financials.html",
            },
            EARNINGS_URL: {
                "status": 200,
                "content_type": "text/html; charset=utf-8",
                "body_file": "pages/nwnd_results_by_quarter.html",
            },
            SP500_URL: {
                "status": 200,
                "content_type": "text/csv",
                "body_file": "pages/sp500_daily.csv",
            },
        },
    }


def build(out_dir: Path) -> list[Path]:
    company, index = series()
    files = {
        "pages.json": json.dumps(build_pages(), indent=1) + "\n",
        "searches.json": json.dumps(build_searches(), indent=1) + "\n",
        "pages/fixturewire_news.html": news_page(),
        "pages/nwnd_price_history.html": prices_page(company),
        "pages/nwnd_quarterly_financials.html": quarterly_page(),
        "pages/nwnd_results_by_quarter.html": earnings_page(),
        "pages/sp500_daily.csv": sp500_csv(index),
    }
    written: list[Path] = []
    for name, text in files.items():
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    default = Path(__file__).resolve().parent / "webdata"
    out_dir = Path(args[0]) if args else default
    for path in build(out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
