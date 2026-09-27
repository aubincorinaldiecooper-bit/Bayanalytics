"""EDGAR (seed, submissions, company facts), Stooq and benchmark tests against fixtures."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from bayanalytics.config import Settings
from bayanalytics.research.edgar import (
    CONCEPT_MAP,
    FACT_FORMS,
    EdgarClient,
    fiscal_period_label,
    load_company_tickers_seed,
    parse_company_facts,
    parse_company_tickers,
    select_filings,
)
from bayanalytics.research.fixture_provider import FixtureFetcher
from bayanalytics.research.prices import (
    BENCHMARKS,
    StooqPrices,
    benchmark_stooq_symbol,
    parse_stooq_csv,
    select_benchmarks,
    to_stooq_symbol,
)
from bayanalytics.research.provider import ResearchProviderError

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "research" / "apple"
AS_OF = datetime(2026, 9, 26, tzinfo=UTC)
ROW_KEYS = {
    "concept",
    "metric",
    "value",
    "unit",
    "start",
    "end",
    "fy",
    "fp",
    "form",
    "filed",
    "accn",
    "frame",
    "source_id",
    "basis",
    "currency",
}


@pytest.fixture
def settings() -> Settings:
    return Settings(research_contact_email="dev@example.com", research_min_request_interval_s=0.0)


@pytest.fixture
def edgar(settings: Settings) -> EdgarClient:
    return EdgarClient(FixtureFetcher(FIXTURE_DIR), settings)


# --------------------------------------------------------------------------------------
# seed / tickers
# --------------------------------------------------------------------------------------


def test_seed_loads_and_contains_large_caps() -> None:
    seed = load_company_tickers_seed()
    assert len(seed) >= 75
    by_ticker = {row["ticker"]: row for row in seed}
    assert by_ticker["AAPL"]["cik_str"] == 320193
    assert by_ticker["MSFT"]["cik_str"] == 789019
    assert by_ticker["NVDA"]["cik_str"] == 1045810
    assert by_ticker["GOOGL"]["cik_str"] == by_ticker["GOOG"]["cik_str"] == 1652044
    assert by_ticker["XYZ"]["aliases"] == ["SQ"]
    assert by_ticker["AAL"]["cik_str"] == 6201 and by_ticker["AXP"]["cik_str"] == 4962
    for row in seed:
        assert isinstance(row["cik_str"], int) and row["cik_str"] > 0
        assert row["ticker"] == row["ticker"].upper()
        assert row["title"]
        assert row.get("exchange") in (None, "NASDAQ", "NYSE")


def test_parse_company_tickers_edgar_shape() -> None:
    rows = parse_company_tickers(
        {
            "1": {"cik_str": 789019, "ticker": "msft", "title": "MICROSOFT"},
            "0": {"cik_str": "320193", "ticker": "AAPL", "title": "Apple"},
            "bad": {"x": 1},
        }
    )
    assert [r["ticker"] for r in rows] == ["AAPL", "MSFT"]
    assert rows[0]["cik_str"] == 320193
    with pytest.raises(ResearchProviderError):
        parse_company_tickers("nope")


async def test_company_tickers_live_and_seed_fallback(
    edgar: EdgarClient, settings: Settings, tmp_path: Path
) -> None:
    rows = await edgar.company_tickers()
    assert edgar.seed_fallback is False
    assert {r["ticker"] for r in rows} >= {"AAPL", "MSFT", "AAL", "AXP"}
    (tmp_path / "pages.json").write_text('{"fixture": true, "pages": {}}')
    offline = EdgarClient(FixtureFetcher(tmp_path), settings)
    rows = await offline.company_tickers()
    assert offline.seed_fallback is True
    assert any(r["ticker"] == "AAPL" and r["cik_str"] == 320193 for r in rows)


# --------------------------------------------------------------------------------------
# submissions
# --------------------------------------------------------------------------------------


async def test_submissions_parsing_and_filing_urls(edgar: EdgarClient) -> None:
    subs = await edgar.submissions("320193")
    assert subs.cik == 320193
    assert subs.name == "Apple Inc."
    assert subs.tickers == ["AAPL"] and subs.exchanges == ["Nasdaq"]
    assert subs.sic == "3571" and subs.sic_description == "Electronic Computers"
    assert subs.fiscal_year_end == "0930"
    assert subs.state_of_incorporation == "CA"
    assert "APPLE COMPUTER INC" in subs.name_history
    assert subs.url == "https://data.sec.gov/submissions/CIK0000320193.json"
    ten_k = next(
        f for f in subs.filings if f.form == "10-K" and f.filing_date == date(2025, 10, 31)
    )
    assert ten_k.accession == "0000320193-25-000123"
    assert ten_k.report_date == date(2025, 9, 27)
    assert ten_k.primary_document == "aapl-20250927.htm"
    assert (
        ten_k.url
        == "https://www.sec.gov/Archives/edgar/data/320193/000032019325000123/aapl-20250927.htm"
    )
    assert ten_k.index_url == (
        "https://www.sec.gov/Archives/edgar/data/320193/000032019325000123/0000320193-25-000123-index.htm"
    )
    assert {f.form for f in subs.filings} >= {"10-K", "10-Q", "8-K", "4", "DEF 14A"}


async def test_latest_filings_respects_forms_limit_and_as_of(edgar: EdgarClient) -> None:
    subs = await edgar.submissions(320193)
    latest_k = subs.latest_filings(("10-K",), limit=1, as_of=AS_OF)
    assert [f.filing_date for f in latest_k] == [date(2025, 10, 31)]
    future_k = subs.latest_filings(("10-K",), limit=1)
    assert future_k[0].filing_date == date(2026, 10, 30)
    default = subs.latest_filings(limit=6, as_of=AS_OF)
    assert all(f.form in ("10-K", "10-Q", "8-K") for f in default)
    assert all(f.filing_date <= AS_OF.date() for f in default)
    assert [f.filing_date for f in default] == sorted(
        (f.filing_date for f in default), reverse=True
    )
    selected = select_filings(subs, ("10-K", "10-Q", "8-K"), limit=6, as_of=AS_OF)
    assert len(selected) == 6
    assert sum(1 for f in selected if f.form == "10-K") == 1
    assert sum(1 for f in selected if f.form == "10-Q") >= 2
    assert selected[0].filing_date >= selected[-1].filing_date


def test_fiscal_period_label() -> None:
    assert fiscal_period_label(date(2025, 9, 27), "0930", "10-K") == "FY2025"
    assert fiscal_period_label(date(2025, 12, 27), "0930", "10-Q") == "Q1 FY2026"
    assert fiscal_period_label(date(2026, 3, 28), "0930", "10-Q") == "Q2 FY2026"
    assert fiscal_period_label(date(2026, 6, 27), "0930", "10-Q") == "Q3 FY2026"
    assert fiscal_period_label(date(2025, 6, 30), "1231", "10-Q") == "Q2 FY2025"
    assert fiscal_period_label(date(2025, 12, 31), "1231", "10-K") == "FY2025"
    assert fiscal_period_label(date(2026, 1, 31), "0131", "10-K") == "FY2026"
    assert fiscal_period_label(date(2026, 7, 30), "0930", "8-K") == "2026-07-30"
    assert fiscal_period_label(date(2026, 7, 30), None, "10-Q") == "2026-07-30"
    assert fiscal_period_label(None, "0930", "10-K") is None


# --------------------------------------------------------------------------------------
# company facts
# --------------------------------------------------------------------------------------


async def test_company_facts_rows_follow_contract(edgar: EdgarClient) -> None:
    facts = await edgar.company_facts(320193, as_of=AS_OF, symbol="AAPL")
    assert facts.cik == 320193 and facts.entity_name == "Apple Inc."
    assert facts.rows and len(facts) == len(facts.rows)
    assert facts.rows_filtered_after_as_of == 8  # FY2026 10-K rows (7 concepts + 1 dei instant)
    for row in facts.rows:
        assert set(row) == ROW_KEYS
        assert row["form"] in FACT_FORMS
        assert row["filed"] <= "2026-09-26"
        assert row["basis"] == "gaap"
        assert row["source_id"] == facts.source.source_id
        assert isinstance(row["value"], float)
    metrics = {row["metric"] for row in facts.rows}
    assert metrics == {
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "eps_diluted",
        "operating_cash_flow",
        "capex",
        "shares_outstanding",
    }
    revenue = [r for r in facts.rows if r["metric"] == "revenue"]
    assert {r["concept"] for r in revenue} == {"us-gaap:Revenues"}  # first match wins
    assert facts.concepts_used["revenue"] == "us-gaap:Revenues"
    assert all(r["unit"] == "USD" and r["currency"] == "USD" for r in revenue)
    q1 = next(r for r in revenue if r["fp"] == "Q1" and r["fy"] == 2026)
    assert q1["start"] == "2025-09-28" and q1["end"] == "2025-12-27"
    assert (
        q1["frame"] == "CY2025Q4" and q1["form"] == "10-Q" and q1["accn"] == "0000320193-26-000010"
    )
    fy = next(r for r in revenue if r["fp"] == "FY" and r["fy"] == 2025)
    assert fy["start"] == "2024-09-29" and fy["end"] == "2025-09-27" and fy["form"] == "10-K"
    eps = [r for r in facts.rows if r["metric"] == "eps_diluted"]
    assert all(r["unit"] == "USD/shares" for r in eps)
    shares = [r for r in facts.rows if r["metric"] == "shares_outstanding"]
    assert all(
        r["unit"] == "shares" and r["start"] is None and r["currency"] is None for r in shares
    )
    assert all(r["concept"] == "dei:EntityCommonStockSharesOutstanding" for r in shares)
    assert {r["fp"] for r in facts.rows} == {"Q1", "Q2", "Q3", "FY"}
    source = facts.source
    assert source.title == "SEC XBRL company facts"
    assert source.url == "https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json"
    assert source.source_type == "regulatory_filing" and source.redistribution == "allowed"
    assert source.symbol == "AAPL" and source.extraction_method == "json"
    assert source.published_at == datetime(2026, 7, 31, tzinfo=UTC)
    assert source.metadata["rows_filtered_after_as_of"] == 8


async def test_company_facts_without_as_of_keeps_future_rows(edgar: EdgarClient) -> None:
    guarded = await edgar.company_facts(320193, as_of=AS_OF)
    unguarded = await edgar.company_facts(320193)
    assert unguarded.rows_filtered_after_as_of == 0
    assert len(unguarded.rows) == len(guarded.rows) + 8
    assert any(r["filed"] == "2026-10-30" for r in unguarded.rows)


def test_parse_company_facts_filters_forms_units_and_orders_concepts() -> None:
    payload = {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "start": "2025-01-01",
                                "end": "2025-12-31",
                                "val": 100,
                                "accn": "a",
                                "fy": 2025,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2026-02-01",
                            },
                            {
                                "start": "2025-01-01",
                                "end": "2025-12-31",
                                "val": 100,
                                "accn": "b",
                                "fy": 2025,
                                "fp": "FY",
                                "form": "S-1",
                                "filed": "2026-03-01",
                            },
                        ],
                        "EUR": [
                            {"end": "2025-12-31", "val": 1, "form": "10-K", "filed": "2026-02-01"}
                        ],
                    }
                },
                "SalesRevenueNet": {
                    "units": {
                        "USD": [
                            {"end": "2025-12-31", "val": 5, "form": "10-K", "filed": "2026-02-01"}
                        ]
                    }
                },
                "LongTermDebt": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-12-31",
                                "val": 7,
                                "form": "10-K",
                                "filed": "2026-02-01",
                                "frame": "CY2025Q4I",
                            }
                        ]
                    }
                },
                "LongTermDebtCurrent": {
                    "units": {
                        "USD": [
                            {"end": "2025-12-31", "val": 2, "form": "10-K", "filed": "2026-02-01"}
                        ]
                    }
                },
                "DepreciationAndAmortization": {
                    "units": {
                        "USD": [
                            {"end": "2025-12-31", "val": 3, "form": "20-F", "filed": "2026-02-01"}
                        ]
                    }
                },
            }
        }
    }
    rows, filtered, used = parse_company_facts(payload, source_id="src_x", as_of=date(2026, 6, 30))
    assert filtered == 0
    assert used["revenue"] == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    assert [r["metric"] for r in rows] == sorted(r["metric"] for r in rows)
    revenue = [r for r in rows if r["metric"] == "revenue"]
    assert len(revenue) == 1 and revenue[0]["form"] == "10-K" and revenue[0]["value"] == 100.0
    assert [r["value"] for r in rows if r["metric"] == "total_debt"] == [7.0]
    assert [r["value"] for r in rows if r["metric"] == "long_term_debt_current"] == [2.0]
    assert [r["form"] for r in rows if r["metric"] == "depreciation_amortization"] == ["20-F"]
    assert [r["frame"] for r in rows if r["metric"] == "total_debt"] == ["CY2025Q4I"]
    # Everything is filed after this as_of: every candidate concept is examined (a fully
    # filtered concept does not claim its metric), so all five dated 10-K/20-F rows count.
    rows, filtered, used = parse_company_facts(payload, source_id="s", as_of=date(2026, 1, 15))
    assert rows == [] and filtered == 5 and used == {}
    assert [m for _, m in CONCEPT_MAP][:3] == ["revenue", "revenue", "revenue"]


async def test_company_facts_unknown_cik_raises_provider_error(edgar: EdgarClient) -> None:
    with pytest.raises(ResearchProviderError):
        await edgar.company_facts(999999)
    with pytest.raises(ResearchProviderError):
        await edgar.submissions("not-a-cik")


async def test_filing_source_record_and_excerpt(edgar: EdgarClient) -> None:
    subs = await edgar.submissions(320193)
    ten_k = subs.latest_filings(("10-K",), 1, AS_OF)[0]
    excerpt = await edgar.fetch_filing_excerpt(ten_k)
    assert 0 < len(excerpt) <= 600
    assert "FIXTURE" in excerpt
    source = edgar.filing_source_record(
        ten_k, symbol="AAPL", as_of=AS_OF, fiscal_year_end=subs.fiscal_year_end, excerpt=excerpt
    )
    assert source.title == "10-K filed 2025-10-31"
    assert source.fiscal_period == "FY2025"
    assert source.source_type == "regulatory_filing" and source.publisher == "SEC EDGAR"
    assert source.url == ten_k.index_url
    assert source.metadata["primary_document_url"] == ten_k.url
    assert source.freshness == "stale"
    assert len(source.excerpt) <= 600


# --------------------------------------------------------------------------------------
# stooq / benchmarks
# --------------------------------------------------------------------------------------


def test_to_stooq_symbol() -> None:
    assert to_stooq_symbol("AAPL", "NASDAQ") == "aapl.us"
    assert to_stooq_symbol("BRK-B", "NYSE") == "brk-b.us"
    assert to_stooq_symbol("BRK.B") == "brk-b.us"
    assert to_stooq_symbol("^SPX") == "^spx"
    assert to_stooq_symbol("XLK") == "xlk.us"
    assert to_stooq_symbol("SHEL", "LSE") == "shel.uk"
    assert benchmark_stooq_symbol("^SPX") == "^spx" and benchmark_stooq_symbol("XLK") == "xlk.us"


def test_parse_stooq_csv_and_errors() -> None:
    points = parse_stooq_csv(
        "Date,Open,High,Low,Close,Volume\n2026-09-25,1,2,0.5,1.5,100\n2026-09-24,1,2,0.5,1.4,\n"
    )
    assert [p.date for p in points] == [date(2026, 9, 24), date(2026, 9, 25)]
    assert points[0].volume is None and points[1].volume == 100.0
    with pytest.raises(ResearchProviderError, match="no data"):
        parse_stooq_csv("No data\n")
    with pytest.raises(ResearchProviderError):
        parse_stooq_csv("")
    with pytest.raises(ResearchProviderError, match="header"):
        parse_stooq_csv("<html>oops</html>")


async def test_stooq_daily_from_fixture_and_as_of_cut() -> None:
    prices = StooqPrices(FixtureFetcher(FIXTURE_DIR))
    series, source = await prices.daily_with_source(
        "aapl.us", AS_OF, 5 * 366, label="AAPL", exchange="NASDAQ"
    )
    assert series.symbol == "aapl.us" and series.label == "AAPL" and series.exchange == "NASDAQ"
    assert len(series.points) == 520
    assert series.session_date == date(2026, 9, 25)
    assert series.latest is not None and series.latest.date == date(2026, 9, 25)
    assert series.price_type == "latest_close"
    assert series.split_adjusted is True and series.dividend_adjusted is False
    assert series.currency == "USD" and series.exchange_timezone == "America/New_York"
    assert series.source_id == source.source_id
    assert source.source_type == "market_data" and source.redistribution == "metadata_only"
    assert source.terms_note == "Stooq terms: personal use, verify before redistribution"
    assert source.url == "https://stooq.com/q/d/l/?s=aapl.us&i=d"
    assert source.published_at == datetime(2026, 9, 25, tzinfo=UTC)
    cut = await prices.daily("aapl.us", datetime(2026, 9, 1, tzinfo=UTC), 30)
    assert cut.points[-1].date <= date(2026, 9, 1)
    assert cut.points[0].date >= date(2026, 8, 2)
    assert cut.session_date == cut.points[-1].date
    assert all(p.date <= date(2026, 9, 1) for p in cut.points)
    spx = await prices.daily("^spx", AS_OF, 400)
    assert spx.symbol == "^spx" and len(spx.points) > 250
    with pytest.raises(ResearchProviderError):
        await prices.daily("nope.us", AS_OF, 30)


@pytest.mark.parametrize(
    ("sic", "etf"),
    [
        (None, None),
        ("", None),
        ("9999", None),
        ("3571", "XLK"),
        ("3674", "XLK"),
        ("7372", "XLK"),
        ("2834", "XLV"),
        ("8071", "XLV"),
        ("3841", "XLV"),
        ("6022", "XLF"),
        ("6798", "XLRE"),
        ("6512", "XLRE"),
        ("1311", "XLE"),
        ("2911", "XLE"),
        ("3711", "XLY"),
        ("3714", "XLY"),
        ("5331", "XLY"),
        ("2080", "XLP"),
        ("2100", "XLP"),
        ("5141", "XLP"),
        ("3523", "XLI"),
        ("3721", "XLI"),
        ("4213", "XLI"),
        ("8711", "XLI"),
        ("2810", "XLB"),
        ("3312", "XLB"),
        ("1040", "XLB"),
        ("4911", "XLU"),
        ("4813", "XLC"),
        ("7812", "XLC"),
    ],
)
def test_select_benchmarks_by_sic(sic: str | None, etf: str | None) -> None:
    refs = select_benchmarks(sic)
    assert refs[0].role == "broad_market" and refs[0].symbol == "^SPX" and refs[0].name == "S&P 500"
    if etf is None:
        assert len(refs) == 1
    else:
        assert len(refs) == 2
        sector = refs[1]
        assert sector.role == "sector" and sector.symbol == etf
        assert sic in sector.reason and etf in sector.reason
        assert etf in BENCHMARKS
