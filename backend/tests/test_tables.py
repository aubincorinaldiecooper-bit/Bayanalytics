"""Tables in the pages a web search returned (``research.tables``): raw tables from HTML, CSV
and JSON; the deterministic candidate filter; and the readers for daily price history and
reported figures (dates, numbers, units, the ``as_of`` cut, plausibility). All inputs are
synthetic, written inline."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from bayanalytics.research.extract import extract_page
from bayanalytics.research.provider import PageResult
from bayanalytics.research.tables import (
    MIN_PRICE_ROWS,
    FigureCandidate,
    RawTable,
    date_order,
    figure_candidates,
    html_tables,
    page_tables,
    parse_cell_number,
    parse_table_date,
    price_candidates,
    read_figures,
    read_price_history,
    shortlist,
)
from bayanalytics.schemas.common import utcnow

AS_OF = date(2026, 9, 26)


def _weekdays(end: date, count: int) -> list[date]:
    days: list[date] = []
    day = end
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day -= timedelta(days=1)
    return days  # newest first


def _price_html(days: list[date], fmt: str = "{:%b %d, %Y}", extra_rows: str = "") -> str:
    rows = "".join(
        f"<tr><td>{fmt.format(d)}</td><td>{100 + i:.2f}</td><td>{102 + i:.2f}</td>"
        f"<td>{99 + i:.2f}</td><td>${101 + i:,.2f}</td><td>{45 + i}.1M</td></tr>"
        for i, d in enumerate(days)
    )
    return (
        "<html><body><p>FIXTURE daily prices. Currency in USD.</p><table><thead><tr>"
        "<th>Date</th><th>Open</th><th>High</th><th>Low</th><th>Close*</th><th>Volume</th>"
        f"</tr></thead><tbody>{extra_rows}{rows}</tbody></table></body></html>"
    )


def _only_price(html: str) -> tuple:
    (candidate,) = price_candidates(html_tables(html), AS_OF)
    return candidate


# ----------------------------------------------------------------------------- raw tables


def test_html_tables_expand_colspans_and_combine_multi_row_headers() -> None:
    html = (
        "<p>(In millions, except per-share amounts)</p><table><caption>Fixture statement"
        "</caption><tr><td></td><th colspan='2'>Three Months Ended</th></tr>"
        "<tr><td></td><th>June 27, 2026</th><th>June 28, 2025</th></tr>"
        "<tr><td>Total net sales</td><td>$</td><td>94,040</td></tr></table>"
        "<script>var ignored = '<table><tr><td>x</td></tr></table>';</script>"
        '<script type="application/ld+json">{"@type": "Dataset", "data": '
        '[{"date": "2026-06-27", "revenue": 94040}, {"date": "2026-03-28", "revenue": 95360}]}'
        "</script>"
    )
    first, second = html_tables(html)
    assert first.columns == [
        "",
        "Three Months Ended June 27, 2026",
        "Three Months Ended June 28, 2025",
    ]
    assert first.rows == [["Total net sales", "$", "94,040"]]
    assert "Fixture statement" in first.context and "(In millions" in first.context
    assert second.origin == "json" and second.columns == ["date", "revenue"]
    assert second.rows == [["2026-06-27", "94040"], ["2026-03-28", "95360"]]


def test_page_tables_reads_csv_and_json_responses_and_html_records() -> None:
    csv = page_tables(
        "csv", {"columns": ["Date", "Close"], "rows": [["2026-09-25", "1.0"]], "row_count": 1}
    )
    assert csv[0].origin == "csv" and csv[0].rows == [["2026-09-25", "1.0"]]
    columnar = page_tables("json", {"date": ["2026-09-24", "2026-09-25"], "close": [1.5, 2]})
    assert columnar[0].rows == [["2026-09-24", "1.5"], ["2026-09-25", "2"]]
    split = page_tables("json", {"items": [{"columns": ["d", "c"], "data": [["a", 1], ["b", 2]]}]})
    assert split[0].columns == ["d", "c"]
    assert page_tables("json", {"parse_error": True}) == []
    page = PageResult(
        url="https://fixture.example/p",
        final_url="https://fixture.example/p",
        status=200,
        content_type="text/html",
        body=_price_html(_weekdays(AS_OF, 3)),
        fetched_at=utcnow(),
    )
    record = extract_page(page)
    (table,) = page_tables(record.extraction_method, record.structured)
    assert table.columns[4] == "Close*" and len(table.rows) == 3
    served_as_text = page.model_copy(
        update={"content_type": "text/plain", "body": "Date,Close\n2026-09-25,1.0\n"}
    )
    assert extract_page(served_as_text).extraction_method == "csv"


# ----------------------------------------------------------------------------- cells


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-06-30", date(2026, 6, 30)),
        ("2026/06/30", date(2026, 6, 30)),
        ("Jun 30, 2026", date(2026, 6, 30)),
        ("June 30 2026", date(2026, 6, 30)),
        ("Mon, Jun 30, 2026", date(2026, 6, 30)),
        ("30 Jun 2026", date(2026, 6, 30)),
        ("30-Jun-26", date(2026, 6, 30)),
        ("06/30/2026", date(2026, 6, 30)),
        ("30/06/2026", date(2026, 6, 30)),
        ("30.06.2026", date(2026, 6, 30)),
        ("20260630", date(2026, 6, 30)),
        ("Three Months Ended June 27, 2026", date(2026, 6, 27)),
        ("Jun 2026", None),  # a month is not a date
        ("Q3 2026", None),  # a label is not a date
        ("05/06/2026", None),  # ambiguous without the column's order
        ("1782000000", None),  # epoch only in a date column
        ("", None),
    ],
)
def test_parse_table_date(text: str, expected: date | None) -> None:
    assert parse_table_date(text) == expected


def test_numeric_date_order_comes_from_the_column() -> None:
    column = ["05/06/2026", "05/07/2026", "05/13/2026"]
    assert date_order(column) == "mdy"
    assert parse_table_date("05/06/2026", "mdy") == date(2026, 5, 6)
    assert date_order(["13/05/2026", "05/06/2026"]) == "dmy"
    assert parse_table_date("05/06/2026", "dmy") == date(2026, 6, 5)
    assert date_order(["13/05/2026", "05/13/2026"]) == "conflict"
    assert parse_table_date("13/05/2026", "conflict") is None
    assert parse_table_date("1782000000", epoch=True) == date(2026, 6, 21)


@pytest.mark.parametrize(
    ("text", "value", "scale"),
    [
        ("$1,234.50", 1234.5, ""),
        ("(1,234)", -1234.0, ""),
        ("\u22121.5", -1.5, ""),
        ("45.1M", 45_100_000.0, "M"),
        ("987K", 987_000.0, "K"),
        ("1.2B", 1_200_000_000.0, "B"),
        ("87,400 (1)", 87_400.0, ""),
    ],
)
def test_parse_cell_number(text: str, value: float, scale: str) -> None:
    parsed = parse_cell_number(text)
    assert parsed is not None and parsed.value == pytest.approx(value)
    assert parsed.scale_applied == scale


@pytest.mark.parametrize("text", ["", "-", "\u2014", "N/A", "n/m", "Dividend"])
def test_missing_cells_are_not_numbers(text: str) -> None:
    assert parse_cell_number(text) is None


# ----------------------------------------------------------------------------- prices


def test_price_history_is_cut_at_as_of_sorted_and_parsed() -> None:
    days = _weekdays(date(2026, 9, 30), 40)  # Sep 28-30 are after as_of
    dividend = '<tr><td>Aug 10, 2026</td><td colspan="5">0.26 Dividend</td></tr>'
    candidate = _only_price(_price_html(days, extra_rows=dividend))
    assert candidate.value_cols == (1, 2, 3, 4, 5)
    table = read_price_history(candidate, 4, AS_OF)
    assert table is not None
    dates = [p.date for p in table.points]
    assert dates == sorted(dates) and dates[-1] == date(2026, 9, 25)  # oldest -> newest
    assert all(d <= AS_OF for d in dates)
    assert table.after_as_of == 3 and table.rejected_rows == 1  # the dividend row
    latest = table.points[-1]
    assert latest.close == pytest.approx(104.0)  # "$104.00" read as a number
    assert latest.volume == pytest.approx(48_100_000.0)  # "48.1M"
    assert table.currency == "USD" and table.currency_stated


@pytest.mark.parametrize("fmt", ["{:%Y-%m-%d}", "{:%d %b %Y}", "{:%m/%d/%Y}", "{:%d-%b-%y}"])
def test_price_history_reads_several_date_formats(fmt: str) -> None:
    candidate = _only_price(_price_html(_weekdays(AS_OF, 30), fmt))
    table = read_price_history(candidate, 4, AS_OF)
    assert table is not None and len(table.points) == 30


def test_fewer_than_twenty_rows_or_non_daily_rows_are_not_a_price_history() -> None:
    assert price_candidates(html_tables(_price_html(_weekdays(AS_OF, 19))), AS_OF) == []
    monthly = [date(2024 + (m // 12), m % 12 + 1, 1) for m in range(30)]
    assert price_candidates(html_tables(_price_html(monthly)), AS_OF) == []
    # enough rows, but most are after as_of: not enough left
    later = _weekdays(date(2026, 11, 30), 30)
    assert price_candidates(html_tables(_price_html(later)), AS_OF) == []
    assert MIN_PRICE_ROWS == 20


def test_a_close_outside_its_own_range_is_rejected_and_a_bad_column_reads_nothing() -> None:
    days = _weekdays(AS_OF, 25)
    html = _price_html(days).replace("<td>$125.00</td>", "<td>$999.00</td>")
    candidate = _only_price(html)
    table = read_price_history(candidate, 4, AS_OF)
    assert table is not None and table.rejected_rows == 1 and len(table.points) == 24
    assert read_price_history(candidate, candidate.date_col, AS_OF) is None
    # the Open column as the "close" still reads (Laya's choice is what decides the column)
    assert read_price_history(candidate, 1, AS_OF) is not None


# ----------------------------------------------------------------------------- figures

_QUARTERLY = """<p>Fixture financials in millions USD. Fiscal year ends in September.</p>
<table><thead><tr><th>Fiscal Quarter</th><th>TTM</th><th>Q1 2027</th><th>Q3 2026</th>
<th>Q2 2026</th><th>Q1 2026</th><th>Q4 2025</th><th>Q3 2025</th></tr></thead><tbody>
<tr><td>Period Ending</td><td></td><td>Dec 26, 2026</td><td>Jun 27, 2026</td><td>Mar 28, 2026</td>
<td>Dec 27, 2025</td><td>Sep 27, 2025</td><td>Jun 28, 2025</td></tr>
<tr><td>Revenue</td><td>99,000</td><td>30,000</td><td>22,760</td><td>23,810</td><td>29,100</td>
<td>23,300</td><td>21,870</td></tr>
<tr><td>Revenue Growth (YoY)</td><td>4%</td><td>-</td><td>4.1%</td><td>3.7%</td><td>4.3%</td>
<td>4.4%</td><td>3.9%</td></tr>
<tr><td>Cost of Revenue</td><td>1</td><td>1</td><td>13,610</td><td>14,190</td><td>17,170</td>
<td>13,960</td><td>13,250</td></tr>
<tr><td>Gross Profit</td><td>1</td><td>1</td><td>9,150</td><td>9,620</td><td>11,930</td>
<td>9,340</td><td>30,000</td></tr>
<tr><td>Adjusted EBITDA</td><td>1</td><td>1</td><td>7,000</td><td>7,100</td><td>9,000</td>
<td>7,200</td><td>6,900</td></tr>
<tr><td>EPS (Diluted)</td><td>1</td><td>1</td><td>$1.59</td><td>1.67</td><td>2.16</td>
<td>1.61</td><td>1.47</td></tr>
<tr><td>Shares Outstanding (Diluted)</td><td>1</td><td>1</td><td>3,080</td><td>3,090</td>
<td>3,100</td><td>3,110</td><td>3,120</td></tr>
</tbody></table>"""


def _candidate(html: str) -> FigureCandidate:
    (candidate,) = figure_candidates(html_tables(html))
    return candidate


def _read(candidate: FigureCandidate, *metrics: str):
    chosen = {m: shortlist(candidate, m, 6)[0] for m in metrics}
    return read_figures(candidate, chosen, AS_OF)


def test_quarterly_figures_with_periods_as_columns() -> None:
    candidate = _candidate(_QUARTERLY)
    assert candidate.across and candidate.scales == (1e6, None)
    # "Cost of Revenue" and "Revenue Growth (YoY)" are never offered as revenue, nor
    # "Adjusted EBITDA" as operating income
    assert [line.label for line in shortlist(candidate, "revenue", 6)] == ["Revenue"]
    assert shortlist(candidate, "operating_income", 6) == []
    result = _read(candidate, "revenue", "gross_profit", "eps_diluted", "shares_outstanding")
    revenue = {f.end: f for f in result.figures if f.metric == "revenue"}
    assert sorted(revenue) == [
        date(2025, 6, 28),
        date(2025, 9, 27),
        date(2025, 12, 27),
        date(2026, 3, 28),
        date(2026, 6, 27),
    ]  # the TTM column is skipped, the future quarter is after as_of
    latest = revenue[date(2026, 6, 27)]
    assert latest.value == 22_760e6 and latest.unit == "USD" and latest.kind == "fiscal_quarter"
    assert latest.period_label == "Q3 2026" and latest.label == "Revenue"
    assert latest.raw_value == "22,760"
    eps = {f.end: f.value for f in result.figures if f.metric == "eps_diluted"}
    assert eps[date(2026, 6, 27)] == 1.59  # per share: never scaled
    shares = [f for f in result.figures if f.metric == "shares_outstanding"]
    assert shares[0].kind == "instant" and shares[-1].value == 3_080e6
    # gross profit above revenue on 2025-06-28 is implausible and dropped, not repaired
    gross = {f.end for f in result.figures if f.metric == "gross_profit"}
    assert date(2025, 6, 28) not in gross and len(gross) == 4
    assert result.implausible == {"gross margin outside 0-100%": 1}
    assert result.after_as_of == 4  # Q1 2027 for the four metrics
    notes = result.notes(AS_OF)
    assert any("after 2026-09-26 dropped" in n for n in notes)
    assert any("implausible" in n for n in notes)


def test_quarterly_figures_with_periods_as_rows_and_units_in_the_header() -> None:
    html = (
        "<p>Fixture results by quarter.</p><table>"
        "<tr><th>Quarter ended</th><th>Revenue ($M)</th><th>Diluted EPS</th>"
        "<th>Net income</th></tr>"
        "<tr><td>2026-06-27</td><td>22,760</td><td>$1.59</td><td>4.89B</td></tr>"
        "<tr><td>2026-03-28</td><td>23,810</td><td>$1.67</td><td>5.17B</td></tr>"
        "<tr><td>2025-12-27</td><td>29,100</td><td>$2.16</td><td>-</td></tr></table>"
    )
    candidate = _candidate(html)
    assert not candidate.across
    result = _read(candidate, "revenue", "eps_diluted", "net_income")
    by = {(f.metric, f.end): f for f in result.figures}
    assert by[("revenue", date(2026, 6, 27))].value == 22_760e6  # "($M)" in the header
    assert by[("net_income", date(2026, 3, 28))].value == 5.17e9  # the cell's own suffix
    assert ("net_income", date(2025, 12, 27)) not in by  # "-" is missing, not zero
    assert by[("eps_diluted", date(2026, 6, 27))].unit == "USD/shares"
    assert {f.kind for f in result.figures} == {"fiscal_quarter"}  # "Quarter ended"


@pytest.mark.parametrize(
    ("context", "scale"),
    [
        ("Amounts in thousands of U.S. dollars", 1e3),
        ("USD thousands", 1e3),
        ("Figures in $M", 1e6),
        ("(in billions)", 1e9),
        ("Note (b): restated", 1.0),  # a footnote letter is not a unit
    ],
)
def test_units_are_read_from_the_text_before_the_table(context: str, scale: float) -> None:
    html = (
        f"<p>{context}</p><table><tr><th>Period</th><th>2026-06-27</th><th>2026-03-28</th></tr>"
        "<tr><td>Revenue</td><td>1,000</td><td>1,100</td></tr></table>"
    )
    candidate = _candidate(html)
    result = _read(candidate, "revenue")
    assert {f.value for f in result.figures} == {1_000 * scale, 1_100 * scale}
    assert {f.kind for f in result.figures} == {"fiscal_quarter"}  # 91 days apart


def test_label_only_periods_and_ambiguous_spacing_are_not_read() -> None:
    labels_only = (
        "<table><tr><th>Metric</th><th>Q3 2026</th><th>Q2 2026</th></tr>"
        "<tr><td>Revenue</td><td>1,000</td><td>1,100</td></tr></table>"
    )
    assert figure_candidates(html_tables(labels_only)) == []  # a label does not fix a date
    irregular = (
        "<table><tr><th>Metric</th><th>2026-06-27</th><th>2026-01-10</th></tr>"
        "<tr><td>Revenue</td><td>1,000</td><td>1,100</td></tr></table>"
    )
    assert figure_candidates(html_tables(irregular)) == []  # neither quarter nor year apart


def test_plausibility_rejects_out_of_range_values() -> None:
    html = (
        "<p>in millions</p><table><tr><th>Period</th><th>2026-06-27</th><th>2026-03-28</th></tr>"
        "<tr><td>Revenue</td><td>-5</td><td>2,000,000,000,000</td></tr>"
        "<tr><td>Operating income</td><td>900</td><td>10</td></tr>"
        "<tr><td>EPS</td><td>250,000</td><td>1.2</td></tr></table>"
    )
    candidate = _candidate(html)
    result = _read(candidate, "revenue", "operating_income", "eps_diluted")
    assert [(f.metric, f.value) for f in result.figures] == [
        ("eps_diluted", 1.2),
        ("operating_income", 10e6),
        ("operating_income", 900e6),
    ]
    assert result.implausible == {
        "amount out of range": 1,
        "negative revenue": 1,
        "per-share value out of range": 1,
    }


def test_a_percentage_gross_margin_line_is_not_an_amount() -> None:
    html = (
        "<p>in millions</p><table><tr><th>Period</th><th>2026-06-27</th><th>2026-03-28</th></tr>"
        "<tr><td>Net sales</td><td>94,040</td><td>95,360</td></tr>"
        "<tr><td>Gross margin</td><td>43,720</td><td>46.2</td></tr></table>"
    )
    candidate = _candidate(html)
    result = _read(candidate, "revenue", "gross_profit")
    gross = [f for f in result.figures if f.metric == "gross_profit"]
    assert [f.value for f in gross] == [43_720e6]  # "46.2" is a percentage without its sign
    assert result.implausible == {"gross margin line is not an amount": 1}


def test_eps_under_a_section_heading_is_offered_with_the_section() -> None:
    html = (
        "<table><tr><td></td><th>Three Months Ended June 27, 2026</th>"
        "<th>Three Months Ended June 28, 2025</th></tr>"
        "<tr><td>Earnings per share:</td><td></td><td></td></tr>"
        "<tr><td>Basic</td><td>1.60</td><td>1.48</td></tr>"
        "<tr><td>Diluted</td><td>1.59</td><td>1.47</td></tr>"
        "<tr><td>Shares used in computing earnings per share:</td><td></td><td></td></tr>"
        "<tr><td>Diluted</td><td>3,080</td><td>3,120</td></tr></table>"
    )
    candidate = _candidate(html)
    labels = [line.label for line in shortlist(candidate, "eps_diluted", 6)]
    assert labels[0] == "Earnings per share: Diluted"
    assert "Shares used in computing earnings per share: Diluted" not in labels
    assert [line.label for line in shortlist(candidate, "eps_basic", 6)] == [
        "Earnings per share: Basic"
    ]


def test_raw_table_round_trips_and_bounds() -> None:
    table = RawTable(["a", "b"], [["1", "2"]], "ctx", "html")
    assert RawTable.from_dict(table.to_dict()) == table
    assert RawTable.from_dict({"columns": "x"}) is None
    wide = RawTable.from_dict({"columns": [str(i) for i in range(100)], "rows": [["1"] * 100]})
    assert wide is not None and len(wide.columns) == 40
