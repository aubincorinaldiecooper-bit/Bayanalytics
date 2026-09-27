"""Tests for the normalization layer: units, periods, sessions, facts, corporate actions."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from bayanalytics.normalization import NORMALIZATION_VERSION
from bayanalytics.normalization.corporate_actions import (
    apply_split_adjustment,
    comparable_periods,
    detect_identity_breaks,
    parse_split_ratio,
    total_return_note,
)
from bayanalytics.normalization.facts import (
    build_facts,
    classify_freshness,
    derive_fourth_quarter_rows,
    detect_stale_mix,
    freshness_summary,
    normalize_unit,
)
from bayanalytics.normalization.periods import (
    CURRENT_QUARTER_AMBIGUOUS,
    calendar_quarter,
    calendar_to_fiscal,
    classify_duration,
    current_fiscal_quarter,
    is_consecutive_quarters,
    label,
    parse_period_label,
    period_from_xbrl,
    ttm_period,
    ttm_window,
)
from bayanalytics.normalization.sessions import (
    TimestampSet,
    classify_price_timestamp,
    label_series,
    latest_completed_close,
    latest_completed_session,
    session_view,
    to_exchange_time,
)
from bayanalytics.normalization.units import (
    assert_same_currency,
    canonical_scale,
    normalize_currency_code,
    parse_number,
    scale_to_base,
    try_parse_number,
)
from bayanalytics.schemas.evidence import (
    CorporateAction,
    NormalizedFact,
    Period,
    PricePoint,
    PriceSeries,
)

AS_OF = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def test_version_constant():
    assert NORMALIZATION_VERSION == "2026.09-1"


# ======================================================================================
# units
# ======================================================================================


class TestParseNumber:
    @pytest.mark.parametrize(
        ("text", "value", "unit", "currency", "scale"),
        [
            ("$1.2B", 1.2e9, "USD", "USD", "B"),
            ("1.2 billion", 1.2e9, "number", None, "B"),
            ("18.2%", 18.2, "percent", None, ""),
            ("(1,234)", -1234.0, "number", None, ""),
            ("\u22127.4%", -7.4, "percent", None, ""),
            ("€3.4M", 3.4e6, "EUR", "EUR", "M"),
            ("1.5x", 1.5, "ratio", None, ""),
            ("1.5\u00d7", 1.5, "ratio", None, ""),
            ("12,345.67", 12345.67, "number", None, ""),
            ("2.5T", 2.5e12, "number", None, "T"),
            ("950.0K", 950_000.0, "number", None, "K"),
            ("1.2bn", 1.2e9, "number", None, "B"),
            ("3.4 mn", 3.4e6, "number", None, "M"),
            ("-$1.2B", -1.2e9, "USD", "USD", "B"),
            ("$-1.2B", -1.2e9, "USD", "USD", "B"),
            ("($1,234)", -1234.0, "USD", "USD", ""),
            ("USD 1,234", 1234.0, "USD", "USD", ""),
            ("1,234 USD", 1234.0, "USD", "USD", ""),
            ("15.3B shares", 15.3e9, "shares", None, "B"),
            ("42 days", 42.0, "days", None, ""),
            ("+4.5%", 4.5, "percent", None, ""),
            ("£1,000", 1000.0, "GBP", "GBP", ""),
        ],
    )
    def test_cases(self, text, value, unit, currency, scale):
        parsed = parse_number(text)
        assert parsed.value == pytest.approx(value)
        assert parsed.unit == unit
        assert parsed.currency == currency
        assert parsed.scale_applied == scale

    @pytest.mark.parametrize("text", ["abc", "", "1.2 zillion", "1.5x%", "--5"])
    def test_rejects_garbage(self, text):
        with pytest.raises(ValueError):
            parse_number(text)
        assert try_parse_number(text) is None

    def test_try_parse(self):
        assert try_parse_number("$5").value == 5.0
        assert try_parse_number(None) is None


class TestCurrencyAndScale:
    def test_normalize_currency_code(self):
        assert normalize_currency_code("$") == "USD"
        assert normalize_currency_code("US$") == "USD"
        assert normalize_currency_code("usd") == "USD"
        assert normalize_currency_code("€") == "EUR"
        assert normalize_currency_code("chf") == "CHF"
        with pytest.raises(ValueError):
            normalize_currency_code("dollars")
        with pytest.raises(ValueError):
            normalize_currency_code(None)

    def test_scale_to_base(self):
        assert scale_to_base(1.5, "B") == 1.5e9
        assert scale_to_base(1.5, "billion") == 1.5e9
        assert scale_to_base(1.5, "mn") == 1.5e6
        assert scale_to_base(1.5, None) == 1.5
        assert canonical_scale("thousand") == "K"
        with pytest.raises(ValueError):
            scale_to_base(1.0, "zillion")

    def test_assert_same_currency(self):
        usd = [{"currency": "USD"}, {"currency": "usd"}]
        assert assert_same_currency(usd) == "USD"
        assert assert_same_currency([]) is None
        with pytest.raises(ValueError, match=r"EUR.*USD"):
            assert_same_currency([{"currency": "USD"}, {"currency": "EUR"}, {"currency": "USD"}])
        with pytest.raises(ValueError, match="unknown"):
            assert_same_currency([{"currency": "USD"}, {"currency": None}])

    def test_normalize_unit(self):
        assert normalize_unit("USD/shares") == "USD_per_share"
        assert normalize_unit("EUR/shares") == "EUR_per_share"
        assert normalize_unit("pure") == "ratio"
        assert normalize_unit("shares") == "shares"
        assert normalize_unit("usd") == "USD"
        assert normalize_unit("furlongs") == "furlongs"


# ======================================================================================
# periods
# ======================================================================================


class TestPeriods:
    def test_period_kinds_from_xbrl(self):
        quarter = period_from_xbrl("2025-06-29", "2025-09-27", 2025, "Q4", "10-K")
        assert quarter.kind == "fiscal_quarter"
        assert quarter.fiscal_period == "Q4"
        assert quarter.label == "Q4 FY2025"
        year = period_from_xbrl("2024-09-29", "2025-09-27", 2025, "FY", "10-K")
        assert year.kind == "fiscal_year" and year.label == "FY2025"
        instant = period_from_xbrl(None, "2025-09-27", 2025, "FY", "10-K")
        assert instant.kind == "instant" and instant.label == "as of 2025-09-27"
        ytd = period_from_xbrl("2024-09-29", "2025-06-28", 2025, "Q3", "10-Q")
        assert ytd.kind == "ytd" and ytd.label == "Q3 YTD to 2025-06-28"
        odd = period_from_xbrl("2025-01-01", "2025-05-15", 2025, "Q2", "10-Q")
        assert odd.kind == "calendar_range" and odd.label == "2025-01-01 to 2025-05-15"
        # a quarter-length duration in a 10-K labelled FY is the fourth quarter
        q4 = period_from_xbrl(date(2025, 6, 29), date(2025, 9, 27), 2025, "FY", "10-K")
        assert q4.fiscal_period == "Q4"
        ttm = period_from_xbrl("2024-09-29", "2025-09-27", 2025, "Q4", "10-Q", kind_hint="ttm")
        assert ttm.kind == "ttm" and ttm.label == "TTM to 2025-09-27"

    def test_classify_duration_notes(self):
        assert classify_duration(date(2025, 1, 1), date(2025, 3, 31)) == ("fiscal_quarter", None)
        kind, note = classify_duration(date(2025, 1, 1), date(2025, 5, 15))
        assert kind == "calendar_range" and "135-day" in note
        kind, note = classify_duration(date(2025, 1, 1), date(2025, 6, 30))
        assert kind == "ytd" and "cumulative" in note
        assert classify_duration(None, None)[0] == "unknown"

    @pytest.mark.parametrize(
        ("text", "kind", "fy", "fp", "expected_label"),
        [
            ("Q3 FY25", "fiscal_quarter", 2025, "Q3", "Q3 FY2025"),
            ("FY2025", "fiscal_year", 2025, "FY", "FY2025"),
            ("2025-Q4", "fiscal_quarter", 2025, "Q4", "Q4 FY2025"),
            ("Q4 2025", "fiscal_quarter", 2025, "Q4", "Q4 FY2025"),
            ("3Q25", "fiscal_quarter", 2025, "Q3", "Q3 FY2025"),
            ("TTM", "ttm", None, None, "TTM"),
            ("FY 2024", "fiscal_year", 2024, "FY", "FY2024"),
            ("TTM to 2025-06-28", "ttm", None, None, "TTM to 2025-06-28"),
            ("H1 FY2025", "ytd", 2025, "H1", "H1 YTD FY2025"),
        ],
    )
    def test_parse_period_label(self, text, kind, fy, fp, expected_label):
        period = parse_period_label(text)
        assert period.kind == kind
        assert period.fiscal_year == fy
        assert period.fiscal_period == fp
        assert period.label == expected_label

    def test_parse_period_label_rejects_unknown(self):
        with pytest.raises(ValueError):
            parse_period_label("last quarter")

    def test_labels(self):
        assert label(Period(kind="fiscal_year", fiscal_year=2025)) == "FY2025"
        assert (
            label(Period(kind="fiscal_quarter", fiscal_year=2025, fiscal_period="Q3"))
            == "Q3 FY2025"
        )
        assert label(Period(kind="ttm", end=date(2025, 6, 28))) == "TTM to 2025-06-28"
        assert label(Period(kind="instant", end=date(2025, 6, 28))) == "as of 2025-06-28"
        assert (
            label(Period(kind="calendar_range", start=date(2025, 1, 1), end=date(2025, 2, 1)))
            == "2025-01-01 to 2025-02-01"
        )
        assert label(Period()) == "unknown period"

    def test_ttm_window(self):
        quarters = [
            period_from_xbrl(s, e, fy, fp)
            for s, e, fy, fp in [
                ("2024-09-29", "2024-12-28", 2025, "Q1"),
                ("2024-12-29", "2025-03-29", 2025, "Q2"),
                ("2025-03-30", "2025-06-28", 2025, "Q3"),
                ("2025-06-29", "2025-09-27", 2025, "Q4"),
                ("2025-09-28", "2025-12-27", 2026, "Q1"),
            ]
        ]
        window = ttm_window(quarters)
        assert [p.label for p in window] == ["Q2 FY2025", "Q3 FY2025", "Q4 FY2025", "Q1 FY2026"]
        assert ttm_period(window).label == "TTM to 2025-12-27"
        assert ttm_period(window).start == date(2024, 12, 29)
        assert ttm_window(quarters[:3]) is None
        gap = [quarters[0], quarters[1], quarters[3], quarters[4]]  # Q3 missing
        assert ttm_window(gap) is None
        assert ttm_window([*quarters, quarters[-1]]) is not None  # duplicates collapse
        assert is_consecutive_quarters(quarters[0], quarters[1])
        assert not is_consecutive_quarters(quarters[0], quarters[2])
        mislabelled = Period(
            kind="fiscal_quarter",
            fiscal_year=2027,
            fiscal_period="Q1",
            start=date(2024, 12, 29),
            end=date(2025, 3, 29),
        )
        assert not is_consecutive_quarters(quarters[0], mislabelled)

    @pytest.mark.parametrize(
        ("d", "expected"),
        [
            (date(2025, 10, 1), (2026, "Q1")),
            (date(2025, 11, 15), (2026, "Q1")),
            (date(2025, 12, 31), (2026, "Q1")),
            (date(2026, 1, 1), (2026, "Q2")),
            (date(2026, 3, 31), (2026, "Q2")),
            (date(2026, 4, 1), (2026, "Q3")),
            (date(2026, 6, 30), (2026, "Q3")),
            (date(2026, 7, 1), (2026, "Q4")),
            (date(2026, 9, 30), (2026, "Q4")),
        ],
    )
    def test_calendar_to_fiscal_september_year_end(self, d, expected):
        assert calendar_to_fiscal(d, "0930") == expected

    def test_calendar_to_fiscal_other_year_ends(self):
        assert calendar_to_fiscal(date(2026, 3, 31), "1231") == (2026, "Q1")
        assert calendar_to_fiscal(date(2026, 12, 31), "1231") == (2026, "Q4")
        assert calendar_to_fiscal(date(2025, 1, 15), "0131") == (2025, "Q4")
        assert calendar_to_fiscal(date(2025, 2, 1), "0131") == (2026, "Q1")
        assert calendar_quarter(date(2025, 11, 15)) == (2025, "Q4")
        assert calendar_quarter(date(2025, 11, 15)) != calendar_to_fiscal(
            date(2025, 11, 15), "0930"
        )
        with pytest.raises(ValueError):
            calendar_to_fiscal(date(2025, 1, 1), "13-01")

    def test_current_quarter_is_ambiguous_without_fiscal_calendar(self):
        assert current_fiscal_quarter(date(2025, 11, 15), None) == (None, CURRENT_QUARTER_AMBIGUOUS)
        assert current_fiscal_quarter(date(2025, 11, 15), "0930") == ((2026, "Q1"), None)


# ======================================================================================
# sessions
# ======================================================================================


def _series(dates: list[date], price_type="historical_close", **kw) -> PriceSeries:
    return PriceSeries(
        symbol="AAPL",
        source_id="src_px",
        points=[PricePoint(date=d, close=100.0 + i) for i, d in enumerate(dates)],
        price_type=price_type,
        retrieved_at=AS_OF,
        **kw,
    )


class TestSessions:
    @pytest.mark.parametrize(
        ("ts", "price_type", "session_date", "completed"),
        [
            (
                datetime(2026, 9, 25, 12, 0, tzinfo=UTC),
                "pre_market",
                date(2026, 9, 25),
                date(2026, 9, 24),
            ),  # 08:00 NY
            (
                datetime(2026, 9, 25, 13, 30, tzinfo=UTC),
                "intraday",
                date(2026, 9, 25),
                date(2026, 9, 24),
            ),  # 09:30 NY
            (
                datetime(2026, 9, 25, 19, 59, tzinfo=UTC),
                "intraday",
                date(2026, 9, 25),
                date(2026, 9, 24),
            ),  # 15:59 NY
            (
                datetime(2026, 9, 25, 20, 0, tzinfo=UTC),
                "after_hours",
                date(2026, 9, 25),
                date(2026, 9, 25),
            ),  # 16:00 NY
            (
                datetime(2026, 9, 25, 23, 30, tzinfo=UTC),
                "after_hours",
                date(2026, 9, 25),
                date(2026, 9, 25),
            ),  # 19:30 NY
            (
                datetime(2026, 9, 26, 1, 0, tzinfo=UTC),
                "latest_close",
                date(2026, 9, 25),
                date(2026, 9, 25),
            ),  # 21:00 NY Fri
            (
                datetime(2026, 9, 26, 15, 0, tzinfo=UTC),
                "latest_close",
                date(2026, 9, 25),
                date(2026, 9, 25),
            ),  # Saturday
            (
                datetime(2026, 9, 27, 15, 0, tzinfo=UTC),
                "latest_close",
                date(2026, 9, 25),
                date(2026, 9, 25),
            ),  # Sunday
            (
                datetime(2026, 9, 28, 6, 0, tzinfo=UTC),
                "latest_close",
                date(2026, 9, 25),
                date(2026, 9, 25),
            ),  # 02:00 NY Monday
            (
                datetime(2026, 9, 28, 14, 0, tzinfo=UTC),
                "intraday",
                date(2026, 9, 28),
                date(2026, 9, 25),
            ),  # Monday 10:00
        ],
    )
    def test_classify_price_timestamp(self, ts, price_type, session_date, completed):
        info = classify_price_timestamp(ts)
        assert info["price_type"] == price_type
        assert info["session_date"] == session_date
        assert info["latest_completed_session"] == completed
        assert info["exchange_timezone"] == "America/New_York"
        assert info["label"]

    def test_timezone_is_preserved(self):
        # 14:00 in London on a Friday is 09:00 in New York: pre-market, not intraday.
        london = datetime(2026, 9, 25, 14, 0, tzinfo=ZoneInfo("Europe/London"))
        assert classify_price_timestamp(london)["price_type"] == "pre_market"
        # The same instant evaluated for an exchange in London is intraday (LSE hours differ,
        # but the default schedule is applied in the given timezone).
        assert classify_price_timestamp(london, "Europe/London")["price_type"] == "intraday"
        naive = datetime(2026, 9, 25, 14, 0)  # naive -> UTC by contract -> 10:00 NY
        assert classify_price_timestamp(naive)["price_type"] == "intraday"
        assert to_exchange_time(naive).hour == 10

    def test_holidays_are_opt_in(self):
        friday = datetime(2026, 9, 25, 14, 0, tzinfo=UTC)
        assert latest_completed_session(friday) == date(2026, 9, 24)
        info = classify_price_timestamp(friday, holidays={date(2026, 9, 25), date(2026, 9, 24)})
        assert info["price_type"] == "latest_close"
        assert info["session_date"] == date(2026, 9, 23)
        assert "holiday" in info["label"]

    def test_latest_completed_close_excludes_open_session(self):
        series = _series([date(2026, 9, 23), date(2026, 9, 24), date(2026, 9, 25)])
        friday_intraday = datetime(2026, 9, 25, 15, 0, tzinfo=UTC)
        assert latest_completed_close(series, friday_intraday).date == date(2026, 9, 24)
        assert latest_completed_close(series, AS_OF).date == date(2026, 9, 25)
        assert latest_completed_close(series, datetime(2026, 9, 1, tzinfo=UTC)) is None

    def test_label_series_returns_copy(self):
        series = _series([date(2026, 9, 23), date(2026, 9, 24), date(2026, 9, 25)])
        labelled = label_series(series, AS_OF)
        assert labelled is not series
        assert series.price_type == "historical_close" and series.label == ""
        assert labelled.price_type == "latest_close"
        assert labelled.session_date == date(2026, 9, 25)
        assert labelled.label.startswith("latest close 2026-09-25")
        assert labelled.retrieved_at == series.retrieved_at
        intraday = label_series(series, datetime(2026, 9, 25, 15, 0, tzinfo=UTC))
        assert intraday.price_type == "intraday"
        assert "session in progress" in intraday.label
        behind = label_series(_series([date(2026, 9, 21), date(2026, 9, 22)]), AS_OF)
        assert behind.price_type == "latest_close"
        assert "3 completed session(s)" in behind.label
        assert label_series(_series([]), AS_OF).label == "empty price series"
        view = session_view(101.0, labelled, None)
        assert view["price_type"] == "latest_close" and view["session_date"] == "2026-09-25"

    def test_timestamp_set(self):
        ts = TimestampSet(
            event=datetime(2026, 9, 25, 20, 0, tzinfo=UTC),
            published=datetime(2026, 9, 25, 21, 0, tzinfo=UTC),
            retrieved=None,
        )
        assert ts.as_dict()["retrieved_at"] is None
        assert ts.as_dict()["event_at"] == "2026-09-25T20:00:00+00:00"
        assert "retrieved unknown" in ts.format()
        assert ts.latest() == ts.published
        assert ts.earliest() == ts.event
        assert TimestampSet().latest() is None


# ======================================================================================
# facts
# ======================================================================================


def _row(
    metric, value, start, end, fy, fp, form, filed, src="src_a", basis="gaap", unit="USD", **kw
):
    row = {
        "concept": f"us-gaap:{metric}",
        "metric": metric,
        "value": value,
        "unit": unit,
        "start": start,
        "end": end,
        "fy": fy,
        "fp": fp,
        "form": form,
        "filed": filed,
        "accn": None,
        "frame": None,
        "source_id": src,
        "basis": basis,
        "currency": "USD",
    }
    row.update(kw)
    return row


class TestBuildFacts:
    def test_dedupe_units_and_periods(self):
        rows = [
            _row("revenue", 100.0, "2025-06-29", "2025-09-27", 2025, "Q4", "10-K", "2025-10-31"),
            _row("revenue", 100.0, "2025-06-29", "2025-09-27", 2025, "Q4", "10-K", "2025-10-31"),
            _row(
                "eps_diluted",
                2.14,
                "2025-06-29",
                "2025-09-27",
                2025,
                "Q4",
                "10-K",
                "2025-10-31",
                unit="USD/shares",
            ),
            _row(
                "shares_outstanding",
                15e9,
                None,
                "2025-10-17",
                2025,
                "FY",
                "10-K",
                "2025-10-31",
                unit="shares",
            ),
            _row(
                "gross_margin",
                0.45,
                "2025-06-29",
                "2025-09-27",
                2025,
                "Q4",
                "10-K",
                "2025-10-31",
                unit="pure",
            ),
        ]
        build = build_facts(rows, AS_OF)
        by_metric = {f.metric: f for f in build.facts}
        assert len(build.facts) == 4
        assert by_metric["revenue"].period.label == "Q4 FY2025"
        assert by_metric["revenue"].published_at == datetime(2025, 10, 31, tzinfo=UTC)
        assert by_metric["eps_diluted"].unit == "USD_per_share"
        assert by_metric["shares_outstanding"].period.kind == "instant"
        assert by_metric["gross_margin"].unit == "ratio"
        assert any("duplicate" in n for n in build.notes)
        assert build.conflicts == []

    def test_restatement_later_filing_wins_and_original_preserved(self):
        rows = [
            _row(
                "revenue",
                100.0,
                "2025-06-29",
                "2025-09-27",
                2025,
                "Q4",
                "10-K",
                "2025-10-31",
                src="src_a",
            ),
            _row(
                "revenue",
                108.0,
                "2025-06-29",
                "2025-09-27",
                2025,
                "Q4",
                "10-K",
                "2026-01-30",
                src="src_b",
            ),
        ]
        build = build_facts(rows, AS_OF)
        assert len(build.facts) == 1
        fact = build.facts[0]
        assert fact.value == 108.0
        assert fact.restated is True
        assert fact.original_value == 100.0
        assert fact.original_source_id == "src_a"
        assert fact.source_id == "src_b"
        assert any("restated" in n for n in fact.notes)
        assert any("restated" in n for n in build.notes)
        # 8 % is material -> also surfaced as a resolved restatement conflict
        assert len(build.conflicts) == 1
        conflict = build.conflicts[0]
        assert conflict.reason == "restatement" and conflict.status == "resolved_by_primary"
        assert conflict.material is True
        assert [v.value for v in conflict.values] == [100.0, 108.0]

    def test_small_restatement_is_not_material(self):
        rows = [
            _row("revenue", 100.0, "2025-06-29", "2025-09-27", 2025, "Q4", "10-K", "2025-10-31"),
            _row("revenue", 101.0, "2025-06-29", "2025-09-27", 2025, "Q4", "10-K", "2026-01-30"),
        ]
        build = build_facts(rows, AS_OF)
        assert build.facts[0].restated is True and build.facts[0].value == 101.0
        assert build.conflicts == []

    def test_basis_conflict_preserves_both(self):
        rows = [
            _row(
                "eps_diluted",
                2.14,
                "2025-06-29",
                "2025-09-27",
                2025,
                "Q4",
                "10-K",
                "2025-10-31",
                unit="USD/shares",
            ),
            _row(
                "eps_diluted",
                2.46,
                "2025-06-29",
                "2025-09-27",
                2025,
                "Q4",
                "8-K",
                "2025-10-31",
                src="src_pr",
                basis="adjusted",
                unit="USD/shares",
            ),
        ]
        build = build_facts(rows, AS_OF)
        assert len(build.facts) == 2
        assert len(build.conflicts) == 1
        conflict = build.conflicts[0]
        assert conflict.reason == "basis_mismatch"
        assert conflict.material is True
        assert sorted(v.basis for v in conflict.values) == ["adjusted", "gaap"]
        assert conflict.period_label == "Q4 FY2025"

    def test_same_date_disagreement_is_unknown_conflict(self):
        rows = [
            _row(
                "revenue",
                100.0,
                "2025-06-29",
                "2025-09-27",
                2025,
                "Q4",
                "10-K",
                "2025-10-31",
                src="src_a",
            ),
            _row(
                "revenue",
                102.0,
                "2025-06-29",
                "2025-09-27",
                2025,
                "Q4",
                "10-K",
                "2025-10-31",
                src="src_b",
            ),
        ]
        build = build_facts(rows, AS_OF)
        assert len(build.facts) == 2
        assert len(build.conflicts) == 1
        assert build.conflicts[0].reason == "unknown"
        assert build.conflicts[0].status == "unresolved"
        assert build.conflicts[0].material is False  # 2 % apart

    def test_period_mismatch_conflict(self):
        rows = [
            _row(
                "revenue",
                100.0,
                "2025-07-01",
                "2025-09-30",
                2025,
                "Q4",
                "10-K",
                "2025-10-31",
                src="src_a",
            ),
            _row(
                "revenue",
                110.0,
                "2025-06-29",
                "2025-09-27",
                2025,
                "Q4",
                "10-K",
                "2025-10-31",
                src="src_b",
            ),
        ]
        build = build_facts(rows, AS_OF)
        assert len(build.facts) == 2
        assert [c.reason for c in build.conflicts] == ["period_mismatch"]
        assert build.conflicts[0].material is True

    def test_derived_facts(self):
        rows = [
            _row(
                "operating_cash_flow",
                50.0,
                "2025-06-29",
                "2025-09-27",
                2025,
                "Q4",
                "10-K",
                "2025-10-31",
            ),
            _row("capex", -10.0, "2025-06-29", "2025-09-27", 2025, "Q4", "10-K", "2025-10-31"),
            _row(
                "operating_income",
                30.0,
                "2025-06-29",
                "2025-09-27",
                2025,
                "Q4",
                "10-K",
                "2025-10-31",
            ),
            _row(
                "depreciation_amortization",
                3.0,
                "2025-06-29",
                "2025-09-27",
                2025,
                "Q4",
                "10-K",
                "2025-10-31",
            ),
            _row(
                "operating_cash_flow",
                40.0,
                "2025-03-30",
                "2025-06-28",
                2025,
                "Q3",
                "10-Q",
                "2025-08-01",
            ),
        ]
        build = build_facts(rows, AS_OF)
        derived = {f.metric: f for f in build.facts if f.extraction_method == "derived"}
        assert set(derived) == {"free_cash_flow", "ebitda"}
        assert derived["free_cash_flow"].value == 40.0  # capex sign convention normalised
        assert derived["ebitda"].value == 33.0
        assert derived["free_cash_flow"].period.label == "Q4 FY2025"
        ocf_id = next(
            f.fact_id
            for f in build.facts
            if f.metric == "operating_cash_flow" and f.period.label == "Q4 FY2025"
        )
        capex_id = next(f.fact_id for f in build.facts if f.metric == "capex")
        assert (
            ocf_id in derived["free_cash_flow"].notes[0]
            and capex_id in derived["free_cash_flow"].notes[0]
        )
        assert not any(
            f.metric == "free_cash_flow" and f.period.label == "Q3 FY2025" for f in build.facts
        )

    def test_leakage_guard_drops_and_counts(self):
        rows = [
            _row("revenue", 100.0, "2025-06-29", "2025-09-27", 2025, "Q4", "10-K", "2025-10-31"),
            _row(
                "revenue", 120.0, "2026-06-28", "2026-09-26", 2026, "Q4", "10-K", "2026-10-30"
            ),  # filed after as_of
            _row(
                "revenue", 130.0, "2026-09-27", "2026-12-26", 2027, "Q1", "10-Q", "2026-09-01"
            ),  # period ends after
            _row("revenue", None, "2025-06-29", "2025-09-27", 2025, "Q4", "10-K", "2025-10-31"),
            {
                "metric": "revenue",
                "value": 1.0,
                "unit": "USD",
                "start": None,
                "end": None,
                "filed": None,
                "source_id": "s",
            },
        ]
        build = build_facts(rows, AS_OF)
        assert [f.value for f in build.facts] == [100.0]
        assert build.dropped_count == 4
        reasons = [d["reason"] for d in build.dropped]
        assert sum("after as_of" in r for r in reasons) == 2
        assert any("2 row(s) dated after as_of" in n for n in build.notes)

    def test_deterministic_ids_and_factory(self):
        rows = [
            _row("revenue", 100.0, "2025-06-29", "2025-09-27", 2025, "Q4", "10-K", "2025-10-31"),
            _row("net_income", 25.0, "2025-06-29", "2025-09-27", 2025, "Q4", "10-K", "2025-10-31"),
        ]
        first = build_facts(rows, AS_OF)
        second = build_facts(list(reversed(rows)), AS_OF)
        assert [f.model_dump() for f in first.facts] == [f.model_dump() for f in second.facts]
        assert all(f.fact_id.startswith("fact_") for f in first.facts)
        counter = iter(range(100))
        custom = build_facts(rows, AS_OF, fact_id_factory=lambda identity: f"x{next(counter)}")
        assert [f.fact_id for f in custom.facts] == ["x0", "x1"]

    def test_currency_taken_from_unit(self):
        rows = [
            _row(
                "revenue",
                100.0,
                "2025-01-01",
                "2025-03-31",
                2025,
                "Q1",
                "10-Q",
                "2025-05-01",
                unit="EUR",
            )
        ]
        fact = build_facts(rows, AS_OF).facts[0]
        assert fact.unit == "EUR" and fact.currency == "EUR"


class TestFreshness:
    @pytest.mark.parametrize(
        ("age_days", "kind", "expected"),
        [
            (0, "price", "current"),
            (3, "price", "current"),
            (4, "price", "recent"),
            (14, "price", "recent"),
            (15, "price", "stale"),
            (100, "fiscal_quarter", "current"),
            (101, "quarterly", "recent"),
            (200, "ttm", "recent"),
            (201, "instant", "stale"),
            (380, "fiscal_year", "current"),
            (381, "annual", "recent"),
            (500, "annual", "recent"),
            (501, "fiscal_year", "stale"),
        ],
    )
    def test_buckets(self, age_days, kind, expected):
        when = AS_OF.date() - timedelta(days=age_days)
        assert classify_freshness(when, AS_OF, kind) == expected

    def test_unknown_and_future(self):
        assert classify_freshness(None, AS_OF, "price") == "unknown"
        assert classify_freshness(AS_OF.date() + timedelta(days=1), AS_OF, "price") == "unknown"
        assert classify_freshness(AS_OF - timedelta(days=1), AS_OF, "price") == "current"

    def _facts(self, quarter_end: date, annual_end: date | None = None) -> list[NormalizedFact]:
        facts = [
            NormalizedFact(
                fact_id="q",
                metric="revenue",
                value=1.0,
                unit="USD",
                period=Period(
                    kind="fiscal_quarter",
                    fiscal_year=2026,
                    fiscal_period="Q3",
                    end=quarter_end,
                    label="Q3 FY2026",
                ),
                source_id="s",
            )
        ]
        if annual_end:
            facts.append(
                NormalizedFact(
                    fact_id="a",
                    metric="revenue",
                    value=4.0,
                    unit="USD",
                    period=Period(
                        kind="fiscal_year",
                        fiscal_year=2025,
                        fiscal_period="FY",
                        end=annual_end,
                        label="FY2025",
                    ),
                    source_id="s",
                )
            )
        return facts

    def test_summary_counts_and_warnings(self):
        stale_quarter = AS_OF.date() - timedelta(days=231)
        facts = self._facts(stale_quarter, AS_OF.date() - timedelta(days=300))
        prices = _series([date(2026, 9, 24), date(2026, 9, 25)])
        summary = freshness_summary(facts, prices, AS_OF)
        assert summary["facts"]["stale"] == 1 and summary["facts"]["current"] == 1
        assert summary["facts"]["latest_quarter_age_days"] == 231
        assert summary["facts"]["latest_quarter_freshness"] == "stale"
        assert summary["prices"]["freshness"] == "current"
        assert summary["prices"]["latest_date"] == "2026-09-25"
        assert any(
            "latest quarterly fundamentals are 231 days old" in w for w in summary["warnings"]
        )
        fresh = freshness_summary(self._facts(AS_OF.date() - timedelta(days=40)), prices, AS_OF)
        assert fresh["warnings"] == []
        empty = freshness_summary([], None, AS_OF)
        assert "no normalized fundamentals available" in empty["warnings"]
        assert "no price data available" in empty["warnings"]
        assert empty["prices"]["freshness"] == "unknown"

    def test_detect_stale_mix(self):
        prices = _series([date(2026, 9, 24), date(2026, 9, 25)])
        stale = self._facts(AS_OF.date() - timedelta(days=231))
        warnings = detect_stale_mix(prices, stale, AS_OF)
        assert len(warnings) == 1
        assert "stale quarterly fundamentals" in warnings[0] and "231 days old" in warnings[0]
        assert detect_stale_mix(prices, self._facts(AS_OF.date() - timedelta(days=40)), AS_OF) == []
        # after-hours print + annual-only fundamentals
        ah = _series([date(2026, 9, 25)], price_type="after_hours")
        annual_only = [self._facts(AS_OF.date(), AS_OF.date() - timedelta(days=100))[1]]
        warnings = detect_stale_mix(ah, annual_only, AS_OF)
        assert any("after hours" in w for w in warnings)
        assert any("annual fundamentals only" in w for w in warnings)
        old_prices = _series([date(2026, 7, 1)])
        assert any("price data is stale" in w for w in detect_stale_mix(old_prices, stale, AS_OF))
        assert detect_stale_mix(None, stale, AS_OF) == []


# ======================================================================================
# corporate actions
# ======================================================================================


class TestCorporateActions:
    def test_parse_split_ratio(self):
        assert parse_split_ratio("4:1") == 4.0
        assert parse_split_ratio("4-for-1") == 4.0
        assert parse_split_ratio("7 for 1") == 7.0
        assert parse_split_ratio("1:10", "reverse_split") == pytest.approx(0.1)
        assert parse_split_ratio("10:1", "reverse_split") == pytest.approx(0.1)
        assert parse_split_ratio("1:1") is None
        assert parse_split_ratio("") is None
        assert parse_split_ratio("special dividend") is None

    def test_split_adjustment_returns_copy(self):
        series = PriceSeries(
            symbol="AAPL",
            source_id="s",
            points=[
                PricePoint(
                    date=date(2020, 8, 28),
                    open=500.0,
                    high=510.0,
                    low=490.0,
                    close=500.0,
                    volume=100.0,
                ),
                PricePoint(date=date(2020, 8, 31), close=129.0),
            ],
            retrieved_at=AS_OF,
            split_adjusted=False,
        )
        adjusted = apply_split_adjustment(
            series, [CorporateAction(kind="split", effective=date(2020, 8, 31), detail="4:1")]
        )
        assert adjusted is not series
        assert series.points[0].close == 500.0 and series.split_adjusted is False
        assert adjusted.points[0].close == 125.0
        assert (
            adjusted.points[0].open == 125.0
            and adjusted.points[0].high == 127.5
            and adjusted.points[0].low == 122.5
        )
        assert adjusted.points[0].volume == 400.0
        assert adjusted.points[1].close == 129.0
        assert adjusted.split_adjusted is True
        assert "split-adjusted (4:1 on 2020-08-31)" in adjusted.label
        # reverse split multiplies earlier prices
        reverse = apply_split_adjustment(
            series,
            [CorporateAction(kind="reverse_split", effective=date(2020, 8, 31), detail="1:10")],
        )
        assert reverse.points[0].close == pytest.approx(5000.0)
        # already adjusted -> unchanged copy
        again = apply_split_adjustment(
            adjusted, [CorporateAction(kind="split", effective=date(2020, 8, 31), detail="4:1")]
        )
        assert again.points[0].close == 125.0 and again is not adjusted
        # undated split cannot be applied -> honestly unadjusted
        undated = apply_split_adjustment(series, [CorporateAction(kind="split", detail="4:1")])
        assert undated.split_adjusted is False and "not applied" in undated.label
        assert undated.points[0].close == 500.0
        assert apply_split_adjustment(series, []).split_adjusted is True

    def test_identity_break_warnings(self):
        actions = [
            CorporateAction(kind="merger", effective=date(2024, 5, 1), detail="with Y Corp"),
            CorporateAction(kind="acquisition", effective=date(2025, 1, 1)),
            CorporateAction(kind="spin_off", effective=date(2025, 6, 1)),
            CorporateAction(kind="ticker_change", effective=date(2025, 7, 1), detail="FB -> META"),
            CorporateAction(kind="fiscal_year_change", effective=date(2025, 8, 1)),
            CorporateAction(kind="share_class_change"),
            CorporateAction(kind="dividend", effective=date(2025, 9, 1)),
            CorporateAction(
                kind="name_change", effective=date(2025, 3, 1), detail="formerly Old Name Corp"
            ),
        ]
        warnings = detect_identity_breaks(actions)
        assert len(warnings) == 7
        assert warnings[0].startswith("merger on 2024-05-01 (with Y Corp)")
        assert any("ticker change" in w and "FB -> META" in w for w in warnings)
        # A renamed issuer is the same reporting entity: a mild note, not an identity break.
        (renamed,) = [w for w in warnings if w.startswith("name change on 2025-03-01")]
        assert "formerly Old Name Corp" in renamed
        assert "older filings and coverage appear under the former name" in renamed
        assert "the reporting entity is unchanged" in renamed
        assert not any("not the same" in w for w in warnings if "name change" in w)
        assert any("fiscal-year change" in w for w in warnings)
        assert any("unknown date" in w for w in warnings)
        assert not any("dividend" in w for w in warnings)
        assert detect_identity_breaks([]) == []

    def test_comparable_periods(self):
        def fact(fid, end, restated=False):
            return NormalizedFact(
                fact_id=fid,
                metric="revenue",
                value=1.0,
                unit="USD",
                period=Period(kind="fiscal_quarter", end=end, label=f"quarter to {end}"),
                source_id="s",
                restated=restated,
            )

        facts = [
            fact("a", date(2024, 3, 30)),
            fact("b", date(2024, 6, 29), restated=True),
            fact("c", date(2024, 9, 28)),
        ]
        kept, warnings = comparable_periods(
            facts, [CorporateAction(kind="merger", effective=date(2024, 5, 1))]
        )
        assert [f.fact_id for f in kept] == ["b", "c"]
        assert any("1 fact(s) for periods ending before 2024-05-01 excluded" in w for w in warnings)
        assert any("restated periods" in w for w in warnings)
        kept, warnings = comparable_periods(facts, [CorporateAction(kind="acquisition")])
        assert len(kept) == 3
        assert any("without an effective date" in w for w in warnings)
        kept, warnings = comparable_periods(
            facts,
            [
                CorporateAction(
                    kind="restatement", effective=date(2024, 8, 1), detail="revenue recognition"
                )
            ],
        )
        assert len(kept) == 3
        assert any("restatement on 2024-08-01 (revenue recognition)" in w for w in warnings)
        assert comparable_periods(facts, []) == (
            facts,
            [
                "restated periods in comparison (original values preserved on the facts): "
                "revenue quarter to 2024-06-29"
            ],
        )

    def test_total_return_note(self):
        assert total_return_note([]) is None
        assert total_return_note([CorporateAction(kind="split", detail="4:1")]) is None
        note = total_return_note(
            [
                CorporateAction(kind="dividend", effective=date(2026, 2, 1), detail="$0.25/share"),
                CorporateAction(kind="dividend", effective=date(2026, 5, 1), detail="$0.26/share"),
            ]
        )
        assert note.startswith("2 dividend(s) recorded between 2026-02-01 and 2026-05-01")
        assert "price returns" in note and "total return" in note


# ======================================================================================
# derived fourth quarters (EDGAR never tags a Q4 duration)
# ======================================================================================


def _xbrl_row(
    metric: str, value: float, start: str, end: str, fp: str, filed: str, **overrides
) -> dict:
    row = {
        "concept": f"us-gaap:{metric}",
        "metric": metric,
        "value": value,
        "unit": "USD",
        "start": start,
        "end": end,
        "fy": 2025,
        "fp": fp,
        "form": "10-K" if fp == "FY" else "10-Q",
        "filed": filed,
        "accn": f"acc-{fp}",
        "frame": None,
        "source_id": "src_x",
        "basis": "gaap",
        "currency": "USD",
    }
    row.update(overrides)
    return row


class TestDeriveFourthQuarter:
    def test_derived_q4_is_fy_minus_nine_month_ytd(self):
        rows = [
            _xbrl_row("revenue", 400.0, "2024-09-29", "2025-09-27", "FY", "2025-10-31"),
            _xbrl_row("revenue", 290.0, "2024-09-29", "2025-06-28", "Q3", "2025-08-01"),
        ]
        out = derive_fourth_quarter_rows(rows)
        assert out[:2] == rows  # inputs first, unchanged
        (q4,) = [r for r in out if r.get("extraction_method") == "derived_q4"]
        assert q4["value"] == 110.0 and q4["fp"] == "Q4" and q4["metric"] == "revenue"
        assert (q4["start"], q4["end"]) == ("2025-06-29", "2025-09-27")
        # provenance follows the 10-K, and both contributing rows are listed
        assert q4["filed"] == "2025-10-31" and q4["accn"] == "acc-FY"
        assert q4["source_id"] == "src_x" and q4["form"] == "10-K"
        assert [d["accn"] for d in q4["derived_from"]] == ["acc-FY", "acc-Q3"]
        assert q4["concept"].endswith("(derived Q4)")

    def test_q4_is_never_derived_for_per_share_or_instant_metrics(self):
        for metric in ("eps_diluted", "eps_basic", "shares_outstanding", "total_debt"):
            rows = [
                _xbrl_row(metric, 6.0, "2024-09-29", "2025-09-27", "FY", "2025-10-31"),
                _xbrl_row(metric, 4.5, "2024-09-29", "2025-06-28", "Q3", "2025-08-01"),
            ]
            assert derive_fourth_quarter_rows(rows) == rows, metric

    def test_q4_not_derived_when_an_explicit_fourth_quarter_exists(self):
        rows = [
            _xbrl_row("revenue", 400.0, "2024-09-29", "2025-09-27", "FY", "2025-10-31"),
            _xbrl_row("revenue", 290.0, "2024-09-29", "2025-06-28", "Q3", "2025-08-01"),
            _xbrl_row("revenue", 111.0, "2025-06-29", "2025-09-27", "Q4", "2025-10-31"),
        ]
        out = derive_fourth_quarter_rows(rows)
        assert out == rows
        assert [r["value"] for r in out if r["fp"] == "Q4"] == [111.0]

    def test_q4_needs_matching_basis_currency_and_fiscal_year_start(self):
        fy = _xbrl_row("revenue", 400.0, "2024-09-29", "2025-09-27", "FY", "2025-10-31")
        ytd = _xbrl_row("revenue", 290.0, "2024-09-29", "2025-06-28", "Q3", "2025-08-01")
        assert derive_fourth_quarter_rows([fy, {**ytd, "basis": "adjusted"}]) == [
            fy,
            {**ytd, "basis": "adjusted"},
        ]
        assert derive_fourth_quarter_rows([fy, {**ytd, "currency": "EUR"}]) == [
            fy,
            {**ytd, "currency": "EUR"},
        ]
        shifted = {**ytd, "start": "2024-10-01"}
        assert derive_fourth_quarter_rows([fy, shifted]) == [fy, shifted]
        assert derive_fourth_quarter_rows([]) == []
