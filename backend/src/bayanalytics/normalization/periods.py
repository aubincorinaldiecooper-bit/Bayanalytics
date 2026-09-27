"""Fiscal period construction and labelling (AGENT.md sections 3.1, 6 and 34).

Periods carry their kind explicitly (fiscal quarter, fiscal year, TTM, YTD, instant, calendar
range) so two values are never compared without the difference in period being visible.

Fiscal versus calendar: a company's fiscal quarter only coincides with a calendar quarter
when its fiscal year ends on 31 December. ``calendar_to_fiscal`` converts a calendar date
into the (fiscal_year, fiscal_period) it falls in for a given fiscal year end; without that
fiscal calendar "current quarter" is ambiguous and must not be assumed
(:data:`CURRENT_QUARTER_AMBIGUOUS`).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from datetime import date, timedelta

from bayanalytics.schemas.evidence import Period, PeriodKind

QUARTER_DAYS = (80, 100)
HALF_YEAR_DAYS = (170, 200)
NINE_MONTH_DAYS = (255, 290)
YEAR_DAYS = (350, 380)

CURRENT_QUARTER_AMBIGUOUS = (
    '"current quarter" is ambiguous unless the company fiscal calendar (fiscal year end) is '
    "known; calendar quarters were not assumed to be fiscal quarters"
)

_QUARTER_PERIODS = ("Q1", "Q2", "Q3", "Q4")


def _to_date(value: date | str | None) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def classify_duration(
    start: date | None, end: date | None, fp: str | None = None
) -> tuple[PeriodKind, str | None]:
    """Classify a reporting duration into a period kind, with an explanatory note.

    * ``start is None`` -> ``instant`` (a balance-sheet date).
    * 80-100 days -> ``fiscal_quarter``; 350-380 days -> ``fiscal_year``.
    * 170-200 or 255-290 days -> ``ytd`` (six- or nine-month cumulative figures from 10-Qs).
    * anything else -> ``calendar_range`` with a note stating the day count, because the
      value cannot be lined up with a fiscal quarter or year without more information.
    """
    if end is None:
        return "unknown", "period end date missing"
    if start is None:
        return "instant", None
    days = (end - start).days + 1
    if QUARTER_DAYS[0] <= days <= QUARTER_DAYS[1]:
        return "fiscal_quarter", None
    if YEAR_DAYS[0] <= days <= YEAR_DAYS[1]:
        return "fiscal_year", None
    if HALF_YEAR_DAYS[0] <= days <= HALF_YEAR_DAYS[1] or (
        NINE_MONTH_DAYS[0] <= days <= NINE_MONTH_DAYS[1]
    ):
        return "ytd", f"{days}-day cumulative (year-to-date) duration, not a single quarter"
    return "calendar_range", f"{days}-day duration does not match a fiscal quarter or year"


def period_from_xbrl(
    start: date | str | None,
    end: date | str | None,
    fy: int | None = None,
    fp: str | None = None,
    form: str | None = None,
    kind_hint: PeriodKind | None = None,
) -> Period:
    """Build a ``Period`` from XBRL context dates plus the filing's fy/fp/form.

    ``kind_hint`` makes TTM and YTD explicit when the caller already knows the duration is a
    trailing or cumulative window (XBRL itself never labels them). Otherwise the kind follows
    :func:`classify_duration`. For a quarter-length duration reported in a 10-K with
    ``fp == "FY"`` the fiscal period is set to ``Q4`` (the only quarter that ends on the fiscal
    year end). The label is produced by :func:`label`.
    """
    start_date = _to_date(start)
    end_date = _to_date(end)
    fiscal_period = fp.upper() if isinstance(fp, str) and fp else None
    if kind_hint is not None:
        kind: PeriodKind = kind_hint
    else:
        kind, _note = classify_duration(start_date, end_date, fiscal_period)

    if kind == "fiscal_quarter":
        if fiscal_period not in _QUARTER_PERIODS:
            fiscal_period = "Q4" if fiscal_period == "FY" else fiscal_period
    elif kind == "fiscal_year":
        fiscal_period = "FY"
    elif kind == "ytd":
        if fiscal_period == "FY" or fiscal_period is None:
            fiscal_period = None
    elif kind == "ttm":
        fiscal_period = None

    period = Period(
        kind=kind,
        fiscal_year=fy,
        fiscal_period=fiscal_period,
        start=start_date,
        end=end_date,
    )
    period.label = label(period)
    return period


def label(period: Period) -> str:
    """Human label: ``FY2025``, ``Q3 FY2025``, ``TTM to 2025-06-28``, ``as of 2025-09-27``."""
    kind = period.kind
    fy = period.fiscal_year
    fp = period.fiscal_period
    end = period.end.isoformat() if period.end else None
    if kind == "fiscal_year":
        if fy is not None:
            return f"FY{fy}"
        return f"fiscal year to {end}" if end else "fiscal year"
    if kind == "fiscal_quarter":
        if fp and fy is not None:
            return f"{fp} FY{fy}"
        if fp and end:
            return f"{fp} to {end}"
        return f"quarter to {end}" if end else "fiscal quarter"
    if kind == "ttm":
        return f"TTM to {end}" if end else "TTM"
    if kind == "ytd":
        prefix = f"{fp} " if fp else ""
        if end:
            return f"{prefix}YTD to {end}"
        return f"{prefix}YTD FY{fy}" if fy is not None else f"{prefix}YTD"
    if kind == "qtd":
        return f"QTD to {end}" if end else "QTD"
    if kind == "instant":
        return f"as of {end}" if end else "instant"
    if kind == "calendar_range":
        start = period.start.isoformat() if period.start else "?"
        return f"{start} to {end or '?'}"
    return f"unknown period ({end})" if end else "unknown period"


_LABEL_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:FY|FISCAL\s*YEAR|FISCAL)\s*'?(\d{2}|\d{4})$"), "fy"),
    (re.compile(r"^Q([1-4])\s*[- ]?\s*(?:FY)?\s*'?(\d{2}|\d{4})$"), "q_year"),
    (re.compile(r"^(\d{4})\s*[- ]?\s*Q([1-4])$"), "year_q"),
    (re.compile(r"^([1-4])Q\s*'?(\d{2}|\d{4})$"), "q_year"),
    (re.compile(r"^H([12])\s*[- ]?\s*(?:FY)?\s*'?(\d{2}|\d{4})$"), "h_year"),
    (re.compile(r"^TTM(?:\s+TO\s+(\d{4}-\d{2}-\d{2}))?$"), "ttm"),
    (re.compile(r"^YTD(?:\s+TO\s+(\d{4}-\d{2}-\d{2}))?$"), "ytd"),
    (re.compile(r"^(?:AS\s+OF\s+)?(\d{4}-\d{2}-\d{2})$"), "instant"),
)


def _expand_year(text: str) -> int:
    year = int(text)
    if year < 100:
        year += 2000
    return year


def parse_period_label(text: str) -> Period:
    """Parse labels such as ``Q3 FY25``, ``FY2025``, ``2025-Q4``, ``Q4 2025``, ``3Q25``,
    ``TTM``, ``TTM to 2025-06-28``, ``FY 2024``, ``H1 FY2025`` into a ``Period``.

    Two-digit years are 20xx. Raises ``ValueError`` for anything unrecognised; a label is
    never guessed. Dates are not filled in because a label alone does not fix them.
    """
    if text is None:
        raise ValueError("no period label")
    cleaned = " ".join(text.strip().upper().split())
    if not cleaned:
        raise ValueError("empty period label")
    for pattern, kind in _LABEL_PATTERNS:
        match = pattern.match(cleaned)
        if match is None:
            continue
        period: Period
        if kind == "fy":
            period = Period(kind="fiscal_year", fiscal_year=_expand_year(match.group(1)))
            period.fiscal_period = "FY"
        elif kind == "q_year":
            period = Period(
                kind="fiscal_quarter",
                fiscal_year=_expand_year(match.group(2)),
                fiscal_period=f"Q{match.group(1)}",
            )
        elif kind == "year_q":
            period = Period(
                kind="fiscal_quarter",
                fiscal_year=_expand_year(match.group(1)),
                fiscal_period=f"Q{match.group(2)}",
            )
        elif kind == "h_year":
            period = Period(
                kind="ytd",
                fiscal_year=_expand_year(match.group(2)),
                fiscal_period=f"H{match.group(1)}",
            )
        elif kind == "ttm":
            period = Period(kind="ttm", end=_to_date(match.group(1)))
        elif kind == "ytd":
            period = Period(kind="ytd", end=_to_date(match.group(1)))
        else:
            period = Period(kind="instant", end=_to_date(match.group(1)))
        period.label = label(period)
        return period
    raise ValueError(f"unrecognised period label: {text!r}")


def is_consecutive_quarters(earlier: Period, later: Period) -> bool:
    """True when ``later`` is the fiscal quarter immediately after ``earlier``.

    Decided from end dates (a gap of 80-100 days) so it works even when fiscal_year and
    fiscal_period are missing; when both carry fy/fp those must also be sequential.
    """
    if earlier.end is None or later.end is None:
        return False
    gap = (later.end - earlier.end).days
    if not (QUARTER_DAYS[0] <= gap <= QUARTER_DAYS[1]):
        return False
    if (
        earlier.fiscal_year is not None
        and later.fiscal_year is not None
        and earlier.fiscal_period in _QUARTER_PERIODS
        and later.fiscal_period in _QUARTER_PERIODS
    ):
        e_index = _QUARTER_PERIODS.index(earlier.fiscal_period)  # type: ignore[arg-type]
        l_index = _QUARTER_PERIODS.index(later.fiscal_period)  # type: ignore[arg-type]
        if e_index == 3:
            return l_index == 0 and later.fiscal_year == earlier.fiscal_year + 1
        return l_index == e_index + 1 and later.fiscal_year == earlier.fiscal_year
    return True


def ttm_window(quarters: Iterable[Period]) -> list[Period] | None:
    """The four most recent consecutive fiscal quarters, oldest first, or ``None``.

    Only ``fiscal_quarter`` periods with an end date are considered. Duplicates of the same
    period key are collapsed. ``None`` when fewer than four quarters exist or the latest four
    are not consecutive (a gap means the window is not a true trailing twelve months).
    """
    unique: dict[str, Period] = {}
    for period in quarters:
        if period.kind != "fiscal_quarter" or period.end is None:
            continue
        unique.setdefault(period.key(), period)
    ordered = sorted(unique.values(), key=lambda p: p.end)  # type: ignore[arg-type, return-value]
    if len(ordered) < 4:
        return None
    window = ordered[-4:]
    for earlier, later in zip(window, window[1:], strict=False):
        if not is_consecutive_quarters(earlier, later):
            return None
    return window


def ttm_period(window: Sequence[Period]) -> Period:
    """The TTM period covering four consecutive quarters (labelled ``TTM to <end>``)."""
    first, last = window[0], window[-1]
    period = Period(kind="ttm", fiscal_year=last.fiscal_year, start=first.start, end=last.end)
    period.label = label(period)
    return period


def _parse_fiscal_year_end(fiscal_year_end: str) -> tuple[int, int]:
    digits = re.sub(r"\D", "", fiscal_year_end or "")
    if len(digits) != 4:
        raise ValueError(f"fiscal_year_end must be MMDD, got {fiscal_year_end!r}")
    month, day = int(digits[:2]), int(digits[2:])
    if not 1 <= month <= 12 or not 1 <= day <= 31:
        raise ValueError(f"fiscal_year_end must be MMDD, got {fiscal_year_end!r}")
    return month, day


def _fye_in_year(year: int, month: int, day: int) -> date:
    while True:
        try:
            return date(year, month, day)
        except ValueError:
            day -= 1


def calendar_to_fiscal(d: date, fiscal_year_end: str) -> tuple[int, str]:
    """Map a calendar date to ``(fiscal_year, fiscal_period)`` for a fiscal year ending on
    ``fiscal_year_end`` (``"MMDD"``, e.g. ``"0930"`` for Apple, ``"1231"`` for a calendar year).

    The fiscal year is named by the calendar year in which it ends (Apple's FY2026 runs
    2025-10-01 .. 2026-09-30, so 2025-11-15 -> ``(2026, "Q1")``). Quarters are whole months
    counted from the fiscal year start; 52/53-week calendars whose quarters end on a fixed
    weekday can differ by a few days around quarter boundaries, which is why this function
    is for labelling calendar dates and never replaces the periods reported in a filing.
    """
    month, day = _parse_fiscal_year_end(fiscal_year_end)
    fye_this_year = _fye_in_year(d.year, month, day)
    fiscal_year = d.year if d <= fye_this_year else d.year + 1
    fy_start = _fye_in_year(fiscal_year - 1, month, day) + timedelta(days=1)
    months_elapsed = (d.year - fy_start.year) * 12 + (d.month - fy_start.month)
    if d.day < fy_start.day:
        months_elapsed -= 1
    months_elapsed = max(0, min(11, months_elapsed))
    quarter = months_elapsed // 3 + 1
    return fiscal_year, f"Q{quarter}"


def current_fiscal_quarter(
    d: date, fiscal_year_end: str | None
) -> tuple[tuple[int, str] | None, str | None]:
    """``((fiscal_year, fiscal_period), None)`` when the fiscal calendar is known, otherwise
    ``(None, CURRENT_QUARTER_AMBIGUOUS)``: the current quarter is never assumed."""
    if not fiscal_year_end:
        return None, CURRENT_QUARTER_AMBIGUOUS
    return calendar_to_fiscal(d, fiscal_year_end), None


def calendar_quarter(d: date) -> tuple[int, str]:
    """The calendar quarter a date falls in, as ``(year, "Q<n>")``. Not a fiscal quarter."""
    return d.year, f"Q{(d.month - 1) // 3 + 1}"
