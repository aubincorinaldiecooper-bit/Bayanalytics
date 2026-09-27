"""Market session and time normalization (AGENT.md section 34).

"Current" has several meanings for a price. This module labels a timestamp as intraday,
pre-market, after-hours or latest completed close in the exchange's own timezone, finds
the latest completed close in a series as of a moment in time, and carries the three
timestamps that matter for any piece of evidence (event, publication, retrieval) without
collapsing them into one.

Regular session hours default to the US equity schedule in ``America/New_York``:
pre-market 04:00-09:30, regular 09:30-16:00, after-hours 16:00-20:00. Weekends and the
hours outside those windows classify as ``latest_close`` for the most recent completed
weekday session.

Exchange holidays are NOT built in. Every function that walks back to a completed session
accepts ``holidays`` (any collection of ``date``); pass the exchange's holiday calendar
there (for example the NYSE closures for the year) and those dates are skipped exactly
like weekends. Without it a holiday close will be reported one session too late.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Collection
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, TypedDict
from zoneinfo import ZoneInfo

from bayanalytics.schemas.common import PriceType
from bayanalytics.schemas.evidence import PricePoint, PriceSeries

PRE_MARKET_OPEN = time(4, 0)
REGULAR_OPEN = time(9, 30)
REGULAR_CLOSE = time(16, 0)
AFTER_HOURS_CLOSE = time(20, 0)

DEFAULT_EXCHANGE_TIMEZONE = "America/New_York"


class SessionInfo(TypedDict):
    price_type: PriceType
    session_date: date
    latest_completed_session: date
    label: str
    exchange_time: str
    exchange_timezone: str


def to_exchange_time(
    ts: datetime, exchange_timezone: str = DEFAULT_EXCHANGE_TIMEZONE, assume_naive: str = "UTC"
) -> datetime:
    """Convert ``ts`` to the exchange timezone. A naive datetime has no timezone; it is
    interpreted in ``assume_naive`` (UTC by default) rather than guessed."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=ZoneInfo(assume_naive) if assume_naive != "UTC" else UTC)
    return ts.astimezone(ZoneInfo(exchange_timezone))


def is_trading_day(d: date, holidays: Collection[date] | None = None) -> bool:
    """Weekday and not in ``holidays`` (which the caller supplies; none are built in)."""
    if d.weekday() >= 5:
        return False
    return holidays is None or d not in holidays


def previous_trading_day(d: date, holidays: Collection[date] | None = None) -> date:
    """The most recent trading day strictly before ``d``."""
    candidate = d - timedelta(days=1)
    while not is_trading_day(candidate, holidays):
        candidate -= timedelta(days=1)
    return candidate


def latest_completed_session(
    ts: datetime,
    exchange_timezone: str = DEFAULT_EXCHANGE_TIMEZONE,
    holidays: Collection[date] | None = None,
) -> date:
    """The date of the most recent regular session that had closed at ``ts``.

    A weekday at or after 16:00 exchange time counts as that day; earlier in the day, on a
    weekend or on a supplied holiday it walks back to the previous trading day.
    """
    local = to_exchange_time(ts, exchange_timezone)
    today = local.date()
    if is_trading_day(today, holidays) and local.time() >= REGULAR_CLOSE:
        return today
    return previous_trading_day(today, holidays)


def classify_price_timestamp(
    ts: datetime,
    exchange_timezone: str = DEFAULT_EXCHANGE_TIMEZONE,
    holidays: Collection[date] | None = None,
) -> SessionInfo:
    """Label a price timestamp by market session in the exchange's timezone.

    ``price_type`` is ``pre_market`` (04:00-09:30), ``intraday`` (09:30-16:00), ``after_hours``
    (16:00-20:00) or ``latest_close`` (any other time, weekends, holidays). ``session_date`` is
    the session the timestamp belongs to (for ``latest_close`` it is the most recent
    completed session); ``latest_completed_session`` is always the last session that had
    closed. A pre-market or after-hours price is never the regular-session close, and the
    label says so.
    """
    local = to_exchange_time(ts, exchange_timezone)
    today = local.date()
    clock = local.time()
    completed = latest_completed_session(ts, exchange_timezone, holidays)
    stamp = local.strftime("%Y-%m-%d %H:%M")
    trading = is_trading_day(today, holidays)

    price_type: PriceType
    if trading and PRE_MARKET_OPEN <= clock < REGULAR_OPEN:
        price_type = "pre_market"
        session_date = today
        text = (
            f"pre-market {stamp} {exchange_timezone} (not a regular-session close; "
            f"latest completed close {completed.isoformat()})"
        )
    elif trading and REGULAR_OPEN <= clock < REGULAR_CLOSE:
        price_type = "intraday"
        session_date = today
        text = (
            f"intraday {stamp} {exchange_timezone} (session in progress; "
            f"latest completed close {completed.isoformat()})"
        )
    elif trading and REGULAR_CLOSE <= clock < AFTER_HOURS_CLOSE:
        price_type = "after_hours"
        session_date = today
        text = (
            f"after-hours {stamp} {exchange_timezone} (not the regular-session close; "
            f"regular session {completed.isoformat()} closed)"
        )
    else:
        price_type = "latest_close"
        session_date = completed
        why = "weekend" if today.weekday() >= 5 else ("holiday" if not trading else "closed")
        text = (
            f"latest close {completed.isoformat()} (market {why} at {stamp} {exchange_timezone})"
        )
    return SessionInfo(
        price_type=price_type,
        session_date=session_date,
        latest_completed_session=completed,
        label=text,
        exchange_time=local.isoformat(),
        exchange_timezone=exchange_timezone,
    )


def latest_completed_close(
    series: PriceSeries, as_of: datetime, holidays: Collection[date] | None = None
) -> PricePoint | None:
    """The last point of ``series`` dated on or before the latest completed session at
    ``as_of`` (in the series' exchange timezone). Points after that date, including a partial
    session in progress, are never returned. ``None`` when nothing qualifies."""
    cutoff = latest_completed_session(as_of, series.exchange_timezone, holidays)
    candidates = [point for point in series.points if point.date <= cutoff]
    if not candidates:
        return None
    return max(candidates, key=lambda point: point.date)


def label_series(
    series: PriceSeries, as_of: datetime, holidays: Collection[date] | None = None
) -> PriceSeries:
    """Return a copy of ``series`` with ``price_type``, ``session_date`` and ``label`` set for
    its latest point relative to ``as_of``. The input is not mutated.

    * the latest point falls on a session still in progress at ``as_of`` -> ``intraday``;
    * the latest point is the latest completed session -> ``latest_close``;
    * the latest point is older -> ``latest_close`` with the number of completed sessions
      the series is behind, so a stale series is never mistaken for current;
    * an empty series is labelled as such.
    The rest of the points are historical closes and keep their dates.
    """
    if not series.points:
        return series.model_copy(update={"label": "empty price series", "session_date": None})
    latest_point = max(series.points, key=lambda point: point.date)
    info = classify_price_timestamp(as_of, series.exchange_timezone, holidays)
    completed = info["latest_completed_session"]
    latest_date = latest_point.date
    price_type: PriceType
    if latest_date > completed:
        price_type = "intraday" if info["price_type"] == "intraday" else "historical_close"
        if price_type == "intraday":
            text = (
                f"intraday {latest_date.isoformat()} (session in progress at "
                f"{info['exchange_time']}; latest completed close {completed.isoformat()})"
            )
        else:
            text = (
                f"price dated {latest_date.isoformat()} is after the latest completed session "
                f"{completed.isoformat()} at {info['exchange_time']}; treat as unverified"
            )
    elif latest_date == completed:
        price_type = "latest_close"
        text = f"latest close {latest_date.isoformat()} ({series.exchange_timezone})"
    else:
        price_type = "latest_close"
        behind = 0
        cursor = completed
        while cursor > latest_date:
            behind += 1
            cursor = previous_trading_day(cursor, holidays)
        text = (
            f"latest available close {latest_date.isoformat()}; {behind} completed session(s) "
            f"after it up to {completed.isoformat()} are not in the series"
        )
    return series.model_copy(
        update={"price_type": price_type, "session_date": latest_date, "label": text}
    )


@dataclasses.dataclass(frozen=True)
class TimestampSet:
    """The three timestamps that matter for a piece of evidence, kept separate.

    ``event`` is when the thing happened (a close, a filing period end, an announcement),
    ``published`` when the source published it, ``retrieved`` when we fetched it. Any of them
    may be unknown; none of them is ever substituted for another.
    """

    event: datetime | None = None
    published: datetime | None = None
    retrieved: datetime | None = None

    def as_dict(self) -> dict[str, str | None]:
        return {
            "event_at": self.event.isoformat() if self.event else None,
            "published_at": self.published.isoformat() if self.published else None,
            "retrieved_at": self.retrieved.isoformat() if self.retrieved else None,
        }

    def format(self) -> str:
        parts = []
        for name, value in (
            ("event", self.event),
            ("published", self.published),
            ("retrieved", self.retrieved),
        ):
            parts.append(f"{name} {value.isoformat() if value else 'unknown'}")
        return " · ".join(parts)

    def latest(self) -> datetime | None:
        known = [value for value in (self.event, self.published, self.retrieved) if value]
        return max(known) if known else None

    def earliest(self) -> datetime | None:
        known = [value for value in (self.event, self.published, self.retrieved) if value]
        return min(known) if known else None


def session_view(price: float, series: PriceSeries, retrieved_at: datetime | None) -> dict[str, Any]:
    """The section-34 example shape for one price: type, exchange, timezone, session date."""
    return {
        "price": price,
        "price_type": series.price_type,
        "exchange": series.exchange,
        "exchange_timezone": series.exchange_timezone,
        "session_date": series.session_date.isoformat() if series.session_date else None,
        "retrieved_at": (retrieved_at or series.retrieved_at).isoformat(),
        "label": series.label,
    }
