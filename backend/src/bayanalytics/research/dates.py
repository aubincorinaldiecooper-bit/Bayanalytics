"""Lenient timestamp parsing for publisher metadata and search backends.

Publishers and search engines emit dates in many shapes (ISO 8601 with or without zone,
RFC 2822, "January 5, 2026", bare dates). Everything parsed here is returned tz-aware in UTC
so freshness comparisons against ``as_of`` never mix naive and aware datetimes.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime

_TEXT_FORMATS: tuple[str, ...] = (
    "%B %d, %Y %H:%M",
    "%B %d, %Y",
    "%b %d, %Y %H:%M",
    "%b %d, %Y",
    "%d %B %Y",
    "%d %b %Y",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%Y%m%d",
)


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_datetime_lenient(value: object) -> datetime | None:
    """Parse ``value`` into a UTC datetime, returning ``None`` when nothing sensible matches."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    text = str(value).strip()
    if not text:
        return None
    if len(text) > 64:
        text = text[:64]
    iso = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        return ensure_utc(datetime.fromisoformat(iso))
    except ValueError:
        pass
    try:
        parsed = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        parsed = None
    if parsed is not None:
        return ensure_utc(parsed)
    for fmt in _TEXT_FORMATS:
        try:
            return ensure_utc(datetime.strptime(text, fmt))
        except ValueError:
            continue
    return None


def parse_date_lenient(value: object) -> date | None:
    parsed = parse_datetime_lenient(value)
    return parsed.date() if parsed else None
