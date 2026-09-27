"""Unit, scale and currency normalization (AGENT.md section 3.1).

The rule is simple: nothing enters a comparison with an inconsistent unit silently. This
module parses the number formats found in filings, press releases and market-data pages into
``(value, unit, currency, scale_applied)`` and provides the guards that refuse to mix
currencies.

Unit names produced here match ``NormalizedFact.unit``: a currency code (``USD``, ``EUR``)
for money, ``percent``, ``ratio``, ``shares``, ``days`` and ``number`` when the text carries no
unit at all. Money per share uses ``<CCY>_per_share``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, NamedTuple

CURRENCY_SYMBOL_TO_CODE: dict[str, str] = {
    "$": "USD",
    "US$": "USD",
    "USD": "USD",
    "€": "EUR",
    "EUR": "EUR",
    "£": "GBP",
    "GBP": "GBP",
    # The yen sign is also used for the renminbi; a bare sign is mapped to JPY and callers
    # that know better should pass an explicit code.
    "¥": "JPY",
    "JPY": "JPY",
    "CN¥": "CNY",
    "RMB": "CNY",
    "CNY": "CNY",
    "C$": "CAD",
    "CA$": "CAD",
    "CAD": "CAD",
    "A$": "AUD",
    "AU$": "AUD",
    "AUD": "AUD",
    "HK$": "HKD",
    "HKD": "HKD",
    "CHF": "CHF",
    "₹": "INR",
    "INR": "INR",
    "₩": "KRW",
    "KRW": "KRW",
    "NZ$": "NZD",
    "NZD": "NZD",
    "SEK": "SEK",
    "NOK": "NOK",
    "DKK": "DKK",
    "SGD": "SGD",
    "S$": "SGD",
    "TWD": "TWD",
    "NT$": "TWD",
    "BRL": "BRL",
    "R$": "BRL",
    "MXN": "MXN",
    "ZAR": "ZAR",
}

SCALE_MULTIPLIERS: dict[str, float] = {
    "": 1.0,
    "K": 1e3,
    "M": 1e6,
    "B": 1e9,
    "T": 1e12,
}

_SCALE_WORDS: dict[str, str] = {
    "k": "K",
    "thousand": "K",
    "thousands": "K",
    "m": "M",
    "mm": "M",
    "mn": "M",
    "mil": "M",
    "million": "M",
    "millions": "M",
    "b": "B",
    "bn": "B",
    "bil": "B",
    "billion": "B",
    "billions": "B",
    "t": "T",
    "tn": "T",
    "trillion": "T",
    "trillions": "T",
}

_UNIT_WORDS: dict[str, str] = {
    "%": "percent",
    "percent": "percent",
    "pct": "percent",
    "x": "ratio",
    "×": "ratio",
    "times": "ratio",
    "shares": "shares",
    "share": "shares",
    "shs": "shares",
    "days": "days",
    "day": "days",
    "bp": "bp",
    "bps": "bp",
}

_MINUS_CHARS = "−–—-"  # unicode minus, en dash, em dash, hyphen-minus


class ParsedNumber(NamedTuple):
    value: float
    unit: str
    currency: str | None
    scale_applied: str


_SYMBOLS_BY_LENGTH = sorted(CURRENCY_SYMBOL_TO_CODE, key=len, reverse=True)
_NUMBER_RE = re.compile(r"^(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?$|^\.(\d+)$")


def normalize_currency_code(symbol_or_code: str | None) -> str:
    """Map a currency symbol or code (``$``, ``US$``, ``usd``, ``€``) to its ISO 4217 code.

    Raises ``ValueError`` for anything unrecognised: an unknown currency must never be
    guessed. A bare ``¥`` maps to ``JPY`` (see the table for the caveat).
    """
    if symbol_or_code is None:
        raise ValueError("currency is missing")
    text = symbol_or_code.strip()
    if not text:
        raise ValueError("currency is empty")
    if text in CURRENCY_SYMBOL_TO_CODE:
        return CURRENCY_SYMBOL_TO_CODE[text]
    upper = text.upper()
    if upper in CURRENCY_SYMBOL_TO_CODE:
        return CURRENCY_SYMBOL_TO_CODE[upper]
    if re.fullmatch(r"[A-Z]{3}", upper):
        return upper
    raise ValueError(f"unrecognised currency: {symbol_or_code!r}")


def scale_to_base(value: float, scale: str | None) -> float:
    """Multiply ``value`` by the scale it was quoted in ("M", "billion", "bn", ...).

    ``None`` or ``""`` means no scale. Unknown scale words raise ``ValueError``.
    """
    key = canonical_scale(scale)
    return float(value) * SCALE_MULTIPLIERS[key]


def canonical_scale(scale: str | None) -> str:
    """Normalise a scale token to one of ``"", "K", "M", "B", "T"``."""
    if scale is None:
        return ""
    token = scale.strip()
    if token == "":
        return ""
    if token in SCALE_MULTIPLIERS:
        return token
    lowered = token.lower()
    if lowered in _SCALE_WORDS:
        return _SCALE_WORDS[lowered]
    raise ValueError(f"unrecognised scale: {scale!r}")


def _strip_currency(text: str) -> tuple[str, str | None]:
    """Remove a leading or trailing currency symbol/code and return (rest, currency)."""
    for symbol in _SYMBOLS_BY_LENGTH:
        if text.startswith(symbol):
            rest = text[len(symbol) :].lstrip()
            if _looks_numeric_start(rest):
                return rest, CURRENCY_SYMBOL_TO_CODE[symbol]
        if text.endswith(symbol) and len(text) > len(symbol):
            rest = text[: -len(symbol)].rstrip()
            boundary_ok = symbol.isalpha() is False or rest[-1:] in " )0123456789."
            if boundary_ok and rest:
                return rest, CURRENCY_SYMBOL_TO_CODE[symbol]
    upper = text.upper()
    for symbol in _SYMBOLS_BY_LENGTH:
        if symbol.isalpha() and upper.startswith(symbol + " "):
            return text[len(symbol) :].lstrip(), CURRENCY_SYMBOL_TO_CODE[symbol]
        if symbol.isalpha() and upper.endswith(" " + symbol):
            return text[: -len(symbol)].rstrip(), CURRENCY_SYMBOL_TO_CODE[symbol]
    return text, None


def _looks_numeric_start(text: str) -> bool:
    return bool(text) and (text[0].isdigit() or text[0] in _MINUS_CHARS + "+(.")


def parse_number(text: str) -> ParsedNumber:
    """Parse a human-formatted number into ``(value, unit, currency, scale_applied)``.

    Handles currency symbols and codes (``$1.2B``, ``€3.4M``, ``USD 1,234``, ``1,234 USD``),
    thousands separators (``12,345.67``), accounting negatives ``(1,234)``, unicode minus
    ``−7.4%``, percent, multiples (``1.5x`` / ``1.5×``), share counts (``15.3B shares``), and
    scale suffixes or words (``K/M/B/T``, ``mn/bn``, ``million``, ``billion``). ``value`` is the
    number in base units (``$1.2B`` -> ``1.2e9``) and ``scale_applied`` records the canonical
    scale (``"B"``) that was multiplied in, or ``""``.

    Raises ``ValueError`` when the text is not a number; see :func:`try_parse_number` for the
    non-raising form. Nothing is ever guessed: a currency is reported only when one is
    written in the text.
    """
    if text is None:
        raise ValueError("no text to parse")
    original = text
    text = text.strip().replace(" ", " ")
    if not text:
        raise ValueError("empty text")

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1].strip()

    text, currency = _strip_currency(text)

    if text and text[0] in _MINUS_CHARS:
        negative = True
        text = text[1:].lstrip()
    elif text.startswith("+"):
        text = text[1:].lstrip()

    if currency is None:
        text, currency = _strip_currency(text)
        if text and text[0] in _MINUS_CHARS:
            negative = True
            text = text[1:].lstrip()

    unit = "number"
    scale = ""
    if text.endswith("%"):
        unit = "percent"
        text = text[:-1].rstrip()

    # Trailing unit words / scale words, possibly several ("1.2 billion shares").
    tokens = text.split()
    while len(tokens) > 1:
        tail = tokens[-1].lower()
        if tail in _UNIT_WORDS:
            unit = _merge_unit(unit, _UNIT_WORDS[tail])
            tokens.pop()
            continue
        if tail in _SCALE_WORDS:
            scale = _SCALE_WORDS[tail]
            tokens.pop()
            continue
        break
    text = " ".join(tokens)

    # Attached suffixes: "1.2B", "1.5x", "18.2pct", "1.2bn", "3.4M".
    match = re.fullmatch(r"([^A-Za-z×%]+?)\s*([A-Za-z×]+)?", text)
    if match is None:
        raise ValueError(f"not a number: {original!r}")
    number_text, suffix = match.group(1).strip(), match.group(2)
    if suffix:
        lowered = suffix.lower()
        if lowered in _UNIT_WORDS:
            unit = _merge_unit(unit, _UNIT_WORDS[lowered])
        elif lowered in _SCALE_WORDS:
            if scale:
                raise ValueError(f"two scales in {original!r}")
            scale = _SCALE_WORDS[lowered]
        elif lowered in ("percent",):
            unit = "percent"
        else:
            raise ValueError(f"unrecognised suffix {suffix!r} in {original!r}")

    number_text = number_text.replace(" ", "")
    if number_text.startswith("(") and number_text.endswith(")"):
        negative = True
        number_text = number_text[1:-1]
    if not _NUMBER_RE.fullmatch(number_text):
        raise ValueError(f"not a number: {original!r}")
    value = float(number_text.replace(",", ""))
    value = scale_to_base(value, scale)
    if negative:
        value = -value

    if currency is not None:
        unit = f"{currency}_per_share" if unit == "shares" else currency
    return ParsedNumber(value=value, unit=unit, currency=currency, scale_applied=scale)


def _merge_unit(current: str, incoming: str) -> str:
    if current == "number":
        return incoming
    if current == incoming:
        return current
    raise ValueError(f"conflicting units {current!r} and {incoming!r}")


def try_parse_number(text: str | None) -> ParsedNumber | None:
    """:func:`parse_number` that returns ``None`` instead of raising."""
    if text is None:
        return None
    try:
        return parse_number(text)
    except ValueError:
        return None


def _currency_of(item: Any) -> str | None:
    if isinstance(item, dict):
        return item.get("currency")
    return getattr(item, "currency", None)


def assert_same_currency(facts: Iterable[Any]) -> str | None:
    """Ensure every fact (anything with a ``currency`` attribute or key) shares one currency.

    Returns that currency (``None`` for an empty input). Raises ``ValueError`` naming every
    currency seen when they differ; a fact without a currency counts as ``"unknown"`` and is
    also refused next to a known one, because "probably the same currency" is exactly the
    silent mixing the spec forbids.
    """
    seen: dict[str, int] = {}
    for fact in facts:
        code = _currency_of(fact)
        label = "unknown" if code is None else str(code).upper()
        seen[label] = seen.get(label, 0) + 1
    if not seen:
        return None
    if len(seen) > 1:
        listing = ", ".join(f"{code} ({count})" for code, count in sorted(seen.items()))
        raise ValueError(f"mixed currencies: {listing}")
    only = next(iter(seen))
    return None if only == "unknown" else only
