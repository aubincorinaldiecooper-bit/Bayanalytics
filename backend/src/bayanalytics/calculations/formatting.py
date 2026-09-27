"""Display formatting for calculated values (numerical typography).

Values are stored at full precision everywhere else; this module is the only place that
rounds. The conventions:

* money (``USD`` or any 3-letter currency code): compact scale with one decimal for
  K/M/B/T (``$42.8B``, ``$950.0K``), two decimals below one thousand (``$12.34``), sign in
  front of the symbol (``-$1.2B``).
* ``USD_per_share`` (or ``<CCY>_per_share``): two decimals (``$6.13``).
* ``percent``: one decimal and a percent sign; the value is already in percent units
  (``18.2`` -> ``18.2%``, ``-7.4`` -> ``-7.4%``).
* ``percentile``: an ordinal rank (``72nd percentile``).
* ``ratio``: one decimal and the multiplication sign U+00D7 (``34.1x`` with that sign).
* ``coefficient``: two decimals, no sign (``1.15`` for a beta).
* ``shares``: compact scale plus the word (``15.3B shares``).
* ``days``: whole days (``42 days``).
* ``bp``: whole basis points with an explicit sign (``+120 bp``).

Tabular consistency: within one comparison never change scale. :func:`format_pair` picks a
single scale for two values so ``$42.8B`` is never shown next to ``$950.0M``.
"""

from __future__ import annotations

from collections.abc import Sequence

from bayanalytics.calculations.primitives import is_number

UNAVAILABLE = "unavailable"
MULTIPLY_SIGN = "\u00d7"

CURRENCY_SYMBOLS: dict[str, str] = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "CN¥",
    "CAD": "C$",
    "AUD": "A$",
    "HKD": "HK$",
    "CHF": "CHF ",
    "INR": "₹",
    "KRW": "₩",
}

# (threshold, suffix, divisor) from largest to smallest.
_SCALES: tuple[tuple[float, str, float], ...] = (
    (1e12, "T", 1e12),
    (1e9, "B", 1e9),
    (1e6, "M", 1e6),
    (1e3, "K", 1e3),
)
_SCALE_DIVISORS: dict[str, float] = {"T": 1e12, "B": 1e9, "M": 1e6, "K": 1e3, "": 1.0}


def pick_scale(value: float) -> str:
    """The compact scale suffix ("T", "B", "M", "K" or "") for ``abs(value)``."""
    magnitude = abs(value)
    for threshold, suffix, _divisor in _SCALES:
        if magnitude >= threshold:
            return suffix
    return ""


def _format_scaled(value: float, scale: str, decimals: int) -> str:
    """``abs(value)`` divided by the scale, rounded, with roll-over into the next scale when
    rounding produces ``1000.0`` (so ``999.96M`` becomes ``$1.0B`` rather than ``$1000.0M``)."""
    magnitude = abs(value)
    divisor = _SCALE_DIVISORS[scale]
    mantissa = magnitude / divisor
    text = f"{mantissa:.{decimals}f}"
    if scale and float(text) >= 1000.0:
        order = ["", "K", "M", "B", "T"]
        position = order.index(scale)
        if position + 1 < len(order):
            next_scale = order[position + 1]
            return _format_scaled(value, next_scale, decimals)
    return f"{text}{scale}"


def _money_symbol(unit: str, currency: str | None) -> str:
    code = currency or unit
    if code.endswith("_per_share"):
        code = code[: -len("_per_share")]
    return CURRENCY_SYMBOLS.get(code.upper(), f"{code.upper()} ")


def _is_money_unit(unit: str) -> bool:
    return unit.isalpha() and unit.isupper() and len(unit) == 3


def _is_per_share_unit(unit: str) -> bool:
    return unit.endswith("_per_share") and _is_money_unit(unit[: -len("_per_share")])


def format_money(
    value: float, unit: str = "USD", currency: str | None = None, decimals: int = 1, scale=None
) -> str:
    """Compact money: ``$42.8B``, ``$950.0K``, ``$12.34``, ``-$1.2B``.

    ``scale`` forces a suffix ("" for none) so a pair of values can share one; when omitted the
    natural scale of the value is used. Values below one thousand (or ``scale == ""``) show
    two decimals.
    """
    symbol = _money_symbol(unit, currency)
    sign = "-" if value < 0 else ""
    chosen = pick_scale(value) if scale is None else scale
    if chosen == "":
        return f"{sign}{symbol}{abs(value):.2f}"
    return f"{sign}{symbol}{_format_scaled(value, chosen, decimals)}"


def _ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def format_value(
    value: float | None,
    unit: str,
    *,
    currency: str | None = None,
    decimals: int | None = None,
    scale: str | None = None,
) -> str:
    """Render ``value`` in the typography for ``unit`` (see the module docstring).

    ``None`` (or a non-finite number) renders as ``"unavailable"``. ``decimals`` overrides the
    default precision for compact money/shares (``format_value(1_230_000, "USD", decimals=2)``
    gives ``$1.23M``); ``scale`` forces the compact suffix. Unknown units fall back to a
    general-purpose number followed by the unit name so nothing is ever hidden.
    """
    if not is_number(value):
        return UNAVAILABLE
    number = float(value)
    if unit == "percent":
        digits = 1 if decimals is None else decimals
        return f"{number:.{digits}f}%"
    if unit == "percentile":
        return f"{_ordinal(round(number))} percentile"
    if unit == "ratio":
        digits = 1 if decimals is None else decimals
        return f"{number:.{digits}f}{MULTIPLY_SIGN}"
    if unit == "coefficient":  # beta and other dimensionless coefficients
        digits = 2 if decimals is None else decimals
        return f"{number:.{digits}f}"
    if unit == "bp":
        whole = round(number)
        sign = "+" if whole > 0 else ""
        return f"{sign}{whole} bp"
    if unit == "days":
        return f"{round(number)} days"
    if unit == "shares":
        digits = 1 if decimals is None else decimals
        chosen = pick_scale(number) if scale is None else scale
        sign = "-" if number < 0 else ""
        if chosen == "":
            return f"{sign}{abs(number):,.0f} shares"
        return f"{sign}{_format_scaled(number, chosen, digits)} shares"
    if unit == "observations":
        return f"{round(number)} observations"
    if _is_per_share_unit(unit):
        symbol = _money_symbol(unit, currency)
        digits = 2 if decimals is None else decimals
        sign = "-" if number < 0 else ""
        return f"{sign}{symbol}{abs(number):.{digits}f}"
    if _is_money_unit(unit):
        digits = 1 if decimals is None else decimals
        return format_money(number, unit, currency, digits, scale)
    digits = 4 if decimals is None else decimals
    text = f"{number:,.{digits}f}".rstrip("0").rstrip(".")
    return f"{text} {unit}".strip()


def format_pair(
    a: float | None,
    b: float | None,
    unit: str,
    *,
    currency: str | None = None,
    decimals: int | None = None,
) -> tuple[str, str]:
    """Format two values of the same unit with one shared scale.

    For compact units (money, shares) the scale of the larger magnitude is used for both, so
    ``format_pair(42.8e9, 950e6, "USD")`` gives ``("$42.8B", "$0.9B")`` rather than mixing
    billions and millions in one comparison. Units without a scale format independently. A
    missing value renders as ``"unavailable"`` and does not influence the other's scale.
    """
    return tuple(format_series([a, b], unit, currency=currency, decimals=decimals))  # type: ignore[return-value]


def format_series(
    values: Sequence[float | None],
    unit: str,
    *,
    currency: str | None = None,
    decimals: int | None = None,
) -> list[str]:
    """Format any number of values of one unit with a single shared scale (see format_pair)."""
    compact = unit == "shares" or _is_money_unit(unit)
    shared: str | None = None
    if compact:
        present = [float(v) for v in values if is_number(v)]
        if present:
            shared = pick_scale(max(present, key=abs))
    return [
        format_value(v, unit, currency=currency, decimals=decimals, scale=shared) for v in values
    ]


def format_change(value: float | None, unit: str, *, currency: str | None = None) -> str:
    """Like :func:`format_value` but with an explicit leading sign for positive changes."""
    text = format_value(value, unit, currency=currency)
    if text == UNAVAILABLE or unit == "bp":
        return text
    if is_number(value) and float(value) > 0:
        return f"+{text}"
    return text
