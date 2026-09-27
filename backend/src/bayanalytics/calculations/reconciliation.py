"""Valuation-vs-fundamentals reconciliation (AGENT.md sections 3.3, 6, 12, 25).

Decomposes a period's price change into the part the company earned (EPS growth) and the
part the market paid for (the change in the trailing multiple). With ``R`` the price return
over the window, ``g`` the growth of trailing diluted EPS over the same window and ``m`` the
implied change in the trailing P/E::

    (1 + R) = (1 + g) x (1 + m)        so        m = (1 + R) / (1 + g) - 1

Additively ``R = g + m + g*m``, so the contributions reported are ``g`` (earnings), ``m``
(multiple) and ``g*m`` (their interaction), each in percentage points of the start price.
The product owner's example, price +35 %, EPS +10 %: ``m = 1.35 / 1.10 - 1 = 22.7 %`` of the
move is multiple expansion, +10 points is earnings growth and +2.3 points the interaction.

Every number is a ``CalculationResult`` record of the registry, per window of ``N`` years:
``reconciliation_price_return_Ny`` (R), ``reconciliation_eps_growth_Ny`` (g),
``reconciliation_revenue_growth_Ny`` (context) and the headline
``valuation_reconciliation_Ny`` (m), whose ``meta["reconciliation"]`` carries the
contributions, the verdict, the rule that produced it and the thresholds. Each record has its
formula, recorded operands with provenance, period labels and a computed / unavailable status
with named missing operands; nothing is inferred.

**Verdict rules** (:data:`VERDICT_RULES`, first match wins; ``g``, ``m``, ``R`` in percentage
points; thresholds :data:`NEGLIGIBLE_POINTS` = 0.5 and :data:`EQUAL_POINTS` = 1.0)::

    |g| < 0.5 and |m| < 0.5                              no material change
    |g| < 0.5, m >= +0.5                                 multiple expansion with flat earnings
    |g| < 0.5, m <= -0.5                                 multiple contraction with flat earnings
    g >= +0.5, |m| < 0.5                                 earnings growth at a steady multiple
    g <= -0.5, |m| < 0.5                                 earnings decline at a steady multiple
    g >= +0.5, m >= +0.5, |m| - |g| > 1.0                mostly multiple expansion
    g >= +0.5, m >= +0.5, |g| - |m| > 1.0                mostly earnings growth
    g >= +0.5, m >= +0.5, ||m| - |g|| <= 1.0             earnings and multiple contributed equally
    g <= -0.5, m <= -0.5, |m| - |g| > 1.0                mostly multiple contraction
    g <= -0.5, m <= -0.5, |g| - |m| > 1.0                mostly earnings decline
    g <= -0.5, m <= -0.5, ||m| - |g|| <= 1.0             earnings and multiple contracted equally
    g >= +0.5, m <= -0.5, R <= -0.5                      de-rating despite growth
    g >= +0.5, m <= -0.5, R > -0.5                       earnings growth offset by de-rating
    g <= -0.5, m >= +0.5, R >= +0.5                      re-rating despite earnings decline
    g <= -0.5, m >= +0.5, R < +0.5                       earnings decline offset by re-rating

The verdict is derived from the record's own values only, so Spark restates it and never
derives one. A multiple needs positive earnings at both ends: when either EPS is zero or
negative the decomposition has no meaning and the records are unavailable, never
approximated. The operand selection (:func:`trailing_pair`) mirrors how the trailing P/E is
defined: the current TTM against the TTM that was current ``years`` earlier, or the
fiscal-year pair when four consecutive quarters are missing at either end (labelled as such,
never mixed).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from bayanalytics.calculations import primitives as prim
from bayanalytics.calculations.formatting import format_change, format_value
from bayanalytics.calculations.operands import Operand, OperandResolver, shift_months
from bayanalytics.schemas.calculations import CalculationResult

WINDOW_YEARS: tuple[int, ...] = (1, 3)
"""Reconciliation windows: one year always, three years when the history reaches back."""

NEGLIGIBLE_POINTS = 0.5
"""A component (EPS growth, multiple change) or a price return smaller than this many
percentage points in magnitude counts as flat for the verdict."""

EQUAL_POINTS = 1.0
"""Two same-sign components whose magnitudes differ by at most this many percentage points
contributed equally; beyond it the larger one is what the move "mostly" was."""

THRESHOLDS: dict[str, float] = {
    "negligible_points": NEGLIGIBLE_POINTS,
    "equal_points": EQUAL_POINTS,
}

COMPONENT_NAMES: tuple[str, ...] = ("price_return", "eps_growth", "revenue_growth")
"""Per window, ``reconciliation_<component>_<N>y`` records sit next to the headline."""


def headline_name(years: int) -> str:
    return f"valuation_reconciliation_{years}y"


def component_name(component: str, years: int) -> str:
    return f"reconciliation_{component}_{years}y"


_Rule = tuple[str, str, Callable[[float, float, float], bool]]


def _flat(x: float) -> bool:
    return abs(x) < NEGLIGIBLE_POINTS


def _up(x: float) -> bool:
    return x >= NEGLIGIBLE_POINTS


def _down(x: float) -> bool:
    return x <= -NEGLIGIBLE_POINTS


VERDICT_RULES: tuple[_Rule, ...] = (
    ("no material change", "|g| < 0.5 and |m| < 0.5", lambda r, g, m: _flat(g) and _flat(m)),
    (
        "multiple expansion with flat earnings",
        "|g| < 0.5, m >= +0.5",
        lambda r, g, m: _flat(g) and _up(m),
    ),
    (
        "multiple contraction with flat earnings",
        "|g| < 0.5, m <= -0.5",
        lambda r, g, m: _flat(g) and _down(m),
    ),
    (
        "earnings growth at a steady multiple",
        "g >= +0.5, |m| < 0.5",
        lambda r, g, m: _up(g) and _flat(m),
    ),
    (
        "earnings decline at a steady multiple",
        "g <= -0.5, |m| < 0.5",
        lambda r, g, m: _down(g) and _flat(m),
    ),
    (
        "mostly multiple expansion",
        "g >= +0.5, m >= +0.5, |m| - |g| > 1.0",
        lambda r, g, m: _up(g) and _up(m) and abs(m) - abs(g) > EQUAL_POINTS,
    ),
    (
        "mostly earnings growth",
        "g >= +0.5, m >= +0.5, |g| - |m| > 1.0",
        lambda r, g, m: _up(g) and _up(m) and abs(g) - abs(m) > EQUAL_POINTS,
    ),
    (
        "earnings and multiple contributed equally",
        "g >= +0.5, m >= +0.5, ||m| - |g|| <= 1.0",
        lambda r, g, m: _up(g) and _up(m),
    ),
    (
        "mostly multiple contraction",
        "g <= -0.5, m <= -0.5, |m| - |g| > 1.0",
        lambda r, g, m: _down(g) and _down(m) and abs(m) - abs(g) > EQUAL_POINTS,
    ),
    (
        "mostly earnings decline",
        "g <= -0.5, m <= -0.5, |g| - |m| > 1.0",
        lambda r, g, m: _down(g) and _down(m) and abs(g) - abs(m) > EQUAL_POINTS,
    ),
    (
        "earnings and multiple contracted equally",
        "g <= -0.5, m <= -0.5, ||m| - |g|| <= 1.0",
        lambda r, g, m: _down(g) and _down(m),
    ),
    (
        "de-rating despite growth",
        "g >= +0.5, m <= -0.5, R <= -0.5",
        lambda r, g, m: _up(g) and _down(m) and _down(r),
    ),
    (
        "earnings growth offset by de-rating",
        "g >= +0.5, m <= -0.5, R > -0.5",
        lambda r, g, m: _up(g) and _down(m),
    ),
    (
        "re-rating despite earnings decline",
        "g <= -0.5, m >= +0.5, R >= +0.5",
        lambda r, g, m: _down(g) and _up(m) and _up(r),
    ),
    (
        "earnings decline offset by re-rating",
        "g <= -0.5, m >= +0.5, R < +0.5",
        lambda r, g, m: _down(g) and _up(m),
    ),
)
"""(label, documented condition, predicate over percentage points), first match wins. The
fifteen conditions are exhaustive over every (R, g, m) the identity can produce."""

VERDICTS: tuple[str, ...] = tuple(label for label, _rule, _test in VERDICT_RULES)
"""Every verdict label the decomposition can produce (fixed vocabulary for clients)."""


@dataclass(frozen=True)
class Verdict:
    label: str
    rule: str  # the documented condition that matched, in percentage points


def verdict(price_return: float, eps_growth: float, multiple_change: float) -> Verdict:
    """The verdict for fractions ``R``, ``g``, ``m`` (converted to percentage points and
    matched against :data:`VERDICT_RULES` in order)."""
    r, g, m = price_return * 100.0, eps_growth * 100.0, multiple_change * 100.0
    for label, rule, test in VERDICT_RULES:
        if test(r, g, m):
            return Verdict(label, rule)
    raise AssertionError(f"verdict rules are not exhaustive for R={r!r}, g={g!r}, m={m!r}")


@dataclass(frozen=True)
class Decomposition:
    """One reconciled window. Fractions throughout (``0.35`` is +35 %)."""

    price_return: float
    eps_growth: float
    multiple_change: float
    earnings_contribution: float  # = eps_growth, in fraction of the start price
    multiple_contribution: float  # = multiple_change
    interaction: float  # = eps_growth * multiple_change
    earnings_share: float | None  # contribution / price_return; None when the price did not move
    multiple_share: float | None
    interaction_share: float | None
    verdict: Verdict

    def as_dict(self) -> dict[str, Any]:
        """Percent units for the contract (``22.7`` means 22.7 %), shares as fractions."""
        return {
            "price_return_pct": self.price_return * 100.0,
            "eps_growth_pct": self.eps_growth * 100.0,
            "multiple_change_pct": self.multiple_change * 100.0,
            "contributions_pct": {
                "earnings": self.earnings_contribution * 100.0,
                "multiple": self.multiple_contribution * 100.0,
                "interaction": self.interaction * 100.0,
            },
            "shares_of_move": {
                "earnings": self.earnings_share,
                "multiple": self.multiple_share,
                "interaction": self.interaction_share,
            },
            "identity": "(1 + price_return) = (1 + eps_growth) x (1 + multiple_change)",
            "contributions_formula": "price_return = eps_growth + multiple_change "
            "+ eps_growth x multiple_change",
            "verdict": self.verdict.label,
            "verdict_rule": self.verdict.rule,
            "thresholds": dict(THRESHOLDS),
        }


def eps_ratio_growth(eps_current: float | None, eps_previous: float | None) -> float | None:
    """``eps_current / eps_previous - 1`` with both strictly positive, else ``None``.

    Unlike :func:`primitives.growth_rate` (which keeps the sign meaningful through a loss by
    dividing by ``|previous|``) the multiplicative identity needs the plain ratio, and a ratio
    through zero or negative earnings is not a growth of a multiple's denominator.
    """
    if not prim.is_number(eps_current) or not prim.is_number(eps_previous):
        return None
    if eps_current <= 0 or eps_previous <= 0:  # type: ignore[operator]
        return None
    return float(eps_current) / float(eps_previous) - 1.0  # type: ignore[arg-type]


def implied_multiple_change(price_return: float | None, eps_growth: float | None) -> float | None:
    """``m = (1 + R) / (1 + g) - 1``; ``None`` when an input is missing or ``1 + g <= 0``."""
    if not prim.is_number(price_return) or not prim.is_number(eps_growth):
        return None
    base = 1.0 + float(eps_growth)  # type: ignore[arg-type]
    if base <= 0:
        return None
    return (1.0 + float(price_return)) / base - 1.0  # type: ignore[arg-type]


def decompose(price_return: float | None, eps_growth: float | None) -> Decomposition | None:
    """Full decomposition of ``price_return`` given ``eps_growth`` (both fractions)."""
    m = implied_multiple_change(price_return, eps_growth)
    if m is None:
        return None
    r = float(price_return)  # type: ignore[arg-type]
    g = float(eps_growth)  # type: ignore[arg-type]
    interaction = g * m

    def share(part: float) -> float | None:
        return None if r == 0 else part / r

    return Decomposition(
        price_return=r,
        eps_growth=g,
        multiple_change=m,
        earnings_contribution=g,
        multiple_contribution=m,
        interaction=interaction,
        earnings_share=share(g),
        multiple_share=share(m),
        interaction_share=share(interaction),
        verdict=verdict(r, g, m),
    )


# ------------------------------------------------------------------ registry glue ----------


def multiple_change_from_operands(
    start_close: float,
    end_close: float,
    eps_current: float,
    eps_previous: float,
    revenue_current: float | None = None,
    revenue_previous: float | None = None,
    pe_percentile: float | None = None,
) -> float | None:
    """The headline value: the implied multiple change as a fraction (``None`` when the
    price return or the EPS ratio is not meaningful). The optional operands are context only
    and never influence the value."""
    price_return = prim.period_return([start_close, end_close])
    eps_growth = eps_ratio_growth(eps_current, eps_previous)
    return implied_multiple_change(price_return, eps_growth)


def explain_unavailable(values: dict[str, Any]) -> str | None:
    """Why a reconciliation record has no value for present operands."""
    for name in ("eps_current", "eps_previous"):
        value = values.get(name)
        if isinstance(value, int | float) and value <= 0:
            return f"trailing P/E not meaningful: {name} = {value!r} (non-positive earnings)"
    start = values.get("start_close")
    if isinstance(start, int | float) and start <= 0:
        return f"price return undefined from a non-positive start close ({start!r})"
    if values.get("revenue_previous") == 0:
        return "growth from a zero revenue base is undefined"
    return None


def detail_for(years: int) -> Callable[[dict[str, Any]], tuple[dict[str, Any], list[str]]]:
    """The headline's ``detail`` hook: the decomposition under ``meta["reconciliation"]``
    (with the names of the component records of the same window) and the deterministic
    verdict sentence as a note."""

    def detail(values: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        price_return = prim.period_return([values["start_close"], values["end_close"]])
        eps_growth = eps_ratio_growth(values["eps_current"], values["eps_previous"])
        parts = decompose(price_return, eps_growth)
        if parts is None:  # pragma: no cover - compute() calls detail only after a value
            return {}, []
        revenue_growth = prim.growth_rate(
            values.get("revenue_current"), values.get("revenue_previous")
        )
        percentile = values.get("pe_percentile")
        record = parts.as_dict()
        record["revenue_growth_pct"] = None if revenue_growth is None else revenue_growth * 100.0
        record["pe_percentile"] = float(percentile) if prim.is_number(percentile) else None
        record["component_calcs"] = {c: component_name(c, years) for c in COMPONENT_NAMES}
        note = sentence(parts, revenue_growth, record["pe_percentile"])
        return {"reconciliation": record}, [note]

    return detail


def sentence(
    parts: Decomposition, revenue_growth: float | None, pe_percentile: float | None
) -> str:
    """One deterministic sentence an analyst can quote, e.g. ``mostly multiple expansion: of
    the +35.0% price change, +10.0 points came from EPS growth, +22.7 points from the change
    in the multiple and +2.3 points from their interaction; revenue grew 8.0%; trailing P/E
    sits at its 92nd percentile of its available history``."""
    pct = format_change(parts.price_return * 100.0, "percent")
    text = (
        f"{parts.verdict.label}: of the {pct} price change, "
        f"{_points(parts.earnings_contribution)} came from EPS growth, "
        f"{_points(parts.multiple_contribution)} from the change in the multiple and "
        f"{_points(parts.interaction)} from their interaction"
    )
    if revenue_growth is not None:
        direction = "grew" if revenue_growth >= 0 else "declined"
        text += f"; revenue {direction} {format_value(abs(revenue_growth) * 100.0, 'percent')}"
    else:
        text += "; revenue growth over the window is unavailable"
    if pe_percentile is not None:
        text += (
            f"; trailing P/E sits at its {format_value(pe_percentile, 'percentile')} "
            "of its available history"
        )
    else:
        text += "; the trailing P/E's historical percentile is unavailable"
    return text


def _points(fraction: float) -> str:
    value = fraction * 100.0
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f} points"


def bundle_view(calculations: list[CalculationResult]) -> dict[str, Any]:
    """The computed reconciliations for the Spark bundle: values, verdict and rule exactly as
    recorded, so Spark restates the verdict instead of deriving one."""
    view: dict[str, Any] = {}
    for calc in calculations:
        record = calc.meta.get("reconciliation") if calc.status == "computed" else None
        if not calc.name.startswith("valuation_reconciliation_") or not isinstance(record, dict):
            continue
        view[calc.name] = {
            "calc_id": calc.calc_id,
            "period": calc.period_label,
            "verdict": record.get("verdict"),
            "verdict_rule": record.get("verdict_rule"),
            "multiple_change": calc.display,
            "price_return_pct": record.get("price_return_pct"),
            "eps_growth_pct": record.get("eps_growth_pct"),
            "revenue_growth_pct": record.get("revenue_growth_pct"),
            "contributions_pct": record.get("contributions_pct"),
            "pe_percentile": record.get("pe_percentile"),
            "statement": calc.notes[-1] if calc.notes else None,
        }
    return view


# ------------------------------------------------------------------ operand selection ------


@dataclass(frozen=True)
class TrailingPair:
    """A metric measured over two matching trailing periods ``years`` apart."""

    current: Operand | None
    previous: Operand | None
    basis: str | None  # "ttm" or "fiscal_year"
    notes: tuple[str, ...]

    @property
    def period_label(self) -> str | None:
        if self.current is None or self.previous is None:
            return None
        return f"{self.current.period_label} vs {self.previous.period_label}"


def trailing_pair(resolver: OperandResolver, metric: str, years: int) -> TrailingPair:
    """``metric`` over the latest TTM and the TTM that was current ``years`` earlier (the
    window whose last quarter ends ``12 * years`` months before the latest TTM's end, within
    20 days); when either end lacks four consecutive quarters, the latest fiscal year and the
    fiscal year ``years`` before it, labelled as a fiscal-year pair. Never one of each."""
    current = resolver.ttm(metric)
    previous: Operand | None = None
    notes: list[str] = []
    if current is not None and current.period_end is not None:
        target = shift_months(current.period_end, -12 * years)
        previous = resolver.ttm_ending_near(metric, target)
        if previous is None:
            notes.append(
                f"no trailing-twelve-month {metric} window ends within 20 days of "
                f"{target.isoformat()} ({years}y before {current.period_label})"
            )
    if current is not None and previous is not None:
        return TrailingPair(current, previous, "ttm", tuple(notes))
    fy_current = resolver.latest_fy(metric)
    fy_previous = resolver.fy_years_before(metric, fy_current, years) if fy_current else None
    if fy_current is not None and fy_previous is not None:
        notes.append(
            f"trailing-twelve-month {metric} pair unavailable; fiscal-year pair used "
            f"({fy_current.period_label} vs {fy_previous.period_label})"
        )
        return TrailingPair(fy_current, fy_previous, "fiscal_year", tuple(notes))
    if current is None:
        notes.append(f"no trailing-twelve-month {metric} (four consecutive quarters missing)")
    if fy_current is None:
        notes.append(f"no fiscal-year {metric}")
    elif fy_previous is None:
        notes.append(f"no fiscal-year {metric} {years} year(s) before {fy_current.period_label}")
    # Report whichever end exists so the missing one is named precisely by the record.
    return TrailingPair(current or fy_current, previous, None, tuple(notes))
