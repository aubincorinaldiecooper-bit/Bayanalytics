"""Deterministic operand extraction from ``NormalizedEvidence``.

The registry decides *what* to compute; this module decides *which facts and prices* are the
operands, always by the same rules so the same evidence yields the same operands:

* facts are filtered to periods ending on or before ``as_of`` (and filings published on or
  before it) and sorted by (period end, basis, publication, fact id);
* for one metric and period, GAAP is preferred over adjusted over unknown basis, then the
  latest publication; any other basis reported for that period is mentioned in a note, the
  conflict itself lives in ``evidence.conflicts``;
* TTM is the sum of the latest four *consecutive* fiscal quarters; when they are not all
  present the latest fiscal year is used instead, labelled as a fiscal year (never a mix of
  three quarters and a year, never an annualised quarter);
* the latest close is the latest *completed* regular session at ``as_of``
  (:mod:`bayanalytics.normalization.sessions`), never a partial session;
* market cap pairs that close with the latest ``shares_outstanding`` fact and records both
  dates, noting when they differ;
* benchmark series are looked up by role in ``evidence.benchmarks``; a missing series is a
  missing operand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from itertools import pairwise
from typing import Any

from bayanalytics.normalization.periods import (
    is_consecutive_quarters,
    ttm_period,
    ttm_window,
)
from bayanalytics.normalization.periods import (
    label as period_label_of,
)
from bayanalytics.normalization.sessions import latest_completed_close
from bayanalytics.schemas.calculations import CalculationInput
from bayanalytics.schemas.evidence import (
    NormalizedEvidence,
    NormalizedFact,
    Period,
    PricePoint,
    PriceSeries,
)

_BASIS_RANK: dict[str, int] = {"gaap": 0, "adjusted": 1, "unknown": 2}
PRICE_METRIC = "price"


@dataclass(frozen=True)
class Operand:
    """One resolved operand with the provenance needed to reproduce the calculation."""

    metric: str
    value: float
    unit: str
    period_label: str
    period_kind: str
    period_end: date | None
    fact_id: str | None = None
    fact_ids: tuple[str, ...] = ()
    source_id: str | None = None
    published_at: datetime | None = None
    currency: str | None = None
    notes: tuple[str, ...] = ()

    def to_input(self, name: str) -> CalculationInput:
        return CalculationInput(
            name=name,
            value=self.value,
            unit=self.unit,
            source_id=self.source_id,
            period_label=self.period_label,
            fact_id=self.fact_id,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "period_label": self.period_label,
            "period_kind": self.period_kind,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "fact_id": self.fact_id,
            "fact_ids": list(self.fact_ids),
            "source_id": self.source_id,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "currency": self.currency,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class SeriesOperand:
    """A window of closes, oldest first, with the dates and provenance to reproduce it."""

    name: str
    closes: tuple[float, ...]
    dates: tuple[date, ...]
    source_id: str | None
    symbol: str | None
    requested_start: date | None = None
    notes: tuple[str, ...] = ()

    @property
    def start(self) -> date | None:
        return self.dates[0] if self.dates else None

    @property
    def end(self) -> date | None:
        return self.dates[-1] if self.dates else None

    @property
    def period_label(self) -> str:
        if not self.dates:
            return "empty"
        return f"{self.start.isoformat()} to {self.end.isoformat()}"  # type: ignore[union-attr]

    def to_input(self) -> CalculationInput:
        return CalculationInput(
            name=self.name,
            value=float(len(self.closes)),
            unit="observations",
            source_id=self.source_id,
            period_label=self.period_label,
            fact_id=None,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "source_id": self.source_id,
            "points": len(self.closes),
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "requested_start": self.requested_start.isoformat() if self.requested_start else None,
            "first_close": self.closes[0] if self.closes else None,
            "last_close": self.closes[-1] if self.closes else None,
            "notes": list(self.notes),
        }


@dataclass
class Aligned:
    """Operands for several metrics that share one period, or the reason they could not."""

    operands: dict[str, Operand] = field(default_factory=dict)
    basis: str | None = None  # "ttm", "fiscal_year", "fiscal_quarter"
    period_label: str | None = None
    missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and bool(self.operands)


def shift_months(d: date, months: int) -> date:
    """Move a date by whole months, clamping the day to the target month's length."""
    total = d.year * 12 + (d.month - 1) + months
    year, month = divmod(total, 12)
    month += 1
    day = d.day
    while True:
        try:
            return date(year, month, day)
        except ValueError:
            day -= 1


def _fact_sort_key(fact: NormalizedFact) -> tuple[Any, ...]:
    return (
        fact.period.end or date.min,
        _BASIS_RANK.get(fact.basis, 3),
        -(fact.published_at.timestamp() if fact.published_at else 0.0),
        fact.fact_id,
    )


class OperandResolver:
    """Pulls operands out of one ``NormalizedEvidence`` for one ``as_of`` moment."""

    def __init__(
        self,
        evidence: NormalizedEvidence,
        as_of: datetime,
        holidays: Sequence[date] | None = None,
    ) -> None:
        self.evidence = evidence
        self.as_of = as_of
        self.as_of_date = as_of.date()
        self.holidays = set(holidays) if holidays else None
        self._by_metric: dict[str, list[NormalizedFact]] = {}
        for fact in evidence.facts:
            if fact.period.end is not None and fact.period.end > self.as_of_date:
                continue
            if fact.published_at is not None and fact.published_at.date() > self.as_of_date:
                continue
            self._by_metric.setdefault(fact.metric, []).append(fact)
        for facts in self._by_metric.values():
            facts.sort(key=_fact_sort_key)

    # ---------------------------------------------------------------- facts ----------

    def facts(self, metric: str, kind: str | None = None) -> list[NormalizedFact]:
        facts = self._by_metric.get(metric, [])
        if kind is None:
            return list(facts)
        return [f for f in facts if f.period.kind == kind]

    def _choose(self, candidates: list[NormalizedFact]) -> NormalizedFact | None:
        """Pick one fact for a single period: GAAP first, then latest publication."""
        if not candidates:
            return None
        ordered = sorted(candidates, key=_fact_sort_key)
        return ordered[0]

    def _operand(self, fact: NormalizedFact, siblings: list[NormalizedFact]) -> Operand:
        notes: list[str] = []
        others = [s for s in siblings if s.fact_id != fact.fact_id and s.basis != fact.basis]
        for other in sorted(others, key=_fact_sort_key):
            notes.append(
                f"{other.basis} value {other.value!r} ({other.source_id}) also reported for "
                f"{fact.period.label}; {fact.basis} used"
            )
        if fact.restated:
            notes.append(
                f"restated from {fact.original_value!r} ({fact.original_source_id}) "
                f"to {fact.value!r}"
            )
        if fact.extraction_method == "derived":
            notes.append("derived fact: " + "; ".join(fact.notes))
        return Operand(
            metric=fact.metric,
            value=fact.value,
            unit=fact.unit,
            period_label=fact.period.label or period_label_of(fact.period),
            period_kind=fact.period.kind,
            period_end=fact.period.end,
            fact_id=fact.fact_id,
            fact_ids=(fact.fact_id,),
            source_id=fact.source_id,
            published_at=fact.published_at,
            currency=fact.currency,
            notes=tuple(notes),
        )

    def _latest_of_kind(self, metric: str, kind: str) -> Operand | None:
        facts = self.facts(metric, kind)
        if not facts:
            return None
        latest_end = max(f.period.end or date.min for f in facts)
        same_period = [f for f in facts if (f.period.end or date.min) == latest_end]
        chosen = self._choose(same_period)
        return self._operand(chosen, same_period) if chosen else None

    def for_period(self, metric: str, period: Period) -> Operand | None:
        facts = [f for f in self.facts(metric) if f.period.key() == period.key()]
        if not facts:
            # Same labelled period under a different key (e.g. dates differ by a day).
            facts = [
                f
                for f in self.facts(metric, period.kind)
                if period.fiscal_year is not None
                and f.period.fiscal_year == period.fiscal_year
                and f.period.fiscal_period == period.fiscal_period
            ]
        chosen = self._choose(facts)
        return self._operand(chosen, facts) if chosen else None

    def latest_fy(self, metric: str) -> Operand | None:
        return self._latest_of_kind(metric, "fiscal_year")

    def fy(self, metric: str, fiscal_year: int) -> Operand | None:
        facts = [
            f for f in self.facts(metric, "fiscal_year") if f.period.fiscal_year == fiscal_year
        ]
        chosen = self._choose(facts)
        return self._operand(chosen, facts) if chosen else None

    def fy_years_before(self, metric: str, latest: Operand, years: int) -> Operand | None:
        """The fiscal-year fact ``years`` before ``latest`` (by fiscal_year, else by end date)."""
        facts = self.facts(metric, "fiscal_year")
        fy = self.period_of(latest)
        if fy is not None and fy.fiscal_year is not None:
            target = [f for f in facts if f.period.fiscal_year == fy.fiscal_year - years]
        else:
            target = []
        if not target and latest.period_end is not None:
            lower = latest.period_end - timedelta(days=366 * years + 20)
            upper = latest.period_end - timedelta(days=365 * years - 20)
            target = [f for f in facts if f.period.end and lower <= f.period.end <= upper]
        chosen = self._choose(target)
        return self._operand(chosen, target) if chosen else None

    def latest_fq(self, metric: str) -> Operand | None:
        return self._latest_of_kind(metric, "fiscal_quarter")

    def period_of(self, operand: Operand) -> Period | None:
        for fact in self.facts(operand.metric):
            if fact.fact_id == operand.fact_id:
                return fact.period
        return None

    def prior_year_quarter(self, metric: str, quarter: Operand) -> Operand | None:
        """The same fiscal quarter one year earlier (fy - 1, same fp; else end ~365 days back)."""
        period = self.period_of(quarter)
        facts = self._same_basis(self.facts(metric, "fiscal_quarter"), quarter)
        target: list[NormalizedFact] = []
        if period is not None and period.fiscal_year is not None and period.fiscal_period:
            target = [
                f
                for f in facts
                if f.period.fiscal_year == period.fiscal_year - 1
                and f.period.fiscal_period == period.fiscal_period
            ]
        if not target and quarter.period_end is not None:
            lower = quarter.period_end - timedelta(days=380)
            upper = quarter.period_end - timedelta(days=350)
            target = [f for f in facts if f.period.end and lower <= f.period.end <= upper]
        chosen = self._choose(target)
        return self._operand(chosen, target) if chosen else None

    def previous_quarter(self, metric: str, quarter: Operand) -> Operand | None:
        """The fiscal quarter immediately before ``quarter`` (consecutive by end date)."""
        period = self.period_of(quarter)
        if period is None:
            return None
        facts = [
            f
            for f in self._same_basis(self.facts(metric, "fiscal_quarter"), quarter)
            if f.period.end
            and period.end
            and f.period.end < period.end
            and is_consecutive_quarters(f.period, period)
        ]
        chosen = self._choose(facts)
        return self._operand(chosen, facts) if chosen else None

    @staticmethod
    def _same_basis(facts: list[NormalizedFact], reference: Operand) -> list[NormalizedFact]:
        """Only facts reported in the reference operand's unit and currency: a USD quarter is
        never compared with a EUR one, nor a per-share figure with a total."""
        return [f for f in facts if f.unit == reference.unit and f.currency == reference.currency]

    def ttm(self, metric: str) -> Operand | None:
        """Sum of the latest four consecutive fiscal quarters, or ``None``."""
        facts = self.facts(metric, "fiscal_quarter")
        window = ttm_window(f.period for f in facts)
        if window is None:
            return None
        return self._ttm_from_window(metric, facts, window)

    def ttm_ending_near(
        self, metric: str, target_end: date, tolerance_days: int = 20
    ) -> Operand | None:
        """The TTM whose last fiscal quarter ends within ``tolerance_days`` of ``target_end``
        (the quarter closest to it wins), made of that quarter and the three consecutive
        quarters before it; ``None`` when no such window exists. This is how "the same
        trailing period one or three years earlier" is located for period-over-period
        comparisons: by end date, never by assuming a calendar."""
        facts = self.facts(metric, "fiscal_quarter")
        unique: dict[str, Period] = {}
        for fact in facts:
            if fact.period.end is not None:
                unique.setdefault(fact.period.key(), fact.period)
        ordered = sorted(unique.values(), key=lambda p: p.end)  # type: ignore[arg-type, return-value]
        candidates = [
            p
            for p in ordered
            if abs((p.end - target_end).days) <= tolerance_days  # type: ignore[operator]
        ]
        if not candidates:
            return None
        last = min(candidates, key=lambda p: (abs((p.end - target_end).days), p.end))  # type: ignore[operator]
        index = ordered.index(last)
        if index < 3:
            return None
        window = ordered[index - 3 : index + 1]
        if not all(is_consecutive_quarters(a, b) for a, b in pairwise(window)):
            return None
        return self._ttm_from_window(metric, facts, window)

    def _ttm_from_window(
        self, metric: str, facts: list[NormalizedFact], window: Sequence[Period]
    ) -> Operand | None:
        chosen: list[NormalizedFact] = []
        notes: list[str] = []
        for period in window:
            same = [f for f in facts if f.period.key() == period.key()]
            fact = self._choose(same)
            if fact is None:
                return None
            chosen.append(fact)
            op = self._operand(fact, same)
            notes.extend(op.notes)
        units = {f.unit for f in chosen}
        currencies = {f.currency for f in chosen}
        if len(units) != 1 or len(currencies) != 1:
            return None
        period = ttm_period(window)
        total = float(sum(f.value for f in chosen))
        return Operand(
            metric=metric,
            value=total,
            unit=chosen[-1].unit,
            period_label=period.label,
            period_kind="ttm",
            period_end=period.end,
            fact_id=None,
            fact_ids=tuple(f.fact_id for f in chosen),
            source_id=chosen[-1].source_id,
            published_at=max((f.published_at for f in chosen if f.published_at), default=None),
            currency=chosen[-1].currency,
            notes=(
                "TTM = " + " + ".join(f"{f.period.label} {f.value!r}" for f in chosen),
                *notes,
            ),
        )

    def ttm_or_fy(self, metric: str) -> Operand | None:
        """TTM when four consecutive quarters exist, otherwise the latest fiscal year,
        labelled as such (never mixed)."""
        operand = self.ttm(metric)
        if operand is not None:
            return operand
        fy = self.latest_fy(metric)
        if fy is None:
            return None
        return Operand(
            **{
                **fy.__dict__,
                "notes": (
                    *fy.notes,
                    "four consecutive fiscal quarters unavailable; latest fiscal year used",
                ),
            }
        )

    def latest_balance(self, metric: str) -> Operand | None:
        """Latest balance-sheet style value: the instant fact with the latest date, else the
        fact of any kind with the latest period end."""
        instant = self._latest_of_kind(metric, "instant")
        if instant is not None:
            return instant
        facts = self.facts(metric)
        if not facts:
            return None
        latest_end = max(f.period.end or date.min for f in facts)
        same = [f for f in facts if (f.period.end or date.min) == latest_end]
        chosen = self._choose(same)
        return self._operand(chosen, same) if chosen else None

    def aligned(
        self, metrics: Sequence[str], kinds: Sequence[str] = ("ttm", "fiscal_year")
    ) -> Aligned:
        """Operands for all ``metrics`` sharing one period, trying ``kinds`` in order.

        ``"ttm"`` requires every metric to have a TTM ending on the same date; ``"fiscal_year"``
        and ``"fiscal_quarter"`` pick the latest such period for which every metric has a fact.
        """
        result = Aligned()
        absent = [m for m in metrics if not self.facts(m)]
        if absent:
            result.missing = absent
            return result
        for kind in kinds:
            if kind == "ttm":
                ops = {m: self.ttm(m) for m in metrics}
                if all(ops.values()) and len({op.period_end for op in ops.values()}) == 1:  # type: ignore[union-attr]
                    result.operands = ops  # type: ignore[assignment]
                    result.basis = "ttm"
                    result.period_label = next(iter(ops.values())).period_label  # type: ignore[union-attr]
                    return result
                if all(ops.values()):
                    result.notes.append("TTM windows end on different dates across metrics")
                else:
                    result.notes.append(
                        "four consecutive fiscal quarters unavailable for: "
                        + ", ".join(m for m, op in ops.items() if op is None)
                    )
                continue
            candidates = self._common_periods(metrics, kind)
            if candidates:
                period = candidates[-1]
                ops = {m: self.for_period(m, period) for m in metrics}
                if all(ops.values()):
                    result.operands = ops  # type: ignore[assignment]
                    result.basis = kind
                    result.period_label = period.label
                    if kind != kinds[0]:
                        result.notes.append(
                            f"{kinds[0]} unavailable; {period.label} used for all operands"
                        )
                    return result
            result.notes.append(f"no {kind} period has all of: {', '.join(metrics)}")
        result.missing = list(metrics)
        return result

    def _common_periods(self, metrics: Sequence[str], kind: str) -> list[Period]:
        """Periods of ``kind`` (by end date) present for every metric, oldest first."""
        sets: list[dict[date, Period]] = []
        for metric in metrics:
            by_end = {f.period.end: f.period for f in self.facts(metric, kind) if f.period.end}
            sets.append(by_end)
        if not sets:
            return []
        common = set(sets[0])
        for other in sets[1:]:
            common &= set(other)
        return [sets[0][end] for end in sorted(common)]

    # --------------------------------------------------------------- prices ----------

    def price_series(self) -> PriceSeries | None:
        return self.evidence.prices

    def benchmark_series(self, role: str) -> tuple[PriceSeries | None, dict[str, Any]]:
        """The benchmark series for a role plus a record of which benchmark was used."""
        series = self.evidence.benchmarks.get(role)
        record: dict[str, Any] = {
            "role": role,
            "symbol": series.symbol if series is not None else None,
            "name": (series.label or None) if series is not None else None,
        }
        return series, record

    def latest_close_point(self, series: PriceSeries | None = None) -> PricePoint | None:
        series = self.evidence.prices if series is None else series
        if series is None or not series.points:
            return None
        return latest_completed_close(series, self.as_of, self.holidays)

    def latest_close(self, series: PriceSeries | None = None) -> Operand | None:
        series = self.evidence.prices if series is None else series
        point = self.latest_close_point(series)
        if point is None or series is None:
            return None
        notes = [f"latest completed close as of {self.as_of.isoformat()}"]
        if (
            series.price_type in ("intraday", "pre_market", "after_hours")
            and series.session_date == point.date
        ):
            notes.append(
                f"series latest point is labelled {series.price_type}; the completed close was used"
            )
        if not series.split_adjusted:
            notes.append("price series is not split-adjusted")
        return Operand(
            metric=PRICE_METRIC,
            value=point.close,
            unit=f"{series.currency}_per_share",
            period_label=f"close {point.date.isoformat()}",
            period_kind="latest_close",
            period_end=point.date,
            fact_id=None,
            source_id=series.source_id,
            published_at=None,
            currency=series.currency,
            notes=tuple(notes),
        )

    def market_cap_operands(self) -> tuple[Operand | None, Operand | None, list[str]]:
        """Latest close and latest shares_outstanding with a note when their dates differ."""
        price = self.latest_close()
        shares = self.latest_balance("shares_outstanding")
        notes: list[str] = []
        if price is not None and shares is not None and shares.period_end and price.period_end:
            gap = (price.period_end - shares.period_end).days
            if gap != 0:
                notes.append(
                    f"market cap pairs the {price.period_end.isoformat()} close with shares "
                    f"outstanding as of {shares.period_end.isoformat()} ({abs(gap)} days "
                    f"{'earlier' if gap > 0 else 'later'}); share count may have changed since"
                )
        return price, shares, notes

    def closes_on_or_before(self, series: PriceSeries, d: date) -> list[PricePoint]:
        return sorted((p for p in series.points if p.date <= d), key=lambda p: p.date)

    def close_on_or_before(self, series: PriceSeries, d: date) -> PricePoint | None:
        points = self.closes_on_or_before(series, d)
        return points[-1] if points else None

    def window(
        self,
        name: str,
        series: PriceSeries | None,
        *,
        months: int | None = None,
        trading_days: int | None = None,
        ytd: bool = False,
        end_date: date | None = None,
    ) -> SeriesOperand | None:
        """Closes from a start point to the latest completed close.

        ``months`` starts at the last close on or before ``end - months`` (calendar); ``ytd``
        starts at the last close of the previous calendar year; ``trading_days`` takes the last
        N closes. ``None`` when the series does not reach back far enough: a shorter window
        would be a different calculation, not an approximation of this one.
        """
        if series is None or not series.points:
            return None
        end_point = self.latest_close_point(series)
        if end_point is None:
            return None
        end_day = end_date or end_point.date
        if end_date is not None:
            candidate = self.close_on_or_before(series, end_date)
            if candidate is None:
                return None
            end_point = candidate
            end_day = end_point.date
        points = self.closes_on_or_before(series, end_day)
        notes: list[str] = []
        requested_start: date | None = None
        if trading_days is not None:
            if len(points) < trading_days:
                return None
            chosen = points[-trading_days:]
        else:
            if ytd:
                requested_start = date(end_day.year - 1, 12, 31)
            elif months is not None:
                requested_start = shift_months(end_day, -months)
            else:
                requested_start = points[0].date
            start_point = self.close_on_or_before(series, requested_start)
            if start_point is None:
                return None
            if start_point.date != requested_start:
                notes.append(
                    f"window start {requested_start.isoformat()} was not a session; "
                    f"previous close {start_point.date.isoformat()} used"
                )
            chosen = [p for p in points if start_point.date <= p.date <= end_day]
        return SeriesOperand(
            name=name,
            closes=tuple(p.close for p in chosen),
            dates=tuple(p.date for p in chosen),
            source_id=series.source_id,
            symbol=series.symbol,
            requested_start=requested_start,
            notes=tuple(notes),
        )

    @staticmethod
    def align(a: SeriesOperand, b: SeriesOperand) -> tuple[SeriesOperand, SeriesOperand]:
        """Restrict two windows to their common dates (for beta / correlation)."""
        common = sorted(set(a.dates) & set(b.dates))
        index_a = dict(zip(a.dates, a.closes, strict=True))
        index_b = dict(zip(b.dates, b.closes, strict=True))
        dropped = (len(a.dates) - len(common), len(b.dates) - len(common))
        note = f"aligned on {len(common)} common sessions (dropped {dropped[0]} / {dropped[1]})"
        return (
            SeriesOperand(
                name=a.name,
                closes=tuple(index_a[d] for d in common),
                dates=tuple(common),
                source_id=a.source_id,
                symbol=a.symbol,
                requested_start=a.requested_start,
                notes=(*a.notes, note),
            ),
            SeriesOperand(
                name=b.name,
                closes=tuple(index_b[d] for d in common),
                dates=tuple(common),
                source_id=b.source_id,
                symbol=b.symbol,
                requested_start=b.requested_start,
                notes=(*b.notes, note),
            ),
        )

    def pe_history(self, years: int | None = 5) -> tuple[list[dict[str, Any]], list[str]]:
        """Trailing P/E at each historical fiscal-quarter end within ``years`` of as_of, or at
        every quarter end the retrieved EPS and price history cover when ``years`` is ``None``.

        Each point uses the close on or before the quarter end (at most 10 days earlier) and
        the TTM diluted EPS made of that quarter and the three consecutive quarters before it.
        Quarters with non-positive TTM EPS are skipped (no meaningful multiple). The window
        is therefore exactly what the sources provide: no point is interpolated or extended.
        """
        series = self.evidence.prices
        notes: list[str] = []
        if series is None or not series.points:
            return [], ["no price series for P/E history"]
        facts = self.facts("eps_diluted", "fiscal_quarter")
        unique: dict[str, NormalizedFact] = {}
        for fact in facts:
            key = fact.period.key()
            if key not in unique or _fact_sort_key(fact) < _fact_sort_key(unique[key]):
                unique[key] = fact
        ordered = sorted(unique.values(), key=lambda f: f.period.end or date.min)
        cutoff = shift_months(self.as_of_date, -12 * years) if years is not None else None
        history: list[dict[str, Any]] = []
        for index in range(3, len(ordered)):
            window = ordered[index - 3 : index + 1]
            if not all(
                is_consecutive_quarters(earlier.period, later.period)
                for earlier, later in pairwise(window)
            ):
                continue
            quarter_end = window[-1].period.end
            if quarter_end is None or quarter_end > self.as_of_date:
                continue
            if cutoff is not None and quarter_end < cutoff:
                continue
            eps_ttm = float(sum(f.value for f in window))
            point = self.close_on_or_before(series, quarter_end)
            if point is None or (quarter_end - point.date).days > 10:
                notes.append(f"no close within 10 days before {quarter_end.isoformat()}")
                continue
            if eps_ttm <= 0:
                notes.append(f"TTM EPS not positive at {quarter_end.isoformat()}; skipped")
                continue
            history.append(
                {
                    "quarter_end": quarter_end.isoformat(),
                    "period_label": window[-1].period.label,
                    "close_date": point.date.isoformat(),
                    "close": point.close,
                    "eps_ttm": eps_ttm,
                    "pe": point.close / eps_ttm,
                    "fact_ids": [f.fact_id for f in window],
                }
            )
        return history, notes
