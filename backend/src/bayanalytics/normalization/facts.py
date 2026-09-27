"""Build ``NormalizedFact`` records from extracted rows (AGENT.md sections 3.1, 6, 22, 32).

Input row contract (produced by the research module; agreed, do not change here)::

    {
      "concept": "us-gaap:Revenues",        # source concept name
      "metric": "revenue",                  # canonical metric name (calculations.registry)
      "value": 94_036_000_000.0,
      "unit": "USD" | "USD/shares" | "shares" | "pure" | "<CCY>" | "<CCY>/shares",
      "start": "2025-06-29" | None,         # None for an instant (balance-sheet) fact
      "end": "2025-09-27",
      "fy": 2025 | None, "fp": "FY" | "Q1".."Q4" | None,
      "form": "10-K" | "10-Q" | "8-K" | ...,
      "filed": "2025-10-31",                # publication date of the filing
      "accn": str | None, "frame": str | None,
      "source_id": "src_...",
      "basis": "gaap" | "adjusted" | "unknown",   # default gaap (XBRL is GAAP)
      "currency": "USD",                    # default USD
    }

What ``build_facts`` guarantees:

* leakage guard: a row filed after ``as_of`` (or whose period ends after ``as_of``) is dropped
  and counted, never used;
* units are normalized (``USD/shares`` -> ``USD_per_share``, ``pure`` -> ``ratio``) and the
  currency is carried on every fact;
* identical rows (same metric, period, basis and value) collapse into one fact;
* a later filing that changes a previously reported value is a restatement: the later
  value wins, the original value and source are preserved on the fact
  (``restated=True``) with a note, and a material restatement (> 5 %) is also surfaced as a
  ``Conflict`` with reason ``restatement``;
* two values for the same metric and period that are not a restatement (different bases,
  or the same filing date) become a ``Conflict``; nothing is dropped;
* derived facts (``free_cash_flow``, ``ebitda``) are added per period with
  ``extraction_method="derived"`` and the input fact ids in their notes.

Freshness (section 6) is classified from the period end for fundamentals and from the
price date for prices; the summary and stale-mix warnings make it impossible to pair a
current price with old fundamentals without saying so.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from bayanalytics.normalization.periods import classify_duration, period_from_xbrl
from bayanalytics.schemas.common import Freshness
from bayanalytics.schemas.evidence import (
    Conflict,
    ConflictValue,
    NormalizedFact,
    Period,
    PriceSeries,
)

RESTATEMENT_TOLERANCE = 0.005  # 0.5 % relative: below this two values are the same number
MATERIAL_THRESHOLD = 0.05  # 5 % relative: a conflict or restatement worth the analyst's eye

# Freshness thresholds in days: (current_max, recent_max). Beyond recent_max -> stale.
FRESHNESS_THRESHOLDS: dict[str, tuple[int, int]] = {
    "price": (3, 14),
    "quarterly": (100, 200),
    "annual": (380, 500),
}
_KIND_TO_BUCKET: dict[str, str] = {
    "price": "price",
    "prices": "price",
    "news": "price",
    "quarterly": "quarterly",
    "fiscal_quarter": "quarterly",
    "ttm": "quarterly",
    "ytd": "quarterly",
    "qtd": "quarterly",
    "instant": "quarterly",
    "annual": "annual",
    "fiscal_year": "annual",
}

DERIVED_METRICS: dict[str, tuple[str, str, str]] = {
    # derived metric -> (left operand, operator, right operand)
    "free_cash_flow": ("operating_cash_flow", "-|", "capex"),
    "ebitda": ("operating_income", "+", "depreciation_amortization"),
}

FactIdFactory = Callable[[dict[str, Any]], str]


@dataclass
class FactBuild:
    facts: list[NormalizedFact] = field(default_factory=list)
    conflicts: list[Conflict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)


def _iso_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _as_date(value: datetime | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    return value


def normalize_unit(unit: str | None) -> str:
    """``USD/shares`` -> ``USD_per_share``, ``pure`` -> ``ratio``, ``shares`` stays, a bare
    currency code stays. Unknown units are kept verbatim (never silently mapped)."""
    if unit is None:
        return "unknown"
    text = unit.strip()
    lowered = text.lower()
    if lowered == "pure":
        return "ratio"
    if lowered in ("percent", "%"):
        return "percent"
    if lowered == "shares":
        return "shares"
    if "/" in text:
        left, right = text.split("/", 1)
        if right.strip().lower() in ("shares", "share"):
            return f"{left.strip().upper()}_per_share"
        return text
    if len(text) == 3 and text.isalpha():
        return text.upper()
    return text


def default_fact_id(row: dict[str, Any]) -> str:
    """Deterministic fact id from the row's identity (metric, period, basis, source, value,
    filing date) so the same evidence always yields the same ids."""
    parts = [
        str(row.get("metric")),
        str(row.get("period_key")),
        str(row.get("basis")),
        str(row.get("source_id")),
        repr(row.get("value")),
        str(row.get("filed")),
    ]
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()  # noqa: S324
    return f"fact_{digest[:12]}"


def _relative_diff(a: float, b: float) -> float:
    base = max(abs(a), abs(b))
    if base == 0:
        return 0.0
    return abs(a - b) / base


@dataclass
class _Row:
    metric: str
    value: float
    unit: str
    currency: str
    period: Period
    basis: str
    source_id: str
    filed: date | None
    concept: str | None
    form: str | None
    accn: str | None
    frame: str | None
    raw: dict[str, Any]
    period_note: str | None

    def identity(self) -> tuple[str, str, str]:
        return self.metric, self.period.key(), self.basis


def _prepare_rows(
    rows: Iterable[dict[str, Any]], as_of: datetime, build: FactBuild
) -> list[_Row]:
    as_of_date = as_of.date() if isinstance(as_of, datetime) else as_of
    prepared: list[_Row] = []
    for index, row in enumerate(rows):
        metric = row.get("metric")
        value = row.get("value")
        if not metric or value is None or isinstance(value, bool):
            build.dropped.append({"row": row, "reason": "missing metric or value", "index": index})
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            build.dropped.append({"row": row, "reason": "value is not numeric", "index": index})
            continue
        try:
            start = _iso_date(row.get("start"))
            end = _iso_date(row.get("end"))
            filed = _iso_date(row.get("filed"))
        except ValueError:
            build.dropped.append({"row": row, "reason": "unparseable date", "index": index})
            continue
        if end is None:
            build.dropped.append({"row": row, "reason": "missing period end", "index": index})
            continue
        if filed is not None and filed > as_of_date:
            build.dropped.append(
                {"row": row, "reason": f"filed {filed.isoformat()} after as_of", "index": index}
            )
            continue
        if end > as_of_date:
            build.dropped.append(
                {"row": row, "reason": f"period end {end.isoformat()} after as_of", "index": index}
            )
            continue
        period = period_from_xbrl(start, end, row.get("fy"), row.get("fp"), row.get("form"))
        _kind, period_note = classify_duration(start, end, row.get("fp"))
        unit = normalize_unit(row.get("unit"))
        currency = str(row.get("currency") or "USD").upper()
        if len(unit) == 3 and unit.isalpha() and unit != currency:
            currency = unit
        basis = str(row.get("basis") or "gaap")
        if basis not in ("gaap", "adjusted", "unknown"):
            basis = "unknown"
        prepared.append(
            _Row(
                metric=str(metric),
                value=number,
                unit=unit,
                currency=currency,
                period=period,
                basis=basis,
                source_id=str(row.get("source_id") or "unknown"),
                filed=filed,
                concept=row.get("concept"),
                form=row.get("form"),
                accn=row.get("accn"),
                frame=row.get("frame"),
                raw=row,
                period_note=period_note,
            )
        )
    return prepared


def _sort_key(row: _Row) -> tuple[Any, ...]:
    return (
        row.metric,
        row.period.end or date.min,
        row.period.kind,
        row.basis,
        row.filed or date.min,
        row.source_id,
        row.value,
    )


def _make_fact(row: _Row, fact_id_factory: FactIdFactory) -> NormalizedFact:
    identity = {
        "metric": row.metric,
        "period_key": row.period.key(),
        "basis": row.basis,
        "source_id": row.source_id,
        "value": row.value,
        "filed": row.filed.isoformat() if row.filed else None,
    }
    notes: list[str] = []
    if row.period_note:
        notes.append(row.period_note)
    if row.filed is None:
        notes.append("filing date unknown; freshness classified from the period end only")
    return NormalizedFact(
        fact_id=fact_id_factory(identity),
        metric=row.metric,
        value=row.value,
        unit=row.unit,
        currency=row.currency,
        period=row.period,
        basis=row.basis,  # type: ignore[arg-type]
        source_id=row.source_id,
        raw_value=str(row.raw.get("raw_value", row.raw.get("value"))),
        extraction_method=str(row.raw.get("extraction_method") or "xbrl"),
        published_at=datetime.combine(row.filed, datetime.min.time(), tzinfo=UTC)
        if row.filed
        else None,
        notes=notes,
    )


def build_facts(
    rows: Iterable[dict[str, Any]],
    as_of: datetime,
    fact_id_factory: FactIdFactory | None = None,
) -> FactBuild:
    """Normalize extracted rows into facts, conflicts and notes (see the module docstring).

    Deterministic: the same rows and ``as_of`` produce the same facts in the same order with
    the same ids (the default id is a hash of the fact's identity).
    """
    factory = fact_id_factory or default_fact_id
    build = FactBuild()
    prepared = sorted(_prepare_rows(rows, as_of, build), key=_sort_key)

    # 1. exact duplicates: same metric, period, basis and value.
    seen: set[tuple[str, str, str, float]] = set()
    unique: list[_Row] = []
    duplicates = 0
    for row in prepared:
        key = (row.metric, row.period.key(), row.basis, row.value)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(row)
    if duplicates:
        build.notes.append(f"{duplicates} duplicate row(s) with identical values collapsed")

    # 2. group by identity (metric, period, basis) for restatements and conflicts.
    groups: dict[tuple[str, str, str], list[_Row]] = {}
    for row in unique:
        groups.setdefault(row.identity(), []).append(row)

    facts: list[NormalizedFact] = []
    winners: dict[tuple[str, str, str], NormalizedFact] = {}
    for identity, members in groups.items():
        members = sorted(members, key=lambda r: (r.filed or date.min, r.source_id, r.value))
        if len(members) == 1:
            fact = _make_fact(members[0], factory)
            facts.append(fact)
            winners[identity] = fact
            continue
        _resolve_group(members, identity, factory, facts, winners, build)

    # 3. cross-basis conflicts for the same metric and period (gaap vs adjusted).
    _cross_basis_conflicts(winners, build)
    # 4. period mismatches: same metric + fy/fp labelled differently with different values.
    _period_mismatch_conflicts(winners, build)
    # 5. derived facts.
    _derive(winners, factory, facts, build)

    facts.sort(
        key=lambda f: (
            f.metric,
            f.period.end or date.min,
            f.period.kind,
            f.basis,
            f.published_at or datetime.min.replace(tzinfo=UTC),
            f.fact_id,
        )
    )
    build.facts = facts
    if build.dropped:
        leaked = sum(1 for d in build.dropped if "after as_of" in d["reason"])
        if leaked:
            build.notes.append(
                f"{leaked} row(s) dated after as_of {as_of.date().isoformat()} dropped (leakage guard)"
            )
        other = len(build.dropped) - leaked
        if other:
            build.notes.append(f"{other} row(s) dropped as unusable (missing metric, value or date)")
    return build


def _resolve_group(
    members: list[_Row],
    identity: tuple[str, str, str],
    factory: FactIdFactory,
    facts: list[NormalizedFact],
    winners: dict[tuple[str, str, str], NormalizedFact],
    build: FactBuild,
) -> None:
    """Handle several distinct values for one (metric, period, basis)."""
    metric, _key, basis = identity
    period_label = members[0].period.label
    # Same filing date (or unknown dates) with different values -> conflict, not restatement.
    by_filed: dict[date | None, list[_Row]] = {}
    for row in members:
        by_filed.setdefault(row.filed, []).append(row)
    for filed, rows in by_filed.items():
        if len(rows) > 1:
            values = [r.value for r in rows]
            diff = _relative_diff(min(values), max(values))
            if diff > RESTATEMENT_TOLERANCE:
                build.conflicts.append(
                    Conflict(
                        metric=metric,
                        period_label=period_label,
                        status="unresolved",
                        reason="unknown",
                        values=[
                            ConflictValue(
                                value=r.value,
                                basis=r.basis,  # type: ignore[arg-type]
                                unit=r.unit,
                                source_id=r.source_id,
                                published_at=datetime.combine(
                                    r.filed, datetime.min.time(), tzinfo=UTC
                                )
                                if r.filed
                                else None,
                                period_label=period_label,
                            )
                            for r in rows
                        ],
                        material=diff > MATERIAL_THRESHOLD,
                        note=(
                            f"{metric} {period_label} ({basis}) reported as "
                            + " vs ".join(f"{r.value!r} ({r.source_id})" for r in rows)
                            + (
                                f" on the same filing date {filed.isoformat()}"
                                if filed
                                else " with unknown filing dates"
                            )
                            + "; both values preserved"
                        ),
                    )
                )
    # Restatement chain across filing dates: the latest filed value wins.
    dated = sorted(
        (r for r in members if r.filed is not None), key=lambda r: (r.filed, r.source_id)  # type: ignore[arg-type,return-value]
    )
    undated = [r for r in members if r.filed is None]
    if len(dated) >= 2 and dated[0].filed != dated[-1].filed:
        original, latest = dated[0], dated[-1]
        fact = _make_fact(latest, factory)
        diff = _relative_diff(original.value, latest.value)
        if diff > RESTATEMENT_TOLERANCE:
            fact.restated = True
            fact.original_value = original.value
            fact.original_source_id = original.source_id
            note = (
                f"restated: {metric} {period_label} ({basis}) originally {original.value!r} "
                f"({original.source_id}, filed {original.filed.isoformat()}), "  # type: ignore[union-attr]
                f"now {latest.value!r} ({latest.source_id}, filed {latest.filed.isoformat()}); "  # type: ignore[union-attr]
                f"{diff * 100:.2f}% change"
            )
            fact.notes.append(note)
            build.notes.append(note)
            if diff > MATERIAL_THRESHOLD:
                build.conflicts.append(
                    Conflict(
                        metric=metric,
                        period_label=period_label,
                        status="resolved_by_primary",
                        reason="restatement",
                        values=[
                            ConflictValue(
                                value=r.value,
                                basis=r.basis,  # type: ignore[arg-type]
                                unit=r.unit,
                                source_id=r.source_id,
                                published_at=datetime.combine(
                                    r.filed, datetime.min.time(), tzinfo=UTC
                                ),
                                period_label=period_label,
                            )
                            for r in (original, latest)
                        ],
                        material=True,
                        note="material restatement; the later filing is used, the original is preserved",
                    )
                )
        else:
            fact.notes.append(
                f"later filing {latest.filed.isoformat()} agrees with {original.filed.isoformat()} "  # type: ignore[union-attr]
                f"within {RESTATEMENT_TOLERANCE * 100:.1f}%"
            )
        facts.append(fact)
        winners[identity] = fact
        # Preserve intermediate and same-date alternates as their own facts (never dropped).
        # The original is already carried on the winner (original_value / original_source_id).
        for row in dated[1:]:
            if row is latest:
                continue
            alt = _make_fact(row, factory)
            alt.notes.append(f"superseded by the value filed {latest.filed.isoformat()}")  # type: ignore[union-attr]
            facts.append(alt)
        for row in undated:
            alt = _make_fact(row, factory)
            alt.notes.append("filing date unknown; not used as the current value")
            facts.append(alt)
        return
    # No restatement chain (all the same date, or undated): keep every value; the first in
    # deterministic order is the group's representative for derived facts and cross checks.
    representative: NormalizedFact | None = None
    for row in members:
        fact = _make_fact(row, factory)
        facts.append(fact)
        if representative is None:
            representative = fact
    if representative is not None:
        winners[identity] = representative


def _cross_basis_conflicts(
    winners: dict[tuple[str, str, str], NormalizedFact], build: FactBuild
) -> None:
    by_metric_period: dict[tuple[str, str], list[NormalizedFact]] = {}
    for (metric, key, _basis), fact in winners.items():
        by_metric_period.setdefault((metric, key), []).append(fact)
    for (metric, _key), facts in sorted(by_metric_period.items()):
        if len(facts) < 2:
            continue
        facts = sorted(facts, key=lambda f: (f.basis, f.fact_id))
        values = [f.value for f in facts]
        diff = _relative_diff(min(values), max(values))
        if diff <= RESTATEMENT_TOLERANCE:
            continue
        build.conflicts.append(
            Conflict(
                metric=metric,
                period_label=facts[0].period.label,
                status="conflict",
                reason="basis_mismatch",
                values=[
                    ConflictValue(
                        value=f.value,
                        basis=f.basis,
                        unit=f.unit,
                        source_id=f.source_id,
                        published_at=f.published_at,
                        period_label=f.period.label,
                    )
                    for f in facts
                ],
                material=diff > MATERIAL_THRESHOLD,
                note=(
                    f"{metric} {facts[0].period.label}: "
                    + " vs ".join(f"{f.basis} {f.value!r} ({f.source_id})" for f in facts)
                    + f"; {diff * 100:.2f}% apart; both preserved"
                ),
            )
        )


def _period_mismatch_conflicts(
    winners: dict[tuple[str, str, str], NormalizedFact], build: FactBuild
) -> None:
    by_label: dict[tuple[str, int, str, str], list[NormalizedFact]] = {}
    for (metric, _key, basis), fact in winners.items():
        period = fact.period
        if period.fiscal_year is None or not period.fiscal_period:
            continue
        if period.kind not in ("fiscal_quarter", "fiscal_year"):
            continue
        by_label.setdefault((metric, period.fiscal_year, period.fiscal_period, basis), []).append(
            fact
        )
    for (metric, fy, fp, _basis), facts in sorted(by_label.items()):
        if len(facts) < 2:
            continue
        facts = sorted(facts, key=lambda f: (f.period.end or date.min, f.fact_id))
        values = [f.value for f in facts]
        diff = _relative_diff(min(values), max(values))
        if diff <= RESTATEMENT_TOLERANCE:
            continue
        build.conflicts.append(
            Conflict(
                metric=metric,
                period_label=f"{fp} FY{fy}",
                status="conflict",
                reason="period_mismatch",
                values=[
                    ConflictValue(
                        value=f.value,
                        basis=f.basis,
                        unit=f.unit,
                        source_id=f.source_id,
                        published_at=f.published_at,
                        period_label=f"{f.period.start} to {f.period.end}",
                    )
                    for f in facts
                ],
                material=diff > MATERIAL_THRESHOLD,
                note=(
                    f"{metric} {fp} FY{fy} reported over different date ranges: "
                    + " vs ".join(
                        f"{f.value!r} ({f.period.start} to {f.period.end}, {f.source_id})"
                        for f in facts
                    )
                    + "; both preserved"
                ),
            )
        )


def _derive(
    winners: dict[tuple[str, str, str], NormalizedFact],
    factory: FactIdFactory,
    facts: list[NormalizedFact],
    build: FactBuild,
) -> None:
    existing = {(f.metric, f.period.key(), f.basis) for f in facts}
    for derived, (left_name, operator, right_name) in DERIVED_METRICS.items():
        for (metric, key, basis), left in sorted(winners.items()):
            if metric != left_name:
                continue
            right = winners.get((right_name, key, basis))
            if right is None:
                continue
            if (derived, key, basis) in existing:
                continue
            if left.currency != right.currency or left.unit != right.unit:
                build.notes.append(
                    f"{derived} not derived for {left.period.label}: {left_name} in "
                    f"{left.unit}/{left.currency} vs {right_name} in {right.unit}/{right.currency}"
                )
                continue
            if operator == "-|":
                value = left.value - abs(right.value)
                formula = f"{left_name} - |{right_name}|"
            else:
                value = left.value + right.value
                formula = f"{left_name} + {right_name}"
            identity = {
                "metric": derived,
                "period_key": key,
                "basis": basis,
                "source_id": "derived",
                "value": value,
                "filed": None,
            }
            fact = NormalizedFact(
                fact_id=factory(identity),
                metric=derived,
                value=value,
                unit=left.unit,
                currency=left.currency,
                period=left.period,
                basis=left.basis,
                source_id=left.source_id,
                raw_value=None,
                extraction_method="derived",
                published_at=max(
                    (p for p in (left.published_at, right.published_at) if p is not None),
                    default=None,
                ),
                notes=[
                    f"derived: {formula} = {left.value!r} {'-' if operator == '-|' else '+'} "
                    f"{abs(right.value) if operator == '-|' else right.value!r} "
                    f"from facts {left.fact_id}, {right.fact_id}"
                ],
            )
            facts.append(fact)
            existing.add((derived, key, basis))


# --------------------------------------------------------------------------------------
# Freshness (section 6)
# --------------------------------------------------------------------------------------


def classify_freshness(
    when: datetime | date | None, as_of: datetime | date, kind: str = "quarterly"
) -> Freshness:
    """Bucket a date's age at ``as_of`` as current / recent / stale / unknown.

    ``kind`` selects the thresholds: prices (``"price"``) are current up to 3 days and recent
    up to 14; quarterly fundamentals (``"quarterly"``, ``"fiscal_quarter"``, ``"ttm"``,
    ``"instant"``) are current up to 100 days after the period end and recent up to 200;
    annual (``"annual"``, ``"fiscal_year"``) 380 / 500. ``None`` -> ``unknown``, and so is a
    date after ``as_of`` (it cannot be aged and probably leaked).
    """
    when_date = _as_date(when)
    as_of_date = _as_date(as_of)
    if when_date is None or as_of_date is None:
        return "unknown"
    age = (as_of_date - when_date).days
    if age < 0:
        return "unknown"
    bucket = _KIND_TO_BUCKET.get(kind, "quarterly")
    current_max, recent_max = FRESHNESS_THRESHOLDS[bucket]
    if age <= current_max:
        return "current"
    if age <= recent_max:
        return "recent"
    return "stale"


def fact_freshness(fact: NormalizedFact, as_of: datetime | date) -> Freshness:
    """Freshness of one fact from its period end (fundamentals age from the period, not
    from the filing date: a 10-K filed yesterday still describes a year that ended months
    ago)."""
    return classify_freshness(fact.period.end, as_of, fact.period.kind)


def _latest_period_end(facts: Sequence[NormalizedFact], kinds: tuple[str, ...]) -> date | None:
    ends = [f.period.end for f in facts if f.period.kind in kinds and f.period.end is not None]
    return max(ends) if ends else None


def freshness_summary(
    facts: Sequence[NormalizedFact], prices: PriceSeries | None, as_of: datetime
) -> dict[str, Any]:
    """Counts per freshness bucket plus the dates and warnings that explain them.

    Shape::

        {"as_of": iso, "facts": {"current": n, "recent": n, "stale": n, "unknown": n,
                                  "oldest_period_end": iso|None, "newest_period_end": iso|None,
                                  "latest_quarter_end": iso|None, "latest_quarter_age_days": n|None,
                                  "latest_quarter_freshness": ..., "latest_annual_end": ...,
                                  "latest_annual_age_days": ..., "latest_annual_freshness": ...},
         "prices": {"freshness": ..., "latest_date": iso|None, "age_days": n|None,
                    "price_type": ..., "points": n},
         "warnings": [...]}
    """
    as_of_date = as_of.date()
    counts: dict[str, int] = {"current": 0, "recent": 0, "stale": 0, "unknown": 0}
    for fact in facts:
        counts[fact_freshness(fact, as_of)] += 1
    ends = [f.period.end for f in facts if f.period.end is not None]
    quarter_end = _latest_period_end(facts, ("fiscal_quarter",))
    annual_end = _latest_period_end(facts, ("fiscal_year",))
    warnings: list[str] = []

    def age(d: date | None) -> int | None:
        return (as_of_date - d).days if d else None

    fact_block: dict[str, Any] = {
        **counts,
        "total": len(facts),
        "oldest_period_end": min(ends).isoformat() if ends else None,
        "newest_period_end": max(ends).isoformat() if ends else None,
        "latest_quarter_end": quarter_end.isoformat() if quarter_end else None,
        "latest_quarter_age_days": age(quarter_end),
        "latest_quarter_freshness": classify_freshness(quarter_end, as_of, "quarterly")
        if quarter_end
        else "unknown",
        "latest_annual_end": annual_end.isoformat() if annual_end else None,
        "latest_annual_age_days": age(annual_end),
        "latest_annual_freshness": classify_freshness(annual_end, as_of, "annual")
        if annual_end
        else "unknown",
    }
    if not facts:
        warnings.append("no normalized fundamentals available")
    else:
        if quarter_end is None:
            warnings.append("no quarterly fundamentals available")
        elif fact_block["latest_quarter_freshness"] != "current":
            warnings.append(
                f"latest quarterly fundamentals are {age(quarter_end)} days old "
                f"(period ended {quarter_end.isoformat()}; {fact_block['latest_quarter_freshness']})"
            )
        if annual_end is not None and fact_block["latest_annual_freshness"] == "stale":
            warnings.append(
                f"latest annual fundamentals are {age(annual_end)} days old "
                f"(fiscal year ended {annual_end.isoformat()}; stale)"
            )
        if counts["unknown"]:
            warnings.append(f"{counts['unknown']} fact(s) have no usable period date")

    price_block: dict[str, Any]
    if prices is None or not prices.points:
        price_block = {
            "freshness": "unknown",
            "latest_date": None,
            "age_days": None,
            "price_type": prices.price_type if prices else None,
            "points": 0,
        }
        warnings.append("no price data available")
    else:
        latest = max(point.date for point in prices.points)
        price_freshness = classify_freshness(latest, as_of, "price")
        price_block = {
            "freshness": price_freshness,
            "latest_date": latest.isoformat(),
            "age_days": age(latest),
            "price_type": prices.price_type,
            "points": len(prices.points),
        }
        if price_freshness != "current":
            warnings.append(
                f"latest price is {age(latest)} days old ({latest.isoformat()}; {price_freshness})"
            )
    return {
        "as_of": as_of.isoformat(),
        "facts": fact_block,
        "prices": price_block,
        "warnings": warnings,
    }


def detect_stale_mix(
    price_series: PriceSeries | None, facts: Sequence[NormalizedFact], as_of: datetime
) -> list[str]:
    """Warnings for every way a current price could be silently paired with stale
    fundamentals (section 6). Empty when nothing is wrong."""
    warnings: list[str] = []
    if price_series is None or not price_series.points:
        return warnings
    as_of_date = as_of.date()
    latest_price_date = max(point.date for point in price_series.points)
    price_freshness = classify_freshness(latest_price_date, as_of, "price")
    price_desc = f"{price_series.price_type.replace('_', ' ')} price dated {latest_price_date.isoformat()}"
    if price_series.price_type in ("intraday", "pre_market", "after_hours"):
        warnings.append(
            f"latest price is a {price_series.price_type.replace('_', ' ')} print, not a "
            "regular-session close; label it when comparing with closing series"
        )
    if price_freshness == "stale":
        warnings.append(
            f"price data is stale: latest point {latest_price_date.isoformat()} is "
            f"{(as_of_date - latest_price_date).days} days before as_of"
        )
        return warnings
    quarter_end = _latest_period_end(facts, ("fiscal_quarter",))
    annual_end = _latest_period_end(facts, ("fiscal_year",))
    if quarter_end is not None:
        freshness = classify_freshness(quarter_end, as_of, "quarterly")
        if freshness == "stale":
            warnings.append(
                f"{price_desc} would be paired with stale quarterly fundamentals "
                f"(latest quarter ended {quarter_end.isoformat()}, "
                f"{(as_of_date - quarter_end).days} days old)"
            )
    elif annual_end is not None:
        freshness = classify_freshness(annual_end, as_of, "annual")
        if freshness == "stale":
            warnings.append(
                f"{price_desc} would be paired with stale annual fundamentals "
                f"(fiscal year ended {annual_end.isoformat()}, "
                f"{(as_of_date - annual_end).days} days old)"
            )
        else:
            warnings.append(
                f"{price_desc} is paired with annual fundamentals only (no quarterly facts); "
                f"latest fiscal year ended {annual_end.isoformat()}"
            )
    elif facts:
        warnings.append(f"{price_desc} cannot be aged against fundamentals: no dated periods")
    else:
        warnings.append(f"{price_desc} has no fundamentals to pair with")
    return warnings
