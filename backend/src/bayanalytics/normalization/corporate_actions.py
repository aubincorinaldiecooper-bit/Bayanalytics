"""Corporate-action handling for historical comparisons (AGENT.md section 32).

Not a corporate-actions engine: the goal is to prevent obviously misleading historical
comparisons. Splits are applied to price series, identity breaks (mergers, acquisitions,
spin-offs, ticker changes, fiscal-year changes) produce warnings and exclude periods that
describe a different business, restatements are flagged, and dividends trigger the
price-return versus total-return note.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from bayanalytics.schemas.evidence import CorporateAction, NormalizedFact, PriceSeries

IDENTITY_BREAK_KINDS: frozenset[str] = frozenset(
    {"merger", "acquisition", "spin_off", "ticker_change", "share_class_change"}
)
COMPARABILITY_BREAK_KINDS: frozenset[str] = frozenset({"merger", "acquisition", "spin_off"})

_RATIO_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?::|-?\s*for\s*-?|/|to)\s*(\d+(?:\.\d+)?)", re.IGNORECASE
)


def parse_split_ratio(detail: str, kind: str = "split") -> float | None:
    """Parse ``"4:1"``, ``"4-for-1"``, ``"4 for 1"``, ``"1:10"`` into new-shares-per-old-share.

    A 4:1 forward split returns ``4.0`` (pre-split prices are divided by 4); a 1:10 reverse
    split returns ``0.1`` (pre-split prices are multiplied by 10). For ``kind ==
    "reverse_split"`` a ratio written the other way round (``"10:1"``) is inverted, since a
    reverse split always reduces the share count. ``None`` when nothing parses.
    """
    if not detail:
        return None
    match = _RATIO_RE.search(detail)
    if match is None:
        return None
    new_shares = float(match.group(1))
    old_shares = float(match.group(2))
    if new_shares <= 0 or old_shares <= 0:
        return None
    ratio = new_shares / old_shares
    if kind == "reverse_split" and ratio > 1:
        ratio = 1.0 / ratio
    if kind == "split" and ratio == 1:
        return None
    return ratio


def _scaled(value: float | None, ratio: float) -> float | None:
    return None if value is None else value / ratio


def apply_split_adjustment(series: PriceSeries, actions: Sequence[CorporateAction]) -> PriceSeries:
    """Return a split-adjusted copy of ``series`` (the input is never mutated).

    For every split / reverse split with an effective date and a parsable ratio, points
    dated before the effective date have open/high/low/close divided by the ratio and
    volume multiplied by it. A series already marked ``split_adjusted`` is returned as an
    unchanged copy so an adjustment is never applied twice. Actions without a date or ratio
    are noted in the label rather than guessed.
    """
    if series.split_adjusted:
        return series.model_copy(deep=True)
    applied: list[str] = []
    skipped: list[str] = []
    points = [point.model_copy() for point in series.points]
    for action in sorted(
        (a for a in actions if a.kind in ("split", "reverse_split")),
        key=lambda a: (a.effective is None, a.effective),
    ):
        ratio = parse_split_ratio(action.detail, action.kind)
        if action.effective is None or ratio is None:
            skipped.append(f"{action.kind} {action.detail!r} (no date or ratio; not applied)")
            continue
        for point in points:
            if point.date < action.effective:
                point.open = _scaled(point.open, ratio)
                point.high = _scaled(point.high, ratio)
                point.low = _scaled(point.low, ratio)
                point.close = point.close / ratio
                if point.volume is not None:
                    point.volume = point.volume * ratio
        applied.append(f"{action.detail} on {action.effective.isoformat()}")
    label_parts = [series.label] if series.label else []
    if applied:
        label_parts.append("split-adjusted (" + "; ".join(applied) + ")")
    if skipped:
        label_parts.append("unapplied: " + "; ".join(skipped))
    # Adjusted means every known split is accounted for; a split that could not be applied
    # (no date or ratio) leaves the series honestly unadjusted.
    return series.model_copy(
        update={
            "points": points,
            "split_adjusted": not skipped,
            "label": "; ".join(label_parts),
        }
    )


def _when(action: CorporateAction) -> str:
    return action.effective.isoformat() if action.effective else "unknown date"


def detect_identity_breaks(actions: Sequence[CorporateAction]) -> list[str]:
    """Warnings for events after which the company is not the same comparable entity."""
    warnings: list[str] = []
    for action in sorted(actions, key=lambda a: (a.effective is None, a.effective, a.kind)):
        detail = f" ({action.detail})" if action.detail else ""
        if action.kind in ("merger", "acquisition"):
            warnings.append(
                f"{action.kind} on {_when(action)}{detail}: pre- and post-{action.kind} periods "
                "describe different businesses; do not compare them as the same entity"
            )
        elif action.kind == "spin_off":
            warnings.append(
                f"spin-off on {_when(action)}{detail}: financials before the spin-off include the "
                "separated business; comparisons across this date are not like-for-like"
            )
        elif action.kind == "ticker_change":
            warnings.append(
                f"ticker change on {_when(action)}{detail}: price history under the previous "
                "identifier must be linked explicitly; identifier history preserved"
            )
        elif action.kind == "name_change":
            warnings.append(
                f"name change on {_when(action)}{detail}: older filings and coverage appear "
                "under the former name; the reporting entity is unchanged"
            )
        elif action.kind == "share_class_change":
            warnings.append(
                f"share-class change on {_when(action)}{detail}: per-share figures before and "
                "after may not be comparable"
            )
        elif action.kind == "fiscal_year_change":
            warnings.append(
                f"fiscal-year change on {_when(action)}{detail}: fiscal periods before and after "
                "cover different months; quarter and year labels are not aligned across it"
            )
    return warnings


def comparable_periods(
    facts: Sequence[NormalizedFact], actions: Sequence[CorporateAction]
) -> tuple[list[NormalizedFact], list[str]]:
    """Facts that may be compared as one entity, plus warnings.

    Periods ending before a merger, acquisition or spin-off are excluded (they describe a
    different business); an identity break without a date excludes nothing but warns that
    the affected periods cannot be determined. Restated facts and restatement /
    accounting-policy / fiscal-year-change actions are flagged but kept, so the analyst
    sees them. Nothing is modified.
    """
    warnings = detect_identity_breaks(actions)
    cutoffs = [
        a.effective
        for a in actions
        if a.kind in COMPARABILITY_BREAK_KINDS and a.effective is not None
    ]
    undated = [a for a in actions if a.kind in COMPARABILITY_BREAK_KINDS and a.effective is None]
    for action in undated:
        warnings.append(
            f"{action.kind} without an effective date: cannot tell which periods predate it; "
            "treat all cross-period comparisons with caution"
        )
    kept: list[NormalizedFact] = []
    excluded = 0
    latest_cutoff = max(cutoffs) if cutoffs else None
    for fact in facts:
        end = fact.period.end
        if latest_cutoff is not None and end is not None and end < latest_cutoff:
            excluded += 1
            continue
        kept.append(fact)
    if excluded:
        warnings.append(
            f"{excluded} fact(s) for periods ending before {latest_cutoff.isoformat()} excluded "  # type: ignore[union-attr]
            "from like-for-like comparison (pre-transaction entity)"
        )
    restated = sorted({f"{f.metric} {f.period.label}" for f in kept if f.restated})
    if restated:
        warnings.append(
            "restated periods in comparison (original values preserved on the facts): "
            + ", ".join(restated)
        )
    for action in actions:
        if action.kind == "restatement":
            warnings.append(
                f"restatement on {_when(action)}"
                + (f" ({action.detail})" if action.detail else "")
                + ": previously reported figures may differ from the current filing"
            )
        elif action.kind == "accounting_policy_change":
            warnings.append(
                f"accounting-policy change on {_when(action)}"
                + (f" ({action.detail})" if action.detail else "")
                + ": figures before and after are not on the same basis"
            )
    return kept, warnings


def total_return_note(actions: Sequence[CorporateAction]) -> str | None:
    """A note distinguishing price return from total return when dividends are present."""
    dividends = [a for a in actions if a.kind == "dividend"]
    if not dividends:
        return None
    dated = sorted(a.effective for a in dividends if a.effective is not None)
    span = ""
    if dated:
        span = f" between {dated[0].isoformat()} and {dated[-1].isoformat()}"
    details = [a.detail for a in dividends if a.detail]
    sample = f" (e.g. {details[0]})" if details else ""
    return (
        f"{len(dividends)} dividend(s) recorded{span}{sample}: returns shown are price returns "
        "and exclude dividends; total return (with dividends reinvested) would be higher"
    )
