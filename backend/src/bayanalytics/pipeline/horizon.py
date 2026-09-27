"""Deterministic time-horizon resolution (AGENT.md section 28).

Rules: an explicit request wins; otherwise the query is scanned for unambiguous horizon
language. Anything ambiguous, or "what might happen" style questions, resolves to a
multi-horizon assessment. No model is involved.
"""

from __future__ import annotations

import re

from bayanalytics.schemas.common import ResolvedHorizon

_PATTERNS: dict[str, list[str]] = {
    "near_term": [
        r"\bthis week\b",
        r"\bnext week\b",
        r"\bnext (?:few|couple of|several) (?:days|weeks)\b",
        r"\bcoming (?:days|weeks)\b",
        r"\b(?:short|near)[- ]term\b",
        r"\bintraday\b",
        r"\btomorrow\b",
        r"\bthis month\b",
        r"\bin the (?:next|coming) (?:\d+ )?(?:days|weeks)\b",
    ],
    "next_cycle": [
        r"\bnext quarter\b",
        r"\bthis quarter\b",
        r"\b(?:next|upcoming|coming) earnings\b",
        r"\bearnings (?:report|call|release|season)\b",
        r"\bnext report\b",
        r"\bq[1-4]\b",
        r"\bnext cycle\b",
    ],
    "medium_term": [
        r"\b(?:6|six)(?:\s*(?:-|to|\u2013)+\s*(?:12|twelve))? months?\b",
        r"\b(?:12|twelve) months?\b",
        r"\bnext year\b",
        r"\b(?:one|1)[- ]year\b",
        r"\bover the (?:next|coming) year\b",
        r"\bmedium[- ]term\b",
        r"\bintermediate[- ]term\b",
        r"\b(?:1[3-9]|2[0-4]) months\b",
    ],
    "long_term": [
        r"\blong[- ]term\b",
        r"\bmulti[- ]year\b",
        r"\bdecade\b",
        r"\b(?:[2-9]|\d{2}|two|three|five|ten)\+?[- ]years?\b",
        r"\bover the (?:next|coming) (?:\d+|several|many) years\b",
        r"\bsecular\b",
    ],
    "multi_horizon": [
        r"\bwhat (?:might|could|will|may) happen\b",
        r"\bwhat(?:'s| is) next\b",
        r"\bwhat happens next\b",
        r"\boutlook\b",
        r"\bwhere is .* (?:headed|going)\b",
        r"\bacross horizons?\b",
        r"\ball horizons\b",
    ],
}

_COMPILED = {name: [re.compile(p, re.IGNORECASE) for p in pats] for name, pats in _PATTERNS.items()}


def detect_horizons(query: str) -> list[str]:
    found: list[str] = []
    for name, patterns in _COMPILED.items():
        if any(p.search(query) for p in patterns):
            found.append(name)
    return found


def resolve_horizon(query: str, requested: str = "auto") -> ResolvedHorizon:
    if requested != "auto":
        return requested  # type: ignore[return-value]
    found = detect_horizons(query)
    specific = [h for h in found if h != "multi_horizon"]
    if len(specific) == 1:
        # "What might happen to NVDA next quarter?" names a period: that period wins.
        return specific[0]  # type: ignore[return-value]
    return "multi_horizon"


def horizons_for(resolved: str) -> list[str]:
    if resolved == "multi_horizon":
        return ["near_term", "next_cycle", "medium_term", "long_term"]
    return [resolved]
