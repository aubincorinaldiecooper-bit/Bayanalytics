"""Parse Spark's sectioned markdown answer into ``Assessment`` / ``HorizonAssessment``.

Tolerant by design: headings may use ``##``, ``#`` or ``**bold**`` forms, sections may be
missing or reordered, and text before the first heading is treated as noise unless no Summary
section exists. Stance and confidence are parsed for completeness only; the orchestrator
overrides them with Laya's recorded decision (Spark explains, it does not decide).
"""

from __future__ import annotations

import re
from typing import Any, cast

from bayanalytics.schemas.common import SINGLE_HORIZONS, Stance
from bayanalytics.schemas.results import Assessment, EvidenceItem, HorizonAssessment
from bayanalytics.spark.prompt import SECTION_HEADINGS, STANCES

SECTION_KEYS: dict[str, str] = {
    heading: re.sub(r"[^a-z0-9]+", "_", heading.lower()).strip("_") for heading in SECTION_HEADINGS
}
"""Heading -> canonical key (``What changed`` -> ``what_changed``)."""

HORIZON_KEY_PREFIX = "horizon_"
EVIDENCE_LIST_KEYS: tuple[str, ...] = ("what_changed", "bull_evidence", "bear_evidence", "risks")
CONTEXT_DICT_KEYS: tuple[str, ...] = (
    "fundamentals",
    "valuation",
    "benchmark_context",
    "historical_context",
    "market_context",
)

_NORMALIZED_HEADINGS: dict[str, str] = {
    re.sub(r"[^a-z0-9]+", "", heading.lower()): key for heading, key in SECTION_KEYS.items()
}
_HASH_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*#*\s*$")
_BOLD_HEADING_RE = re.compile(r"^\s{0,3}(?:\*\*|__)\s*(.+?)\s*(?:\*\*|__)\s*:?\s*$")
_NUMBER_PREFIX_RE = re.compile(r"^\d+[.)]\s*")
_HORIZON_RE = re.compile(r"^horizon\s*[:\-]\s*(.+)$", re.IGNORECASE)
_STANCE_RE = re.compile(
    r"^\s*(?:[-*]\s*)?\**\s*stance\s*\**\s*:\s*\**\s*(bullish|neutral|bearish|mixed)\b",
    re.IGNORECASE | re.MULTILINE,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+(.*)$")
_CITATION_BLOCK_RE = re.compile(r"\[([^\[\]]*?src_[^\[\]]*)\]")
_SOURCE_ID_RE = re.compile(r"src_[A-Za-z0-9]+")
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*$")


def canonical_heading(text: str) -> str | None:
    """Map raw heading text to a canonical key, or ``None`` when it is not a known heading."""
    raw = _NUMBER_PREFIX_RE.sub("", text.strip().strip("*_ ").rstrip(":").strip())
    horizon = _HORIZON_RE.match(raw)
    if horizon:
        name = re.sub(r"[^a-z0-9]+", "_", horizon.group(1).lower()).strip("_")
        for known in (*SINGLE_HORIZONS, "multi_horizon"):
            if name == known or name.startswith(f"{known}_"):
                name = known
                break
        return f"{HORIZON_KEY_PREFIX}{name}" if name else None
    return _NORMALIZED_HEADINGS.get(re.sub(r"[^a-z0-9]+", "", raw.lower()))


def parse_sections(text: str) -> dict[str, str]:
    """Split the answer into ``{canonical_key: body}``; everything becomes summary if unheaded."""
    lines = [line for line in text.splitlines() if not _FENCE_RE.match(line)]
    sections: dict[str, list[str]] = {}
    order: list[str] = []
    current: str | None = None
    preamble: list[str] = []
    for line in lines:
        key = _heading_key(line)
        if key is not None:
            current = key
            if key not in sections:
                sections[key] = []
                order.append(key)
            continue
        if current is None:
            preamble.append(line)
        else:
            sections[current].append(line)
    result = {key: "\n".join(sections[key]).strip() for key in order}
    preamble_text = "\n".join(preamble).strip()
    if not result:
        return {"summary": preamble_text} if preamble_text else {}
    if "summary" not in result and preamble_text:
        result = {"summary": preamble_text, **result}
    return result


def _heading_key(line: str) -> str | None:
    match = _HASH_HEADING_RE.match(line)
    if match:
        return canonical_heading(match.group(1))
    match = _BOLD_HEADING_RE.match(line)
    if match:
        return canonical_heading(match.group(1))
    return None


def extract_citations(text: str) -> list[str]:
    """Ordered, de-duplicated ``src_`` ids cited in square brackets."""
    seen: list[str] = []
    for block in _CITATION_BLOCK_RE.findall(text):
        for source_id in _SOURCE_ID_RE.findall(block):
            if source_id not in seen:
                seen.append(source_id)
    return seen


def parse_bullets(section_text: str) -> list[str]:
    """Bullet items (continuation lines joined); paragraphs when the section has no bullets."""
    _, bullets = split_prose_and_bullets(section_text)
    if bullets:
        return bullets
    return [p for p in re.split(r"\n\s*\n", section_text.strip()) if p.strip()]


def split_prose_and_bullets(section_text: str) -> tuple[str, list[str]]:
    prose: list[str] = []
    bullets: list[str] = []
    for line in section_text.splitlines():
        match = _BULLET_RE.match(line)
        if match:
            bullets.append(match.group(1).strip())
        elif bullets and line.strip() and line.startswith((" ", "\t")):
            bullets[-1] = f"{bullets[-1]} {line.strip()}"
        else:
            prose.append(line)
    prose_text = re.sub(r"\n{3,}", "\n\n", "\n".join(prose)).strip()
    return prose_text, [b for b in bullets if b]


def parse_stance(horizon_section_text: str) -> Stance | None:
    match = _STANCE_RE.search(horizon_section_text)
    if not match:
        return None
    value = match.group(1).lower()
    return cast(Stance, value) if value in STANCES else None


def conflict_notes(sections: dict[str, str]) -> list[str]:
    """Spark's prose about conflicts (``Assessment.conflicts`` holds structured records only)."""
    return parse_bullets(sections.get("conflicts", ""))


def to_assessment(
    sections: dict[str, str], known_source_ids: set[str]
) -> tuple[Assessment, dict[str, HorizonAssessment], list[str]]:
    warnings: list[str] = []
    unknown: list[str] = []

    def item(text: str, section: str, stance: Stance | None = None) -> EvidenceItem:
        cited = extract_citations(text)
        known = [c for c in cited if c in known_source_ids]
        for c in cited:
            if c not in known_source_ids and c not in unknown:
                unknown.append(c)
        if not cited:
            warnings.append(f"uncited claim in {section}")
        return EvidenceItem(text=text, source_ids=known, stance=stance)

    def items(key: str, stance: Stance | None = None) -> list[EvidenceItem]:
        return [item(b, key, stance) for b in parse_bullets(sections.get(key, ""))]

    def context(key: str) -> dict[str, Any]:
        body = sections.get(key, "")
        if not body:
            return {}
        cited = extract_citations(body)
        for c in cited:
            if c not in known_source_ids and c not in unknown:
                unknown.append(c)
        return {
            "text": body,
            "items": parse_bullets(body),
            "source_ids": [c for c in cited if c in known_source_ids],
        }

    summary = sections.get("summary", "")
    if not summary:
        warnings.append("spark response has no summary section")
    for c in extract_citations(summary):
        if c not in known_source_ids and c not in unknown:
            unknown.append(c)

    assessment = Assessment(
        summary=summary,
        what_changed=items("what_changed"),
        fundamentals=context("fundamentals"),
        valuation=context("valuation"),
        benchmark_context=context("benchmark_context"),
        historical_context=context("historical_context"),
        market_context=context("market_context"),
        bull_evidence=items("bull_evidence", "bullish"),
        bear_evidence=items("bear_evidence", "bearish"),
        risks=items("risks"),
        uncertainties=parse_bullets(sections.get("uncertainties", "")),
        follow_up_questions=parse_bullets(sections.get("follow_up_questions", "")),
    )

    horizons: dict[str, HorizonAssessment] = {}
    for key, body in sections.items():
        if not key.startswith(HORIZON_KEY_PREFIX):
            continue
        name = key[len(HORIZON_KEY_PREFIX) :]
        stance = parse_stance(body)
        if stance is None:
            warnings.append(f"spark horizon section '{name}' has no stance line")
        stripped = _STANCE_RE.sub("", body, count=1).strip()
        prose, bullets = split_prose_and_bullets(stripped)
        key_evidence = [item(b, key) for b in bullets]
        summary_text = prose or (bullets[0] if bullets else "")
        for c in extract_citations(prose):
            if c not in known_source_ids and c not in unknown:
                unknown.append(c)
        horizons[name] = HorizonAssessment(
            horizon=name,
            stance=stance or "mixed",
            summary=summary_text,
            key_evidence=key_evidence,
        )

    for source_id in unknown:
        warnings.append(f"spark cited unknown source [{source_id}]")
    return assessment, horizons, warnings
