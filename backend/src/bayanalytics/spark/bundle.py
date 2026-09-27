"""Context budgeting for the Spark evidence bundle (AGENT.md sections 8 and 24).

``fit_bundle`` applies the section-24 overflow policy in order and records every trim as a
human-readable string the orchestrator surfaces as an uncertainty. Protected sections are never
touched: ``calculated_metrics``, ``conflicts``, ``uncertainties``, ``laya_assessments``,
``current_metrics``, ``freshness``, ``question_focus``.

Every token count is **measured**: ``fit_bundle`` renders the candidate prompt and asks the
Spark session for the number of tokens the server will see (its own chat template and
tokenizer), so "fits" here is exactly "fits in llama-server".
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from bayanalytics.instruments.base import SparkEvidenceBundle
from bayanalytics.schemas.common import is_primary_source, source_rank
from bayanalytics.spark.base import SparkMessage, SparkRunOptions
from bayanalytics.spark.prompt import build_messages, clean_text

PromptCounter = Callable[[list[SparkMessage]], Awaitable[int]]
"""Measured token count of a message list as the server will process it."""

OUTPUT_MARGIN_TOKENS = 64
MAX_ANALOGUES = 3
MAX_EVENTS = 8
TRIMMED_EXCERPT_CHARS = 300
LOW_RANK_THRESHOLD = 7
"""Source ranks at or above this (secondary_commentary, unverified_web) are the low-ranked
excerpts dropped in step 2; other non-primary excerpts survive until step 6."""

OVERFLOW_TRIM = "evidence exceeds context ceiling even after trimming"


def reserved_output_tokens(options: SparkRunOptions | None) -> int:
    opts = options or SparkRunOptions()
    return opts.max_tokens + OUTPUT_MARGIN_TOKENS


@dataclass
class FitResult:
    bundle: SparkEvidenceBundle
    trims: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    """Measured size of the final prompt (system + user messages, templated)."""
    budget: int = 0
    measurements: int = 0
    """How many prompt measurements the fit needed (each is one server round trip)."""

    @property
    def overflow(self) -> bool:
        return OVERFLOW_TRIM in self.trims


async def fit_bundle(
    bundle: SparkEvidenceBundle,
    context_ceiling: int,
    options: SparkRunOptions | None,
    count_prompt: PromptCounter,
) -> FitResult:
    """Trim ``bundle`` until the full prompt (system + instructions + evidence) fits.

    Budget = ``context_ceiling - reserved_output_tokens(options)``; every "does it fit" is a
    measurement of the rendered prompt by ``count_prompt``. Steps, in order, each applied only
    while still over budget:

    1. remove duplicate excerpts (same source id and text);
    2. drop excerpts from low-ranked non-primary sources, least authoritative first;
    3. keep at most 3 historical analogues;
    4. keep at most 8 important events (material ones first);
    5. truncate excerpt text to 300 characters;
    6. drop every remaining non-primary excerpt.

    If the bundle still does not fit, the trims end with ``OVERFLOW_TRIM`` and the caller
    decides (record an uncertainty, suggest Deep, or refuse).
    """
    budget = context_ceiling - reserved_output_tokens(options)
    trims: list[str] = []
    current = bundle.model_copy(deep=True)
    result = FitResult(bundle=current, trims=trims, budget=budget)

    async def fits() -> bool:
        result.prompt_tokens = await count_prompt(build_messages(current, options))
        result.measurements += 1
        return result.prompt_tokens <= budget

    def done() -> FitResult:
        result.bundle = current
        return result

    if await fits():
        return done()

    # 1. duplicates
    excerpts, removed = _dedupe_excerpts(current.excerpts)
    if removed:
        current = current.model_copy(update={"excerpts": excerpts})
        trims.append(f"removed {removed} duplicate excerpt(s)")
        if await fits():
            return done()

    # 2. low-ranked non-primary excerpts, least authoritative first
    meta = _source_meta(current)
    dropped = 0
    while not await fits():
        index = _worst_excerpt_index(current.excerpts, meta, min_rank=LOW_RANK_THRESHOLD)
        if index is None:
            break
        excerpts = list(current.excerpts)
        del excerpts[index]
        current = current.model_copy(update={"excerpts": excerpts})
        dropped += 1
    if dropped:
        trims.append(f"dropped {dropped} excerpt(s) from low-ranked non-primary sources")
        if await fits():
            return done()

    # 3. analogues beyond 3
    if len(current.historical_analogues) > MAX_ANALOGUES:
        removed = len(current.historical_analogues) - MAX_ANALOGUES
        current = current.model_copy(
            update={"historical_analogues": list(current.historical_analogues[:MAX_ANALOGUES])}
        )
        trims.append(f"dropped {removed} historical analogue(s) beyond the first {MAX_ANALOGUES}")
        if await fits():
            return done()

    # 4. events beyond 8, material first
    if len(current.important_events) > MAX_EVENTS:
        removed = len(current.important_events) - MAX_EVENTS
        ordered = sorted(
            enumerate(current.important_events),
            key=lambda item: (not item[1].get("material"), item[0]),
        )
        kept = sorted(ordered[:MAX_EVENTS], key=lambda item: item[0])
        current = current.model_copy(update={"important_events": [e for _, e in kept]})
        trims.append(f"dropped {removed} non-material event(s) beyond the first {MAX_EVENTS}")
        if await fits():
            return done()

    # 5. truncate excerpt text
    excerpts, truncated = _truncate_excerpts(current.excerpts, TRIMMED_EXCERPT_CHARS)
    if truncated:
        current = current.model_copy(update={"excerpts": excerpts})
        trims.append(f"truncated {truncated} excerpt(s) to {TRIMMED_EXCERPT_CHARS} characters")
        if await fits():
            return done()

    # 6. drop every remaining non-primary excerpt
    excerpts = [e for e in current.excerpts if _is_primary_excerpt(e, meta)]
    removed = len(current.excerpts) - len(excerpts)
    if removed:
        current = current.model_copy(update={"excerpts": excerpts})
        trims.append(f"dropped {removed} remaining excerpt(s) from non-primary sources")
        if await fits():
            return done()

    if not await fits():
        trims.append(OVERFLOW_TRIM)
    return done()


# --- helpers ----------------------------------------------------------------------------------


def _excerpt_text(entry: dict[str, Any]) -> str:
    for key in ("text", "excerpt", "fact"):
        value = entry.get(key)
        if value:
            return str(value)
    return ""


def _excerpt_source(entry: dict[str, Any]) -> str:
    source = entry.get("source_id")
    if not source:
        ids = entry.get("source_ids")
        if isinstance(ids, list) and ids:
            source = ids[0]
    return str(source or "")


def _dedupe_excerpts(excerpts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[tuple[str, str]] = set()
    kept: list[dict[str, Any]] = []
    for entry in excerpts:
        digest = hashlib.sha1(clean_text(_excerpt_text(entry)).lower().encode()).hexdigest()
        key = (_excerpt_source(entry), digest)
        if key in seen:
            continue
        seen.add(key)
        kept.append(entry)
    return kept, len(excerpts) - len(kept)


def _source_meta(bundle: SparkEvidenceBundle) -> dict[str, tuple[int, bool]]:
    """source_id -> (rank, is_primary) from ``bundle.sources``."""
    meta: dict[str, tuple[int, bool]] = {}
    for source in bundle.sources:
        if not isinstance(source, dict) or not source.get("source_id"):
            continue
        source_type = str(source.get("source_type") or "unverified_web")
        rank = source.get("rank")
        rank = int(rank) if isinstance(rank, int | float) else source_rank(source_type)
        primary = source.get("is_primary")
        primary = bool(primary) if primary is not None else is_primary_source(source_type)
        meta[str(source["source_id"])] = (rank, primary)
    return meta


def _excerpt_rank(entry: dict[str, Any], meta: dict[str, tuple[int, bool]]) -> tuple[int, bool]:
    rank = entry.get("rank")
    primary = entry.get("is_primary")
    if rank is None or primary is None:
        source_type = entry.get("source_type")
        if source_type:
            rank = rank if rank is not None else source_rank(str(source_type))
            primary = primary if primary is not None else is_primary_source(str(source_type))
    known = meta.get(_excerpt_source(entry), (8, False))
    return (
        int(rank) if isinstance(rank, int | float) else known[0],
        bool(primary) if primary is not None else known[1],
    )


def _is_primary_excerpt(entry: dict[str, Any], meta: dict[str, tuple[int, bool]]) -> bool:
    return _excerpt_rank(entry, meta)[1]


def _worst_excerpt_index(
    excerpts: list[dict[str, Any]], meta: dict[str, tuple[int, bool]], min_rank: int
) -> int | None:
    worst: tuple[int, int] | None = None
    for index, entry in enumerate(excerpts):
        rank, primary = _excerpt_rank(entry, meta)
        if primary or rank < min_rank:
            continue
        candidate = (rank, index)
        if worst is None or candidate > worst:
            worst = candidate
    return None if worst is None else worst[1]


def _truncate_excerpts(
    excerpts: list[dict[str, Any]], limit: int
) -> tuple[list[dict[str, Any]], int]:
    out: list[dict[str, Any]] = []
    truncated = 0
    for entry in excerpts:
        text = _excerpt_text(entry)
        if len(text) > limit:
            new_entry = dict(entry)
            for key in ("text", "excerpt", "fact"):
                if new_entry.get(key):
                    new_entry[key] = clean_text(text, limit)
                    break
            out.append(new_entry)
            truncated += 1
        else:
            out.append(entry)
    return out, truncated
