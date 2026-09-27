"""State compaction and question validation for Laya (AGENT.md sections 1.2, 3.1 and 30).

Laya's English checkpoint renders every question as::

    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]

and then silently truncates the *state* to whatever is left of ``max_len`` (512) tokens; the
option/instruction head is squeezed into ``head_max_len`` (192) tokens, again silently, and only
throws when the options do not fit at all. The spec forbids relying on that silent truncation, so
this module compacts states deterministically *before* the call and rejects heads that would be
truncated.

Every token figure here is **measured** with the loaded bundle's own tokenizer, through the
``TokenCounter`` the worker exposes (``LayaClient.count_tokens``). Nothing is estimated: the
sequence arithmetic below mirrors ``@receptron/laya``'s ``buildSequence`` exactly, so "fits here"
means "fits in the worker".
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from bayanalytics.laya.base import LAYA_HEAD_MAX_LEN, LAYA_MAX_CHOICE_OPTIONS, LAYA_MAX_LEN
from bayanalytics.schemas.decisions import LayaQuestion

TokenCounter = Callable[[Sequence[str]], Awaitable[list[int]]]
"""Measured token counts for a batch of texts (no special tokens), in order."""

# Sequence frame around head + state: [CLS], [SEP] after the instructions, [SEP] after the
# options and the final [SEP] (``buildSequence``: ``room = maxLen - seq.length - 1``).
SEQUENCE_SPECIALS = 4
MIN_STATE_BUDGET_TOKENS = 64

MAX_STRING_CHARS = 240
MAX_LIST_ITEMS = 8
# Laya keeps at most 48 tokens per option text before it starts shrinking heads.
LAYA_MAX_OPTION_TOKENS = 48
# Below this many tokens left for the instructions Laya shrinks every option evenly.
MIN_INSTRUCTION_TOKENS = 16
ELLIPSIS = "…"

# Keys kept first (in this order) when nothing more specific is requested. Unknown keys follow
# alphabetically, so the least structured, most verbose material is dropped first.
DEFAULT_STATE_PRIORITY: tuple[str, ...] = (
    "instrument",
    "symbol",
    "company",
    "question",
    "horizon",
    "period",
    "as_of",
    "facts",
    "metrics",
    "calculations",
    "evidence_gaps",
    "sources_count",
    "freshness",
    "guidance_hint",
    "sentiment_hint",
    "evidence",
    "history",
    "notes",
)

_NOUL_OPTIONS = ("false: no, the statement does not hold", "true: yes, the statement holds")
_MASK = "[MASK]"


# ---- serialisation ------------------------------------------------------------------------


def laya_json(obj: Any) -> str:
    """Serialise the way ``@receptron/laya`` renders a JSON state (Python ``json.dumps`` style)."""
    if isinstance(obj, str):
        return obj
    return json.dumps(obj, ensure_ascii=False, default=str)


def canonical_json(obj: Any) -> str:
    """Stable JSON for digests: sorted keys, compact separators, non-JSON values stringified."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def state_digest(state: Any) -> str:
    """sha256 hex of the canonical JSON of ``state`` (stable across key order)."""
    return hashlib.sha256(canonical_json(state).encode("utf-8")).hexdigest()


def _scrub(text: str) -> str:
    """``buildSequence`` replaces literal ``[MASK]`` markers in text with a space."""
    return text.replace(_MASK, " ")


# ---- question rendering (exactly as Laya renders it) --------------------------------------


def rendered_options(question: LayaQuestion) -> list[str]:
    """Option texts exactly as Laya renders them (``sequence.ts`` ``renderOptions``)."""
    if question.type == "choice":
        criteria = question.criteria
        if isinstance(criteria, Mapping):
            return [f"{k}: {v}" if v else str(k) for k, v in criteria.items()]
        return [str(k) for k in criteria or []]
    if question.type == "score":
        return [f"level {i}: {c}" for i, c in enumerate(question.criteria or [])]
    return list(_NOUL_OPTIONS)


def instruction_text(question: LayaQuestion) -> str:
    """The instruction part of the head, as encoded by ``buildSequence``."""
    instructions = question.instructions
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False, default=str)
    return _scrub(f"{question.type} question: {instructions}")


def option_texts(question: LayaQuestion) -> list[str]:
    """Option texts as encoded (each is prefixed with a space before tokenisation)."""
    return [" " + _scrub(text) for text in rendered_options(question)]


@dataclass(frozen=True)
class HeadMeasure:
    """Measured head of one question: instruction tokens and per-option tokens."""

    instruction_tokens: int
    option_tokens: tuple[int, ...]

    @property
    def options_total(self) -> int:
        # one [MASK] marker per option plus its text
        return sum(1 + n for n in self.option_tokens)

    @property
    def total(self) -> int:
        return self.instruction_tokens + self.options_total

    def truncation(self) -> str | None:
        """Why Laya would silently squeeze this head, or ``None`` when it fits intact."""
        if any(n > LAYA_MAX_OPTION_TOKENS for n in self.option_tokens):
            return f"an option exceeds {LAYA_MAX_OPTION_TOKENS} tokens"
        room = LAYA_HEAD_MAX_LEN - self.options_total
        if room < MIN_INSTRUCTION_TOKENS:
            return (
                f"options take {self.options_total} of {LAYA_HEAD_MAX_LEN} head tokens; "
                f"fewer than {MIN_INSTRUCTION_TOKENS} left for the instructions"
            )
        if self.instruction_tokens > room:
            return (
                f"instructions of {self.instruction_tokens} tokens exceed the {room} tokens "
                f"left in head_max_len={LAYA_HEAD_MAX_LEN}"
            )
        return None


async def measure_heads(
    questions: Mapping[str, LayaQuestion], counter: TokenCounter
) -> dict[str, HeadMeasure]:
    """Measure every question head in one tokenizer round trip and reject heads Laya would
    truncate (``ValueError`` names the question and the reason)."""
    texts: list[str] = []
    layout: list[tuple[str, int]] = []
    for key, question in questions.items():
        if not isinstance(question, LayaQuestion):
            question = LayaQuestion.model_validate(question)
        options = option_texts(question)
        texts.append(instruction_text(question))
        texts.extend(options)
        layout.append((key, len(options)))
    counts = await counter(texts)
    if len(counts) != len(texts):
        raise ValueError("token counter returned the wrong number of counts")
    measured: dict[str, HeadMeasure] = {}
    cursor = 0
    for key, n_options in layout:
        head = HeadMeasure(counts[cursor], tuple(counts[cursor + 1 : cursor + 1 + n_options]))
        cursor += 1 + n_options
        problem = head.truncation()
        if problem is not None:
            raise ValueError(f"question {key!r}: {problem}")
        measured[key] = head
    return measured


def state_budget_for(heads: Mapping[str, HeadMeasure], max_len: int = LAYA_MAX_LEN) -> int:
    """State token budget for a batch: ``max_len`` minus the frame and the largest measured head
    (every question's sequence carries the whole state, so the largest head binds)."""
    largest = max((h.total for h in heads.values()), default=0)
    room = max_len - SEQUENCE_SPECIALS - largest
    return max(MIN_STATE_BUDGET_TOKENS, room)


# ---- compaction ---------------------------------------------------------------------------


def _shrink(value: Any, max_str: int, max_list: int) -> Any:
    if isinstance(value, str):
        if len(value) > max_str:
            return value[:max_str].rstrip() + ELLIPSIS
        return value
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _shrink(v, max_str, max_list) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value) if not isinstance(value, (set, frozenset)) else sorted(value, key=str)
        return [_shrink(v, max_str, max_list) for v in items[:max_list]]
    return str(value)


def _ordered_keys(state: Mapping[str, Any], priority: Sequence[str] | None) -> list[str]:
    prio = list(priority) if priority is not None else list(DEFAULT_STATE_PRIORITY)
    seen: set[str] = set()
    ordered: list[str] = []
    for key in prio:
        if key in state and key not in seen:
            ordered.append(key)
            seen.add(key)
    ordered.extend(sorted((str(k) for k in state if k not in seen), key=str))
    return ordered


async def _count_one(counter: TokenCounter, obj: Any) -> int:
    counts = await counter([_scrub(laya_json(obj))])
    if len(counts) != 1:
        raise ValueError("token counter returned the wrong number of counts")
    return counts[0]


async def _compact(
    state: Mapping[str, Any],
    budget_tokens: int,
    counter: TokenCounter,
    priority: Sequence[str] | None,
) -> tuple[dict[str, Any], list[str]]:
    budget = max(1, int(budget_tokens))
    ordered = _ordered_keys(state, priority)
    compacted: dict[str, Any] = {
        key: _shrink(state[key], MAX_STRING_CHARS, MAX_LIST_ITEMS) for key in ordered
    }
    dropped: list[str] = []
    if not compacted:
        return compacted, dropped
    total = await _count_one(counter, compacted)
    if total <= budget:
        return compacted, dropped

    # Over budget: measure every key on its own (one round trip), drop from the tail until the
    # per-key sum fits, then confirm with a measurement of the assembled state.
    per_key = await counter([_scrub(laya_json({k: v})) for k, v in compacted.items()])
    if len(per_key) != len(compacted):
        raise ValueError("token counter returned the wrong number of counts")
    sizes = dict(zip(compacted, per_key, strict=True))
    running = sum(per_key)
    while len(compacted) > 1 and running > budget:
        key = next(reversed(compacted))
        del compacted[key]
        dropped.append(key)
        running -= sizes[key]
    total = await _count_one(counter, compacted)
    while len(compacted) > 1 and total > budget:
        key = next(reversed(compacted))
        del compacted[key]
        dropped.append(key)
        total = await _count_one(counter, compacted)
    if total > budget:
        # One oversized key left: tighten it rather than sending an empty state.
        key = next(iter(compacted))
        compacted[key] = _shrink(compacted[key], max(32, budget * 2), 4)
    return compacted, dropped


async def compact_state(
    state: Mapping[str, Any] | str,
    budget_tokens: int,
    counter: TokenCounter,
    priority: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Deterministically shrink ``state`` under ``budget_tokens`` as measured by ``counter``.

    Keys are kept in ``priority`` order (then alphabetically); long strings are cut to
    ``MAX_STRING_CHARS`` with an ellipsis, lists to ``MAX_LIST_ITEMS`` items, and the lowest
    priority keys are dropped until the measured state fits. Returns the compact state and the
    dropped keys, in drop order. Counter failures propagate (the caller reports them as a Laya
    failure); malformed input degrades to a best-effort copy.
    """
    if isinstance(state, str):
        return await _compact({"text": state}, budget_tokens, counter, priority)
    if not isinstance(state, Mapping):
        return await _compact({"value": state}, budget_tokens, counter, priority)
    return await _compact(state, budget_tokens, counter, priority)


# ---- question validation ------------------------------------------------------------------


def normalise_option(key: str) -> str:
    return " ".join(str(key).strip().lower().replace("-", "_").split())


def validate_questions(questions: Mapping[str, LayaQuestion]) -> None:
    """Structural checks that need no tokenizer: reject question batches Laya would refuse.

    Raises ``ValueError`` when a choice has ``LAYA_MAX_CHOICE_OPTIONS`` (20) or more options,
    when option keys collide after normalisation, when a score has fewer than two levels or when
    instructions are empty. Token-length checks are ``measure_heads`` (measured, per call).
    """
    if not questions:
        raise ValueError("at least one Laya question is required")
    for key, question in questions.items():
        if not isinstance(question, LayaQuestion):
            question = LayaQuestion.model_validate(question)
        if question.type == "choice":
            criteria = question.criteria or {}
            option_keys = [str(k) for k in criteria]
            if not option_keys:
                raise ValueError(f"question {key!r}: a choice needs at least one option")
            if len(option_keys) >= LAYA_MAX_CHOICE_OPTIONS:
                raise ValueError(
                    f"question {key!r}: {len(option_keys)} options; Laya needs fewer than "
                    f"{LAYA_MAX_CHOICE_OPTIONS} per choice"
                )
            normalised = [normalise_option(k) for k in option_keys]
            if any(not n for n in normalised):
                raise ValueError(f"question {key!r}: empty option key")
            if len(set(normalised)) != len(normalised):
                raise ValueError(
                    f"question {key!r}: option keys are not distinct after normalisation"
                )
        elif question.type == "score":
            levels = question.criteria or []
            if len(levels) < 2:
                raise ValueError(f"question {key!r}: a score needs at least two ordered levels")
            if len(levels) >= LAYA_MAX_CHOICE_OPTIONS:
                raise ValueError(
                    f"question {key!r}: {len(levels)} levels; Laya needs fewer than "
                    f"{LAYA_MAX_CHOICE_OPTIONS}"
                )
        if not str(question.instructions).strip():
            raise ValueError(f"question {key!r}: instructions are empty")
