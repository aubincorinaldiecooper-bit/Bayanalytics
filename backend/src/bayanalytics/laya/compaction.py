"""State compaction and question validation for Laya (AGENT.md sections 1.2, 3.1 and 30).

Laya's English checkpoint renders every question as::

    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]

and then silently truncates the *state* to whatever is left of ``max_len`` (512) tokens; the
option/instruction head is squeezed into ``head_max_len`` (192) tokens, again silently, and only
throws when the options do not fit at all. The spec forbids relying on that silent truncation, so
this module compacts states deterministically *before* the call and rejects heads that would be
truncated.

Every token figure here is an **estimate** (the tokenizer lives in the Node worker); the real
``usage.input_tokens`` comes back from the worker with each answer. The estimate is deliberately
conservative (about 1 token per 3.5 characters, plus one per JSON key) so that "fits by estimate"
implies "fits for the real tokenizer" for English states.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from bayanalytics.laya.base import LAYA_HEAD_MAX_LEN, LAYA_MAX_CHOICE_OPTIONS, LAYA_MAX_LEN
from bayanalytics.schemas.decisions import LayaQuestion

CHARS_PER_TOKEN = 3.5

# Header allowance: the fixed frame around the state. "<type> question: " plus the finance
# schemas' instructions (20-45 tokens), the option markers and texts of the compact trend /
# stance questions (20-50 tokens) and the three [SEP]/[CLS] specials fit comfortably in 96
# tokens. Batches with a longer head (``research_intent`` renders ten described options) must
# use ``state_budget_for`` which subtracts the largest estimated head in the batch instead.
HEADER_ALLOWANCE_TOKENS = 96
DEFAULT_STATE_BUDGET_TOKENS = LAYA_MAX_LEN - HEADER_ALLOWANCE_TOKENS  # 416
MIN_STATE_BUDGET_TOKENS = 64

MAX_STRING_CHARS = 240
MAX_LIST_ITEMS = 8
# Laya keeps at most 48 tokens per option text before it starts shrinking heads.
LAYA_MAX_OPTION_TOKENS = 48
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


# ---- token estimates ----------------------------------------------------------------------


def _count_keys(obj: Any) -> int:
    if isinstance(obj, Mapping):
        return len(obj) + sum(_count_keys(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_count_keys(v) for v in obj)
    return 0


def estimate_tokens(text_or_obj: Any) -> int:
    """Conservative token estimate: ceil(chars / 3.5) plus one token per JSON key.

    A heuristic only. Objects are measured as Laya serialises them; the authoritative count is
    the ``usage.input_tokens`` the worker returns.
    """
    if isinstance(text_or_obj, str):
        return math.ceil(len(text_or_obj) / CHARS_PER_TOKEN)
    return math.ceil(len(laya_json(text_or_obj)) / CHARS_PER_TOKEN) + _count_keys(text_or_obj)


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


def estimate_head_tokens(question: LayaQuestion) -> int:
    """Estimated tokens of the question head: instructions plus one marker and text per option."""
    head = estimate_tokens(f"{question.type} question: {question.instructions}")
    options = sum(1 + estimate_tokens(" " + text) for text in rendered_options(question))
    return head + options


def state_budget_for(questions: Mapping[str, LayaQuestion], max_len: int = LAYA_MAX_LEN) -> int:
    """State token budget for a batch: ``max_len`` minus the largest head, capped at the default."""
    largest = max((estimate_head_tokens(q) for q in questions.values()), default=0)
    room = max_len - largest - 4  # [CLS] + three [SEP]
    return max(MIN_STATE_BUDGET_TOKENS, min(DEFAULT_STATE_BUDGET_TOKENS, room))


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


def _compact(
    state: Mapping[str, Any], budget_tokens: int, priority: Sequence[str] | None
) -> tuple[dict[str, Any], list[str]]:
    budget = max(1, int(budget_tokens))
    ordered = _ordered_keys(state, priority)
    compacted: dict[str, Any] = {
        key: _shrink(state[key], MAX_STRING_CHARS, MAX_LIST_ITEMS) for key in ordered
    }
    dropped: list[str] = []
    while len(compacted) > 1 and estimate_tokens(compacted) > budget:
        key = next(reversed(compacted))
        del compacted[key]
        dropped.append(key)
    if compacted and estimate_tokens(compacted) > budget:
        # One oversized key left: tighten it rather than sending an empty state.
        key = next(iter(compacted))
        compacted[key] = _shrink(compacted[key], max(32, budget * 2), 4)
    return compacted, dropped


def compact_state(
    state: Mapping[str, Any] | str,
    budget_tokens: int = DEFAULT_STATE_BUDGET_TOKENS,
    priority: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Deterministically shrink ``state`` under ``budget_tokens`` (estimated).

    Keys are kept in ``priority`` order (then alphabetically); long strings are cut to
    ``MAX_STRING_CHARS`` with an ellipsis, lists to ``MAX_LIST_ITEMS`` items, and the lowest
    priority keys are dropped until the estimate fits. Returns the compact state and the dropped
    keys, in drop order. Never raises: on an unexpected input it returns a best-effort copy.
    """
    try:
        if isinstance(state, str):
            return _compact({"text": state}, budget_tokens, priority)
        if not isinstance(state, Mapping):
            return _compact({"value": state}, budget_tokens, priority)
        return _compact(state, budget_tokens, priority)
    except Exception:  # pragma: no cover - defensive; compaction must never break an analysis
        try:
            return dict(state) if isinstance(state, Mapping) else {"value": str(state)}, []
        except Exception:
            return {}, []


# ---- question validation ------------------------------------------------------------------


def normalise_option(key: str) -> str:
    return " ".join(str(key).strip().lower().replace("-", "_").split())


def validate_questions(questions: Mapping[str, LayaQuestion]) -> None:
    """Reject question batches Laya would truncate or refuse.

    Raises ``ValueError`` when a choice has ``LAYA_MAX_CHOICE_OPTIONS`` (20) or more options, when
    option keys collide after normalisation, when a single option would exceed Laya's 48-token
    cap, or when the estimated head (instructions plus rendered options) exceeds
    ``head_max_len`` (192). Runs before every ``system_one`` and at import time on the finance
    schemas.
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
        for text in rendered_options(question):
            if estimate_tokens(" " + text) > LAYA_MAX_OPTION_TOKENS:
                raise ValueError(
                    f"question {key!r}: option {text[:40]!r} exceeds "
                    f"{LAYA_MAX_OPTION_TOKENS} tokens"
                )
        head = estimate_head_tokens(question)
        if head > LAYA_HEAD_MAX_LEN:
            raise ValueError(
                f"question {key!r}: estimated head of {head} tokens exceeds "
                f"head_max_len={LAYA_HEAD_MAX_LEN}"
            )
