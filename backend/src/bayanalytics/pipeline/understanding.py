"""Spark pass 1: a short, structured interpretation of the analyst's question.

Runs after the instrument (and any explicit horizon) is resolved and before research. Spark is
asked to convert the question into a ``QueryUnderstanding`` (an intent, the requirements an
answer needs, a comparison focus and three flags) and nothing else: no evidence, no numbers, no
answer. The generation is constrained to the model's JSON schema (llama-server compiles
``response_format`` ``json_schema`` into a grammar) and the output is validated strictly.

Pass 1 uses the same Spark runtime and profile as the final synthesis (pass 2) through its own
short session: the Spark lock is held only while this generation runs and is released before
research starts. It is internal: tokens are discarded, and it emits no ``spark.started`` /
``spark.token`` / ``spark.completed`` events (a model load it triggers still emits the
manager's ``spark.loading``, so a client can see ``spark.loading`` before research). Prompts are
never logged or persisted; the raw output goes to DEBUG logs only.

Outcomes:

- a valid interpretation: source ``spark`` (values outside the vocabularies are dropped, with a
  product-level note);
- malformed or empty output, an out-of-vocabulary intent, or a generation cut off by the token
  limit: the broad interpretation (a general assessment, no requirements), source ``fallback``,
  with the uncertainty ``FALLBACK_NOTE``;
- a Spark runtime failure (start, memory, inference, cancel, shutdown) propagates: the final
  synthesis could not run either.

Latency is measured separately (``query_understanding_*`` telemetry and the ``understanding``
stage timer): the wall clock, and within it the wait for the Spark lane, any model load and the
generation itself. It must be measured on the reference hardware before any optimisation.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext
from bayanalytics.instruments.base import InstrumentIdentity
from bayanalytics.schemas.common import HORIZON_LABELS, Profile
from bayanalytics.schemas.questions import (
    COMPARISON_FOCI,
    COMPARISON_FOCUS_DESCRIPTIONS,
    FLAG_DESCRIPTIONS,
    MAX_REQUIREMENTS,
    QUESTION_INTENT_DESCRIPTIONS,
    QUESTION_INTENTS,
    REQUIREMENT_DESCRIPTIONS,
    REQUIREMENT_NAMES,
    InterpretationSource,
    QueryUnderstanding,
    query_understanding_schema,
)
from bayanalytics.spark.base import SparkClient, SparkMessage, SparkRunOptions
from bayanalytics.spark.prompt import clean_text

log = logging.getLogger(__name__)

TIMER_NAME = "understanding"
QUESTION_MAX_CHARS = 300
NAME_MAX_CHARS = 120

FALLBACK_NOTE = "the question could not be interpreted; a general assessment was produced"
OUT_OF_VOCABULARY_NOTE = (
    "part of the question's interpretation was outside the supported values and was ignored"
)
_FLAGS: tuple[str, ...] = tuple(FLAG_DESCRIPTIONS)


def _system_prompt() -> str:
    lines = [
        "Convert the analyst's question about a listed company into the JSON object "
        "described. Do not answer the question. No prose. Use only the listed values. A broad "
        "request such as 'Assess Apple' is general_assessment with no requirements.",
        "",
        "intent (exactly one):",
        *(f"- {name}: {QUESTION_INTENT_DESCRIPTIONS[name]}" for name in QUESTION_INTENTS),
        "",
        f"requirements (only what an answer needs, at most {MAX_REQUIREMENTS}, may be empty):",
        *(f"- {name}: {REQUIREMENT_DESCRIPTIONS[name]}" for name in REQUIREMENT_NAMES),
        "",
        "comparison_focus (exactly one):",
        *(f"- {name}: {COMPARISON_FOCUS_DESCRIPTIONS[name]}" for name in COMPARISON_FOCI),
        "",
        *(f"{flag}: {text}." for flag, text in FLAG_DESCRIPTIONS.items()),
    ]
    return "\n".join(lines)


SYSTEM_PROMPT = _system_prompt()
RESPONSE_SCHEMA: dict[str, Any] = query_understanding_schema()


class UnderstandingStats(BaseModel):
    """Measured only; ``None`` when the runtime did not report a figure."""

    wall_ms: float  # wait + load + generation
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    load_ms: float | None = None  # only when this pass loaded the profile
    wait_ms: float | None = None  # queued behind another analysis's Spark turn
    generation_ms: float | None = None  # the pass itself: prompt processing and decoding


class Understanding(BaseModel):
    understanding: QueryUnderstanding
    source: InterpretationSource
    notes: list[str] = Field(default_factory=list)  # product-level sentences (uncertainties)
    stats: UnderstandingStats


def understanding_messages(
    query: str, identity: InstrumentIdentity, resolved_horizon: str
) -> list[SparkMessage]:
    """The pass-1 prompt: the rules above plus the question, the instrument and the horizon.
    No evidence and no calculations."""
    company = clean_text(identity.name, NAME_MAX_CHARS)
    symbol = clean_text(identity.symbol, 16)
    user = "\n".join(
        [
            f"Question: {clean_text(query, QUESTION_MAX_CHARS)}",
            f"Company: {company} ({symbol})" if company else f"Company: {symbol}",
            f"Horizon: {HORIZON_LABELS.get(resolved_horizon, clean_text(resolved_horizon, 40))}",
        ]
    )
    return [
        SparkMessage(role="system", content=SYSTEM_PROMPT),
        SparkMessage(role="user", content=user),
    ]


def understanding_options(settings: Settings) -> SparkRunOptions:
    return SparkRunOptions(
        max_tokens=settings.spark_understanding_max_tokens,
        temperature=0.0,
        json_schema=RESPONSE_SCHEMA,
    )


def parse_understanding(
    text: str, *, truncated: bool = False
) -> tuple[QueryUnderstanding | None, list[str], str | None]:
    """``(interpretation, notes, fallback_reason)``; the interpretation is ``None`` when the
    output cannot be used at all (then ``fallback_reason`` is a keyword, never model text)."""
    if truncated:
        return None, [], "output_token_limit"
    raw = (text or "").strip()
    if not raw:
        return None, [], "empty_output"
    try:
        data = json.loads(raw)
    except ValueError:
        return None, [], "malformed_json"
    if not isinstance(data, dict):
        return None, [], "not_an_object"
    try:
        return QueryUnderstanding.model_validate(data), [], None
    except ValidationError:
        pass
    # Strict validation failed: keep what is valid, drop the rest (with one note), and give
    # up only when the intent itself is unusable.
    intent = data.get("intent")
    if not isinstance(intent, str) or intent not in QUESTION_INTENTS:
        return None, [], "invalid_intent"
    dropped = False
    requirements: list[str] = []
    raw_requirements = data.get("requirements")
    if isinstance(raw_requirements, list):
        for value in raw_requirements:
            if isinstance(value, str) and value in REQUIREMENT_NAMES:
                if value not in requirements:
                    requirements.append(value)
            else:
                dropped = True
        if len(requirements) > MAX_REQUIREMENTS:
            requirements = requirements[:MAX_REQUIREMENTS]
            dropped = True
    else:
        dropped = True
    comparison = data.get("comparison_focus")
    if not isinstance(comparison, str) or comparison not in COMPARISON_FOCI:
        comparison = "none"
        dropped = True
    flags: dict[str, bool] = {}
    for flag in _FLAGS:
        value = data.get(flag)
        if isinstance(value, bool):
            flags[flag] = value
        else:
            flags[flag] = False
            dropped = True
    if set(data) - {"intent", "requirements", "comparison_focus", *_FLAGS}:
        dropped = True
    understanding = QueryUnderstanding.model_validate(
        {
            "intent": intent,
            "requirements": requirements,
            "comparison_focus": comparison,
            **flags,
        }
    )
    return understanding, [OUT_OF_VOCABULARY_NOTE] if dropped else [], None


async def _discard(_text: str) -> None:
    """Pass 1 is internal: nothing it generates is streamed to the client."""
    return None


async def understand_question(
    query: str,
    identity: InstrumentIdentity,
    resolved_horizon: str,
    spark: SparkClient,
    profile: Profile,
    ctx: AnalysisContext,
    settings: Settings,
) -> Understanding:
    """Run Spark pass 1 and return the validated interpretation (or the broad fallback)."""
    messages = understanding_messages(query, identity, resolved_horizon)
    options = understanding_options(settings)
    started = time.perf_counter()
    with ctx.timers.span(TIMER_NAME):
        async with spark.session(profile, ctx) as session:
            entered = time.perf_counter()
            generation = await session.generate(messages, _discard, options)
            generated = time.perf_counter()
    wall_ms = (time.perf_counter() - started) * 1000.0
    load_ms = generation.stats.load_ms
    # Entering the session is the wait for the lane plus any model load it triggered.
    wait_ms = max(0.0, (entered - started) * 1000.0 - (load_ms or 0.0))
    stats = UnderstandingStats(
        wall_ms=round(wall_ms, 3),
        prompt_tokens=generation.stats.prompt_tokens,
        output_tokens=generation.stats.output_tokens,
        load_ms=load_ms,
        wait_ms=round(wait_ms, 3),
        generation_ms=round((generated - entered) * 1000.0, 3),
    )
    log.debug("query understanding raw output: %r", generation.text)
    truncated = generation.truncated or generation.stats.finish_reason == "length"
    parsed, notes, reason = parse_understanding(generation.text, truncated=truncated)
    if parsed is None:
        log.info("query understanding fell back to a general assessment: %s", reason)
        ctx.diagnostics["query_understanding_fallback"] = reason
        return Understanding(
            understanding=QueryUnderstanding.broad(),
            source="fallback",
            notes=[FALLBACK_NOTE],
            stats=stats,
        )
    log.info(
        "query understood: intent=%s requirements=%d dropped_values=%s wall_ms=%.0f",
        parsed.intent,
        len(parsed.requirements),
        bool(notes),
        wall_ms,
    )
    return Understanding(understanding=parsed, source="spark", notes=notes, stats=stats)
