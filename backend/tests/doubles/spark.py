"""``ScriptedSpark``: a ``SparkClient`` test double that streams a scripted, sectioned text.

No model, no llama-server. It mirrors the real client's observable behaviour: one session at a
time (the lock is held for the whole session), ``spark.loading`` on the first session of each
profile, cancel checks between chunks that raise ``ctx.cancel.reason`` (so a shutdown shows
INTERRUPTED, a user cancel CANCELLED), ``DEEP_PROFILE_UNAVAILABLE`` when Deep is disabled, and
one generation per session.

Every figure in its stats is measured or ``None``:

* ``prompt_tokens`` is the count ``count_prompt_tokens`` measures with the doubles' tokenizer
  (``doubles.tokens``, one token per word or punctuation mark over each message's content),
  the same measurement the bundle fit used;
* ``output_tokens`` is ``None``: no model produced the text, so there is no model count;
* ``tokens_per_second`` is measured (doubles' tokens over wall-clock) only when a real
  per-chunk delay was configured, otherwise ``None``;
* ``load_ms``, ``resident_rss_mb``, ``peak_rss_mb`` and ``runtime_version`` are ``None``.

The scripted text keeps every answer section so the parser and assembly paths are exercised.
Its lines are labelled ``[scripted]`` and cite ids found in the user message; it never claims to
be an assessment. Pass ``text_factory`` to script something else.

Structured generations (options carrying ``json_schema``: Spark pass 1, query understanding)
return a scripted interpretation instead. **The double does not understand language**: it looks
the question (the ``Question:`` line of the pass-1 prompt) up in the test-supplied
``interpretations`` mapping, or passes it to the test-supplied callback, and otherwise returns
the broad interpretation (a general assessment with no requirements). Tests built on it prove
that the pipeline turns an interpretation into requirements, plans and checks, not that a model
interprets questions well. A dict is serialised as JSON; a string is returned verbatim (for
malformed-output tests). Structured generations are recorded in ``understandings`` and their
sessions are not listed in ``sessions`` (``runs`` and ``sessions`` stay the synthesis record);
like the real client, they add nothing to the synthesis stage timer or its diagnostics.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from collections.abc import AsyncIterator, Callable, Mapping
from typing import Any

from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.schemas.capabilities import ProfileCapability
from bayanalytics.schemas.common import ErrorCode, Profile
from bayanalytics.schemas.questions import QueryUnderstanding
from bayanalytics.spark.base import (
    ProfileSpec,
    SparkGeneration,
    SparkMessage,
    SparkRunOptions,
    SparkStreamStats,
    TokenCallback,
)
from bayanalytics.spark.profiles import profile_specs
from bayanalytics.spark.prompt import HORIZON_HEADING_PREFIX, SECTION_HEADINGS

from .tokens import count_tokens

TextFactory = Callable[[list[SparkMessage]], str]
Interpretation = Mapping[str, Any] | str
InterpretationScript = Mapping[str, Interpretation] | Callable[[str], Interpretation | None]

BROAD_INTERPRETATION: dict[str, Any] = QueryUnderstanding.broad().model_dump()
_QUESTION_RE = re.compile(r"^Question:\s*(.*)$", re.MULTILINE)

_INSTRUMENT_RE = re.compile(r"^Instrument:\s*(.+?)(?:\s*\|.*)?$", re.MULTILINE)
_HORIZON_PLAN_RE = re.compile(
    rf"^##\s*{re.escape(HORIZON_HEADING_PREFIX)}(\w+)\s*\nStance:\s*(\w+)", re.MULTILINE
)
_SOURCE_ID_RE = re.compile(r"\[(src_[A-Za-z0-9]+)\]")

_SECTION_TEXT: dict[str, str] = {
    "Summary": (
        "[scripted] Scripted synthesis for {instrument}: the sections below exercise the "
        "answer format with ids cited from the bundle and are not an assessment {cite}."
    ),
    "What changed": (
        "- [scripted] The bundle lists its most recent reporting period under important "
        "events {cite}."
    ),
    "Fundamentals": (
        "[scripted] Figures are quoted exactly as provided in the calculated metrics; nothing "
        "was recomputed {cite}."
    ),
    "Valuation": (
        "[scripted] Valuation lines repeat the calculated ratios in the bundle and their "
        "period labels {cite}."
    ),
    "Benchmark context": (
        "[scripted] Benchmark lines repeat the benchmark context supplied by the "
        "deterministic layer {cite}."
    ),
    "Historical context": (
        "[scripted] Historical lines repeat the listed analogues in the bundle {cite}."
    ),
    "Market context": (
        "[scripted] Market lines repeat the secondary excerpts in the bundle {cite}."
    ),
    "Bull evidence": (
        "- [scripted] A bullet citing the first primary-source excerpt in the bundle {cite}."
    ),
    "Bear evidence": ("- [scripted] A bullet citing a qualifying statement in the bundle {cite}."),
    "Risks": ("- [scripted] A bullet repeating the freshness note from the bundle {cite}."),
    "Conflicts": "- [scripted] conflicts were not assessed; see the structured conflicts list.",
    "Uncertainties": (
        "- [scripted] This text was produced by a test double without a language model; "
        "it is not an assessment."
    ),
    "Follow-up questions": (
        "- [scripted] Which upcoming disclosure would most change the current picture?"
    ),
}


def scripted_text(messages: list[SparkMessage]) -> str:
    """The default script: every heading, filled from the user message when possible."""
    user = next((m.content for m in reversed(messages) if m.role == "user"), "")
    match = _INSTRUMENT_RE.search(user)
    instrument = match.group(1).strip() if match else "the company"
    ids: list[str] = []
    for source_id in _SOURCE_ID_RE.findall(user):
        if source_id not in ids:
            ids.append(source_id)
    cite = f"[{ids[0]}]" if ids else ""
    cite_alt = f"[{ids[1]}]" if len(ids) > 1 else cite
    blocks: list[str] = []
    for index, heading in enumerate(SECTION_HEADINGS):
        body = _SECTION_TEXT[heading].format(
            instrument=instrument, cite=cite_alt if index % 2 else cite
        )
        blocks.append(f"## {heading}\n{body.replace(' .', '.')}")
    for horizon, stance in _HORIZON_PLAN_RE.findall(user):
        bullet = (
            f"- [scripted] Over the {horizon.replace('_', ' ')} horizon this bullet cites "
            f"evidence in the bundle for the stated stance {cite}."
        )
        blocks.append(
            f"## {HORIZON_HEADING_PREFIX}{horizon}\nStance: {stance}\n{bullet.replace(' .', '.')}"
        )
    return "\n\n".join(blocks) + "\n"


class ScriptedSpark:
    """Implements ``bayanalytics.spark.base.SparkClient`` without a model."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        deep_available: bool = True,
        delay_s: float = 0.0,
        text_factory: TextFactory | None = None,
        chunk_chars: int = 24,
        interpretations: InterpretationScript | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._specs = profile_specs(self._settings)
        self.deep_available = deep_available
        self.delay_s = delay_s
        self._text_factory: TextFactory = text_factory or scripted_text
        self._chunk_chars = max(1, chunk_chars)
        self._lock = asyncio.Lock()
        self._waiting = 0  # sessions queued for the lane (as in the real client)
        self._loaded: set[Profile] = set()
        self._interpretations = interpretations
        self.runs: list[dict[str, Any]] = []
        self.sessions: list[dict[str, Any]] = []
        self.understandings: list[dict[str, Any]] = []  # pass-1 (json_schema) generations

    # --- SparkClient protocol -----------------------------------------------------------

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        self._loaded.clear()

    @property
    def busy(self) -> bool:
        return self._lock.locked() or self._waiting > 0

    @contextlib.asynccontextmanager
    async def _turn(self) -> AsyncIterator[None]:
        self._waiting += 1
        try:
            await self._lock.acquire()
        finally:
            self._waiting -= 1
        try:
            yield
        finally:
            self._lock.release()

    def availability(self, profile: Profile) -> ProfileCapability:
        spec = self._specs[profile]
        if profile == "deep" and not self.deep_available:
            return ProfileCapability(
                available=False,
                context_ceiling=spec.context_ceiling,
                reason="the deep profile is disabled in this scripted double",
                code=ErrorCode.DEEP_PROFILE_UNAVAILABLE,
            )
        return ProfileCapability(available=True, context_ceiling=spec.context_ceiling)

    def profile_spec(self, profile: Profile) -> ProfileSpec:
        return self._specs[profile]

    @contextlib.asynccontextmanager
    async def session(
        self, profile: Profile, ctx: AnalysisContext
    ) -> AsyncIterator[ScriptedSession]:
        """Exclusive turn with ``profile``: the lock is held until the block ends."""
        ctx.check_cancelled()
        async with self._turn():
            ctx.check_cancelled()
            if profile == "deep" and not self.deep_available:
                raise AnalysisError(
                    ErrorCode.DEEP_PROFILE_UNAVAILABLE,
                    details={"reason": "deep profile disabled in the scripted double"},
                )
            spec = self._specs[profile]
            loaded_now = profile not in self._loaded
            if loaded_now:
                await ctx.event(
                    "spark.loading",
                    profile=profile,
                    context_ceiling=spec.context_ceiling,
                    kv_cache_type=spec.kv_cache_type,
                )
                self._loaded.add(profile)
            ctx.diagnostics["spark_loaded_now"] = loaded_now
            session = ScriptedSession(self, profile, spec, ctx)
            self.sessions.append(session.record)
            yield session

    async def run(
        self,
        profile: Profile,
        messages: list[SparkMessage],
        on_token: TokenCallback,
        ctx: AnalysisContext,
        options: SparkRunOptions | None = None,
    ) -> SparkGeneration:
        async with self.session(profile, ctx) as session:
            return await session.generate(messages, on_token, options)

    # --- internals ----------------------------------------------------------------------

    def interpretation_text(self, messages: list[SparkMessage]) -> str:
        """The scripted pass-1 output: a lookup of the question text, never an understanding."""
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        match = _QUESTION_RE.search(user)
        question = match.group(1).strip() if match else ""
        script = self._interpretations
        scripted: Interpretation | None = None
        if callable(script):
            scripted = script(question)
        elif script is not None:
            scripted = script.get(question)
        if scripted is None:
            scripted = BROAD_INTERPRETATION
        return scripted if isinstance(scripted, str) else json.dumps(dict(scripted))

    def _default_options(self, options: SparkRunOptions | None) -> SparkRunOptions:
        return options or SparkRunOptions(
            max_tokens=self._settings.spark_max_output_tokens,
            temperature=self._settings.spark_temperature,
        )

    async def _generate(
        self,
        profile: Profile,
        spec: ProfileSpec,
        messages: list[SparkMessage],
        on_token: TokenCallback,
        ctx: AnalysisContext,
        opts: SparkRunOptions,
    ) -> SparkGeneration:
        structured = opts.json_schema is not None
        text = self.interpretation_text(messages) if structured else self._text_factory(messages)
        record: dict[str, Any] = {
            "profile": profile,
            "messages": messages,
            "options": opts,
            "text": text,
            "completed": False,
        }
        (self.understandings if structured else self.runs).append(record)
        prompt_tokens = prompt_token_count(messages)

        started = time.perf_counter()
        if not structured:
            ctx.timers.start("spark")
        ttft_ms: float | None = None
        emitted = 0
        try:
            for chunk in _chunks(text, self._chunk_chars):
                ctx.check_cancelled()
                if self.delay_s > 0:
                    await asyncio.sleep(self.delay_s)
                else:
                    await asyncio.sleep(0)
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1000.0
                emitted += 1
                await on_token(chunk)
            ctx.check_cancelled()
        finally:
            total_ms = (time.perf_counter() - started) * 1000.0
            if not structured:
                ctx.timers.stop("spark")
            record["chunks_emitted"] = emitted

        record["completed"] = True
        if ttft_ms is not None and not structured:
            ctx.diagnostics["spark_ttft_ms"] = ttft_ms
        # Throughput is a wall-clock measurement only when the stream really took time;
        # without a configured delay the clock says nothing about a model, so None.
        tps: float | None = None
        if self.delay_s > 0 and total_ms > 0:
            tps = count_tokens(text) / (total_ms / 1000.0)
        stats = SparkStreamStats(
            profile=profile,
            context_ceiling=spec.context_ceiling,
            kv_cache_type=spec.kv_cache_type,
            load_ms=None,
            time_to_first_token_ms=ttft_ms,
            total_ms=total_ms,
            prompt_tokens=prompt_tokens,
            output_tokens=None,
            tokens_per_second=tps,
            resident_rss_mb=None,
            peak_rss_mb=None,
            finish_reason="stop",
            runtime_version=None,
        )
        return SparkGeneration(text=text, stats=stats, truncated=False)


class ScriptedSession:
    """``SparkSession`` for ``ScriptedSpark`` (valid only inside ``session``)."""

    def __init__(
        self, client: ScriptedSpark, profile: Profile, spec: ProfileSpec, ctx: AnalysisContext
    ) -> None:
        self._client = client
        self._profile = profile
        self._spec = spec
        self._ctx = ctx
        self.generated = False
        self.record: dict[str, Any] = {"profile": profile, "measurements": []}

    @property
    def spec(self) -> ProfileSpec:
        return self._spec

    async def count_prompt_tokens(self, messages: list[SparkMessage]) -> int:
        self._ctx.check_cancelled()
        count = prompt_token_count(messages)
        self.record["measurements"].append(count)
        return count

    async def generate(
        self,
        messages: list[SparkMessage],
        on_token: TokenCallback,
        options: SparkRunOptions | None = None,
    ) -> SparkGeneration:
        if self.generated:
            raise AnalysisError(
                ErrorCode.INTERNAL_ERROR, details={"reason": "one generation per spark session"}
            )
        self.generated = True
        opts = self._client._default_options(options)
        if opts.json_schema is not None and self.record in self._client.sessions:
            # A pass-1 session: keep ``sessions`` the record of synthesis sessions.
            self._client.sessions.remove(self.record)
            self.record["structured"] = True
        self._ctx.check_cancelled()
        return await self._client._generate(
            self._profile, self._spec, messages, on_token, self._ctx, opts
        )


def prompt_token_count(messages: list[SparkMessage]) -> int:
    """The doubles' measurement of a prompt: tokens over each message's content."""
    return sum(count_tokens(m.content) for m in messages)


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]
