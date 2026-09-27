"""``MockSpark``: a ``SparkClient`` that streams a canned sectioned answer with no model.

Used by tests, demos and ``spark_mode = "mock"``. It mimics the real client's observable
behaviour: one request at a time, ``spark.loading`` on the first run of each profile, cancel
checks between chunks, ``DEEP_PROFILE_UNAVAILABLE`` when Deep is disabled, and measured stats.
The default text fills placeholders from the user message (instrument line, horizon plan and
cited source ids as rendered by ``prompt.build_messages``); pass ``text_factory`` to override.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable
from typing import Any

from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.schemas.capabilities import ProfileCapability
from bayanalytics.schemas.common import ErrorCode, Profile
from bayanalytics.spark.base import (
    ProfileSpec,
    SparkGeneration,
    SparkMessage,
    SparkRunOptions,
    SparkStreamStats,
    TokenCallback,
)
from bayanalytics.spark.bundle import estimate_tokens
from bayanalytics.spark.profiles import profile_specs
from bayanalytics.spark.prompt import HORIZON_HEADING_PREFIX, SECTION_HEADINGS

TextFactory = Callable[[list[SparkMessage]], str]

_INSTRUMENT_RE = re.compile(r"^Instrument:\s*(.+?)(?:\s*\|.*)?$", re.MULTILINE)
_HORIZON_PLAN_RE = re.compile(
    rf"^##\s*{re.escape(HORIZON_HEADING_PREFIX)}(\w+)\s*\nStance:\s*(\w+)", re.MULTILINE
)
_SOURCE_ID_RE = re.compile(r"\[(src_[A-Za-z0-9]+)\]")

_SECTION_TEXT: dict[str, str] = {
    "Summary": (
        "Evidence suggests {instrument} is in a period where current signals are mixed: the "
        "deterministic metrics in the bundle point one way while the most recent excerpts "
        "qualify that view {cite}."
    ),
    "What changed": (
        "- The most recent reporting period shows a change the evidence records as material {cite}."
    ),
    "Fundamentals": (
        "Reported figures are quoted exactly as provided in the calculated metrics; nothing "
        "was recomputed {cite}."
    ),
    "Valuation": (
        "Valuation context is limited to the calculated ratios in the bundle and their period "
        "labels {cite}."
    ),
    "Benchmark context": (
        "Relative performance is read from the benchmark context supplied by the deterministic "
        "layer {cite}."
    ),
    "Historical context": (
        "Historically similar periods in the listed analogues offer only a partial precedent "
        "{cite}."
    ),
    "Market context": (
        "Media and market commentary in the excerpts is treated as secondary evidence {cite}."
    ),
    "Bull evidence": (
        "- The strongest supporting evidence is the primary-source excerpt cited here {cite}."
    ),
    "Bear evidence": (
        "- The strongest contradictory evidence is the qualifying statement cited here {cite}."
    ),
    "Risks": (
        "- Freshness risk: part of the evidence may be stale relative to the as-of date {cite}."
    ),
    "Conflicts": "- No unresolved source conflicts were listed in the evidence.",
    "Uncertainties": (
        "- Mock synthesis: this text is a placeholder produced without a language model."
    ),
    "Follow-up questions": ("- Which upcoming disclosure would most change the current picture?"),
}


def default_text(messages: list[SparkMessage]) -> str:
    """Canned answer with every heading, filled from the user message when possible."""
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
        verb = "supports" if stance == "bullish" else "qualifies"
        bullet = (
            f"- Over the {horizon.replace('_', ' ')} horizon the evidence in the bundle "
            f"{verb} this stance {cite}."
        )
        blocks.append(
            f"## {HORIZON_HEADING_PREFIX}{horizon}\nStance: {stance}\n{bullet.replace(' .', '.')}"
        )
    return "\n\n".join(blocks) + "\n"


class MockSpark:
    """Implements ``bayanalytics.spark.base.SparkClient`` without a model."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        deep_available: bool = True,
        delay_s: float = 0.0,
        text_factory: TextFactory | None = None,
        chunk_chars: int = 24,
    ) -> None:
        self._settings = settings or Settings()
        self._specs = profile_specs(self._settings)
        self.deep_available = deep_available
        self.delay_s = delay_s
        self._text_factory: TextFactory = text_factory or default_text
        self._chunk_chars = max(1, chunk_chars)
        self._lock = asyncio.Lock()
        self._loaded: set[Profile] = set()
        self.runs: list[dict[str, Any]] = []

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        self._loaded.clear()

    def availability(self, profile: Profile) -> ProfileCapability:
        spec = self._specs[profile]
        if profile == "deep" and not self.deep_available:
            return ProfileCapability(
                available=False,
                context_ceiling=spec.context_ceiling,
                reason="the deep profile is disabled in this mock configuration",
                code=ErrorCode.DEEP_PROFILE_UNAVAILABLE,
            )
        return ProfileCapability(available=True, context_ceiling=spec.context_ceiling)

    def profile_spec(self, profile: Profile) -> ProfileSpec:
        return self._specs[profile]

    async def run(
        self,
        profile: Profile,
        messages: list[SparkMessage],
        on_token: TokenCallback,
        ctx: AnalysisContext,
        options: SparkRunOptions | None = None,
    ) -> SparkGeneration:
        opts = options or SparkRunOptions(
            max_tokens=self._settings.spark_max_output_tokens,
            temperature=self._settings.spark_temperature,
        )
        ctx.check_cancelled()
        async with self._lock:
            ctx.check_cancelled()
            if profile == "deep" and not self.deep_available:
                raise AnalysisError(
                    ErrorCode.DEEP_PROFILE_UNAVAILABLE,
                    details={"reason": "deep profile disabled in mock"},
                )
            spec = self._specs[profile]
            load_ms: float | None = None
            if profile not in self._loaded:
                await ctx.event(
                    "spark.loading",
                    profile=profile,
                    context_ceiling=spec.context_ceiling,
                    kv_cache_type=spec.kv_cache_type,
                )
                self._loaded.add(profile)
                load_ms = 0.0
            text = self._text_factory(messages)
            record: dict[str, Any] = {
                "profile": profile,
                "messages": messages,
                "options": opts,
                "text": text,
                "completed": False,
            }
            self.runs.append(record)

            started = time.perf_counter()
            ctx.timers.start("spark")
            ttft_ms: float | None = None
            emitted = 0
            try:
                for chunk in _chunks(text, self._chunk_chars):
                    if ctx.cancel.cancelled:
                        raise AnalysisError(ErrorCode.CANCELLED)
                    if self.delay_s > 0:
                        await asyncio.sleep(self.delay_s)
                    else:
                        await asyncio.sleep(0)
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000.0
                    emitted += 1
                    await on_token(chunk)
                if ctx.cancel.cancelled:
                    raise AnalysisError(ErrorCode.CANCELLED)
            finally:
                total_ms = (time.perf_counter() - started) * 1000.0
                ctx.timers.stop("spark")
                record["chunks_emitted"] = emitted

            record["completed"] = True
            output_tokens = estimate_tokens(text)
            prompt_tokens = sum(estimate_tokens(m.content) for m in messages)
            if ttft_ms is not None:
                ctx.diagnostics["spark_ttft_ms"] = ttft_ms
            # Without a real delay the wall clock says nothing about throughput: report None.
            tps = output_tokens / (total_ms / 1000.0) if self.delay_s > 0 and total_ms else None
            stats = SparkStreamStats(
                profile=profile,
                context_ceiling=spec.context_ceiling,
                kv_cache_type=spec.kv_cache_type,
                load_ms=load_ms,
                time_to_first_token_ms=ttft_ms,
                total_ms=total_ms,
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                tokens_per_second=tps,
                resident_rss_mb=None,
                peak_rss_mb=None,
                finish_reason="stop",
                runtime_version="mock",
            )
            return SparkGeneration(text=text, stats=stats, truncated=False)


def _chunks(text: str, size: int) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]
