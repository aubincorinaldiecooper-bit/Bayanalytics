"""``LlamaSparkClient``: the llama-server backed ``SparkClient``.

One Spark request at a time (AGENT.md section 38): ``run`` takes an ``asyncio.Lock``, lets the
manager switch the loaded profile if needed, streams ``POST /v1/chat/completions`` and releases
the lock. Cancellation mid-stream closes the response and raises ``CANCELLED``; partial text is
never returned as a completed generation (section 24, model/runtime failure).

Error details never contain prompt or evidence content: they carry a reason keyword, an HTTP
status or an exception class name only.

``start()`` does not warm-load a profile. Section 29 leaves residency to measurement; when the
measured numbers justify it, warm-loading Fast at startup is a one-line ``manager.ensure`` call
here, guarded by a setting that does not exist yet.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx

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
from bayanalytics.spark.manager import LlamaServerManager, SpawnFn
from bayanalytics.spark.profiles import (
    MemoryProbe,
    check_availability,
    profile_specs,
    psutil_probe,
    read_lockfile,
    version_fields,
)

logger = logging.getLogger("bayanalytics.spark")

CONNECT_TIMEOUT_S = 10.0


class LlamaSparkClient:
    """Implements ``bayanalytics.spark.base.SparkClient`` on top of llama-server."""

    def __init__(
        self,
        settings: Settings,
        *,
        manager: LlamaServerManager | None = None,
        http: httpx.AsyncClient | None = None,
        probe: MemoryProbe | None = None,
        spawn: SpawnFn | None = None,
    ) -> None:
        self._settings = settings
        self._probe: MemoryProbe = probe or psutil_probe
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.spark_request_timeout_s, connect=CONNECT_TIMEOUT_S)
        )
        self._manager = manager or LlamaServerManager(
            settings, probe=self._probe, spawn=spawn, http=self._http
        )
        self._lock = asyncio.Lock()
        self._specs = profile_specs(settings)
        self.lockfile = read_lockfile(settings.spark_lockfile)

    # --- SparkClient protocol -----------------------------------------------------------

    @property
    def manager(self) -> LlamaServerManager:
        return self._manager

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    async def start(self) -> None:
        """No warm load (see module docstring). External mode: seed the capability cache."""
        if self._settings.spark_mode == "external":
            try:
                await self._manager.probe_external()
            except Exception as exc:  # capability seeding is best effort
                logger.debug("spark external probe failed: %s", type(exc).__name__)

    async def close(self) -> None:
        await self._manager.stop()
        if self._owns_http:
            await self._http.aclose()

    def availability(self, profile: Profile) -> ProfileCapability:
        return check_availability(
            profile,
            self._settings,
            self._probe,
            self._manager.model_present(),
            self._manager.runtime_present(),
            external_healthy=self._manager.external_healthy,
            external_n_ctx=self._manager.n_ctx if self._manager.external else None,
        )

    def profile_spec(self, profile: Profile) -> ProfileSpec:
        return self._specs[profile]

    async def version_info(self) -> dict[str, str | None]:
        """``VersionInfo`` fields for the job record (lockfile + measured runtime version)."""
        runtime = await self._manager.runtime_version()
        return version_fields(self.lockfile, self._settings, runtime)

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
            outcome = await self._manager.ensure(profile, ctx)
            spec = self._specs[profile]
            runtime_version = await self._manager.runtime_version()
            load_ms = outcome.load_ms if outcome.loaded_now else None
            if load_ms is not None:
                ctx.diagnostics["spark_load_ms"] = load_ms
            ctx.diagnostics["spark_loaded_now"] = bool(outcome.loaded_now)
            with self._manager.request_scope():
                return await self._generate(
                    profile, spec, messages, on_token, ctx, opts, load_ms, runtime_version
                )

    # --- streaming ----------------------------------------------------------------------

    async def _generate(
        self,
        profile: Profile,
        spec: ProfileSpec,
        messages: list[SparkMessage],
        on_token: TokenCallback,
        ctx: AnalysisContext,
        opts: SparkRunOptions,
        load_ms: float | None,
        runtime_version: str | None,
    ) -> SparkGeneration:
        payload: dict[str, Any] = {
            "messages": [m.model_dump() for m in messages],
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": opts.max_tokens,
            "temperature": opts.temperature,
            "top_p": opts.top_p,
            "stop": list(opts.stop),
            "cache_prompt": True,
        }
        url = f"{self._manager.base_url}/v1/chat/completions"
        timeout = httpx.Timeout(self._settings.spark_request_timeout_s, connect=CONNECT_TIMEOUT_S)

        parts: list[str] = []
        rss_samples: list[float] = []
        ttft_ms: float | None = None
        finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        timings: dict[str, Any] | None = None
        delta_count = 0
        done = False

        rss = self._manager.rss_mb()
        if rss is not None:
            rss_samples.append(rss)
        started = time.perf_counter()
        ctx.timers.start("spark")
        try:
            async with self._http.stream("POST", url, json=payload, timeout=timeout) as response:
                if response.status_code != 200:
                    raise AnalysisError(
                        ErrorCode.SPARK_INFERENCE_FAILED,
                        details={"reason": "http_status", "status": response.status_code},
                    )
                async for raw_line in _cancellable_lines(response, ctx):
                    line = raw_line.strip()
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("error:"):
                        raise AnalysisError(
                            ErrorCode.SPARK_INFERENCE_FAILED, details={"reason": "server_error"}
                        )
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    chunk = _parse_chunk(data)
                    if "error" in chunk:
                        raise AnalysisError(
                            ErrorCode.SPARK_INFERENCE_FAILED, details={"reason": "server_error"}
                        )
                    if isinstance(chunk.get("usage"), dict):
                        usage = chunk["usage"]
                    if isinstance(chunk.get("timings"), dict):
                        timings = chunk["timings"]
                    choices = chunk.get("choices") or []
                    if not choices or not isinstance(choices[0], dict):
                        continue
                    choice = choices[0]
                    if choice.get("finish_reason"):
                        finish_reason = str(choice["finish_reason"])
                    delta = choice.get("delta") or {}
                    text = delta.get("content") if isinstance(delta, dict) else None
                    if not text:
                        continue
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - started) * 1000.0
                        rss = self._manager.rss_mb()
                        if rss is not None:
                            rss_samples.append(rss)
                    delta_count += 1
                    parts.append(text)
                    await on_token(text)
        except (httpx.HTTPError, httpx.StreamError) as exc:
            raise AnalysisError(
                ErrorCode.SPARK_INFERENCE_FAILED, details={"reason": type(exc).__name__}
            ) from exc
        finally:
            total_ms = (time.perf_counter() - started) * 1000.0
            ctx.timers.stop("spark")

        if not done and finish_reason is None:
            raise AnalysisError(
                ErrorCode.SPARK_INFERENCE_FAILED, details={"reason": "stream_ended_early"}
            )
        if ctx.cancel.cancelled:
            raise AnalysisError(ctx.cancel.reason)

        rss = self._manager.rss_mb()
        if rss is not None:
            rss_samples.append(rss)
        prompt_tokens, output_tokens, tps = _token_stats(usage, timings, delta_count)
        ctx.diagnostics["spark_streamed_deltas"] = delta_count
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
            resident_rss_mb=rss_samples[-1] if rss_samples else None,
            peak_rss_mb=max(rss_samples) if rss_samples else None,
            finish_reason=finish_reason,
            runtime_version=runtime_version,
        )
        if ttft_ms is not None:
            ctx.diagnostics["spark_ttft_ms"] = ttft_ms
        logger.info(
            "spark generated profile=%s ttft_ms=%s total_ms=%.0f prompt_tokens=%s "
            "output_tokens=%s finish=%s",
            profile,
            None if ttft_ms is None else round(ttft_ms),
            total_ms,
            prompt_tokens,
            output_tokens,
            finish_reason,
        )
        return SparkGeneration(
            text="".join(parts), stats=stats, truncated=finish_reason == "length"
        )


def _parse_chunk(data: str) -> dict[str, Any]:
    try:
        chunk = json.loads(data)
    except ValueError as exc:
        raise AnalysisError(
            ErrorCode.SPARK_INFERENCE_FAILED, details={"reason": "malformed_sse"}
        ) from exc
    if not isinstance(chunk, dict):
        raise AnalysisError(ErrorCode.SPARK_INFERENCE_FAILED, details={"reason": "malformed_sse"})
    return chunk


def _token_stats(
    usage: dict[str, Any] | None,
    timings: dict[str, Any] | None,
    delta_count: int,
) -> tuple[int | None, int | None, float | None]:
    """Prompt / output token counts and tokens per second.

    Token counts come from llama-server's ``usage`` (or ``timings``); tokens per second only
    from ``timings``. When neither is present the output count falls back to the number of
    streamed deltas (an estimate) and tokens per second stays ``None`` so an estimate is never
    reported as a measurement.
    """
    prompt_tokens = _int_or_none(usage, "prompt_tokens") or _int_or_none(timings, "prompt_n")
    output_tokens = _int_or_none(usage, "completion_tokens") or _int_or_none(timings, "predicted_n")
    tps: float | None = None
    if timings:
        value = timings.get("predicted_per_second")
        if isinstance(value, int | float) and value > 0:
            tps = float(value)
        else:
            predicted_n = _int_or_none(timings, "predicted_n")
            predicted_ms = timings.get("predicted_ms")
            if predicted_n and isinstance(predicted_ms, int | float) and predicted_ms > 0:
                tps = predicted_n / (predicted_ms / 1000.0)
    # When neither usage nor timings is present the count is unknown; the streamed-delta
    # count is only a diagnostic (see ``_generate``), never a measured statistic.
    return prompt_tokens, output_tokens, tps


def _int_or_none(source: dict[str, Any] | None, key: str) -> int | None:
    if not source:
        return None
    value = source.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and value >= 0:
        return int(value)
    return None


async def _cancellable_lines(response: httpx.Response, ctx: AnalysisContext) -> AsyncIterator[str]:
    """Yield SSE lines while watching the cancel token even when the server is silent.

    During prompt processing llama-server sends nothing, sometimes for minutes on a CPU; a
    plain ``async for`` would only notice a cancel at the next line. Racing each read against
    the token lets ``POST /cancel`` close the response (llama-server aborts the request) and
    release the Spark lock promptly.
    """
    iterator = response.aiter_lines().__aiter__()
    cancel_wait = asyncio.ensure_future(ctx.cancel.wait())
    pending: asyncio.Task[str] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(iterator.__anext__())
            done, _ = await asyncio.wait(
                {pending, cancel_wait}, return_when=asyncio.FIRST_COMPLETED
            )
            if cancel_wait in done and pending not in done:
                raise AnalysisError(ctx.cancel.reason)
            task, pending = pending, None
            try:
                line = task.result()
            except StopAsyncIteration:
                return
            if ctx.cancel.cancelled:
                raise AnalysisError(ctx.cancel.reason)
            yield line
    finally:
        cancel_wait.cancel()
        if pending is not None and not pending.done():
            pending.cancel()
            with contextlib.suppress(BaseException):
                await pending
