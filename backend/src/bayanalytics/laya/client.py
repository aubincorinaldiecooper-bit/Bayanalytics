"""``LayaWorkerClient``: the Python side of the Laya worker boundary (AGENT.md sections 36, 38).

One persistent ``node worker.mjs`` child speaks newline-delimited JSON over stdin/stdout. A single
``asyncio.Lock`` serialises every request so ordering stays unambiguous; responses are matched by
id. If the worker dies, the in-flight request fails and the next call performs at most
``laya_max_restarts`` controlled restarts (respawn + reload) before failing fast. Failures surface
as ``AnalysisError(LAYA_INFERENCE_FAILED)`` whose details never contain state contents.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import psutil

from bayanalytics.config import Settings
from bayanalytics.errors import AnalysisError
from bayanalytics.laya.base import LAYA_HEAD_MAX_LEN, LAYA_MAX_LEN, LayaHealth, LayaLoadInfo
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.decisions import (
    ChoiceAnswer,
    LayaAnswer,
    LayaQuestion,
    LayaResult,
    LayaUsage,
    NoulAnswer,
    ScoreAnswer,
)

log = logging.getLogger(__name__)
worker_log = logging.getLogger("bayanalytics.laya.worker")

WORKER_SCRIPT = "worker.mjs"
CLOSE_TIMEOUT_S = 5.0
TERMINATE_TIMEOUT_S = 2.0
REAP_TIMEOUT_S = 2.0
HEALTH_TIMEOUT_S = 10.0
MAX_ERROR_MESSAGE_CHARS = 300
STREAM_LIMIT = 8 * 1024 * 1024
_MB = 1048576.0


def _worker_error(
    code: str, message: str, *, retryable: bool | None = None, **details: Any
) -> AnalysisError:
    """LAYA_INFERENCE_FAILED with a worker code and a truncated message (never state contents)."""
    payload: dict[str, Any] = {"worker_code": code, "message": message[:MAX_ERROR_MESSAGE_CHARS]}
    payload.update(details)
    return AnalysisError(ErrorCode.LAYA_INFERENCE_FAILED, retryable=retryable, details=payload)


def parse_answer(raw: Any) -> LayaAnswer:
    """Map a worker answer object onto the typed answer by shape (choice / noul / score)."""
    if not isinstance(raw, Mapping):
        raise ValueError("answer is not an object")
    if "choice" in raw:
        probabilities = raw.get("probabilities") or {}
        if not isinstance(probabilities, Mapping):
            raise ValueError("choice probabilities are not an object")
        return ChoiceAnswer(
            choice=str(raw["choice"]),
            probabilities={str(k): float(v) for k, v in probabilities.items()},
        )
    if "noul" in raw:
        return NoulAnswer(noul=float(raw["noul"]))
    if "score" in raw:
        distribution: list[float] | None = None
        probabilities = raw.get("probabilities")
        if isinstance(probabilities, Mapping) and probabilities:
            try:
                ordered = sorted(probabilities, key=lambda k: int(k))
                distribution = [float(probabilities[k]) for k in ordered]
            except (TypeError, ValueError):
                distribution = None
        elif isinstance(probabilities, list):
            distribution = [float(v) for v in probabilities]
        return ScoreAnswer(score=float(raw["score"]), distribution=distribution)
    raise ValueError("answer has none of choice / score / noul")


def parse_result(
    raw: Mapping[str, Any], expected_keys: Mapping[str, Any] | None = None
) -> LayaResult:
    answers_raw = raw.get("answers")
    if not isinstance(answers_raw, Mapping):
        raise ValueError("result has no answers object")
    if expected_keys is not None:
        missing = [k for k in expected_keys if k not in answers_raw]
        if missing:
            raise ValueError(f"missing answers for {len(missing)} question(s)")
    answers = {str(k): parse_answer(v) for k, v in answers_raw.items()}
    usage_raw = raw.get("usage") or {}
    tokens = usage_raw.get("input_tokens") if isinstance(usage_raw, Mapping) else None
    usage = LayaUsage(input_tokens=int(tokens) if isinstance(tokens, (int, float)) else None)
    latency = raw.get("latency_ms")
    return LayaResult(
        answers=answers,
        usage=usage,
        latency_ms=float(latency) if isinstance(latency, (int, float)) else None,
    )


async def _drain_stderr(stream: asyncio.StreamReader, pid: int) -> None:
    """Forward the worker's stderr to the backend logger; stdout is protocol only."""
    try:
        while True:
            line = await stream.readline()
            if not line:
                return
            worker_log.info("pid %s: %s", pid, line.decode("utf-8", "replace").rstrip())
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - a broken stderr pipe is harmless
        log.debug("laya worker stderr reader stopped: %s", exc)


class LayaWorkerClient:
    """Persistent Node worker client implementing ``LayaClient``."""

    def __init__(
        self,
        settings: Settings,
        env: Mapping[str, str] | None = None,
        *,
        revision: str | None = None,
    ) -> None:
        self._settings = settings
        self._env_overrides = dict(env or {})
        self._revision = revision
        self._lock = asyncio.Lock()
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._ps: psutil.Process | None = None
        self._seq = 0
        self._abandoned: set[str] = set()
        self._dead = False
        self._ever_started = False
        self._loaded = False
        self._was_loaded = False
        self._closed = False
        self._restarts = 0
        self._load_info: LayaLoadInfo | None = None
        self._started_at: float | None = None
        self._stats: dict[str, Any] = {
            "load_ms": None,
            "resident_rss_mb": None,
            "peak_rss_mb": None,
            "restarts": 0,
            "requests": 0,
            "warm_inference_ms": None,
        }

    # ---- public surface --------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, Any]:
        """load_ms, resident_rss_mb, peak_rss_mb, restarts, requests, warm_inference_ms.

        ``requests`` counts ``system_one`` calls sent to the worker.
        """
        out = dict(self._stats)
        out["restarts"] = self._restarts
        return out

    @property
    def load_info(self) -> LayaLoadInfo | None:
        return self._load_info

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def restarts(self) -> int:
        return self._restarts

    @property
    def started_at(self) -> float | None:
        """``time.monotonic()`` when the current worker process was spawned."""
        return self._started_at

    async def load(self) -> LayaLoadInfo:
        """Start the worker if needed and load the model. Idempotent."""
        async with self._lock:
            await self._ensure_worker_locked()
            if self._loaded and self._load_info is not None:
                return self._load_info
            return await self._load_locked()

    async def system_one(
        self, state: dict[str, Any] | str, questions: dict[str, LayaQuestion]
    ) -> LayaResult:
        """One ``systemOne`` round trip. Loads first if the model is not resident yet."""
        payload_questions = {
            key: q.to_laya() if isinstance(q, LayaQuestion) else dict(q)
            for key, q in questions.items()
        }
        async with self._lock:
            await self._ensure_worker_locked()
            if not self._loaded:
                await self._load_locked()
            started = time.perf_counter()
            self._stats["requests"] += 1
            raw = await self._request_locked(
                "system_one",
                {"state": state, "questions": payload_questions},
                self._settings.laya_request_timeout_s,
            )
            round_trip_ms = (time.perf_counter() - started) * 1000.0
            try:
                result = parse_result(raw, expected_keys=payload_questions)
            except (ValueError, TypeError, KeyError) as exc:
                raise _worker_error(
                    "PROTOCOL_ERROR", f"unparseable answers: {type(exc).__name__}: {exc}"
                ) from exc
            if result.latency_ms is None:
                result.latency_ms = round(round_trip_ms, 1)
            self._stats["warm_inference_ms"] = result.latency_ms
            self._sample_rss()
            return result

    async def health(self) -> LayaHealth:
        """Never raises. ``ok`` means the worker process is alive and answering."""
        proc = self._proc
        restarts = self._restarts
        if self._closed:
            return LayaHealth(ok=False, loaded=False, restarts=restarts, detail="closed")
        if proc is None:
            return LayaHealth(ok=False, loaded=False, restarts=restarts, detail="not started")
        if self._dead or proc.returncode is not None:
            detail = f"worker exited (code {proc.returncode})"
            if restarts >= self._settings.laya_max_restarts:
                detail += "; restarts exhausted"
            return LayaHealth(
                ok=False, loaded=False, pid=proc.pid, restarts=restarts, detail=detail
            )
        if self._lock.locked():
            return LayaHealth(
                ok=True,
                loaded=self._loaded,
                pid=proc.pid,
                resident_rss_mb=self._sample_rss(),
                restarts=restarts,
                detail="busy",
            )
        try:
            async with self._lock:
                timeout_s = min(HEALTH_TIMEOUT_S, self._settings.laya_request_timeout_s)
                raw = await self._request_locked("health", {}, timeout_s)
        except AnalysisError as exc:
            return LayaHealth(
                ok=False,
                loaded=False,
                pid=proc.pid,
                restarts=self._restarts,
                detail=str(exc.details.get("worker_code", "error")),
            )
        rss = self._sample_rss()
        if rss is None and isinstance(raw.get("rss_mb"), (int, float)):
            rss = float(raw["rss_mb"])
        return LayaHealth(
            ok=True,
            loaded=bool(raw.get("loaded", self._loaded)),
            pid=proc.pid,
            resident_rss_mb=rss,
            restarts=self._restarts,
        )

    async def close(self) -> None:
        """Polite close, then terminate, then kill. Safe to call twice."""
        self._closed = True
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        self._loaded = False
        if proc.returncode is None and not self._dead and not self._abandoned:
            acquired = False
            try:
                await asyncio.wait_for(self._lock.acquire(), CLOSE_TIMEOUT_S)
                acquired = True
                await self._request_on(proc, "close", {}, CLOSE_TIMEOUT_S)
            except (AnalysisError, TimeoutError, OSError):
                pass
            finally:
                if acquired:
                    self._lock.release()
            try:
                await asyncio.wait_for(proc.wait(), CLOSE_TIMEOUT_S)
            except TimeoutError:
                pass
        await self._stop(proc)
        await self._cleanup_proc(proc)

    # ---- process management ----------------------------------------------------------------

    async def _ensure_worker_locked(self) -> None:
        if self._closed:
            raise _worker_error("CLOSED", "client is closed", retryable=False)
        proc = self._proc
        if proc is not None and proc.returncode is None and not self._dead:
            return
        if proc is None and not self._ever_started:
            await self._spawn_locked()
            return
        # The worker died: at most laya_max_restarts controlled restarts, never a loop.
        max_restarts = self._settings.laya_max_restarts
        if self._restarts >= max_restarts:
            raise _worker_error(
                "WORKER_EXITED",
                "worker exited and the restart budget is exhausted",
                retryable=False,
                restarts_exhausted=True,
                restarts=self._restarts,
            )
        self._restarts += 1
        self._stats["restarts"] = self._restarts
        log.warning("laya worker died; controlled restart %s/%s", self._restarts, max_restarts)
        if proc is not None:
            await self._stop(proc)
            await self._cleanup_proc(proc)
        await self._spawn_locked()
        if self._was_loaded:
            await self._load_locked()

    async def _spawn_locked(self) -> None:
        settings = self._settings
        worker_dir = Path(settings.laya_worker_dir)
        script = worker_dir / WORKER_SCRIPT
        if not script.is_file():
            raise _worker_error(
                "SPAWN_FAILED", f"worker script not found: {script}", retryable=False
            )
        env = {**os.environ, **self._env_overrides}
        try:
            proc = await asyncio.create_subprocess_exec(
                settings.laya_node_bin,
                WORKER_SCRIPT,
                cwd=str(worker_dir),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                limit=STREAM_LIMIT,
            )
        except (OSError, ValueError) as exc:
            raise _worker_error(
                "SPAWN_FAILED", f"{type(exc).__name__}: {exc}", retryable=False
            ) from exc
        self._proc = proc
        self._dead = False
        self._loaded = False
        self._abandoned = set()
        self._ever_started = True
        self._started_at = time.monotonic()
        try:
            self._ps = psutil.Process(proc.pid)
        except psutil.Error:
            self._ps = None
        assert proc.stderr is not None
        self._stderr_task = asyncio.create_task(_drain_stderr(proc.stderr, proc.pid))
        log.info("laya worker started (pid %s)", proc.pid)

    async def _load_locked(self) -> LayaLoadInfo:
        settings = self._settings
        params: dict[str, Any] = {}
        if settings.laya_model_dir is not None:
            params["modelDir"] = str(settings.laya_model_dir)
        if settings.laya_cache_dir is not None:
            params["cacheDir"] = str(settings.laya_cache_dir)
        if settings.laya_threads:
            params["threads"] = int(settings.laya_threads)
        if self._revision:
            params["revision"] = self._revision
        started = time.perf_counter()
        raw = await self._request_locked("load", params, settings.laya_load_timeout_s)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        load_ms = raw.get("load_ms")
        rss = self._sample_rss()
        if rss is None and isinstance(raw.get("rss_mb"), (int, float)):
            rss = float(raw["rss_mb"])
        info = LayaLoadInfo(
            load_ms=float(load_ms) if isinstance(load_ms, (int, float)) else round(elapsed_ms, 1),
            resident_rss_mb=rss,
            package_version=str(raw["package_version"]) if raw.get("package_version") else None,
            model_dir=str(raw["model_dir"]) if raw.get("model_dir") else None,
            max_len=int(raw.get("max_len") or LAYA_MAX_LEN),
            head_max_len=int(raw.get("head_max_len") or LAYA_HEAD_MAX_LEN),
        )
        self._loaded = True
        self._was_loaded = True
        self._load_info = info
        self._stats["load_ms"] = info.load_ms
        log.info(
            "laya loaded in %.0f ms (package %s, rss %s MB)",
            info.load_ms,
            info.package_version,
            None if rss is None else round(rss),
        )
        return info

    async def _request_locked(
        self, op: str, params: dict[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.returncode is not None or self._dead:
            raise _worker_error("WORKER_EXITED", f"worker is not running (op {op})")
        return await self._request_on(proc, op, params, timeout_s)

    async def _request_on(
        self,
        proc: asyncio.subprocess.Process,
        op: str,
        params: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        """Write one request line and read its response, skipping late replies to timed-out ids."""
        assert proc.stdin is not None and proc.stdout is not None
        self._seq += 1
        rid = str(self._seq)
        try:
            line = json.dumps(
                {"id": rid, "op": op, "params": params},
                ensure_ascii=False,
                default=str,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise _worker_error(
                "BAD_STATE",
                f"request is not JSON-serialisable: {type(exc).__name__}",
                retryable=False,
            ) from exc
        try:
            proc.stdin.write(line.encode("utf-8") + b"\n")
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            code = await self._mark_dead(proc)
            raise _worker_error(
                "WORKER_EXITED", f"worker pipe broke during {op}", exit_code=code
            ) from exc
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._abandoned.add(rid)
                raise _worker_error("TIMEOUT", f"{op} exceeded {timeout_s:g}s")
            try:
                raw = await asyncio.wait_for(proc.stdout.readline(), remaining)
            except TimeoutError:
                self._abandoned.add(rid)
                raise _worker_error("TIMEOUT", f"{op} exceeded {timeout_s:g}s") from None
            if not raw:
                code = await self._mark_dead(proc)
                raise _worker_error("WORKER_EXITED", f"worker exited during {op}", exit_code=code)
            try:
                response = json.loads(raw)
            except ValueError as exc:
                await self._mark_dead(proc, kill=True)
                raise _worker_error("PROTOCOL_ERROR", "non-JSON line on worker stdout") from exc
            if not isinstance(response, Mapping):
                await self._mark_dead(proc, kill=True)
                raise _worker_error("PROTOCOL_ERROR", "worker response is not an object")
            response_id = response.get("id")
            response_id = str(response_id) if response_id is not None else None
            if response_id != rid:
                if response_id in self._abandoned:
                    self._abandoned.discard(response_id)
                    continue  # late answer to a request that already timed out
                await self._mark_dead(proc, kill=True)
                raise _worker_error("PROTOCOL_ERROR", "worker response id mismatch")
            if response.get("ok"):
                result = response.get("result")
                return dict(result) if isinstance(result, Mapping) else {}
            error = response.get("error") or {}
            if not isinstance(error, Mapping):
                error = {}
            raise _worker_error(
                str(error.get("code") or "UNKNOWN"), str(error.get("message") or "worker error")
            )

    async def _mark_dead(
        self, proc: asyncio.subprocess.Process, *, kill: bool = False
    ) -> int | None:
        self._dead = True
        self._loaded = False
        if kill:
            await self._stop(proc)
        try:
            await asyncio.wait_for(proc.wait(), REAP_TIMEOUT_S)
        except TimeoutError:
            await self._stop(proc)
        await self._cleanup_proc(proc)
        return proc.returncode

    async def _stop(self, proc: asyncio.subprocess.Process) -> None:
        """terminate(), then kill(); ignores a process that is already gone."""
        if proc.returncode is not None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), TERMINATE_TIMEOUT_S)
            return
        except TimeoutError:
            pass
        try:
            proc.kill()
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(proc.wait(), TERMINATE_TIMEOUT_S)
        except TimeoutError:  # pragma: no cover - the kernel did not honour SIGKILL
            log.error("laya worker pid %s did not exit after kill()", proc.pid)

    async def _cleanup_proc(self, proc: asyncio.subprocess.Process) -> None:
        if proc.stdin is not None and not proc.stdin.is_closing():
            proc.stdin.close()
        task = self._stderr_task
        if task is not None and not task.done():
            # The reader ends by itself once the child's stderr closes; cancel only if it lingers.
            await asyncio.wait({task}, timeout=REAP_TIMEOUT_S)
            if not task.done():
                task.cancel()
                await asyncio.wait({task}, timeout=1.0)
        self._stderr_task = None
        self._ps = None

    def _sample_rss(self) -> float | None:
        """Best-effort child RSS in MB via psutil; updates resident/peak stats, never raises."""
        ps = self._ps
        if ps is None:
            return None
        try:
            rss = ps.memory_info().rss / _MB
        except (psutil.Error, OSError):
            return None
        self._stats["resident_rss_mb"] = rss
        peak = self._stats.get("peak_rss_mb")
        self._stats["peak_rss_mb"] = rss if peak is None else max(peak, rss)
        return rss
