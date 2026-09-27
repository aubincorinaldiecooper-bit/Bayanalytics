"""llama-server process manager: one managed process, one loaded profile at a time.

AGENT.md references: section 29 (load time measured separately from inference), section 35
(profile snapshot per request, safe local allocation), section 36 (llama.cpp behind a process
boundary, called over localhost HTTP), section 38 (the Spark lock owns Fast / Deep transitions;
never restart while a request is in flight).

Every request carries the server's API key (managed: generated per process unless
``BAY_SPARK_API_KEY`` is set, passed through ``LLAMA_API_KEY`` in the child environment;
external: ``BAY_SPARK_API_KEY`` when that server needs one).

Modes (``settings.spark_mode``):

- ``managed``: this class spawns ``llama-server`` with the profile's context size and KV cache
  type, waits for ``GET /health`` to return 200, and restarts it when a different profile is
  requested. Command line (llama.cpp b10828+)::

      llama-server -m <gguf> -c <ctx> --host 127.0.0.1 --port <port> [-t <threads>]
                   -ctk <kv> -ctv <kv> [-fa on] --no-webui --jinja

  ``-fa on`` is added whenever the KV cache type is not ``f16`` because quantized V caches need
  flash attention.
- ``external``: an operator runs llama-server; ``ensure`` only verifies ``/health`` and reads
  ``/props`` for the loaded context size. The profile is whatever that server has.

The process' stdout is discarded; stderr is inherited so llama-server's own startup log stays
visible in the backend terminal. Nothing here logs prompt content. Every llama-server child
(the server and the ``--version`` probe) runs with the allow-listed environment from
:mod:`bayanalytics.procenv`, so it never sees backend settings or secrets.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import shutil
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx
import psutil
from pydantic import BaseModel

from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.procenv import child_env
from bayanalytics.schemas.common import ErrorCode, Profile
from bayanalytics.spark.base import ProfileSpec
from bayanalytics.spark.profiles import (
    MemoryProbe,
    assert_can_allocate,
    profile_specs,
    profile_unavailable_code,
    psutil_probe,
)

logger = logging.getLogger("bayanalytics.spark")

PROCESS_STOP_GRACE_S = 10.0
HEALTH_POLL_INTERVAL_S = 0.5
HEALTH_REQUEST_TIMEOUT_S = 2.0
VERSION_TIMEOUT_S = 15.0
_MIB = 1024.0 * 1024.0


class ProcessLike(Protocol):
    """The subset of ``asyncio.subprocess.Process`` the manager uses (fakeable in tests)."""

    pid: int
    returncode: int | None

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    async def wait(self) -> int: ...


SpawnFn = Callable[[list[str], dict[str, str]], Awaitable[ProcessLike]]
"""Spawn llama-server from ``argv`` with ``env`` added to the allow-listed child environment."""
VersionProbe = Callable[[], Awaitable[str | None]]


class LoadOutcome(BaseModel):
    loaded_now: bool
    load_ms: float | None = None
    profile: Profile
    external: bool = False
    n_ctx: int | None = None


async def default_spawn(argv: list[str], env: dict[str, str]) -> ProcessLike:
    """Spawn llama-server with an allow-listed environment (never the backend's own) plus
    ``env`` (the API key travels here, not on the command line)."""
    return await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=None,
        env=child_env(env),
    )


class LlamaServerManager:
    def __init__(
        self,
        settings: Settings,
        probe: MemoryProbe | None = None,
        spawn: SpawnFn | None = None,
        http: httpx.AsyncClient | None = None,
        *,
        version_probe: VersionProbe | None = None,
        health_poll_interval_s: float = HEALTH_POLL_INTERVAL_S,
    ) -> None:
        self._settings = settings
        self._probe: MemoryProbe = probe or psutil_probe
        self._spawn: SpawnFn = spawn or default_spawn
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(timeout=HEALTH_REQUEST_TIMEOUT_S)
        self._version_probe: VersionProbe = version_probe or self._run_version_command
        self._poll_interval = health_poll_interval_s
        self._specs = profile_specs(settings)

        # Managed servers always run behind a key: the configured one, else a per-process
        # random key. External servers use the configured key when there is one.
        self.api_key: str | None = settings.spark_api_key or (
            None if settings.spark_mode == "external" else secrets.token_urlsafe(24)
        )
        parts = urlsplit(settings.spark_server_url)
        self.host = parts.hostname or "127.0.0.1"
        self.port = parts.port or 8081
        if settings.spark_mode == "external":
            self.base_url = settings.spark_server_url.rstrip("/")
        else:
            self.base_url = f"http://{self.host}:{self.port}"

        self.loaded_profile: Profile | None = None
        self.process: ProcessLike | None = None
        self.pid: int | None = None
        self.started_at: float | None = None
        self.load_ms: float | None = None
        self.n_ctx: int | None = None
        self.external = settings.spark_mode == "external"
        self.external_healthy: bool | None = None
        self.peak_rss_mb: float | None = None
        self._in_flight = False
        self._runtime_version: str | None = None
        self._version_checked = False
        self._props_build: str | None = None

    # --- state --------------------------------------------------------------------------

    @property
    def alive(self) -> bool:
        if self.external:
            return bool(self.external_healthy)
        return self.process is not None and self.process.returncode is None

    @property
    def in_flight(self) -> bool:
        return self._in_flight

    def spec(self, profile: Profile) -> ProfileSpec:
        return self._specs[profile]

    @contextmanager
    def request_scope(self) -> Iterator[None]:
        """Marks a generation as in flight so a restart during it is a programming error."""
        self._in_flight = True
        try:
            yield
        finally:
            self._in_flight = False

    # --- ensure -------------------------------------------------------------------------

    async def ensure(self, profile: Profile, ctx: AnalysisContext) -> LoadOutcome:
        """Make ``profile`` the loaded profile. Caller must hold the Spark lock."""
        if self._in_flight:
            raise AnalysisError(
                ErrorCode.INTERNAL_ERROR,
                details={"reason": "spark profile change attempted while a request is in flight"},
            )
        if self._settings.spark_mode == "external":
            return await self._ensure_external(profile)
        if self._settings.spark_mode != "managed":
            raise AnalysisError(
                ErrorCode.SPARK_START_FAILED,
                details={"stage": "config", "reason": "unsupported spark mode"},
            )
        return await self._ensure_managed(profile, ctx)

    async def _ensure_managed(self, profile: Profile, ctx: AnalysisContext) -> LoadOutcome:
        if self.loaded_profile == profile and self.alive and await self._health_ok():
            return LoadOutcome(loaded_now=False, load_ms=self.load_ms, profile=profile)

        await self.stop()
        if not self.model_present():
            raise AnalysisError(
                ErrorCode.SPARK_START_FAILED,
                "The synthesis model could not be started: model artifact not found.",
                details={"stage": "preflight", "reason": "model artifact not found"},
            )
        spec = self._specs[profile]
        assert_can_allocate(profile, self._settings, self._probe)

        await ctx.event(
            "spark.loading",
            profile=profile,
            context_ceiling=spec.context_ceiling,
            kv_cache_type=spec.kv_cache_type,
        )
        logger.info(
            "spark loading profile=%s ctx=%d kv=%s",
            profile,
            spec.context_ceiling,
            spec.kv_cache_type,
        )
        argv = self.build_argv(profile)
        logger.debug("spark argv=%s", argv)
        started = time.perf_counter()
        try:
            process = await self._spawn(argv, self.child_env_extra())
        except FileNotFoundError as exc:
            raise AnalysisError(
                ErrorCode.SPARK_START_FAILED,
                "The synthesis model could not be started: llama-server not found.",
                details={"stage": "spawn", "reason": "llama-server not found"},
            ) from exc
        except OSError as exc:
            raise AnalysisError(
                ErrorCode.SPARK_START_FAILED,
                details={"stage": "spawn", "reason": type(exc).__name__},
            ) from exc
        self.process = process
        self.pid = process.pid
        self.started_at = time.time()
        self.peak_rss_mb = None
        try:
            await self._wait_healthy(process)
        except AnalysisError:
            await self.stop()
            raise
        self.load_ms = (time.perf_counter() - started) * 1000.0
        self.loaded_profile = profile
        self.rss_mb()
        props = await self._fetch_props()
        self.n_ctx = _props_n_ctx(props) or spec.context_ceiling
        logger.info(
            "spark loaded profile=%s load_ms=%.0f rss_mb=%s",
            profile,
            self.load_ms,
            None if self.peak_rss_mb is None else round(self.peak_rss_mb),
        )
        return LoadOutcome(loaded_now=True, load_ms=self.load_ms, profile=profile, n_ctx=self.n_ctx)

    async def _ensure_external(self, profile: Profile) -> LoadOutcome:
        healthy = await self._health_ok()
        self.external_healthy = healthy
        if not healthy:
            raise AnalysisError(
                ErrorCode.SPARK_START_FAILED,
                "The external synthesis server is not ready.",
                details={"stage": "external_health"},
            )
        props = await self._fetch_props()
        self.n_ctx = _props_n_ctx(props)
        spec = self._specs[profile]
        if self.n_ctx is not None and self.n_ctx < spec.context_ceiling:
            raise AnalysisError(
                profile_unavailable_code(profile),
                f"The external synthesis server is loaded with a {self.n_ctx} token context; "
                f"the {profile} profile needs {spec.context_ceiling}.",
                details={"n_ctx": self.n_ctx, "context_ceiling": spec.context_ceiling},
            )
        loaded_now = self.loaded_profile != profile
        self.loaded_profile = profile
        return LoadOutcome(
            loaded_now=loaded_now, load_ms=None, profile=profile, external=True, n_ctx=self.n_ctx
        )

    async def probe_external(self) -> bool:
        """Best-effort health + props refresh for ``/capabilities`` in external mode."""
        healthy = await self._health_ok()
        self.external_healthy = healthy
        if healthy:
            props = await self._fetch_props()
            n_ctx = _props_n_ctx(props)
            if n_ctx is not None:
                self.n_ctx = n_ctx
        return healthy

    @property
    def auth_headers(self) -> dict[str, str]:
        """``Authorization`` for every llama-server request (empty without a key)."""
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def child_env_extra(self) -> dict[str, str]:
        """Environment added for a managed llama-server: its API key (``LLAMA_API_KEY``)."""
        return {"LLAMA_API_KEY": self.api_key} if self.api_key else {}

    def model_present(self) -> bool:
        model_path = self._settings.spark_model_path
        return model_path is not None and Path(model_path).is_file()

    def runtime_present(self) -> bool:
        binary = self._settings.spark_llama_server_bin
        return shutil.which(binary) is not None or Path(binary).is_file()

    def build_argv(self, profile: Profile) -> list[str]:
        settings = self._settings
        spec = self._specs[profile]
        argv = [
            settings.spark_llama_server_bin,
            "-m",
            str(settings.spark_model_path),
            "-c",
            str(spec.context_ceiling),
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if settings.spark_threads:
            argv += ["-t", str(settings.spark_threads)]
        argv += ["-ctk", spec.kv_cache_type, "-ctv", spec.kv_cache_type]
        if spec.kv_cache_type != "f16":
            argv += ["-fa", "on"]
        argv += ["--no-webui", "--jinja"]
        return argv

    async def _wait_healthy(self, process: ProcessLike) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._settings.spark_start_timeout_s
        while True:
            if process.returncode is not None:
                raise AnalysisError(
                    ErrorCode.SPARK_START_FAILED,
                    "The synthesis model process exited during startup.",
                    details={"stage": "process_exit", "returncode": process.returncode},
                )
            if await self._health_ok():
                return
            if loop.time() >= deadline:
                raise AnalysisError(
                    ErrorCode.SPARK_START_FAILED,
                    "The synthesis model did not become ready in time.",
                    details={
                        "stage": "health_wait",
                        "timeout_s": self._settings.spark_start_timeout_s,
                    },
                )
            await asyncio.sleep(self._poll_interval)

    async def _health_ok(self) -> bool:
        try:
            response = await self._http.get(
                f"{self.base_url}/health",
                timeout=HEALTH_REQUEST_TIMEOUT_S,
                headers=self.auth_headers,
            )
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    async def _fetch_props(self) -> dict[str, Any] | None:
        try:
            response = await self._http.get(
                f"{self.base_url}/props",
                timeout=HEALTH_REQUEST_TIMEOUT_S,
                headers=self.auth_headers,
            )
            if response.status_code != 200:
                return None
            props = response.json()
        except (httpx.HTTPError, ValueError):
            return None
        if not isinstance(props, dict):
            return None
        build = props.get("build_info")
        if isinstance(build, str) and build:
            self._props_build = build
        return props

    # --- lifecycle ----------------------------------------------------------------------

    async def stop(self) -> None:
        process = self.process
        if process is not None and process.returncode is None:
            logger.info("spark stopping profile=%s", self.loaded_profile)
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), PROCESS_STOP_GRACE_S)
            except TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
        self.process = None
        self.pid = None
        self.loaded_profile = None
        self.started_at = None
        if not self.external:
            self.n_ctx = None

    async def aclose(self) -> None:
        await self.stop()
        if self._owns_http:
            await self._http.aclose()

    # --- measurements -------------------------------------------------------------------

    def rss_mb(self) -> float | None:
        """Resident set size of the managed process in MB (``None`` when not managed)."""
        if self.pid is None or self.process is None or self.process.returncode is not None:
            return None
        try:
            rss = psutil.Process(self.pid).memory_info().rss / _MIB
        except (psutil.Error, OSError):
            return None
        if self.peak_rss_mb is None or rss > self.peak_rss_mb:
            self.peak_rss_mb = rss
        return rss

    async def runtime_version(self) -> str | None:
        """``llama-server --version`` output (cached, best effort), else ``/props`` build info."""
        if not self._version_checked:
            self._version_checked = True
            try:
                self._runtime_version = await self._version_probe()
            except Exception as exc:  # best effort by contract
                logger.debug("spark version probe failed: %s", type(exc).__name__)
                self._runtime_version = None
        return self._runtime_version or self._props_build

    async def _run_version_command(self) -> str | None:
        if not self.runtime_present():
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                self._settings.spark_llama_server_bin,
                "--version",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=child_env(),
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), VERSION_TIMEOUT_S)
        except (OSError, TimeoutError):
            return None
        for stream in (stdout, stderr):
            for line in stream.decode("utf-8", "replace").splitlines():
                text = line.strip()
                if text:
                    return text[:200]
        return None


def _props_n_ctx(props: dict[str, Any] | None) -> int | None:
    if not props:
        return None
    defaults = props.get("default_generation_settings")
    if isinstance(defaults, dict):
        value = defaults.get("n_ctx")
        if isinstance(value, int) and value > 0:
            return value
    value = props.get("n_ctx")
    if isinstance(value, int) and value > 0:
        return value
    return None
