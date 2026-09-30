"""Runtime container: every process-wide dependency the API and the pipeline share.

Built once per process by ``build_runtime`` (see ``wiring.py``) and attached to the FastAPI
app state. Tests build it with doubles injected through ``build_runtime``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from bayanalytics.config import Settings
from bayanalytics.errors import AnalysisError
from bayanalytics.jobs.bus import AnalysisEventBus
from bayanalytics.jobs.runner import AnalysisRunner
from bayanalytics.laya.base import LayaClient
from bayanalytics.schemas.capabilities import Capabilities, ComponentHealth, Health
from bayanalytics.schemas.results import ExecutionInfo
from bayanalytics.spark.base import SparkClient
from bayanalytics.store.base import AnalysisStore
from bayanalytics.whisper.base import Transcriber

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    settings: Settings
    store: AnalysisStore
    bus: AnalysisEventBus
    runner: AnalysisRunner
    laya: LayaClient
    spark: SparkClient
    transcriber: Transcriber
    research: Any = None  # the web research provider; opaque to the API layer
    extras: dict[str, Any] = field(default_factory=dict)
    _started: bool = False
    _health_cache: tuple[float, Health] | None = None

    @property
    def search_configured(self) -> bool:
        """Whether the research provider has a web search backend to ask (the only source of
        evidence; an analysis cannot run without one)."""
        return bool(getattr(self.research, "search_configured", False))

    def execution_info(self) -> ExecutionInfo:
        s = self.settings
        return ExecutionInfo(
            spark_mode=s.spark_mode,
            whisper_mode=s.whisper_mode,
            deployment=s.deployment,
            search_configured=self.search_configured,
        )

    async def start(self) -> None:
        if self._started:
            return
        self.extras["execution"] = self.execution_info()
        if not self.search_configured:
            log.warning(
                "BAY_RESEARCH_SEARCH_URL is not set: web search is not configured, so analyses "
                "cannot run (evidence comes only from web search)"
            )
        await self.store.start()
        await self.runner.start()
        # Laya is the frequent decision layer: keep it resident for the process lifetime.
        try:
            info = await self.laya.load()
        except AnalysisError as exc:
            details = exc.details or {}
            raise RuntimeError(
                "the Laya decision model failed to load "
                f"({details.get('worker_code') or exc.code}: {details.get('reason')}); "
                "set BAY_LAYA_MODEL_DIR to the bundle or run scripts/install_laya_worker.sh"
            ) from exc
        self.extras["laya_load"] = info.model_dump()
        log.info(
            "laya loaded in %.0f ms (package %s)", info.load_ms, info.package_version or "unknown"
        )
        await self.spark.start()
        await self._probe_spark_version()
        voice = self.transcriber.available()
        log.info("voice input %s", "available" if voice else "disabled")
        self.extras["voice_available_at_start"] = voice
        self._started = True

    async def _probe_spark_version(self) -> None:
        """Record what the Spark client measured about its runtime and artifact (best effort)."""
        version_info = getattr(self.spark, "version_info", None)
        if not callable(version_info):
            return
        try:
            info = await version_info()
        except Exception:  # pragma: no cover - version probing is best effort
            log.debug("spark version probe failed", exc_info=True)
            return
        self.extras["spark_version"] = (
            info.model_dump() if hasattr(info, "model_dump") else dict(info or {})
        )

    async def refresh_spark(self) -> None:
        """External llama-server mode: re-check the server so one started after this backend
        becomes available without a restart. Managed mode needs no refresh."""
        if self.settings.spark_mode != "external":
            return
        manager = getattr(self.spark, "manager", None)
        probe = getattr(manager, "probe_external", None)
        if callable(probe):
            try:
                if await probe():
                    await self._probe_spark_version()
            except Exception:  # pragma: no cover - a probe failure just keeps the old state
                log.debug("spark external probe failed", exc_info=True)

    async def close(self) -> None:
        # Stop accepting new jobs, cancel/finish active work, then close runtimes.
        await self.runner.shutdown()
        closers = [("spark", self.spark.close), ("laya", self.laya.close)]
        aclose = getattr(self.research, "aclose", None)
        if callable(aclose):
            closers.append(("research", aclose))
        closers.append(("transcriber", self.transcriber.close))
        for name, closer in closers:
            try:
                await closer()
            except Exception:  # pragma: no cover - best effort shutdown
                log.exception("error closing %s", name)
        await self.store.close()
        self._started = False

    # -- capability / health views ---------------------------------------------------------
    def capabilities(self) -> Capabilities:
        profiles = {
            name: self.spark.availability(name)  # type: ignore[arg-type]
            for name in ("fast", "deep")
        }
        return Capabilities(
            profiles=profiles,
            voice=self.transcriber.available(),
            deployment=self.settings.deployment,
            research=self.research is not None,
            web_search=self.search_configured,
            execution=self.execution_info(),
        )

    async def health(self, version: str) -> Health:
        # Cached briefly: the Laya round trip takes the worker lock, and unauthenticated
        # pollers must not contend with analyses.
        now = time.monotonic()
        cached = self._health_cache
        if cached is not None and now - cached[0] < self.settings.health_cache_s:
            return cached[1].model_copy(update={"active_analyses": self.runner.active_count})
        health = await self._health(version)
        self._health_cache = (now, health)
        return health

    async def _health(self, version: str) -> Health:
        await self.refresh_spark()
        components: list[ComponentHealth] = []
        try:
            laya = await self.laya.health()
            components.append(
                ComponentHealth(
                    name="laya",
                    status="ok" if laya.ok and laya.loaded else "degraded",
                    detail=laya.detail,
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            components.append(
                ComponentHealth(name="laya", status="down", detail=type(exc).__name__)
            )
        fast = self.spark.availability("fast")
        components.append(
            ComponentHealth(
                name="spark", status="ok" if fast.available else "degraded", detail=fast.reason
            )
        )
        components.append(
            ComponentHealth(
                name="whisper",
                status="ok" if self.transcriber.available() else "disabled",
            )
        )
        try:
            count_active = getattr(self.store, "count_active", None)
            if callable(count_active):
                await count_active()
            components.append(ComponentHealth(name="store", status="ok"))
        except Exception as exc:
            components.append(
                ComponentHealth(name="store", status="down", detail=type(exc).__name__)
            )
        overall = "ok"
        if any(c.status == "down" for c in components):
            overall = "down"
        elif any(c.status == "degraded" for c in components):
            overall = "degraded"
        return Health(
            status=overall,
            version=version,
            components=components,
            active_analyses=self.runner.active_count,
            execution=self.execution_info(),
        )
