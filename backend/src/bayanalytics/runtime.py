"""Runtime container: every process-wide dependency the API and the pipeline share.

Built once per process by ``build_runtime`` (see ``wiring.py``) and attached to the FastAPI
app state. Tests build it with mock runtimes and the in-memory store.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from bayanalytics.config import Settings
from bayanalytics.jobs.bus import AnalysisEventBus
from bayanalytics.jobs.runner import AnalysisRunner
from bayanalytics.laya.base import LayaClient
from bayanalytics.schemas.capabilities import Capabilities, ComponentHealth, Health
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
    research: Any = None  # research stack (provider, edgar, prices); opaque to the API layer
    extras: dict[str, Any] = field(default_factory=dict)
    _started: bool = False
    _health_cache: tuple[float, Health] | None = None

    async def start(self) -> None:
        if self._started:
            return
        await self.store.start()
        await self.runner.start()
        # Laya is the frequent decision layer: keep it resident for the process lifetime.
        info = await self.laya.load()
        self.extras["laya_load"] = info.model_dump()
        log.info("laya loaded in %.0f ms", info.load_ms)
        await self.spark.start()
        voice = self.transcriber.available()
        log.info("voice input %s", "available" if voice else "disabled")
        self.extras["voice_available_at_start"] = voice
        self._started = True

    async def close(self) -> None:
        # Stop accepting new jobs, cancel/finish active work, then close runtimes.
        await self.runner.shutdown()
        for name, closer in (("spark", self.spark.close), ("laya", self.laya.close)):
            try:
                await closer()
            except Exception:  # pragma: no cover - best effort shutdown
                log.exception("error closing %s", name)
        try:
            await self.transcriber.close()
        except Exception:  # pragma: no cover
            log.exception("error closing transcriber")
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
        components.append(ComponentHealth(name="store", status="ok"))
        overall = "ok"
        if any(c.status == "down" for c in components):
            overall = "degraded"
        return Health(
            status=overall,
            version=version,
            components=components,
            active_analyses=self.runner.active_count,
        )
