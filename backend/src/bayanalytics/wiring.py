"""Build the process runtime from settings.

Every component is the real implementation (Node Laya worker, llama-server client, whisper.cpp
CLI or disabled voice, HTTP research stack, in-memory or Postgres store). Tests inject their
own doubles through the keyword arguments of ``build_runtime``; nothing here selects a mock.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from bayanalytics.config import Settings
from bayanalytics.instruments.equity import EquityAnalyzer
from bayanalytics.instruments.identity import InstrumentResolver
from bayanalytics.jobs.bus import AnalysisEventBus
from bayanalytics.jobs.runner import AnalysisRunner
from bayanalytics.laya.base import LayaClient
from bayanalytics.laya.client import LayaWorkerClient
from bayanalytics.laya.schemas import LAYA_SCHEMA_VERSION
from bayanalytics.laya.wrapper import LayaFinanceWrapper
from bayanalytics.normalization import NORMALIZATION_VERSION
from bayanalytics.pipeline.orchestrator import run_analysis
from bayanalytics.research.http_provider import build_research_stack
from bayanalytics.runtime import Runtime
from bayanalytics.spark.base import SparkClient
from bayanalytics.spark.client import LlamaSparkClient
from bayanalytics.spark.profiles import read_lockfile
from bayanalytics.store import build_store
from bayanalytics.whisper import build_transcriber

log = logging.getLogger(__name__)


def build_laya(settings: Settings) -> LayaClient:
    return LayaWorkerClient(settings)


def build_spark(settings: Settings) -> SparkClient:
    return LlamaSparkClient(settings)


def build_runtime(
    settings: Settings,
    *,
    store: Any | None = None,
    laya: LayaClient | None = None,
    spark: SparkClient | None = None,
    transcriber: Any | None = None,
    research: Any | None = None,
) -> Runtime:
    store = store or build_store(settings)
    laya = laya or build_laya(settings)
    spark = spark or build_spark(settings)
    transcriber = transcriber or build_transcriber(settings)
    research = research or build_research_stack(settings)
    wrapper = LayaFinanceWrapper(laya, LAYA_SCHEMA_VERSION)

    async def resolver_factory() -> InstrumentResolver:
        """The instrument is the ticker in the request: no directory, no network."""
        return InstrumentResolver()

    def analyzer_factory() -> EquityAnalyzer:
        return EquityAnalyzer(settings, research, wrapper, resolver_factory)

    spark_lock = read_lockfile(settings.spark_lockfile) if settings.spark_lockfile else None
    bus = AnalysisEventBus(store)
    runtime = Runtime(
        settings=settings,
        store=store,
        bus=bus,
        runner=None,  # type: ignore[arg-type]  # set below (needs the runtime for the pipeline)
        laya=laya,
        spark=spark,
        transcriber=transcriber,
        research=research,
        extras={
            "analyzer_factory": analyzer_factory,
            "resolver_factory": resolver_factory,
            "spark_lock": spark_lock or {},
        },
    )

    def versions() -> dict[str, Any]:
        """Schema versions are code constants; runtime versions are whatever the probes at
        startup measured (``Runtime.start``), never configured labels."""
        spark_version = runtime.extras.get("spark_version") or {}
        return {
            "normalization_version": NORMALIZATION_VERSION,
            "laya_schema_version": LAYA_SCHEMA_VERSION,
            "spark_artifact": spark_version.get("spark_artifact"),
            "spark_runtime": spark_version.get("spark_runtime"),
        }

    runtime.runner = AnalysisRunner(
        store,
        bus,
        functools.partial(run_analysis, rt=runtime),
        versions=versions,
        max_active=settings.max_active_analyses,
    )
    return runtime
