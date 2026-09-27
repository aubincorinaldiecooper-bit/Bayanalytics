"""Build the process runtime from settings (mock, fixture and real runtimes share one shape)."""

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
from bayanalytics.laya.schemas import LAYA_SCHEMA_VERSION
from bayanalytics.laya.wrapper import LayaFinanceWrapper
from bayanalytics.normalization import NORMALIZATION_VERSION
from bayanalytics.pipeline.orchestrator import run_analysis
from bayanalytics.research.edgar import load_company_tickers_seed
from bayanalytics.research.http_provider import build_research_stack
from bayanalytics.runtime import Runtime
from bayanalytics.spark.base import SparkClient
from bayanalytics.spark.profiles import read_lockfile
from bayanalytics.store import build_store
from bayanalytics.whisper import build_transcriber

log = logging.getLogger(__name__)


def build_laya(settings: Settings) -> LayaClient:
    if settings.laya_mode == "mock":
        from bayanalytics.laya.mock import MockLaya

        return MockLaya()
    from bayanalytics.laya.client import LayaWorkerClient

    return LayaWorkerClient(settings)


def build_spark(settings: Settings) -> SparkClient:
    if settings.spark_mode == "mock":
        from bayanalytics.spark.mock import MockSpark

        return MockSpark()
    from bayanalytics.spark.client import LlamaSparkClient

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
    _provider, edgar, _prices = research
    wrapper = LayaFinanceWrapper(laya, LAYA_SCHEMA_VERSION)

    async def resolver_factory() -> InstrumentResolver:
        rows: list[dict] = []
        try:
            rows = await edgar.company_tickers()
        except Exception as exc:  # network unavailable: fall back to the bundled seed
            log.warning("company ticker list unavailable (%s); using seed", type(exc).__name__)
        if not rows:
            rows = load_company_tickers_seed()
        return InstrumentResolver(rows)

    def analyzer_factory() -> EquityAnalyzer:
        return EquityAnalyzer(settings, research, wrapper, resolver_factory)

    spark_lock = read_lockfile(settings.spark_lockfile) if settings.spark_lockfile else None
    versions = {
        "normalization_version": NORMALIZATION_VERSION,
        "laya_schema_version": LAYA_SCHEMA_VERSION,
        "spark_artifact": settings.spark_artifact,
        "spark_runtime": (spark_lock or {}).get("llama_cpp_version"),
    }
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
        extras={"analyzer_factory": analyzer_factory, "spark_lock": spark_lock or {}},
    )
    runtime.runner = AnalysisRunner(
        store,
        bus,
        functools.partial(run_analysis, rt=runtime),
        versions=versions,
        max_active=settings.max_active_analyses,
    )
    return runtime
