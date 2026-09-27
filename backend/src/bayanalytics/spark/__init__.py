"""Spark synthesis layer: llama-server client, prompt, bundle fitting and parsing."""

from typing import Any

from bayanalytics.config import Settings
from bayanalytics.spark.base import (
    ProfileSpec,
    SparkClient,
    SparkGeneration,
    SparkMessage,
    SparkRunOptions,
    SparkSession,
    SparkStreamStats,
    TokenCallback,
)
from bayanalytics.spark.bundle import (
    OVERFLOW_TRIM,
    FitResult,
    PromptCounter,
    fit_bundle,
    reserved_output_tokens,
)
from bayanalytics.spark.client import LlamaSparkClient
from bayanalytics.spark.manager import LlamaServerManager, LoadOutcome
from bayanalytics.spark.parse import (
    conflict_notes,
    extract_citations,
    parse_bullets,
    parse_sections,
    parse_stance,
    to_assessment,
)
from bayanalytics.spark.profiles import (
    MemorySnapshot,
    assert_can_allocate,
    check_availability,
    profile_specs,
    psutil_probe,
    read_lockfile,
    version_fields,
)
from bayanalytics.spark.prompt import SECTION_HEADINGS, SYSTEM_PROMPT, build_messages, render_bundle

__all__ = [
    "OVERFLOW_TRIM",
    "SECTION_HEADINGS",
    "SYSTEM_PROMPT",
    "FitResult",
    "LlamaServerManager",
    "LlamaSparkClient",
    "LoadOutcome",
    "MemorySnapshot",
    "ProfileSpec",
    "PromptCounter",
    "SparkClient",
    "SparkGeneration",
    "SparkMessage",
    "SparkRunOptions",
    "SparkSession",
    "SparkStreamStats",
    "TokenCallback",
    "assert_can_allocate",
    "build_messages",
    "check_availability",
    "conflict_notes",
    "extract_citations",
    "fit_bundle",
    "make_spark_client",
    "parse_bullets",
    "parse_sections",
    "parse_stance",
    "profile_specs",
    "psutil_probe",
    "read_lockfile",
    "render_bundle",
    "reserved_output_tokens",
    "to_assessment",
    "version_fields",
]


def make_spark_client(settings: Settings, **kwargs: Any) -> SparkClient:
    return LlamaSparkClient(settings, **kwargs)
