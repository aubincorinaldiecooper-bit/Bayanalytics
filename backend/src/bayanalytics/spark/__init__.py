"""Spark synthesis layer: llama-server client, mock, prompt, bundle fitting and parsing."""

from bayanalytics.spark.base import (
    ProfileSpec,
    SparkClient,
    SparkGeneration,
    SparkMessage,
    SparkRunOptions,
    SparkStreamStats,
    TokenCallback,
)
from bayanalytics.spark.bundle import (
    bundle_tokens,
    estimate_tokens,
    fit_bundle,
    reserved_output_tokens,
    system_tokens,
)
from bayanalytics.spark.client import LlamaSparkClient
from bayanalytics.spark.manager import LlamaServerManager, LoadOutcome
from bayanalytics.spark.mock import MockSpark
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
    "SECTION_HEADINGS",
    "SYSTEM_PROMPT",
    "LlamaServerManager",
    "LlamaSparkClient",
    "LoadOutcome",
    "MemorySnapshot",
    "MockSpark",
    "ProfileSpec",
    "SparkClient",
    "SparkGeneration",
    "SparkMessage",
    "SparkRunOptions",
    "SparkStreamStats",
    "TokenCallback",
    "assert_can_allocate",
    "build_messages",
    "bundle_tokens",
    "check_availability",
    "conflict_notes",
    "estimate_tokens",
    "extract_citations",
    "fit_bundle",
    "parse_bullets",
    "parse_sections",
    "parse_stance",
    "profile_specs",
    "psutil_probe",
    "read_lockfile",
    "render_bundle",
    "reserved_output_tokens",
    "system_tokens",
    "to_assessment",
    "version_fields",
]


def make_spark_client(settings, **kwargs):  # type: ignore[no-untyped-def]
    """``MockSpark`` for ``spark_mode == "mock"``, otherwise ``LlamaSparkClient``."""
    if settings.spark_mode == "mock":
        return MockSpark(settings, **kwargs)
    return LlamaSparkClient(settings, **kwargs)
