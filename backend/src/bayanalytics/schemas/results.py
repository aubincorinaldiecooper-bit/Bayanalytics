"""Final structured analysis result (AGENT.md sections 12 and 37.3)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from bayanalytics.schemas.calculations import CalculationResult
from bayanalytics.schemas.common import AnalysisStatus, Profile, ResolvedHorizon, Stance
from bayanalytics.schemas.decisions import LayaDecision
from bayanalytics.schemas.errors import ErrorPayload
from bayanalytics.schemas.evidence import Conflict, SourceRecord


class InstrumentView(BaseModel):
    symbol: str
    exchange: str | None = None
    name: str
    cik: str | None = None
    sector: str | None = None
    instrument_type: str = "equity"


class EvidenceItem(BaseModel):
    """A claim tied to the sources that support it."""

    text: str
    source_ids: list[str] = Field(default_factory=list)
    stance: Stance | None = None
    metric: str | None = None
    period_label: str | None = None
    decision_id: str | None = None
    calc_id: str | None = None


class Assessment(BaseModel):
    summary: str = ""
    what_changed: list[EvidenceItem] = Field(default_factory=list)
    fundamentals: dict[str, Any] = Field(default_factory=dict)
    valuation: dict[str, Any] = Field(default_factory=dict)
    benchmark_context: dict[str, Any] = Field(default_factory=dict)
    historical_context: dict[str, Any] = Field(default_factory=dict)
    market_context: dict[str, Any] = Field(default_factory=dict)
    bull_evidence: list[EvidenceItem] = Field(default_factory=list)
    bear_evidence: list[EvidenceItem] = Field(default_factory=list)
    risks: list[EvidenceItem] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


class HorizonAssessment(BaseModel):
    horizon: str
    stance: Stance = "mixed"
    confidence: float = 0.0  # confidence in the Laya decision, never an outcome probability
    summary: str = ""
    key_evidence: list[EvidenceItem] = Field(default_factory=list)
    decision_id: str | None = None
    synthesized: bool = True  # False when Spark produced no section for this horizon
    low_confidence: bool = False  # Laya's stance confidence was below the floor


class ResearchStats(BaseModel):
    search_rounds: int = 0
    queries_issued: int = 0
    sources_fetched: int = 0
    sources_rejected: int = 0
    duplicate_sources_removed: int = 0
    evidence_gaps_remaining: int = 0
    retrieval_total_ms: float = 0.0
    intents: list[str] = Field(default_factory=list)
    termination_reason: str | None = None


class VersionInfo(BaseModel):
    normalization_version: str = ""
    laya_schema_version: str = ""
    laya_package_version: str = ""
    spark_artifact: str = ""
    spark_runtime: str | None = None
    spark_gguf_sha256: str | None = None
    spark_hf_revision: str | None = None


class Telemetry(BaseModel):
    """Measured, never estimated. Fields stay ``None`` when a runtime did not report them."""

    profile: Profile | None = None
    context_ceiling: int | None = None
    kv_cache_type: str | None = None

    retrieval_ms: float | None = None
    normalization_ms: float | None = None
    laya_ms: float | None = None
    math_ms: float | None = None
    spark_load_ms: float | None = None
    spark_time_to_first_token_ms: float | None = None
    spark_total_ms: float | None = None
    total_request_ms: float | None = None

    laya_load_ms: float | None = None
    laya_warm_inference_ms: float | None = None
    whisper_load_ms: float | None = None

    spark_prompt_tokens: int | None = None
    spark_output_tokens: int | None = None
    spark_tokens_per_second: float | None = None

    process_peak_rss_mb: float | None = None
    laya_resident_ram_mb: float | None = None
    laya_peak_rss_mb: float | None = None
    spark_resident_ram_mb: float | None = None
    spark_peak_rss_mb: float | None = None
    whisper_peak_rss_mb: float | None = None
    system_total_ram_mb: float | None = None
    system_peak_ram_mb: float | None = None
    swap_used_mb: float | None = None
    cpu_percent: float | None = None

    research: ResearchStats = Field(default_factory=ResearchStats)
    versions: VersionInfo = Field(default_factory=VersionInfo)


class AnalysisResult(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    query: str
    instrument: InstrumentView | None = None
    profile: Profile
    horizon: ResolvedHorizon
    as_of: datetime
    created_at: datetime
    completed_at: datetime | None = None
    assessment: Assessment = Field(default_factory=Assessment)
    horizon_assessments: dict[str, HorizonAssessment] = Field(default_factory=dict)
    sources: list[SourceRecord] = Field(default_factory=list)
    calculations: list[CalculationResult] = Field(default_factory=list)
    laya_decisions: list[LayaDecision] = Field(default_factory=list)
    freshness_summary: dict[str, Any] = Field(default_factory=dict)
    streamed_text: str = ""  # exactly what was streamed as spark.token, for recovery
    telemetry: Telemetry = Field(default_factory=Telemetry)
    error: ErrorPayload | None = None
    partial: bool = False  # true when cancelled/failed with some content preserved
