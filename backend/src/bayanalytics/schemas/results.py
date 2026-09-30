"""Final structured analysis result (AGENT.md sections 12 and 37.3)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from bayanalytics.schemas.calculations import CalculationResult
from bayanalytics.schemas.common import AnalysisStatus, Profile, ResolvedHorizon, Stance
from bayanalytics.schemas.decisions import LayaDecision
from bayanalytics.schemas.errors import ErrorPayload
from bayanalytics.schemas.evidence import Conflict, SourceRecord
from bayanalytics.schemas.questions import RequirementsReport


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
    queries_failed: int = 0  # search backend errors (the loop continued without them)
    structured_failures: int = 0  # EDGAR / price endpoint errors after the first success
    sources_fetched: int = 0
    sources_rejected: int = 0
    duplicate_sources_removed: int = 0
    evidence_gaps_remaining: int = 0
    retrieval_total_ms: float = 0.0
    intents: list[str] = Field(default_factory=list)
    termination_reason: str | None = None


class ExecutionInfo(BaseModel):
    """How this backend process is wired: the llama-server mode (managed by the backend or an
    external server), whether voice input is on, the deployment target and whether a web search
    backend is configured. Runtime versions are measured separately in ``VersionInfo``."""

    spark_mode: str = "managed"
    whisper_mode: str = "disabled"
    deployment: str = "local"
    search_configured: bool = False


class VersionInfo(BaseModel):
    normalization_version: str = ""
    laya_schema_version: str = ""
    # Measured at runtime: the worker reports its installed package version, the Spark
    # artifact comes from the download lockfile and the loaded model, the runtime from
    # ``llama-server --version`` or ``/props``. ``None`` means not reported, never a default.
    laya_package_version: str | None = None
    spark_artifact: str | None = None
    spark_runtime: str | None = None
    spark_gguf_sha256: str | None = None
    spark_hf_revision: str | None = None
    execution: ExecutionInfo = Field(default_factory=ExecutionInfo)


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

    # Spark pass 1 (query understanding), measured separately from the synthesis above.
    query_understanding_ms: float | None = None  # wall clock of the pass, lock wait included
    query_understanding_prompt_tokens: int | None = None  # llama-server usage
    query_understanding_output_tokens: int | None = None  # llama-server usage
    query_understanding_load_ms: float | None = None  # only when this pass loaded the profile
    query_understanding_wait_ms: float | None = None  # queued for the Spark lane (no load)
    query_understanding_generation_ms: float | None = None  # prompt processing + decoding

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


class StanceChange(BaseModel):
    """One stance compared with the prior assessment (``scope`` is ``overall`` or a horizon)."""

    scope: str
    previous: Stance | None = None
    current: Stance | None = None
    changed: bool = False  # both known and different
    compared_horizons: list[str] = Field(default_factory=list)
    """``overall`` scope only: the horizons both assessments covered. The overall stance is
    computed over these alone, so a run that covers fewer or other horizons is not reported
    as a change of thesis."""


class MetricChange(BaseModel):
    """A deterministic calculation present in both assessments: value then, value now."""

    name: str
    unit: str
    previous_value: float | None = None
    current_value: float | None = None
    delta: float | None = None  # current - previous, in the calculation's unit
    previous_display: str = ""
    current_display: str = ""
    previous_period: str | None = None
    current_period: str | None = None
    previous_calc_id: str | None = None
    current_calc_id: str | None = None


class FreshnessChange(BaseModel):
    """Did the information set move on since the prior assessment (new quarter, newer close)?"""

    previous_latest_quarter_end: str | None = None
    current_latest_quarter_end: str | None = None
    new_quarter: bool = False
    previous_price_date: str | None = None
    current_price_date: str | None = None
    newer_prices: bool = False


class ThesisDiff(BaseModel):
    """Structured comparison with the latest earlier completed assessment of the same
    instrument (AGENT.md section 12 "what changed"). Only structured fields are compared:
    stances, deterministic calculation values, conflicts, uncertainties and freshness; the
    prior narrative is never reused."""

    previous_analysis_id: str
    previous_as_of: datetime
    previous_created_at: datetime
    previous_horizon: ResolvedHorizon
    overall: StanceChange
    horizons: list[StanceChange] = Field(default_factory=list)
    metrics: list[MetricChange] = Field(default_factory=list)
    new_conflicts: list[str] = Field(default_factory=list)
    resolved_conflicts: list[str] = Field(default_factory=list)
    new_uncertainties: list[str] = Field(default_factory=list)
    resolved_uncertainties: list[str] = Field(default_factory=list)
    freshness: FreshnessChange = Field(default_factory=FreshnessChange)
    stance_changed: bool = False  # the overall stance or any shared horizon stance moved
    horizon_scope_changed: bool = False  # the runs assessed different sets of horizons
    summary: list[str] = Field(default_factory=list)  # deterministic one-line statements


PricePointRow = tuple[str, float | None, float | None, float | None, float, float | None]


class MarketSeries(BaseModel):
    """Daily price points for one symbol, as sent to clients when price display is enabled.

    ``points`` rows are ``[date, open, high, low, close, volume]``, oldest first."""

    role: Literal["company", "broad_market", "sector"]
    symbol: str
    name: str
    source_id: str
    currency: str = "USD"
    interval: Literal["1d"] = "1d"
    points: list[PricePointRow] = Field(default_factory=list)


class FundamentalQuarter(BaseModel):
    label: str
    end: str  # ISO date
    revenue: float | None = None
    gross_margin_pct: float | None = None


class MarketFundamentals(BaseModel):
    """Quarterly revenue and gross margin from the normalized SEC facts, oldest first."""

    currency: str = "USD"
    quarters: list[FundamentalQuarter] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class MarketView(BaseModel):
    price_display: bool = False
    series: list[MarketSeries] = Field(default_factory=list)
    fundamentals: MarketFundamentals | None = None


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
    thesis_diff: ThesisDiff | None = None  # None when no prior completed assessment exists
    # How the question was interpreted (product labels) and which of its requirements were
    # met; None when the analysis stopped before the question was interpreted.
    requirements: RequirementsReport | None = None
    streamed_text: str = ""  # exactly what was streamed as spark.token, for recovery
    market: MarketView = Field(default_factory=MarketView)
    telemetry: Telemetry = Field(default_factory=Telemetry)
    error: ErrorPayload | None = None
    partial: bool = False  # true when cancelled/failed with some content preserved
