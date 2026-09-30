"""Instrument adapter boundary (AGENT.md section 23). MVP implements ``EquityAnalyzer`` only."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from bayanalytics.context import AnalysisContext
from bayanalytics.schemas.calculations import CalculationResult
from bayanalytics.schemas.common import Profile, ResolvedHorizon
from bayanalytics.schemas.decisions import LayaDecision, LayaQuestionSet
from bayanalytics.schemas.evidence import NormalizedEvidence, SourceRecord
from bayanalytics.schemas.questions import AnalyticalRequirements


class InstrumentCandidate(BaseModel):
    symbol: str
    exchange: str | None = None
    name: str = ""
    cik: str | None = None
    score: float = 0.0


class InstrumentIdentity(BaseModel):
    """The instrument as the analyst named it (``instruments.identity``). ``name``, ``cik``
    and ``sector`` stay empty: there is no company directory, only the ticker."""

    symbol: str
    exchange: str | None = None
    name: str = ""
    cik: str | None = None
    instrument_type: Literal["equity"] = "equity"
    sector: str | None = None
    exchange_timezone: str = "America/New_York"
    confidence: float = 1.0
    resolution_method: str = ""


class ResearchBudget(BaseModel):
    max_rounds: int = 4
    max_sources: int = 24
    max_fetch_per_round: int = 6
    timeout_s: float = 240.0


class AnalysisRequest(BaseModel):
    """Internal, immutable snapshot of what an analysis asked for."""

    analysis_id: str
    query: str
    profile: Profile
    requested_horizon: str
    resolved_horizon: ResolvedHorizon
    as_of: datetime
    budget: ResearchBudget = Field(default_factory=ResearchBudget)
    # What the interpreted question needs (pipeline/understanding.py, pipeline/questions.py,
    # instruments/questions.py); None until the question was interpreted.
    requirements: AnalyticalRequirements | None = None


class LayaDecisions(BaseModel):
    decisions: list[LayaDecision] = Field(default_factory=list)

    def by_type(self, decision_type: str) -> list[LayaDecision]:
        return [d for d in self.decisions if d.decision_type == decision_type]

    def latest(self, decision_type: str) -> LayaDecision | None:
        matches = self.by_type(decision_type)
        return matches[-1] if matches else None


class CalculatedMetrics(BaseModel):
    calculations: list[CalculationResult] = Field(default_factory=list)

    def by_name(self, name: str) -> CalculationResult | None:
        for calc in self.calculations:
            if calc.name == name and calc.status == "computed":
                return calc
        return None


class SparkEvidenceBundle(BaseModel):
    """Strict structured handoff to Spark (AGENT.md section 16, phase 3)."""

    instrument: dict[str, Any] = Field(default_factory=dict)
    request: dict[str, Any] = Field(default_factory=dict)
    current_metrics: dict[str, Any] = Field(default_factory=dict)
    historical_metrics: dict[str, Any] = Field(default_factory=dict)
    laya_assessments: dict[str, Any] = Field(default_factory=dict)
    important_events: list[dict[str, Any]] = Field(default_factory=list)
    historical_analogues: list[dict[str, Any]] = Field(default_factory=list)
    calculated_metrics: dict[str, Any] = Field(default_factory=dict)
    benchmark_context: dict[str, Any] = Field(default_factory=dict)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    excerpts: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    freshness: dict[str, Any] = Field(default_factory=dict)
    horizons: list[str] = Field(default_factory=list)
    prior_assessment: dict[str, Any] | None = None
    """Bounded, evidence-only comparison with the prior completed assessment of the same
    instrument (``pipeline.thesis.prior_assessment_block``); ``None`` when there is none."""
    # What the analyst asked (intent and requirement labels, focus sentence, horizons to
    # emphasise) and what the analysis could not meet, so the synthesis addresses the question;
    # empty for a general assessment with nothing narrower.
    question_focus: dict[str, Any] = Field(default_factory=dict)


class InstrumentAnalyzer(Protocol):
    async def identify(self, user_input: str, ctx: AnalysisContext) -> InstrumentIdentity: ...

    async def retrieve(
        self, identity: InstrumentIdentity, request: AnalysisRequest, ctx: AnalysisContext
    ) -> list[SourceRecord]: ...

    async def normalize(
        self, records: list[SourceRecord], ctx: AnalysisContext
    ) -> NormalizedEvidence: ...

    def build_laya_questions(
        self, evidence: NormalizedEvidence, request: AnalysisRequest
    ) -> list[LayaQuestionSet]: ...

    async def calculate(
        self, evidence: NormalizedEvidence, decisions: LayaDecisions, ctx: AnalysisContext
    ) -> CalculatedMetrics: ...

    def build_spark_bundle(
        self,
        evidence: NormalizedEvidence,
        decisions: LayaDecisions,
        calculations: CalculatedMetrics,
        request: AnalysisRequest,
    ) -> SparkEvidenceBundle: ...
