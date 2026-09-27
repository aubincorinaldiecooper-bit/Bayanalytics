"""Evidence, provenance and normalized-fact schemas (AGENT.md sections 5, 6, 22, 32, 34)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

from bayanalytics.schemas.common import (
    Basis,
    Freshness,
    PriceType,
    Redistribution,
    SourceType,
    is_primary_source,
    source_rank,
)

PeriodKind = Literal[
    "fiscal_quarter",
    "fiscal_year",
    "ttm",
    "ytd",
    "qtd",
    "instant",
    "calendar_range",
    "unknown",
]


class Period(BaseModel):
    """A labelled reporting or observation period. Never compare periods of different kinds
    without labelling the difference."""

    kind: PeriodKind = "unknown"
    fiscal_year: int | None = None
    fiscal_period: str | None = None  # "FY", "Q1".."Q4", "H1", "H2"
    start: date | None = None
    end: date | None = None
    label: str = ""

    def key(self) -> str:
        return f"{self.kind}|{self.fiscal_year}|{self.fiscal_period}|{self.start}|{self.end}"


class SourceRecord(BaseModel):
    """Provenance for one retrieved item. Stored for every source, kept or rejected."""

    source_id: str
    url: str
    title: str
    publisher: str | None = None
    source_type: SourceType = "unverified_web"
    published_at: datetime | None = None
    retrieved_at: datetime
    symbol: str | None = None
    fiscal_period: str | None = None
    excerpt: str = ""  # short evidence excerpt, never a full article
    content_hash: str | None = None
    extraction_method: str = "unknown"
    freshness: Freshness = "unknown"
    redistribution: Redistribution = "unknown"
    terms_note: str | None = None
    rejected_reason: str | None = None
    research_intent: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_primary(self) -> bool:
        return is_primary_source(self.source_type)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def rank(self) -> int:
        return source_rank(self.source_type)

    def public_view(self) -> dict[str, Any]:
        """The fields the SSE stream and the result expose for a source."""
        return {
            "source_id": self.source_id,
            "title": self.title,
            "publisher": self.publisher,
            "source_type": self.source_type,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "retrieved_at": self.retrieved_at.isoformat(),
            "url": self.url,
            "fiscal_period": self.fiscal_period,
            "freshness": self.freshness,
            "is_primary": self.is_primary,
        }


class NormalizedFact(BaseModel):
    """One normalized numerical fact with full provenance."""

    fact_id: str
    metric: str  # e.g. revenue, net_income, eps_diluted, operating_cash_flow, capex
    value: float
    unit: str  # USD, USD_per_share, shares, percent, ratio, days
    currency: str | None = None
    period: Period = Field(default_factory=Period)
    basis: Basis = "unknown"
    source_id: str
    raw_value: str | None = None
    extraction_method: str = "unknown"
    published_at: datetime | None = None
    restated: bool = False
    original_value: float | None = None
    original_source_id: str | None = None
    notes: list[str] = Field(default_factory=list)


class ConflictValue(BaseModel):
    value: float
    basis: Basis = "unknown"
    unit: str | None = None
    source_id: str
    published_at: datetime | None = None
    period_label: str | None = None


ConflictReason = Literal[
    "definition_mismatch",
    "basis_mismatch",
    "period_mismatch",
    "currency_mismatch",
    "restatement",
    "split_adjustment",
    "unknown",
]


class Conflict(BaseModel):
    """Two or more credible sources disagree. Preserved, never normalized away."""

    metric: str
    period_label: str | None = None
    status: Literal["conflict", "resolved_by_primary", "unresolved"] = "conflict"
    reason: ConflictReason = "unknown"
    values: list[ConflictValue]
    material: bool = False
    note: str | None = None


class PricePoint(BaseModel):
    date: date
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float
    volume: float | None = None


class PriceSeries(BaseModel):
    symbol: str
    source_id: str
    points: list[PricePoint]
    price_type: PriceType = "historical_close"
    exchange: str | None = None
    exchange_timezone: str = "America/New_York"
    session_date: date | None = None
    retrieved_at: datetime
    split_adjusted: bool = True
    dividend_adjusted: bool = False
    currency: str = "USD"
    label: str = ""

    @property
    def latest(self) -> PricePoint | None:
        return self.points[-1] if self.points else None


class EventSegment(BaseModel):
    """A compact, Laya-sized slice of company history (AGENT.md section 3.4)."""

    segment_id: str
    period: Period
    summary: dict[str, Any] = Field(default_factory=dict)
    source_ids: list[str] = Field(default_factory=list)
    laya: dict[str, Any] = Field(default_factory=dict)  # filled by scoring


class CorporateAction(BaseModel):
    kind: Literal[
        "split",
        "reverse_split",
        "ticker_change",
        "merger",
        "acquisition",
        "spin_off",
        "share_class_change",
        "dividend",
        "restatement",
        "fiscal_year_change",
        "accounting_policy_change",
    ]
    effective: date | None = None
    detail: str = ""
    source_id: str | None = None


class BenchmarkRef(BaseModel):
    role: Literal["broad_market", "sector"]
    symbol: str
    name: str
    reason: str = ""


class NormalizedEvidence(BaseModel):
    """Everything the wrapper hands to Laya, the calculators and the Spark bundle builder."""

    symbol: str
    as_of: datetime
    facts: list[NormalizedFact] = Field(default_factory=list)
    prices: PriceSeries | None = None
    benchmarks: dict[str, PriceSeries] = Field(default_factory=dict)
    benchmark_refs: list[BenchmarkRef] = Field(default_factory=list)
    sources: list[SourceRecord] = Field(default_factory=list)
    conflicts: list[Conflict] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    segments: list[EventSegment] = Field(default_factory=list)
    corporate_actions: list[CorporateAction] = Field(default_factory=list)
    text_evidence: list[dict[str, Any]] = Field(default_factory=list)  # {source_id, fact, ...}
    freshness_summary: dict[str, Any] = Field(default_factory=dict)
    normalization_version: str = ""

    def source_by_id(self, source_id: str) -> SourceRecord | None:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        return None

    def facts_for(self, metric: str) -> list[NormalizedFact]:
        return [fact for fact in self.facts if fact.metric == metric]
