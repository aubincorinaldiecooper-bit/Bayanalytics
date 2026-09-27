"""Request/response bodies for the analysis API (AGENT.md section 37)."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from bayanalytics.schemas.common import AnalysisStatus, Horizon, Profile, ResolvedHorizon
from bayanalytics.schemas.results import InstrumentView


class InstrumentRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str = Field(min_length=1, max_length=16)
    exchange: str | None = Field(default=None, max_length=32)

    @field_validator("symbol")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper()


class CreateAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    instrument: InstrumentRef | None = None
    profile: Profile = "fast"
    horizon: Horizon = "auto"

    @field_validator("query")
    @classmethod
    def _strip(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped


class CreateAnalysisResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    profile: Profile
    resolved_horizon: ResolvedHorizon


class CancelAnalysisResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    cancel_requested: bool


class AnalysisSummary(BaseModel):
    """One row of the history list: everything a sidebar entry renders, nothing more.

    Built entirely from the persisted job record, so listing a page never reads results,
    sources or events.
    """

    analysis_id: str
    query: str
    instrument: InstrumentView | None = None
    profile: Profile
    horizon: ResolvedHorizon
    status: AnalysisStatus
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None
    error_code: str | None = None


class AnalysisListResponse(BaseModel):
    """Page of analyses, newest first.

    ``next_cursor`` is opaque: pass it back as ``?cursor=`` for the following page, or
    ``None`` when the list is exhausted. Keeping the page in an envelope (rather than a bare
    array) is what lets a later ``owner_id`` filter be added without changing this shape.
    """

    analyses: list[AnalysisSummary] = Field(default_factory=list)
    next_cursor: str | None = None
