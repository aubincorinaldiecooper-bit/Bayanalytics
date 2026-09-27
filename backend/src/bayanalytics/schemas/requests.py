"""Request/response bodies for the analysis API (AGENT.md section 37)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from bayanalytics.schemas.common import AnalysisStatus, Horizon, Profile, ResolvedHorizon


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
