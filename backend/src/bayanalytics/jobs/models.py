"""Durable analysis job state (AGENT.md section 38)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from bayanalytics.instruments.base import InstrumentIdentity, ResearchBudget
from bayanalytics.schemas.common import (
    TERMINAL_STATUSES,
    AnalysisStatus,
    Profile,
    ResolvedHorizon,
    utcnow,
)
from bayanalytics.schemas.errors import ErrorPayload
from bayanalytics.schemas.requests import InstrumentRef


class AnalysisJob(BaseModel):
    analysis_id: str
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    query: str
    instrument_ref: InstrumentRef | None = None
    instrument: InstrumentIdentity | None = None
    profile: Profile
    requested_horizon: str
    resolved_horizon: ResolvedHorizon
    as_of: datetime = Field(default_factory=utcnow)
    research_budget: ResearchBudget = Field(default_factory=ResearchBudget)
    source_ids: list[str] = Field(default_factory=list)
    normalization_version: str = ""
    laya_schema_version: str = ""
    spark_artifact: str = ""
    spark_runtime: str | None = None
    status: AnalysisStatus = "queued"
    error: ErrorPayload | None = None
    cancel_requested: bool = False
    last_seq: int = 0
    telemetry: dict[str, Any] = Field(default_factory=dict)

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def touch(self) -> None:
        self.updated_at = utcnow()
