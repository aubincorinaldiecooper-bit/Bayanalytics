"""Health and capability responses (AGENT.md section 37.6)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.results import ExecutionInfo


class ProfileCapability(BaseModel):
    available: bool
    context_ceiling: int
    reason: str | None = None
    code: ErrorCode | None = None


class MarketCapability(BaseModel):
    price_display: bool = False


class Capabilities(BaseModel):
    profiles: dict[str, ProfileCapability]
    voice: bool
    deployment: str
    research: bool = True
    web_search: bool = False  # a search backend is configured (BAY_RESEARCH_SEARCH_URL)
    execution: ExecutionInfo = Field(default_factory=ExecutionInfo)
    market: MarketCapability = Field(default_factory=MarketCapability)


class ComponentHealth(BaseModel):
    name: str
    status: str  # ok | degraded | down | disabled
    detail: str | None = None


class Health(BaseModel):
    status: str
    version: str
    components: list[ComponentHealth] = Field(default_factory=list)
    active_analyses: int = 0
    execution: ExecutionInfo = Field(default_factory=ExecutionInfo)
