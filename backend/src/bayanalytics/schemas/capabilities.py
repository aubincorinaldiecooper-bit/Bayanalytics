"""Health and capability responses (AGENT.md section 37.6)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from bayanalytics.schemas.common import ErrorCode


class ProfileCapability(BaseModel):
    available: bool
    context_ceiling: int
    reason: str | None = None
    code: ErrorCode | None = None


class Capabilities(BaseModel):
    profiles: dict[str, ProfileCapability]
    voice: bool
    deployment: str
    research: bool = True


class ComponentHealth(BaseModel):
    name: str
    status: str  # ok | degraded | down | disabled
    detail: str | None = None


class Health(BaseModel):
    status: str
    version: str
    components: list[ComponentHealth] = Field(default_factory=list)
    active_analyses: int = 0
