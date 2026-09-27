"""Spark runtime boundary. Implementations: llama-server backed client and ``MockSpark``.

One Spark request at a time. The client owns the Fast/Deep profile transition under its own
lock: acquire -> restart llama-server if the loaded profile differs -> generate -> release.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from bayanalytics.context import AnalysisContext
from bayanalytics.schemas.capabilities import ProfileCapability
from bayanalytics.schemas.common import Profile

TokenCallback = Callable[[str], Awaitable[None]]


class SparkMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ProfileSpec(BaseModel):
    name: Profile
    context_ceiling: int
    kv_cache_type: str
    min_available_mb: int


class SparkStreamStats(BaseModel):
    profile: Profile
    context_ceiling: int
    kv_cache_type: str
    load_ms: float | None = None
    time_to_first_token_ms: float | None = None
    total_ms: float
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    tokens_per_second: float | None = None
    resident_rss_mb: float | None = None
    peak_rss_mb: float | None = None
    finish_reason: str | None = None
    runtime_version: str | None = None


class SparkGeneration(BaseModel):
    text: str
    stats: SparkStreamStats
    truncated: bool = False


class SparkRunOptions(BaseModel):
    max_tokens: int = 1400
    temperature: float = 0.2
    top_p: float = 0.9
    stop: list[str] = Field(default_factory=list)


class SparkClient(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    def availability(self, profile: Profile) -> ProfileCapability: ...

    def profile_spec(self, profile: Profile) -> ProfileSpec: ...

    async def run(
        self,
        profile: Profile,
        messages: list[SparkMessage],
        on_token: TokenCallback,
        ctx: AnalysisContext,
        options: SparkRunOptions | None = None,
    ) -> SparkGeneration: ...
