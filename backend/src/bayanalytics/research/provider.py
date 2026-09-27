"""Provider-neutral web research boundary (AGENT.md section 21).

Web research means only: search -> open/fetch -> extract -> return source + content + metadata.
The finance layer owns strategy, gaps, Laya decisions, normalization and provenance. A provider
owns only access to web evidence. Retrieved content is evidence, never instruction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from bayanalytics.schemas.common import SourceType


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str = ""
    engine: str | None = None
    published_at: datetime | None = None
    score: float | None = None
    source_type_hint: SourceType | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PageResult(BaseModel):
    url: str
    final_url: str
    status: int
    content_type: str | None = None
    body: str = ""  # decoded text body (HTML, JSON, CSV or plain text), size-capped
    fetched_at: datetime
    headers: dict[str, str] = Field(default_factory=dict)
    from_cache: bool = False


class EvidenceRecord(BaseModel):
    """Extracted evidence for one URL: metadata + short text, never the whole page."""

    url: str
    final_url: str
    title: str
    publisher: str | None = None
    source_type: SourceType = "unverified_web"
    published_at: datetime | None = None
    retrieved_at: datetime
    text: str = ""  # extracted main text, capped by the provider
    excerpt: str = ""  # short lead excerpt safe to store and display
    content_hash: str
    extraction_method: str
    language: str | None = None
    structured: dict[str, Any] = Field(default_factory=dict)  # parsed JSON/CSV payloads
    metadata: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ResearchProvider(Protocol):
    async def search(self, query: str) -> list[SearchResult]: ...

    async def open(self, url: str) -> PageResult: ...

    async def extract(self, url: str) -> EvidenceRecord: ...


class ResearchProviderError(Exception):
    """Raised by providers when web access itself fails (network, backend down, blocked)."""
