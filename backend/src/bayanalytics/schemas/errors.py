"""Error envelope returned by the API and carried inside ``analysis.failed`` events."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from bayanalytics.schemas.common import ErrorCode


class ErrorPayload(BaseModel):
    code: ErrorCode
    message: str
    retryable: bool = False
    details: dict[str, Any] | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorPayload
