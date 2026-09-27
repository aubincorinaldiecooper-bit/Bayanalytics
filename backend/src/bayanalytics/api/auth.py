"""Optional API-key gate (AGENT.md section 20).

Local MVP: loopback binding, no key. Anything else: ``BAY_API_KEY`` must be set and every
request must carry it as ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``. The cloud
auth provider is intentionally deferred; this is the minimum that keeps the API from being
exposed without any credential.
"""

from __future__ import annotations

import hmac

from fastapi import Request

from bayanalytics.errors import AnalysisError
from bayanalytics.schemas.common import ErrorCode


def _presented_key(request: Request) -> str | None:
    header = request.headers.get("authorization")
    if header and header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("x-api-key")


async def require_api_key(request: Request) -> None:
    settings = request.app.state.runtime.settings
    expected = settings.api_key
    if not expected:
        return
    presented = _presented_key(request)
    if not presented or not hmac.compare_digest(presented, expected):
        raise AnalysisError(ErrorCode.UNAUTHORIZED)
