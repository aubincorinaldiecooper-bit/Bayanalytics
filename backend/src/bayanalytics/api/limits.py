"""Request body limits enforced before a body is buffered."""

from __future__ import annotations

import json

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.errors import ErrorEnvelope, ErrorPayload

_BODY_METHODS = {"POST", "PUT", "PATCH"}


class BodyLimitMiddleware:
    """Reject over-sized bodies by ``Content-Length`` (413) and require a length for bodies
    at all (411), so neither JSON nor multipart uploads are spooled before validation."""

    def __init__(
        self, app: ASGIApp, *, default_limit: int, upload_limit: int, upload_path_suffix: str
    ) -> None:
        self.app = app
        self.default_limit = default_limit
        self.upload_limit = upload_limit
        self.upload_path_suffix = upload_path_suffix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") not in _BODY_METHODS:
            await self.app(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])
        }
        path = scope.get("path", "")
        limit = (
            self.upload_limit
            if path.rstrip("/").endswith(self.upload_path_suffix)
            else self.default_limit
        )
        length = headers.get("content-length")
        if length is None:
            if headers.get("transfer-encoding", "").lower() == "chunked":
                await self._reject(send, 411, "Request bodies must declare Content-Length.")
                return
            await self.app(scope, receive, send)
            return
        try:
            size = int(length)
        except ValueError:
            await self._reject(send, 400, "Invalid Content-Length.")
            return
        if size > limit:
            await self._reject(send, 413, f"Request body exceeds the {limit} byte limit.")
            return
        await self.app(scope, receive, send)

    @staticmethod
    async def _reject(send: Send, status: int, message: str) -> None:
        payload = ErrorEnvelope(
            error=ErrorPayload(code=ErrorCode.INVALID_REQUEST, message=message, retryable=False)
        )
        body = json.dumps(payload.model_dump(mode="json")).encode()
        start: Message = {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
        await send(start)
        await send({"type": "http.response.body", "body": body})
