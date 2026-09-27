"""Exception handlers producing the structured error envelope for every failure."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from bayanalytics.errors import AnalysisError
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.errors import ErrorEnvelope, ErrorPayload

log = logging.getLogger(__name__)


def _envelope(payload: ErrorPayload, status: int) -> JSONResponse:
    return JSONResponse(
        status_code=status, content=ErrorEnvelope(error=payload).model_dump(mode="json")
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AnalysisError)
    async def _analysis_error(_request: Request, exc: AnalysisError) -> JSONResponse:
        response = _envelope(exc.payload(), exc.http_status)
        if exc.code == ErrorCode.TOO_MANY_ANALYSES:
            response.headers["Retry-After"] = "5"
        return response

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        details = {
            "errors": [{"loc": list(e.get("loc", ())), "msg": e.get("msg")} for e in exc.errors()]
        }
        payload = ErrorPayload(
            code=ErrorCode.INVALID_REQUEST,
            message="The request was invalid.",
            retryable=False,
            details=details,
        )
        return _envelope(payload, 422)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = ErrorCode.NOT_FOUND if exc.status_code == 404 else ErrorCode.INVALID_REQUEST
        if exc.status_code >= 500:
            code = ErrorCode.INTERNAL_ERROR
        payload = ErrorPayload(code=code, message=str(exc.detail), retryable=exc.status_code >= 500)
        return _envelope(payload, exc.status_code)

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        payload = AnalysisError.from_exception(exc).payload()
        return _envelope(payload, 500)
