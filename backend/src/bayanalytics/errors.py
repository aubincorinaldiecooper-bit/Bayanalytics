"""Structured error contract (AGENT.md section 39).

Every failure that reaches the API or the SSE stream is an ``AnalysisError`` carrying one of the
locked error codes. Raw exceptions never reach clients: unknown failures become INTERNAL_ERROR.
"""

from __future__ import annotations

from typing import Any

from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.errors import ErrorPayload

# code -> (default message, retryable, http status)
_DEFAULTS: dict[ErrorCode, tuple[str, bool, int]] = {
    ErrorCode.AMBIGUOUS_INSTRUMENT: (
        "The company could not be identified unambiguously. Specify the ticker or exchange.",
        False,
        422,
    ),
    ErrorCode.INSUFFICIENT_EVIDENCE: (
        "Not enough current public evidence was found to make a reliable assessment.",
        False,
        422,
    ),
    ErrorCode.STALE_EVIDENCE: (
        "The available evidence is too old to support a current assessment.",
        True,
        422,
    ),
    ErrorCode.RESEARCH_UNAVAILABLE: (
        "Public research is unavailable right now. The analysis was not completed.",
        True,
        503,
    ),
    ErrorCode.SOURCE_CONFLICT: (
        "Credible sources disagree on a material value and the conflict could not be resolved.",
        False,
        422,
    ),
    ErrorCode.MISSING_CALCULATION_INPUT: (
        "A required input for a deterministic calculation was not available.",
        False,
        422,
    ),
    ErrorCode.FAST_PROFILE_UNAVAILABLE: (
        "The Fast analysis profile is not available on this machine.",
        True,
        503,
    ),
    ErrorCode.DEEP_PROFILE_UNAVAILABLE: (
        "Deep analysis is not available on this machine right now. Try Fast.",
        True,
        503,
    ),
    ErrorCode.MEMORY_PRESSURE: (
        "There is not enough free memory to run this analysis safely.",
        True,
        503,
    ),
    ErrorCode.SPARK_START_FAILED: ("The synthesis model could not be started.", True, 503),
    ErrorCode.SPARK_INFERENCE_FAILED: ("The synthesis model failed while generating.", True, 502),
    ErrorCode.LAYA_INFERENCE_FAILED: ("The decision model failed.", True, 502),
    ErrorCode.WHISPER_FAILED: ("Speech transcription failed.", True, 502),
    ErrorCode.INTERRUPTED: (
        "The analysis was interrupted by a backend restart and was not resumed.",
        True,
        409,
    ),
    ErrorCode.CANCELLED: ("The analysis was cancelled.", False, 409),
    ErrorCode.INTERNAL_ERROR: ("An internal error occurred.", True, 500),
    ErrorCode.NOT_FOUND: ("No analysis exists with that id.", False, 404),
    ErrorCode.INVALID_REQUEST: ("The request was invalid.", False, 422),
}


class AnalysisError(Exception):
    """A failure with a stable code, a user-safe message and optional structured details."""

    def __init__(
        self,
        code: ErrorCode | str,
        message: str | None = None,
        *,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
        http_status: int | None = None,
    ) -> None:
        self.code = ErrorCode(code)
        default_message, default_retryable, default_status = _DEFAULTS[self.code]
        self.message = message or default_message
        self.retryable = default_retryable if retryable is None else retryable
        self.details = details or {}
        self.http_status = http_status or default_status
        super().__init__(f"{self.code.value}: {self.message}")

    def payload(self) -> ErrorPayload:
        return ErrorPayload(
            code=self.code,
            message=self.message,
            retryable=self.retryable,
            details=self.details or None,
        )

    @classmethod
    def from_exception(cls, exc: BaseException) -> AnalysisError:
        if isinstance(exc, AnalysisError):
            return exc
        return cls(ErrorCode.INTERNAL_ERROR, details={"exception": type(exc).__name__})


def default_message(code: ErrorCode) -> str:
    return _DEFAULTS[code][0]
