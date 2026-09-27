"""Analysis endpoints: create, stream events, read, cancel (AGENT.md 37.1-37.4)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Header, Query
from fastapi.responses import Response

from bayanalytics.api.deps import RuntimeDep
from bayanalytics.api.sse import event_stream, parse_after_seq, sse_response
from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.base import ResearchBudget
from bayanalytics.pipeline.horizon import resolve_horizon
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.events import sse_comment
from bayanalytics.schemas.requests import (
    CancelAnalysisResponse,
    CreateAnalysisRequest,
    CreateAnalysisResponse,
)
from bayanalytics.schemas.results import AnalysisResult, InstrumentView

router = APIRouter(prefix="/analyses", tags=["analyses"])


@router.post("", response_model=CreateAnalysisResponse, status_code=202)
async def create_analysis(body: CreateAnalysisRequest, rt: RuntimeDep) -> CreateAnalysisResponse:
    capability = rt.spark.availability(body.profile)
    if not capability.available:
        # An external llama-server started after this backend is picked up here.
        await rt.refresh_spark()
        capability = rt.spark.availability(body.profile)
    if not capability.available:
        code = capability.code or (
            ErrorCode.DEEP_PROFILE_UNAVAILABLE
            if body.profile == "deep"
            else ErrorCode.FAST_PROFILE_UNAVAILABLE
        )
        raise AnalysisError(code, details={"reason": capability.reason})
    # Resolve the instrument synchronously so an ambiguous query is answered here, with
    # candidates, instead of creating an analysis that fails a few milliseconds later
    # (frontend contract section 18: "Which company did you mean?").
    resolver_factory = rt.extras.get("resolver_factory")
    if resolver_factory is not None:
        resolver = await resolver_factory()
        resolver.resolve(body.query, body.instrument)
    resolved = resolve_horizon(body.query, body.horizon)
    settings = rt.settings
    budget = ResearchBudget(
        max_rounds=settings.research_max_rounds,
        max_sources=settings.research_max_sources,
        max_fetch_per_round=settings.research_max_fetch_per_round,
        timeout_s=settings.research_timeout_s,
    )
    job = await rt.runner.submit(
        body, resolved_horizon=resolved, budget=budget, as_of=settings.eval_as_of
    )
    return CreateAnalysisResponse(
        analysis_id=job.analysis_id,
        status=job.status,
        profile=job.profile,
        resolved_horizon=job.resolved_horizon,
    )


@router.get("/{analysis_id}/events")
async def stream_events(
    analysis_id: str,
    rt: RuntimeDep,
    last_event_id: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
    after: Annotated[int | None, Query(ge=0)] = None,
) -> Response:
    job = await rt.store.get_job(analysis_id)
    if job is None:
        raise AnalysisError(ErrorCode.NOT_FOUND)
    after_seq = parse_after_seq(last_event_id, after)
    if job.terminal and job.last_seq and after_seq >= job.last_seq:
        # Nothing left to send: close at once rather than holding a zombie connection.
        return sse_response(_closed_stream())
    return sse_response(
        event_stream(rt.bus, analysis_id, after_seq, keepalive_s=rt.settings.sse_keepalive_s)
    )


async def _closed_stream() -> AsyncIterator[str]:
    yield sse_comment("stream complete")


@router.get("/{analysis_id}", response_model=AnalysisResult, response_model_exclude_none=False)
async def get_analysis(analysis_id: str, rt: RuntimeDep) -> AnalysisResult:
    result = await rt.store.get_result(analysis_id)
    if result is not None:
        return result
    job = await rt.store.get_job(analysis_id)
    if job is None:
        raise AnalysisError(ErrorCode.NOT_FOUND)
    instrument = (
        InstrumentView(
            symbol=job.instrument.symbol,
            exchange=job.instrument.exchange,
            name=job.instrument.name,
            cik=job.instrument.cik,
            sector=job.instrument.sector,
        )
        if job.instrument
        else None
    )
    # Running (or failed-before-result) analysis: return the durable artifacts persisted so
    # far, so a client recovering from a dropped stream sees sources and calculations.
    sources = await _optional(rt.store, "get_sources", analysis_id)
    calculations = await _optional(rt.store, "get_calculations", analysis_id)
    decisions = await _optional(rt.store, "get_decisions", analysis_id)
    return AnalysisResult(
        analysis_id=job.analysis_id,
        status=job.status,
        query=job.query,
        instrument=instrument,
        profile=job.profile,
        horizon=job.resolved_horizon,
        as_of=job.as_of,
        created_at=job.created_at,
        completed_at=job.finished_at,
        sources=sources,
        calculations=calculations,
        laya_decisions=decisions,
        error=job.error,
        partial=job.status != "queued",
    )


async def _optional(store: object, method: str, analysis_id: str) -> list:
    getter = getattr(store, method, None)
    if getter is None:
        return []
    try:
        return list(await getter(analysis_id))
    except Exception:  # pragma: no cover - a snapshot must never fail because of extras
        return []


@router.post("/{analysis_id}/cancel", response_model=CancelAnalysisResponse)
async def cancel_analysis(analysis_id: str, rt: RuntimeDep) -> CancelAnalysisResponse:
    job = await rt.runner.cancel(analysis_id)
    if job is None:
        raise AnalysisError(ErrorCode.NOT_FOUND)
    return CancelAnalysisResponse(
        analysis_id=job.analysis_id, status=job.status, cancel_requested=job.cancel_requested
    )
