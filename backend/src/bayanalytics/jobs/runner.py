"""In-process asyncio job runner (AGENT.md section 38).

One task per analysis. The runner owns job status transitions, persistence of the job row and
the terminal event. The pipeline function it is given must always return an ``AnalysisResult``
whose status is completed, failed or cancelled; it never raises.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from bayanalytics.context import AnalysisContext, CancelToken
from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.base import ResearchBudget
from bayanalytics.jobs.bus import AnalysisEventBus
from bayanalytics.jobs.models import AnalysisJob
from bayanalytics.schemas.common import ErrorCode, new_id, utcnow
from bayanalytics.schemas.errors import ErrorPayload
from bayanalytics.schemas.requests import CreateAnalysisRequest
from bayanalytics.schemas.results import AnalysisResult
from bayanalytics.store.base import AnalysisStore

log = logging.getLogger(__name__)

PipelineFn = Callable[[AnalysisJob, AnalysisContext], Awaitable[AnalysisResult]]


class AnalysisRunner:
    def __init__(
        self,
        store: AnalysisStore,
        bus: AnalysisEventBus,
        pipeline: PipelineFn,
        *,
        versions: dict[str, Any] | None = None,
    ) -> None:
        self._store = store
        self._bus = bus
        self._pipeline = pipeline
        self._versions = versions or {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._contexts: dict[str, AnalysisContext] = {}
        self._accepting = True

    # -- lifecycle -----------------------------------------------------------------------
    async def start(self) -> list[str]:
        interrupted = await self._store.mark_interrupted()
        if interrupted:
            log.warning("marked %d analyses INTERRUPTED from a previous process", len(interrupted))
        for analysis_id in interrupted:
            await self._close_interrupted(analysis_id)
        self._accepting = True
        return interrupted

    async def _close_interrupted(self, analysis_id: str) -> None:
        """Append the terminal event for a job a previous process left running, so a client
        that reconnects to its event stream sees ``analysis.failed`` instead of waiting."""
        events = await self._store.list_events(analysis_id)
        if events and events[-1].terminal:
            return
        last_seq = events[-1].seq if events else 0
        self._bus.register(analysis_id, last_seq=last_seq)
        job = await self._store.get_job(analysis_id)
        error = (job.error if job and job.error else None) or AnalysisError(
            ErrorCode.INTERRUPTED
        ).payload()
        try:
            await self._bus.publish(
                analysis_id,
                "analysis.failed",
                {"status": "failed", "error": error.model_dump(mode="json"), "partial": True},
            )
        except RuntimeError:
            return  # already terminal in this bus
        if job is not None:
            job.last_seq = self._bus.last_seq(analysis_id)
            job.touch()
            await self._store.update_job(job)

    async def shutdown(self, timeout_s: float = 10.0) -> None:
        self._accepting = False
        for ctx in list(self._contexts.values()):
            ctx.cancel.cancel()
        tasks = list(self._tasks.values())
        if tasks:
            _done, pending = await asyncio.wait(tasks, timeout=timeout_s)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    @property
    def active_count(self) -> int:
        return len(self._tasks)

    def active_ids(self) -> list[str]:
        return list(self._tasks)

    # -- submission ----------------------------------------------------------------------
    async def submit(
        self,
        request: CreateAnalysisRequest,
        *,
        resolved_horizon: str,
        budget: Any,
        as_of: Any | None = None,
    ) -> AnalysisJob:
        if not self._accepting:
            raise AnalysisError(ErrorCode.INTERNAL_ERROR, "The backend is shutting down.")
        job = AnalysisJob(
            analysis_id=new_id("an"),
            query=request.query,
            instrument_ref=request.instrument,
            profile=request.profile,
            requested_horizon=request.horizon,
            resolved_horizon=resolved_horizon,  # type: ignore[arg-type]
            research_budget=budget if budget is not None else ResearchBudget(),
            normalization_version=str(self._versions.get("normalization_version", "")),
            laya_schema_version=str(self._versions.get("laya_schema_version", "")),
            spark_artifact=str(self._versions.get("spark_artifact", "")),
            spark_runtime=self._versions.get("spark_runtime"),
        )
        if as_of is not None:
            job.as_of = as_of
        await self._store.create_job(job)
        self._bus.register(job.analysis_id)
        ctx = AnalysisContext(analysis_id=job.analysis_id, emit=self._emitter(job.analysis_id))
        self._contexts[job.analysis_id] = ctx
        task = asyncio.create_task(self._run(job, ctx), name=f"analysis:{job.analysis_id}")
        self._tasks[job.analysis_id] = task
        return job

    def _emitter(self, analysis_id: str) -> Callable[[str, dict[str, Any]], Awaitable[None]]:
        async def emit(event: str, data: dict[str, Any]) -> None:
            await self._bus.publish(analysis_id, event, data)

        return emit

    # -- execution -----------------------------------------------------------------------
    async def _run(self, job: AnalysisJob, ctx: AnalysisContext) -> None:
        job.started_at = utcnow()
        try:
            result = await self._pipeline(job, ctx)
        except asyncio.CancelledError:
            result = self._aborted_result(job, ErrorCode.CANCELLED)
        except BaseException as exc:  # the pipeline promised not to raise; keep the contract
            log.exception("pipeline raised for %s", job.analysis_id)
            result = self._aborted_result(job, AnalysisError.from_exception(exc).code)
        try:
            await self._finish(job, result)
        except Exception:
            log.exception("failed to persist terminal state for %s", job.analysis_id)
        finally:
            self._tasks.pop(job.analysis_id, None)
            self._contexts.pop(job.analysis_id, None)
            self._bus.forget(job.analysis_id)

    def _aborted_result(self, job: AnalysisJob, code: ErrorCode) -> AnalysisResult:
        error = AnalysisError(code).payload()
        status = "cancelled" if code == ErrorCode.CANCELLED else "failed"
        return AnalysisResult(
            analysis_id=job.analysis_id,
            status=status,
            query=job.query,
            profile=job.profile,
            horizon=job.resolved_horizon,
            as_of=job.as_of,
            created_at=job.created_at,
            completed_at=utcnow(),
            error=error,
            partial=True,
        )

    async def _finish(self, job: AnalysisJob, result: AnalysisResult) -> None:
        job.status = result.status
        job.error = result.error
        job.finished_at = result.completed_at or utcnow()
        job.telemetry = result.telemetry.model_dump(mode="json")
        job.touch()
        # Persist the result before the terminal event so a client reacting to
        # analysis.completed always finds the structured result on GET.
        await self._store.save_result(result)
        await self._store.update_job(job)
        if self._bus.is_terminal(job.analysis_id):
            job.last_seq = self._bus.last_seq(job.analysis_id)
            await self._store.update_job(job)
            return
        if result.status == "completed":
            await self._bus.publish(
                job.analysis_id,
                "analysis.completed",
                {
                    "status": "completed",
                    "analysis_id": job.analysis_id,
                    "total_request_ms": result.telemetry.total_request_ms,
                },
            )
        else:
            error = result.error or ErrorPayload(
                code=ErrorCode.INTERNAL_ERROR, message="unknown failure", retryable=True
            )
            await self._bus.publish(
                job.analysis_id,
                "analysis.failed",
                {
                    "status": result.status,
                    "error": error.model_dump(mode="json"),
                    "partial": result.partial,
                },
            )
        job.last_seq = self._bus.last_seq(job.analysis_id)
        job.touch()
        await self._store.update_job(job)

    # -- cancellation --------------------------------------------------------------------
    async def cancel(self, analysis_id: str) -> AnalysisJob | None:
        job = await self._store.get_job(analysis_id)
        if job is None:
            return None
        if job.terminal:
            return job
        job.cancel_requested = True
        job.touch()
        await self._store.update_job(job)
        ctx = self._contexts.get(analysis_id)
        if ctx is not None:
            ctx.cancel.cancel()
        return job

    def cancel_token(self, analysis_id: str) -> CancelToken | None:
        ctx = self._contexts.get(analysis_id)
        return ctx.cancel if ctx else None
