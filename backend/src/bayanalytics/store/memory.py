"""In-memory ``AnalysisStore`` for local runs and tests.

Everything lives in dictionaries behind a single ``asyncio.Lock``. Stored objects are deep
copies, and every read returns a fresh deep copy, so callers can never mutate persisted state
by accident. Semantics mirror ``PostgresStore`` exactly so the orchestrator can be tested here
and deployed there.
"""

from __future__ import annotations

import asyncio

from bayanalytics.errors import default_message
from bayanalytics.jobs.models import AnalysisJob
from bayanalytics.schemas.calculations import CalculationResult
from bayanalytics.schemas.common import TERMINAL_STATUSES, ErrorCode, utcnow
from bayanalytics.schemas.decisions import LayaDecision
from bayanalytics.schemas.errors import ErrorPayload
from bayanalytics.schemas.events import AnalysisEvent
from bayanalytics.schemas.evidence import NormalizedFact, SourceRecord
from bayanalytics.schemas.results import AnalysisResult


def interrupted_payload() -> ErrorPayload:
    """The error every non-terminal job receives when the backend restarts (section 38)."""
    return ErrorPayload(
        code=ErrorCode.INTERRUPTED,
        message=default_message(ErrorCode.INTERRUPTED),
        retryable=True,
    )


class InMemoryStore:
    """Dict-backed store. ``create_job``/``update_job`` and ``save_*`` overwrite by id."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._jobs: dict[str, AnalysisJob] = {}
        self._events: dict[str, dict[int, AnalysisEvent]] = {}
        self._sources: dict[str, dict[str, SourceRecord]] = {}
        self._facts: dict[str, dict[str, NormalizedFact]] = {}
        self._decisions: dict[str, dict[str, LayaDecision]] = {}
        self._calculations: dict[str, dict[str, CalculationResult]] = {}
        self._results: dict[str, AnalysisResult] = {}

    # --- lifecycle --------------------------------------------------------------------

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    # --- jobs -------------------------------------------------------------------------

    async def create_job(self, job: AnalysisJob) -> None:
        async with self._lock:
            self._jobs[job.analysis_id] = job.model_copy(deep=True)

    async def update_job(self, job: AnalysisJob) -> None:
        async with self._lock:
            self._jobs[job.analysis_id] = job.model_copy(deep=True)

    async def get_job(self, analysis_id: str) -> AnalysisJob | None:
        async with self._lock:
            job = self._jobs.get(analysis_id)
            return job.model_copy(deep=True) if job is not None else None

    async def list_jobs(self) -> list[AnalysisJob]:
        """All jobs, newest first (helper for diagnostics; not part of the protocol)."""
        async with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
            return [job.model_copy(deep=True) for job in jobs]

    async def count_active(self) -> int:
        """Number of non-terminal jobs (used by ``/health``)."""
        async with self._lock:
            return sum(1 for job in self._jobs.values() if job.status not in TERMINAL_STATUSES)

    # --- events -----------------------------------------------------------------------

    async def append_event(self, event: AnalysisEvent) -> None:
        async with self._lock:
            bucket = self._events.setdefault(event.analysis_id, {})
            if event.seq in bucket:
                raise ValueError(
                    f"duplicate event seq {event.seq} for analysis {event.analysis_id}"
                )
            bucket[event.seq] = event.model_copy(deep=True)

    async def list_events(self, analysis_id: str, after_seq: int = 0) -> list[AnalysisEvent]:
        async with self._lock:
            bucket = self._events.get(analysis_id, {})
            return [bucket[seq].model_copy(deep=True) for seq in sorted(bucket) if seq > after_seq]

    # --- evidence, decisions, calculations ------------------------------------------

    async def save_sources(self, analysis_id: str, sources: list[SourceRecord]) -> None:
        async with self._lock:
            bucket = self._sources.setdefault(analysis_id, {})
            for source in sources:
                bucket[source.source_id] = source.model_copy(deep=True)

    async def save_facts(self, analysis_id: str, facts: list[NormalizedFact]) -> None:
        async with self._lock:
            bucket = self._facts.setdefault(analysis_id, {})
            for fact in facts:
                bucket[fact.fact_id] = fact.model_copy(deep=True)

    async def save_decisions(self, analysis_id: str, decisions: list[LayaDecision]) -> None:
        async with self._lock:
            bucket = self._decisions.setdefault(analysis_id, {})
            for decision in decisions:
                bucket[decision.decision_id] = decision.model_copy(deep=True)

    async def save_calculations(
        self, analysis_id: str, calculations: list[CalculationResult]
    ) -> None:
        async with self._lock:
            bucket = self._calculations.setdefault(analysis_id, {})
            for calc in calculations:
                bucket[calc.calc_id] = calc.model_copy(deep=True)

    async def get_sources(self, analysis_id: str) -> list[SourceRecord]:
        async with self._lock:
            return [s.model_copy(deep=True) for s in self._sources.get(analysis_id, {}).values()]

    async def get_facts(self, analysis_id: str) -> list[NormalizedFact]:
        async with self._lock:
            return [f.model_copy(deep=True) for f in self._facts.get(analysis_id, {}).values()]

    async def get_decisions(self, analysis_id: str) -> list[LayaDecision]:
        async with self._lock:
            return [d.model_copy(deep=True) for d in self._decisions.get(analysis_id, {}).values()]

    async def get_calculations(self, analysis_id: str) -> list[CalculationResult]:
        async with self._lock:
            return [
                c.model_copy(deep=True) for c in self._calculations.get(analysis_id, {}).values()
            ]

    # --- results ----------------------------------------------------------------------

    async def save_result(self, result: AnalysisResult) -> None:
        async with self._lock:
            self._results[result.analysis_id] = result.model_copy(deep=True)

    async def get_result(self, analysis_id: str) -> AnalysisResult | None:
        async with self._lock:
            result = self._results.get(analysis_id)
            return result.model_copy(deep=True) if result is not None else None

    # --- startup recovery -------------------------------------------------------------

    async def mark_interrupted(self) -> list[str]:
        """Fail every non-terminal job with INTERRUPTED (AGENT.md section 38)."""
        now = utcnow()
        affected: list[str] = []
        async with self._lock:
            for analysis_id, job in self._jobs.items():
                if job.status in TERMINAL_STATUSES:
                    continue
                job.status = "failed"
                job.error = interrupted_payload()
                job.finished_at = now
                job.updated_at = now
                affected.append(analysis_id)
        return affected
