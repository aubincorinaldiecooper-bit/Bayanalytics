"""Persistence boundary. Implementations: ``InMemoryStore`` and ``PostgresStore``."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from bayanalytics.jobs.models import AnalysisJob
from bayanalytics.schemas.calculations import CalculationResult
from bayanalytics.schemas.decisions import LayaDecision
from bayanalytics.schemas.events import AnalysisEvent
from bayanalytics.schemas.evidence import NormalizedFact, SourceRecord
from bayanalytics.schemas.results import AnalysisResult

OWNER_SCOPE_UNSUPPORTED = (
    "owner-scoped lookups are not supported: analyses do not carry ownership yet (scoping by "
    "owner needs an owner_id recorded on each analysis); pass owner_id=None for today's "
    "single-tenant behaviour"
)


def refuse_owner_scope(owner_id: str | None) -> None:
    """Raise for any owner filter: jobs carry no owner, so a filter would be silently
    ignored, and a caller must never believe a lookup was scoped to a user when it was not."""
    if owner_id is not None:
        raise NotImplementedError(OWNER_SCOPE_UNSUPPORTED)


class AnalysisStore(Protocol):
    async def start(self) -> None: ...

    async def close(self) -> None: ...

    async def create_job(self, job: AnalysisJob) -> None: ...

    async def update_job(self, job: AnalysisJob) -> None: ...

    async def get_job(self, analysis_id: str) -> AnalysisJob | None: ...

    async def list_jobs(
        self, limit: int = 50, *, before: tuple[datetime, str] | None = None
    ) -> list[AnalysisJob]:
        """Jobs newest first by ``(created_at, analysis_id)``, at most ``limit`` of them.

        ``before`` is that key taken from the last row of the previous page (keyset
        pagination), so a page boundary stays stable while new analyses are created.
        """
        ...

    async def append_event(self, event: AnalysisEvent) -> None: ...

    async def list_events(self, analysis_id: str, after_seq: int = 0) -> list[AnalysisEvent]: ...

    async def save_sources(self, analysis_id: str, sources: list[SourceRecord]) -> None: ...

    async def save_facts(self, analysis_id: str, facts: list[NormalizedFact]) -> None: ...

    async def save_decisions(self, analysis_id: str, decisions: list[LayaDecision]) -> None: ...

    async def save_calculations(
        self, analysis_id: str, calculations: list[CalculationResult]
    ) -> None: ...

    async def save_result(self, result: AnalysisResult) -> None: ...

    async def get_result(self, analysis_id: str) -> AnalysisResult | None: ...

    async def latest_completed_result(
        self,
        symbol: str,
        *,
        before: datetime | None = None,
        owner_id: str | None = None,
    ) -> AnalysisResult | None:
        """The newest ``completed`` result whose job resolved to instrument ``symbol``
        (case-insensitive), or ``None``.

        Newest is by the job's ``(created_at, analysis_id)``; ``before`` is an exclusive
        upper bound on ``created_at`` so an analysis can ask for the assessment that
        preceded it and never see itself or a later run.

        Ownership: jobs carry no owner yet, so this lookup spans every analysis in the store,
        which is only correct for a single-user deployment. ``owner_id`` is the seam for
        scoping it: it must be ``None`` today, and every implementation raises
        ``NotImplementedError`` for any other value (owner scoping needs an ``owner_id``
        column on analyses), so no caller can believe it filtered by user. A multi-user
        deployment must scope this lookup by owner before enabling it.
        """
        ...

    async def mark_interrupted(self) -> list[str]:
        """Mark every non-terminal job as failed/INTERRUPTED. Returns the affected ids."""
        ...
