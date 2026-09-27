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

    async def mark_interrupted(self) -> list[str]:
        """Mark every non-terminal job as failed/INTERRUPTED. Returns the affected ids."""
        ...
