"""ResearchRunner: executes one ``PlannedQuery`` (a web search) at a time for the EquityAnalyzer.

The ``ResearchProvider`` stays pure (search / open / extract). This runner adds what the
finance layer owns: leakage guard against ``as_of``, paywall and thin-content rejection,
deduplication, source classification, provenance records, budget enforcement, counters and
observable events. Every page it opens is a hit of the search it just ran.

Retrieving data is web search's main job: with a Laya wrapper the runner asks Laya which hits
most likely hold the intent's topic and opens them in that order (engine order without Laya),
and for every page with candidate tables it asks Laya which table is the daily price history
or the quarterly results, which column is the close and which line is which figure
(``research.selection``). Deterministic code reads what was chosen (``research.tables``); the
series and figures accumulate in ``MarketData`` with the page they came from. A page with data
is kept even when its readable text is short (a CSV or JSON response has none).

Events emitted HERE (the analyzer emits ``research.started`` / ``research.completed``):

* ``research.query``            {intent, kind, query, label, round}
* ``research.search_results``   {query, intent, round, total, failed, hits}
* ``laya.started`` / ``laya.decision`` / ``laya.completed`` for the ``source_selection`` and
  ``data_identification`` choices
* ``research.fetching``         {url, domain, kind="web", label, intent, round}
* ``research.fetch_skipped``    {url, domain, reason ("duplicate" | "budget"), intent, round}
* ``research.source_found``     SourceRecord.public_view() + {domain, fetch_ms, text_chars,
  redistribution, excerpt (only when redistribution == "allowed"), preview (a small table when
  the page yielded prices or figures, else null), intent, round}
* ``research.source_rejected``  {url, title, reason, domain, fetch_ms, intent, round};
  ``reason`` is always one of ``REJECTION_REASONS`` (fixed keywords, never transport text)
* ``market.series``             a price series the analysis keeps (company or broad market),
  right after its page's ``research.source_found``
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.base import InstrumentIdentity, ResearchBudget
from bayanalytics.laya.wrapper import LayaFinanceWrapper
from bayanalytics.research.dedup import Deduplicator
from bayanalytics.research.http_provider import REASON_NOT_FROM_SEARCH
from bayanalytics.research.intents import PlannedQuery, intent_topic
from bayanalytics.research.market import (
    BROAD_MARKET,
    BROAD_MARKET_NAME,
    BROAD_MARKET_SYMBOL,
    COMPANY,
    MarketData,
)
from bayanalytics.research.provider import (
    EvidenceRecord,
    ResearchProvider,
    ResearchProviderError,
    SearchResult,
)
from bayanalytics.research.selection import PageData, identify_page_data, order_hits, role_for
from bayanalytics.research.sources import (
    canonical_url,
    classify_freshness,
    classify_source,
    domain_of,
    source_record_from_evidence,
)
from bayanalytics.schemas.common import stable_id, utcnow
from bayanalytics.schemas.decisions import LayaDecision
from bayanalytics.schemas.evidence import SourceRecord
from bayanalytics.schemas.results import ResearchStats

log = logging.getLogger(__name__)

MIN_TEXT_CHARS = 200
REASON_LEAKAGE = "published_after_as_of"
REASON_PAYWALLED = "paywalled"
REASON_THIN = "thin_content"
REASON_FETCH_FAILED = "fetch_failed"
REASON_BLOCKED = "blocked_target"
REASON_TIMEOUT = "timeout"
REASON_EXTRACT_FAILED = "extract_failed"
REASON_ROBOTS = "robots_disallowed"
REASON_SEARCH_FAILED = "search_failed"
TERMINATION_MAX_SOURCES = "max_sources"

# The only rejection / failure reasons that leave the runner (events, error details). Raw
# exception text and URLs go to the log, never to clients.
REJECTION_REASONS: frozenset[str] = frozenset(
    {
        REASON_LEAKAGE,
        REASON_PAYWALLED,
        REASON_THIN,
        REASON_FETCH_FAILED,
        REASON_BLOCKED,
        REASON_TIMEOUT,
        REASON_EXTRACT_FAILED,
        REASON_ROBOTS,
        REASON_SEARCH_FAILED,
        REASON_NOT_FROM_SEARCH,
    }
)


def reject_reason(exc: BaseException) -> str:
    """Map a provider failure to a fixed reason keyword (default ``fetch_failed``)."""
    tagged = getattr(exc, "reason", None)
    if isinstance(tagged, str) and tagged in REJECTION_REASONS:
        return tagged
    message = str(exc)
    if message in REJECTION_REASONS:
        return message
    return REASON_FETCH_FAILED


def _elapsed_ms(started: float) -> int:
    return round((time.perf_counter() - started) * 1000)


class RoundResult(BaseModel):
    intent: str
    kind: str
    label: str = ""
    sources: list[SourceRecord] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    rejected: list[tuple[SourceRecord, str]] = Field(default_factory=list)
    duplicates: int = 0
    notes: list[str] = Field(default_factory=list)
    budget_exhausted: bool = False


class ResearchRunner:
    """``laya`` makes the bounded choices (hit order, tables, columns, lines); without it hits
    open in engine order and no table is read. ``market`` accumulates the series and figures
    read from pages; ``decisions`` receives every Laya decision the runner records."""

    def __init__(
        self,
        provider: ResearchProvider,
        settings: Settings,
        budget: ResearchBudget,
        *,
        laya: LayaFinanceWrapper | None = None,
        market: MarketData | None = None,
        decisions: list[LayaDecision] | None = None,
    ) -> None:
        self.provider = provider
        self.settings = settings
        self.budget = budget
        self.laya = laya
        self.market = market if market is not None else MarketData()
        self.decisions: list[LayaDecision] = decisions if decisions is not None else []
        self.stats = ResearchStats()
        self.dedup = Deduplicator()
        self.kept_sources = 0
        self._round = 0

    # -- rounds / budget ------------------------------------------------------------------

    def begin_round(self) -> int:
        self._round += 1
        self.stats.search_rounds = self._round
        return self._round

    @property
    def round_no(self) -> int:
        return self._round

    @property
    def budget_exhausted(self) -> bool:
        return self.kept_sources >= self.budget.max_sources

    def finish(self, reason: str) -> ResearchStats:
        if not self.stats.termination_reason:
            self.stats.termination_reason = reason
        return self.stats

    def _note_intent(self, intent: str) -> None:
        if intent not in self.stats.intents:
            self.stats.intents.append(intent)

    def _check_budget(self) -> bool:
        if self.budget_exhausted:
            self.stats.termination_reason = TERMINATION_MAX_SOURCES
            return True
        return False

    # -- execute --------------------------------------------------------------------------

    async def execute(
        self,
        planned: PlannedQuery,
        identity: InstrumentIdentity,
        as_of: datetime,
        ctx: AnalysisContext,
    ) -> RoundResult:
        ctx.check_cancelled()
        if self._round == 0:
            self.begin_round()
        intent = str(planned.intent)
        self._note_intent(intent)
        result = RoundResult(intent=intent, kind=planned.kind, label=planned.label)
        await ctx.event(
            "research.query",
            intent=intent,
            kind=planned.kind,
            query=planned.query,
            label=planned.label,
            round=self._round,
        )
        ctx.timers.start("retrieval")
        try:
            await self._run_search(planned, identity, as_of, ctx, result)
        finally:
            self.stats.retrieval_total_ms += ctx.timers.stop("retrieval")
        result.budget_exhausted = self.budget_exhausted
        if result.budget_exhausted:
            self.stats.termination_reason = TERMINATION_MAX_SOURCES
        return result

    # -- shared bookkeeping ---------------------------------------------------------------

    async def _keep(
        self,
        source: SourceRecord,
        ctx: AnalysisContext,
        result: RoundResult,
        evidence: EvidenceRecord | None = None,
        *,
        fetch_ms: int | None = None,
        text_chars: int | None = None,
        preview: dict[str, Any] | None = None,
    ) -> None:
        self.kept_sources += 1
        result.sources.append(source)
        if evidence is not None:
            result.evidence.append(evidence)
        # The excerpt is third-party text: it only leaves the backend when the source's terms
        # allow redistribution (the same rule the final sources list follows).
        excerpt = source.excerpt if source.redistribution == "allowed" and source.excerpt else None
        await ctx.event(
            "research.source_found",
            **source.public_view(),
            domain=domain_of(source.url),
            fetch_ms=fetch_ms,
            text_chars=text_chars,
            redistribution=source.redistribution,
            excerpt=excerpt,
            preview=preview,
            intent=result.intent,
            round=self._round,
        )

    async def _reject(
        self,
        source: SourceRecord,
        reason: str,
        ctx: AnalysisContext,
        result: RoundResult,
        *,
        fetch_ms: int | None = None,
    ) -> None:
        source.rejected_reason = reason
        self.stats.sources_rejected += 1
        result.rejected.append((source, reason))
        await ctx.event(
            "research.source_rejected",
            url=source.url,
            title=source.title,
            reason=reason,
            domain=domain_of(source.url),
            fetch_ms=fetch_ms,
            intent=result.intent,
            round=self._round,
        )

    async def _fetching(
        self, ctx: AnalysisContext, url: str, kind: str, intent: str, label: str | None = None
    ) -> None:
        """Announce a request right before it goes out, so clients can show it live."""
        await ctx.event(
            "research.fetching",
            url=url,
            domain=domain_of(url),
            kind=kind,
            label=label,
            intent=intent,
            round=self._round,
        )

    async def _skipped(self, ctx: AnalysisContext, url: str, reason: str, intent: str) -> None:
        await ctx.event(
            "research.fetch_skipped",
            url=url,
            domain=domain_of(url),
            reason=reason,
            intent=intent,
            round=self._round,
        )

    # -- search ---------------------------------------------------------------------------

    async def _search(self, planned: PlannedQuery) -> list[SearchResult]:
        query = planned.query or ""
        search_with = getattr(self.provider, "search_with", None)
        if callable(search_with):
            return await search_with(
                query,
                categories=planned.params.get("categories"),
                time_range=planned.params.get("time_range"),
            )
        return await self.provider.search(query)

    async def _run_search(
        self,
        planned: PlannedQuery,
        identity: InstrumentIdentity,
        as_of: datetime,
        ctx: AnalysisContext,
        result: RoundResult,
    ) -> None:
        self.stats.queries_issued += 1
        try:
            results = await self._search(planned)
        except ResearchProviderError as exc:
            result.notes.append(f"{REASON_SEARCH_FAILED}: {planned.label or planned.kind}")
            log.warning("search failed for %r: %s", planned.query, exc)
            await ctx.event(
                "research.search_results",
                query=planned.query or "",
                intent=result.intent,
                round=self._round,
                total=0,
                failed=True,
                hits=[],
            )
            return
        await ctx.event(
            "research.search_results",
            query=planned.query or "",
            intent=result.intent,
            round=self._round,
            total=len(results),
            failed=False,
            hits=[
                {
                    "url": h.url,
                    "title": h.title,
                    "domain": domain_of(h.url),
                    "published_at": h.published_at.isoformat() if h.published_at else None,
                }
                for h in results[:8]
            ],
        )
        results, decisions = await order_hits(
            self.laya,
            results,
            symbol=identity.symbol,
            topic=intent_topic(planned.intent, planned.params.get("gap")),
            query=planned.query or "",
            ctx=ctx,
        )
        self.decisions.extend(decisions)
        attempts = 0
        for hit in results:
            if attempts >= self.budget.max_fetch_per_round:
                break
            if self._check_budget():
                break
            ctx.check_cancelled()
            if hit.published_at is not None and hit.published_at > as_of:
                await self._reject(
                    self._source_from_hit(hit, identity, result, as_of), REASON_LEAKAGE, ctx, result
                )
                continue
            if self.dedup.known_url(hit.url):
                self.dedup.duplicates += 1
                result.duplicates += 1
                self.stats.duplicate_sources_removed += 1
                await self._skipped(ctx, hit.url, "duplicate", result.intent)
                continue
            self.dedup.seen(hit.url)
            attempts += 1
            await self._fetching(ctx, hit.url, "web", result.intent, hit.title or None)
            started = time.perf_counter()
            try:
                record = await self.provider.extract(hit.url)
            except ResearchProviderError as exc:
                reason = reject_reason(exc)
                log.debug("source %s rejected (%s): %s", hit.url, reason, exc)
                await self._reject(
                    self._source_from_hit(hit, identity, result, as_of),
                    reason,
                    ctx,
                    result,
                    fetch_ms=_elapsed_ms(started),
                )
                continue
            fetch_ms = _elapsed_ms(started)
            text_chars = len(record.text)
            self.stats.sources_fetched += 1
            if record.published_at is None and hit.published_at is not None:
                record.published_at = hit.published_at
            source = source_record_from_evidence(record, identity.symbol, result.intent, as_of)
            record.source_type = source.source_type
            if record.metadata.get("paywalled"):
                await self._reject(source, REASON_PAYWALLED, ctx, result, fetch_ms=fetch_ms)
                continue
            if record.published_at is not None and record.published_at > as_of:
                await self._reject(source, REASON_LEAKAGE, ctx, result, fetch_ms=fetch_ms)
                continue
            if self.dedup.seen_record(record, check_url=False):
                result.duplicates += 1
                self.stats.duplicate_sources_removed += 1
                await self._skipped(ctx, hit.url, "duplicate", result.intent)
                continue
            page = await self._page_data(record, source, identity, result.intent, as_of, ctx)
            if len(record.text) < MIN_TEXT_CHARS and not page.has_data:
                await self._reject(source, REASON_THIN, ctx, result, fetch_ms=fetch_ms)
                continue
            if page.has_data:
                source.metadata["data"] = page.summary()
            await self._keep(
                source,
                ctx,
                result,
                record,
                fetch_ms=fetch_ms,
                text_chars=text_chars,
                preview=page.preview(),
            )
            await self._use_page_data(page, source, identity, result.intent, as_of, ctx)

    async def _page_data(
        self,
        record: EvidenceRecord,
        source: SourceRecord,
        identity: InstrumentIdentity,
        intent: str,
        as_of: datetime,
        ctx: AnalysisContext,
    ) -> PageData:
        """Laya's choices among the page's candidate tables and the deterministic reading of
        them. A parser failure on hostile content means no data from the page, never a failed
        analysis; a cancellation propagates."""
        try:
            page = await identify_page_data(
                self.laya,
                record,
                source_id=source.source_id,
                symbol=identity.symbol,
                role=role_for(intent),
                as_of=as_of.date(),
                ctx=ctx,
            )
        except AnalysisError:
            raise
        except Exception:
            log.exception("reading tables failed for %s", record.final_url or record.url)
            return PageData()
        self.decisions.extend(page.decisions)
        for note in page.notes:
            self.market.note(f"{domain_of(source.url)}: {note}")
        return page

    async def _use_page_data(
        self,
        page: PageData,
        source: SourceRecord,
        identity: InstrumentIdentity,
        intent: str,
        as_of: datetime,
        ctx: AnalysisContext,
    ) -> None:
        """Keep what the page yielded: its price series (``market.series`` when it becomes the
        company's or the market's series) and its figures."""
        if page.prices is not None:
            if role_for(intent) == BROAD_MARKET:
                payload = self.market.offer_series(
                    BROAD_MARKET, BROAD_MARKET_SYMBOL, BROAD_MARKET_NAME, source, page.prices
                )
            else:
                payload = self.market.offer_series(
                    COMPANY,
                    identity.symbol,
                    identity.name or identity.symbol,
                    source,
                    page.prices,
                    identity.exchange,
                )
            if payload is not None:
                await ctx.event("market.series", **payload.model_dump(mode="json"))
        if page.figures is not None:
            self.market.add_figures(source, page.figures, as_of.date())

    def _source_from_hit(
        self, hit: SearchResult, identity: InstrumentIdentity, result: RoundResult, as_of: datetime
    ) -> SourceRecord:
        source_type, publisher, redistribution, note = classify_source(hit.url)
        return SourceRecord(
            source_id=stable_id("src", canonical_url(hit.url)),
            url=hit.url,
            title=hit.title,
            publisher=publisher,
            source_type=source_type,
            published_at=hit.published_at,
            retrieved_at=utcnow(),
            symbol=identity.symbol,
            excerpt=hit.snippet[:600],
            extraction_method="search_result",
            freshness=classify_freshness(hit.published_at, as_of),
            redistribution=redistribution,
            terms_note=note,
            research_intent=result.intent,
            metadata={"engine": hit.engine, "score": hit.score},
        )
