"""ResearchRunner: executes one ``PlannedQuery`` at a time on behalf of the EquityAnalyzer.

The ``ResearchProvider`` stays pure (search / open / extract). This runner adds what the
finance layer owns: leakage guard against ``as_of``, paywall and thin-content rejection,
deduplication, source classification, provenance records, budget enforcement, counters and
observable events.

Events emitted HERE (the analyzer emits ``research.started`` / ``research.completed``):

* ``research.query``            {intent, kind, query, label, round}
* ``research.source_found``     SourceRecord.public_view() + {intent, round}
* ``research.source_rejected``  {url, title, reason, intent, round}; ``reason`` is always one
  of ``REJECTION_REASONS`` (fixed keywords, never transport error text)

Call sequence for the orchestrator::

    provider, edgar, prices = build_research_stack(settings)
    runner = ResearchRunner(provider, edgar, prices, settings, request.budget)
    for round_no in range(budget.max_rounds):
        runner.begin_round()
        for planned in build_queries(intent, identity, horizon, as_of, gaps):
            result = await runner.execute(planned, identity, as_of, ctx)
            ...  # accumulate result.sources / evidence / facts_rows / price_series
        if runner.budget_exhausted or laya says stop: break
    runner.finish("laya_stop" | "max_rounds" | ...)
    stats = runner.stats
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
from bayanalytics.research.dedup import Deduplicator
from bayanalytics.research.edgar import (
    COMPANY_FACTS_URL,
    DEFAULT_FILING_FORMS,
    SUBMISSIONS_URL,
    EdgarClient,
    EdgarSubmissions,
    normalize_cik,
    select_filings,
)
from bayanalytics.research.extract import content_hash
from bayanalytics.research.intents import PlannedQuery
from bayanalytics.research.market import (
    domain_of,
    facts_preview,
    price_preview,
    series_payload,
    short_date,
    submissions_preview,
)
from bayanalytics.research.prices import StooqPrices, benchmark_stooq_symbol, select_benchmarks
from bayanalytics.research.provider import (
    EvidenceRecord,
    ResearchProvider,
    ResearchProviderError,
    SearchResult,
)
from bayanalytics.research.sources import (
    canonical_url,
    classify_freshness,
    classify_source,
    source_record_from_evidence,
)
from bayanalytics.schemas.common import ErrorCode, stable_id, utcnow
from bayanalytics.schemas.evidence import BenchmarkRef, PriceSeries, SourceRecord
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
    facts_rows: list[dict[str, Any]] = Field(default_factory=list)
    price_series: PriceSeries | None = None
    benchmark_series: dict[str, PriceSeries] = Field(default_factory=dict)
    benchmark_refs: list[BenchmarkRef] = Field(default_factory=list)
    submissions: EdgarSubmissions | None = None
    rejected: list[tuple[SourceRecord, str]] = Field(default_factory=list)
    duplicates: int = 0
    notes: list[str] = Field(default_factory=list)
    budget_exhausted: bool = False


class ResearchRunner:
    def __init__(
        self,
        provider: ResearchProvider,
        edgar: EdgarClient,
        prices: StooqPrices,
        settings: Settings,
        budget: ResearchBudget,
    ) -> None:
        self.provider = provider
        self.edgar = edgar
        self.prices = prices
        self.settings = settings
        self.budget = budget
        self.stats = ResearchStats()
        self.dedup = Deduplicator()
        self.kept_sources = 0
        self._round = 0
        self._structured_ok = False

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
            if planned.kind == "search":
                await self._run_search(planned, identity, as_of, ctx, result)
            elif planned.kind == "edgar_submissions":
                await self._run_submissions(planned, identity, as_of, ctx, result)
            elif planned.kind == "edgar_companyfacts":
                await self._run_companyfacts(planned, identity, as_of, ctx, result)
            elif planned.kind == "prices":
                await self._run_prices(planned, identity, as_of, ctx, result)
            elif planned.kind == "benchmarks":
                await self._run_benchmarks(planned, identity, as_of, ctx, result)
            else:
                result.notes.append(f"unknown query kind {planned.kind}")
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

    async def _structured_rejected(
        self,
        ctx: AnalysisContext,
        url: str,
        title: str,
        reason: str,
        intent: str,
        fetch_ms: int,
    ) -> None:
        """A structured request (EDGAR) that failed: shown as not used, without counting it as a
        rejected source (it is tracked as a structured failure instead)."""
        await ctx.event(
            "research.source_rejected",
            url=url,
            title=title,
            reason=reason,
            domain=domain_of(url),
            fetch_ms=fetch_ms,
            intent=intent,
            round=self._round,
        )

    def _structured_failure(self, exc: Exception, planned: PlannedQuery, what: str) -> None:
        reason = reject_reason(exc)
        log.warning("structured research failed (%s, %s): %s", planned.kind, what, exc)
        if not self._structured_ok:
            raise AnalysisError(
                ErrorCode.RESEARCH_UNAVAILABLE,
                details={"kind": planned.kind, "stage": planned.kind, "reason": reason},
            ) from exc

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
            if len(record.text) < MIN_TEXT_CHARS:
                await self._reject(source, REASON_THIN, ctx, result, fetch_ms=fetch_ms)
                continue
            if self.dedup.seen_record(record, check_url=False):
                result.duplicates += 1
                self.stats.duplicate_sources_removed += 1
                await self._skipped(ctx, hit.url, "duplicate", result.intent)
                continue
            await self._keep(source, ctx, result, record, fetch_ms=fetch_ms, text_chars=text_chars)

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

    # -- EDGAR ----------------------------------------------------------------------------

    def _cik(self, planned: PlannedQuery, identity: InstrumentIdentity) -> str | None:
        cik = planned.params.get("cik") or identity.cik
        return str(cik) if cik not in (None, "") else None

    async def _run_submissions(
        self,
        planned: PlannedQuery,
        identity: InstrumentIdentity,
        as_of: datetime,
        ctx: AnalysisContext,
        result: RoundResult,
    ) -> None:
        cik = self._cik(planned, identity)
        if cik is None:
            result.notes.append("no CIK for identity; EDGAR submissions skipped")
            return
        self.stats.queries_issued += 1
        url = SUBMISSIONS_URL.format(cik=normalize_cik(cik))
        await self._fetching(ctx, url, "edgar_submissions", result.intent, "SEC EDGAR submissions")
        started = time.perf_counter()
        try:
            subs = await self.edgar.submissions(cik)
        except ResearchProviderError as exc:
            await self._structured_rejected(
                ctx,
                url,
                "SEC EDGAR submissions",
                reject_reason(exc),
                result.intent,
                _elapsed_ms(started),
            )
            self._structured_failure(exc, planned, f"submissions CIK {cik}")
            result.notes.append(f"EDGAR submissions unavailable ({reject_reason(exc)})")
            return
        fetch_ms = _elapsed_ms(started)
        self._structured_ok = True
        self.stats.sources_fetched += 1
        result.submissions = subs
        if self._check_budget():
            await self._skipped(ctx, url, "budget", result.intent)
            return
        await self._keep(
            self._submissions_source(subs, identity, as_of, result.intent),
            ctx,
            result,
            fetch_ms=fetch_ms,
            preview=submissions_preview(subs, as_of),
        )
        forms = tuple(planned.params.get("forms") or DEFAULT_FILING_FORMS)
        limit = int(planned.params.get("limit") or 6)
        fetch_budget = self.budget.max_fetch_per_round
        for filing in select_filings(subs, forms, limit, as_of):
            if self._check_budget():
                break
            ctx.check_cancelled()
            excerpt = ""
            filing_ms: int | None = None
            if fetch_budget > 0:
                fetch_budget -= 1
                await self._fetching(
                    ctx,
                    filing.url,
                    "filing",
                    result.intent,
                    f"{filing.form} filed {short_date(filing.filing_date)}",
                )
                started = time.perf_counter()
                try:
                    excerpt = await self.edgar.fetch_filing_excerpt(filing)
                    self.stats.sources_fetched += 1
                except ResearchProviderError as exc:
                    log.debug("filing excerpt %s unavailable: %s", filing.url, exc)
                    result.notes.append(
                        f"{filing.form} {filing.accession}: excerpt unavailable"
                        f" ({reject_reason(exc)})"
                    )
                filing_ms = _elapsed_ms(started)
            source = self.edgar.filing_source_record(
                filing,
                symbol=identity.symbol,
                as_of=as_of,
                fiscal_year_end=subs.fiscal_year_end or identity.fiscal_year_end,
                intent=result.intent,
                excerpt=excerpt,
            )
            evidence = EvidenceRecord(
                url=filing.index_url,
                final_url=filing.url,
                title=source.title,
                publisher="SEC EDGAR",
                source_type="regulatory_filing",
                published_at=source.published_at,
                retrieved_at=source.retrieved_at,
                text=excerpt,
                excerpt=excerpt,
                content_hash=content_hash(excerpt),
                extraction_method="edgar_submissions",
                structured={
                    "form": filing.form,
                    "filing_date": filing.filing_date.isoformat(),
                    "report_date": filing.report_date.isoformat() if filing.report_date else None,
                    "accession": filing.accession,
                    "primary_document": filing.primary_document,
                    "cik": filing.cik,
                },
            )
            await self._keep(
                source,
                ctx,
                result,
                evidence,
                fetch_ms=filing_ms,
                text_chars=len(excerpt) if excerpt else None,
            )

    def _submissions_source(
        self, subs: EdgarSubmissions, identity: InstrumentIdentity, as_of: datetime, intent: str
    ) -> SourceRecord:
        latest = subs.latest_filings(forms=(), limit=1, as_of=as_of)
        published = None
        if latest:
            d = latest[0].filing_date
            published = datetime(d.year, d.month, d.day, tzinfo=as_of.tzinfo)
        tickers = ", ".join(subs.tickers) or "?"
        summary = (
            f"{subs.name} (CIK {subs.cik}); SIC {subs.sic or '?'} {subs.sic_description or ''};"
            f" fiscal year end {subs.fiscal_year_end or '?'}; tickers {tickers}"
        ).strip()
        return SourceRecord(
            source_id=stable_id("src", canonical_url(subs.url)),
            url=subs.url,
            title="SEC EDGAR submissions",
            publisher="SEC EDGAR",
            source_type="regulatory_filing",
            published_at=published,
            retrieved_at=subs.retrieved_at,
            symbol=identity.symbol,
            excerpt=summary[:600],
            extraction_method="json",
            freshness=classify_freshness(published, as_of),
            redistribution="allowed",
            terms_note="US government work; SEC fair-access policy applies to retrieval",
            research_intent=intent,
            metadata={
                "cik": subs.cik,
                "sic": subs.sic,
                "fiscal_year_end": subs.fiscal_year_end,
                "former_names": subs.name_history,
                "from_cache": subs.from_cache,
            },
        )

    async def _run_companyfacts(
        self,
        planned: PlannedQuery,
        identity: InstrumentIdentity,
        as_of: datetime,
        ctx: AnalysisContext,
        result: RoundResult,
    ) -> None:
        cik = self._cik(planned, identity)
        if cik is None:
            result.notes.append("no CIK for identity; EDGAR company facts skipped")
            return
        self.stats.queries_issued += 1
        url = COMPANY_FACTS_URL.format(cik=normalize_cik(cik))
        await self._fetching(
            ctx, url, "edgar_companyfacts", result.intent, "SEC XBRL company facts"
        )
        started = time.perf_counter()
        try:
            facts = await self.edgar.company_facts(
                cik, as_of=as_of, symbol=identity.symbol, intent=result.intent
            )
        except ResearchProviderError as exc:
            await self._structured_rejected(
                ctx,
                url,
                "SEC XBRL company facts",
                reject_reason(exc),
                result.intent,
                _elapsed_ms(started),
            )
            self._structured_failure(exc, planned, f"companyfacts CIK {cik}")
            result.notes.append(f"EDGAR company facts unavailable ({reject_reason(exc)})")
            return
        fetch_ms = _elapsed_ms(started)
        self._structured_ok = True
        self.stats.sources_fetched += 1
        result.facts_rows = facts.rows
        if facts.rows_filtered_after_as_of:
            result.notes.append(
                f"{facts.rows_filtered_after_as_of} XBRL rows filed after as_of were dropped"
            )
        if self._check_budget():
            await self._skipped(ctx, url, "budget", result.intent)
            return
        await self._keep(
            facts.source, ctx, result, fetch_ms=fetch_ms, preview=facts_preview(facts.rows)
        )

    # -- prices / benchmarks --------------------------------------------------------------

    def _days(self, planned: PlannedQuery) -> int:
        days = planned.params.get("days")
        return int(days) if days else int(self.settings.research_price_history_days)

    async def _run_prices(
        self,
        planned: PlannedQuery,
        identity: InstrumentIdentity,
        as_of: datetime,
        ctx: AnalysisContext,
        result: RoundResult,
    ) -> None:
        from bayanalytics.research.prices import to_stooq_symbol

        symbol = planned.params.get("symbol") or to_stooq_symbol(identity.symbol, identity.exchange)
        self.stats.queries_issued += 1
        url = self.prices.url_for(symbol)
        await self._fetching(
            ctx, url, "prices", result.intent, f"Stooq daily prices {identity.symbol}"
        )
        started = time.perf_counter()
        try:
            series, source = await self.prices.daily_with_source(
                symbol,
                as_of,
                self._days(planned),
                label=identity.symbol,
                exchange=planned.params.get("exchange") or identity.exchange,
                intent=result.intent,
            )
        except ResearchProviderError as exc:
            reason = reject_reason(exc)
            log.debug("prices for %s rejected (%s): %s", symbol, reason, exc)
            await self._reject(
                self._price_failure_source(symbol, identity, result.intent),
                reason,
                ctx,
                result,
                fetch_ms=_elapsed_ms(started),
            )
            return
        fetch_ms = _elapsed_ms(started)
        self.stats.sources_fetched += 1
        result.price_series = series
        preview = None
        if self.settings.price_display:
            payload = series_payload(
                series, "company", identity.symbol, identity.name or identity.symbol
            )
            await ctx.event("market.series", **payload.model_dump(mode="json"))
            preview = price_preview(series)
        if self._check_budget():
            await self._skipped(ctx, url, "budget", result.intent)
            return
        await self._keep(source, ctx, result, fetch_ms=fetch_ms, preview=preview)

    async def _run_benchmarks(
        self,
        planned: PlannedQuery,
        identity: InstrumentIdentity,
        as_of: datetime,
        ctx: AnalysisContext,
        result: RoundResult,
    ) -> None:
        sic = planned.params.get("sic") or identity.sic
        refs = select_benchmarks(sic)
        result.benchmark_refs = refs
        for ref in refs:
            ctx.check_cancelled()
            stooq_symbol = benchmark_stooq_symbol(ref.symbol)
            self.stats.queries_issued += 1
            url = self.prices.url_for(stooq_symbol)
            await self._fetching(
                ctx, url, "benchmark", result.intent, f"Stooq daily prices {ref.name}"
            )
            started = time.perf_counter()
            try:
                series, source = await self.prices.daily_with_source(
                    stooq_symbol, as_of, self._days(planned), label=ref.name, intent=result.intent
                )
            except ResearchProviderError as exc:
                reason = reject_reason(exc)
                log.debug("benchmark %s rejected (%s): %s", stooq_symbol, reason, exc)
                await self._reject(
                    self._price_failure_source(stooq_symbol, identity, result.intent),
                    reason,
                    ctx,
                    result,
                    fetch_ms=_elapsed_ms(started),
                )
                continue
            fetch_ms = _elapsed_ms(started)
            self.stats.sources_fetched += 1
            source.metadata["benchmark_role"] = ref.role
            source.metadata["benchmark_reason"] = ref.reason
            result.benchmark_series[ref.symbol] = series
            preview = None
            if self.settings.price_display:
                payload = series_payload(series, ref.role, ref.symbol, ref.name)
                await ctx.event("market.series", **payload.model_dump(mode="json"))
                preview = price_preview(series)
            if self._check_budget():
                await self._skipped(ctx, url, "budget", result.intent)
                break
            await self._keep(source, ctx, result, fetch_ms=fetch_ms, preview=preview)

    def _price_failure_source(
        self, stooq_symbol: str, identity: InstrumentIdentity, intent: str
    ) -> SourceRecord:
        return SourceRecord(
            source_id=stable_id("src", canonical_url(self.prices.url_for(stooq_symbol))),
            url=self.prices.url_for(stooq_symbol),
            title=f"Stooq daily prices {stooq_symbol}",
            publisher="Stooq",
            source_type="market_data",
            retrieved_at=utcnow(),
            symbol=identity.symbol,
            extraction_method="csv",
            redistribution="metadata_only",
            research_intent=intent,
        )
