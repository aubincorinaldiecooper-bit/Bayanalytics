"""ResearchRunner: executes one ``PlannedQuery`` at a time on behalf of the EquityAnalyzer.

The ``ResearchProvider`` stays pure (search / open / extract). This runner adds what the
finance layer owns: leakage guard against ``as_of``, paywall and thin-content rejection,
deduplication, source classification, provenance records, budget enforcement, counters and
observable events.

Events emitted HERE (the analyzer emits ``research.started`` / ``research.completed``):

* ``research.query``            {intent, kind, query, label, round}
* ``research.source_found``     SourceRecord.public_view() + {intent, round}
* ``research.source_rejected``  {url, title, reason, intent, round}

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
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.base import InstrumentIdentity, ResearchBudget
from bayanalytics.research.dedup import Deduplicator
from bayanalytics.research.edgar import (
    DEFAULT_FILING_FORMS,
    EdgarClient,
    EdgarSubmissions,
    select_filings,
)
from bayanalytics.research.extract import content_hash
from bayanalytics.research.intents import PlannedQuery
from bayanalytics.research.prices import StooqPrices, benchmark_stooq_symbol, select_benchmarks
from bayanalytics.research.provider import (
    EvidenceRecord,
    ResearchProvider,
    ResearchProviderError,
    SearchResult,
)
from bayanalytics.research.sources import (
    classify_freshness,
    classify_source,
    source_record_from_evidence,
)
from bayanalytics.schemas.common import ErrorCode, new_id, utcnow
from bayanalytics.schemas.evidence import BenchmarkRef, PriceSeries, SourceRecord
from bayanalytics.schemas.results import ResearchStats

log = logging.getLogger(__name__)

MIN_TEXT_CHARS = 200
REASON_LEAKAGE = "published_after_as_of"
REASON_PAYWALLED = "paywalled"
REASON_THIN = "thin_content"
REASON_FETCH_FAILED = "fetch_failed"
TERMINATION_MAX_SOURCES = "max_sources"


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
    ) -> None:
        self.kept_sources += 1
        result.sources.append(source)
        if evidence is not None:
            result.evidence.append(evidence)
        await ctx.event(
            "research.source_found", **source.public_view(), intent=result.intent, round=self._round
        )

    async def _reject(
        self, source: SourceRecord, reason: str, ctx: AnalysisContext, result: RoundResult
    ) -> None:
        source.rejected_reason = reason
        self.stats.sources_rejected += 1
        result.rejected.append((source, reason))
        await ctx.event(
            "research.source_rejected",
            url=source.url,
            title=source.title,
            reason=reason,
            intent=result.intent,
            round=self._round,
        )

    def _structured_failure(self, exc: Exception, planned: PlannedQuery, url: str) -> None:
        if not self._structured_ok:
            raise AnalysisError(
                ErrorCode.RESEARCH_UNAVAILABLE,
                details={"kind": planned.kind, "url": url, "reason": str(exc)[:300]},
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
            result.notes.append(f"search failed: {exc}")
            log.info("search failed for %r: %s", planned.query, exc)
            return
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
                continue
            self.dedup.seen(hit.url)
            attempts += 1
            try:
                record = await self.provider.extract(hit.url)
            except ResearchProviderError as exc:
                await self._reject(
                    self._source_from_hit(hit, identity, result, as_of),
                    f"{REASON_FETCH_FAILED}: {str(exc)[:160]}",
                    ctx,
                    result,
                )
                continue
            self.stats.sources_fetched += 1
            if record.published_at is None and hit.published_at is not None:
                record.published_at = hit.published_at
            source = source_record_from_evidence(record, identity.symbol, result.intent, as_of)
            record.source_type = source.source_type
            if record.metadata.get("paywalled"):
                await self._reject(source, REASON_PAYWALLED, ctx, result)
                continue
            if record.published_at is not None and record.published_at > as_of:
                await self._reject(source, REASON_LEAKAGE, ctx, result)
                continue
            if len(record.text) < MIN_TEXT_CHARS:
                await self._reject(source, REASON_THIN, ctx, result)
                continue
            if self.dedup.seen_record(record, check_url=False):
                result.duplicates += 1
                self.stats.duplicate_sources_removed += 1
                continue
            await self._keep(source, ctx, result, record)

    def _source_from_hit(
        self, hit: SearchResult, identity: InstrumentIdentity, result: RoundResult, as_of: datetime
    ) -> SourceRecord:
        source_type, publisher, redistribution, note = classify_source(hit.url)
        return SourceRecord(
            source_id=new_id("src"),
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
        try:
            subs = await self.edgar.submissions(cik)
        except ResearchProviderError as exc:
            self._structured_failure(exc, planned, f"submissions CIK {cik}")
            result.notes.append(f"EDGAR submissions unavailable: {exc}")
            return
        self._structured_ok = True
        self.stats.sources_fetched += 1
        result.submissions = subs
        if self._check_budget():
            return
        await self._keep(
            self._submissions_source(subs, identity, as_of, result.intent), ctx, result
        )
        forms = tuple(planned.params.get("forms") or DEFAULT_FILING_FORMS)
        limit = int(planned.params.get("limit") or 6)
        fetch_budget = self.budget.max_fetch_per_round
        for filing in select_filings(subs, forms, limit, as_of):
            if self._check_budget():
                break
            ctx.check_cancelled()
            excerpt = ""
            if fetch_budget > 0:
                fetch_budget -= 1
                try:
                    excerpt = await self.edgar.fetch_filing_excerpt(filing)
                    self.stats.sources_fetched += 1
                except ResearchProviderError as exc:
                    result.notes.append(
                        f"{filing.form} {filing.accession}: excerpt unavailable ({exc})"
                    )
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
            await self._keep(source, ctx, result, evidence)

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
            source_id=new_id("src"),
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
        try:
            facts = await self.edgar.company_facts(
                cik, as_of=as_of, symbol=identity.symbol, intent=result.intent
            )
        except ResearchProviderError as exc:
            self._structured_failure(exc, planned, f"companyfacts CIK {cik}")
            result.notes.append(f"EDGAR company facts unavailable: {exc}")
            return
        self._structured_ok = True
        self.stats.sources_fetched += 1
        result.facts_rows = facts.rows
        if facts.rows_filtered_after_as_of:
            result.notes.append(
                f"{facts.rows_filtered_after_as_of} XBRL rows filed after as_of were dropped"
            )
        if self._check_budget():
            return
        await self._keep(facts.source, ctx, result)

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
            await self._reject(
                self._price_failure_source(symbol, identity, result.intent),
                f"{REASON_FETCH_FAILED}: {str(exc)[:160]}",
                ctx,
                result,
            )
            return
        self.stats.sources_fetched += 1
        result.price_series = series
        if self._check_budget():
            return
        await self._keep(source, ctx, result)

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
            try:
                series, source = await self.prices.daily_with_source(
                    stooq_symbol, as_of, self._days(planned), label=ref.name, intent=result.intent
                )
            except ResearchProviderError as exc:
                await self._reject(
                    self._price_failure_source(stooq_symbol, identity, result.intent),
                    f"{REASON_FETCH_FAILED}: {str(exc)[:160]}",
                    ctx,
                    result,
                )
                continue
            self.stats.sources_fetched += 1
            source.metadata["benchmark_role"] = ref.role
            source.metadata["benchmark_reason"] = ref.reason
            result.benchmark_series[ref.symbol] = series
            if self._check_budget():
                break
            await self._keep(source, ctx, result)

    def _price_failure_source(
        self, stooq_symbol: str, identity: InstrumentIdentity, intent: str
    ) -> SourceRecord:
        return SourceRecord(
            source_id=new_id("src"),
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
