"""EquityAnalyzer: the public-equity adapter (AGENT.md sections 21, 23, 27, 28).

Owns the research strategy for one analysis: a deterministic seed plan by horizon, then a
Laya-directed loop over bounded research intents until evidence is sufficient, retrieval stops
changing the picture, the budget is spent, or the user cancels. Retrieval itself is done by the
research runner (with Laya choosing which hits to open and which tables, columns and lines of a
page hold the data); normalization, Laya question construction, calculation selection and the
Spark bundle are built here from the accumulated evidence: the web pages' text, and the price
series and figures read from their tables, each keeping the page it came from.

A question that names a company instead of a ticker is resolved here by one web search and a
Laya choice among the tickers the results name (stage ``instrument_resolution``).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from bayanalytics.calculations.primitives import annualized_volatility, growth_rate, margin
from bayanalytics.calculations.registry import CALCULATION_PACKS, run_pack
from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.base import (
    AnalysisRequest,
    CalculatedMetrics,
    InstrumentIdentity,
    LayaDecisions,
    SparkEvidenceBundle,
)
from bayanalytics.instruments.identity import (
    MAX_NAME_CANDIDATES,
    MULTIPLE_TICKERS_MESSAGE,
    TICKER_REQUIRED_MESSAGE,
    InstrumentResolver,
    NameCandidate,
    company_phrase,
    name_query,
    ticker_candidates,
)
from bayanalytics.instruments.questions import check_requirements, operand_gaps
from bayanalytics.laya.schemas import (
    INSTRUMENT_CHOICE_KEY,
    STAGE_INSTRUMENT_RESOLUTION,
    history_segment_questions,
    horizon_context_questions,
    instrument_choice_questions,
    overall_scan_questions,
    research_plan_questions,
    text_evidence_questions,
)
from bayanalytics.laya.wrapper import LayaFinanceWrapper
from bayanalytics.normalization import NORMALIZATION_VERSION
from bayanalytics.normalization.corporate_actions import comparable_periods
from bayanalytics.normalization.facts import (
    build_facts,
    detect_stale_mix,
    freshness_summary,
)
from bayanalytics.normalization.periods import label as period_label
from bayanalytics.normalization.sessions import label_series
from bayanalytics.research.intents import (
    PlannedQuery,
    ResearchIntent,
    build_queries,
    gap_to_intent,
    seed_plan,
)
from bayanalytics.research.market import MarketData
from bayanalytics.research.provider import (
    EvidenceRecord,
    ResearchProvider,
    ResearchProviderError,
    SearchResult,
)
from bayanalytics.research.runner import REASON_SEARCH_FAILED, ResearchRunner, RoundResult
from bayanalytics.research.selection import ask_shrinking, chosen_option
from bayanalytics.research.sources import domain_of, web_pages
from bayanalytics.schemas.common import ErrorCode, source_rank, stable_id
from bayanalytics.schemas.decisions import ChoiceAnswer, LayaDecision, LayaQuestionSet, NoulAnswer
from bayanalytics.schemas.evidence import (
    CorporateAction,
    EventSegment,
    NormalizedEvidence,
    NormalizedFact,
    Period,
    PriceSeries,
    SourceRecord,
)
from bayanalytics.schemas.questions import AnalyticalRequirements, RequirementsReport
from bayanalytics.schemas.requests import InstrumentRef
from bayanalytics.schemas.results import ResearchStats

log = logging.getLogger(__name__)

ResolverFactory = Callable[[], Awaitable[InstrumentResolver]]

_MAX_TEXT_EVIDENCE_FOR_LAYA = 8
_MAX_SEGMENTS_FOR_LAYA = 8
_SUFFICIENT_THRESHOLD = 0.7
INSTRUMENT_CHOICE_MIN_CONFIDENCE = 0.6
"""Laya's confidence needed to accept a company it chose among the searched candidates."""
RESOLVE_INTENT = "resolve_instrument"
"""The ``intent`` the name lookup's research events carry (not a research intent Laya picks)."""


@dataclass
class RetrievalState:
    sources: list[SourceRecord] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    rejected: list[tuple[SourceRecord, str]] = field(default_factory=list)
    executed: list[str] = field(default_factory=list)
    rounds: int = 0
    termination_reason: str | None = None
    source_by_id: dict[str, SourceRecord] = field(default_factory=dict)
    evidence_by_source: dict[str, EvidenceRecord] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)  # provider outages, dropped rows, gaps
    searches_issued: int = 0
    queries_failed: int = 0

    def absorb(self, result: RoundResult) -> None:
        for source in result.sources:
            if source.source_id not in self.source_by_id:
                self.source_by_id[source.source_id] = source
                self.sources.append(source)
        for record in result.evidence:
            self.evidence.append(record)
            sid = record.metadata.get("source_id") if record.metadata else None
            if sid:
                self.evidence_by_source[sid] = record
        self.rejected.extend(result.rejected)
        for note in result.notes:
            if note not in self.notes:
                self.notes.append(note)
            if note.lower().startswith(REASON_SEARCH_FAILED):
                self.queries_failed += 1


class EquityAnalyzer:
    """One instance per analysis (it accumulates retrieval state)."""

    def __init__(
        self,
        settings: Settings,
        research_provider: ResearchProvider,
        laya: LayaFinanceWrapper,
        resolver_factory: ResolverFactory,
    ) -> None:
        self.settings = settings
        self.provider = research_provider
        self.laya = laya
        self._resolver_factory = resolver_factory
        self.state = RetrievalState()
        self.stats = ResearchStats()
        self.identity: InstrumentIdentity | None = None
        self.instrument_ref: InstrumentRef | None = None
        self.research_decisions: list[LayaDecision] = []
        self.request_as_of: datetime = datetime.now(tz=UTC)
        # What the question requires (set from the request in retrieve) and, after the
        # calculations, which of it was met.
        self.requirements: AnalyticalRequirements | None = None
        self.requirements_report: RequirementsReport | None = None
        self.operand_gaps: list[str] = []
        # Price series and figures read from the pages research kept (with their pages).
        self.market = MarketData()
        self.market_labels: dict[Any, str] = {}
        self.identity_decisions: list[LayaDecision] = []

    # ------------------------------------------------------------------ identify
    async def identify(self, user_input: str, ctx: AnalysisContext) -> InstrumentIdentity:
        """The ticker the analyst gave; without one, the company the question names, looked up
        by web search and chosen by Laya among the tickers the results name."""
        resolver = await self._resolver_factory()
        try:
            identity = resolver.resolve(user_input, self.instrument_ref)
        except AnalysisError as exc:
            phrase = company_phrase(user_input)
            if phrase is None or not self._name_lookup_applies(exc):
                raise
            identity = await self._resolve_by_name(phrase, ctx)
        self.identity = identity
        return identity

    def _name_lookup_applies(self, exc: AnalysisError) -> bool:
        return (
            exc.code == ErrorCode.AMBIGUOUS_INSTRUMENT
            and (exc.details or {}).get("reason") == "ticker_required"
            and self.instrument_ref is None
            and self.provider is not None
            and bool(getattr(self.provider, "search_configured", True))
        )

    async def _search(self, query: str) -> list[SearchResult]:
        search_with = getattr(self.provider, "search_with", None)
        if callable(search_with):
            return await search_with(query)
        return await self.provider.search(query)

    async def _resolve_by_name(self, phrase: str, ctx: AnalysisContext) -> InstrumentIdentity:
        """One topic-only web search for the phrase, the tickers its results name (ranked by
        distinct websites), and Laya's choice among them. Unsure, tied or no candidates:
        ``AMBIGUOUS_INSTRUMENT`` with the candidates as found (or ``ticker_required``)."""
        query = name_query(phrase)
        await ctx.event(
            "research.query",
            intent=RESOLVE_INTENT,
            kind="search",
            query=query,
            label=f"ticker lookup for {phrase}",
            round=0,
        )
        ticker_required = AnalysisError(
            ErrorCode.AMBIGUOUS_INSTRUMENT,
            TICKER_REQUIRED_MESSAGE,
            details={"reason": "ticker_required", "candidates": []},
        )
        try:
            results = await self._search(query)
        except ResearchProviderError as exc:
            log.warning("ticker lookup search failed: %s", exc)
            await ctx.event(
                "research.search_results",
                query=query,
                intent=RESOLVE_INTENT,
                round=0,
                total=0,
                failed=True,
                hits=[],
            )
            raise ticker_required from exc
        await ctx.event(
            "research.search_results",
            query=query,
            intent=RESOLVE_INTENT,
            round=0,
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
        candidates = ticker_candidates(results)[:MAX_NAME_CANDIDATES]
        if not candidates:
            raise ticker_required
        chosen, confidence = await self._choose_company(phrase, candidates, ctx)
        tied = chosen is not None and any(
            other is not chosen and len(other.domains) == len(chosen.domains)
            for other in candidates
        )
        if chosen is None or confidence < INSTRUMENT_CHOICE_MIN_CONFIDENCE or tied:
            raise AnalysisError(
                ErrorCode.AMBIGUOUS_INSTRUMENT,
                MULTIPLE_TICKERS_MESSAGE,
                details={
                    "reason": "multiple_companies",
                    "candidates": [c.as_candidate().model_dump() for c in candidates],
                },
            )
        return InstrumentIdentity(
            symbol=chosen.symbol,
            exchange=chosen.exchange,
            name=chosen.name,
            confidence=confidence,
            resolution_method="name_search",
        )

    async def _choose_company(
        self, phrase: str, candidates: list[NameCandidate], ctx: AnalysisContext
    ) -> tuple[NameCandidate | None, float]:
        """Laya's choice among the candidates (``None`` when Laya declines, is unavailable, or
        there is no Laya)."""
        if self.laya is None:
            return None, 0.0
        options = [(c.symbol, c.name or c.symbol) for c in candidates]
        state = {
            "company": phrase,
            "candidates": [
                {"option": c.symbol, "name": c.name, "sites": len(c.domains)} for c in candidates
            ],
        }
        decisions = await ask_shrinking(
            self.laya,
            STAGE_INSTRUMENT_RESOLUTION,
            state,
            lambda n: instrument_choice_questions(options[:n]),
            len(options),
            stable_id("name", phrase),
            ctx,
        )
        self.identity_decisions.extend(decisions or [])
        choice, confidence = chosen_option(decisions, INSTRUMENT_CHOICE_KEY)
        return next((c for c in candidates if c.symbol == choice), None), confidence

    # ------------------------------------------------------------------ retrieve
    async def retrieve(
        self, identity: InstrumentIdentity, request: AnalysisRequest, ctx: AnalysisContext
    ) -> list[SourceRecord]:
        budget = request.budget
        self.request_as_of = request.as_of
        self.requirements = request.requirements
        runner = ResearchRunner(
            self.provider,
            self.settings,
            budget,
            laya=self.laya,
            market=self.market,
            decisions=self.research_decisions,
        )
        plan: list[ResearchIntent] = list(seed_plan(request.resolved_horizon, request.requirements))
        gaps: list[str] = []
        with ctx.timers.span("retrieval"):
            try:
                async with asyncio.timeout(budget.timeout_s):
                    await self._research_loop(runner, identity, request, ctx, plan, gaps)
            except TimeoutError:
                self.state.termination_reason = "timeout"
                log.warning("research budget timeout after %.0fs", budget.timeout_s)
            gaps = self.compute_gaps(request.resolved_horizon, request.as_of)
        runner_stats = runner.finish(self.state.termination_reason or "max_rounds")
        self.stats = self._merge_stats(runner_stats, gaps)
        self.stats.termination_reason = self.state.termination_reason
        self.stats.search_rounds = self.state.rounds
        self.stats.intents = list(self.state.executed)
        self.stats.retrieval_total_ms = ctx.timers.elapsed_ms.get("retrieval", 0.0)
        await ctx.event("research.completed", **self.stats.model_dump(mode="json"))
        return list(self.state.sources)

    async def _research_loop(
        self,
        runner: ResearchRunner,
        identity: InstrumentIdentity,
        request: AnalysisRequest,
        ctx: AnalysisContext,
        plan: list[ResearchIntent],
        gaps: list[str],
    ) -> None:
        budget = request.budget
        if budget.max_rounds > 0:
            while self.state.rounds < budget.max_rounds:
                ctx.check_cancelled()
                self.state.rounds += 1
                round_no = self.state.rounds
                runner.begin_round()
                await ctx.event(
                    "research.started",
                    round=round_no,
                    intents=[str(i) for i in plan],
                    evidence_gaps=gaps,
                    **_interpretation_view(request.requirements),
                )
                for intent in plan:
                    if intent == ResearchIntent.stop_research:
                        continue
                    query_gaps = gaps
                    if intent == ResearchIntent.retrieve_missing_metric and self.operand_gaps:
                        # A metric search spells the gap out: the operands the question needs
                        # go first, ahead of the loop's coverage labels.
                        query_gaps = [g for g in gaps if g in self.operand_gaps] + [
                            g for g in gaps if g not in self.operand_gaps
                        ]
                    for planned in build_queries(
                        intent, identity, request.resolved_horizon, request.as_of, query_gaps
                    ):
                        ctx.check_cancelled()
                        await self._execute(runner, planned, identity, request, ctx, round_no)
                    self.state.executed.append(str(intent))
                    if runner.budget_exhausted or len(self.state.sources) >= budget.max_sources:
                        break
                gaps = self.compute_gaps(request.resolved_horizon, request.as_of)
                self.stats = self._merge_stats(runner.stats, gaps)
                if runner.budget_exhausted or len(self.state.sources) >= budget.max_sources:
                    self.state.termination_reason = "max_sources"
                    break
                decision_intent, sufficient = await self._plan_next(identity, request, gaps, ctx)
                if sufficient >= _SUFFICIENT_THRESHOLD:
                    self.state.termination_reason = "evidence_sufficient"
                    break
                if decision_intent == ResearchIntent.stop_research:
                    self.state.termination_reason = "laya_stop"
                    break
                if str(decision_intent) in self.state.executed:
                    # The only intent that could help was already executed: more rounds would
                    # re-issue identical queries (termination rule 2).
                    self.state.termination_reason = "no_new_evidence"
                    break
                plan = [decision_intent]
            else:
                self.state.termination_reason = "max_rounds"

    async def _execute(
        self,
        runner: ResearchRunner,
        planned: PlannedQuery,
        identity: InstrumentIdentity,
        request: AnalysisRequest,
        ctx: AnalysisContext,
        round_no: int,
    ) -> None:
        self.state.searches_issued += 1
        try:
            result = await runner.execute(planned, identity, request.as_of, ctx)
        except AnalysisError:
            raise
        except ResearchProviderError as exc:
            log.warning("research query failed (%s): %s", planned.label, exc)
            return
        self.state.absorb(result)

    def _merge_stats(self, runner_stats: ResearchStats, gaps: list[str]) -> ResearchStats:
        merged = runner_stats.model_copy()
        merged.evidence_gaps_remaining = len(gaps)
        merged.search_rounds = self.state.rounds
        merged.queries_failed = self.state.queries_failed
        return merged

    def compute_gaps(self, horizon: str, as_of: datetime) -> list[str]:
        """Topics the kept web pages do not cover yet, as evidence-gap labels."""
        gaps: list[str] = []
        covered = {s.research_intent for s in self.state.sources if s.research_intent}
        for label, intent in (
            ("earnings_history", ResearchIntent.retrieve_earnings_history),
            ("price_history", ResearchIntent.retrieve_price_history),
            ("sector_benchmark", ResearchIntent.retrieve_sector_benchmark),
            ("latest_filing", ResearchIntent.retrieve_latest_filing),
        ):
            if str(intent) not in covered:
                gaps.append(label)
        recent_news = [
            s
            for s in self.state.sources
            if s.source_type in {"financial_journalism", "earnings_release", "investor_relations"}
            and s.published_at is not None
            and (as_of - s.published_at).days <= 45
        ]
        if not recent_news:
            gaps.append("recent_news")
        if horizon in {"next_cycle", "multi_horizon"} and not any(
            "guidance" in (s.excerpt or "").lower() or "guidance" in s.title.lower()
            for s in self.state.sources
        ):
            gaps.append("guidance_history")
        if horizon in {"long_term", "multi_horizon"} and not any(
            "transcript" in s.title.lower() or "call" in s.title.lower() for s in self.state.sources
        ):
            gaps.append("management_commentary")
        # Operands the question requires that no page has supplied yet (figures by metric,
        # the company's price series, a benchmark series).
        self.operand_gaps = operand_gaps(
            self.requirements,
            self.market.metric_rows(),
            self.market.prices(),
            self.market.benchmarks(),
        )
        for gap in self.operand_gaps:
            if gap not in gaps:
                gaps.append(gap)
        return gaps

    async def _plan_next(
        self,
        identity: InstrumentIdentity,
        request: AnalysisRequest,
        gaps: list[str],
        ctx: AnalysisContext,
    ) -> tuple[ResearchIntent, float]:
        primary = [s for s in self.state.sources if s.is_primary]
        state = {
            "instrument": identity.symbol,
            "horizon": request.resolved_horizon,
            "round": self.state.rounds,
            "sources_count": len(self.state.sources),
            "primary_sources": len(primary),
            "evidence_gaps": gaps,
            "executed_intents": self.state.executed[-6:],
            "latest_sources": [
                {
                    "title": s.title[:80],
                    "type": s.source_type,
                    "date": s.published_at.date().isoformat() if s.published_at else None,
                }
                for s in self.state.sources[-5:]
            ],
        }
        question_set = LayaQuestionSet(
            stage="research_plan", state=state, questions=research_plan_questions()
        )
        await ctx.event(
            "laya.started", stage="research_plan", questions=len(question_set.questions)
        )
        decisions = await self.laya.ask(question_set, ctx)
        for decision in decisions:
            await ctx.event("laya.decision", **decision.event_view())
        await ctx.event("laya.completed", stage="research_plan", decisions=len(decisions))
        self.research_decisions.extend(decisions)
        intent = ResearchIntent.stop_research
        sufficient = 0.0
        stale_matters = 0.0
        for decision in decisions:
            if decision.decision_type == "research_intent" and isinstance(
                decision.answer, ChoiceAnswer
            ):
                try:
                    intent = ResearchIntent(decision.answer.choice)
                except ValueError:
                    intent = ResearchIntent.stop_research
            if decision.decision_type == "evidence_sufficient" and isinstance(
                decision.answer, NoulAnswer
            ):
                sufficient = decision.answer.noul
            if decision.decision_type == "stale_evidence_matters" and isinstance(
                decision.answer, NoulAnswer
            ):
                stale_matters = decision.answer.noul
        # Laya judged the staleness material: refresh coverage once, and say so.
        if (
            stale_matters >= 0.6
            and str(ResearchIntent.retrieve_recent_news) not in self.state.executed
        ):
            self.state.notes.append(
                "Laya judged the available evidence stale enough to matter; "
                "recent coverage was refreshed"
            )
            return ResearchIntent.retrieve_recent_news, min(sufficient, 0.5)
        if stale_matters >= 0.6:
            self.state.notes.append(
                "Laya judged the available evidence stale enough to matter (confidence "
                f"{stale_matters:.2f}); treat the assessment with caution"
            )
        # Laya's stop is honoured unless a gap remains whose intent has not been tried yet.
        if intent != ResearchIntent.stop_research or not gaps:
            return intent, sufficient
        for gap in gaps:
            candidate = gap_to_intent(gap)
            if str(candidate) not in self.state.executed:
                return candidate, sufficient
        # A required operand no page supplied: one metric search before stopping (the intent
        # the schema reserves for "a metric a calculation needs").
        if (
            any(gap in self.operand_gaps for gap in gaps)
            and str(ResearchIntent.retrieve_missing_metric) not in self.state.executed
        ):
            return ResearchIntent.retrieve_missing_metric, sufficient
        return ResearchIntent.stop_research, sufficient

    # ------------------------------------------------------------------ normalize
    async def normalize(
        self, records: list[SourceRecord], ctx: AnalysisContext
    ) -> NormalizedEvidence:
        """The pages' text plus what was read from their tables: figures become normalized facts
        (one row per page, so pages that disagree become conflicts), the company's price series
        and the S&P 500 series become the price evidence. The uncertainties say what was found
        and where, or that search returned no page with it."""
        assert self.identity is not None
        as_of = self._as_of
        with ctx.timers.span("normalization"):
            rows, alignment_notes, self.market_labels = self.market.fact_rows()
            fact_build = build_facts(rows, as_of)
            facts: list[NormalizedFact] = fact_build.facts
            company = self.market.prices()
            prices = label_series(company, as_of) if company is not None else None
            benchmarks = {
                role: label_series(series, as_of)
                for role, series in self.market.benchmarks().items()
            }
            uncertainties: list[str] = self.market.data_notes(
                self.identity.symbol, len(web_pages(records)), rows
            )
            uncertainties.extend(self.market.notes)
            uncertainties.extend(alignment_notes)
            uncertainties.extend(fact_build.notes)
            uncertainties.extend(self.state.notes)
            if prices is not None:
                uncertainties.extend(detect_stale_mix(prices, facts, as_of))
            summary = freshness_summary(facts, prices, as_of)
            uncertainties.extend(summary.get("warnings", []))
            actions = self._corporate_actions()
            comparable, comparability_notes = comparable_periods(facts, actions)
            uncertainties.extend(comparability_notes)
            evidence = NormalizedEvidence(
                symbol=self.identity.symbol,
                as_of=as_of,
                facts=facts,
                prices=prices,
                benchmarks=benchmarks,
                sources=list(records),
                conflicts=fact_build.conflicts,
                uncertainties=_dedupe(uncertainties),
                segments=self._segments(comparable, prices),
                corporate_actions=actions,
                text_evidence=self._text_evidence(records),
                freshness_summary=summary,
                normalization_version=NORMALIZATION_VERSION,
            )
        return evidence

    def _corporate_actions(self) -> list[CorporateAction]:
        """Splits, renames, mergers and spin-offs need a corporate-actions source; web pages
        are not parsed into actions, so none are known and ``comparable_periods`` sees none."""
        return []

    @property
    def _as_of(self) -> datetime:
        return self.request_as_of

    def _text_evidence(self, records: list[SourceRecord]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for source in records:
            if not source.excerpt:
                continue
            if source.extraction_method in {"json", "csv"} or source.source_type == "market_data":
                continue  # structured payloads and quote pages are not prose to be judged
            items.append(
                {
                    "source_id": source.source_id,
                    "title": source.title,
                    "publisher": source.publisher,
                    "source_type": source.source_type,
                    "published_at": source.published_at.isoformat()
                    if source.published_at
                    else None,
                    "fact": source.excerpt[:600],
                    "rank": source_rank(source.source_type),
                    "is_primary": source.is_primary,
                    "freshness": source.freshness,
                }
            )
        # Most authoritative rank first; within a rank the newest coverage first (section 6),
        # undated items last.
        items.sort(key=lambda i: (i["rank"], i["published_at"] is None, i["published_at"] or ""))
        by_rank: dict[int, list[dict[str, Any]]] = {}
        for item in items:
            by_rank.setdefault(item["rank"], []).append(item)
        ordered: list[dict[str, Any]] = []
        for rank in sorted(by_rank):
            dated = [i for i in by_rank[rank] if i["published_at"]]
            undated = [i for i in by_rank[rank] if not i["published_at"]]
            ordered.extend(sorted(dated, key=lambda i: i["published_at"], reverse=True))
            ordered.extend(undated)
        return ordered

    def _segments(
        self, facts: list[NormalizedFact], prices: PriceSeries | None = None
    ) -> list[EventSegment]:
        by_period: dict[str, dict[str, NormalizedFact]] = {}
        periods: dict[str, Period] = {}
        for fact in facts:
            if fact.period.kind != "fiscal_quarter":
                continue
            key = fact.period.key()
            by_period.setdefault(key, {})[fact.metric] = fact
            periods[key] = fact.period
        ordered = sorted(periods.items(), key=lambda kv: kv[1].end or datetime.min.date())
        segments: list[EventSegment] = []
        prev_year: dict[tuple[int | None, str | None], dict[str, NormalizedFact]] = {}
        by_end: list[tuple[Period, dict[str, NormalizedFact]]] = []
        for key, period in ordered:
            metrics = by_period[key]
            summary: dict[str, Any] = {"period": period_label(period)}
            for metric in (
                "revenue",
                "net_income",
                "eps_diluted",
                "operating_income",
                "free_cash_flow",
            ):
                fact = metrics.get(metric)
                if fact is not None:
                    summary[metric] = fact.value
            if period.fiscal_year is not None and period.fiscal_period:
                prior = prev_year.get((period.fiscal_year - 1, period.fiscal_period))
            else:
                # Unlabelled quarters (read from web pages) pair by end date: the quarter
                # that ended 350-380 days earlier.
                prior = next(
                    (
                        m
                        for p, m in by_end
                        if period.end and p.end and 350 <= (period.end - p.end).days <= 380
                    ),
                    None,
                )
            if prior and "revenue" in metrics and "revenue" in prior:
                growth = growth_rate(metrics["revenue"].value, prior["revenue"].value)
                if growth is not None:
                    summary["revenue_growth_yoy"] = round(growth * 100, 2)
            if "operating_income" in metrics and "revenue" in metrics:
                m = margin(metrics["operating_income"].value, metrics["revenue"].value)
                if m is not None:
                    summary["operating_margin_pct"] = round(m * 100, 2)
            vol = _segment_volatility(prices, period)
            if vol is not None:
                summary["volatility_annualized_pct"] = round(vol * 100, 2)
            prev_year[(period.fiscal_year, period.fiscal_period)] = metrics
            by_end.append((period, metrics))
            segments.append(
                EventSegment(
                    segment_id=stable_id("seg", self.identity.symbol, period_label(period)),
                    period=period,
                    summary=summary,
                    source_ids=sorted({f.source_id for f in metrics.values()}),
                )
            )
        return segments[-_MAX_SEGMENTS_FOR_LAYA * 2 :]

    # ------------------------------------------------------------------ laya questions
    def build_laya_questions(
        self, evidence: NormalizedEvidence, request: AnalysisRequest
    ) -> list[LayaQuestionSet]:
        sets: list[LayaQuestionSet] = []
        latest = self._latest_metrics(evidence)
        scan_state = {
            "instrument": evidence.symbol,
            "horizon": request.resolved_horizon,
            "as_of": evidence.as_of.date().isoformat(),
            "latest_metrics": latest,
            "sources": len(evidence.sources),
            "primary_sources": sum(1 for s in evidence.sources if s.is_primary),
            "conflicts": [c.metric for c in evidence.conflicts][:5],
            "freshness_warnings": evidence.freshness_summary.get("warnings", [])[:3],
            "recent_headlines": [t["title"][:90] for t in evidence.text_evidence[:5]],
        }
        sets.append(
            LayaQuestionSet(
                stage="evidence_scan", state=scan_state, questions=overall_scan_questions()
            )
        )
        for segment in evidence.segments[-_MAX_SEGMENTS_FOR_LAYA:]:
            sets.append(
                LayaQuestionSet(
                    stage="history_scan",
                    state={"instrument": evidence.symbol, **segment.summary},
                    questions=history_segment_questions(
                        include_volatility="volatility_annualized_pct" in segment.summary
                    ),
                    segment_id=segment.segment_id,
                )
            )
        for item in evidence.text_evidence[:_MAX_TEXT_EVIDENCE_FOR_LAYA]:
            sets.append(
                LayaQuestionSet(
                    stage="text_evidence",
                    state={
                        "instrument": evidence.symbol,
                        "title": item["title"],
                        "publisher": item["publisher"],
                        "source_type": item["source_type"],
                        "published_at": item["published_at"],
                        "excerpt": item["fact"][:400],
                    },
                    questions=text_evidence_questions(),
                    segment_id=item["source_id"],
                )
            )
        return sets

    def build_horizon_questions(
        self,
        evidence: NormalizedEvidence,
        decisions: LayaDecisions,
        calculations: CalculatedMetrics,
        request: AnalysisRequest,
        horizons: list[str],
    ) -> list[LayaQuestionSet]:
        state = {
            "instrument": evidence.symbol,
            "horizon": request.resolved_horizon,
            "calculations": {
                c.name: c.display for c in calculations.calculations if c.status == "computed"
            },
            "scan": {
                d.decision_type: d.decision
                for d in decisions.decisions
                if d.stage == "evidence_scan"
            },
            "material_periods": [
                d.segment_id
                for d in decisions.decisions
                if d.decision_type == "material_change"
                and d.stage == "history_scan"
                and isinstance(d.answer, NoulAnswer)
                and d.answer.noul >= 0.6
            ][:4],
            "conflicts": len(evidence.conflicts),
            "freshness_warnings": evidence.freshness_summary.get("warnings", [])[:2],
        }
        return [
            LayaQuestionSet(
                stage="horizon", state=state, questions=horizon_context_questions(horizons)
            )
        ]

    # ------------------------------------------------------------------ calculate
    async def calculate(
        self, evidence: NormalizedEvidence, decisions: LayaDecisions, ctx: AnalysisContext
    ) -> CalculatedMetrics:
        pack = "all_standard"
        chosen = decisions.latest("calculation_pack")
        if chosen is not None and isinstance(chosen.answer, ChoiceAnswer):
            if chosen.answer.choice in CALCULATION_PACKS:
                pack = chosen.answer.choice
        packs = [pack] if pack == "all_standard" else [pack, "growth_and_margins"]
        results = []
        seen: set[str] = set()
        with ctx.timers.span("math"):
            for name in packs:
                for calc in run_pack(name, evidence, evidence.as_of):
                    if calc.name in seen:
                        continue
                    seen.add(calc.name)
                    results.append(calc)
            # The question's required calculations (and the ones it reports when available)
            # always run, whichever pack Laya chose; they come from their own packs so the
            # records stay identical to a pack run.
            wanted_by_question = (
                [*self.requirements.required_calculations, *self.requirements.also_calculated]
                if self.requirements
                else []
            )
            required = [name for name in wanted_by_question if name not in seen]
            added: list[str] = []
            for pack_name, names in CALCULATION_PACKS.items():
                wanted = [n for n in required if n in names and n not in seen]
                if pack_name == "all_standard" or not wanted:
                    continue
                for calc in run_pack(pack_name, evidence, evidence.as_of):
                    if calc.name in wanted and calc.name not in seen:
                        seen.add(calc.name)
                        results.append(calc)
                        added.append(calc.name)
        ctx.diagnostics["calculation_pack"] = pack
        if added:
            ctx.diagnostics["required_calculations_added"] = added
        return CalculatedMetrics(calculations=results)

    def validate_requirements(
        self,
        evidence: NormalizedEvidence | None,
        calculations: CalculatedMetrics,
        *,
        prior_available: bool | None = None,
    ) -> RequirementsReport | None:
        """Which of the question's requirements the analysis met (None before the question
        was interpreted). ``prior_available`` is whether a prior completed assessment exists
        (``None``: the lookup was not reached).

        Unmet requirements are uncertainties for the result and the Spark bundle; they never
        fail the analysis and leave ``INSUFFICIENT_EVIDENCE`` to the evidence gate.
        """
        if self.requirements is None:
            return None
        self.requirements_report = check_requirements(
            self.requirements,
            evidence,
            calculations,
            self.state.executed,
            prior_available=prior_available,
            symbol=self.identity.symbol if self.identity else None,
        )
        return self.requirements_report

    # ------------------------------------------------------------------ spark bundle
    def build_spark_bundle(
        self,
        evidence: NormalizedEvidence,
        decisions: LayaDecisions,
        calculations: CalculatedMetrics,
        request: AnalysisRequest,
    ) -> SparkEvidenceBundle:
        assert self.identity is not None
        horizons = [request.resolved_horizon]
        if request.resolved_horizon == "multi_horizon":
            horizons = ["near_term", "next_cycle", "medium_term", "long_term"]
        computed = {
            c.name: {
                "value": c.value,
                "unit": c.unit,
                "display": c.display,
                "period": c.period_label,
                "calc_id": c.calc_id,
            }
            for c in calculations.calculations
            if c.status == "computed"
        }
        unavailable = [c.name for c in calculations.calculations if c.status != "computed"]
        laya_assessments: dict[str, Any] = {}
        scan_view: dict[str, Any] = {}
        for decision in decisions.decisions:
            if decision.stage not in {"evidence_scan", "horizon"}:
                continue
            if decision.decision_type.startswith("horizon_stance_"):
                laya_assessments[decision.decision_type.removeprefix("horizon_stance_")] = {
                    "stance": decision.decision,
                    "confidence": round(decision.confidence, 3),
                }
            else:
                scan_view[decision.decision_type] = {
                    "decision": decision.decision,
                    "confidence": round(decision.confidence, 3),
                }
        laya_assessments["scan"] = scan_view
        important_events = []
        for segment in evidence.segments:
            seg_decisions = [d for d in decisions.decisions if d.segment_id == segment.segment_id]
            material = any(
                d.decision_type == "material_change"
                and isinstance(d.answer, NoulAnswer)
                and d.answer.noul >= 0.6
                for d in seg_decisions
            )
            entry = {
                "period": segment.summary.get("period"),
                "material": material,
                "summary": {k: v for k, v in segment.summary.items() if k != "period"},
                "laya": {d.decision_type: d.decision for d in seg_decisions},
                "source_ids": segment.source_ids,
            }
            important_events.append(entry)
        excerpts = []
        for item in evidence.text_evidence:
            stance = next(
                (
                    d.decision
                    for d in decisions.decisions
                    if d.segment_id == item["source_id"] and d.decision_type == "evidence_stance"
                ),
                None,
            )
            excerpts.append(
                {
                    "source_id": item["source_id"],
                    "title": item["title"],
                    "publisher": item["publisher"],
                    "published_at": item["published_at"],
                    "source_type": item["source_type"],
                    "is_primary": item["is_primary"],
                    "rank": item["rank"],
                    "text": item["fact"],
                    "laya_stance": stance,
                }
            )
        sources = [
            {**s.public_view(), "rank": s.rank} for s in evidence.sources if not s.rejected_reason
        ]
        requirements = request.requirements or self.requirements
        report = self.requirements_report
        question_focus: dict[str, Any] = {}
        question_notes: list[str] = []
        if requirements is not None and not requirements.broad:
            question_focus = {
                "intent": requirements.intent_label,
                "requirements": requirements.requirement_labels,
                "focus": requirements.focus,
                "horizons_emphasis": list(requirements.horizons_emphasis),
                "recent_period": requirements.recent_period,
                "unmet_requirements": list(report.uncertainties) if report else [],
            }
        elif requirements is not None:
            # A general assessment runs exactly as before; only an interpretation note (the
            # fallback, a rejected interpretation) reaches Spark, as an uncertainty.
            question_notes = list(report.uncertainties if report else requirements.notes)
        return SparkEvidenceBundle(
            instrument={
                "symbol": self.identity.symbol,
                "name": self.identity.name,
                "exchange": self.identity.exchange,
                "sector": self.identity.sector,
            },
            request={
                "query": request.query,
                "profile": request.profile,
                "horizon": request.resolved_horizon,
                "as_of": request.as_of.isoformat(),
            },
            current_metrics=self._latest_metrics(evidence),
            historical_metrics={
                "segments": [e["summary"] | {"period": e["period"]} for e in important_events[-8:]]
            },
            laya_assessments=laya_assessments,
            important_events=[e for e in important_events if e["material"]]
            or important_events[-3:],
            historical_analogues=_historical_analogues(evidence, decisions),
            calculated_metrics={"computed": computed, "unavailable": unavailable},
            benchmark_context={},
            sources=sources,
            excerpts=excerpts,
            conflicts=[c.model_dump(mode="json") for c in evidence.conflicts],
            uncertainties=_dedupe([*evidence.uncertainties, *question_notes]),
            freshness=evidence.freshness_summary,
            horizons=horizons,
            question_focus=question_focus,
        )

    # ------------------------------------------------------------------ helpers
    def _latest_metrics(self, evidence: NormalizedEvidence) -> dict[str, Any]:
        latest: dict[str, Any] = {}
        for metric in (
            "revenue",
            "net_income",
            "eps_diluted",
            "operating_income",
            "operating_cash_flow",
            "free_cash_flow",
            "shares_outstanding",
        ):
            facts = [f for f in evidence.facts_for(metric) if f.period.end is not None]
            if not facts:
                continue
            quarterly = [f for f in facts if f.period.kind == "fiscal_quarter"]
            pool = quarterly or facts
            fact = max(pool, key=lambda f: f.period.end)  # type: ignore[arg-type,return-value]
            latest[metric] = {
                "value": fact.value,
                "unit": fact.unit,
                "period": period_label(fact.period),
                "source_id": fact.source_id,
            }
        if evidence.prices and evidence.prices.latest:
            latest["price"] = {
                "value": evidence.prices.latest.close,
                "price_type": evidence.prices.price_type,
                "session_date": evidence.prices.session_date.isoformat()
                if evidence.prices.session_date
                else None,
                "source_id": evidence.prices.source_id,
            }
        return latest


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _interpretation_view(requirements: AnalyticalRequirements | None) -> dict[str, Any]:
    """What ``research.started`` carries about the question: product labels only (never the
    raw interpretation); null / empty before the question was interpreted."""
    if requirements is None:
        return {"question_intent": None, "requirements": [], "interpretation_source": None}
    return {
        "question_intent": requirements.intent_label,
        "requirements": requirements.requirement_labels,
        "interpretation_source": requirements.source,
    }


def _segment_volatility(prices: PriceSeries | None, period: Period) -> float | None:
    """Annualized volatility of daily closes inside the period (None below 20 closes)."""
    if prices is None or period.start is None or period.end is None:
        return None
    closes = [p.close for p in prices.points if period.start <= p.date <= period.end]
    if len(closes) < 20:
        return None
    return annualized_volatility(closes)


def _historical_analogues(
    evidence: NormalizedEvidence, decisions: LayaDecisions, limit: int = 3
) -> list[dict[str, Any]]:
    """Past periods most similar to the latest one on the deterministic segment summary
    (revenue growth, operating margin, volatility), preferring periods Laya scored as
    material or unusual. A heuristic over retrieved history, not a model judgement."""
    segments = [s for s in evidence.segments if s.summary.get("revenue_growth_yoy") is not None]
    if len(segments) < 2:
        return []
    latest = segments[-1]
    flags: dict[str, float] = {}
    for decision in decisions.decisions:
        if decision.stage != "history_scan" or decision.segment_id is None:
            continue
        if decision.decision_type in {"historically_unusual", "material_change"} and isinstance(
            decision.answer, NoulAnswer
        ):
            flags[decision.segment_id] = max(
                flags.get(decision.segment_id, 0.0), decision.answer.noul
            )

    def distance(seg: EventSegment) -> float:
        d = abs(
            float(seg.summary.get("revenue_growth_yoy", 0.0))
            - float(latest.summary.get("revenue_growth_yoy", 0.0))
        )
        lm, sm = latest.summary.get("operating_margin_pct"), seg.summary.get("operating_margin_pct")
        if lm is not None and sm is not None:
            d += abs(float(sm) - float(lm))
        lv, sv = (
            latest.summary.get("volatility_annualized_pct"),
            seg.summary.get("volatility_annualized_pct"),
        )
        if lv is not None and sv is not None:
            d += abs(float(sv) - float(lv)) / 2
        return d - 5.0 * flags.get(seg.segment_id, 0.0)

    ranked = sorted(segments[:-1], key=distance)[:limit]
    return [
        {
            "period": seg.summary.get("period"),
            "label": seg.summary.get("period"),
            "summary": {k: v for k, v in seg.summary.items() if k != "period"},
            "similarity_distance": round(distance(seg), 3),
            "laya_material_or_unusual": round(flags.get(seg.segment_id, 0.0), 3),
            "source_ids": seg.source_ids,
            "method": "deterministic nearest-neighbour on growth/margin/volatility",
        }
        for seg in ranked
    ]
