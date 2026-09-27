"""EquityAnalyzer: the public-equity adapter (AGENT.md sections 21, 23, 27, 28).

Owns the research strategy for one analysis: a deterministic seed plan by horizon, then a
Laya-directed loop over bounded research intents until evidence is sufficient, retrieval stops
changing the picture, the budget is spent, or the user cancels. Retrieval itself is done by the
research runner; normalization, Laya question construction, calculation selection and the Spark
bundle are built here from the accumulated evidence.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from bayanalytics.calculations.primitives import growth_rate, margin
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
from bayanalytics.instruments.identity import InstrumentResolver
from bayanalytics.laya.schemas import (
    history_segment_questions,
    horizon_context_questions,
    overall_scan_questions,
    research_plan_questions,
    text_evidence_questions,
)
from bayanalytics.laya.wrapper import LayaFinanceWrapper
from bayanalytics.normalization import NORMALIZATION_VERSION
from bayanalytics.normalization.facts import (
    build_facts,
    derive_fourth_quarter_rows,
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
from bayanalytics.research.provider import EvidenceRecord, ResearchProviderError
from bayanalytics.research.runner import ResearchRunner, RoundResult
from bayanalytics.schemas.common import ErrorCode, new_id, source_rank
from bayanalytics.schemas.decisions import ChoiceAnswer, LayaDecision, LayaQuestionSet, NoulAnswer
from bayanalytics.schemas.evidence import (
    BenchmarkRef,
    EventSegment,
    NormalizedEvidence,
    NormalizedFact,
    Period,
    PriceSeries,
    SourceRecord,
)
from bayanalytics.schemas.requests import InstrumentRef
from bayanalytics.schemas.results import ResearchStats

log = logging.getLogger(__name__)

ResolverFactory = Callable[[], Awaitable[InstrumentResolver]]

_MAX_TEXT_EVIDENCE_FOR_LAYA = 8
_MAX_SEGMENTS_FOR_LAYA = 8
_SUFFICIENT_THRESHOLD = 0.7


@dataclass
class RetrievalState:
    sources: list[SourceRecord] = field(default_factory=list)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    price_series: PriceSeries | None = None
    benchmark_series: dict[str, PriceSeries] = field(default_factory=dict)
    benchmark_refs: list[BenchmarkRef] = field(default_factory=list)
    rejected: list[tuple[SourceRecord, str]] = field(default_factory=list)
    executed: list[str] = field(default_factory=list)
    rounds: int = 0
    termination_reason: str | None = None
    source_by_id: dict[str, SourceRecord] = field(default_factory=dict)
    evidence_by_source: dict[str, EvidenceRecord] = field(default_factory=dict)

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
        self.rows.extend(result.facts_rows)
        if result.price_series is not None:
            self.price_series = result.price_series
        self.benchmark_series.update(result.benchmark_series)
        for ref in result.benchmark_refs:
            if all(r.symbol != ref.symbol for r in self.benchmark_refs):
                self.benchmark_refs.append(ref)
        self.rejected.extend(result.rejected)


class EquityAnalyzer:
    """One instance per analysis (it accumulates retrieval state)."""

    def __init__(
        self,
        settings: Settings,
        research_stack: Any,
        laya: LayaFinanceWrapper,
        resolver_factory: ResolverFactory,
    ) -> None:
        self.settings = settings
        self.provider, self.edgar, self.prices = research_stack
        self.laya = laya
        self._resolver_factory = resolver_factory
        self.state = RetrievalState()
        self.stats = ResearchStats()
        self.identity: InstrumentIdentity | None = None
        self.instrument_ref: InstrumentRef | None = None
        self.research_decisions: list[LayaDecision] = []
        self.request_as_of: datetime = datetime.now(tz=UTC)
        self._submissions: Any = None

    # ------------------------------------------------------------------ identify
    async def identify(self, user_input: str, ctx: AnalysisContext) -> InstrumentIdentity:
        resolver = await self._resolver_factory()
        identity = resolver.resolve(user_input, self.instrument_ref)
        self.identity = identity
        return identity

    async def enrich_identity(self, identity: InstrumentIdentity, ctx: AnalysisContext) -> None:
        """Fill exchange / SIC / fiscal-year-end from EDGAR submissions (first structured call)."""
        if not identity.cik:
            return
        try:
            submissions = await self.edgar.submissions(identity.cik)
        except ResearchProviderError as exc:
            raise AnalysisError(
                ErrorCode.RESEARCH_UNAVAILABLE, details={"stage": "edgar_submissions"}
            ) from exc
        identity.sic = getattr(submissions, "sic", None) or identity.sic
        identity.sector = getattr(submissions, "sic_description", None) or identity.sector
        identity.fiscal_year_end = getattr(submissions, "fiscal_year_end", None)
        exchanges = getattr(submissions, "exchanges", None) or []
        if not identity.exchange and exchanges:
            identity.exchange = str(exchanges[0]).upper()
        former = getattr(submissions, "former_names", None) or []
        identity.ticker_history = [str(n) for n in former][:6]
        if getattr(submissions, "name", None):
            identity.name = submissions.name
        self._submissions = submissions

    # ------------------------------------------------------------------ retrieve
    async def retrieve(
        self, identity: InstrumentIdentity, request: AnalysisRequest, ctx: AnalysisContext
    ) -> list[SourceRecord]:
        budget = request.budget
        self.request_as_of = request.as_of
        runner = ResearchRunner(self.provider, self.edgar, self.prices, self.settings, budget)
        plan: list[ResearchIntent] = list(seed_plan(request.resolved_horizon))
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
        if True:
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
                )
                for intent in plan:
                    if intent == ResearchIntent.stop_research:
                        continue
                    for planned in build_queries(
                        intent, identity, request.resolved_horizon, request.as_of, gaps
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
        return merged

    def compute_gaps(self, horizon: str, as_of: datetime) -> list[str]:
        gaps: list[str] = []
        revenue_quarters = {
            (r.get("fy"), r.get("fp"))
            for r in self.state.rows
            if r.get("metric") == "revenue" and r.get("fp") in {"Q1", "Q2", "Q3", "Q4"}
        }
        if len(revenue_quarters) < 4:
            gaps.append("earnings_history")
        if self.state.price_series is None or not self.state.price_series.points:
            gaps.append("price_history")
        if not self.state.benchmark_series:
            gaps.append("sector_benchmark")
        if not any(s.source_type == "regulatory_filing" for s in self.state.sources):
            gaps.append("latest_filing")
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
        return gaps

    async def _plan_next(
        self,
        identity: InstrumentIdentity,
        request: AnalysisRequest,
        gaps: list[str],
        ctx: AnalysisContext,
    ) -> tuple[ResearchIntent, float]:
        primary = [s for s in self.state.sources if s.is_primary]
        freshness = freshness_summary(
            self._quick_facts(request.as_of), self.state.price_series, request.as_of
        )
        state = {
            "instrument": identity.symbol,
            "horizon": request.resolved_horizon,
            "round": self.state.rounds,
            "sources_count": len(self.state.sources),
            "primary_sources": len(primary),
            "facts_count": len(self.state.rows),
            "has_price_history": self.state.price_series is not None,
            "has_benchmark": bool(self.state.benchmark_series),
            "evidence_gaps": gaps,
            "executed_intents": self.state.executed[-6:],
            "freshness": freshness.get("warnings", [])[:3],
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
        decisions = await self.laya.ask(question_set, ctx)
        await ctx.event(
            "laya.started", stage="research_plan", questions=len(question_set.questions)
        )
        for decision in decisions:
            await ctx.event("laya.decision", **decision.event_view())
        await ctx.event("laya.completed", stage="research_plan", decisions=len(decisions))
        self.research_decisions.extend(decisions)
        intent = ResearchIntent.stop_research
        sufficient = 0.0
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
        # Laya's stop is honoured unless a gap remains whose intent has not been tried yet.
        if intent != ResearchIntent.stop_research or not gaps:
            return intent, sufficient
        for gap in gaps:
            candidate = gap_to_intent(gap)
            if str(candidate) not in self.state.executed:
                return candidate, sufficient
        return ResearchIntent.stop_research, sufficient

    # ------------------------------------------------------------------ normalize
    def _quick_facts(self, as_of: datetime) -> list[NormalizedFact]:
        try:
            return build_facts(derive_fourth_quarter_rows(self.state.rows), as_of).facts
        except Exception:  # pragma: no cover - defensive; normalize() surfaces real errors
            return []

    async def normalize(
        self, records: list[SourceRecord], ctx: AnalysisContext
    ) -> NormalizedEvidence:
        assert self.identity is not None
        as_of = self._as_of
        with ctx.timers.span("normalization"):
            fact_build = build_facts(derive_fourth_quarter_rows(self.state.rows), as_of)
            prices = (
                label_series(self.state.price_series, as_of) if self.state.price_series else None
            )
            benchmarks = {
                key: label_series(series, as_of)
                for key, series in self.state.benchmark_series.items()
            }
            uncertainties: list[str] = list(fact_build.notes)
            if prices is not None:
                uncertainties.extend(detect_stale_mix(prices, fact_build.facts, as_of))
            summary = freshness_summary(fact_build.facts, prices, as_of)
            uncertainties.extend(summary.get("warnings", []))
            if fact_build.dropped:
                uncertainties.append(
                    f"{len(fact_build.dropped)} facts published after the as-of date were excluded"
                )
            text_evidence = self._text_evidence(records)
            segments = self._segments(fact_build.facts)
            evidence = NormalizedEvidence(
                symbol=self.identity.symbol,
                as_of=as_of,
                facts=fact_build.facts,
                prices=prices,
                benchmarks=benchmarks,
                benchmark_refs=list(self.state.benchmark_refs),
                sources=list(records),
                conflicts=fact_build.conflicts,
                uncertainties=_dedupe(uncertainties),
                segments=segments,
                text_evidence=text_evidence,
                freshness_summary=summary,
                normalization_version=NORMALIZATION_VERSION,
            )
        return evidence

    @property
    def _as_of(self) -> datetime:
        return self.request_as_of

    def _text_evidence(self, records: list[SourceRecord]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for source in records:
            if not source.excerpt:
                continue
            if source.extraction_method in {"json", "csv"} or source.source_type == "market_data":
                continue  # structured endpoints are facts/prices, not prose to be judged
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
        items.sort(key=lambda i: (i["rank"], i["published_at"] or ""))
        return items

    def _segments(self, facts: list[NormalizedFact]) -> list[EventSegment]:
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
            prior = prev_year.get(
                (period.fiscal_year - 1 if period.fiscal_year else None, period.fiscal_period)
            )
            if prior and "revenue" in metrics and "revenue" in prior:
                growth = growth_rate(metrics["revenue"].value, prior["revenue"].value)
                if growth is not None:
                    summary["revenue_growth_yoy"] = round(growth * 100, 2)
            if "operating_income" in metrics and "revenue" in metrics:
                m = margin(metrics["operating_income"].value, metrics["revenue"].value)
                if m is not None:
                    summary["operating_margin_pct"] = round(m * 100, 2)
            prev_year[(period.fiscal_year, period.fiscal_period)] = metrics
            segments.append(
                EventSegment(
                    segment_id=new_id("seg"),
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
                    questions=history_segment_questions(),
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
        ctx.diagnostics["calculation_pack"] = pack
        return CalculatedMetrics(calculations=results)

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
        return SparkEvidenceBundle(
            instrument={
                "symbol": self.identity.symbol,
                "name": self.identity.name,
                "exchange": self.identity.exchange,
                "sector": self.identity.sector,
                "fiscal_year_end": self.identity.fiscal_year_end,
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
            historical_analogues=[],
            calculated_metrics={"computed": computed, "unavailable": unavailable},
            benchmark_context={
                "benchmarks": [r.model_dump() for r in evidence.benchmark_refs],
            },
            sources=sources,
            excerpts=excerpts,
            conflicts=[c.model_dump(mode="json") for c in evidence.conflicts],
            uncertainties=list(evidence.uncertainties),
            freshness=evidence.freshness_summary,
            horizons=horizons,
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
