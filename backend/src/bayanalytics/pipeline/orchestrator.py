"""The vertical slice (AGENT.md sections 19, 27, 40).

resolve instrument -> Spark pass 1 (question understanding) -> Laya validation of the proposed
requirements -> deterministic requirements -> Laya-directed research -> normalize + provenance
-> Laya scoring -> deterministic calculations -> Laya horizon stances -> Spark pass 2 synthesis
(streamed) -> structured, sourced result. Every stage emits recorded system state; nothing
hidden is streamed. The function never raises: failures become a result with ``status``
failed/cancelled and a structured error, preserving whatever content was already produced as
``partial``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from bayanalytics.calculations.reconciliation import bundle_view as reconciliation_bundle_view
from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.base import (
    AnalysisRequest,
    CalculatedMetrics,
    InstrumentIdentity,
    LayaDecisions,
)
from bayanalytics.instruments.equity import EquityAnalyzer
from bayanalytics.jobs.models import AnalysisJob
from bayanalytics.pipeline.assemble import finalize_assessment, merge_horizons
from bayanalytics.pipeline.horizon import horizons_for
from bayanalytics.pipeline.questions import resolve_requirements
from bayanalytics.pipeline.thesis import (
    diff_assessments,
    find_prior_assessment,
    no_prior_assessment_note,
    prior_assessment_block,
    snapshot_of_assembled,
    snapshot_of_draft,
)
from bayanalytics.pipeline.understanding import (
    QUERY_UNDERSTANDING_STAGE,
    SYNTHESIS_STAGE,
    Understanding,
    understand_question,
)
from bayanalytics.pipeline.understanding import TIMER_NAME as UNDERSTANDING_TIMER
from bayanalytics.research.market import fundamentals_view, market_series
from bayanalytics.research.sources import domain_of, web_pages
from bayanalytics.runtime import Runtime
from bayanalytics.schemas.common import ErrorCode, utcnow
from bayanalytics.schemas.decisions import LayaDecision
from bayanalytics.schemas.evidence import NormalizedEvidence, SourceRecord
from bayanalytics.schemas.questions import RequirementsReport
from bayanalytics.schemas.results import (
    AnalysisResult,
    Assessment,
    InstrumentView,
    MarketView,
    Telemetry,
    ThesisDiff,
    VersionInfo,
)
from bayanalytics.spark.base import SparkMessage, SparkRunOptions
from bayanalytics.spark.bundle import OVERFLOW_TRIM, fit_bundle
from bayanalytics.spark.parse import parse_sections, to_assessment
from bayanalytics.spark.prompt import build_messages
from bayanalytics.telemetry.tracker import PeakTracker

log = logging.getLogger(__name__)

AnalyzerFactory = Callable[[], EquityAnalyzer]


class _Draft:
    """Everything produced so far, so a failure or cancellation can still return it."""

    def __init__(self, job: AnalysisJob) -> None:
        self.job = job
        self.identity: InstrumentIdentity | None = None
        self.sources: list[SourceRecord] = []
        self.evidence: NormalizedEvidence | None = None
        self.decisions = LayaDecisions()
        self.calculations = CalculatedMetrics()
        self.streamed_text = ""
        self.assessment = Assessment()
        self.horizon_assessments: dict[str, Any] = {}
        self.telemetry = Telemetry(profile=job.profile)
        self.extra_uncertainties: list[str] = []
        self.freshness_summary: dict[str, Any] = {}
        self.partial_synthesis = False
        self.prior: AnalysisResult | None = None  # the last completed assessment, if any
        self.thesis_diff: ThesisDiff | None = None
        self.understanding: Understanding | None = None  # Spark pass 1, once it ran
        self.requirements: RequirementsReport | None = None
        self.market = MarketView()

    def instrument_view(self) -> InstrumentView | None:
        if self.identity is None:
            return None
        return InstrumentView(
            symbol=self.identity.symbol,
            exchange=self.identity.exchange,
            name=self.identity.name,
            cik=self.identity.cik,
            sector=self.identity.sector,
        )

    def result(self, status: str, error: AnalysisError | None = None) -> AnalysisResult:
        return AnalysisResult(
            analysis_id=self.job.analysis_id,
            status=status,  # type: ignore[arg-type]
            query=self.job.query,
            instrument=self.instrument_view(),
            profile=self.job.profile,
            horizon=self.job.resolved_horizon,
            as_of=self.job.as_of,
            created_at=self.job.created_at,
            completed_at=utcnow(),
            assessment=self.assessment,
            horizon_assessments=self.horizon_assessments,
            sources=self.sources,
            calculations=self.calculations.calculations,
            laya_decisions=self.decisions.decisions,
            freshness_summary=self.freshness_summary,
            thesis_diff=self.thesis_diff,
            requirements=self.requirements,
            streamed_text=self.streamed_text,
            market=self.market,
            telemetry=self.telemetry,
            error=error.payload() if error else None,
            partial=status != "completed" or self.partial_synthesis,
        )


async def run_analysis(job: AnalysisJob, ctx: AnalysisContext, rt: Runtime) -> AnalysisResult:
    draft = _Draft(job)
    draft.market.price_display = rt.settings.price_display
    analyzer: EquityAnalyzer = rt.extras["analyzer_factory"]()
    analyzer.instrument_ref = job.instrument_ref
    request = AnalysisRequest(
        analysis_id=job.analysis_id,
        query=job.query,
        profile=job.profile,
        requested_horizon=job.requested_horizon,
        resolved_horizon=job.resolved_horizon,
        as_of=job.as_of,
        budget=job.research_budget,
    )
    horizons = horizons_for(job.resolved_horizon)
    tracker = PeakTracker(child_pids=lambda: _child_pids(rt))
    ctx.timers.start("total")
    try:
        async with tracker:
            execution = rt.extras.get("execution")
            await ctx.event(
                "analysis.started",
                query=job.query,
                profile=job.profile,
                resolved_horizon=job.resolved_horizon,
                as_of=job.as_of.isoformat(),
                execution=execution.model_dump() if execution is not None else None,
            )
            # 1. instrument -------------------------------------------------------------
            await _set_status(rt, job, "resolving_instrument")
            identity = await analyzer.identify(job.query, ctx)
            await analyzer.enrich_identity(identity, ctx)
            draft.identity = identity
            job.instrument = identity
            await rt.store.update_job(job)
            await ctx.event(
                "instrument.resolved",
                symbol=identity.symbol,
                exchange=identity.exchange,
                name=identity.name,
                cik=identity.cik,
                sector=identity.sector,
                resolution_method=identity.resolution_method,
                confidence=round(identity.confidence, 3),
            )
            # 2. research ---------------------------------------------------------------
            # Evidence comes only from web search: without a search backend there is nothing
            # to research, so fail before any model time is spent.
            if not rt.search_configured:
                raise _search_not_configured()
            await _set_status(rt, job, "researching")
            # 2a. what the question requires: Spark pass 1 interprets it (a short structured
            # reading on its own Spark session, lock released before research), Laya confirms
            # or drops each proposed requirement (question_validation), and the requirements
            # builder turns what survives into intents, calculations, operands and checks.
            # Pass 1 takes its turn on the one Spark lane like any other request (first come,
            # first served); a client is told only when it actually has to wait.
            if _spark_busy(rt.spark):
                await ctx.event(
                    "spark.queued",
                    profile=job.profile,
                    stage=QUERY_UNDERSTANDING_STAGE,
                    active_analyses=rt.runner.active_count,
                )
            draft.understanding = await understand_question(
                job.query,
                identity,
                job.resolved_horizon,
                rt.spark,
                job.profile,
                ctx,
                rt.settings,
            )
            requirements, question_decisions = await resolve_requirements(
                draft.understanding,
                job.query,
                identity,
                job.resolved_horizon,
                analyzer.laya,
                ctx,
            )
            draft.decisions.decisions.extend(question_decisions)
            request = request.model_copy(update={"requirements": requirements})
            sources = await analyzer.retrieve(identity, request, ctx)
            draft.sources = sources
            draft.decisions.decisions.extend(analyzer.research_decisions)
            job.source_ids = [s.source_id for s in sources]
            await rt.store.save_sources(job.analysis_id, sources)
            # 3. normalize --------------------------------------------------------------
            await _set_status(rt, job, "normalizing")
            evidence = await analyzer.normalize(sources, ctx)
            draft.evidence = evidence
            await rt.store.save_facts(job.analysis_id, evidence.facts)
            await ctx.event(
                "normalization.completed",
                facts=len(evidence.facts),
                conflicts=len(evidence.conflicts),
                uncertainties=len(evidence.uncertainties),
                sources=len(evidence.sources),
                segments=len(evidence.segments),
                freshness=_freshness_view(evidence.freshness_summary),
            )
            fundamentals = fundamentals_view(evidence, job.as_of)
            draft.market.fundamentals = fundamentals
            if fundamentals is not None:
                await ctx.event("market.fundamentals", **fundamentals.model_dump(mode="json"))
            if rt.settings.price_display:
                draft.market.series = market_series(
                    evidence, identity.symbol, identity.name or identity.symbol
                )
            _evidence_gate(
                evidence,
                analyzer.compute_gaps(job.resolved_horizon, job.as_of),
                searches_issued=analyzer.state.searches_issued,
                searches_failed=analyzer.state.queries_failed,
            )
            # 4. Laya scoring -----------------------------------------------------------
            await _set_status(rt, job, "scoring")
            question_sets = analyzer.build_laya_questions(evidence, request)
            await ctx.event(
                "laya.started",
                stage="evidence_scan",
                question_sets=len(question_sets),
                questions=sum(len(s.questions) for s in question_sets),
            )
            decisions = await analyzer.laya.ask_many(question_sets, ctx)
            await _emit_decisions(ctx, decisions)
            draft.decisions.decisions.extend(decisions)
            await ctx.event("laya.completed", stage="evidence_scan", decisions=len(decisions))
            # 5. calculations -----------------------------------------------------------
            await _set_status(rt, job, "calculating")
            await ctx.event("calculation.started", pack=_pack_name(draft.decisions))
            calculations = await analyzer.calculate(evidence, draft.decisions, ctx)
            draft.calculations = calculations
            for calc in calculations.calculations:
                await ctx.event("calculation.completed", **calc.event_view())
            await rt.store.save_calculations(job.analysis_id, calculations.calculations)
            # 5b. horizon stances (Laya, with calculations in state) ----------------------
            horizon_sets = analyzer.build_horizon_questions(
                evidence, draft.decisions, calculations, request, horizons
            )
            await ctx.event(
                "laya.started",
                stage="horizon",
                question_sets=len(horizon_sets),
                questions=sum(len(s.questions) for s in horizon_sets),
            )
            horizon_decisions = await analyzer.laya.ask_many(horizon_sets, ctx)
            await _emit_decisions(ctx, horizon_decisions)
            draft.decisions.decisions.extend(horizon_decisions)
            await ctx.event("laya.completed", stage="horizon", decisions=len(horizon_decisions))
            await rt.store.save_decisions(job.analysis_id, draft.decisions.decisions)
            # 5c. prior assessment: the thesis diff against the last completed run ---------
            # Single-user seam: no owner is recorded on jobs, so the lookup is unscoped and
            # says so explicitly (the store refuses any owner filter it cannot honour).
            draft.prior = await find_prior_assessment(
                rt.store, identity.symbol, before=job.created_at, owner_id=None
            )
            if draft.prior is None:
                draft.extra_uncertainties.append(no_prior_assessment_note(identity.symbol))
            else:
                current = snapshot_of_draft(
                    evidence, draft.decisions, calculations, horizons, draft.extra_uncertainties
                )
                draft.thesis_diff = diff_assessments(draft.prior, current)
            # 5d. requirement acceptance checks (after the prior lookup, which the
            # prior_assessment requirement needs): unmet requirements are uncertainties for
            # the result and for Spark, never a failure.
            draft.requirements = analyzer.validate_requirements(
                evidence, calculations, prior_available=draft.prior is not None
            )
            if draft.requirements is not None:
                draft.extra_uncertainties.extend(draft.requirements.uncertainties)
            # 6. Spark synthesis --------------------------------------------------------
            await _set_status(rt, job, "synthesizing")
            options = SparkRunOptions(
                max_tokens=rt.settings.spark_max_output_tokens,
                temperature=rt.settings.spark_temperature,
            )
            bundle = analyzer.build_spark_bundle(evidence, draft.decisions, calculations, request)
            if draft.thesis_diff is not None:
                bundle.prior_assessment = prior_assessment_block(draft.thesis_diff)
            else:
                bundle.uncertainties.append(no_prior_assessment_note(identity.symbol))
            if reconciled := reconciliation_bundle_view(calculations.calculations):
                bundle.calculated_metrics["reconciliation"] = reconciled
            started = False
            prompt_tokens: int | None = None

            async def emit_started() -> None:
                nonlocal started
                if started:
                    return
                started = True
                await ctx.event(
                    "spark.started",
                    profile=job.profile,
                    context_ceiling=spec.context_ceiling,
                    prompt_tokens=prompt_tokens,
                    horizons=horizons,
                )

            async def on_token(text: str) -> None:
                # spark.loading (if a load happens) is emitted by the client before the first
                # token, so the answer state begins here, at the first real token.
                await emit_started()
                draft.streamed_text += text
                await ctx.event("spark.token", text=text)

            if _spark_busy(rt.spark):
                await ctx.event(
                    "spark.queued",
                    profile=job.profile,
                    stage=SYNTHESIS_STAGE,
                    active_analyses=rt.runner.active_count,
                )
            async with rt.spark.session(job.profile, ctx) as session:
                spec = session.spec
                # The prompt is sized by the server's own tokenizer and chat template, so the
                # overflow decision below is a measurement, not an estimate.
                fit = await fit_bundle(
                    bundle, spec.context_ceiling, options, session.count_prompt_tokens
                )
                prompt_tokens = fit.prompt_tokens
                ctx.diagnostics["spark_prompt_tokens"] = fit.prompt_tokens
                ctx.diagnostics["spark_fit_measurements"] = fit.measurements
                draft.extra_uncertainties.extend(t for t in fit.trims if t != OVERFLOW_TRIM)
                if fit.overflow:
                    raise AnalysisError(
                        ErrorCode.SPARK_INFERENCE_FAILED,
                        "The evidence bundle exceeds this profile's context ceiling even after "
                        "trimming. Try the Deep profile.",
                        retryable=True,
                        details={
                            "reason": "context_overflow",
                            "profile": job.profile,
                            "context_ceiling": spec.context_ceiling,
                            "prompt_tokens": fit.prompt_tokens,
                            "budget": fit.budget,
                        },
                    )
                messages: list[SparkMessage] = build_messages(fit.bundle, options)
                generation = await session.generate(messages, on_token, options)
            await emit_started()  # an empty generation still marks the answer state
            stats = generation.stats
            await ctx.event(
                "spark.completed",
                prompt_tokens=stats.prompt_tokens,
                output_tokens=stats.output_tokens,
                time_to_first_token_ms=stats.time_to_first_token_ms,
                total_ms=stats.total_ms,
                tokens_per_second=stats.tokens_per_second,
                truncated=generation.truncated,
            )
            if generation.truncated:
                draft.extra_uncertainties.append(
                    "the synthesis hit the output token limit and may be incomplete"
                )
            # 7. assemble ---------------------------------------------------------------
            sections = parse_sections(generation.text)
            known_ids = {s.source_id for s in sources}
            assessment, parsed_horizons, warnings = to_assessment(sections, known_ids)
            draft.extra_uncertainties.extend(warnings)
            draft.assessment = finalize_assessment(
                assessment, evidence, draft.decisions, calculations, draft.extra_uncertainties
            )
            draft.horizon_assessments, horizon_notes = merge_horizons(
                horizons,
                parsed_horizons,
                draft.decisions,
                draft.assessment.bull_evidence,
                draft.assessment.bear_evidence,
                known_ids,
            )
            for note in horizon_notes:
                if note not in draft.assessment.uncertainties:
                    draft.assessment.uncertainties.append(note)
            for entry in ctx.diagnostics.get("laya_truncated", []):
                note = (
                    f"Laya state was truncated in stage {entry.get('stage')}; "
                    "those decisions saw partial evidence"
                )
                if note not in draft.assessment.uncertainties:
                    draft.assessment.uncertainties.append(note)
            draft.freshness_summary = dict(evidence.freshness_summary)
            if draft.prior is not None:
                # Re-diff against the assembled result so the stored diff reflects the final
                # stances, conflicts and uncertainties (same rule set as the pre-Spark diff).
                draft.thesis_diff = diff_assessments(
                    draft.prior,
                    snapshot_of_assembled(
                        draft.horizon_assessments,
                        calculations,
                        draft.assessment.conflicts,
                        draft.assessment.uncertainties,
                        draft.freshness_summary,
                    ),
                )
            # A cut-off synthesis or a missing horizon section is reported, never passed off
            # as a complete assessment (section 24).
            draft.partial_synthesis = generation.truncated or not all(
                h.synthesized for h in draft.horizon_assessments.values()
            )
            ctx.timers.stop("total")
            draft.telemetry = _telemetry(rt, ctx, draft, analyzer, tracker, stats)
            return draft.result("completed")
    except AnalysisError as exc:
        ctx.timers.stop("total")
        status = "cancelled" if exc.code == ErrorCode.CANCELLED else "failed"
        log.info("analysis %s %s: %s", job.analysis_id, status, exc.code)
        await _salvage(rt, job, draft, analyzer)
        draft.telemetry = _telemetry(rt, ctx, draft, analyzer, tracker, None)
        if draft.evidence is not None:
            draft.assessment.conflicts = list(draft.evidence.conflicts)
            draft.assessment.uncertainties = list(draft.evidence.uncertainties)
            draft.freshness_summary = dict(draft.evidence.freshness_summary)
        return draft.result(status, exc)
    except Exception as exc:
        ctx.timers.stop("total")
        log.exception("analysis %s crashed", job.analysis_id)
        await _salvage(rt, job, draft, analyzer)
        draft.telemetry = _telemetry(rt, ctx, draft, analyzer, tracker, None)
        return draft.result("failed", AnalysisError.from_exception(exc))


# ------------------------------------------------------------------------- helpers


async def _salvage(rt: Runtime, job: AnalysisJob, draft: _Draft, analyzer: EquityAnalyzer) -> None:
    """Keep everything already found when a stage fails, so the partial result is inspectable."""
    if not draft.sources and analyzer.state.sources:
        draft.sources = list(analyzer.state.sources)
        job.source_ids = [s.source_id for s in draft.sources]
    if analyzer.research_decisions and not any(
        d.stage == "research_plan" for d in draft.decisions.decisions
    ):
        draft.decisions.decisions.extend(analyzer.research_decisions)
    if draft.identity is None and analyzer.identity is not None:
        draft.identity = analyzer.identity
    if draft.requirements is None and analyzer.requirements is not None:
        # The question was interpreted even if the calculations did not run: report it, with
        # every requirement the analysis never reached listed as unmet.
        draft.requirements = analyzer.validate_requirements(
            draft.evidence,
            draft.calculations,
            prior_available=None if draft.prior is None else True,
        )
    try:
        if draft.sources:
            await rt.store.save_sources(job.analysis_id, draft.sources)
        if draft.decisions.decisions:
            await rt.store.save_decisions(job.analysis_id, draft.decisions.decisions)
    except Exception:  # pragma: no cover - persistence of partials is best effort
        log.exception("could not persist partial artifacts for %s", job.analysis_id)


async def _set_status(rt: Runtime, job: AnalysisJob, status: str) -> None:
    job.status = status  # type: ignore[assignment]
    job.touch()
    await rt.store.update_job(job)


async def _emit_decisions(ctx: AnalysisContext, decisions: list[LayaDecision]) -> None:
    for decision in decisions:
        await ctx.event("laya.decision", **decision.event_view())


def _pack_name(decisions: LayaDecisions) -> str:
    chosen = decisions.latest("calculation_pack")
    return str(chosen.decision) if chosen is not None else "all_standard"


MIN_WEB_SOURCES = 2
"""Kept web pages with extracted text an assessment needs (from ``MIN_WEB_DOMAINS`` sites)."""
MIN_WEB_DOMAINS = 2


def _evidence_gate(
    evidence: NormalizedEvidence,
    gaps: list[str],
    *,
    searches_issued: int = 0,
    searches_failed: int = 0,
) -> None:
    """Refuse to synthesise from too little web evidence (AGENT.md section 24).

    Research is web search only, so the floor is what search can supply: at least
    ``MIN_WEB_SOURCES`` kept pages with extracted text, from at least ``MIN_WEB_DOMAINS``
    different sites (one site repeating itself is not corroboration). When every search of
    the run failed the search backend is down: that is a retryable outage, not a verdict on
    the company. Financial figures and prices are not required: calculations report their
    missing operands and the result says what it is based on.
    """
    if searches_issued and searches_failed >= searches_issued:
        raise AnalysisError(
            ErrorCode.RESEARCH_UNAVAILABLE,
            "Web search failed for every query, so no evidence could be gathered. "
            "Try again shortly.",
            details={
                "reason": "search_failed",
                "searches_issued": searches_issued,
                "searches_failed": searches_failed,
            },
        )
    pages = web_pages(evidence.sources)
    domains = sorted({domain_of(s.url) for s in pages} - {""})
    missing: list[str] = []
    if len(pages) < MIN_WEB_SOURCES:
        missing.append(
            f"at least {MIN_WEB_SOURCES} web pages with readable text (found {len(pages)})"
        )
    if len(domains) < MIN_WEB_DOMAINS:
        missing.append(
            f"pages from at least {MIN_WEB_DOMAINS} different websites (found {len(domains)})"
        )
    if missing:
        raise AnalysisError(
            ErrorCode.INSUFFICIENT_EVIDENCE,
            details={
                "missing": missing,
                "evidence_gaps": gaps,
                "sources": len(evidence.sources),
                "web_pages": len(pages),
                "domains": domains,
                "facts": len(evidence.facts),
            },
        )
    facts_buckets = (evidence.freshness_summary or {}).get("facts") or {}
    prices_fresh = ((evidence.freshness_summary or {}).get("prices") or {}).get("freshness")
    only_stale_facts = bool(facts_buckets.get("stale")) and not (
        facts_buckets.get("current") or facts_buckets.get("recent")
    )
    if only_stale_facts and prices_fresh in (None, "stale") and not evidence.text_evidence:
        raise AnalysisError(
            ErrorCode.STALE_EVIDENCE,
            details={"freshness": {"facts": facts_buckets, "prices": prices_fresh}},
        )


def _search_not_configured() -> AnalysisError:
    return AnalysisError(
        ErrorCode.RESEARCH_UNAVAILABLE,
        "Web search is not configured on this server (BAY_RESEARCH_SEARCH_URL), so no "
        "evidence can be gathered.",
        retryable=False,
        details={"reason": "search_not_configured"},
    )


def _spark_busy(spark: Any) -> bool:
    """One Spark request runs at a time; report when this analysis has to wait for the lane.
    The client's ``busy`` counts a running turn and any queued ones (with pass 1 and pass 2 of
    several analyses contending, a turn is often handed over to a waiter that has not run yet).
    """
    return bool(getattr(spark, "busy", False))


def _child_pids(rt: Runtime) -> dict[str, int]:
    pids: dict[str, int] = {}
    for name, client in (("laya", rt.laya), ("spark", rt.spark)):
        pid = getattr(client, "pid", None)
        if pid is None:
            pid = getattr(getattr(client, "manager", None), "pid", None)
        if callable(pid):
            pid = pid()
        if isinstance(pid, int) and pid > 0:
            pids[name] = pid
    return pids


def _telemetry(
    rt: Runtime,
    ctx: AnalysisContext,
    draft: _Draft,
    analyzer: EquityAnalyzer,
    tracker: PeakTracker,
    spark_stats: Any,
) -> Telemetry:
    elapsed = ctx.timers.elapsed_ms
    spec = rt.spark.profile_spec(draft.job.profile)
    telemetry = Telemetry(
        profile=draft.job.profile,
        context_ceiling=spec.context_ceiling,
        kv_cache_type=spec.kv_cache_type,
        retrieval_ms=elapsed.get("retrieval"),
        normalization_ms=elapsed.get("normalization"),
        laya_ms=elapsed.get("laya"),
        math_ms=elapsed.get("math"),
        spark_total_ms=elapsed.get("spark"),
        total_request_ms=elapsed.get("total"),
        research=analyzer.stats,
        versions=VersionInfo(
            normalization_version=draft.job.normalization_version,
            laya_schema_version=draft.job.laya_schema_version,
            laya_package_version=(rt.extras.get("laya_load") or {}).get("package_version"),
            spark_artifact=(rt.extras.get("spark_version") or {}).get("spark_artifact")
            or draft.job.spark_artifact,
            spark_runtime=(rt.extras.get("spark_version") or {}).get("spark_runtime")
            or draft.job.spark_runtime,
            spark_gguf_sha256=rt.extras.get("spark_lock", {}).get("gguf_sha256"),
            spark_hf_revision=rt.extras.get("spark_lock", {}).get("hf_revision"),
            execution=rt.extras.get("execution") or VersionInfo().execution,
        ),
    )
    laya_load = rt.extras.get("laya_load") or {}
    telemetry.laya_load_ms = laya_load.get("load_ms")
    laya_stats = getattr(rt.laya, "stats", None)
    if isinstance(laya_stats, dict):
        telemetry.laya_resident_ram_mb = laya_stats.get("resident_rss_mb")
        telemetry.laya_warm_inference_ms = laya_stats.get("warm_inference_ms")
    telemetry.query_understanding_ms = elapsed.get(UNDERSTANDING_TIMER)
    if draft.understanding is not None:
        understood = draft.understanding.stats
        telemetry.query_understanding_prompt_tokens = understood.prompt_tokens
        telemetry.query_understanding_output_tokens = understood.output_tokens
        telemetry.query_understanding_load_ms = understood.load_ms
        telemetry.query_understanding_wait_ms = understood.wait_ms
        telemetry.query_understanding_generation_ms = understood.generation_ms
    if spark_stats is not None:
        telemetry.spark_load_ms = spark_stats.load_ms
        telemetry.spark_time_to_first_token_ms = spark_stats.time_to_first_token_ms
        telemetry.spark_total_ms = spark_stats.total_ms
        telemetry.spark_prompt_tokens = spark_stats.prompt_tokens
        telemetry.spark_output_tokens = spark_stats.output_tokens
        telemetry.spark_tokens_per_second = spark_stats.tokens_per_second
        telemetry.spark_resident_ram_mb = spark_stats.resident_rss_mb
        if spark_stats.runtime_version:
            telemetry.versions.spark_runtime = spark_stats.runtime_version
    try:
        telemetry = tracker.to_telemetry(telemetry)
    except Exception:  # pragma: no cover - telemetry must never break a result
        log.exception("telemetry merge failed")
    return telemetry


def _freshness_view(summary: dict[str, Any] | None) -> dict[str, Any]:
    summary = summary or {}
    facts = summary.get("facts") or {}
    prices = summary.get("prices") or {}
    return {
        "facts": {k: facts.get(k) for k in ("current", "recent", "stale", "unknown", "total")},
        "latest_quarter_end": facts.get("latest_quarter_end"),
        "prices": prices.get("freshness"),
        "warnings": len(summary.get("warnings") or []),
    }
