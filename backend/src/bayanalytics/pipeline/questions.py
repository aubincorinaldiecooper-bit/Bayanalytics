"""Question validation stage: Laya constrains Spark's interpretation, Python builds the work.

Runs after Spark pass 1 (:mod:`bayanalytics.pipeline.understanding`) and before the first
retrieval round. Laya answers one noul per requirement Spark proposed (its requirements plus
those its flags imply) and ``requirements_supported``, over a compact state holding the
question, the instrument, the horizon and the proposed intent and requirements as product
labels (stage ``question_validation``, recorded through ``laya.started`` / ``laya.decision`` /
``laya.completed`` and persisted with every other decision). The combination rule is ordinary
code (:func:`bayanalytics.instruments.questions.combine_validation`): Laya can drop a proposed
requirement or reject the whole interpretation, never add one. With nothing proposed (a broad
request, or the fallback) there is nothing to validate and Laya is not asked.
"""

from __future__ import annotations

import logging

from bayanalytics.context import AnalysisContext
from bayanalytics.instruments.base import InstrumentIdentity
from bayanalytics.instruments.questions import (
    build_requirements,
    combine_validation,
    proposed_requirements,
    rejected_requirements,
)
from bayanalytics.laya.schemas import STAGE_QUESTION_VALIDATION, requirement_validation_questions
from bayanalytics.laya.wrapper import LayaFinanceWrapper
from bayanalytics.pipeline.understanding import QUESTION_MAX_CHARS, Understanding
from bayanalytics.schemas.decisions import LayaDecision, LayaQuestionSet
from bayanalytics.schemas.questions import (
    QUESTION_INTENT_LABELS,
    REQUIREMENT_LABELS,
    AnalyticalRequirements,
)
from bayanalytics.spark.prompt import clean_text

log = logging.getLogger(__name__)


async def resolve_requirements(
    understanding: Understanding,
    query: str,
    identity: InstrumentIdentity,
    resolved_horizon: str,
    laya: LayaFinanceWrapper,
    ctx: AnalysisContext,
) -> tuple[AnalyticalRequirements, list[LayaDecision]]:
    """Validate the interpretation with Laya (when it proposes anything) and build the
    requirements; returns them with the Laya decisions taken."""
    interpretation = understanding.understanding
    proposal = proposed_requirements(interpretation)
    if not proposal:
        requirements = build_requirements(
            interpretation, source=understanding.source, kept=[], notes=understanding.notes
        )
        log.info("question requirements: intent=%s, nothing to validate", requirements.intent)
        return requirements, []
    state = {
        "question": clean_text(query, QUESTION_MAX_CHARS),
        "instrument": identity.symbol,
        "horizon": resolved_horizon,
        "intent": QUESTION_INTENT_LABELS[interpretation.intent],
        "requirements": [REQUIREMENT_LABELS[name] for name in proposal],
    }
    question_set = LayaQuestionSet(
        stage=STAGE_QUESTION_VALIDATION,
        state=state,
        questions=requirement_validation_questions(proposal),
    )
    await ctx.event(
        "laya.started", stage=STAGE_QUESTION_VALIDATION, questions=len(question_set.questions)
    )
    decisions = await laya.ask(question_set, ctx)
    for decision in decisions:
        await ctx.event("laya.decision", **decision.event_view())
    await ctx.event("laya.completed", stage=STAGE_QUESTION_VALIDATION, decisions=len(decisions))
    outcome = combine_validation(proposal, decisions)
    if outcome.rejected:
        requirements = rejected_requirements(proposal, understanding.notes, outcome.decision_ids)
    else:
        requirements = build_requirements(
            interpretation,
            source=understanding.source,
            kept=outcome.kept,
            dropped=outcome.dropped,
            notes=understanding.notes,
            decision_ids=outcome.decision_ids,
        )
    log.info(
        "question requirements: intent=%s kept=%d dropped=%d rejected=%s",
        requirements.intent,
        len(requirements.requirements),
        len(requirements.dropped_by_validation),
        outcome.rejected,
    )
    return requirements, decisions
