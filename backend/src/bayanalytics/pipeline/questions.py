"""Question classification stage: rules first, one bounded Laya choice when they are unclear.

Runs at the start of the research phase, before the first retrieval round. The deterministic
classifier (:mod:`bayanalytics.instruments.questions`) decides where it can; when it is unclear
or low-confidence, Laya answers ``question_kind_questions`` over a compact state (stage
``question_scan``, recorded like every other Laya decision through ``laya.started`` /
``laya.decision`` / ``laya.completed``). Laya only picks the kind: the requirements are a table
lookup and the research plan is built by ordinary code from them.
"""

from __future__ import annotations

import logging

from bayanalytics.context import AnalysisContext
from bayanalytics.instruments.questions import (
    classification_from_laya,
    classify_question,
    needs_laya,
    requirements_for,
)
from bayanalytics.laya.schemas import STAGE_QUESTION_SCAN, question_kind_questions
from bayanalytics.laya.wrapper import LayaFinanceWrapper
from bayanalytics.schemas.decisions import LayaDecision, LayaQuestionSet
from bayanalytics.schemas.questions import AnalyticalRequirements

log = logging.getLogger(__name__)

MAX_CANDIDATES_IN_STATE = 3
MAX_CUES_IN_STATE = 6


async def resolve_requirements(
    query: str,
    resolved_horizon: str,
    laya: LayaFinanceWrapper,
    ctx: AnalysisContext,
    *,
    instrument: str | None = None,
) -> tuple[AnalyticalRequirements, list[LayaDecision]]:
    """Classify ``query`` and return its requirements plus the Laya decisions it took (if any)."""
    classification = classify_question(query, resolved_horizon)
    decisions: list[LayaDecision] = []
    if needs_laya(classification):
        state = {
            "instrument": instrument,
            "question": query,
            "horizon": resolved_horizon,
            "rule_candidates": classification.candidates[:MAX_CANDIDATES_IN_STATE],
            "rule_cues": classification.cues[:MAX_CUES_IN_STATE],
        }
        question_set = LayaQuestionSet(
            stage=STAGE_QUESTION_SCAN, state=state, questions=question_kind_questions()
        )
        await ctx.event(
            "laya.started", stage=STAGE_QUESTION_SCAN, questions=len(question_set.questions)
        )
        decisions = await laya.ask(question_set, ctx)
        for decision in decisions:
            await ctx.event("laya.decision", **decision.event_view())
        await ctx.event("laya.completed", stage=STAGE_QUESTION_SCAN, decisions=len(decisions))
        classification = classification_from_laya(classification, decisions)
    requirements = requirements_for(classification)
    log.info(
        "question classified as %s by %s (confidence %.2f)",
        requirements.question_kind,
        requirements.source,
        requirements.confidence,
    )
    return requirements, decisions
