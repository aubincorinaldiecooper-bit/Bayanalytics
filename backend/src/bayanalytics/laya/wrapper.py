"""``LayaFinanceWrapper``: turns question sets into auditable ``LayaDecision`` records.

Per call: cancellation check, head/option validation, deterministic state compaction (dropped
keys go to ``ctx.diagnostics["laya_dropped_keys"]``), one ``system_one`` round trip, one decision
per answer. The wrapper emits no events (the orchestrator owns ``laya.*`` events) and accumulates
wall-clock Laya time in ``ctx.timers`` under ``"laya"``. Failures propagate as
``AnalysisError(LAYA_INFERENCE_FAILED)``; state contents never appear in an error.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.laya.base import LAYA_MAX_LEN, LayaClient
from bayanalytics.laya.compaction import (
    compact_state,
    state_budget_for,
    state_digest,
    validate_questions,
)
from bayanalytics.laya.schemas import LAYA_SCHEMA_VERSION
from bayanalytics.schemas.common import ErrorCode, new_id, utcnow
from bayanalytics.schemas.decisions import (
    LayaDecision,
    LayaQuestionSet,
    LayaResult,
    answer_confidence,
)

TIMER_NAME = "laya"
DROPPED_KEYS_DIAGNOSTIC = "laya_dropped_keys"


class LayaFinanceWrapper:
    def __init__(
        self,
        client: LayaClient,
        schema_version: str = LAYA_SCHEMA_VERSION,
        *,
        state_priority: Sequence[str] | None = None,
        state_budget_tokens: int | None = None,
    ) -> None:
        self._client = client
        self._schema_version = schema_version
        self._state_priority = list(state_priority) if state_priority is not None else None
        self._state_budget = state_budget_tokens

    @property
    def client(self) -> LayaClient:
        return self._client

    @property
    def schema_version(self) -> str:
        return self._schema_version

    async def ask(self, question_set: LayaQuestionSet, ctx: AnalysisContext) -> list[LayaDecision]:
        ctx.check_cancelled()
        questions = question_set.questions
        try:
            validate_questions(questions)
        except ValueError as exc:
            raise AnalysisError(
                ErrorCode.LAYA_INFERENCE_FAILED,
                retryable=False,
                details={"reason": "invalid_questions", "message": str(exc)[:300]},
            ) from exc

        budget = self._state_budget or state_budget_for(questions)
        state, dropped = compact_state(question_set.state, budget, self._state_priority)
        if dropped:
            entries = ctx.diagnostics.setdefault(DROPPED_KEYS_DIAGNOSTIC, [])
            entries.append(
                {
                    "stage": question_set.stage,
                    "segment_id": question_set.segment_id,
                    "keys": list(dropped),
                }
            )
        digest = state_digest(state)

        started = time.perf_counter()
        try:
            result = await self._client.system_one(state, questions)
        except AnalysisError:
            raise
        except Exception as exc:
            raise AnalysisError(
                ErrorCode.LAYA_INFERENCE_FAILED,
                details={"reason": "client_failure", "exception": type(exc).__name__},
            ) from exc
        finally:
            ctx.timers.mark(TIMER_NAME, (time.perf_counter() - started) * 1000.0)

        missing = [key for key in questions if key not in result.answers]
        if missing:
            raise AnalysisError(
                ErrorCode.LAYA_INFERENCE_FAILED,
                details={"reason": "missing_answers", "count": len(missing)},
            )

        created_at = utcnow()
        decisions: list[LayaDecision] = []
        for key, question in questions.items():
            answer = result.answers[key]
            decisions.append(
                LayaDecision(
                    decision_id=new_id("dec"),
                    stage=question_set.stage,
                    decision_type=key,
                    question=question,
                    answer=answer,
                    confidence=answer_confidence(answer),
                    state_digest=digest,
                    state_tokens=result.usage.input_tokens,
                    segment_id=question_set.segment_id,
                    created_at=created_at,
                    schema_version=self._schema_version,
                )
            )
        self._check_truncation(question_set, result, ctx)
        return decisions

    async def ask_many(
        self, sets: Sequence[LayaQuestionSet], ctx: AnalysisContext
    ) -> list[LayaDecision]:
        """Sequential: one Laya call at a time (section 38 local concurrency policy)."""
        decisions: list[LayaDecision] = []
        for question_set in sets:
            decisions.extend(await self.ask(question_set, ctx))
        return decisions

    @staticmethod
    def as_dict(decisions: Sequence[LayaDecision]) -> dict[str, Any]:
        """``{decision_type: decision value}`` for quick branching in the orchestrator."""
        return {d.decision_type: d.decision for d in decisions}

    @staticmethod
    def _check_truncation(
        question_set: LayaQuestionSet, result: LayaResult, ctx: AnalysisContext
    ) -> None:
        """Laya truncates the state to ``max_len`` silently; the worker's real token count is
        per batch (every question's sequence includes the state), so the per-question average
        reaching the limit means the compacted state was still too long."""
        tokens = result.usage.input_tokens
        count = len(question_set.questions)
        if not tokens or not count:
            return
        per_question = tokens / count
        if per_question >= LAYA_MAX_LEN:
            ctx.diagnostics.setdefault("laya_truncated", []).append(
                {
                    "stage": question_set.stage,
                    "segment_id": question_set.segment_id,
                    "tokens": per_question,
                }
            )
