"""Laya question / answer / decision schemas (AGENT.md sections 1.2, 2, 3.2)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

LayaQuestionType = Literal["choice", "score", "noul"]


class LayaQuestion(BaseModel):
    """One typed question in the exact shape ``@receptron/laya`` ``systemOne`` accepts."""

    type: LayaQuestionType
    instructions: str
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def _check_criteria(self) -> LayaQuestion:
        if self.type == "choice" and not isinstance(self.criteria, dict):
            raise ValueError("choice questions need a criteria mapping option -> description")
        if self.type == "score" and not isinstance(self.criteria, list):
            raise ValueError("score questions need an ordered criteria list")
        if self.type == "noul" and self.criteria is not None:
            raise ValueError("noul questions take no criteria")
        return self

    def to_laya(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.type, "instructions": self.instructions}
        if self.criteria is not None:
            payload["criteria"] = self.criteria
        return payload


class ChoiceAnswer(BaseModel):
    choice: str
    probabilities: dict[str, float]


class ScoreAnswer(BaseModel):
    score: float
    distribution: list[float] | None = None


class NoulAnswer(BaseModel):
    noul: float


LayaAnswer = ChoiceAnswer | ScoreAnswer | NoulAnswer


class LayaUsage(BaseModel):
    input_tokens: int | None = None


class LayaResult(BaseModel):
    answers: dict[str, LayaAnswer]
    usage: LayaUsage = Field(default_factory=LayaUsage)
    latency_ms: float | None = None


def answer_confidence(answer: LayaAnswer) -> float:
    """Confidence in the structured decision itself (never a market-outcome probability)."""
    if isinstance(answer, ChoiceAnswer):
        return max(answer.probabilities.values(), default=0.0)
    if isinstance(answer, NoulAnswer):
        return max(answer.noul, 1.0 - answer.noul)
    if answer.distribution:
        return max(answer.distribution)
    return 0.0


def answer_value(answer: LayaAnswer) -> str | float:
    if isinstance(answer, ChoiceAnswer):
        return answer.choice
    if isinstance(answer, NoulAnswer):
        return answer.noul
    return answer.score


class LayaDecision(BaseModel):
    """A recorded, auditable Laya decision."""

    decision_id: str
    stage: str  # research_plan, evidence_scan, history_scan, horizon, calculation, ...
    decision_type: str  # the question key
    question: LayaQuestion
    answer: LayaAnswer
    confidence: float
    state_digest: str  # sha256 of the compacted state JSON
    state_tokens: int | None = None
    segment_id: str | None = None
    created_at: datetime
    schema_version: str = ""

    @property
    def decision(self) -> str | float:
        return answer_value(self.answer)

    def event_view(self) -> dict[str, Any]:
        value = self.decision
        return {
            "decision_id": self.decision_id,
            "stage": self.stage,
            "decision_type": self.decision_type,
            "decision": value if isinstance(value, str) else round(float(value), 4),
            "confidence": round(self.confidence, 4),
            "segment_id": self.segment_id,
        }


class LayaQuestionSet(BaseModel):
    """A batch of questions that share one compact state."""

    stage: str
    state: dict[str, Any]
    questions: dict[str, LayaQuestion]
    segment_id: str | None = None
