"""Question classification and analytical requirements (the contract side).

The analyst's question decides what an analysis must retrieve and compute. The query is mapped
onto one of a bounded set of question kinds, first by deterministic rules
(:mod:`bayanalytics.instruments.questions`) and, when the rules are unclear or barely
separate two kinds, by one bounded Laya choice (stage ``question_scan``). The kind selects
``AnalyticalRequirements``: the research intents, the calculations and the operands the
question needs, plus the focus sentence Spark is given. After the calculations a
``RequirementsReport`` records which requirements were met; unmet ones become uncertainties on
the result and are shown to Spark, and they never fail the analysis.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

QuestionKind = Literal[
    "general_assessment",
    "thesis_change",
    "valuation",
    "growth",
    "profitability_margins",
    "relative_performance",
    "risk_volatility",
    "guidance_outlook",
    "earnings_reaction",
    "balance_sheet_liquidity",
    "dividends_capital_return",
]
QUESTION_KINDS: tuple[str, ...] = QuestionKind.__args__  # type: ignore[attr-defined]
UNCLEAR = "unclear"
"""The rules' verdict when nothing fired or two kinds tied; never a final kind."""

ClassificationSource = Literal["rules", "laya"]

# Short descriptions: the option texts of the Laya choice (measured against the head limit at
# runtime, so they stay terse) and the labels used in uncertainties.
QUESTION_KIND_DESCRIPTIONS: dict[str, str] = {
    "general_assessment": "overall view of the company",
    "thesis_change": "did new results change the thesis",
    "valuation": "cheap or expensive, multiples",
    "growth": "revenue or earnings growth",
    "profitability_margins": "margins and profitability",
    "relative_performance": "returns versus market or sector",
    "risk_volatility": "risk, volatility, drawdowns",
    "guidance_outlook": "guidance and outlook",
    "earnings_reaction": "latest results and the reaction",
    "balance_sheet_liquidity": "cash, debt, liquidity",
    "dividends_capital_return": "dividends and buybacks",
}
QUESTION_KIND_LABELS: dict[str, str] = {
    "general_assessment": "a general assessment",
    "thesis_change": "whether the thesis changed",
    "valuation": "valuation",
    "growth": "growth",
    "profitability_margins": "margins and profitability",
    "relative_performance": "relative performance",
    "risk_volatility": "risk and volatility",
    "guidance_outlook": "guidance and outlook",
    "earnings_reaction": "the latest results and the market's reaction",
    "balance_sheet_liquidity": "the balance sheet and liquidity",
    "dividends_capital_return": "dividends and capital return",
}
assert set(QUESTION_KIND_DESCRIPTIONS) == set(QUESTION_KINDS) == set(QUESTION_KIND_LABELS)


class QuestionClassification(BaseModel):
    """What the query was classified as, by whom and on which cues."""

    kind: str  # a QuestionKind, or UNCLEAR while the rules alone could not decide
    confidence: float = 0.0
    source: ClassificationSource = "rules"
    cues: list[str] = Field(default_factory=list)  # "kind:matched text", rules only
    candidates: list[str] = Field(default_factory=list)  # kinds the rules scored, best first
    recent_period: bool = False  # the question asks about a specific recent period
    decision_id: str | None = None  # the Laya decision that chose the kind (source == laya)
    note: str | None = None  # why the classification is weaker than it looks

    @property
    def unclear(self) -> bool:
        return self.kind == UNCLEAR


class AnalyticalRequirements(BaseModel):
    """Explicit requirements derived from the question kind (one row of the per-kind table)."""

    question_kind: QuestionKind
    confidence: float
    source: ClassificationSource
    cues: list[str] = Field(default_factory=list)
    recent_period: bool = False
    required_research_intents: list[str] = Field(default_factory=list)
    required_calculations: list[str] = Field(default_factory=list)
    required_operands: list[str] = Field(default_factory=list)
    focus: str = ""
    horizons_emphasis: list[str] = Field(default_factory=list)
    decision_id: str | None = None
    note: str | None = None

    def classification(self) -> QuestionClassification:
        return QuestionClassification(
            kind=self.question_kind,
            confidence=self.confidence,
            source=self.source,
            cues=list(self.cues),
            recent_period=self.recent_period,
            decision_id=self.decision_id,
            note=self.note,
        )


class MissingRequirement(BaseModel):
    name: str
    reason: str
    missing_inputs: list[str] = Field(default_factory=list)


class RequirementsReport(BaseModel):
    """Which of the question's requirements the analysis met (``AnalysisResult.requirements``)."""

    classification: QuestionClassification
    focus: str = ""
    horizons_emphasis: list[str] = Field(default_factory=list)
    required_research_intents: list[str] = Field(default_factory=list)
    executed_research_intents: list[str] = Field(default_factory=list)
    missing_research_intents: list[str] = Field(default_factory=list)
    required_calculations: list[str] = Field(default_factory=list)
    satisfied_calculations: list[str] = Field(default_factory=list)
    missing_calculations: list[MissingRequirement] = Field(default_factory=list)
    required_operands: list[str] = Field(default_factory=list)
    satisfied_operands: list[str] = Field(default_factory=list)
    missing_operands: list[MissingRequirement] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        return not (
            self.missing_calculations or self.missing_operands or self.missing_research_intents
        )
