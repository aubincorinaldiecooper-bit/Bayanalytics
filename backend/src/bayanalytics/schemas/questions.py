"""The analyst's question, interpreted: bounded vocabularies and the requirements contract.

What a question asks is never decided by keyword rules. After the instrument and any explicit
horizon are resolved deterministically, Spark (pass 1) converts the question into a
``QueryUnderstanding``: one ``QuestionIntent``, up to ``MAX_REQUIREMENTS`` ``Requirement`` values,
a ``ComparisonFocus`` and three flags, constrained by the model's JSON schema at generation time
and validated strictly afterwards. Laya then confirms or drops each proposed requirement (stage
``question_validation``), and ordinary Python turns what survives into
``AnalyticalRequirements``: research intents, registry calculations, operands and acceptance
checks. After the calculations a ``RequirementsReport`` records what was met; unmet
requirements are uncertainties, never failures.

Every value below is a product-level name with a documented one-liner (the pass-1 prompt lists
them) and a product label (what events and results show). Results and events carry labels only:
never the raw interpretation, a prompt or model reasoning.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---- question intents -----------------------------------------------------------------------

QuestionIntent = Literal[
    "general_assessment",
    "valuation",
    "valuation_vs_fundamentals",
    "growth",
    "profitability",
    "event_impact",
    "relative_performance",
    "risk",
    "balance_sheet",
    "capital_return",
    "guidance_outlook",
]
QUESTION_INTENTS: tuple[str, ...] = QuestionIntent.__args__  # type: ignore[attr-defined]
GENERAL_ASSESSMENT = "general_assessment"

QUESTION_INTENT_DESCRIPTIONS: dict[str, str] = {
    "general_assessment": "a broad view of the company with no narrower topic",
    "valuation": "whether the shares look cheap or expensive",
    "valuation_vs_fundamentals": "whether the price is justified by earnings and growth",
    "growth": "how fast revenue and earnings are growing and whether that lasts",
    "profitability": "margins, profits and cash generation",
    "event_impact": "what the latest results or a specific event changed",
    "relative_performance": "how the stock did against the market, sector or peers",
    "risk": "volatility, drawdowns and what could go wrong",
    "balance_sheet": "cash, debt and financial strength",
    "capital_return": "dividends, buybacks and payouts",
    "guidance_outlook": "management guidance and the outlook",
}
QUESTION_INTENT_LABELS: dict[str, str] = {
    "general_assessment": "General assessment",
    "valuation": "Valuation",
    "valuation_vs_fundamentals": "Valuation versus fundamentals",
    "growth": "Growth",
    "profitability": "Profitability",
    "event_impact": "Event impact",
    "relative_performance": "Relative performance",
    "risk": "Risk",
    "balance_sheet": "Balance sheet",
    "capital_return": "Capital return",
    "guidance_outlook": "Guidance and outlook",
}

# ---- requirements ---------------------------------------------------------------------------

Requirement = Literal[
    "valuation_multiples",
    "valuation_history",
    "price_vs_earnings",
    "earnings_trajectory",
    "revenue_trajectory",
    "margin_trajectory",
    "cash_flow",
    "price_performance",
    "benchmark_comparison",
    "volatility_drawdown",
    "balance_sheet",
    "capital_return",
    "guidance",
    "latest_period",
    "prior_assessment",
    "recent_coverage",
]
REQUIREMENT_NAMES: tuple[str, ...] = Requirement.__args__  # type: ignore[attr-defined]
MAX_REQUIREMENTS = 8

REQUIREMENT_DESCRIPTIONS: dict[str, str] = {
    "valuation_multiples": "current P/E, price-to-sales and free-cash-flow yield",
    "valuation_history": "where the P/E sits within the company's own history",
    "price_vs_earnings": "how much of the price move earnings growth explains",
    "earnings_trajectory": "the direction of earnings per share over time",
    "revenue_trajectory": "the direction of revenue over time",
    "margin_trajectory": "the direction of gross, operating and net margins",
    "cash_flow": "free cash flow and its margin",
    "price_performance": "the stock's own price returns",
    "benchmark_comparison": "returns and risk against the market and sector",
    "volatility_drawdown": "realised volatility and drawdowns",
    "balance_sheet": "cash, debt, equity and assets",
    "capital_return": "cash available for dividends and buybacks",
    "guidance": "management guidance and commentary",
    "latest_period": "the newest reported quarter",
    "prior_assessment": "comparison with the previous assessment of this company",
    "recent_coverage": "news from recent weeks",
}
REQUIREMENT_LABELS: dict[str, str] = {
    "valuation_multiples": "Valuation multiples",
    "valuation_history": "Valuation history",
    "price_vs_earnings": "Price versus earnings",
    "earnings_trajectory": "Earnings trajectory",
    "revenue_trajectory": "Revenue trajectory",
    "margin_trajectory": "Margin trajectory",
    "cash_flow": "Cash flow",
    "price_performance": "Price performance",
    "benchmark_comparison": "Benchmark comparison",
    "volatility_drawdown": "Volatility and drawdown",
    "balance_sheet": "Balance sheet",
    "capital_return": "Capital return",
    "guidance": "Guidance",
    "latest_period": "Latest period",
    "prior_assessment": "Prior assessment",
    "recent_coverage": "Recent coverage",
}

# ---- comparison focus -----------------------------------------------------------------------

ComparisonFocus = Literal["own_history", "market", "sector", "peers", "none"]
COMPARISON_FOCI: tuple[str, ...] = ComparisonFocus.__args__  # type: ignore[attr-defined]
COMPARISON_FOCUS_DESCRIPTIONS: dict[str, str] = {
    "own_history": "the company against its own past",
    "market": "the stock against the broad market",
    "sector": "the stock against its sector",
    "peers": "the company against named competitors",
    "none": "no comparison asked for",
}
COMPARISON_FOCUS_LABELS: dict[str, str] = {
    "own_history": "the company's own history",
    "market": "the broad market",
    "sector": "the sector",
    "peers": "peers",
    "none": "",
}

FLAG_DESCRIPTIONS: dict[str, str] = {
    "needs_benchmark": "true when the question compares the stock with the market or a sector",
    "needs_prior_assessment": (
        "true when the question asks whether something changed the view or the thesis"
    ),
    "recent_period_focus": (
        "true when the question is about the latest quarter, latest results or a recent event"
    ),
}

InterpretationSource = Literal["spark", "fallback"]

assert set(QUESTION_INTENT_DESCRIPTIONS) == set(QUESTION_INTENTS) == set(QUESTION_INTENT_LABELS)
assert set(REQUIREMENT_DESCRIPTIONS) == set(REQUIREMENT_NAMES) == set(REQUIREMENT_LABELS)
assert set(COMPARISON_FOCUS_DESCRIPTIONS) == set(COMPARISON_FOCI) == set(COMPARISON_FOCUS_LABELS)


class QueryUnderstanding(BaseModel):
    """Spark pass 1's structured reading of the question (never an answer to it).

    Every field is required and no other key is allowed; the JSON schema of this model is what
    the generation is constrained to (:func:`query_understanding_schema`). ``requirements`` is
    deduplicated in order and capped at ``MAX_REQUIREMENTS``.
    """

    model_config = ConfigDict(extra="forbid", strict=True, title="query_understanding")

    intent: QuestionIntent
    requirements: list[Requirement] = Field(max_length=MAX_REQUIREMENTS)
    comparison_focus: ComparisonFocus
    needs_benchmark: bool
    needs_prior_assessment: bool
    recent_period_focus: bool

    @field_validator("requirements")
    @classmethod
    def _dedupe(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))

    @classmethod
    def broad(cls) -> QueryUnderstanding:
        """The interpretation of a broad request ("Assess Apple"): nothing narrower."""
        return cls(
            intent=GENERAL_ASSESSMENT,
            requirements=[],
            comparison_focus="none",
            needs_benchmark=False,
            needs_prior_assessment=False,
            recent_period_focus=False,
        )


_ANNOTATIONS = frozenset({"title", "description"})


def _constraints_only(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _constraints_only(v) for k, v in node.items() if k not in _ANNOTATIONS}
    if isinstance(node, list):
        return [_constraints_only(v) for v in node]
    return node


def query_understanding_schema() -> dict[str, Any]:
    """The JSON schema pass 1 is constrained to: enums, every field required,
    ``additionalProperties: false``. Titles and descriptions are dropped (they constrain
    nothing); the top-level ``title`` names the schema in the ``response_format`` request."""
    schema = QueryUnderstanding.model_json_schema()
    compact = _constraints_only(schema)
    compact["title"] = schema["title"]
    return compact


# ---- requirements (built by instruments/questions.py) --------------------------------------


class AnalyticalRequirements(BaseModel):
    """What the interpreted question requires, composed from the per-requirement table.

    ``requirements`` are the ones kept after Laya's validation (plus those implied by the
    interpretation's flags); ``dropped_by_validation`` the ones Laya did not confirm. The rest
    is the union of the kept requirements' table rows (stable order, deduplicated).
    """

    intent: QuestionIntent
    requirements: list[Requirement] = Field(default_factory=list)
    comparison_focus: ComparisonFocus = "none"
    source: InterpretationSource = "fallback"
    dropped_by_validation: list[Requirement] = Field(default_factory=list)
    recent_period: bool = False
    required_research_intents: list[str] = Field(default_factory=list)
    required_calculations: list[str] = Field(default_factory=list)
    also_calculated: list[str] = Field(default_factory=list)
    """Run with the required ones when their inputs exist, never reported as unmet (the
    three-year reconciliation exists only when the history reaches back that far)."""
    required_operands: list[str] = Field(default_factory=list)
    focus: str = ""
    horizons_emphasis: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # product-level interpretation notes
    decision_ids: list[str] = Field(default_factory=list)  # Laya question_validation decisions

    @property
    def broad(self) -> bool:
        """A general assessment with nothing narrower: the analysis runs exactly as before."""
        return self.intent == GENERAL_ASSESSMENT and not self.requirements

    @property
    def intent_label(self) -> str:
        return QUESTION_INTENT_LABELS[self.intent]

    @property
    def requirement_labels(self) -> list[str]:
        return [REQUIREMENT_LABELS[r] for r in self.requirements]


class MissingRequirement(BaseModel):
    name: str
    reason: str
    missing_inputs: list[str] = Field(default_factory=list)
    requirement: str | None = None  # the product label of the requirement it belongs to


class RequirementsReport(BaseModel):
    """What the question required and what the analysis met (``AnalysisResult.requirements``).

    Product labels and registry names only: never the raw interpretation, a prompt or model
    reasoning.
    """

    question_intent: str
    requirements: list[str] = Field(default_factory=list)
    interpretation_source: InterpretationSource
    dropped_by_validation: list[str] = Field(default_factory=list)
    focus: str = ""
    horizons_emphasis: list[str] = Field(default_factory=list)
    satisfied_requirements: list[str] = Field(default_factory=list)
    unmet_requirements: list[MissingRequirement] = Field(default_factory=list)
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
            self.unmet_requirements
            or self.missing_calculations
            or self.missing_operands
            or self.missing_research_intents
        )
