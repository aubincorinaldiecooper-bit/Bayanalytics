"""Deterministic calculation records (AGENT.md section 3.3, 25)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class CalculationInput(BaseModel):
    name: str
    value: float | None
    unit: str | None = None
    source_id: str | None = None
    period_label: str | None = None
    fact_id: str | None = None


class CalculationResult(BaseModel):
    """Every calculation records its formula and every operand so it can be reproduced."""

    calc_id: str
    name: str
    formula: str
    inputs: list[CalculationInput] = Field(default_factory=list)
    value: float | None = None
    unit: str = "ratio"
    period_label: str | None = None
    status: Literal["computed", "unavailable"] = "computed"
    missing_inputs: list[str] = Field(default_factory=list)
    display: str = ""
    notes: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    def event_view(self) -> dict[str, Any]:
        return {
            "calc_id": self.calc_id,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "display": self.display,
            "status": self.status,
            "period_label": self.period_label,
            "missing_inputs": self.missing_inputs,
        }
