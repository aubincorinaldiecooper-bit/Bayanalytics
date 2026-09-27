"""Deterministic finance calculations (AGENT.md section 3.3).

Entry points:

* :func:`run_pack` / :func:`run_packs` - run a named calculation pack on ``NormalizedEvidence``;
* :func:`compute` - run one named calculation on explicit operands;
* :func:`require` - opt-in: raise ``MISSING_CALCULATION_INPUT`` for an unavailable result;
* :data:`CALCULATION_PACKS`, :data:`SPECS`, :data:`CANONICAL_METRICS`, :data:`METRIC_UNITS`;
* :func:`format_value` / :func:`format_pair` - display typography;
* :mod:`bayanalytics.calculations.primitives` - the pure arithmetic.
"""

from bayanalytics.calculations import primitives
from bayanalytics.calculations.formatting import (
    UNAVAILABLE,
    format_change,
    format_pair,
    format_series,
    format_value,
)
from bayanalytics.calculations.operands import Operand, OperandResolver, SeriesOperand
from bayanalytics.calculations.registry import (
    CALCULATION_PACKS,
    CANONICAL_METRICS,
    METRIC_UNITS,
    SPECS,
    UNITS,
    CalculationSpec,
    compute,
    require,
    require_all,
    run_pack,
    run_packs,
    spec_for,
)

__all__ = [
    "CALCULATION_PACKS",
    "CANONICAL_METRICS",
    "METRIC_UNITS",
    "SPECS",
    "UNAVAILABLE",
    "UNITS",
    "CalculationSpec",
    "Operand",
    "OperandResolver",
    "SeriesOperand",
    "compute",
    "format_change",
    "format_pair",
    "format_series",
    "format_value",
    "primitives",
    "require",
    "require_all",
    "run_pack",
    "run_packs",
    "spec_for",
]
