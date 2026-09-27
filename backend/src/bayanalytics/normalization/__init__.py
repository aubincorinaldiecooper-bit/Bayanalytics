"""Input normalization: units, currencies, fiscal periods, market sessions, facts,
freshness and corporate actions (AGENT.md sections 3.1, 6, 22, 32, 34).

``NORMALIZATION_VERSION`` is recorded on every job and result so a stored analysis can be
matched to the rules that produced it; bump it whenever a rule here changes a value or a
label.
"""

from bayanalytics.normalization.corporate_actions import (
    apply_split_adjustment,
    comparable_periods,
    detect_identity_breaks,
    parse_split_ratio,
    total_return_note,
)
from bayanalytics.normalization.facts import (
    FactBuild,
    build_facts,
    classify_freshness,
    detect_stale_mix,
    fact_freshness,
    freshness_summary,
    normalize_unit,
)
from bayanalytics.normalization.periods import (
    CURRENT_QUARTER_AMBIGUOUS,
    calendar_quarter,
    calendar_to_fiscal,
    classify_duration,
    current_fiscal_quarter,
    is_consecutive_quarters,
    label,
    parse_period_label,
    period_from_xbrl,
    ttm_period,
    ttm_window,
)
from bayanalytics.normalization.sessions import (
    TimestampSet,
    classify_price_timestamp,
    label_series,
    latest_completed_close,
    latest_completed_session,
    session_view,
    to_exchange_time,
)
from bayanalytics.normalization.units import (
    ParsedNumber,
    assert_same_currency,
    canonical_scale,
    normalize_currency_code,
    parse_number,
    scale_to_base,
    try_parse_number,
)

NORMALIZATION_VERSION = "2026.09-1"

__all__ = [
    "CURRENT_QUARTER_AMBIGUOUS",
    "NORMALIZATION_VERSION",
    "FactBuild",
    "ParsedNumber",
    "TimestampSet",
    "apply_split_adjustment",
    "assert_same_currency",
    "build_facts",
    "calendar_quarter",
    "calendar_to_fiscal",
    "canonical_scale",
    "classify_duration",
    "classify_freshness",
    "classify_price_timestamp",
    "comparable_periods",
    "current_fiscal_quarter",
    "detect_identity_breaks",
    "detect_stale_mix",
    "fact_freshness",
    "freshness_summary",
    "is_consecutive_quarters",
    "label",
    "label_series",
    "latest_completed_close",
    "latest_completed_session",
    "normalize_currency_code",
    "normalize_unit",
    "parse_number",
    "parse_period_label",
    "parse_split_ratio",
    "period_from_xbrl",
    "scale_to_base",
    "session_view",
    "to_exchange_time",
    "total_return_note",
    "try_parse_number",
    "ttm_period",
    "ttm_window",
]
