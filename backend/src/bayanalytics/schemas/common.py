"""Shared literals, enums and small helpers used by every schema."""

from __future__ import annotations

import secrets
from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

Profile = Literal["fast", "deep"]
PROFILES: tuple[Profile, ...] = ("fast", "deep")

Horizon = Literal["auto", "near_term", "next_cycle", "medium_term", "long_term", "multi_horizon"]
ResolvedHorizon = Literal["near_term", "next_cycle", "medium_term", "long_term", "multi_horizon"]
SingleHorizon = Literal["near_term", "next_cycle", "medium_term", "long_term"]
SINGLE_HORIZONS: tuple[SingleHorizon, ...] = (
    "near_term",
    "next_cycle",
    "medium_term",
    "long_term",
)
HORIZON_LABELS: dict[str, str] = {
    "near_term": "Near term (days to several weeks)",
    "next_cycle": "Next cycle (next earnings / quarter)",
    "medium_term": "Medium term (6-12 months)",
    "long_term": "Long term (multi-year)",
    "multi_horizon": "Multi-horizon",
}

AnalysisStatus = Literal[
    "queued",
    "resolving_instrument",
    "researching",
    "normalizing",
    "scoring",
    "calculating",
    "synthesizing",
    "completed",
    "failed",
    "cancelled",
]
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})


class ErrorCode(StrEnum):
    AMBIGUOUS_INSTRUMENT = "AMBIGUOUS_INSTRUMENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    STALE_EVIDENCE = "STALE_EVIDENCE"
    RESEARCH_UNAVAILABLE = "RESEARCH_UNAVAILABLE"
    SOURCE_CONFLICT = "SOURCE_CONFLICT"
    MISSING_CALCULATION_INPUT = "MISSING_CALCULATION_INPUT"
    FAST_PROFILE_UNAVAILABLE = "FAST_PROFILE_UNAVAILABLE"
    DEEP_PROFILE_UNAVAILABLE = "DEEP_PROFILE_UNAVAILABLE"
    MEMORY_PRESSURE = "MEMORY_PRESSURE"
    SPARK_START_FAILED = "SPARK_START_FAILED"
    SPARK_INFERENCE_FAILED = "SPARK_INFERENCE_FAILED"
    LAYA_INFERENCE_FAILED = "LAYA_INFERENCE_FAILED"
    WHISPER_FAILED = "WHISPER_FAILED"
    INTERRUPTED = "INTERRUPTED"
    CANCELLED = "CANCELLED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    # HTTP-level codes outside the analysis failure set (never emitted on the SSE stream).
    NOT_FOUND = "NOT_FOUND"
    INVALID_REQUEST = "INVALID_REQUEST"


Stance = Literal["bullish", "neutral", "bearish", "mixed"]

# Source hierarchy (AGENT.md section 22). Lower rank = more authoritative. Contextual, not
# absolute: the ranking is exposed to Laya and the analyst, never used to silently drop data.
SourceType = Literal[
    "regulatory_filing",
    "exchange_data",
    "investor_relations",
    "financial_statement",
    "earnings_release",
    "earnings_transcript",
    "market_data",
    "financial_journalism",
    "secondary_commentary",
    "unverified_web",
]
SOURCE_HIERARCHY: dict[str, int] = {
    "regulatory_filing": 1,
    "exchange_data": 1,
    "investor_relations": 2,
    "financial_statement": 3,
    "earnings_release": 4,
    "earnings_transcript": 4,
    "market_data": 5,
    "financial_journalism": 6,
    "secondary_commentary": 7,
    "unverified_web": 8,
}
PRIMARY_SOURCE_TYPES: frozenset[str] = frozenset(
    {
        "regulatory_filing",
        "exchange_data",
        "investor_relations",
        "financial_statement",
        "earnings_release",
        "earnings_transcript",
    }
)

Freshness = Literal["current", "recent", "stale", "unknown"]
Basis = Literal["gaap", "adjusted", "unknown"]
PriceType = Literal["intraday", "latest_close", "pre_market", "after_hours", "historical_close"]
Redistribution = Literal["allowed", "metadata_only", "unknown"]


def utcnow() -> datetime:
    return datetime.now(tz=UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(8)}"


def source_rank(source_type: str) -> int:
    return SOURCE_HIERARCHY.get(source_type, 8)


def is_primary_source(source_type: str) -> bool:
    return source_type in PRIMARY_SOURCE_TYPES
