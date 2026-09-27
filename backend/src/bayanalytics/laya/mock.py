"""Deterministic, rule-based Laya double (``BAY_LAYA_MODE=mock`` and tests of other modules).

No subprocess, no model. Answers are derived from a handful of well-known state keys
(``revenue_growth_yoy``, ``operating_margin_change_bp``, ``price_return_1m``,
``volatility_30d_annualized``, ``pe_5y_percentile``, ``evidence_gaps``, ``sources_count``,
``freshness``, ``guidance_hint``, ``sentiment_hint``, ...) with sane defaults when they are
missing, so orchestration code sees plausible, stable decisions. Every choice answer returns a
probability for every option summing to one; every call is recorded in ``calls``.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from bayanalytics.laya.base import LayaHealth, LayaLoadInfo
from bayanalytics.laya.compaction import estimate_head_tokens, estimate_tokens
from bayanalytics.schemas.common import PRIMARY_SOURCE_TYPES
from bayanalytics.schemas.decisions import (
    ChoiceAnswer,
    LayaAnswer,
    LayaQuestion,
    LayaResult,
    LayaUsage,
    NoulAnswer,
    ScoreAnswer,
)

# Evidence-gap labels (as the research layer names them) -> bounded research intent.
GAP_TO_INTENT: dict[str, str] = {
    "latest_filing": "retrieve_latest_filing",
    "filing": "retrieve_latest_filing",
    "recent_news": "retrieve_recent_news",
    "news": "retrieve_recent_news",
    "historical_coverage": "retrieve_historical_coverage",
    "history": "retrieve_historical_coverage",
    "coverage": "retrieve_historical_coverage",
    "price_history": "retrieve_price_history",
    "prices": "retrieve_price_history",
    "returns": "retrieve_price_history",
    "sector_benchmark": "retrieve_sector_benchmark",
    "benchmark": "retrieve_sector_benchmark",
    "peers": "retrieve_sector_benchmark",
    "earnings_history": "retrieve_earnings_history",
    "earnings": "retrieve_earnings_history",
    "guidance_history": "retrieve_guidance_history",
    "guidance": "retrieve_guidance_history",
    "management_commentary": "retrieve_management_commentary",
    "commentary": "retrieve_management_commentary",
    "transcript": "retrieve_management_commentary",
    "missing_metric": "retrieve_missing_metric",
    "metric": "retrieve_missing_metric",
}

_POSITIVE_HINTS = {"raised", "improving", "up", "positive", "better", "strong", "beat"}
_NEGATIVE_HINTS = {"lowered", "cut", "deteriorating", "down", "negative", "worse", "weak", "miss"}


@dataclass
class MockCall:
    state: Any
    questions: dict[str, LayaQuestion]
    result: LayaResult | None = None
    at: float = field(default_factory=time.time)


# ---- state readers -------------------------------------------------------------------------


def _num(state: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = state.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


def _text(state: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def _hint_sign(hint: str) -> float | None:
    if not hint:
        return None
    if any(word in hint for word in _POSITIVE_HINTS):
        return 1.0
    if any(word in hint for word in _NEGATIVE_HINTS):
        return -1.0
    return 0.0


def _signal(value: float | None, threshold: float) -> float | None:
    if value is None:
        return None
    if value > threshold:
        return 1.0
    if value < -threshold:
        return -1.0
    return 0.0


# ---- answer builders ----------------------------------------------------------------------


def _choice_probs(keys: list[str], preferred: str, top: float) -> dict[str, float]:
    if preferred not in keys:
        preferred = keys[0]
    if len(keys) == 1:
        return {keys[0]: 1.0}
    rest = (1.0 - top) / (len(keys) - 1)
    return {k: (top if k == preferred else rest) for k in keys}


def _choice_answer(question: LayaQuestion, preferred: str, top: float = 0.7) -> ChoiceAnswer:
    criteria = question.criteria
    keys = [str(k) for k in (criteria if isinstance(criteria, Mapping) else criteria or [])]
    probs = _choice_probs(keys, preferred, top)
    return ChoiceAnswer(choice=max(probs, key=probs.__getitem__), probabilities=probs)


def _score_answer(question: LayaQuestion, level: float, sharpness: float = 1.5) -> ScoreAnswer:
    levels = list(question.criteria or [])
    n = max(1, len(levels))
    level = min(max(level, 0.0), float(n - 1))
    weights = [math.exp(-abs(i - level) * sharpness) for i in range(n)]
    total = sum(weights)
    dist = [w / total for w in weights]
    score = sum(i * p for i, p in enumerate(dist))
    return ScoreAnswer(score=round(score, 4), distribution=[round(p, 6) for p in dist])


def _stance_from(signals: list[float | None], top_when_clear: float = 0.72) -> tuple[str, float]:
    present = [s for s in signals if s is not None]
    if not present:
        return "neutral", 0.5
    positives = sum(1 for s in present if s > 0)
    negatives = sum(1 for s in present if s < 0)
    if positives and negatives:
        return "mixed", 0.6
    if positives:
        return "bullish", top_when_clear
    if negatives:
        return "bearish", top_when_clear
    return "neutral", 0.6


# ---- rules ---------------------------------------------------------------------------------


def _rule_research_intent(state: Mapping[str, Any], q: LayaQuestion) -> ChoiceAnswer:
    keys = [str(k) for k in (q.criteria or {})]
    gaps = state.get("evidence_gaps")
    if isinstance(gaps, (list, tuple)):
        for gap in gaps:
            label = str(gap).strip().lower()
            intent = GAP_TO_INTENT.get(label)
            if intent is None:
                if label in keys:
                    intent = label
                elif f"retrieve_{label}" in keys:
                    intent = f"retrieve_{label}"
                else:
                    for fragment, mapped in GAP_TO_INTENT.items():
                        if fragment in label:
                            intent = mapped
                            break
            if intent in keys:
                return _choice_answer(q, intent, 0.75)
            if intent is not None and "retrieve_missing_metric" in keys:
                return _choice_answer(q, "retrieve_missing_metric", 0.55)
    return _choice_answer(q, "stop_research", 0.8)


def _rule_calculation_pack(state: Mapping[str, Any], q: LayaQuestion) -> ChoiceAnswer:
    has_growth = _num(state, "revenue_growth_yoy", "operating_margin_change_bp") is not None
    has_valuation = _num(state, "pe_5y_percentile", "pe_ratio") is not None
    has_benchmark = (
        _num(state, "benchmark_return_1m", "benchmark_return", "excess_return") is not None
    )
    has_volatility = _num(state, "volatility_30d_annualized", "max_drawdown") is not None
    flags = [has_growth, has_valuation, has_benchmark, has_volatility]
    if sum(flags) != 1:
        return _choice_answer(q, "all_standard", 0.6)
    if has_growth:
        return _choice_answer(q, "growth_and_margins")
    if has_valuation:
        return _choice_answer(q, "valuation_vs_history")
    if has_benchmark:
        return _choice_answer(q, "returns_vs_benchmark")
    return _choice_answer(q, "volatility_and_drawdown")


def _horizon_signals(state: Mapping[str, Any], horizon: str) -> list[float | None]:
    growth = _signal(_num(state, "revenue_growth_yoy"), 0.02)
    margins = _signal(_num(state, "operating_margin_change_bp"), 50.0)
    momentum = _signal(_num(state, "price_return_1m", "price_return_3m"), 0.03)
    guidance = _hint_sign(_text(state, "guidance_hint", "guidance"))
    sentiment = _hint_sign(_text(state, "sentiment_hint", "sentiment"))
    pe_pct = _num(state, "pe_5y_percentile")
    valuation = None if pe_pct is None else (-1.0 if pe_pct >= 80 else 1.0 if pe_pct <= 20 else 0.0)
    durability = _hint_sign(_text(state, "moat_hint", "durability_hint"))
    if horizon == "near_term":
        return [momentum, sentiment]
    if horizon == "next_cycle":
        return [guidance, growth, margins]
    if horizon == "medium_term":
        return [valuation, growth, durability]
    if horizon == "long_term":
        return [durability, growth, valuation]
    return [growth, margins, momentum, guidance, sentiment]


def _material(state: Mapping[str, Any]) -> bool:
    growth = _num(state, "revenue_growth_yoy")
    margins = _num(state, "operating_margin_change_bp")
    momentum = _num(state, "price_return_1m")
    return (
        (growth is not None and abs(growth) > 0.10)
        or (margins is not None and abs(margins) > 200)
        or (momentum is not None and abs(momentum) > 0.10)
        or _hint_sign(_text(state, "guidance_hint", "guidance")) not in (None, 0.0)
    )


class MockLaya:
    """Rule-based ``LayaClient``; see the module docstring for the rules.

    ``force`` pins answers per question key (a choice key, a score level or a noul probability);
    ``raise_error`` makes ``system_one`` raise it, for failure-path tests.
    """

    def __init__(
        self,
        *,
        force: Mapping[str, Any] | None = None,
        raise_error: BaseException | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        self.calls: list[MockCall] = []
        self.force: dict[str, Any] = dict(force or {})
        self.raise_error = raise_error
        self.latency_ms = latency_ms
        self._loaded = False
        self._closed = False
        self._requests = 0

    # ---- LayaClient ------------------------------------------------------------------------

    async def load(self) -> LayaLoadInfo:
        self._loaded = True
        self._closed = False
        return LayaLoadInfo(
            load_ms=0.0, resident_rss_mb=0.0, package_version="mock", model_dir="mock"
        )

    async def system_one(
        self, state: dict[str, Any] | str, questions: dict[str, LayaQuestion]
    ) -> LayaResult:
        call = MockCall(state=state, questions=dict(questions))
        self.calls.append(call)
        self._requests += 1
        if self.raise_error is not None:
            raise self.raise_error
        if not questions:
            raise ValueError("system_one: at least one question is required")
        view: Mapping[str, Any] = state if isinstance(state, Mapping) else {"text": state}
        answers = {key: self._answer(key, question, view) for key, question in questions.items()}
        state_tokens = min(estimate_tokens(state), 512)
        usage = sum(state_tokens + estimate_head_tokens(q) + 4 for q in questions.values())
        result = LayaResult(
            answers=answers, usage=LayaUsage(input_tokens=usage), latency_ms=self.latency_ms
        )
        call.result = result
        return result

    async def health(self) -> LayaHealth:
        return LayaHealth(
            ok=not self._closed,
            loaded=self._loaded and not self._closed,
            pid=os.getpid(),
            resident_rss_mb=0.0,
            restarts=0,
            detail="mock",
        )

    async def close(self) -> None:
        self._closed = True
        self._loaded = False

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "load_ms": 0.0,
            "resident_rss_mb": 0.0,
            "peak_rss_mb": 0.0,
            "restarts": 0,
            "requests": self._requests,
            "warm_inference_ms": self.latency_ms,
        }

    # ---- rules -----------------------------------------------------------------------------

    def _answer(self, key: str, q: LayaQuestion, state: Mapping[str, Any]) -> LayaAnswer:
        if key in self.force:
            return self._forced(q, self.force[key])
        if q.type == "choice":
            return self._choice(key, q, state)
        if q.type == "score":
            return self._score(key, q, state)
        return NoulAnswer(noul=self._noul(key, state))

    @staticmethod
    def _forced(q: LayaQuestion, value: Any) -> LayaAnswer:
        if q.type == "choice":
            return _choice_answer(q, str(value), 0.9)
        if q.type == "score":
            return _score_answer(q, float(value), sharpness=3.0)
        return NoulAnswer(noul=min(max(float(value), 0.0), 1.0))

    def _choice(self, key: str, q: LayaQuestion, state: Mapping[str, Any]) -> ChoiceAnswer:
        if key == "research_intent":
            return _rule_research_intent(state, q)
        if key == "calculation_pack":
            return _rule_calculation_pack(state, q)
        if key == "guidance_trend":
            sign = _hint_sign(_text(state, "guidance_hint", "guidance"))
            if sign is None or sign == 0.0:
                return _choice_answer(q, "unchanged", 0.6 if sign is None else 0.7)
            return _choice_answer(q, "improving" if sign > 0 else "deteriorating")
        if key == "sentiment_trend":
            sign = _hint_sign(_text(state, "sentiment_hint", "sentiment"))
            if sign is None:
                sign = _signal(_num(state, "price_return_1m"), 0.03)
            if sign is None or sign == 0.0:
                return _choice_answer(q, "stable", 0.6)
            return _choice_answer(q, "improving" if sign > 0 else "weakening")
        if key == "volatility_regime":
            vol = _num(state, "volatility_30d_annualized", "volatility_annualized")
            if vol is None:
                return _choice_answer(q, "normal", 0.5)
            return _choice_answer(
                q, "low" if vol < 0.20 else "elevated" if vol > 0.40 else "normal"
            )
        if key == "margin_direction":
            sign = _signal(_num(state, "operating_margin_change_bp"), 50.0)
            if sign is None or sign == 0.0:
                return _choice_answer(q, "stable", 0.5 if sign is None else 0.7)
            return _choice_answer(q, "expanding" if sign > 0 else "contracting")
        if key == "benchmark_relative":
            excess = _num(state, "excess_return")
            if excess is None:
                own = _num(state, "price_return_1m", "price_return")
                bench = _num(state, "benchmark_return_1m", "benchmark_return")
                excess = None if own is None or bench is None else own - bench
            sign = _signal(excess, 0.02)
            if sign is None or sign == 0.0:
                return _choice_answer(q, "in_line", 0.5 if sign is None else 0.7)
            return _choice_answer(q, "outperforming" if sign > 0 else "underperforming")
        if key == "drawdown_nature":
            own = _num(state, "max_drawdown", "drawdown")
            bench = _num(state, "benchmark_drawdown", "benchmark_max_drawdown")
            if own is None or bench is None:
                return _choice_answer(q, "mixed", 0.5)
            ratio = abs(bench) / abs(own) if own else 1.0
            if ratio >= 0.75:
                return _choice_answer(q, "market_wide")
            if ratio <= 0.35:
                return _choice_answer(q, "idiosyncratic")
            return _choice_answer(q, "mixed", 0.6)
        if key == "evidence_stance" or key.startswith("horizon_stance_"):
            horizon = key.removeprefix("horizon_stance_") if key != "evidence_stance" else ""
            stance, top = _stance_from(_horizon_signals(state, horizon))
            return _choice_answer(q, stance, top)
        keys = [
            str(k) for k in (q.criteria if isinstance(q.criteria, Mapping) else q.criteria or [])
        ]
        return _choice_answer(q, keys[0], 0.5)

    def _score(self, key: str, q: LayaQuestion, state: Mapping[str, Any]) -> ScoreAnswer:
        n = len(q.criteria or [])
        middle = (n - 1) / 2
        if key == "valuation_extremeness":
            pct = _num(state, "pe_5y_percentile")
            if pct is None:
                return _score_answer(q, middle, sharpness=0.8)
            return _score_answer(q, min(max(pct, 0.0), 100.0) / 100.0 * (n - 1))
        if key == "revenue_momentum":
            growth = _num(state, "revenue_growth_yoy")
            if growth is None:
                return _score_answer(q, middle, sharpness=0.8)
            level = 0 if growth <= -0.10 else 1 if growth < 0 else 2 if growth <= 0.05 else 3
            if growth > 0.25:
                level = 4
            return _score_answer(q, float(level))
        if key == "growth_durability":
            growth = _num(state, "revenue_growth_yoy", "revenue_cagr_3y")
            streak = _num(state, "growth_streak_quarters")
            level = middle
            if growth is not None:
                level = 1.0 if growth < 0 else 2.0 if growth < 0.05 else 3.0
                if streak is not None and streak >= 8 and growth >= 0.05:
                    level = 4.0
                if streak is not None and streak <= 1 and growth < 0:
                    level = 0.0
            return _score_answer(q, level, sharpness=1.0 if growth is None else 1.5)
        return _score_answer(q, middle, sharpness=0.8)

    def _noul(self, key: str, state: Mapping[str, Any]) -> float:
        if key == "evidence_sufficient":
            count = _num(state, "sources_count")
            return 0.8 if count is not None and count >= 6 else 0.3
        if key == "material_change":
            return 0.75 if _material(state) else 0.3
        if key == "escalate_to_spark":
            conflicts = _num(state, "conflicts_count")
            return 0.7 if _material(state) or (conflicts or 0) > 0 else 0.35
        if key == "historically_unusual":
            pct = _num(state, "pe_5y_percentile")
            vol = _num(state, "volatility_30d_annualized")
            growth = _num(state, "revenue_growth_yoy")
            unusual = (
                (pct is not None and (pct >= 90 or pct <= 10))
                or (vol is not None and vol > 0.5)
                or (growth is not None and abs(growth) > 0.3)
            )
            return 0.7 if unusual else 0.3
        if key == "source_is_material":
            source_type = _text(state, "source_type")
            if source_type in PRIMARY_SOURCE_TYPES:
                return 0.75
            return 0.45 if source_type else 0.5
        if key == "stale_evidence_matters":
            freshness = _text(state, "freshness")
            return 0.7 if freshness in {"stale", "unknown"} else 0.25
        return 0.5
