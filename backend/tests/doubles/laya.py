"""``RuleLaya``: a deterministic, rule-based ``LayaClient`` double (tests only).

No subprocess, no model. Answers are derived from a handful of well-known state keys
(``revenue_growth_yoy``, ``operating_margin_change_bp``, ``price_return_1m``,
``volatility_30d_annualized``, ``pe_5y_percentile``, ``evidence_gaps``, ``sources_count``,
``freshness``, ``guidance_hint``, ``sentiment_hint``, ...) with stable defaults when they are
missing, so orchestration code sees plausible, repeatable decisions. The
``question_validation`` nouls (``requirement_<name>``, ``requirements_supported``) confirm at
0.8 unless ``force`` pins them; the double never reads the question text (the instructions).
The dynamic research choices read their options' descriptions instead, as a model would:
``instrument_choice`` picks the candidate whose name contains the question's company phrase
and that the most websites named (unsure on a tie), ``open_order`` ranks hits by the words
their site and title share with the topic, ``price_table`` / ``figures_table`` /
``close_column`` / ``line_<metric>`` pick the option whose header or label names the thing
asked for, else ``none``. Every choice answer returns a probability for every option summing
to one; every call is recorded in ``calls``.

Token figures are measured with the doubles' word/punctuation tokenizer (``doubles.tokens``):
``count_tokens`` returns one count per text and ``usage.input_tokens`` is the sequence length
Laya would build per question (frame + head + state, capped at ``max_len``). Nothing here is a
characters-per-token estimate, and ``stats`` keeps every runtime measurement ``None``.
"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bayanalytics.laya.base import LAYA_MAX_LEN, LayaHealth, LayaLoadInfo
from bayanalytics.laya.compaction import (
    SEQUENCE_SPECIALS,
    instruction_text,
    laya_json,
    option_texts,
)
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

from .tokens import count_many, count_tokens

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
class LayaCall:
    state: Any
    questions: dict[str, LayaQuestion]
    result: LayaResult | None = None
    at: float = field(default_factory=time.time)


def sequence_tokens(state: Any, question: LayaQuestion) -> int:
    """Length of the sequence Laya builds for one question over ``state``, measured with the
    doubles' tokenizer: frame + instruction head + one [MASK] per option + state, capped at
    ``LAYA_MAX_LEN`` exactly where Laya would truncate the state."""
    state_tokens = count_tokens(laya_json(state).replace("[MASK]", " "))
    head = count_tokens(instruction_text(question)) + sum(
        1 + count_tokens(text) for text in option_texts(question)
    )
    return min(LAYA_MAX_LEN, SEQUENCE_SPECIALS + head + state_tokens)


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


def _options(q: LayaQuestion) -> dict[str, str]:
    criteria = q.criteria
    if isinstance(criteria, Mapping):
        return {str(k): str(v) for k, v in criteria.items()}
    return {str(k): "" for k in criteria or []}


def _words(text: str) -> set[str]:
    return {w for w in re.split(r"[^a-z0-9%]+", text.lower()) if len(w) >= 3}


def _rule_open_order(state: Mapping[str, Any], q: LayaQuestion) -> ChoiceAnswer:
    topic = _words(_text(state, "question"))
    options = _options(q)
    hits = state.get("hits")
    for hit in hits if isinstance(hits, list) else []:
        if isinstance(hit, Mapping) and str(hit.get("option")) in options:
            options[str(hit["option"])] = f"{hit.get('title', '')} {hit.get('site', '')}"
    weights = {key: 1.0 + 2.0 * len(topic & _words(text)) for key, text in options.items()}
    total = sum(weights.values())
    probs = {key: weight / total for key, weight in weights.items()}
    best = max(probs, key=lambda k: (probs[k], -list(probs).index(k)))
    return ChoiceAnswer(choice=best, probabilities=probs)


def _rule_instrument(state: Mapping[str, Any], q: LayaQuestion) -> ChoiceAnswer:
    phrase = _words(_text(state, "company"))
    sites: dict[str, float] = {}
    candidates = state.get("candidates")
    for item in candidates if isinstance(candidates, list) else []:
        if isinstance(item, Mapping) and isinstance(item.get("sites"), (int, float)):
            sites[str(item.get("option"))] = float(item["sites"])
    matching = [
        key
        for key, name in _options(q).items()
        if key != "none" and phrase and phrase <= _words(name)
    ]
    if not matching:
        return _choice_answer(q, "none", 0.7)
    most = max(sites.get(key, 0.0) for key in matching)
    best = [key for key in matching if sites.get(key, 0.0) == most]
    return _choice_answer(q, best[0], 0.8 if len(best) == 1 else 0.4)


_LINE_RULES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    # metric -> (every word group must appear, none of these may appear)
    "revenue": (("revenue|sales",), ("cost", "growth", "per")),
    "gross_profit": (("gross",), ("%",)),
    "operating_income": (("operating", "income|profit"), ("cash", "margin")),
    "net_income": (("net", "income|earnings|profit"), ("per", "share")),
    "eps_diluted": (("diluted|eps",), ("basic", "shares")),
    "eps_basic": (("basic", "eps|share"), ("shares",)),
    "operating_cash_flow": (("operating", "cash"), ()),
    "capex": (("capital|capex|property",), ("proceeds", "sale")),
    "free_cash_flow": (("free", "cash"), ()),
    "shares_outstanding": (("shares", "outstanding"), ()),
}


def _rule_line(metric: str, q: LayaQuestion) -> ChoiceAnswer:
    need, avoid = _LINE_RULES.get(metric, ((), ()))
    for key, text in _options(q).items():
        if key == "none":
            continue
        words = _words(text) | ({"%"} if "%" in text else set())
        if all(any(alt in words for alt in group.split("|")) for group in need) and not (
            words & set(avoid)
        ):
            return _choice_answer(q, key, 0.8)
    return _choice_answer(q, "none", 0.8)


def _rule_option_named(q: LayaQuestion, *names: str) -> ChoiceAnswer:
    for key, text in _options(q).items():
        if key != "none" and any(name in _words(text) for name in names):
            return _choice_answer(q, key, 0.8)
    return _choice_answer(q, "none", 0.8)


def _rule_close_column(q: LayaQuestion) -> ChoiceAnswer:
    for key, text in _options(q).items():
        if key != "none" and re.match(r"\s*close\b", text.lower()):
            return _choice_answer(q, key, 0.8)
    return _choice_answer(q, "none", 0.8)


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


VALIDATION_CONFIRM = 0.8
"""Default noul for the question_validation keys: the proposed requirement is confirmed."""


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


class RuleLaya:
    """Rule-based ``LayaClient``; see the module docstring for the rules.

    ``force`` pins answers per question key (a choice key, a score level or a noul probability;
    a ``(choice, probability)`` pair pins a choice at that probability, e.g. an unsure one);
    ``raise_error`` makes ``system_one`` raise it, for failure-path tests; ``latency_ms`` makes
    every ``system_one`` really wait that long (the reported latency is the measured wait).
    """

    def __init__(
        self,
        *,
        force: Mapping[str, Any] | None = None,
        raise_error: BaseException | None = None,
        latency_ms: float = 0.0,
    ) -> None:
        self.calls: list[LayaCall] = []
        self.token_calls: list[list[str]] = []
        self.force: dict[str, Any] = dict(force or {})
        self.raise_error = raise_error
        self.latency_ms = latency_ms
        self._loaded = False
        self._closed = False

    # ---- LayaClient ------------------------------------------------------------------------

    async def load(self) -> LayaLoadInfo:
        started = time.perf_counter()
        self._loaded = True
        self._closed = False
        # Nothing is loaded, so nothing about a bundle or a process can be reported.
        return LayaLoadInfo(
            load_ms=round((time.perf_counter() - started) * 1000.0, 3),
            resident_rss_mb=None,
            package_version=None,
            model_dir=None,
        )

    async def system_one(
        self, state: dict[str, Any] | str, questions: dict[str, LayaQuestion]
    ) -> LayaResult:
        call = LayaCall(state=state, questions=dict(questions))
        self.calls.append(call)
        if self.raise_error is not None:
            raise self.raise_error
        if not questions:
            raise ValueError("system_one: at least one question is required")
        started = time.perf_counter()
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)
        view: Mapping[str, Any] = state if isinstance(state, Mapping) else {"text": state}
        answers = {key: self._answer(key, question, view) for key, question in questions.items()}
        usage = sum(sequence_tokens(state, q) for q in questions.values())
        result = LayaResult(
            answers=answers,
            usage=LayaUsage(input_tokens=usage),
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
        call.result = result
        return result

    async def count_tokens(self, texts: Sequence[str]) -> list[int]:
        """One count per text from the doubles' tokenizer (no special tokens)."""
        items = [str(t) for t in texts]
        self.token_calls.append(items)
        return count_many(items)

    async def health(self) -> LayaHealth:
        return LayaHealth(
            ok=not self._closed,
            loaded=self._loaded and not self._closed,
            pid=None,
            resident_rss_mb=None,
            restarts=0,
            detail=None,
        )

    async def close(self) -> None:
        self._closed = True
        self._loaded = False

    @property
    def stats(self) -> dict[str, Any]:
        """A double measures no worker: every measurement is ``None`` so telemetry never
        reports a fake number as a measured one (only the call count is real)."""
        return {
            "load_ms": None,
            "resident_rss_mb": None,
            "peak_rss_mb": None,
            "restarts": 0,
            "requests": len(self.calls),
            "warm_inference_ms": None,
        }

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
        if q.type == "choice" and isinstance(value, tuple):
            choice, top = value
            return _choice_answer(q, str(choice), float(top))
        if q.type == "choice":
            return _choice_answer(q, str(value), 0.9)
        if q.type == "score":
            return _score_answer(q, float(value), sharpness=3.0)
        return NoulAnswer(noul=min(max(float(value), 0.0), 1.0))

    def _choice(self, key: str, q: LayaQuestion, state: Mapping[str, Any]) -> ChoiceAnswer:
        if key == "research_intent":
            return _rule_research_intent(state, q)
        if key == "open_order":
            return _rule_open_order(state, q)
        if key == "instrument_choice":
            return _rule_instrument(state, q)
        if key == "price_table":
            return _rule_option_named(q, "close", "price")
        if key == "figures_table":
            return _rule_option_named(q, "revenue", "sales", "income", "eps")
        if key == "close_column":
            return _rule_close_column(q)
        if key.startswith("line_"):
            return _rule_line(key.removeprefix("line_"), q)
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
        if key == "requirements_supported" or key.startswith("requirement_"):
            return VALIDATION_CONFIRM
        return 0.5
