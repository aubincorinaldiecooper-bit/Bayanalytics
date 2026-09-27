"""Finance question schemas, measured head sizing, compaction, ``RuleLaya`` and the wrapper.

Every token figure in these tests comes from the test's own ``counter`` (one token per word or
punctuation mark, the rule the doubles and ``stub_laya.mjs`` share); nothing is estimated.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import pytest

from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.laya import schemas
from bayanalytics.laya.base import LAYA_HEAD_MAX_LEN, LAYA_MAX_LEN
from bayanalytics.laya.compaction import (
    DEFAULT_STATE_PRIORITY,
    MAX_LIST_ITEMS,
    MAX_STRING_CHARS,
    MIN_STATE_BUDGET_TOKENS,
    SEQUENCE_SPECIALS,
    HeadMeasure,
    canonical_json,
    compact_state,
    instruction_text,
    laya_json,
    measure_heads,
    option_texts,
    rendered_options,
    state_budget_for,
    state_digest,
    validate_questions,
)
from bayanalytics.laya.wrapper import LayaFinanceWrapper
from bayanalytics.schemas.common import SINGLE_HORIZONS, ErrorCode
from bayanalytics.schemas.decisions import (
    ChoiceAnswer,
    LayaQuestion,
    LayaQuestionSet,
    NoulAnswer,
    ScoreAnswer,
    answer_confidence,
)
from doubles import RuleLaya

_TOKEN = re.compile(r"\w+|[^\w\s]")


def tokens(text: Any) -> int:
    return len(_TOKEN.findall(str(text)))


async def counter(texts: Sequence[str]) -> list[int]:
    """The test's own ``TokenCounter``: one token per word or punctuation mark."""
    return [tokens(t) for t in texts]


def state_tokens(state: Any) -> int:
    """What the counter sees for a state: Laya's JSON rendering with [MASK] scrubbed."""
    return tokens(laya_json(state).replace("[MASK]", " "))


def head_tokens(question: LayaQuestion) -> int:
    return tokens(instruction_text(question)) + sum(
        1 + tokens(text) for text in option_texts(question)
    )


# ---- schemas ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [
        schemas.question_kind_questions,
        schemas.research_plan_questions,
        schemas.evidence_scan_questions,
        schemas.history_segment_questions,
        lambda: schemas.history_segment_questions(include_volatility=True),
        schemas.calculation_questions,
        lambda: schemas.horizon_questions(SINGLE_HORIZONS),
        lambda: schemas.horizon_questions(["multi_horizon"]),
        schemas.synthesis_gate_questions,
        schemas.overall_scan_questions,
        schemas.text_evidence_questions,
        lambda: schemas.horizon_context_questions(SINGLE_HORIZONS),
    ],
)
async def test_every_builder_validates_and_fits_the_measured_head(builder) -> None:
    batch = builder()
    assert batch
    validate_questions(batch)
    heads = await measure_heads(batch, counter)
    assert set(heads) == set(batch)
    for key, question in batch.items():
        assert isinstance(question, LayaQuestion)
        assert question.to_laya()["type"] == question.type
        head = heads[key]
        assert head.truncation() is None, key
        assert head.total <= LAYA_HEAD_MAX_LEN, key
        assert head == HeadMeasure(
            tokens(instruction_text(question)),
            tuple(tokens(t) for t in option_texts(question)),
        )
        assert len(head.option_tokens) == len(rendered_options(question))
    largest = max(h.total for h in heads.values())
    budget = state_budget_for(heads)
    assert budget == LAYA_MAX_LEN - SEQUENCE_SPECIALS - largest
    assert MIN_STATE_BUDGET_TOKENS <= budget < LAYA_MAX_LEN


def test_required_question_shapes() -> None:
    schemas.validate_all_builders()
    plan = schemas.research_plan_questions()
    intent = plan["research_intent"]
    assert intent.type == "choice"
    assert tuple(intent.criteria) == schemas.RESEARCH_INTENTS
    assert len(set(intent.criteria.values())) == len(intent.criteria)  # distinct descriptions
    assert plan["evidence_sufficient"].type == "noul"

    for key in (
        "evidence_sufficient",
        "material_change",
        "historically_unusual",
        "escalate_to_spark",
        "source_is_material",
        "stale_evidence_matters",
    ):
        assert schemas.ALL_QUESTIONS[key].type == "noul", key

    expected_choices = {
        "guidance_trend": ("deteriorating", "unchanged", "improving"),
        "sentiment_trend": ("weakening", "stable", "improving"),
        "volatility_regime": ("low", "normal", "elevated"),
        "margin_direction": ("contracting", "stable", "expanding"),
        "evidence_stance": ("bullish", "neutral", "bearish", "mixed"),
        "calculation_pack": schemas.CALCULATION_PACKS,
        "benchmark_relative": ("underperforming", "in_line", "outperforming"),
        "drawdown_nature": ("idiosyncratic", "market_wide", "mixed"),
    }
    for key, options in expected_choices.items():
        question = schemas.ALL_QUESTIONS[key]
        assert question.type == "choice" and tuple(question.criteria) == options, key

    assert schemas.ALL_QUESTIONS["valuation_extremeness"].criteria == list(schemas.VALUATION_LEVELS)
    for key in ("revenue_momentum", "growth_durability"):
        question = schemas.ALL_QUESTIONS[key]
        assert question.type == "score" and len(question.criteria) == 5, key

    horizon = schemas.horizon_questions(["near_term", "long_term"])
    assert list(horizon) == ["horizon_stance_near_term", "horizon_stance_long_term"]
    assert "volatility" in horizon["horizon_stance_near_term"].instructions
    assert "earnings proximity" in horizon["horizon_stance_near_term"].instructions
    assert "guidance" in schemas.HORIZON_STANCE["next_cycle"].instructions
    assert "valuation" in schemas.HORIZON_STANCE["medium_term"].instructions
    assert "market structure" in horizon["horizon_stance_long_term"].instructions
    for question in horizon.values():
        assert tuple(question.criteria) == schemas.STANCES
    with pytest.raises(ValueError):
        schemas.horizon_questions(["auto"])
    with pytest.raises(ValueError):
        schemas.horizon_questions([])
    # history questions only ask about volatility when the state carries a measured one
    assert "volatility_regime" not in schemas.history_segment_questions()
    assert "volatility_regime" in schemas.history_segment_questions(include_volatility=True)

    # Builders hand out copies: mutating one batch never leaks into the module constants.
    batch = schemas.evidence_scan_questions()
    batch["guidance_trend"].instructions = "mutated"
    assert schemas.evidence_scan_questions()["guidance_trend"].instructions != "mutated"


def test_validate_questions_is_structural_only() -> None:
    too_many = LayaQuestion(
        type="choice",
        instructions="pick",
        criteria={f"option_{i}": f"description {i}" for i in range(25)},
    )
    with pytest.raises(ValueError, match="options"):
        validate_questions({"q": too_many})
    duplicate = LayaQuestion(
        type="choice", instructions="pick", criteria={"Yes": "affirm", "yes ": "affirm again"}
    )
    with pytest.raises(ValueError, match="distinct"):
        validate_questions({"q": duplicate})
    with pytest.raises(ValueError, match="at least one option"):
        validate_questions({"q": LayaQuestion(type="choice", instructions="pick", criteria={})})
    with pytest.raises(ValueError, match="empty option key"):
        blank = LayaQuestion(type="choice", instructions="p", criteria={" ": "d"})
        validate_questions({"q": blank})
    with pytest.raises(ValueError, match="empty"):
        validate_questions({"q": LayaQuestion(type="noul", instructions="   ")})
    with pytest.raises(ValueError):
        validate_questions({})
    with pytest.raises(ValueError, match="levels"):
        validate_questions({"q": LayaQuestion(type="score", instructions="x", criteria=["one"])})
    with pytest.raises(ValueError, match="levels"):
        twenty = LayaQuestion(type="score", instructions="x", criteria=[str(i) for i in range(20)])
        validate_questions({"q": twenty})
    validate_questions({"ok": LayaQuestion(type="noul", instructions="fine?")})
    # Token lengths are not the structural check's business: these pass here and are caught,
    # measured, by measure_heads.
    long_head = LayaQuestion(
        type="choice",
        instructions="Consider carefully. " * 70,
        criteria={"a": "first option", "b": "second option"},
    )
    long_option = LayaQuestion(
        type="choice", instructions="pick", criteria={"a": "word " * 60, "b": "short"}
    )
    validate_questions({"head": long_head, "option": long_option})


async def test_measure_heads_measures_exactly_as_laya_renders() -> None:
    noul = LayaQuestion(type="noul", instructions="fine?")
    masked = LayaQuestion(
        type="choice", instructions="[MASK] pick", criteria={"a": "x [MASK] y", "b": ""}
    )
    score = LayaQuestion(type="score", instructions="rate", criteria=["low", "mid", "high"])
    heads = await measure_heads({"n": noul, "m": masked, "s": score}, counter)
    # "noul question: fine?" -> 5; the two fixed noul options -> 9 and 7 tokens each
    assert heads["n"] == HeadMeasure(5, (9, 7))
    assert heads["n"].options_total == (1 + 9) + (1 + 7) and heads["n"].total == 23
    # [MASK] markers are replaced by a space before counting, exactly as buildSequence does;
    # an option with an empty description renders as the bare key.
    assert heads["m"] == HeadMeasure(4, (4, 1))
    assert rendered_options(masked) == ["a: x [MASK] y", "b"]
    assert option_texts(masked) == [" a: x   y", " b"]
    assert heads["s"].option_tokens == (4, 4, 4)  # "level 0: low"
    assert rendered_options(score) == ["level 0: low", "level 1: mid", "level 2: high"]
    # Dict inputs are accepted (validated into LayaQuestion), and the counter must answer for
    # every text it was given.
    same = await measure_heads({"n": noul.to_laya()}, counter)
    assert same["n"] == heads["n"]

    async def short(texts: Sequence[str]) -> list[int]:
        return [1]

    with pytest.raises(ValueError, match="wrong number"):
        await measure_heads({"n": noul, "s": score}, short)


async def test_measure_heads_rejects_heads_laya_would_truncate() -> None:
    long_option = LayaQuestion(
        type="choice", instructions="pick", criteria={"a": "word " * 60, "b": "short"}
    )
    with pytest.raises(ValueError, match=r"question 'q': an option exceeds 48 tokens"):
        await measure_heads({"q": long_option}, counter)
    assert tokens(option_texts(long_option)[0]) == 62 > 48

    long_head = LayaQuestion(
        type="choice",
        instructions="Consider carefully. " * 70,
        criteria={"a": "first option", "b": "second option"},
    )
    head = await measure_heads(
        {"fine": LayaQuestion(type="choice", instructions="pick", criteria={"a": "b"})}, counter
    )
    assert head["fine"].truncation() is None
    with pytest.raises(ValueError, match=r"question 'q': instructions of 213 tokens") as info:
        await measure_heads({"q": long_head}, counter)
    assert "left in head_max_len=192" in str(info.value)

    crowded = LayaQuestion(
        type="choice",
        instructions="pick",
        criteria={f"o{i}": "one two three four five six seven" for i in range(19)},
    )
    with pytest.raises(ValueError, match="fewer than 16 left for the instructions"):
        await measure_heads({"q": crowded}, counter)

    # The failing question is named even in a batch of otherwise fine questions.
    with pytest.raises(ValueError, match="question 'bad'"):
        await measure_heads(
            {"ok": LayaQuestion(type="noul", instructions="fine?"), "bad": long_option}, counter
        )


def test_state_budget_follows_the_largest_measured_head() -> None:
    assert state_budget_for({}) == LAYA_MAX_LEN - SEQUENCE_SPECIALS
    small = HeadMeasure(3, (4, 4))
    large = HeadMeasure(100, tuple([20] * 8))
    assert small.total == 13 and large.total == 100 + 8 * 21
    assert state_budget_for({"a": small, "b": large}) == LAYA_MAX_LEN - SEQUENCE_SPECIALS - 268
    assert state_budget_for({"a": small}, max_len=300) == 300 - SEQUENCE_SPECIALS - 13
    huge = HeadMeasure(LAYA_HEAD_MAX_LEN, ())
    assert state_budget_for({"h": huge}, max_len=200) == MIN_STATE_BUDGET_TOKENS


# ---- compaction ---------------------------------------------------------------------------


def _big_state() -> dict:
    return {
        "symbol": "ACME",
        "instrument": {"symbol": "ACME", "name": "Acme Corp"},
        "facts": [
            {"fact": f"Revenue fact {i} " + "x" * 300, "period": f"2025-Q{i % 4 + 1}"}
            for i in range(20)
        ],
        "notes": "n" * 2000,
        "zeta": ["z" * 100] * 30,
        "sources_count": 7,
        "freshness": "current",
    }


def _priority_order(state: dict) -> list[str]:
    ordered = [k for k in DEFAULT_STATE_PRIORITY if k in state]
    return ordered + sorted(k for k in state if k not in DEFAULT_STATE_PRIORITY)


async def test_compaction_respects_measured_budget_and_is_deterministic() -> None:
    state = _big_state()
    compact, dropped = await compact_state(state, 120, counter)
    assert state_tokens(compact) <= 120
    assert dropped, "an oversized state must drop something"
    # Priority keys survive; the alphabetical tail goes first and keys come off the tail.
    assert "symbol" in compact and "instrument" in compact
    assert dropped[0] == "zeta"
    assert list(compact)[:2] == ["instrument", "symbol"]
    assert list(compact) + dropped[::-1] == _priority_order(state)
    assert all(isinstance(v, (dict, str, int, list)) for v in compact.values())

    again, dropped_again = await compact_state(state, 120, counter)
    assert again == compact and dropped_again == dropped
    shuffled = dict(reversed(list(state.items())))
    assert await compact_state(shuffled, 120, counter) == (compact, dropped)

    # A larger budget keeps more (never less) and still measures under it.
    roomier, dropped_roomier = await compact_state(state, 400, counter)
    assert state_tokens(roomier) <= 400
    assert set(roomier) >= set(compact) and len(dropped_roomier) <= len(dropped)
    assert list(roomier) + dropped_roomier[::-1] == _priority_order(state)

    # Shrinking rules without dropping: 240 characters per string, 8 items per list.
    loose, dropped_loose = await compact_state(state, 100_000, counter)
    assert dropped_loose == []
    assert len(loose["notes"]) == MAX_STRING_CHARS + 1 and loose["notes"].endswith("…")
    assert len(loose["facts"]) == MAX_LIST_ITEMS == 8
    assert all(len(f["fact"]) <= MAX_STRING_CHARS + 1 for f in loose["facts"])
    assert len(loose["zeta"]) == 8 and loose["sources_count"] == 7
    assert list(loose) == _priority_order(state)

    # Custom priority is honoured and never raises on odd input.
    custom, custom_dropped = await compact_state(state, 60, counter, priority=["zeta", "symbol"])
    assert next(iter(custom)) == "zeta" and "symbol" in custom
    assert state_tokens(custom) <= 60 and custom_dropped and custom_dropped[-1] == "facts"
    compacted_text, _ = await compact_state("just text " * 500, 50, counter)
    assert set(compacted_text) == {"text"} and compacted_text["text"].endswith("…")
    assert state_tokens(compacted_text) <= 50
    assert await compact_state({}, 10, counter) == ({}, [])
    assert await compact_state([1, 2, 3], 100, counter) == ({"value": [1, 2, 3]}, [])  # type: ignore[arg-type]


async def test_compaction_measures_the_rendered_state_and_propagates_counter_failures() -> None:
    seen: list[list[str]] = []

    async def capture(texts: Sequence[str]) -> list[int]:
        seen.append(list(texts))
        return [tokens(t) for t in texts]

    compact, dropped = await compact_state({"a": "b [MASK] c", "z": 1}, 100, capture)
    assert (compact, dropped) == ({"a": "b [MASK] c", "z": 1}, [])
    # One measurement of the whole assembled state, rendered the way Laya renders JSON and
    # with the [MASK] marker scrubbed the way buildSequence scrubs it.
    assert seen == [['{"a": "b   c", "z": 1}']]

    async def broken(texts: Sequence[str]) -> list[int]:
        raise RuntimeError("tokenizer down")

    with pytest.raises(RuntimeError, match="tokenizer down"):
        await compact_state(_big_state(), 100, broken)

    async def wrong_count(texts: Sequence[str]) -> list[int]:
        return [1, 2]

    with pytest.raises(ValueError, match="wrong number"):
        await compact_state({"a": 1}, 100, wrong_count)


def test_state_digest_stable_across_key_order() -> None:
    a = {"x": 1, "y": {"b": 2, "a": [1, 2]}, "z": "é"}
    b = {"z": "é", "y": {"a": [1, 2], "b": 2}, "x": 1}
    assert state_digest(a) == state_digest(b)
    assert len(state_digest(a)) == 64
    assert canonical_json(a) == canonical_json(b)
    assert state_digest({"x": 1}) != state_digest({"x": 2})
    assert laya_json("plain") == "plain" and laya_json({"k": "v"}) == '{"k": "v"}'


# ---- RuleLaya -----------------------------------------------------------------------------


async def test_rule_laya_answers_all_questions_with_valid_probabilities() -> None:
    laya = RuleLaya()
    info = await laya.load()
    # A double loads nothing: it reports no package, no bundle and no resident memory.
    assert info.package_version is None and info.model_dir is None
    assert info.resident_rss_mb is None and info.load_ms >= 0
    batches = [
        schemas.research_plan_questions(),
        schemas.evidence_scan_questions(),
        schemas.history_segment_questions(),
        schemas.calculation_questions(),
        schemas.horizon_questions(["multi_horizon"]),
        schemas.synthesis_gate_questions(),
    ]
    state = {
        "symbol": "ACME",
        "revenue_growth_yoy": 0.18,
        "operating_margin_change_bp": 120,
        "price_return_1m": 0.05,
        "volatility_30d_annualized": 0.55,
        "pe_5y_percentile": 92,
        "evidence_gaps": ["guidance", "news"],
        "sources_count": 7,
        "freshness": "current",
        "guidance_hint": "raised",
    }
    for batch in batches:
        result = await laya.system_one(state, batch)
        assert set(result.answers) == set(batch)
        # usage is the sequence length Laya would build per question, measured by the same
        # tokenizer the test uses, never a characters-per-token guess
        expected = sum(
            min(LAYA_MAX_LEN, SEQUENCE_SPECIALS + head_tokens(q) + state_tokens(state))
            for q in batch.values()
        )
        assert result.usage.input_tokens == expected > 0
        assert result.latency_ms is not None and result.latency_ms >= 0
        for key, question in batch.items():
            answer = result.answers[key]
            if question.type == "choice":
                assert isinstance(answer, ChoiceAnswer)
                assert set(answer.probabilities) == set(question.criteria)
                assert sum(answer.probabilities.values()) == pytest.approx(1.0)
                assert answer.choice == max(answer.probabilities, key=answer.probabilities.get)
            elif question.type == "score":
                assert isinstance(answer, ScoreAnswer)
                assert 0.0 <= answer.score <= len(question.criteria) - 1
                assert answer.distribution and len(answer.distribution) == len(question.criteria)
                assert sum(answer.distribution) == pytest.approx(1.0)
            else:
                assert isinstance(answer, NoulAnswer) and 0.0 <= answer.noul <= 1.0
    assert len(laya.calls) == len(batches)
    assert laya.calls[0].state is state
    assert laya.calls[0].result is not None
    health = await laya.health()
    assert health.ok and health.loaded and health.pid is None and health.detail is None
    assert laya.stats == {
        "load_ms": None,
        "resident_rss_mb": None,
        "peak_rss_mb": None,
        "restarts": 0,
        "requests": len(batches),
        "warm_inference_ms": None,
    }
    await laya.close()
    assert not (await laya.health()).loaded


async def test_rule_laya_counts_tokens_deterministically() -> None:
    laya = RuleLaya()
    texts = ["", "a b, c", "Hello, world!", "x" * 100, "one\ntwo  three", "état!"]
    counts = await laya.count_tokens(texts)
    assert counts == [tokens(t) for t in texts] == [0, 4, 4, 1, 3, 2]
    assert await laya.count_tokens([]) == []
    assert laya.token_calls == [texts, []]
    assert await laya.count_tokens(texts) == counts  # same text, same count
    # Big states are capped exactly at max_len per question, as Laya truncates.
    huge = {"text": "word " * 2000}
    question = schemas.research_plan_questions()
    result = await laya.system_one(huge, question)
    assert result.usage.input_tokens == LAYA_MAX_LEN * len(question)
    assert laya.stats["requests"] == 1


async def test_rule_laya_rules_follow_state() -> None:
    laya = RuleLaya()
    plan = schemas.research_plan_questions()

    r = await laya.system_one({"evidence_gaps": ["guidance"], "sources_count": 2}, plan)
    assert r.answers["research_intent"].choice == "retrieve_guidance_history"
    assert r.answers["evidence_sufficient"].noul == 0.3
    r = await laya.system_one({"evidence_gaps": [], "sources_count": 6}, plan)
    assert r.answers["research_intent"].choice == "stop_research"
    assert r.answers["evidence_sufficient"].noul == 0.8
    r = await laya.system_one({"evidence_gaps": ["retrieve_price_history"]}, plan)
    assert r.answers["research_intent"].choice == "retrieve_price_history"
    r = await laya.system_one({"freshness": "stale"}, plan)
    assert r.answers["stale_evidence_matters"].noul == 0.7

    scan = schemas.evidence_scan_questions()
    r = await laya.system_one(
        {"guidance_hint": "lowered", "source_type": "regulatory_filing"}, scan
    )
    assert r.answers["guidance_trend"].choice == "deteriorating"
    assert r.answers["source_is_material"].noul == 0.75
    assert r.answers["evidence_stance"].choice == "bearish"
    r = await laya.system_one({}, scan)
    assert r.answers["guidance_trend"].choice == "unchanged"
    assert r.answers["evidence_stance"].choice == "neutral"
    r = await laya.system_one({"revenue_growth_yoy": 0.2, "price_return_1m": -0.2}, scan)
    assert r.answers["evidence_stance"].choice == "mixed"

    calc = schemas.calculation_questions()
    r = await laya.system_one({"pe_5y_percentile": 100}, calc)
    assert r.answers["valuation_extremeness"].score > 3.0
    assert r.answers["calculation_pack"].choice == "valuation_vs_history"
    r = await laya.system_one(
        {"pe_5y_percentile": 0, "price_return_1m": 0.1, "benchmark_return_1m": 0.01}, calc
    )
    assert r.answers["valuation_extremeness"].score < 1.0
    assert r.answers["benchmark_relative"].choice == "outperforming"
    assert r.answers["calculation_pack"].choice == "all_standard"

    history = schemas.history_segment_questions(include_volatility=True)
    r = await laya.system_one(
        {"volatility_30d_annualized": 0.1, "operating_margin_change_bp": -300}, history
    )
    assert r.answers["volatility_regime"].choice == "low"
    assert r.answers["margin_direction"].choice == "contracting"
    assert r.answers["material_change"].noul == 0.75

    forced = RuleLaya(
        force={"research_intent": "retrieve_recent_news", "evidence_sufficient": 0.99}
    )
    r = await forced.system_one({"evidence_gaps": ["guidance"]}, plan)
    assert r.answers["research_intent"].choice == "retrieve_recent_news"
    assert r.answers["evidence_sufficient"].noul == 0.99

    failing = RuleLaya(raise_error=AnalysisError(ErrorCode.LAYA_INFERENCE_FAILED))
    with pytest.raises(AnalysisError):
        await failing.system_one({}, plan)
    assert len(failing.calls) == 1


# ---- wrapper ------------------------------------------------------------------------------


def _question_set(**overrides) -> LayaQuestionSet:
    base = {
        "stage": schemas.STAGE_EVIDENCE_SCAN,
        "state": {
            "symbol": "ACME",
            "revenue_growth_yoy": 0.12,
            "guidance_hint": "raised",
            "source_type": "earnings_release",
        },
        "questions": {
            **schemas.evidence_scan_questions(),
            "revenue_momentum": schemas.ALL_QUESTIONS["revenue_momentum"].model_copy(deep=True),
        },
        "segment_id": "src_1",
    }
    base.update(overrides)
    return LayaQuestionSet(**base)


async def test_wrapper_builds_decisions_and_marks_timer() -> None:
    laya = RuleLaya(latency_ms=1.5)
    wrapper = LayaFinanceWrapper(laya)
    ctx = AnalysisContext(analysis_id="an_test")
    question_set = _question_set()

    decisions = await wrapper.ask(question_set, ctx)
    assert [d.decision_type for d in decisions] == list(question_set.questions)
    compacted_state = laya.calls[0].state
    expected_digest = state_digest(compacted_state)
    by_type = {d.decision_type: d for d in decisions}
    for decision in decisions:
        assert decision.decision_id.startswith("dec_")
        assert decision.stage == schemas.STAGE_EVIDENCE_SCAN
        assert decision.segment_id == "src_1"
        assert decision.schema_version == schemas.LAYA_SCHEMA_VERSION
        assert decision.state_digest == expected_digest
        assert decision.state_tokens == laya.calls[0].result.usage.input_tokens
        assert decision.created_at.tzinfo is not None
        assert decision.confidence == answer_confidence(decision.answer)
        assert decision.question == question_set.questions[decision.decision_type]
        view = decision.event_view()
        assert view["decision_type"] == decision.decision_type

    choice = by_type["guidance_trend"]
    assert isinstance(choice.answer, ChoiceAnswer)
    assert choice.decision == "improving"
    assert choice.confidence == pytest.approx(max(choice.answer.probabilities.values()))

    noul = by_type["source_is_material"]
    assert isinstance(noul.answer, NoulAnswer)
    assert noul.confidence == pytest.approx(max(noul.answer.noul, 1 - noul.answer.noul))

    score = by_type["revenue_momentum"]
    assert isinstance(score.answer, ScoreAnswer)
    assert score.confidence == pytest.approx(max(score.answer.distribution))

    assert ctx.timers.elapsed_ms["laya"] > 0
    assert "laya_dropped_keys" not in ctx.diagnostics
    assert len(decisions) == len(set(d.decision_id for d in decisions))
    # decision ids are stable: the same compacted state and question give the same id
    again = await wrapper.ask(question_set, AnalysisContext(analysis_id="an_again"))
    assert [d.decision_id for d in again] == [d.decision_id for d in decisions]

    # Custom schema version flows through; ask_many is sequential and flat.
    versioned = LayaFinanceWrapper(laya, schema_version="finance-test")
    more = await versioned.ask_many([question_set, _question_set(segment_id="src_2")], ctx)
    assert len(more) == 2 * len(question_set.questions)
    assert {d.segment_id for d in more} == {"src_1", "src_2"}
    assert all(d.schema_version == "finance-test" for d in more)
    assert len(laya.calls) == 4
    assert LayaFinanceWrapper.as_dict(decisions)["guidance_trend"] == "improving"


async def test_wrapper_measures_heads_once_per_batch_and_compacts_under_that_budget() -> None:
    laya = RuleLaya()
    wrapper = LayaFinanceWrapper(laya)
    ctx = AnalysisContext(analysis_id="an_big")
    # Large in *tokens*, not merely in bytes: the shrink rules alone leave this over budget.
    state = {
        "symbol": "ACME",
        "sources_count": 9,
        "notes": "one word after another " * 400,
        "zzz_dump": ["y " * 150] * 40,
        "yyy_dump": "x y " * 1000,
    }
    questions = schemas.research_plan_questions()
    question_set = _question_set(
        stage=schemas.STAGE_RESEARCH_PLAN, state=state, questions=questions, segment_id=None
    )
    decisions = await wrapper.ask(question_set, ctx)
    sent = laya.calls[0].state
    heads = await measure_heads(questions, counter)
    assert state_tokens(sent) <= state_budget_for(heads)
    assert "symbol" in sent and sent["sources_count"] == 9
    entries = ctx.diagnostics["laya_dropped_keys"]
    assert entries[0]["stage"] == schemas.STAGE_RESEARCH_PLAN
    assert entries[0]["keys"] == ["zzz_dump"]  # the alphabetical tail goes first
    assert "zzz_dump" not in sent and len(sent["notes"]) == 241
    assert all(d.state_digest == state_digest(sent) for d in decisions)
    assert {d.decision_type for d in decisions} == set(question_set.questions)
    assert decisions[0].segment_id is None
    # The head measurement went through the client's tokenizer (one batched call holding every
    # instruction and option text) and is cached: a second ask of the same batch only measures
    # the state again.
    head_text = instruction_text(questions["research_intent"])
    head_calls = [c for c in laya.token_calls if head_text in c]
    assert len(head_calls) == 1
    assert len(head_calls[0]) == sum(1 + len(option_texts(q)) for q in questions.values())
    await wrapper.ask(question_set, AnalysisContext(analysis_id="an_two"))
    assert len([c for c in laya.token_calls if head_text in c]) == 1
    assert len(laya.token_calls) > len(head_calls) + 1

    # An explicit budget overrides the measured one.
    tight = LayaFinanceWrapper(RuleLaya(), state_budget_tokens=80)
    await tight.ask(question_set, AnalysisContext(analysis_id="an_tight"))
    assert state_tokens(tight.client.calls[0].state) <= 80


async def test_wrapper_error_paths() -> None:
    ctx = AnalysisContext(analysis_id="an_err")
    bad = _question_set(
        questions={
            "q": LayaQuestion(
                type="choice", instructions="x", criteria={f"o{i}": "d" for i in range(30)}
            )
        }
    )
    with pytest.raises(AnalysisError) as info:
        await LayaFinanceWrapper(RuleLaya()).ask(bad, ctx)
    assert info.value.code is ErrorCode.LAYA_INFERENCE_FAILED
    assert info.value.details["reason"] == "invalid_questions"

    # A head Laya would silently truncate is refused before any state is sent.
    laya = RuleLaya()
    truncating = _question_set(
        questions={"q": LayaQuestion(type="choice", instructions="x", criteria={"a": "word " * 60})}
    )
    with pytest.raises(AnalysisError) as info:
        await LayaFinanceWrapper(laya).ask(truncating, ctx)
    assert info.value.code is ErrorCode.LAYA_INFERENCE_FAILED
    assert info.value.details["reason"] == "invalid_questions"
    assert "exceeds 48 tokens" in info.value.details["message"]
    assert laya.calls == []

    with pytest.raises(AnalysisError) as info:
        await LayaFinanceWrapper(RuleLaya(raise_error=RuntimeError("boom"))).ask(
            _question_set(), ctx
        )
    assert info.value.code is ErrorCode.LAYA_INFERENCE_FAILED
    assert info.value.details == {"reason": "client_failure", "exception": "RuntimeError"}
    assert ctx.timers.elapsed_ms["laya"] >= 0

    class BrokenTokenizer(RuleLaya):
        async def count_tokens(self, texts: Sequence[str]) -> list[int]:
            raise OSError("worker gone")

    with pytest.raises(AnalysisError) as info:
        await LayaFinanceWrapper(BrokenTokenizer()).ask(_question_set(), ctx)
    assert info.value.details == {"reason": "client_failure", "exception": "OSError"}

    cancelled = AnalysisContext(analysis_id="an_cancel")
    cancelled.cancel.cancel()
    laya = RuleLaya()
    with pytest.raises(AnalysisError) as info:
        await LayaFinanceWrapper(laya).ask(_question_set(), cancelled)
    assert info.value.code is ErrorCode.CANCELLED
    assert laya.calls == [] and laya.token_calls == []
