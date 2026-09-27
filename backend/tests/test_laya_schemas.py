"""Finance question schemas, compaction, MockLaya and LayaFinanceWrapper."""

from __future__ import annotations

import pytest

from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.laya import schemas
from bayanalytics.laya.base import LAYA_HEAD_MAX_LEN, LAYA_MAX_LEN
from bayanalytics.laya.compaction import (
    DEFAULT_STATE_BUDGET_TOKENS,
    HEADER_ALLOWANCE_TOKENS,
    canonical_json,
    compact_state,
    estimate_head_tokens,
    estimate_tokens,
    state_budget_for,
    state_digest,
    validate_questions,
)
from bayanalytics.laya.mock import MockLaya
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

# ---- schemas ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [
        schemas.research_plan_questions,
        schemas.evidence_scan_questions,
        schemas.history_segment_questions,
        schemas.calculation_questions,
        lambda: schemas.horizon_questions(SINGLE_HORIZONS),
        lambda: schemas.horizon_questions(["multi_horizon"]),
        schemas.synthesis_gate_questions,
    ],
)
def test_every_builder_validates(builder) -> None:
    batch = builder()
    assert batch
    validate_questions(batch)
    for key, question in batch.items():
        assert isinstance(question, LayaQuestion)
        assert estimate_head_tokens(question) <= LAYA_HEAD_MAX_LEN, key
        assert question.to_laya()["type"] == question.type
    assert LAYA_MAX_LEN - HEADER_ALLOWANCE_TOKENS == DEFAULT_STATE_BUDGET_TOKENS
    assert 64 <= state_budget_for(batch) <= DEFAULT_STATE_BUDGET_TOKENS


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

    # Builders hand out copies: mutating one batch never leaks into the module constants.
    batch = schemas.evidence_scan_questions()
    batch["guidance_trend"].instructions = "mutated"
    assert schemas.evidence_scan_questions()["guidance_trend"].instructions != "mutated"


def test_head_length_guard_rejects_bad_choices() -> None:
    too_many = LayaQuestion(
        type="choice",
        instructions="pick",
        criteria={f"option_{i}": f"description {i}" for i in range(25)},
    )
    with pytest.raises(ValueError, match="options"):
        validate_questions({"q": too_many})

    long_head = LayaQuestion(
        type="choice",
        instructions="Consider carefully. " * 40,
        criteria={"a": "first option", "b": "second option"},
    )
    with pytest.raises(ValueError, match="head"):
        validate_questions({"q": long_head})

    long_option = LayaQuestion(
        type="choice",
        instructions="pick",
        criteria={"a": "word " * 60, "b": "short"},
    )
    with pytest.raises(ValueError, match="exceeds"):
        validate_questions({"q": long_option})

    duplicate = LayaQuestion(
        type="choice", instructions="pick", criteria={"Yes": "affirm", "yes ": "affirm again"}
    )
    with pytest.raises(ValueError, match="distinct"):
        validate_questions({"q": duplicate})

    with pytest.raises(ValueError):
        validate_questions({})
    with pytest.raises(ValueError, match="levels"):
        validate_questions({"q": LayaQuestion(type="score", instructions="x", criteria=["one"])})
    validate_questions({"ok": LayaQuestion(type="noul", instructions="fine?")})


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


def test_compaction_respects_budget_and_is_deterministic() -> None:
    state = _big_state()
    compact, dropped = compact_state(state, budget_tokens=200)
    assert estimate_tokens(compact) <= 200
    assert dropped, "an oversized state must drop something"
    # Priority keys survive; the alphabetical tail goes first.
    assert "symbol" in compact and "instrument" in compact
    assert dropped[0] == "zeta"
    assert list(compact) == [k for k in compact]  # insertion order is the priority order
    assert list(compact)[:2] == ["instrument", "symbol"]

    again, dropped_again = compact_state(state, budget_tokens=200)
    assert again == compact and dropped_again == dropped
    shuffled = dict(reversed(list(state.items())))
    assert compact_state(shuffled, budget_tokens=200) == (compact, dropped)

    # Shrinking rules without dropping.
    loose, dropped_loose = compact_state(state, budget_tokens=100_000)
    assert dropped_loose == []
    assert len(loose["notes"]) == 241 and loose["notes"].endswith("…")
    assert len(loose["facts"]) == 8
    assert all(len(f["fact"]) <= 241 for f in loose["facts"])
    assert loose["sources_count"] == 7

    # Custom priority is honoured and never raises on odd input.
    custom, custom_dropped = compact_state(state, budget_tokens=60, priority=["zeta", "symbol"])
    assert next(iter(custom)) == "zeta" or "zeta" not in custom
    assert "symbol" in custom or custom_dropped
    compacted_text, _ = compact_state("just text " * 500, budget_tokens=50)
    assert estimate_tokens(compacted_text) <= 200
    assert compact_state({}, budget_tokens=10) == ({}, [])


def test_state_digest_stable_across_key_order() -> None:
    a = {"x": 1, "y": {"b": 2, "a": [1, 2]}, "z": "é"}
    b = {"z": "é", "y": {"a": [1, 2], "b": 2}, "x": 1}
    assert state_digest(a) == state_digest(b)
    assert len(state_digest(a)) == 64
    assert canonical_json(a) == canonical_json(b)
    assert state_digest({"x": 1}) != state_digest({"x": 2})
    assert estimate_tokens("") == 0 and estimate_tokens("abcd") == 2
    assert estimate_tokens({"k": "v"}) > estimate_tokens('{"k": "v"}')


# ---- MockLaya -----------------------------------------------------------------------------


async def test_mock_laya_answers_all_questions_with_valid_probabilities() -> None:
    mock = MockLaya()
    await mock.load()
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
        result = await mock.system_one(state, batch)
        assert set(result.answers) == set(batch)
        assert result.usage.input_tokens and result.usage.input_tokens > 0
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
    assert len(mock.calls) == len(batches)
    assert mock.calls[0].state is state
    assert mock.calls[0].result is not None
    assert (await mock.health()).ok and (await mock.health()).loaded
    assert mock.stats["requests"] == len(batches)
    await mock.close()
    assert not (await mock.health()).loaded


async def test_mock_laya_rules_follow_state() -> None:
    mock = MockLaya()
    plan = schemas.research_plan_questions()

    r = await mock.system_one({"evidence_gaps": ["guidance"], "sources_count": 2}, plan)
    assert r.answers["research_intent"].choice == "retrieve_guidance_history"
    assert r.answers["evidence_sufficient"].noul == 0.3
    r = await mock.system_one({"evidence_gaps": [], "sources_count": 6}, plan)
    assert r.answers["research_intent"].choice == "stop_research"
    assert r.answers["evidence_sufficient"].noul == 0.8
    r = await mock.system_one({"evidence_gaps": ["retrieve_price_history"]}, plan)
    assert r.answers["research_intent"].choice == "retrieve_price_history"
    r = await mock.system_one({"freshness": "stale"}, plan)
    assert r.answers["stale_evidence_matters"].noul == 0.7

    scan = schemas.evidence_scan_questions()
    r = await mock.system_one(
        {"guidance_hint": "lowered", "source_type": "regulatory_filing"}, scan
    )
    assert r.answers["guidance_trend"].choice == "deteriorating"
    assert r.answers["source_is_material"].noul == 0.75
    assert r.answers["evidence_stance"].choice == "bearish"
    r = await mock.system_one({}, scan)
    assert r.answers["guidance_trend"].choice == "unchanged"
    assert r.answers["evidence_stance"].choice == "neutral"
    r = await mock.system_one({"revenue_growth_yoy": 0.2, "price_return_1m": -0.2}, scan)
    assert r.answers["evidence_stance"].choice == "mixed"

    calc = schemas.calculation_questions()
    r = await mock.system_one({"pe_5y_percentile": 100}, calc)
    assert r.answers["valuation_extremeness"].score > 3.0
    assert r.answers["calculation_pack"].choice == "valuation_vs_history"
    r = await mock.system_one(
        {"pe_5y_percentile": 0, "price_return_1m": 0.1, "benchmark_return_1m": 0.01}, calc
    )
    assert r.answers["valuation_extremeness"].score < 1.0
    assert r.answers["benchmark_relative"].choice == "outperforming"
    assert r.answers["calculation_pack"].choice == "all_standard"

    history = schemas.history_segment_questions()
    r = await mock.system_one(
        {"volatility_30d_annualized": 0.1, "operating_margin_change_bp": -300}, history
    )
    assert r.answers["volatility_regime"].choice == "low"
    assert r.answers["margin_direction"].choice == "contracting"
    assert r.answers["material_change"].noul == 0.75

    forced = MockLaya(
        force={"research_intent": "retrieve_recent_news", "evidence_sufficient": 0.99}
    )
    r = await forced.system_one({"evidence_gaps": ["guidance"]}, plan)
    assert r.answers["research_intent"].choice == "retrieve_recent_news"
    assert r.answers["evidence_sufficient"].noul == 0.99

    failing = MockLaya(raise_error=AnalysisError(ErrorCode.LAYA_INFERENCE_FAILED))
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
    mock = MockLaya(latency_ms=1.5)
    wrapper = LayaFinanceWrapper(mock)
    ctx = AnalysisContext(analysis_id="an_test")
    question_set = _question_set()

    decisions = await wrapper.ask(question_set, ctx)
    assert [d.decision_type for d in decisions] == list(question_set.questions)
    compacted_state = mock.calls[0].state
    expected_digest = state_digest(compacted_state)
    by_type = {d.decision_type: d for d in decisions}
    for decision in decisions:
        assert decision.decision_id.startswith("dec_")
        assert decision.stage == schemas.STAGE_EVIDENCE_SCAN
        assert decision.segment_id == "src_1"
        assert decision.schema_version == schemas.LAYA_SCHEMA_VERSION
        assert decision.state_digest == expected_digest
        assert decision.state_tokens == mock.calls[0].result.usage.input_tokens
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

    # Custom schema version flows through; ask_many is sequential and flat.
    versioned = LayaFinanceWrapper(mock, schema_version="finance-test")
    more = await versioned.ask_many([question_set, _question_set(segment_id="src_2")], ctx)
    assert len(more) == 2 * len(question_set.questions)
    assert {d.segment_id for d in more} == {"src_1", "src_2"}
    assert all(d.schema_version == "finance-test" for d in more)
    assert len(mock.calls) == 3
    assert LayaFinanceWrapper.as_dict(decisions)["guidance_trend"] == "improving"


async def test_wrapper_compacts_state_and_records_dropped_keys() -> None:
    mock = MockLaya()
    wrapper = LayaFinanceWrapper(mock)
    ctx = AnalysisContext(analysis_id="an_big")
    state = {
        "symbol": "ACME",
        "sources_count": 9,
        "notes": "n" * 5000,
        "zzz_dump": ["y" * 200] * 40,
        "yyy_dump": "x" * 3000,
    }
    question_set = _question_set(
        stage=schemas.STAGE_RESEARCH_PLAN,
        state=state,
        questions=schemas.research_plan_questions(),
        segment_id=None,
    )
    decisions = await wrapper.ask(question_set, ctx)
    sent = mock.calls[0].state
    assert estimate_tokens(sent) <= state_budget_for(question_set.questions)
    assert "symbol" in sent and sent["sources_count"] == 9
    entries = ctx.diagnostics["laya_dropped_keys"]
    assert entries[0]["stage"] == schemas.STAGE_RESEARCH_PLAN
    assert "zzz_dump" in entries[0]["keys"]
    assert all(d.state_digest == state_digest(sent) for d in decisions)
    assert {d.decision_type for d in decisions} == set(question_set.questions)
    assert decisions[0].segment_id is None


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
        await LayaFinanceWrapper(MockLaya()).ask(bad, ctx)
    assert info.value.code is ErrorCode.LAYA_INFERENCE_FAILED
    assert info.value.details["reason"] == "invalid_questions"

    with pytest.raises(AnalysisError) as info:
        await LayaFinanceWrapper(MockLaya(raise_error=RuntimeError("boom"))).ask(
            _question_set(), ctx
        )
    assert info.value.code is ErrorCode.LAYA_INFERENCE_FAILED
    assert info.value.details == {"reason": "client_failure", "exception": "RuntimeError"}
    assert ctx.timers.elapsed_ms["laya"] >= 0

    cancelled = AnalysisContext(analysis_id="an_cancel")
    cancelled.cancel.cancel()
    mock = MockLaya()
    with pytest.raises(AnalysisError) as info:
        await LayaFinanceWrapper(mock).ask(_question_set(), cancelled)
    assert info.value.code is ErrorCode.CANCELLED
    assert mock.calls == []
