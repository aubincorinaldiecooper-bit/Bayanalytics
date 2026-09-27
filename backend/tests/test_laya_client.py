"""LayaWorkerClient against the real worker.mjs with the test-only stub Laya module."""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

import bayanalytics.laya as laya_pkg
from bayanalytics.config import Settings
from bayanalytics.errors import AnalysisError
from bayanalytics.laya.client import LayaWorkerClient, parse_answer
from bayanalytics.procenv import child_env
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.decisions import ChoiceAnswer, LayaQuestion, NoulAnswer, ScoreAnswer

WORKER_DIR = Path(laya_pkg.__file__).resolve().parent / "worker"
STUB = Path(__file__).resolve().parent / "doubles" / "stub_laya.mjs"
SECRET = "SECRET_STATE_VALUE_7f3a"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "laya_worker_dir": WORKER_DIR,
        "laya_load_timeout_s": 15.0,
        "laya_request_timeout_s": 5.0,
        "laya_max_restarts": 1,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def make_client(**overrides: object) -> LayaWorkerClient:
    return LayaWorkerClient(make_settings(**overrides), env={"LAYA_MODULE": str(STUB)})


def questions(marker: str = "") -> dict[str, LayaQuestion]:
    return {
        "pick": LayaQuestion(
            type="choice",
            instructions=f"pick one {marker}",
            criteria={"alpha": "first", "beta": "second", "gamma": "third"},
        ),
        "level": LayaQuestion(type="score", instructions="level", criteria=["low", "mid", "high"]),
        "flag": LayaQuestion(type="noul", instructions="is it?"),
    }


@pytest.fixture
async def client() -> AsyncIterator[LayaWorkerClient]:
    c = make_client()
    try:
        yield c
    finally:
        await c.close()


async def test_load_health_system_one_round_trip(client: LayaWorkerClient) -> None:
    info = await client.load()
    assert info.load_ms >= 0
    assert info.max_len == 512 and info.head_max_len == 192
    assert info.resident_rss_mb is not None and info.resident_rss_mb > 0
    # The module was injected by path: no package version is measured, so none is reported.
    assert info.package_version is None
    assert client.stats["load_ms"] == info.load_ms

    again = await client.load()  # idempotent
    assert again == info

    health = await client.health()
    assert health.ok and health.loaded
    assert health.pid == client.pid and health.restarts == 0

    result = await client.system_one({"symbol": "ACME", "note": "x"}, questions())
    pick = result.answers["pick"]
    assert isinstance(pick, ChoiceAnswer)
    assert pick.choice == "alpha"
    assert set(pick.probabilities) == {"alpha", "beta", "gamma"}
    assert sum(pick.probabilities.values()) == pytest.approx(1.0)
    level = result.answers["level"]
    assert isinstance(level, ScoreAnswer)
    assert level.score == 1.0
    assert level.distribution == pytest.approx([1 / 3] * 3, abs=1e-3)
    flag = result.answers["flag"]
    assert isinstance(flag, NoulAnswer) and flag.noul == 0.5
    assert result.usage.input_tokens and result.usage.input_tokens > 0
    assert result.latency_ms is not None and result.latency_ms >= 0

    stats = client.stats
    assert stats["requests"] == 1
    assert stats["restarts"] == 0
    assert stats["warm_inference_ms"] == result.latency_ms
    assert stats["peak_rss_mb"] is not None and stats["peak_rss_mb"] >= stats["resident_rss_mb"]


async def test_system_one_auto_loads(client: LayaWorkerClient) -> None:
    result = await client.system_one("plain text state", questions())
    assert isinstance(result.answers["flag"], NoulAnswer)
    assert client.load_info is not None
    assert (await client.health()).loaded


# --- count_tokens: measured by the worker with the loaded module's tokenizer ---------------


async def _worker_round_trip(requests: list[dict]) -> list[dict]:
    """Drive ``worker.mjs`` over its NDJSON protocol directly (no Python client)."""
    proc = await asyncio.create_subprocess_exec(
        "node",
        "worker.mjs",
        cwd=str(WORKER_DIR),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=child_env({"LAYA_MODULE": str(STUB)}),
    )
    assert proc.stdin is not None and proc.stdout is not None
    responses: list[dict] = []
    try:
        for request in requests:
            proc.stdin.write((json.dumps(request) + "\n").encode())
            await proc.stdin.drain()
            line = await asyncio.wait_for(proc.stdout.readline(), 15.0)
            responses.append(json.loads(line))
    finally:
        proc.stdin.close()
        try:
            await asyncio.wait_for(proc.wait(), 5.0)
        except TimeoutError:
            proc.kill()
            await proc.wait()
    return responses


async def test_worker_count_tokens_op_answers_through_the_stub_tokenizer() -> None:
    before_load, loaded, counted, empty, bad = await _worker_round_trip(
        [
            {"id": "1", "op": "count_tokens", "params": {"texts": ["a b, c"]}},
            {"id": "2", "op": "load", "params": {}},
            {"id": "3", "op": "count_tokens", "params": {"texts": ["a b, c"]}},
            {"id": "4", "op": "count_tokens", "params": {"texts": []}},
            {"id": "5", "op": "count_tokens", "params": {"texts": ["ok", 7]}},
        ]
    )
    # The worker itself never counts without a loaded module: that is the client's job.
    assert before_load["id"] == "1" and before_load["ok"] is False
    assert before_load["error"]["code"] == "NOT_LOADED"
    assert loaded["ok"] is True and loaded["result"]["loaded"] is True
    assert loaded["result"]["package_version"] is None  # injected by path, not the package
    assert counted == {"id": "3", "ok": True, "result": {"counts": [4]}}
    assert empty["result"] == {"counts": []}
    assert bad["ok"] is False and bad["error"]["code"] == "BAD_REQUEST"


async def test_client_count_tokens_returns_the_worker_counts(client: LayaWorkerClient) -> None:
    # Counting before load triggers the load, exactly like system_one does.
    assert client.load_info is None
    assert await client.count_tokens(["a b, c"]) == [4]
    assert client.load_info is not None and (await client.health()).loaded
    assert await client.count_tokens(["", "one", "x-y", "Hello, world!", "a  b"]) == [
        0,
        1,
        3,
        4,
        2,
    ]
    assert await client.count_tokens([]) == []  # no round trip for nothing
    assert client.stats["requests"] == 0  # count_tokens is not a system_one request
    assert client.restarts == 0


async def test_request_timeout_maps_to_laya_inference_failed() -> None:
    client = make_client(laya_request_timeout_s=0.3)
    try:
        await client.load()
        with pytest.raises(AnalysisError) as info:
            await client.system_one({"secret": SECRET}, questions("__slow__"))
        err = info.value
        assert err.code is ErrorCode.LAYA_INFERENCE_FAILED
        assert err.details["worker_code"] == "TIMEOUT"
        assert SECRET not in str(err) and SECRET not in str(err.details)
        assert client.restarts == 0
    finally:
        await client.close()


async def test_worker_error_maps_with_worker_code(client: LayaWorkerClient) -> None:
    await client.load()
    with pytest.raises(AnalysisError) as info:
        await client.system_one({"secret": SECRET}, questions("__throw__"))
    err = info.value
    assert err.code is ErrorCode.LAYA_INFERENCE_FAILED
    assert err.details["worker_code"] == "LAYA_ERROR"
    assert err.details["reason"] == "library_error"
    assert "message" not in err.details  # the worker's free text stays in the server log
    assert SECRET not in str(err.details)
    # The worker survives a thrown error: no restart, next call works.
    result = await client.system_one({"a": 1}, questions())
    assert isinstance(result.answers["pick"], ChoiceAnswer)
    assert client.restarts == 0


async def test_crash_restarts_once_then_fails_fast(client: LayaWorkerClient) -> None:
    await client.load()
    first_pid = client.pid

    with pytest.raises(AnalysisError) as info:
        await client.system_one({"secret": SECRET}, questions("__crash__"))
    assert info.value.details["worker_code"] == "WORKER_EXITED"
    assert info.value.details.get("exit_code") == 3
    assert SECRET not in str(info.value.details)
    health = await client.health()
    assert not health.ok and not health.loaded

    # The next call performs exactly one controlled restart (respawn + reload) and succeeds.
    result = await client.system_one({"a": 1}, questions())
    assert result.answers["pick"].choice == "alpha"  # type: ignore[union-attr]
    assert client.restarts == 1 and client.stats["restarts"] == 1
    assert client.pid != first_pid
    health = await client.health()
    assert health.ok and health.loaded and health.restarts == 1

    # A second crash exhausts the budget: further calls fail fast without spawning anything.
    with pytest.raises(AnalysisError) as info:
        await client.system_one({"a": 1}, questions("__crash__"))
    assert info.value.details["worker_code"] == "WORKER_EXITED"
    dead_pid = client.pid
    with pytest.raises(AnalysisError) as info:
        await client.system_one({"a": 1}, questions())
    assert info.value.code is ErrorCode.LAYA_INFERENCE_FAILED
    assert info.value.details["restarts_exhausted"] is True
    assert info.value.retryable is False
    with pytest.raises(AnalysisError) as info:
        await client.load()
    assert info.value.details["restarts_exhausted"] is True
    assert client.pid == dead_pid and client.restarts == 1
    health = await client.health()
    assert not health.ok and "exhausted" in (health.detail or "")


async def test_close_is_idempotent() -> None:
    client = make_client()
    await client.load()
    await client.close()
    await client.close()
    health = await client.health()
    assert not health.ok and health.detail == "closed"
    with pytest.raises(AnalysisError) as info:
        await client.system_one({"a": 1}, questions())
    assert info.value.details["worker_code"] == "CLOSED"
    await client.close()


async def test_close_without_start_and_missing_node() -> None:
    never_started = make_client()
    await never_started.close()
    health = await never_started.health()
    assert not health.ok

    client = make_client(laya_node_bin="/nonexistent/node-binary")
    try:
        with pytest.raises(AnalysisError) as info:
            await client.load()
        assert info.value.code is ErrorCode.LAYA_INFERENCE_FAILED
        assert info.value.details["worker_code"] == "SPAWN_FAILED"
        assert not (await client.health()).ok
    finally:
        await client.close()


async def test_concurrent_calls_are_serialised(client: LayaWorkerClient) -> None:
    await client.load()

    async def one(i: int) -> tuple[int, str, str]:
        qs = {
            f"pick{i}": LayaQuestion(
                type="choice",
                instructions=f"call {i}",
                criteria={f"opt{i}_a": "a", f"opt{i}_b": "b"},
            ),
            "flag": LayaQuestion(type="noul", instructions=f"flag {i}"),
        }
        result = await client.system_one({"call": i}, qs)
        answer = result.answers[f"pick{i}"]
        assert isinstance(answer, ChoiceAnswer)
        assert set(result.answers) == {f"pick{i}", "flag"}
        return i, answer.choice, next(iter(answer.probabilities))

    results = await asyncio.gather(*(one(i) for i in range(5)))
    for i, choice, first_key in results:
        assert choice == f"opt{i}_a"
        assert first_key == f"opt{i}_a"
    assert client.stats["requests"] == 5
    assert client.restarts == 0


def test_parse_answer_shapes() -> None:
    choice = parse_answer({"type": "choice", "choice": "x", "probabilities": {"x": 0.9, "y": 0.1}})
    assert isinstance(choice, ChoiceAnswer) and choice.probabilities["y"] == 0.1
    score = parse_answer({"score": 1.5, "probabilities": {"1": 0.5, "0": 0.0, "2": 0.5}})
    assert isinstance(score, ScoreAnswer) and score.distribution == [0.0, 0.5, 0.5]
    noul = parse_answer({"noul": 0.25, "rl_agent": {"act_probability": 0.1}})
    assert isinstance(noul, NoulAnswer) and noul.noul == 0.25
    with pytest.raises(ValueError):
        parse_answer({"something": 1})
