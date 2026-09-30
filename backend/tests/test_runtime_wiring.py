"""Runtime container, the ``build_runtime`` seams, CLI options and body-limit envelopes."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal, get_args

import httpx
import pytest

from bayanalytics.config import Settings, set_settings
from bayanalytics.errors import AnalysisError
from bayanalytics.main import create_app
from bayanalytics.runtime import Runtime
from bayanalytics.schemas.capabilities import ProfileCapability
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.results import ExecutionInfo, VersionInfo
from bayanalytics.wiring import build_runtime
from doubles import FixedTranscriber, RuleLaya, ScriptedSpark, fixture_research_stack
from test_core_jobs_api import FakeSpark, _client, _pipeline_ok, _runtime

FIXTURES = Path(__file__).parent / "fixtures" / "research" / "apple"


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {"log_level": "WARNING"}
    base.update(overrides)
    return Settings(**base)


def _full_runtime(
    settings: Settings | None = None, *, fixture_dir: Path = FIXTURES, **doubles: Any
) -> Runtime:
    settings = settings or _settings()
    return build_runtime(
        settings,
        laya=doubles.get("laya") or RuleLaya(),
        spark=doubles.get("spark") or ScriptedSpark(),
        transcriber=FixedTranscriber(),
        research=fixture_research_stack(settings, fixture_dir),
    )


def test_execution_and_version_info_carry_no_configured_labels() -> None:
    assert set(ExecutionInfo.model_fields) == {
        "spark_mode",
        "whisper_mode",
        "deployment",
        "search_configured",
    }
    info = VersionInfo()
    assert info.laya_package_version is None and info.spark_artifact is None
    assert info.spark_runtime is None and info.execution == ExecutionInfo()
    for field in (
        "research_provider",
        "research_fixture_dir",
        "laya_mode",
        "laya_package_version",
        "spark_artifact",
        "synthetic",
    ):
        assert field not in Settings.model_fields, field
    assert get_args(Settings.model_fields["spark_mode"].annotation) == ("managed", "external")
    assert get_args(Settings.model_fields["whisper_mode"].annotation) == ("cli", "disabled")
    assert Settings.model_fields["spark_mode"].annotation == Literal["managed", "external"]
    assert Settings().graceful_shutdown_s == 10
    assert Settings.from_env({"BAY_GRACEFUL_SHUTDOWN_S": "4"}).graceful_shutdown_s == 4
    rt = _full_runtime()
    assert rt.execution_info() == ExecutionInfo(
        spark_mode="managed", whisper_mode="disabled", deployment="local", search_configured=False
    )
    assert rt.capabilities().execution == rt.execution_info()
    assert rt.capabilities().research is True and rt.capabilities().voice is True
    # price points stay server-side unless the operator turns price display on
    assert Settings().price_display is False and rt.capabilities().market.price_display is False
    assert Settings.from_env({"BAY_PRICE_DISPLAY": "true"}).price_display is True


async def test_runtime_start_reports_a_laya_load_failure_and_a_missing_search_url(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rt = _runtime(_pipeline_ok)

    async def failing_load() -> Any:
        raise AnalysisError(
            ErrorCode.LAYA_INFERENCE_FAILED,
            details={"worker_code": "LOAD_FAILED", "reason": "load_failed"},
        )

    rt.laya.load = failing_load  # type: ignore[method-assign]
    with (
        caplog.at_level("WARNING", logger="bayanalytics.runtime"),
        pytest.raises(RuntimeError, match="LOAD_FAILED: load_failed") as info,
    ):
        await rt.start()
    assert "Laya decision model failed to load" in str(info.value)
    assert "BAY_LAYA_MODEL_DIR" in str(info.value)
    assert isinstance(info.value.__cause__, AnalysisError)
    assert any("BAY_RESEARCH_SEARCH_URL is not set" in r.getMessage() for r in caplog.records)
    # The application lifespan surfaces the same error instead of a half-started server.
    app = create_app(rt.settings, runtime=rt)
    with pytest.raises(RuntimeError, match="Laya decision model failed to load"):
        async with app.router.lifespan_context(app):
            raise AssertionError("unreachable")


async def test_runtime_close_closes_the_research_provider() -> None:
    settings = _settings()
    stack = fixture_research_stack(settings, FIXTURES)
    provider = stack[0]
    rt = build_runtime(
        settings,
        laya=RuleLaya(),
        spark=ScriptedSpark(),
        transcriber=FixedTranscriber(),
        research=stack,
    )
    await rt.start()
    assert rt.search_configured is False and provider.closed is False
    assert rt.extras["laya_load"]["package_version"] is None  # a double measures no package
    await rt.close()
    assert provider.closed is True
    assert not (await rt.laya.health()).loaded
    await rt.close()  # idempotent


class _ProbingSpark(FakeSpark):
    """External-server double: unavailable until its health probe says otherwise."""

    def __init__(self) -> None:
        super().__init__(deep=True)
        self.probes = 0
        self.healthy = False  # the live server state, seen only through a probe
        self.probed_healthy = False  # what the last probe reported (the real client's cache)
        self.manager = self

    async def probe_external(self) -> bool:
        self.probes += 1
        self.probed_healthy = self.healthy
        return self.healthy

    async def version_info(self) -> dict[str, str | None]:
        return {"spark_artifact": None, "spark_runtime": "b-external"}

    def availability(self, profile: str) -> ProfileCapability:
        if not self.probed_healthy:
            return ProfileCapability(
                available=False,
                context_ceiling=32768,
                reason="the external llama-server did not pass its health check",
                code=ErrorCode.SPARK_START_FAILED,
            )
        return super().availability(profile)


async def test_refresh_spark_reprobes_only_an_external_server() -> None:
    managed = _runtime(_pipeline_ok)
    managed.spark = _ProbingSpark()
    await managed.refresh_spark()
    assert managed.spark.probes == 0  # a managed server needs no refresh

    external = _runtime(_pipeline_ok)
    external.settings = external.settings.model_copy(update={"spark_mode": "external"})
    external.spark = _ProbingSpark()
    await external.refresh_spark()
    assert external.spark.probes == 1 and "spark_version" not in external.extras
    external.spark.healthy = True
    await external.refresh_spark()
    assert external.spark.probes == 2
    assert external.extras["spark_version"] == {
        "spark_artifact": None,
        "spark_runtime": "b-external",
    }


async def test_post_and_health_reprobe_an_external_server_that_came_up() -> None:
    rt = _runtime(_pipeline_ok)
    rt.settings = rt.settings.model_copy(update={"spark_mode": "external"})
    spark = _ProbingSpark()
    rt.spark = spark
    async with _client(rt) as client:
        probes_after_start = spark.probes
        first = await client.post("/api/v1/analyses", json={"query": "Assess Apple."})
        assert first.status_code == 503
        assert first.json()["error"]["code"] == "SPARK_START_FAILED"
        assert spark.probes == probes_after_start + 1
        spark.healthy = True  # an operator started llama-server after this backend
        second = await client.post("/api/v1/analyses", json={"query": "Assess Apple."})
        assert second.status_code == 202, second.text
        assert spark.probes == probes_after_start + 2
        [_ async for _ in rt.bus.stream(second.json()["analysis_id"])]
        health = (await client.get("/api/v1/health")).json()
        assert spark.probes == probes_after_start + 3  # health re-probes too
        assert health["execution"]["spark_mode"] == "external"
        assert next(c for c in health["components"] if c["name"] == "spark")["status"] == "ok"


async def test_wiring_resolver_factory_reports_a_missing_ticker_directory(tmp_path: Path) -> None:
    (tmp_path / "pages.json").write_text('{"fixture": true, "pages": {}}')
    settings = _settings()
    rt = _full_runtime(settings, fixture_dir=tmp_path)
    with pytest.raises(AnalysisError) as info:
        await rt.extras["resolver_factory"]()
    outage = info.value
    assert outage.code == ErrorCode.RESEARCH_UNAVAILABLE and outage.retryable is True
    assert outage.details == {"stage": "ticker_directory", "reason": "ticker_directory_unavailable"}
    assert "directory is unavailable" in outage.message
    app = create_app(settings, runtime=rt)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/v1/analyses", json={"query": "Assess Apple."})
    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "RESEARCH_UNAVAILABLE"
    assert resp.json()["error"]["details"]["reason"] == "ticker_directory_unavailable"
    assert rt.runner.active_count == 0  # nothing was admitted
    # With the directory reachable the resolver holds the whole list; there is no partial mode.
    good = _full_runtime(settings)
    resolver = await good.extras["resolver_factory"]()
    assert resolver.resolve("Assess Apple.").symbol == "AAPL"
    assert not hasattr(resolver, "directory_complete")


async def test_wiring_versions_are_measured_not_configured(tmp_path: Path) -> None:
    lock = tmp_path / "spark.lock.json"
    lock.write_text(
        json.dumps(
            {
                "hf_repo": "example/spark-gguf",
                "hf_revision": "rev-1",
                "gguf_quantization": "Q4_K_M",
                "gguf_sha256": "cafe",
                "gguf_file": "spark.gguf",
                "llama_cpp_version": "b-lock",
            }
        )
    )

    class VersionedSpark(ScriptedSpark):
        async def version_info(self) -> dict[str, str | None]:
            return {
                "spark_artifact": "example/spark-gguf:Q4_K_M",
                "spark_runtime": "b-measured",
                "spark_gguf_sha256": "cafe",
                "spark_hf_revision": "rev-1",
            }

    rt = _full_runtime(_settings(spark_lockfile=lock), spark=VersionedSpark())
    versions = rt.runner._versions
    assert callable(versions)
    # Before start nothing has been measured, whatever the lockfile says.
    assert versions() == {
        "normalization_version": "2026.09-1",
        "laya_schema_version": "finance-v1",
        "spark_artifact": None,
        "spark_runtime": None,
    }
    assert rt.extras["spark_lock"]["gguf_sha256"] == "cafe"
    await rt.start()
    assert versions()["spark_artifact"] == "example/spark-gguf:Q4_K_M"
    assert versions()["spark_runtime"] == "b-measured"
    await rt.close()
    # A Spark without a version probe leaves both None even after start.
    plain = _full_runtime()
    await plain.start()
    assert plain.runner._versions()["spark_artifact"] is None
    assert plain.runner._versions()["spark_runtime"] is None
    assert plain.extras["spark_lock"] == {}
    await plain.close()


def test_cli_serve_passes_graceful_shutdown_to_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    import uvicorn

    from bayanalytics import cli

    recorded: dict[str, Any] = {}

    class FakeServer:
        def __init__(self, config: uvicorn.Config) -> None:
            recorded["config"] = config
            self.started = True

        def run(self, sockets: list[Any] | None = None) -> None:
            recorded["sockets"] = sockets

    monkeypatch.setattr(uvicorn, "Server", FakeServer)
    monkeypatch.setenv("BAY_GRACEFUL_SHUTDOWN_S", "3")
    monkeypatch.setenv("BAY_LOG_LEVEL", "WARNING")
    try:
        assert cli.main(["serve", "--port", "8123"]) == 0
    finally:
        set_settings(None)
    config = recorded["config"]
    assert config.timeout_graceful_shutdown == 3
    assert config.port == 8123 and config.host == "127.0.0.1"
    assert config.log_level == "warning"
    assert config.app.title == "BayAnalytics"
    assert recorded["sockets"] is None


async def test_body_limit_envelopes_are_compact_json() -> None:
    rt = _runtime(_pipeline_ok)
    rt.settings = rt.settings.model_copy(update={"max_request_body_bytes": 200})
    async with _client(rt) as client:
        big = await client.post("/api/v1/analyses", json={"query": "x" * 500})
        assert big.status_code == 413
        payload = big.json()
        assert big.content == json.dumps(payload, separators=(",", ":")).encode()
        assert b": " not in big.content and b", " not in big.content
        assert big.headers["content-type"] == "application/json"
        assert big.headers["content-length"] == str(len(big.content))
        error = payload["error"]
        assert error["code"] == "INVALID_REQUEST" and error["retryable"] is False
        assert error["message"] == "Request body exceeds the 200 byte limit."

        async def chunks() -> AsyncIterator[bytes]:
            yield b'{"query": "Assess Apple."}'

        chunked = await client.post(
            "/api/v1/analyses", content=chunks(), headers={"content-type": "application/json"}
        )
        assert chunked.status_code == 411
        assert chunked.content == json.dumps(chunked.json(), separators=(",", ":")).encode()
        assert chunked.json()["error"]["message"] == "Request bodies must declare Content-Length."
        assert rt.runner.active_count == 0
