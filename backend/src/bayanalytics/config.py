"""Backend settings.

Everything is read from environment variables with the ``BAY_`` prefix (``DATABASE_URL`` is
also honoured without the prefix because managed Postgres providers set it that way).
Secrets never appear in ``redacted()`` output, which is the only form that may be logged.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

Deployment = Literal["local", "cloud"]

_SECRET_FIELDS = {"database_url", "api_key", "spark_api_key"}
_PATH_FIELDS_ARE_MASKED = True
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_DEFAULT_WORKER_DIR = Path(__file__).resolve().parent / "laya" / "worker"


def _env_bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(value: str | None, default: list[str]) -> list[str]:
    if value is None or value.strip() == "":
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _env_path(value: str | None) -> Path | None:
    if value is None or value.strip() == "":
        return None
    return Path(value).expanduser()


def _env_datetime(value: str | None) -> datetime | None:
    if value is None or value.strip() == "":
        return None
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


class Settings(BaseModel):
    # --- server -------------------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    deployment: Deployment = "local"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    log_level: str = "INFO"
    api_prefix: str = "/api/v1"
    # Required on every request when set (Authorization: Bearer <key> or X-API-Key). The
    # backend refuses to start on a non-loopback host without it (AGENT.md section 20).
    api_key: str | None = None
    max_active_analyses: int = 4
    max_request_body_bytes: int = 64 * 1024
    max_upload_bytes: int = 25 * 1024 * 1024
    health_cache_s: float = 2.0
    sse_keepalive_s: float = 15.0
    graceful_shutdown_s: int = 10  # uvicorn waits this long for open connections on stop

    # --- persistence --------------------------------------------------------------------
    database_url: str | None = None
    database_pool_min: int = 1
    database_pool_max: int = 4

    # --- research -----------------------------------------------------------------------
    research_search_url: str | None = None  # SearXNG base URL (carried over from GNSIS)
    research_contact_email: str | None = None  # required by SEC EDGAR fair-access policy
    research_user_agent: str = "BayAnalytics/0.1"
    research_cache_dir: Path | None = None
    research_max_rounds: int = 4
    research_max_sources: int = 24
    research_max_fetch_per_round: int = 6
    research_timeout_s: float = 240.0
    research_fetch_timeout_s: float = 20.0
    research_min_request_interval_s: float = 0.25
    research_price_history_days: int = 5 * 366
    eval_as_of: datetime | None = None  # leakage guard: drop evidence published after this

    # --- laya ---------------------------------------------------------------------------
    laya_node_bin: str = "node"
    laya_worker_dir: Path = _DEFAULT_WORKER_DIR
    laya_model_dir: Path | None = None
    laya_cache_dir: Path | None = None
    laya_threads: int | None = None
    laya_load_timeout_s: float = 900.0
    laya_request_timeout_s: float = 120.0
    laya_max_restarts: int = 1

    # --- spark --------------------------------------------------------------------------
    spark_mode: Literal["managed", "external"] = "managed"
    spark_server_url: str = "http://127.0.0.1:8081"
    # Managed mode: the key llama-server is started with (a random per-process key when unset)
    # so no other local process can drive the model. External mode: the key that server uses.
    spark_api_key: str | None = None
    spark_llama_server_bin: str = "llama-server"
    spark_model_path: Path | None = None
    spark_lockfile: Path | None = None
    spark_threads: int | None = None
    spark_fast_ctx: int = 32768
    spark_deep_ctx: int = 131072
    spark_fast_kv: str = "f16"
    spark_deep_kv: str = "q4_0"
    # Provisional planning threshold; replace from measured RAM on the reference machine.
    spark_deep_min_available_mb: int = 4096
    spark_fast_min_available_mb: int = 1536
    spark_max_output_tokens: int = 1400
    spark_temperature: float = 0.2
    # Spark pass 1 (query understanding): a short JSON interpretation, never an answer.
    spark_understanding_max_tokens: int = 192
    spark_start_timeout_s: float = 300.0
    spark_request_timeout_s: float = 900.0

    # --- whisper ------------------------------------------------------------------------
    whisper_mode: Literal["cli", "disabled"] = "disabled"
    whisper_bin: str = "whisper-cli"
    whisper_model_path: Path | None = None
    whisper_threads: int | None = None
    ffmpeg_bin: str = "ffmpeg"
    whisper_timeout_s: float = 120.0

    # --- schema versions recorded on every job (code constants, not runtime measurements) ---
    normalization_version: str = "2026.09-1"
    laya_schema_version: str = "finance-v1"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        e = dict(os.environ if env is None else env)

        def get(name: str, default: str | None = None) -> str | None:
            return e.get(f"BAY_{name}", default)

        kwargs: dict[str, Any] = {}
        for field_name, field in cls.model_fields.items():
            raw = get(field_name.upper())
            if field_name == "database_url" and raw is None:
                raw = e.get("DATABASE_URL")
            if raw is None:
                continue
            annotation = field.annotation
            if annotation is bool:
                kwargs[field_name] = _env_bool(raw, bool(field.default))
            elif annotation == list[str]:
                kwargs[field_name] = _env_list(raw, [])
            elif annotation in (Path, Path | None):
                kwargs[field_name] = _env_path(raw)
            elif annotation in (datetime | None,):
                kwargs[field_name] = _env_datetime(raw)
            elif raw == "":
                continue
            else:
                kwargs[field_name] = raw
        return cls(**kwargs)

    def redacted(self) -> dict[str, Any]:
        """Settings safe for logs: secrets replaced, paths reduced to basenames, email masked."""
        out: dict[str, Any] = {}
        for name, value in self.model_dump().items():
            if name in _SECRET_FIELDS:
                out[name] = "***" if value else None
            elif isinstance(value, Path):
                out[name] = f".../{value.name}" if value.name else str(value)
            elif name == "research_contact_email" and value:
                out[name] = "***"
            else:
                out[name] = value
        return out

    @property
    def is_loopback(self) -> bool:
        return self.host in _LOOPBACK_HOSTS

    @property
    def user_agent(self) -> str:
        if self.research_contact_email:
            return f"{self.research_user_agent} ({self.research_contact_email})"
        return self.research_user_agent


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def set_settings(settings: Settings | None) -> None:
    global _settings
    _settings = settings
