"""Spark execution profiles, safe-allocation gating and the artifact lockfile.

AGENT.md references: section 1.3 (artifact / reproducibility rule), section 9 (measure, do not
estimate), section 35 (Fast / Deep profiles, safe local allocation), section 39 (error codes).

Threshold note
--------------
``spark_fast_min_available_mb`` and ``spark_deep_min_available_mb`` are *provisional planning
inputs*. Section 9 forbids treating estimated KV-cache numbers as engineering measurements, so
these thresholds exist only to keep the 8 GB reference machine out of obvious swap death until
the real Fast / Deep footprints have been measured with the locked artifact and runtime. Replace
them with measured values; never tune product behaviour around them.

Reasons produced here are shown to users and returned by ``/capabilities``: they describe memory
in MB and never contain file paths, hostnames or command lines.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil

from bayanalytics.config import Settings
from bayanalytics.errors import AnalysisError
from bayanalytics.schemas.capabilities import ProfileCapability
from bayanalytics.schemas.common import ErrorCode, Profile
from bayanalytics.spark.base import ProfileSpec

logger = logging.getLogger("bayanalytics.spark")

# Swap usage above this share of the swap device is treated as memory pressure. Provisional.
SWAP_PRESSURE_PERCENT = 50.0

LOCKFILE_KEYS: tuple[str, ...] = (
    "hf_repo",
    "hf_revision",
    "gguf_quantization",
    "gguf_sha256",
    "gguf_file",
    "llama_cpp_version",
    "chat_template_source",
)


@dataclass(frozen=True)
class MemorySnapshot:
    """One reading of system memory, in MB. ``swap_percent`` is used-swap over swap-total."""

    available_mb: float
    total_mb: float
    swap_used_mb: float
    swap_percent: float


MemoryProbe = Callable[[], MemorySnapshot]
"""A zero-argument callable returning a ``MemorySnapshot``. Injected in tests."""


def psutil_probe() -> MemorySnapshot:
    """Default probe: ``psutil.virtual_memory()`` / ``psutil.swap_memory()``."""
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return MemorySnapshot(
        available_mb=vm.available / _MIB,
        total_mb=vm.total / _MIB,
        swap_used_mb=swap.used / _MIB,
        swap_percent=float(swap.percent),
    )


def static_probe(
    available_mb: float,
    total_mb: float = 8192.0,
    swap_used_mb: float = 0.0,
    swap_percent: float = 0.0,
) -> MemoryProbe:
    """A probe that always returns the same snapshot (tests and demos)."""
    snapshot = MemorySnapshot(available_mb, total_mb, swap_used_mb, swap_percent)
    return lambda: snapshot


_MIB = 1024.0 * 1024.0


def profile_specs(settings: Settings) -> dict[Profile, ProfileSpec]:
    return {
        "fast": ProfileSpec(
            name="fast",
            context_ceiling=settings.spark_fast_ctx,
            kv_cache_type=settings.spark_fast_kv,
            min_available_mb=settings.spark_fast_min_available_mb,
        ),
        "deep": ProfileSpec(
            name="deep",
            context_ceiling=settings.spark_deep_ctx,
            kv_cache_type=settings.spark_deep_kv,
            min_available_mb=settings.spark_deep_min_available_mb,
        ),
    }


def profile_unavailable_code(profile: Profile) -> ErrorCode:
    if profile == "deep":
        return ErrorCode.DEEP_PROFILE_UNAVAILABLE
    return ErrorCode.FAST_PROFILE_UNAVAILABLE


def memory_shortfall(spec: ProfileSpec, snapshot: MemorySnapshot) -> str | None:
    """A user-facing reason when ``snapshot`` cannot host ``spec`` safely, else ``None``."""
    if snapshot.available_mb < spec.min_available_mb:
        return (
            f"the {spec.name} profile needs about {spec.min_available_mb} MB of free memory "
            f"(provisional threshold); {snapshot.available_mb:.0f} MB of "
            f"{snapshot.total_mb:.0f} MB are available"
        )
    if snapshot.swap_percent > SWAP_PRESSURE_PERCENT:
        return (
            f"swap usage is {snapshot.swap_percent:.0f}% ({snapshot.swap_used_mb:.0f} MB in "
            f"use), which indicates memory pressure"
        )
    return None


def check_availability(
    profile: Profile,
    settings: Settings,
    probe: MemoryProbe,
    model_present: bool,
    runtime_present: bool,
    *,
    external_healthy: bool | None = None,
    external_n_ctx: int | None = None,
) -> ProfileCapability:
    """Capability answer for ``/capabilities`` (AGENT.md section 37.6).

    - managed mode: artifact and runtime presence first (SPARK_START_FAILED), then memory
      arithmetic (FAST_/DEEP_PROFILE_UNAVAILABLE);
    - external mode: availability comes from the last health check (``external_healthy``) and
      the context size the external server reports (``external_n_ctx``), never from memory
      arithmetic, because that server's memory is not ours to reason about.
    """
    spec = profile_specs(settings)[profile]
    ceiling = spec.context_ceiling

    if settings.spark_mode == "external":
        if external_healthy is False:
            return ProfileCapability(
                available=False,
                context_ceiling=ceiling,
                reason="the external llama-server did not pass its health check",
                code=ErrorCode.SPARK_START_FAILED,
            )
        if external_n_ctx is not None and external_n_ctx < ceiling:
            return ProfileCapability(
                available=False,
                context_ceiling=ceiling,
                reason=(
                    f"the external llama-server is loaded with a {external_n_ctx} token "
                    f"context; the {profile} profile needs {ceiling}"
                ),
                code=profile_unavailable_code(profile),
            )
        return ProfileCapability(available=True, context_ceiling=ceiling)

    if not model_present:
        return ProfileCapability(
            available=False,
            context_ceiling=ceiling,
            reason="model artifact not found",
            code=ErrorCode.SPARK_START_FAILED,
        )
    if not runtime_present:
        return ProfileCapability(
            available=False,
            context_ceiling=ceiling,
            reason="llama-server not found",
            code=ErrorCode.SPARK_START_FAILED,
        )
    shortfall = memory_shortfall(spec, probe())
    if shortfall is not None:
        return ProfileCapability(
            available=False,
            context_ceiling=ceiling,
            reason=shortfall,
            code=profile_unavailable_code(profile),
        )
    return ProfileCapability(available=True, context_ceiling=ceiling)


def assert_can_allocate(profile: Profile, settings: Settings, probe: MemoryProbe) -> MemorySnapshot:
    """Request-time gate run right before a managed llama-server is (re)started.

    Raises ``DEEP_PROFILE_UNAVAILABLE`` when Deep cannot be allocated safely and
    ``MEMORY_PRESSURE`` for swap pressure or a Fast shortfall. A Deep request that cannot be
    allocated is a structured error, never a silent fallback to Fast (AGENT.md section 35).
    Returns the snapshot so the caller can record it in telemetry.
    """
    spec = profile_specs(settings)[profile]
    snapshot = probe()
    details = {
        "profile": profile,
        "available_mb": round(snapshot.available_mb),
        "required_mb": spec.min_available_mb,
        "swap_percent": round(snapshot.swap_percent, 1),
    }
    if snapshot.swap_percent > SWAP_PRESSURE_PERCENT:
        raise AnalysisError(
            ErrorCode.MEMORY_PRESSURE,
            f"There is not enough free memory to run this analysis safely: "
            f"{memory_shortfall(spec, snapshot)}.",
            details=details,
        )
    if snapshot.available_mb < spec.min_available_mb:
        reason = memory_shortfall(spec, snapshot)
        if profile == "deep":
            raise AnalysisError(
                ErrorCode.DEEP_PROFILE_UNAVAILABLE,
                f"Deep analysis cannot be allocated safely right now: {reason}. Try Fast.",
                details=details,
            )
        raise AnalysisError(
            ErrorCode.MEMORY_PRESSURE,
            f"There is not enough free memory to run this analysis safely: {reason}.",
            details=details,
        )
    return snapshot


def read_lockfile(path: Path | str | None) -> dict[str, Any] | None:
    """Read ``models/spark.lock.json`` written by the model setup script.

    Expected keys: ``hf_repo``, ``hf_revision``, ``gguf_quantization``, ``gguf_sha256``,
    ``gguf_file``, ``llama_cpp_version``, ``chat_template_source``. Missing keys are returned
    as ``None``; a missing or unreadable file yields ``None`` (the job then records only the
    configured artifact name).
    """
    if path is None:
        return None
    file = Path(path)
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("spark lockfile unreadable: %s", type(exc).__name__)
        return None
    if not isinstance(raw, dict):
        logger.warning("spark lockfile is not a JSON object")
        return None
    lock = {key: raw.get(key) for key in LOCKFILE_KEYS}
    for key, value in raw.items():
        if key not in lock:
            lock[key] = value
    return lock


def version_fields(
    lock: dict[str, Any] | None,
    settings: Settings,
    runtime_version: str | None = None,
) -> dict[str, str | None]:
    """The ``VersionInfo`` fields Spark is responsible for (``VersionInfo(**fields)`` works).

    The artifact is named only from the download lockfile (a record of a verified download);
    without a lockfile it is ``None``, never a configured label.
    """
    lock = lock or {}
    artifact: str | None = None
    if lock.get("hf_repo") and lock.get("gguf_quantization"):
        artifact = f"{lock['hf_repo']}:{lock['gguf_quantization']}"
    runtime = runtime_version or _as_str(lock.get("llama_cpp_version"))
    return {
        "spark_artifact": artifact,
        "spark_runtime": runtime,
        "spark_gguf_sha256": _as_str(lock.get("gguf_sha256")),
        "spark_hf_revision": _as_str(lock.get("hf_revision")),
    }


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)
