"""Measured process and system memory samples (AGENT.md sections 9, 19 and 29).

Everything here is read from the OS through ``psutil`` / ``resource``; nothing is estimated.
Values are in mebibytes (``1024 * 1024`` bytes) as floats.

Peak RSS sources:

- own process: ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` (kilobytes on Linux, bytes on
  macOS; both handled),
- other pids on Linux: ``VmHWM`` from ``/proc/<pid>/status``,
- other pids on Windows: ``memory_info().peak_wset``,
- otherwise ``None`` (``PeakTracker`` still tracks the sampled maximum).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import psutil

try:  # ``resource`` does not exist on Windows.
    import resource
except ImportError:  # pragma: no cover - non-POSIX
    resource = None  # type: ignore[assignment]

_MB = 1024.0 * 1024.0


@dataclass(frozen=True, slots=True)
class ProcessSample:
    rss_mb: float
    peak_rss_mb: float | None
    cpu_percent: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SystemSample:
    total_mb: float
    available_mb: float
    used_mb: float  # total - available: what is actually unavailable to new allocations
    swap_used_mb: float
    swap_percent: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# psutil computes cpu_percent from the delta since the previous call on the *same* Process
# object, so we keep one per pid. ``is_running()`` also detects pid reuse via create_time.
_processes: dict[int, psutil.Process] = {}


def _process(pid: int) -> psutil.Process:
    proc = _processes.get(pid)
    if proc is None or not proc.is_running():
        proc = psutil.Process(pid)  # raises psutil.NoSuchProcess for a dead pid
        _processes[pid] = proc
    return proc


def own_peak_rss_bytes() -> int | None:
    """Lifetime peak RSS of this process, in bytes, or ``None`` when unsupported."""
    if resource is None:
        return None
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(ru_maxrss)  # macOS reports bytes
    return int(ru_maxrss) * 1024  # Linux (and the BSDs) report kilobytes


def _linux_peak_rss_bytes(pid: int) -> int | None:
    try:
        with open(f"/proc/{pid}/status", encoding="ascii", errors="replace") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def peak_rss_bytes(pid: int, proc: psutil.Process | None = None) -> int | None:
    if pid == os.getpid():
        return own_peak_rss_bytes()
    if sys.platform.startswith("linux"):
        return _linux_peak_rss_bytes(pid)
    if sys.platform == "win32":  # pragma: no cover - not a target platform
        proc = proc or _process(pid)
        peak = getattr(proc.memory_info(), "peak_wset", None)
        return int(peak) if peak is not None else None
    return None


def sample_process(pid: int | None = None) -> ProcessSample:
    """Sample one process. Raises ``psutil.NoSuchProcess`` for a dead pid.

    ``cpu_percent`` is measured over the interval since the previous sample of the same pid;
    the first sample of a pid therefore reports ``0.0`` (psutil semantics, not an estimate).
    """
    target = os.getpid() if pid is None else int(pid)
    try:
        proc = _process(target)
        rss = proc.memory_info().rss
        cpu = proc.cpu_percent(interval=None)
    except psutil.Error:
        _processes.pop(target, None)
        raise
    peak = peak_rss_bytes(target, proc)
    return ProcessSample(
        rss_mb=rss / _MB,
        peak_rss_mb=(peak / _MB) if peak is not None else None,
        cpu_percent=float(cpu),
    )


def sample_system() -> SystemSample:
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return SystemSample(
        total_mb=vm.total / _MB,
        available_mb=vm.available / _MB,
        used_mb=max(0.0, (vm.total - vm.available) / _MB),
        swap_used_mb=swap.used / _MB,
        swap_percent=float(swap.percent),
    )


def sample_children(pids: Mapping[str, int | None]) -> dict[str, ProcessSample | None]:
    """Sample named child processes (``{"laya": pid, "spark": pid}``).

    Never raises: a dead, unknown or inaccessible pid yields a ``None`` entry.
    """
    out: dict[str, ProcessSample | None] = {}
    for name, pid in pids.items():
        if pid is None:
            out[name] = None
            continue
        try:
            out[name] = sample_process(pid)
        except (psutil.Error, OSError, ValueError, TypeError):
            out[name] = None
    return out


def forget_process(pid: int) -> None:
    """Drop the cached ``psutil.Process`` for a pid (call after a child exits)."""
    _processes.pop(pid, None)
