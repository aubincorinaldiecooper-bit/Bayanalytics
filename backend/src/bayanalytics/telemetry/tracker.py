"""Background peak sampler for one analysis (AGENT.md sections 9, 19, 29).

Usage in the orchestrator::

    async with PeakTracker(child_pids=lambda: {"laya": laya.pid, "spark": spark.pid}) as peaks:
        ...run the analysis...
    result.telemetry = peaks.to_telemetry(result.telemetry)

The tracker samples the backend process, the system and the named child processes every
``interval_s`` seconds in a background task, folding each sample into running maxima. It takes
one sample on entry and one on exit so even a very short window has data. The sampling task
never raises: a failed sample is counted in ``errors`` and sampling continues. Cancelling the
surrounding task stops the tracker cleanly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from typing import Any

from bayanalytics.schemas.results import Telemetry
from bayanalytics.telemetry.memory import (
    ProcessSample,
    sample_children,
    sample_process,
    sample_system,
)

logger = logging.getLogger(__name__)

ChildPids = Callable[[], Mapping[str, int | None]]

# Children whose peaks map onto named Telemetry fields.
_TELEMETRY_CHILDREN = ("laya", "spark", "whisper")


def _max(current: float | None, value: float | None) -> float | None:
    if value is None:
        return current
    return value if current is None or value > current else current


def _min(current: float | None, value: float | None) -> float | None:
    if value is None:
        return current
    return value if current is None or value < current else current


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, 2)


class PeakTracker:
    def __init__(
        self,
        interval_s: float = 0.5,
        child_pids: ChildPids | None = None,
        pid: int | None = None,
    ) -> None:
        self.interval_s = max(0.01, float(interval_s))
        self._child_pids = child_pids
        self._pid = pid
        self._task: asyncio.Task[None] | None = None
        self._started: float | None = None
        self._stopped: float | None = None

        self.samples = 0
        self.errors = 0
        self.process_peak_rss_mb: float | None = None  # max sampled inside this window
        self.process_last_rss_mb: float | None = None
        self.process_lifetime_peak_rss_mb: float | None = None  # ru_maxrss, whole process
        self.system_total_mb: float | None = None
        self.system_peak_used_mb: float | None = None
        self.system_min_available_mb: float | None = None
        self.swap_peak_used_mb: float | None = None
        self.swap_peak_percent: float | None = None
        self.cpu_max_percent: float | None = None
        self.children: dict[str, dict[str, Any]] = {}

    # --- sampling ---------------------------------------------------------------------

    def sample(self) -> None:
        """Take one sample and fold it into the maxima. Never raises."""
        try:
            self._sample_once()
            self.samples += 1
        except Exception as exc:
            self.errors += 1
            logger.debug("telemetry sample failed: %s", type(exc).__name__)

    def _sample_once(self) -> None:
        proc = sample_process(self._pid)
        self.process_last_rss_mb = proc.rss_mb
        self.process_peak_rss_mb = _max(self.process_peak_rss_mb, proc.rss_mb)
        self.process_lifetime_peak_rss_mb = _max(
            self.process_lifetime_peak_rss_mb, proc.peak_rss_mb
        )
        self.cpu_max_percent = _max(self.cpu_max_percent, proc.cpu_percent)

        system = sample_system()
        self.system_total_mb = system.total_mb
        self.system_peak_used_mb = _max(self.system_peak_used_mb, system.used_mb)
        self.system_min_available_mb = _min(self.system_min_available_mb, system.available_mb)
        self.swap_peak_used_mb = _max(self.swap_peak_used_mb, system.swap_used_mb)
        self.swap_peak_percent = _max(self.swap_peak_percent, system.swap_percent)

        if self._child_pids is None:
            return
        pids = dict(self._child_pids())
        for name, sample in sample_children(pids).items():
            self._fold_child(name, sample)

    def _fold_child(self, name: str, sample: ProcessSample | None) -> None:
        entry = self.children.setdefault(
            name,
            {
                "peak_rss_mb": None,
                "last_rss_mb": None,
                "lifetime_peak_rss_mb": None,
                "cpu_max_percent": None,
                "samples": 0,
                "alive": False,
            },
        )
        if sample is None:
            entry["alive"] = False
            return
        entry["alive"] = True
        entry["samples"] += 1
        entry["last_rss_mb"] = sample.rss_mb
        entry["peak_rss_mb"] = _max(entry["peak_rss_mb"], sample.rss_mb)
        entry["lifetime_peak_rss_mb"] = _max(entry["lifetime_peak_rss_mb"], sample.peak_rss_mb)
        entry["cpu_max_percent"] = _max(entry["cpu_max_percent"], sample.cpu_percent)

    async def _run(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.interval_s)
                self.sample()
        except asyncio.CancelledError:
            return  # clean stop; the final sample is taken by stop()

    # --- lifecycle --------------------------------------------------------------------

    async def start(self) -> None:
        if self._task is not None:
            return
        self._started = time.perf_counter()
        self._stopped = None
        self.sample()
        self._task = asyncio.create_task(self._run(), name="bay-peak-tracker")

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise  # the *caller* is being cancelled; do not swallow that
        self.sample()
        self._stopped = time.perf_counter()

    async def __aenter__(self) -> PeakTracker:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.stop()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def duration_ms(self) -> float | None:
        if self._started is None:
            return None
        end = self._stopped if self._stopped is not None else time.perf_counter()
        return (end - self._started) * 1000.0

    # --- output -----------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        children = {
            name: {
                "peak_rss_mb": _round(entry["peak_rss_mb"]),
                "last_rss_mb": _round(entry["last_rss_mb"]),
                "lifetime_peak_rss_mb": _round(entry["lifetime_peak_rss_mb"]),
                "cpu_max_percent": _round(entry["cpu_max_percent"]),
                "samples": entry["samples"],
                "alive": entry["alive"],
            }
            for name, entry in self.children.items()
        }
        return {
            "samples": self.samples,
            "errors": self.errors,
            "interval_s": self.interval_s,
            "duration_ms": _round(self.duration_ms),
            "process_peak_rss_mb": _round(self.process_peak_rss_mb),
            "process_last_rss_mb": _round(self.process_last_rss_mb),
            "process_lifetime_peak_rss_mb": _round(self.process_lifetime_peak_rss_mb),
            "system_total_mb": _round(self.system_total_mb),
            "system_peak_used_mb": _round(self.system_peak_used_mb),
            "system_min_available_mb": _round(self.system_min_available_mb),
            "swap_peak_used_mb": _round(self.swap_peak_used_mb),
            "swap_peak_percent": _round(self.swap_peak_percent),
            "cpu_max_percent": _round(self.cpu_max_percent),
            "children": children,
        }

    def to_telemetry(self, base: Telemetry) -> Telemetry:
        """Copy measured peaks into a ``Telemetry``; fields without a sample stay untouched."""
        update: dict[str, Any] = {}
        if self.process_peak_rss_mb is not None:
            update["process_peak_rss_mb"] = _round(self.process_peak_rss_mb)
        if self.system_total_mb is not None:
            update["system_total_ram_mb"] = _round(self.system_total_mb)
        if self.system_peak_used_mb is not None:
            update["system_peak_ram_mb"] = _round(self.system_peak_used_mb)
        if self.swap_peak_used_mb is not None:
            update["swap_used_mb"] = _round(self.swap_peak_used_mb)
        if self.cpu_max_percent is not None:
            update["cpu_percent"] = _round(self.cpu_max_percent)
        for name in _TELEMETRY_CHILDREN:
            entry = self.children.get(name)
            if entry is not None and entry["peak_rss_mb"] is not None:
                update[f"{name}_peak_rss_mb"] = _round(entry["peak_rss_mb"])
        return base.model_copy(update=update)
