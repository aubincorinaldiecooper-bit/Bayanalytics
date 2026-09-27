"""Memory / CPU instrumentation: samples are measured, peaks are tracked, nothing raises."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import psutil
import pytest

from bayanalytics.schemas.results import Telemetry
from bayanalytics.telemetry import (
    PeakTracker,
    ProcessSample,
    SystemSample,
    own_peak_rss_bytes,
    sample_children,
    sample_process,
    sample_system,
)
from bayanalytics.telemetry import tracker as tracker_module


def test_sample_process_returns_sane_numbers() -> None:
    sample = sample_process()
    assert isinstance(sample, ProcessSample)
    assert sample.rss_mb > 1.0
    assert sample.cpu_percent >= 0.0
    assert sample.peak_rss_mb is not None
    assert sample.peak_rss_mb >= sample.rss_mb - 4.0  # same RSS accounting, tiny lag allowed
    same = sample_process(os.getpid())
    assert abs(same.rss_mb - sample.rss_mb) < 64.0
    peak = own_peak_rss_bytes()
    assert peak is not None and peak > 1024 * 1024
    if sys.platform.startswith("linux"):
        assert peak < 1024**4  # kilobytes were converted to bytes, not left as KB or squared


def test_sample_system_returns_sane_numbers() -> None:
    sample = sample_system()
    assert isinstance(sample, SystemSample)
    assert sample.total_mb > 100.0
    assert 0.0 <= sample.available_mb <= sample.total_mb
    assert 0.0 <= sample.used_mb <= sample.total_mb
    assert abs(sample.used_mb + sample.available_mb - sample.total_mb) < 1.0
    assert sample.swap_used_mb >= 0.0
    assert 0.0 <= sample.swap_percent <= 100.0
    assert set(sample.as_dict()) == {
        "total_mb",
        "available_mb",
        "used_mb",
        "swap_used_mb",
        "swap_percent",
    }


def test_sample_children_dead_pid_yields_none() -> None:
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    assert finished.wait() == 0
    dead_pid = finished.pid
    impossible_pid = 2**22 + 4242  # above Linux pid_max
    samples = sample_children(
        {"me": os.getpid(), "dead": dead_pid, "impossible": impossible_pid, "unset": None}
    )
    assert set(samples) == {"me", "dead", "impossible", "unset"}
    assert samples["me"] is not None and samples["me"].rss_mb > 0
    assert samples["impossible"] is None
    assert samples["unset"] is None
    if not psutil.pid_exists(dead_pid):  # pid reuse is possible but very unlikely
        assert samples["dead"] is None
    with pytest.raises(psutil.NoSuchProcess):
        sample_process(impossible_pid)


async def test_peak_tracker_records_burst_and_stops_cleanly() -> None:
    baseline = sample_process().rss_mb
    tracker = PeakTracker(interval_s=0.02)
    async with tracker as active:
        assert active is tracker
        assert tracker.running
        blob = b"x" * (48 * 1024 * 1024)  # touches every page: RSS really grows
        await asyncio.sleep(0.15)
        del blob
    assert not tracker.running
    snap = tracker.snapshot()
    assert snap["samples"] >= 4
    assert snap["errors"] == 0
    assert snap["process_peak_rss_mb"] is not None
    assert snap["process_peak_rss_mb"] >= baseline + 30.0
    assert snap["process_lifetime_peak_rss_mb"] >= snap["process_peak_rss_mb"] - 4.0
    assert snap["system_total_mb"] and snap["system_peak_used_mb"] is not None
    assert snap["system_min_available_mb"] is not None
    assert snap["swap_peak_used_mb"] is not None and snap["swap_peak_used_mb"] >= 0.0
    assert snap["cpu_max_percent"] is not None and snap["cpu_max_percent"] >= 0.0
    assert snap["duration_ms"] and snap["duration_ms"] >= 100.0
    assert snap["children"] == {}

    telemetry = tracker.to_telemetry(Telemetry(profile="fast", retrieval_ms=5.0))
    assert telemetry.profile == "fast" and telemetry.retrieval_ms == 5.0
    assert telemetry.process_peak_rss_mb == snap["process_peak_rss_mb"]
    assert telemetry.system_total_ram_mb == snap["system_total_mb"]
    assert telemetry.system_peak_ram_mb == snap["system_peak_used_mb"]
    assert telemetry.swap_used_mb == snap["swap_peak_used_mb"]
    assert telemetry.cpu_percent == snap["cpu_max_percent"]
    assert telemetry.laya_peak_rss_mb is None and telemetry.spark_peak_rss_mb is None


async def test_peak_tracker_tracks_named_children() -> None:
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import sys, time; blob = b'y' * (24 * 1024 * 1024); sys.stdout.write('ready\\n'); "
        "sys.stdout.flush(); time.sleep(5)",
        stdout=asyncio.subprocess.PIPE,
    )
    assert child.stdout is not None
    await child.stdout.readline()  # the child has allocated its blob
    pids: dict[str, int | None] = {"laya": child.pid, "spark": None}
    try:
        async with PeakTracker(interval_s=0.02, child_pids=lambda: pids) as tracker:
            await asyncio.sleep(0.1)
            child.kill()
            await child.wait()
            await asyncio.sleep(0.06)  # a sample after the child died
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()
    snap = tracker.snapshot()
    laya = snap["children"]["laya"]
    assert laya["samples"] >= 1
    assert laya["peak_rss_mb"] is not None and laya["peak_rss_mb"] >= 20.0
    assert laya["alive"] is False  # last sample saw a dead pid
    if sys.platform.startswith("linux"):
        assert laya["lifetime_peak_rss_mb"] is not None and laya["lifetime_peak_rss_mb"] >= 20.0
    assert snap["children"]["spark"] == {
        "peak_rss_mb": None,
        "last_rss_mb": None,
        "lifetime_peak_rss_mb": None,
        "cpu_max_percent": None,
        "samples": 0,
        "alive": False,
    }
    telemetry = tracker.to_telemetry(Telemetry())
    assert telemetry.laya_peak_rss_mb == laya["peak_rss_mb"]
    assert telemetry.spark_peak_rss_mb is None
    assert telemetry.whisper_peak_rss_mb is None


def test_to_telemetry_copies_only_present_fields() -> None:
    tracker = PeakTracker()
    base = Telemetry(profile="deep", laya_resident_ram_mb=1900.0)
    untouched = tracker.to_telemetry(base)
    assert untouched == base

    tracker.process_peak_rss_mb = 812.345
    tracker.system_total_mb = 8192.0
    tracker.system_peak_used_mb = 6100.126
    tracker.swap_peak_used_mb = 12.5
    tracker.cpu_max_percent = 355.0
    tracker.children = {
        "laya": {
            "peak_rss_mb": 2048.5,
            "last_rss_mb": 2000.0,
            "lifetime_peak_rss_mb": None,
            "cpu_max_percent": 90.0,
            "samples": 3,
            "alive": True,
        },
        "spark": {
            "peak_rss_mb": 1500.25,
            "last_rss_mb": 1500.0,
            "lifetime_peak_rss_mb": None,
            "cpu_max_percent": 380.0,
            "samples": 3,
            "alive": True,
        },
        "whisper": {
            "peak_rss_mb": None,
            "last_rss_mb": None,
            "lifetime_peak_rss_mb": None,
            "cpu_max_percent": None,
            "samples": 0,
            "alive": False,
        },
        "other": {
            "peak_rss_mb": 5.0,
            "last_rss_mb": 5.0,
            "lifetime_peak_rss_mb": None,
            "cpu_max_percent": 1.0,
            "samples": 1,
            "alive": True,
        },
    }
    out = tracker.to_telemetry(base)
    assert out.profile == "deep"
    assert out.laya_resident_ram_mb == 1900.0  # not owned by the tracker, left alone
    assert out.process_peak_rss_mb == 812.35
    assert out.system_total_ram_mb == 8192.0
    assert out.system_peak_ram_mb == 6100.13
    assert out.swap_used_mb == 12.5
    assert out.cpu_percent == 355.0
    assert out.laya_peak_rss_mb == 2048.5
    assert out.spark_peak_rss_mb == 1500.25
    assert out.whisper_peak_rss_mb is None
    assert base.process_peak_rss_mb is None  # original not mutated


async def test_peak_tracker_survives_cancellation() -> None:
    tracker = PeakTracker(interval_s=0.01)
    entered = asyncio.Event()

    async def work() -> None:
        async with tracker:
            entered.set()
            await asyncio.sleep(30)

    task = asyncio.create_task(work())
    await entered.wait()
    await asyncio.sleep(0.03)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert task.cancelled()
    assert not tracker.running
    assert tracker.samples >= 2


async def test_sampling_errors_never_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(_pid: int | None = None) -> ProcessSample:
        raise psutil.AccessDenied(pid=1)

    monkeypatch.setattr(tracker_module, "sample_process", boom)
    async with PeakTracker(interval_s=0.01) as tracker:
        await asyncio.sleep(0.05)
    assert tracker.errors >= 2
    assert tracker.samples == 0
    assert tracker.snapshot()["process_peak_rss_mb"] is None
    assert tracker.to_telemetry(Telemetry()) == Telemetry()
