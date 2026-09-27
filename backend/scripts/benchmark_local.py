#!/usr/bin/env python3
"""Phase 1 local model proof: load, warm inference, RAM and latency for Laya, Spark and Whisper.

Measures, never estimates (AGENT.md sections 9, 29, 35). Writes bench/benchmark_<ts>.json with:
  laya_load_ms, laya_warm_inference_ms, laya_resident_ram_mb
  per profile: spark_load_ms, spark_time_to_first_token_ms, spark_total_inference_ms,
               tokens_per_second, spark_resident_ram_mb, context_ceiling, kv_cache_type
  whisper_transcription_ms (when a WAV is given)
  system_total_ram_mb, system_peak_ram_mb, swap_used_mb, process_peak_rss_mb

Run on the reference machine with the real runtimes configured in .env:
    set -a; source .env; set +a; python scripts/benchmark_local.py --profiles fast deep
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bayanalytics.config import Settings  # noqa: E402
from bayanalytics.context import AnalysisContext  # noqa: E402
from bayanalytics.laya.schemas import evidence_scan_questions  # noqa: E402
from bayanalytics.spark.base import SparkMessage, SparkRunOptions  # noqa: E402
from bayanalytics.telemetry.memory import sample_process, sample_system  # noqa: E402
from bayanalytics.telemetry.tracker import PeakTracker  # noqa: E402
from bayanalytics.wiring import build_laya, build_spark  # noqa: E402
from bayanalytics.whisper import build_transcriber  # noqa: E402

BENCH_STATE = {
    "instrument": "AAPL",
    "horizon": "multi_horizon",
    "latest_metrics": {"revenue": {"value": 9.4e10, "period": "Q3 FY2026"}},
    "sources": 9,
    "primary_sources": 3,
    "conflicts": [],
    "freshness_warnings": [],
    "recent_headlines": ["Fixture results beat expectations", "Guidance unchanged"],
}


async def bench_laya(settings: Settings, out: dict) -> None:
    laya = build_laya(settings)
    t0 = time.perf_counter()
    info = await laya.load()
    out["laya_load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    out["laya_reported_load_ms"] = info.load_ms
    out["laya_resident_ram_mb"] = info.resident_rss_mb
    questions = evidence_scan_questions()
    await laya.system_one(BENCH_STATE, questions)  # cold call
    latencies = []
    for _ in range(5):
        t1 = time.perf_counter()
        result = await laya.system_one(BENCH_STATE, questions)
        latencies.append((time.perf_counter() - t1) * 1000)
    out["laya_warm_inference_ms"] = round(sum(latencies) / len(latencies), 1)
    out["laya_input_tokens"] = result.usage.input_tokens
    health = await laya.health()
    out["laya_resident_ram_mb_after"] = health.resident_rss_mb
    await laya.close()


async def bench_spark(settings: Settings, profiles: list[str], out: dict) -> None:
    spark = build_spark(settings)
    await spark.start()
    ctx = AnalysisContext(analysis_id="bench")
    messages = [
        SparkMessage(role="system", content="You are a concise financial analyst."),
        SparkMessage(
            role="user",
            content="In five sentences, explain what revenue growth, operating margin and free "
            "cash flow margin say about a company's operating health. Do not invent numbers.",
        ),
    ]
    for profile in profiles:
        capability = spark.availability(profile)  # type: ignore[arg-type]
        entry: dict = {"available": capability.available, "reason": capability.reason}
        if capability.available:
            tokens: list[str] = []

            async def on_token(text: str, _tokens: list[str] = tokens) -> None:
                _tokens.append(text)

            t0 = time.perf_counter()
            try:
                gen = await spark.run(profile, messages, on_token, ctx, SparkRunOptions(max_tokens=200))  # type: ignore[arg-type]
                stats = gen.stats
                entry.update(
                    {
                        "context_ceiling": stats.context_ceiling,
                        "kv_cache_type": stats.kv_cache_type,
                        "spark_load_ms": stats.load_ms,
                        "spark_time_to_first_token_ms": stats.time_to_first_token_ms,
                        "spark_total_inference_ms": stats.total_ms,
                        "tokens_per_second": stats.tokens_per_second,
                        "prompt_tokens": stats.prompt_tokens,
                        "output_tokens": stats.output_tokens,
                        "spark_resident_ram_mb": stats.resident_rss_mb,
                        "spark_peak_rss_mb": stats.peak_rss_mb,
                        "wall_ms": round((time.perf_counter() - t0) * 1000, 1),
                        "runtime_version": stats.runtime_version,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - benchmark reports, never hides
                entry["error"] = f"{type(exc).__name__}: {exc}"[:300]
        out.setdefault("spark", {})[profile] = entry
    await spark.close()


async def bench_whisper(settings: Settings, wav: Path | None, out: dict) -> None:
    transcriber = build_transcriber(settings)
    out["whisper_available"] = transcriber.available()
    if wav is None or not transcriber.available():
        return
    t0 = time.perf_counter()
    result = await transcriber.transcribe(wav.read_bytes(), wav.name, "audio/wav")
    out["whisper_transcription_ms"] = result.transcription_ms
    out["whisper_wall_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    out["whisper_audio_duration_ms"] = result.duration_ms
    out["whisper_text"] = result.text[:200]
    stats = getattr(transcriber, "stats", {})
    out["whisper_peak_rss_mb"] = stats.get("whisper_peak_rss_mb") if isinstance(stats, dict) else None


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs="*", default=["fast"])
    parser.add_argument("--wav", type=Path, default=None)
    parser.add_argument("--skip-laya", action="store_true")
    parser.add_argument("--skip-spark", action="store_true")
    parser.add_argument("--out", default="bench")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    out: dict = {
        "started_at": datetime.now(tz=UTC).isoformat(),
        "settings": {k: v for k, v in settings.redacted().items() if k.startswith(("laya", "spark", "whisper"))},
        "system_before": sample_system().__dict__ if hasattr(sample_system(), "__dict__") else str(sample_system()),
    }
    async with PeakTracker(interval_s=0.25) as tracker:
        if not args.skip_laya:
            await bench_laya(settings, out)
        if not args.skip_spark:
            await bench_spark(settings, args.profiles, out)
        await bench_whisper(settings, args.wav, out)
    out["peaks"] = tracker.snapshot()
    proc = sample_process()
    out["process_peak_rss_mb"] = getattr(proc, "peak_rss_mb", None)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"benchmark_{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
    path.write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))
    print("written", path)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
