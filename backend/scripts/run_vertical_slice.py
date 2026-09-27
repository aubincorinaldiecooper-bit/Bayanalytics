#!/usr/bin/env python3
"""Run the first engineering task end to end (AGENT.md section 19): "Assess Apple."

Builds the runtime from the environment (real models when configured, mocks/fixtures otherwise),
submits one analysis, prints every SSE event as it happens, then prints the telemetry block and
writes the full structured result to bench/vertical_slice_<analysis_id>.json.

Examples:
    # fixture research + mock models (no network, no weights):
    BAY_RESEARCH_PROVIDER=fixture BAY_RESEARCH_FIXTURE_DIR=tests/fixtures/research/apple \
    BAY_LAYA_MODE=mock BAY_SPARK_MODE=mock python scripts/run_vertical_slice.py

    # real stack on the Mac (see README runbook):
    set -a; source .env; set +a; python scripts/run_vertical_slice.py --profile fast
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bayanalytics.config import Settings  # noqa: E402
from bayanalytics.instruments.base import ResearchBudget  # noqa: E402
from bayanalytics.pipeline.horizon import resolve_horizon  # noqa: E402
from bayanalytics.schemas.requests import CreateAnalysisRequest  # noqa: E402
from bayanalytics.wiring import build_runtime  # noqa: E402


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="Assess Apple.")
    parser.add_argument("--profile", default="fast", choices=["fast", "deep"])
    parser.add_argument("--horizon", default="auto")
    parser.add_argument("--out", default="bench")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    rt = build_runtime(settings)
    await rt.start()
    try:
        caps = rt.capabilities()
        print("capabilities:", json.dumps(caps.model_dump(mode="json")))
        if not caps.profiles[args.profile].available:
            print(f"profile {args.profile} unavailable: {caps.profiles[args.profile].reason}")
            return 2
        request = CreateAnalysisRequest(query=args.query, profile=args.profile, horizon=args.horizon)  # type: ignore[arg-type]
        job = await rt.runner.submit(
            request,
            resolved_horizon=resolve_horizon(request.query, request.horizon),
            budget=ResearchBudget(
                max_rounds=settings.research_max_rounds,
                max_sources=settings.research_max_sources,
                max_fetch_per_round=settings.research_max_fetch_per_round,
            ),
            as_of=settings.eval_as_of,
        )
        print(f"analysis {job.analysis_id} queued (profile={job.profile}, horizon={job.resolved_horizon})")
        streamed: list[str] = []
        async for event in rt.bus.stream(job.analysis_id):
            if event.event == "spark.token":
                streamed.append(event.data.get("text", ""))
                sys.stdout.write(event.data.get("text", ""))
                sys.stdout.flush()
                continue
            data = {k: v for k, v in event.data.items() if k not in {"analysis_id"}}
            print(f"\n[{event.seq:03d}] {event.event} {json.dumps(data, default=str)[:300]}")
        await asyncio.sleep(0.05)
        result = await rt.store.get_result(job.analysis_id)
        assert result is not None
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"vertical_slice_{job.analysis_id}.json"
        path.write_text(json.dumps(result.model_dump(mode="json"), indent=2))
        print("\n\nstatus:", result.status, "| error:", result.error.code if result.error else None)
        print("sources:", len(result.sources), "| calculations:", len(result.calculations), "| decisions:", len(result.laya_decisions))
        print("telemetry:", json.dumps(result.telemetry.model_dump(mode="json"), indent=2))
        print("result written to", path)
        return 0 if result.status == "completed" else 1
    finally:
        await rt.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
