# BayAnalytics

CPU-only financial analysis backend for public equities. Ask about a listed company and get a
fast, sourced, explainable, multi-horizon assessment built from public information:

```
public research → normalization + provenance → Laya (bounded decisions)
→ deterministic calculations → Spark X2.5 1.7B Q4_K_M (synthesis) → analyst
```

No GPU is required. The reference machine is an 8 GB Intel Mac; the same backend runs on
CPU-only cloud infrastructure with the same API contract.

This repository holds the **backend** (`backend/`). The Next.js + Beautiful UI frontend lives in
a separate repository and is a thin client of this API.

## Repository layout

```
backend/
  src/bayanalytics/
    api/            FastAPI routes: analyses, SSE events, cancel, transcriptions, health, capabilities
    jobs/           asyncio job runner, per-analysis event bus, durable job model
    pipeline/       orchestrator (the vertical slice), horizon resolution, result assembly
    instruments/    InstrumentAnalyzer boundary, identity resolution, EquityAnalyzer
    research/       ResearchProvider boundary, SearXNG search, fetch/extract/dedup, SEC EDGAR, prices, intents
    normalization/  units, fiscal periods, market sessions, facts + conflicts + freshness, corporate actions
    laya/           Node worker (@receptron/laya@0.1.2, NDJSON), Python client, finance question schemas, mock
    calculations/   deterministic finance math, reproducible calculation records, formatting
    spark/          llama-server manager, Fast/Deep profiles, streaming client, prompt, bundle fitting, parser, mock
    store/          AnalysisStore boundary, in-memory store, Postgres (asyncpg) store + schema
    telemetry/      RSS / system RAM / swap / CPU sampling, peak tracker
    whisper/        whisper.cpp transcriber (optional voice input), mock
  tests/            unit + integration tests (mock runtimes, fixture research; no network)
  scripts/          model setup, benchmark, vertical-slice runner
  models/           downloaded weights (git-ignored) + spark.lock.json
docs/ARCHITECTURE.md
```

## Quick start (development, no models)

```bash
cd backend
uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q
BAY_RESEARCH_PROVIDER=fixture BAY_RESEARCH_FIXTURE_DIR=tests/fixtures/research/apple \
BAY_LAYA_MODE=mock BAY_SPARK_MODE=mock .venv/bin/python scripts/run_vertical_slice.py
```

The last command runs `Assess Apple.` through the whole pipeline with fixture research and mock
Laya/Spark, printing every SSE event, the streamed answer and the telemetry block. Fixture data is
synthetic and labelled as such; it never reaches a real analysis path.

## Runbook: the 8 GB Intel Mac

1. **Tooling**: Xcode command line tools, `cmake`, Python 3.12, Node 20+ (22 recommended), `uv`.
2. **Backend**: `cd backend && uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -e ".[dev]"`.
3. **Laya**: `scripts/install_laya_worker.sh` (installs the pinned `@receptron/laya@0.1.2`, downloads the
   ~1.7 GB ONNX bundle into `~/.cache/receptron-laya`, prints load time and RSS).
4. **llama.cpp**: `scripts/setup_llama_cpp.sh` (builds `llama-server` at tag `b10828`, the first
   with Spark X2.5 support; records the commit in `models/spark.lock.json`).
5. **Spark weights**: `pip install huggingface_hub && scripts/download_spark.py` (official
   `XHToken/Spark-X2.5-1.7B-GGUF`, Q4_K_M only; records revision + sha256 in the lockfile).
6. **Whisper (optional)**: `scripts/setup_whisper.sh`.
7. **Configuration**: copy `backend/.env.example` to `backend/.env`, fill in the paths the scripts
   printed, `BAY_RESEARCH_CONTACT_EMAIL` (SEC EDGAR requires it), `BAY_RESEARCH_SEARCH_URL` (a SearXNG
   instance, the search backend carried over from GNSIS) and `DATABASE_URL`.
8. **Database**: `set -a; source .env; set +a; .venv/bin/bayanalytics migrate`.
9. **Measure before trusting**: `.venv/bin/python scripts/benchmark_local.py --profiles fast deep`
   records load times, TTFT, tokens/s, resident RAM, system peak RAM and swap per profile. Choose
   `BAY_SPARK_*_KV` and the `*_MIN_AVAILABLE_MB` thresholds from these numbers, not from estimates.
10. **Run**: `.venv/bin/bayanalytics serve` (binds 127.0.0.1:8000), then
    `.venv/bin/python scripts/run_vertical_slice.py --profile fast`.

## Persistence (Railway Postgres)

A Railway project `bayanalytics` with a `Postgres` service (postgres-ssl image, 5 GB volume,
us-west2) was provisioned for this backend. Its public TCP endpoint is
`altaria.proxy.rlwy.net:14222`; copy the password from the Railway dashboard (service
`Postgres` → Variables → `POSTGRES_PASSWORD` or `DATABASE_PUBLIC_URL`) into `DATABASE_URL`:

```
DATABASE_URL=postgresql://postgres:<password>@altaria.proxy.rlwy.net:14222/railway?sslmode=require
```

The image uses a self-signed certificate; `sslmode=require` encrypts without verifying the CA.
Without `DATABASE_URL` the backend uses an in-memory store (development only).

## API contract (`/api/v1`)

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/analyses` | create an analysis `{query, instrument?, profile: fast\|deep, horizon}` → `202 {analysis_id, status, profile, resolved_horizon}` |
| GET | `/analyses/{id}/events` | SSE stream of recorded system state (`Last-Event-ID` replay supported) |
| GET | `/analyses/{id}` | the structured result (or the job snapshot while running) |
| POST | `/analyses/{id}/cancel` | best-effort cancellation |
| POST | `/transcriptions` | multipart `audio` → `{text, duration_ms, transcription_ms}` (never starts an analysis) |
| GET | `/health` | process and component health |
| GET | `/capabilities` | Fast/Deep availability with reasons, voice, deployment |

Errors always use `{"error": {"code", "message", "retryable", "details?"}}` with the locked codes
(`AMBIGUOUS_INSTRUMENT`, `INSUFFICIENT_EVIDENCE`, `STALE_EVIDENCE`, `RESEARCH_UNAVAILABLE`,
`SOURCE_CONFLICT`, `MISSING_CALCULATION_INPUT`, `FAST_PROFILE_UNAVAILABLE`,
`DEEP_PROFILE_UNAVAILABLE`, `MEMORY_PRESSURE`, `SPARK_START_FAILED`, `SPARK_INFERENCE_FAILED`,
`LAYA_INFERENCE_FAILED`, `WHISPER_FAILED`, `INTERRUPTED`, `CANCELLED`, `INTERNAL_ERROR`) plus two
HTTP-level codes (`NOT_FOUND`, `INVALID_REQUEST`) that never appear on the event stream.

See `docs/ARCHITECTURE.md` for the event sequence, retrieval-loop termination rules, the
evidence-only boundary and the profile/memory policy.

## What is verified where

| Verified in CI / this repository | Must be verified on the reference machine |
| --- | --- |
| API contract, SSE replay, cancellation, INTERRUPTED marking | Laya ONNX load time and resident RAM |
| Laya worker protocol (real Node worker, stub model) and one controlled restart | Real `systemOne` latency and calibration on finance states |
| Spark streaming client, profile switch command lines, memory-pressure errors (fake server) | llama-server build, GGUF sha256, TTFT, tokens/s, KV precision per profile |
| Deterministic calculations (reproducibility, units, formatting) | Live SEC EDGAR / price / SearXNG retrieval quality |
| Normalization: periods, sessions, restatements, conflicts, freshness, leakage guard | Postgres schema against the Railway instance (`bayanalytics migrate`) |
| End-to-end vertical slice with fixture research + mock models | End-to-end "Assess Apple." with real research and models, with telemetry |

The container this backend was written in has no access to HuggingFace, sec.gov or public search
engines, so none of the right-hand column has been executed yet. The scripts above are the path.
