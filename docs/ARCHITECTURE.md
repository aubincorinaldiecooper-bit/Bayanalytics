# BayAnalytics backend architecture

CPU-only public-equity analysis. One typed (or spoken) question about a listed company becomes a
sourced, multi-horizon assessment produced by a small stack:

```
typed query / microphone
        ↓ (Whisper Tiny, voice only)
instrument identification            instruments/identity.py
        ↓
active retrieval loop                instruments/equity.py + research/*
   Laya picks a bounded intent  →  deterministic query template  →  search / EDGAR / prices
        ↓
normalization + provenance           normalization/*  (facts, periods, sessions, conflicts)
        ↓
Laya finance wrapper                 laya/*  (question schemas, compaction, worker client)
        ↓
deterministic calculations           calculations/*  (registry, packs, formatting)
        ↓
structured evidence bundle           instruments/equity.py::build_spark_bundle
        ↓
Spark X2.5 1.7B Q4_K_M (llama.cpp)   spark/*  (profiles, manager, streaming client, parser)
        ↓
assessment + sources + uncertainty   pipeline/assemble.py
```

## Process topology (local)

```
Next.js client  ──HTTP/SSE──▶  FastAPI (uvicorn, 127.0.0.1:8000)
                                  ├── asyncio job runner (one task per analysis)
                                  ├── Laya worker   : node worker.mjs  (NDJSON over stdin/stdout, resident)
                                  ├── Spark runtime : llama-server     (localhost HTTP, one request at a time)
                                  ├── whisper-cli   : subprocess per transcription (optional)
                                  └── Postgres      : asyncpg pool (Railway) or in-memory store
```

The same API contract runs unchanged against a CPU-only cloud worker; only endpoint location,
model process location, authentication and persistence configuration differ.

## Model roles

| Model | Role | Runtime | Boundary |
| --- | --- | --- | --- |
| Laya (`@receptron/laya@0.1.2`) | fast bounded decisions: research intent, materiality, trends, stances, calculation pack | ONNX Runtime CPU in a persistent Node 20+ worker | `laya/worker/worker.mjs` exposes only `load`, `system_one`, `health`, `close` |
| Spark X2.5 1.7B Q4_K_M | synthesis and explanation over the compact evidence bundle | llama.cpp `llama-server` ≥ b10828 | `spark/manager.py` owns Fast/Deep restarts under the Spark lock |
| Whisper Tiny | speech-to-text only | whisper.cpp `whisper-cli` | `whisper/client.py`; never auto-submits an analysis |

Deterministic Python code does every calculation. Laya chooses the pack and consumes the
results; Spark receives them as given and is told never to recompute or invent numbers.

## Analysis lifecycle and events

Job statuses: `queued → resolving_instrument → researching → normalizing → scoring →
calculating → synthesizing → completed | failed | cancelled`. Every transition is persisted.
The SSE stream (`GET /api/v1/analyses/{id}/events`) carries only recorded system state:

```
analysis.started → instrument.resolved
→ research.started / research.query / research.source_found / research.source_rejected
  (per round; then laya.started → laya.decision × n → laya.completed for the research_plan
  stage, which chooses the next bounded intent)
→ research.completed → normalization.completed
→ laya.started → laya.decision × n → laya.completed        (evidence_scan, history_scan, text_evidence)
→ calculation.started → calculation.completed × n
→ laya.started → laya.decision × n → laya.completed        (horizon stances)
→ spark.queued? → spark.loading? → spark.started → spark.token × n → spark.completed
→ analysis.completed | analysis.failed
```

Events carry `id: <seq>`; reconnecting with `Last-Event-ID` (or `?after=`) replays the tail
from the store. No prompts, hidden reasoning or raw model traces are ever streamed.

## Retrieval loop termination

`EquityAnalyzer.retrieve` runs a horizon-specific seed plan, then asks Laya for the next bounded
intent and an `evidence_sufficient` probability after every round. It stops when evidence is
sufficient (≥ 0.7), Laya chooses `stop_research`, the chosen intent was already executed with no
gaps left, `max_sources` is reached, `max_rounds` is reached, or the user cancels. The counters
`search_rounds, queries_issued, sources_fetched, sources_rejected, duplicate_sources_removed,
evidence_gaps_remaining, retrieval_total_ms` are returned in `telemetry.research`.

## Evidence-only boundary

Retrieved pages are data. They are reduced to metadata + a short excerpt (≤ 600 chars) + a
content hash before Laya sees them (as a compact state object) or Spark sees them (inside an
`<EVIDENCE>` block whose system prompt says instructions found there must be ignored). Tool
calls only originate from application logic and Laya's bounded choices.

## Persistence

Postgres (Railway project `bayanalytics`) holds `analyses`, `analysis_events`, `sources`,
`facts`, `laya_decisions`, `calculations`, `results` and `schema_migrations`. Without
`DATABASE_URL` the in-memory store is used (development and tests; finished analyses are
evicted beyond a cap). On startup every non-terminal job left by a previous process is marked
`failed` / `INTERRUPTED` and receives a terminal `analysis.failed` event; nothing is resumed
silently. A backend shutdown interrupts running jobs the same way (never reported as a user
cancellation).

## Profiles and memory safety

| Profile | Context ceiling | KV cache (provisional) | Use |
| --- | --- | --- | --- |
| fast | 32,768 | f16 | default, everyday analysis |
| deep | 131,072 | q4_0 (requires `-fa on`) | heavy historical analysis |

The profile is snapshotted at `POST /analyses` and never changes mid-analysis. Before starting a
Deep server the manager checks available RAM and swap against configurable thresholds and
returns `DEEP_PROFILE_UNAVAILABLE` / `MEMORY_PRESSURE` instead of forcing the machine into
pressure. Thresholds and KV precision are planning inputs until measured with
`scripts/benchmark_local.py` on the reference machine.

## Historical evaluation hook

`BAY_EVAL_AS_OF` freezes the information set: every source published after that timestamp and
every XBRL fact filed after it is dropped and counted (`leakage guard`). `RecordingProvider` /
`RecordingFetcher` (research/fixture_provider.py) can capture live responses into the fixture
layout but are not yet wired to a setting. The rest of the section-14 harness (per-request
`as_of`, date-bounded search templates, an evaluations table and outcome comparison) is not
built; see the README "What is verified where" table.

## Request guardrails

Loopback binding needs no credential; any other host refuses to start without `BAY_API_KEY`
(sent as `Authorization: Bearer` or `X-API-Key`). Request bodies are limited by
`Content-Length` before they are buffered (64 KiB for JSON, 25 MiB for audio uploads), at most
`BAY_MAX_ACTIVE_ANALYSES` analyses run at once (`429 TOO_MANY_ANALYSES` beyond that), the
retrieval loop is bounded by `ResearchBudget.timeout_s`, and the fetcher refuses loopback,
link-local and private targets on the initial URL and on every redirect hop.
