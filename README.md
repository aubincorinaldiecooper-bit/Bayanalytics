# BayAnalytics

CPU-only financial analysis backend for public equities. It is designed to turn one question
about a listed company into a sourced, explainable, multi-horizon assessment built from public
information:

```
question → Spark X2.5 1.7B Q4_K_M (pass 1: what the question requires) → Laya (bounded validation)
→ public research → normalization + provenance → Laya (bounded decisions)
→ deterministic calculations → Spark (pass 2: synthesis) → analyst
```

Spark understands the question first. Laya constrains. Research and math execute. Spark
explains at the end. No keyword or regex rule decides what a question is about.

No GPU is required. The reference machine is an 8 GB Intel Mac. The same API contract is
intended to run unchanged on CPU-only cloud infrastructure; no cloud deployment exists yet.

This repository holds the **backend** (`backend/`). The Next.js + Beautiful UI frontend lives in
a separate repository and is a thin client of this API.

**Nothing in the product is a mock, a fixture or a placeholder.** Every component the backend
starts is the real one (the `@receptron/laya` Node worker, `llama-server`, `whisper-cli`, a
SearXNG instance, Postgres or the in-memory store), every number in
`telemetry` is measured or `null`, and every token count that sizes a prompt or a Laya state is
taken from the real tokenizer (llama-server `/apply-template` + `/tokenize`; the Laya bundle's
tokenizer through the worker's `count_tokens` op). Test doubles and the synthetic Apple fixture
exist only under `backend/tests/doubles` and `backend/tests/fixtures`; they are not importable
from the product and no setting can select them.

## Repository layout

```
backend/
  src/bayanalytics/
    api/            FastAPI routes: analyses, SSE events, cancel, transcriptions, health, capabilities
    jobs/           asyncio job runner (admission control), per-analysis event bus, durable job model
    pipeline/       orchestrator (the vertical slice), horizon resolution, question understanding (Spark pass 1) and validation, thesis diff, result assembly
    instruments/    InstrumentAnalyzer boundary, identity resolution, analytical requirements builder, EquityAnalyzer
    research/       ResearchProvider boundary (web search only), SearXNG search, fetch (SSRF gate)/extract/dedup, intents
    normalization/  units, fiscal periods, market sessions, facts + conflicts + freshness, corporate actions
    laya/           Node worker (@receptron/laya@0.1.2, NDJSON), Python client, finance question schemas, measured compaction
    calculations/   deterministic finance math, reproducible calculation records, formatting
    spark/          llama-server manager, Fast/Deep profiles, streaming client, prompt, measured bundle fitting, parser
    store/          AnalysisStore boundary, in-memory store, Postgres (asyncpg) store + schema
    telemetry/      RSS / system RAM / swap / CPU sampling, peak tracker
    whisper/        whisper.cpp transcriber (optional voice input) or disabled
    procenv.py      environment allow-list for every child process
  tests/            unit + integration tests; doubles under tests/doubles, synthetic fixture under tests/fixtures
  scripts/          model setup, benchmark, vertical-slice runner
  models/           downloaded weights (git-ignored) + spark.lock.json
docs/ARCHITECTURE.md
THIRD_PARTY_NOTICES.md
```

## Development

All commands below run from `backend/`; relative paths such as `models/…` and `bench/…` assume it.

```bash
cd backend
uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q          # test doubles + synthetic fixture; no network, no weights
.venv/bin/ruff check src tests scripts
```

The tests exercise the real code paths (runner, event bus, orchestrator, API, the real
`worker.mjs` protocol with a stub module, the real Spark client against a fake llama-server over
`httpx.MockTransport`, the real HTTP fetcher's target policy) with doubles injected through
`build_runtime(...)`. Running an analysis needs the real models: there is no "demo mode".

## Runbook: the 8 GB Intel Mac

1. **Tooling**: Xcode command line tools, `git`, `cmake`, Python 3.12, Node 20+ (22 recommended),
   `uv`, `ffmpeg` (any browser recording that is not 16 kHz mono WAV goes through it), and a
   reachable SearXNG instance: the search service in `search/` (SearXNG) is the backend's only
   data source, and without it analyses cannot run (the backend says so at startup).
2. **Backend**: `cd backend && uv venv --python 3.12 .venv && uv pip install -p .venv/bin/python -e ".[dev]"`.
   The venv has no `pip`; use `uv pip install -p .venv/bin/python …` for extras.
3. **Laya**: `scripts/install_laya_worker.sh` installs the pinned `@receptron/laya@0.1.2` (and its
   tokenizer package), downloads the ~1.7 GB ONNX bundle into `$LAYA_CACHE`
   (default `~/.cache/receptron-laya`, or `BAY_LAYA_CACHE_DIR` when set) and prints load time,
   RSS and the bundle directory. Put that directory in `BAY_LAYA_MODEL_DIR` (or the cache in
   `BAY_LAYA_CACHE_DIR`) so the backend and the script use one copy.
4. **llama.cpp**: `scripts/setup_llama_cpp.sh` builds `llama-server` at tag `b10828` (the minimum
   version the spec requires for Spark X2.5; its `/apply-template` and `/tokenize` endpoints are
   what the backend measures prompts with) and records the commit in `models/spark.lock.json`.
5. **Spark weights**: `uv pip install -p .venv/bin/python huggingface_hub &&
   .venv/bin/python scripts/download_spark.py` (official `XHToken/Spark-X2.5-1.7B-GGUF`, Q4_K_M
   only; records revision + sha256 in the lockfile, which is where `telemetry.versions.spark_artifact`
   comes from).
6. **Whisper (optional)**: `scripts/setup_whisper.sh` (whisper.cpp at tag `v1.9.4`, Whisper Tiny).
7. **Configuration**: copy `backend/.env.example` to `backend/.env`, replace every placeholder
   (a literal `<…>` breaks `source`), fill in the paths the scripts printed,
   `BAY_RESEARCH_SEARCH_URL` and `DATABASE_URL` (`BAY_RESEARCH_CONTACT_EMAIL`, optional, is
   appended to the User-Agent of every fetch). The backend binds `127.0.0.1` and needs no key there; any other host refuses to
   start unless `BAY_API_KEY` is set, and clients then send `Authorization: Bearer <key>` or
   `X-API-Key`. `BAY_MAX_ACTIVE_ANALYSES` (default 4) bounds concurrent analyses; further requests
   get `429 TOO_MANY_ANALYSES`.
8. **Database**: `set -a; source .env; set +a; .venv/bin/bayanalytics migrate`.
9. **Measure before trusting**: `.venv/bin/python scripts/benchmark_local.py --profiles fast deep`
   (with `.env` exported) records Laya load time and warm latency, and per Spark profile the load
   time, TTFT, tokens/s, llama-server RSS and the prompt size measured before the run next to the
   server's own `usage.prompt_tokens` (they must agree), plus system peak RAM and swap for the whole
   run. Choose `BAY_SPARK_*_KV` and the `*_MIN_AVAILABLE_MB` thresholds from these numbers.
10. **Run**: either `.venv/bin/bayanalytics serve` (binds 127.0.0.1:8000) and drive it over HTTP,
    or `.venv/bin/python scripts/run_vertical_slice.py --profile fast` on its own (it starts its own
    Laya worker and llama-server). Do not run both at once on the 8 GB machine.

## Persistence (Railway Postgres)

A Railway project `bayanalytics` with a `Postgres` service was provisioned for this backend;
its public TCP endpoint is `altaria.proxy.rlwy.net:14222`. The service's own `DATABASE_URL`
variable is the private-network form and only works inside Railway; from the Mac build the URL
from the dashboard values (service `Postgres` → Variables → `POSTGRES_USER`, `POSTGRES_PASSWORD`,
`POSTGRES_DB`):

```
DATABASE_URL=postgresql://postgres:<POSTGRES_PASSWORD>@altaria.proxy.rlwy.net:14222/railway?sslmode=require
```

`BAY_DATABASE_URL` is honoured as well. Schema version 2 adds the index the prior-assessment
lookup uses (`analyses_instrument_symbol_idx`); `bayanalytics migrate` applies it in place. The
image uses a self-signed certificate, so
`sslmode=require` encrypts without verifying the CA (`verify-full` needs the CA via
`?sslrootcert=`). Without a database URL the backend uses an in-memory store (development only;
finished analyses are evicted after 200). The Postgres store, migrations and the INTERRUPTED
recovery path have been exercised against a local PostgreSQL 16 (see below) and the Railway
deploy runs `bayanalytics migrate` against the Railway instance before each release. The Postgres
tests run when `BAY_TEST_DATABASE_URL` points at a disposable database.

## Deploying on Railway

The backend service (`bayanalytics-api-live`) builds `backend/Dockerfile`, which Railway detects at
the root of the service's source directory. Railway's `railway.json` config-as-code is deprecated,
so the settings live on the service:

| Setting | Value |
| --- | --- |
| Source | this repository, branch `main`, root directory `/backend` |
| Start command | `/bin/sh -c "exec bayanalytics serve --host :: --port ${PORT:-8000}"` |
| Pre-deploy command | `bayanalytics migrate` |
| Healthcheck | `/healthz`, timeout 1200 s (200 only when every component reports `ok`) |
| Variables | `BAY_API_KEY`, `DATABASE_URL` (the Postgres service's private URL), `BAY_DEPLOYMENT=cloud`, `PORT=8000` |

Railway runs a Dockerfile service's start command in exec form, hence the `sh -c` for `${PORT}`.
With `--host ::`, `bayanalytics serve` opens one dual-stack socket (`IPV6_V6ONLY` off) and hands it
to uvicorn: Railway's healthcheck and public edge arrive over IPv4, the private network over IPv6.
Plain `uvicorn --host ::` would not do this, because asyncio sets `IPV6_V6ONLY` on the sockets it
binds, and `0.0.0.0` leaves no IPv6 listener for the private network. The frontend reaches the backend over the private network with
`BAY_API_URL=http://${{bayanalytics-api-live.RAILWAY_PRIVATE_DOMAIN}}:8000/api/v1` and
`BAY_API_KEY=${{bayanalytics-api-live.BAY_API_KEY}}`, so the key has one source of truth.

## Data sources: web search only

The backend aggregates what the configured web search returns and nothing else (the owner's
rule, see `CLAUDE.md`). The search service in `search/` (SearXNG) is its only data source: it
contacts that search engine (`BAY_RESEARCH_SEARCH_URL`; on Railway
`http://${{searxng.RAILWAY_PRIVATE_DOMAIN}}:8080`) and the pages those searches returned, with
their redirects and `robots.txt`. `HttpResearchProvider` refuses to open any URL its own search
did not return, query templates carry no `site:` operator or provider name, and
`tests/test_source_policy.py` fails the build if product code gains a hard-coded data URL. Web
pages are evidence as text: they are not parsed into financial figures or price series, so the
deterministic calculations report their missing operands and every result says it is based only
on the web pages found by search. The instrument is the ticker in the question (`$AAPL`, `AAPL`
or an explicit `instrument`); without one `POST /analyses` answers `AMBIGUOUS_INSTRUMENT` with
`details.reason = ticker_required`.

## Live research monitoring

While an analysis runs, the event stream shows exactly what research is doing: each web search
with its hit list (`research.search_results`), each request right before it goes out
(`research.fetching`), and how it ended (`research.source_found`, `research.source_rejected` with
a reason, or `research.fetch_skipped`). Found sources carry the domain, request time, text size
and the excerpt only when the source's redistribution terms allow it.

## API contract (`/api/v1`)

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/analyses` | create an analysis `{query, instrument?, profile: fast\|deep, horizon}` → `202 {analysis_id, status, profile, resolved_horizon}` |
| GET | `/analyses` | history for the sidebar: `?limit=1..100&cursor=` → `{analyses: [summary], next_cursor}`, newest first |
| GET | `/analyses/{id}/events` | SSE stream of recorded system state (`Last-Event-ID` replay supported) |
| GET | `/analyses/{id}` | the structured result (or the persisted artifacts so far while running or after an interruption) |
| POST | `/analyses/{id}/cancel` | best-effort cancellation |
| POST | `/transcriptions` | multipart `audio` → `{text, duration_ms, transcription_ms}` (never starts an analysis) |
| GET | `/health` | process and component health (`down` if any component is down, `degraded` if any is degraded) |
| GET | `/capabilities` | Fast/Deep availability with reasons, voice, deployment, `execution` (Spark mode, voice mode, search configured) |

Errors always use `{"error": {"code", "message", "retryable", "details?"}}` with the locked codes
(`AMBIGUOUS_INSTRUMENT`, `INSUFFICIENT_EVIDENCE`, `STALE_EVIDENCE`, `RESEARCH_UNAVAILABLE`,
`SOURCE_CONFLICT`, `MISSING_CALCULATION_INPUT`, `FAST_PROFILE_UNAVAILABLE`,
`DEEP_PROFILE_UNAVAILABLE`, `MEMORY_PRESSURE`, `SPARK_START_FAILED`, `SPARK_INFERENCE_FAILED`,
`LAYA_INFERENCE_FAILED`, `WHISPER_FAILED`, `INTERRUPTED`, `CANCELLED`, `INTERNAL_ERROR`) plus four
HTTP-level codes (`NOT_FOUND`, `INVALID_REQUEST`, `UNAUTHORIZED`, `TOO_MANY_ANALYSES`) that never
appear on the event stream.

Notes for the client (from a real-HTTP simulation of the frontend reducer):

- `AMBIGUOUS_INSTRUMENT` is answered synchronously by `POST /analyses` (422 with
  `details.candidates`); no analysis is created, so the "Which company did you mean?" picker
  runs before any stream is opened. If a live ticker refresh changes the answer, the same code
  can still arrive as `analysis.failed`.
- `RESEARCH_UNAVAILABLE` covers the search backend: not configured
  (`details.reason = search_not_configured`, not retryable) or every search of the run failed
  (`search_failed`, retryable). A search that fails while others answer lowers
  `telemetry.research.queries_failed` and adds an uncertainty. `INSUFFICIENT_EVIDENCE` means
  fewer than two kept web pages with readable text, or pages from fewer than two websites
  (`details.missing` says which).
- A user cancel ends with `analysis.failed` whose `status` is `cancelled` and `error.code` is
  `CANCELLED`; branch on `status`, and treat the terminal event as authoritative (the cancel
  response echoes the pre-cancel stage). A backend stop or crash is `INTERRUPTED`, never `CANCELLED`.
- `laya.started` / `laya.decision` / `laya.completed` carry a `stage`: `question_validation`
  (only when Spark's interpretation proposed requirements) and `research_plan` happen inside the
  research phase, `evidence_scan`, `history_scan` and `text_evidence` are the scoring phase, and
  `horizon` runs after the calculations. Map by stage, not by first occurrence.
- The question is interpreted before research starts (see "Question understanding" in
  `docs/ARCHITECTURE.md`): Spark pass 1 produces a short structured interpretation, Laya
  confirms or drops each proposed requirement, and Python builds the research plan and
  calculations from what survives. Events and results carry product labels only, never the raw
  interpretation, a prompt or model reasoning. Every `research.started` carries
  `question_intent` (a label such as "Valuation", "Growth", "Event impact", "Relative
  performance" or "General assessment"), `requirements` (labels such as `["Valuation
  history", "Earnings trajectory"]`, empty for a general assessment) and
  `interpretation_source` (`spark`, or `fallback` when the interpretation was unusable or
  rejected and a general assessment runs). The result's `requirements` field holds
  `question_intent`, `requirements`, `interpretation_source`, `dropped_by_validation` (labels
  Laya did not confirm), the `focus` sentence Spark was given, `horizons_emphasis`,
  `satisfied_requirements` / `unmet_requirements` (label and reason), and for research
  intents, calculations and operands the `required_*` / `executed_*` / `satisfied_*` /
  `missing_*` lists (with reasons and missing inputs) plus the `uncertainties` those gaps
  produced (also merged into `assessment.uncertainties`). Unmet requirements never fail an
  analysis; `INSUFFICIENT_EVIDENCE` keeps its meaning.
- Spark pass 1 is internal and emits no `spark.started` / `spark.token` / `spark.completed`,
  but a model load it triggers emits `spark.loading`: the first analysis after a start (or a
  profile switch) shows `spark.loading` between `instrument.resolved` and `research.started`,
  and the synthesis then normally shows none. A Spark runtime failure in pass 1
  (`SPARK_START_FAILED`, `MEMORY_PRESSURE`, `SPARK_INFERENCE_FAILED`) does not stop the
  analysis: it continues as a general assessment (`interpretation_source: "fallback"`, no
  requirements, an uncertainty), research and the calculations run, and the synthesis tries
  the runtime again; only if that also fails does the analysis fail, with its research and
  calculations preserved. An unavailable profile, a cancellation or a shutdown still stops it.
- `spark.queued` (with `profile`, `stage` and `active_analyses`) is emitted only when the
  analysis actually has to wait for the single Spark lane: `stage` is `query_understanding`
  before pass 1 (between `instrument.resolved` and `research.started`) or `synthesis` before
  pass 2. The lane serves requests first come, first served; no stage has priority, so nothing
  starves. `spark.loading` appears only when a model load actually happens; `spark.started`
  arrives with the first token and carries the measured `prompt_tokens`.
- `telemetry` adds `query_understanding_ms` (wall clock of pass 1, lock wait included),
  `query_understanding_prompt_tokens` / `query_understanding_output_tokens` (llama-server
  usage), `query_understanding_load_ms` (only when pass 1 loaded the model; `spark_load_ms`
  is then `null`), `query_understanding_wait_ms` (queued behind another analysis's Spark turn)
  and `query_understanding_generation_ms` (the pass itself: prompt processing and decoding);
  each is `null` when not reported. `BAY_SPARK_UNDERSTANDING_MAX_TOKENS`
  (default 192) bounds pass 1's output.
- A result with `partial: true` and `status: completed` means the synthesis was cut off or a
  horizon section is missing (`horizon_assessments[*].synthesized`).
- `POST /transcriptions` takes the audio as multipart field `audio`.
- `thesis_diff` (additive, on the result): when the store holds an earlier completed analysis
  of the same instrument (`AnalysisStore.latest_completed_result`, created before this one),
  the result carries a structured comparison with it: `previous_analysis_id`, `previous_as_of`,
  `previous_created_at`, `previous_horizon`, `overall` and `horizons` (`scope`, `previous`,
  `current`, `changed`), `metrics` (each calculation both runs computed: `previous_value`,
  `current_value`, `delta`, displays, period labels, calc ids), `new_conflicts` /
  `resolved_conflicts`, `new_uncertainties` / `resolved_uncertainties`, `freshness`
  (`new_quarter`, `newer_prices` with both dates), `stance_changed`, `horizon_scope_changed`
  and a deterministic `summary` list. The overall stance is compared over the horizons both
  runs assessed (`overall.compared_horizons`), so a run that covers fewer or other horizons
  is not reported as a change of thesis. It is `null` when no prior assessment exists, and `assessment.uncertainties`
  then says so. Only structured fields are compared; the earlier narrative is never reused.
  Spark receives the same comparison as a bounded "Prior assessment" block in its evidence.
  The lookup is single-tenant: analyses carry no owner yet, so it spans the whole store;
  `latest_completed_result(..., owner_id=None)` is the seam for scoping it per user and
  raises `NotImplementedError` for any non-null `owner_id` rather than ignoring it. A
  multi-user deployment must scope this lookup by owner before enabling it.
- Calculations added to the `valuation_vs_history` pack, per window of 1 and 3 years: the
  component records `reconciliation_price_return_Ny` (R), `reconciliation_eps_growth_Ny` (g,
  TTM diluted EPS now vs the TTM current N years earlier, fiscal-year pair when quarters are
  missing) and `reconciliation_revenue_growth_Ny`, and the headline
  `valuation_reconciliation_Ny`, whose value is the implied multiple change
  `m = (1 + R) / (1 + g) - 1` in percent. Its `meta.reconciliation` carries `price_return_pct`,
  `eps_growth_pct`, `contributions_pct` (earnings, multiple, interaction), `shares_of_move`,
  `revenue_growth_pct`, `pe_percentile`, `component_calcs`, and the verdict with the rule
  and thresholds that produced it (`verdict`, `verdict_rule`, `thresholds`). Verdicts come
  from a fixed rule table over R, g and m in percentage points (a component below 0.5 point
  is flat; same-sign components within 1.0 point contributed equally, otherwise the larger
  one is what the move "mostly" was; the sign of R, at the same 0.5-point threshold, decides
  "de-rating despite growth" vs "earnings growth offset by de-rating"); the full table is in
  `calculations/reconciliation.py`. Spark receives the verdicts under
  `calculated_metrics.reconciliation` and is told to restate them as given.
  `pe_history_percentile` ranks the current trailing P/E within every quarter-end trailing
  P/E the available EPS and price history can produce; its period label states the
  window used ("over 19 available quarters, Q3 FY2021 to Q2 FY2026"), and `pe_5y_percentile`
  (five-year cap) is kept with the same labelling.
  As everywhere, a missing operand makes the record `unavailable` with `missing_inputs` named.
- Keepalive comments are sent every `BAY_SSE_KEEPALIVE_S` seconds (default 15); reconnecting with
  the terminal event's id closes immediately.
- `429 TOO_MANY_ANALYSES` carries `Retry-After: 5`.

See `docs/ARCHITECTURE.md` for the event sequence, retrieval-loop termination rules, the
evidence-only boundary, the measured token accounting and the profile/memory policy.

## What is verified where

Everything in the left column runs in CI on every push (`.github/workflows/ci.yml`, Python 3.12
and Node 22) with test doubles injected through `build_runtime`, a fake llama-server over
`httpx.MockTransport`, the real `worker.mjs` with a stub Laya module and the synthetic Apple
fixture. The middle column was executed in the development container against a local
PostgreSQL 16 and a real uvicorn process. Nothing in the right column has been executed anywhere
yet: the container this backend was written in had no access to Hugging Face, a SearXNG
instance, the Railway database or the reference machine.

| Verified in CI (doubles, fakes, fixture) | Verified in the container (real process, real Postgres 16) | Not yet executed anywhere |
| --- | --- | --- |
| API contract, SSE replay (`Last-Event-ID`), cancellation, INTERRUPTED marking, API-key gate, body limits, admission control (including the store-await race) | `bayanalytics migrate` twice, schema-version refusal, every table populated by real analyses, INTERRUPTED recovery after SIGKILL, JSONB/typed column agreement, no DSN or path leaks | Loading the real `@receptron/laya` ONNX bundle: load time, resident RAM, `systemOne` latency, calibration on finance states |
| Laya worker NDJSON protocol (`load`, `system_one`, `count_tokens`, `health`, `close`) with the real `worker.mjs` and a stub model, one controlled restart per process lifetime | Real uvicorn: health, capabilities, 202/SSE/GET, cancel, reconnect edge cases, 413/411/401 envelopes, SIGTERM and SIGKILL behaviour, process hygiene | Building llama-server b10828, GGUF download and sha256, a real generation, TTFT, tokens/s, KV precision, whether 128K allocates on 8 GB |
| Spark streaming client, Fast/Deep restart command lines, `/apply-template` + `/tokenize` prompt measurement, memory-pressure and Deep-unavailable errors, cancel-while-silent (fake llama-server) | The real worker answering `load`/`health`/`system_one`/`close` with a bogus model directory (fails fast, no orphan) | A real `whisper-cli` run (tests use a fake shell script) |
| Deterministic calculations (reproducibility, units, formatting), derived Q4, valuation-vs-fundamentals reconciliation, historical P/E window reporting (direct unit tests) | | End-to-end "Assess $AAPL." with live SearXNG retrieval and real models, with telemetry |
| Prior-assessment lookup and thesis diff: two runs on the fixture, the second carrying `thesis_diff` and the Spark prompt the prior block (in-memory store) | `latest_completed_result` against a local PostgreSQL 16 | |
| Normalization: periods, sessions (no exchange-holiday calendar), restatements, conflicts, freshness, leakage guard, corporate actions | | The Railway Postgres instance (`bayanalytics migrate` against it) |
| Fetcher target policy (loopback, link-local, private and every redirect hop refused), DOM depth cap, keyword-only rejection reasons, cache policy | | |
| End-to-end "Assess $AAPL." with the synthetic web-search fixture, `RuleLaya` and `ScriptedSpark` doubles; the source policy (`tests/test_source_policy.py`) | | |
| Question understanding: the pass-1 request (`response_format` JSON schema, 192 tokens, temperature 0), fallbacks, event order and lock release through the real Spark client and a fake llama-server; Laya validation and the requirements builder with scripted interpretations (the doubles do not understand language, so these prove the pipeline, not interpretation quality) | | Pass-1 latency and interpretation quality with the real Spark model on the reference machine |

Known limits not yet addressed: no exchange-holiday calendar (a session around an NYSE holiday
is labelled one day late); web pages are not parsed into financial figures, prices or
corporate actions, so those calculations report missing operands; conflict
handling preserves and surfaces disagreements but does not search for a primary source to
resolve them; z-scores, correlations and scenario tables exist as primitives without a
calculation pack; the evaluation harness is limited to the global `BAY_EVAL_AS_OF` guard; free
prose from Spark is checked for citations but its numbers are not cross-checked against the
bundle.

## Licensing

`pyproject.toml` marks the backend proprietary. The SearXNG discovery pattern in
`research/searxng.py` is adapted from GNSIS under MIT; the licence text and the licences of the
runtimes and models the backend uses are listed in `THIRD_PARTY_NOTICES.md`.
