# BayAnalytics backend architecture

CPU-only public-equity analysis. One typed (or spoken) question about a listed company becomes a
sourced, multi-horizon assessment produced by a small stack:

```
typed query / microphone
        ↓ (Whisper Tiny, voice only)
instrument identification            instruments/identity.py
        ↓
question classification              instruments/questions.py + pipeline/questions.py
   deterministic rules  →  bounded Laya choice only when unclear  →  analytical requirements
        ↓
active retrieval loop                instruments/equity.py + research/*
   required intents first, then the horizon seed; required operands become evidence gaps;
   Laya picks a bounded intent  →  deterministic query template  →  search / EDGAR / prices
        ↓
normalization + provenance           normalization/*  (facts, periods, sessions, conflicts)
        ↓
Laya finance wrapper                 laya/*  (question schemas, compaction, worker client)
        ↓
deterministic calculations           calculations/*  (registry, packs, reconciliation, formatting)
   the Laya-chosen pack plus every calculation the question requires
        ↓
required-operand validation          instruments/questions.py::check_requirements
   unmet requirements → uncertainties + AnalysisResult.requirements (never a failure)
        ↓
prior assessment + thesis diff       pipeline/thesis.py  (store lookup, structured comparison)
        ↓
structured evidence bundle           instruments/equity.py::build_spark_bundle (+ question_focus, prior block)
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

The same API contract is intended to run unchanged against a CPU-only cloud worker; only
endpoint location, model process location, authentication and persistence configuration would
differ. No cloud deployment exists yet.

Every process in this diagram is the real one. There are no mock, fixture or demo modes in the
product: test doubles live under `backend/tests/doubles`, are injected through
`build_runtime(...)` by the tests only, and cannot be selected by configuration.

## Model roles

| Model | Role | Runtime | Boundary |
| --- | --- | --- | --- |
| Laya (`@receptron/laya@0.1.2`) | fast bounded decisions: question kind (only when the rules are unclear), research intent, materiality, trends, stances, calculation pack | ONNX Runtime CPU in a persistent Node 20+ worker | `laya/worker/worker.mjs` exposes only `load`, `system_one`, `count_tokens`, `health`, `close` |
| Spark X2.5 1.7B Q4_K_M | synthesis and explanation over the compact evidence bundle | llama.cpp `llama-server` ≥ b10828 | `spark/manager.py` owns Fast/Deep restarts under the Spark lock |
| Whisper Tiny | speech-to-text only | whisper.cpp `whisper-cli` | `whisper/client.py`; never auto-submits an analysis |

Deterministic Python code does every calculation. Laya chooses the pack and consumes the
results; Spark receives them as given and is instructed never to recompute or invent numbers.
Spark's citations are validated against the known sources and uncited items are dropped from
the structured lists; there is no automated numeric check of Spark's free prose.

Two calculation families answer "what is the market paying for" without a model:

- **Valuation-vs-fundamentals reconciliation** (`calculations/reconciliation.py`). With the
  window's price return `R` and the growth `g` of trailing diluted EPS over the same window,
  the implied change in the multiple is `m = (1 + R) / (1 + g) - 1`, and `R = g + m + g·m`
  gives the contributions. Per window of N = 1 and 3 years every number is its own
  `CalculationResult`: `reconciliation_price_return_Ny`, `reconciliation_eps_growth_Ny`,
  `reconciliation_revenue_growth_Ny` and the headline `valuation_reconciliation_Ny` (value
  `m` in percent), each with formula, recorded operands, period labels and named missing
  inputs. The headline's `meta.reconciliation` holds the contributions, the shares of the
  move, revenue growth, the P/E's historical percentile and the verdict with `verdict_rule`
  and `thresholds`; its note is one quotable sentence. The verdict is the first match in a
  fixed table over R, g and m in percentage points (`VERDICT_RULES`: below 0.5 point a
  component is flat, same-sign components within 1.0 point contributed equally, the sign of
  R at 0.5 point separates "despite" from "offset"), so it is a function of the recorded
  values only; Spark gets it under `calculated_metrics.reconciliation` with an instruction to
  restate it as given. EPS pairs are the latest
  TTM against the TTM that was current one (three) year(s) earlier, located by quarter-end
  date, or the fiscal-year pair when quarters are missing (labelled, never mixed); the
  price/EPS lag is recorded. Non-positive EPS at either end makes the record unavailable
  (`not_meaningful`); a missing close or EPS names the operand in `missing_inputs`.
- **Historical P/E window**: `pe_history_percentile` ranks the current trailing P/E within
  every quarter-end trailing P/E the retrieved EPS and price history can form (no year cap,
  at least eight points), and its period label states the window ("over 19 available
  quarters, Q3 FY2021 to Q2 FY2026"); `pe_5y_percentile` keeps the five-year cap with the
  same reporting.

## Prior assessment and thesis diff

"Did the latest quarter change the thesis?" is answered against the assessment this backend
produced last time, not inferred. After the horizon stances the orchestrator asks the store
for the newest completed result of the same instrument created before the current job
(`AnalysisStore.latest_completed_result(symbol, before=job.created_at)`; Postgres joins
`results` with the job's resolved symbol and uses `analyses_instrument_symbol_idx`, schema
version 2). `pipeline/thesis.py::diff_assessments` compares structured fields only: the
overall and per-horizon stances (previous, current, changed), every calculation in
`THESIS_METRICS` that both runs computed (value then, value now, delta), conflicts and
uncertainties that appeared or went away, and the freshness change (new quarter, newer
prices). The result carries it as `thesis_diff` (recomputed once the final stances and
uncertainties are assembled) and Spark's bundle carries `prior_assessment`, a bounded
(≤ 40 lines) evidence-only block rendered inside `<EVIDENCE>` with the previous stances, the
deltas and the prior `as_of`; the instructions ask Spark to state under "What changed"
whether stances and metrics moved against it. Without a prior assessment `thesis_diff` is
`null`, the bundle says none was found and an uncertainty records it. The earlier run's
narrative is never fed back.

The lookup is single-tenant by construction: jobs carry no owner, so it spans every analysis
in the store. `latest_completed_result(symbol, *, before=None, owner_id=None)` is the seam
for multi-user scoping: the orchestrator passes `owner_id=None` explicitly and both stores
raise `NotImplementedError` ("analyses do not carry ownership yet") for any other value, so
no caller can believe the lookup was filtered by user. A multi-user deployment must record
an owner on analyses and scope this lookup by it before enabling it; nothing else (history
listing, routes, schema) anticipates ownership yet.

## Measured token accounting

Nothing sizes a prompt by counting characters:

- **Laya.** Before every `system_one`, the wrapper measures each question head (instructions and
  option texts, exactly as `buildSequence` encodes them) through the worker's `count_tokens` op,
  which uses the loaded bundle's own tokenizer. Heads that Laya would silently squeeze
  (an option over 48 tokens, options leaving fewer than 16 head tokens, instructions over the
  remaining `head_max_len` of 192) are rejected as `LAYA_INFERENCE_FAILED / invalid_questions`.
  The state budget is `max_len (512) − 4 specials − largest measured head`, and the state is
  compacted deterministically (priority order, 240-character strings, 8-item lists, then
  dropping keys from the tail) with per-key measurements until the assembled state measures
  under budget. Measured heads are cached per question batch.
- **Spark.** The orchestrator opens a `SparkClient.session` (the lock is held, the profile is
  resident), and `fit_bundle` renders each candidate prompt and asks the server for its size:
  `/apply-template` applies the model's chat template, `/tokenize` counts it with special
  tokens as a request would. The trim policy of section 24 runs against that measurement;
  the final measured size is what `spark.started` reports as `prompt_tokens`, and the server's
  `usage.prompt_tokens` after generation must agree with it (the benchmark prints both).
- **Telemetry.** `telemetry.versions.laya_package_version` is what the worker read from the
  installed package, `spark_runtime` is `llama-server --version` (or `/props` build info) and
  `spark_artifact` comes from the download lockfile; each is `null` when not measured. Every
  timing and memory figure is sampled from the OS through `psutil`/`resource`.

## Analysis lifecycle and events

Job statuses: `queued → resolving_instrument → researching → normalizing → scoring →
calculating → synthesizing → completed | failed | cancelled`. Every transition is persisted.
The SSE stream (`GET /api/v1/analyses/{id}/events`) carries only recorded system state:

```
analysis.started → instrument.resolved
→ laya.started → laya.decision × 2 → laya.completed        (question_scan; only when the rules
                                                            could not classify the question)
→ research.started / research.query / research.source_found / research.source_rejected
  (per round; research.started carries question_kind, classification_source and confidence;
  then laya.started → laya.decision × n → laya.completed for the research_plan
  stage, which chooses the next bounded intent)
→ research.completed → normalization.completed
→ laya.started → laya.decision × n → laya.completed        (evidence_scan, history_scan, text_evidence)
→ calculation.started → calculation.completed × n
→ laya.started → laya.decision × n → laya.completed        (horizon stances)
→ spark.queued? → spark.loading? → spark.started → spark.token × n → spark.completed
→ analysis.completed | analysis.failed
```

Events carry `id: <seq>`; reconnecting with `Last-Event-ID` (or `?after=`) replays the tail
from the store, and reconnecting at or after the terminal event's id closes at once. No prompts
are streamed. `spark.token` carries only the `content` delta of the chat completion; reasoning
that llama-server separates into `reasoning_content` is dropped. Inline reasoning markup inside
`content`, if the model ever produces any, is not filtered (to be verified on the reference
machine).

## Question classification and analytical requirements

The analyst's question shapes the analysis before any retrieval:

```
user query
  ↓ classify_question: deterministic keyword/regex rules with a confidence and the matched cues
  ↓ question_scan: one bounded Laya choice over the same kinds, only when the rules are unclear
  ↓ AnalyticalRequirements: required research intents, calculations, operands, focus, horizons
  ↓ seed_plan(horizon, requirements): required intents first, then the horizon seed (deduplicated)
  ↓ compute_gaps: required operands missing from the retrieval state are evidence gaps
  ↓ calculate: the Laya-chosen pack plus every required calculation
  ↓ check_requirements: what was met, what was not → uncertainties + result.requirements
  ↓ Spark bundle question_focus → rendered in the instructions, outside the evidence block
```

The kinds are a closed set (`general_assessment`, `thesis_change`, `valuation`, `growth`,
`profitability_margins`, `relative_performance`, `risk_volatility`, `guidance_outlook`,
`earnings_reaction`, `balance_sheet_liquidity`, `dividends_capital_return`). Each rule adds its
weight to one kind. `general_assessment` cues ("assess", "analyze", "should I buy") are framing:
they win only when no specific kind scored, so "Assess Apple's valuation" is a valuation
question, and they never count as a competitor. Among the rest the leading kind wins with
`0.5 + 0.2 × margin − 0.15 × [a competing kind scored]`, capped at 0.95, so one plain cue is
0.7, a decisive phrase 0.9 and a one-point lead over a competing kind 0.55. Below 0.6, or when
nothing fired or two kinds tied, Laya answers
`question_kind_questions` (a choice over the kinds plus a `recent_period_focus` noul) over a
compact state holding the question, the horizon and the rules' candidates; the decision is
recorded like every other one (stage `question_scan`). Laya's pick below 0.4 confidence is not
trusted to narrow the analysis: a general assessment runs and the result says so. Laya only
picks the kind; the requirements are a table lookup (`instruments/questions.py::REQUIREMENTS`,
validated against the intent set, the calculation registry and the canonical metric names at
import time) and the plan is ordinary code. `general_assessment` requires nothing beyond the
horizon seed plan and the Laya-chosen pack, so "Assess X." behaves exactly as before.

Required operands that the structured sources did not carry surface as gaps under their metric
name (`gap_to_intent` maps them to `retrieve_earnings_history`, and once that was executed the
loop escalates to one `retrieve_missing_metric` search whose template spells the metric out);
the termination rules below are unchanged. After the calculations `check_requirements` lists
each required calculation and operand as satisfied or missing (with the missing inputs or the
reason the formula had no meaningful value) and each required intent the budget did not reach;
every gap is one sentence such as "the question asks about valuation but the trailing P/E could
not be computed: missing eps_ttm", added to the assessment's uncertainties and shown to Spark
as "evidence the question needs that could not be retrieved or computed". Nothing here fails an
analysis: `INSUFFICIENT_EVIDENCE` remains the evidence gate's verdict on facts and primary
sources.

## Retrieval loop termination

`EquityAnalyzer.retrieve` runs the seed plan (the question's required intents, then the
horizon's), then asks Laya for the next bounded intent, an `evidence_sufficient` probability
and a `stale_evidence_matters` probability after every round. It stops when evidence is
sufficient (≥ 0.7), Laya chooses `stop_research` and no untried gap remains, the chosen intent
was already executed, `max_sources` is reached, `max_rounds` is reached, the research budget
timeout (`BAY_RESEARCH_TIMEOUT_S`) expires, or the user cancels. When Laya judges the evidence stale enough to matter, one `retrieve_recent_news`
refresh is forced before stopping. The counters `search_rounds, queries_issued, queries_failed,
structured_failures, sources_fetched, sources_rejected, duplicate_sources_removed,
evidence_gaps_remaining, retrieval_total_ms` are returned in `telemetry.research`; provider
outages (a dead SearXNG, an EDGAR error) become uncertainties on the result, and a run whose
facts are missing because a structured source failed ends as `RESEARCH_UNAVAILABLE`
(`structured_source_failed`) rather than `INSUFFICIENT_EVIDENCE`.

## Evidence-only boundary

Retrieved pages are data. They are reduced to metadata + a short excerpt (≤ 600 chars) + a
content hash before Laya sees them (as a compact state object) or Spark sees them (inside an
`<EVIDENCE>` block whose system prompt says instructions found there must be ignored). Tool
calls only originate from application logic and Laya's bounded choices.

## Persistence

Postgres (Railway project `bayanalytics`) holds `analyses`, `analysis_events`, `sources`,
`facts`, `laya_decisions`, `calculations`, `results` and `schema_migrations` (version 2 adds
the expression index on the job's resolved symbol for the prior-assessment lookup). Without
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
every XBRL fact filed after it is dropped and counted (`leakage guard`). The rest of the
section-14 harness (per-request `as_of`, date-bounded search templates, an evaluations table and
outcome comparison) is not built; see the README "What is verified where" table.

## Request guardrails

Loopback binding needs no credential; any other host refuses to start without `BAY_API_KEY`
(sent as `Authorization: Bearer` or `X-API-Key`). Request bodies are limited by
`Content-Length` before they are buffered (64 KiB for JSON, 25 MiB for audio uploads); at most
`BAY_MAX_ACTIVE_ANALYSES` analyses run at once (`429 TOO_MANY_ANALYSES` beyond that, the slot
reserved before the first store write so concurrent requests cannot over-admit); the retrieval
loop is bounded by `ResearchBudget.timeout_s`.

The fetcher (`research/fetch.py`) resolves every host itself and refuses loopback, link-local,
private, multicast and reserved targets on the initial URL and on every redirect hop (redirects
are walked manually, at most 5 hops), with only the configured SearXNG `host:port` exempt; page
bodies are cached on disk only for SEC and Stooq hosts, with a 24-hour sweep. HTML deeper than
200 nested elements is rejected instead of parsed. Rejection reasons and `RESEARCH_UNAVAILABLE`
details are fixed keywords, never URLs or upstream text.

Child processes (the Node worker, `llama-server`, `whisper-cli`, `ffmpeg`) start with the
allow-listed environment of `procenv.child_env` (PATH, locale, HOME, HF cache settings, Laya and
Node knobs, thread and library-path knobs); `DATABASE_URL`, `BAY_API_KEY` and the rest of the
backend's environment never reach them. Worker and runtime error messages are reduced to
keywords before they reach a client; free text with paths goes to the log only. A backend stop
waits `BAY_GRACEFUL_SHUTDOWN_S` (default 10) for open streams, then running analyses are
reported `INTERRUPTED`.
