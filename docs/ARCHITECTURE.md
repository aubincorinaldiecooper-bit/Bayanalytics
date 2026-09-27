# BayAnalytics backend architecture

CPU-only public-equity analysis. One typed (or spoken) question about a listed company becomes a
sourced, multi-horizon assessment produced by a small stack:

```
typed query / microphone
        ↓ (Whisper Tiny, voice only)
instrument identification            instruments/identity.py  (tickers, cashtags, names, possessives)
explicit horizon (if stated)         pipeline/horizon.py      (deterministic phrases only)
        ↓
Spark pass 1: query understanding    pipeline/understanding.py
   a short JSON interpretation constrained to the QueryUnderstanding schema; never an answer
        ↓
Laya question validation             pipeline/questions.py  (confirm or drop each proposed requirement)
        ↓
requirements builder                 instruments/questions.py
   per-requirement rows → research intents (company facts first), calculations, operands, checks
        ↓
active retrieval loop                instruments/equity.py + research/*
   required intents with the horizon seed; required operands become evidence gaps;
   Laya picks a bounded intent  →  deterministic query template  →  search / EDGAR / prices
        ↓
normalization + provenance           normalization/*  (facts, periods, sessions, conflicts)
        ↓
Laya finance wrapper                 laya/*  (question schemas, compaction, worker client)
        ↓
deterministic calculations           calculations/*  (registry, packs, reconciliation, formatting)
   the Laya-chosen pack plus every calculation the question requires
        ↓
prior assessment + thesis diff       pipeline/thesis.py  (store lookup, structured comparison)
        ↓
requirement acceptance checks        instruments/questions.py::check_requirements
   unmet requirements → uncertainties + AnalysisResult.requirements (never a failure)
        ↓
structured evidence bundle           instruments/equity.py::build_spark_bundle (+ question_focus, prior block)
        ↓
Spark pass 2: synthesis (streamed)   spark/*  (profiles, manager, streaming client, parser)
        ↓
assessment + sources + uncertainty   pipeline/assemble.py
```

Spark understands the question first. Laya constrains. Research and math execute. Spark
explains at the end.

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
| Laya (`@receptron/laya@0.1.2`) | fast bounded decisions: confirming or dropping the requirements Spark proposed, research intent, materiality, trends, stances, calculation pack | ONNX Runtime CPU in a persistent Node 20+ worker | `laya/worker/worker.mjs` exposes only `load`, `system_one`, `count_tokens`, `health`, `close` |
| Spark X2.5 1.7B Q4_K_M | pass 1: a short structured interpretation of the question (JSON-schema constrained, internal, never an answer); pass 2: synthesis and explanation over the compact evidence bundle | llama.cpp `llama-server` ≥ b10828 | `spark/manager.py` owns Fast/Deep restarts under the Spark lock; each pass is its own short session |
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
overall and per-horizon stances (previous, current, changed; the overall stance only over
the horizons both runs assessed, so a narrower or different scope is not a change of thesis
and `horizon_scope_changed` records it), every calculation in
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
- **Spark pass 1.** The query-understanding prompt is small and bounded (the rules with every
  allowed value, then the question capped at 300 characters, the company and the horizon; no
  evidence), so it is not fitted; llama-server's `usage` for it is reported as
  `telemetry.query_understanding_prompt_tokens` / `_output_tokens`, `null` when not reported.
- **Spark pass 2.** The orchestrator opens a `SparkClient.session` (the lock is held, the profile is
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
→ spark.queued?                                             (stage query_understanding; only when
                                                            the Spark lane is busy)
→ spark.loading?                                            (only when Spark pass 1 loads the
                                                            model; pass 1 streams nothing else)
→ laya.started → laya.decision × (n + 1) → laya.completed  (question_validation; only when pass 1
                                                            proposed n ≥ 1 requirements)
→ research.started / research.query / research.source_found / research.source_rejected
  (per round; research.started carries question_intent, requirements and interpretation_source;
  then laya.started → laya.decision × n → laya.completed for the research_plan
  stage, which chooses the next bounded intent)
→ research.completed → normalization.completed
→ laya.started → laya.decision × n → laya.completed        (evidence_scan, history_scan, text_evidence)
→ calculation.started → calculation.completed × n
→ laya.started → laya.decision × n → laya.completed        (horizon stances)
→ spark.queued? (stage synthesis) → spark.loading? → spark.started → spark.token × n → spark.completed
→ analysis.completed | analysis.failed
```

Events carry `id: <seq>`; reconnecting with `Last-Event-ID` (or `?after=`) replays the tail
from the store, and reconnecting at or after the terminal event's id closes at once. No prompts
are streamed. `spark.token` carries only the `content` delta of the chat completion; reasoning
that llama-server separates into `reasoning_content` is dropped. Inline reasoning markup inside
`content`, if the model ever produces any, is not filtered (to be verified on the reference
machine).

## Question understanding and analytical requirements

The analyst's question shapes the analysis before any retrieval:

```
user query
  ├─ resolve instrument           deterministic: tickers, cashtags, names, possessives
  ├─ resolve explicit horizon     deterministic, pipeline/horizon.py (only stated periods)
  └─ Spark pass 1                 a short structured QueryUnderstanding (not an answer)
        ↓
  Laya question_validation        one noul per proposed requirement + requirements_supported
        ↓
  requirements builder            per-requirement rows → intents, calculations, operands, checks
        ↓
  research loop → normalization → deterministic math → Laya evidence / stance judgements
        ↓
  check_requirements → uncertainties + result.requirements → Spark pass 2 (synthesis)
```

**No regular expression or keyword rule decides what a question is about.** Deterministic
parsing is limited to resolving the instrument, an explicit horizon phrase and text
normalisation. The meaning of the question comes from Spark pass 1, is constrained by Laya and
executed by ordinary code.

**Bounded vocabularies** (`schemas/questions.py`, each value with a one-line description and a
product label):

- intent: `general_assessment`, `valuation`, `valuation_vs_fundamentals`, `growth`,
  `profitability`, `event_impact`, `relative_performance`, `risk`, `balance_sheet`,
  `capital_return`, `guidance_outlook`;
- requirements (at most 8, deduplicated): `valuation_multiples`, `valuation_history`,
  `price_vs_earnings`, `earnings_trajectory`, `revenue_trajectory`, `margin_trajectory`,
  `cash_flow`, `price_performance`, `benchmark_comparison`, `volatility_drawdown`,
  `balance_sheet`, `capital_return`, `guidance`, `latest_period`, `prior_assessment`,
  `recent_coverage` (labels such as "Valuation history", "Earnings trajectory", "Price
  performance", "Benchmark comparison", "Prior assessment");
- comparison focus: `own_history`, `market`, `sector`, `peers`, `none`;
- flags: `needs_benchmark`, `needs_prior_assessment`, `recent_period_focus`.

**Two Spark passes, one runtime.** Pass 1 (`pipeline/understanding.py`) runs right after the
instrument is resolved, on the same Spark runtime and profile as the synthesis but in its own
short session: the Spark lock is held only while it generates and is released before research
starts. Its system prompt says "Convert the analyst's question about a listed company into the
JSON object described. Do not answer the question. No prose. Use only the listed values. A broad
request such as 'Assess Apple' is general_assessment with no requirements." and lists every
allowed value with its one-liner; the user message is the question (control characters and
evidence markers removed, capped at 300 characters), the company and the resolved horizon. No
evidence and no calculations. The request carries
`response_format: {"type": "json_schema", "json_schema": {"name": "query_understanding",
"schema": …}}` with the `QueryUnderstanding` JSON schema (enums, every field required,
`additionalProperties: false`), which llama-server compiles into a grammar;
`max_tokens = BAY_SPARK_UNDERSTANDING_MAX_TOKENS` (default 192) and temperature 0. There is no
second or third model.

Pass 1 is internal: its tokens are discarded and it emits no `spark.started`, `spark.token` or
`spark.completed`. A model load it triggers still emits the manager's `spark.loading`, so a
client normally sees `spark.loading` between `instrument.resolved` and `research.started`
(pass 2 then finds the profile resident). The output is validated strictly: values outside the
vocabularies are dropped with one product-level note ("part of the question's interpretation
was outside the supported values and was ignored"); malformed or empty JSON, an unusable intent
or a generation cut off by the token limit falls back to the broad interpretation (a general
assessment, no requirements, `interpretation_source: fallback`) with the uncertainty "the
question could not be interpreted; a general assessment was produced". Spark runtime errors
(`SPARK_START_FAILED`, `MEMORY_PRESSURE`, `SPARK_INFERENCE_FAILED`, a profile being unavailable,
`CANCELLED`, `INTERRUPTED`) fail the analysis with that error: the synthesis could not have run
either. Prompts are never logged or persisted; the raw pass-1 output is logged at DEBUG only
and never reaches an event or a result.

**Laya constrains** (`pipeline/questions.py`, stage `question_validation`). When pass 1 proposed
at least one requirement (its list plus the requirements its flags imply: `needs_benchmark` →
`benchmark_comparison`, `needs_prior_assessment` → `prior_assessment`, `recent_period_focus` →
`latest_period`), Laya answers one noul per proposed requirement ("Answering the question
requires valuation history.") and `requirements_supported` ("The proposed requirements fit the
question.") over a compact state holding the question, the instrument, the horizon and the
proposed intent and requirements as labels. The batch is validated and measured like every
other one, emits `laya.started` / `laya.decision` / `laya.completed` with that stage and is
persisted with every other decision. The combination rule is Python
(`instruments/questions.py::combine_validation`): a proposed requirement is kept unless its noul
is below `REQUIREMENT_REJECT_BELOW` (0.3), then it is dropped and recorded in
`dropped_by_validation`; `requirements_supported` below `SUPPORT_REJECT_BELOW` (0.3) rejects the
interpretation (a general assessment runs, every proposed requirement is recorded as dropped,
and an uncertainty says so). Laya only reads proposed requirements, so it can never add one. A
broad interpretation (a general assessment with nothing proposed) has nothing to validate and
Laya is not asked. Which calculation pack runs is still Laya's `calculation_pack` choice,
unchanged.

**Python executes** (`instruments/questions.py`). `REQUIREMENT_TABLE` maps every requirement to
research intents, registry calculations, operands (canonical metric names, `prices`,
`benchmark`, `sector_benchmark`) and a minimum price window; `INTENT_TABLE` maps every intent to
the focus sentence Spark is given and the horizons to emphasise. Both are validated at import
against `ResearchIntent`, `SPECS` and the operand names. `build_requirements` unions the kept
requirements' rows in order (deduplicated), so requirements compose: "valuation history" is
`pe_ttm`, `pe_5y_percentile`, `pe_history_percentile`; "price versus earnings" is the existing
`valuation_reconciliation_1y` record (the three-year record runs alongside when the history reaches
back, never reported as unmet); "benchmark comparison" adds `retrieve_sector_benchmark` and the
relative-return, beta and relative-drawdown records; "prior assessment" retrieves and computes
nothing: it is met when the prior completed assessment exists that the question-agnostic thesis diff
(above) compares against, and the diff is never duplicated.

- *Company facts first.* The research loop ends an intent list as soon as `max_sources` is
  reached, and each search intent can fetch several sources, so searches planned ahead of the
  XBRL company facts could fill the budget and leave the evidence gate without facts. The
  required intents and the seed plan built from them are therefore reordered by a stable
  structural rule (`research/intents.py::facts_first`): company facts and prices, then the other
  structured sources (filings list, benchmarks), then searches.
- *Price window.* A requirement can declare a minimum price window: valuation history and the
  price-versus-earnings reconciliation need `PE_HISTORY_YEARS × 366` days of prices (eight
  quarter-end P/E points and the price three years back), so `build_queries` uses the larger of
  the horizon's window (`PRICE_DAYS`, 400 days for `near_term` / `next_cycle`) and the
  requirements' minimum, for prices and benchmarks alike.
- Required operands the retrieval state lacks become evidence gaps under their metric name
  (`gap_to_intent` maps them to `retrieve_earnings_history`, and once that ran the loop
  escalates to one `retrieve_missing_metric` search whose template spells the metric out); the
  termination rules below are unchanged. Required calculations run whatever pack Laya chose.

After the prior-assessment lookup, `check_requirements` records every requirement as satisfied
or unmet, and each required calculation, operand and intent as satisfied or missing (with the
missing inputs or the reason a formula had no meaningful value). A requirement with no
calculations or operands of its own (guidance, recent coverage) is met only when at least one
kept source was retrieved by one of its intents: an executed search that returned nothing
usable leaves it unmet. Every gap is one sentence such
as "the question needs valuation history but the P/E's five-year percentile could not be
computed: missing eps_ttm, pe_history", added to the assessment's uncertainties and shown to
Spark pass 2 in its instructions (outside the evidence block) with the intent, the requirement
labels, the focus sentence and the horizons to emphasise. Nothing here fails an analysis:
`INSUFFICIENT_EVIDENCE` remains the evidence gate's verdict on facts and primary sources.

A general assessment with no requirements builds nothing: the seed plan is the horizon's, the
calculations are the Laya-chosen pack, Spark pass 2 receives no question focus, and the result
carries `requirements` with the label "General assessment" and empty lists. "Assess X." behaves
exactly as before, apart from the pass-1 call itself.

**Measurement.** Pass 1 is instrumented separately and measured only: `query_understanding_ms`
(wall clock, lock wait and any model load included), `query_understanding_prompt_tokens` and
`query_understanding_output_tokens` (llama-server `usage`), `query_understanding_load_ms` (only
when this pass loaded the profile; the synthesis's `spark_load_ms` is then `null`),
`query_understanding_wait_ms` (queued behind another analysis's Spark turn),
`query_understanding_generation_ms` (prompt processing and decoding: the cost of the semantic
step itself) and the `understanding` stage timer; each is `null` when not reported. Its latency on the 8 GB reference
machine has not been measured yet and must be before any optimisation (prompt caching, a warm
profile, a smaller prompt).

**Waiting.** Pass 1 and pass 2 take turns on the one Spark lane in arrival order
(`asyncio.Lock` is fair and nothing reorders waiters), so a query-understanding pass never
overtakes a synthesis already waiting and no request starves. When the lane is busy at the
moment an analysis needs it, `spark.queued` is emitted with `stage: "query_understanding"` or
`stage: "synthesis"`; an instant turn emits nothing. A client presents both as its ordinary
"thinking" state; the scheduler is not a product concept.

**Failure.** A Spark runtime failure in pass 1 (`SPARK_START_FAILED`, `MEMORY_PRESSURE`,
`SPARK_INFERENCE_FAILED`) makes the interpretation the broad fallback: source `fallback`, no
requirements, one uncertainty, the horizon's seed plan. Research, normalization, the
calculations and the stances run as for any general assessment, and the synthesis tries the
runtime again; if it fails there too, the analysis fails with that code and keeps its sources
and calculations (`partial: true`). An unavailable profile (`FAST_PROFILE_UNAVAILABLE`,
`DEEP_PROFILE_UNAVAILABLE`), a cancellation or a shutdown propagates, and so does anything
unexpected.

## Retrieval loop termination

`EquityAnalyzer.retrieve` runs the seed plan (the question's required intents with the
horizon's, company facts first), then asks Laya for the next bounded intent, an `evidence_sufficient` probability
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
