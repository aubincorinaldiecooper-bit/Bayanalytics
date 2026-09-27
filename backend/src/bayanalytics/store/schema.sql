-- BayAnalytics persistence schema (AGENT.md sections 5, 7, 36 "Persistence", 38).
--
-- Every statement is idempotent: the file is applied inside one transaction on every
-- backend start and by `python -m bayanalytics.store.migrate`. The typed columns exist for
-- querying and indexing; the `job` / `record` / `result` jsonb columns hold the full
-- Pydantic document (model_dump(mode="json")) and are the source of truth when reading.
--
-- Version bookkeeping: bump SCHEMA_VERSION in postgres.py together with the INSERT below.

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     integer PRIMARY KEY,
    applied_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS analyses (
    analysis_id       text PRIMARY KEY,
    status            text NOT NULL,
    profile           text NOT NULL,
    resolved_horizon  text NOT NULL,
    query             text NOT NULL,
    created_at        timestamptz NOT NULL,
    updated_at        timestamptz NOT NULL,
    finished_at       timestamptz,
    error             jsonb,
    job               jsonb NOT NULL
);

CREATE INDEX IF NOT EXISTS analyses_status_idx ON analyses (status);

CREATE INDEX IF NOT EXISTS analyses_created_at_idx ON analyses (created_at DESC);

-- Version 2: the prior-assessment lookup (latest completed result per instrument) filters on
-- the resolved symbol stored in the job document and orders by creation time.
CREATE INDEX IF NOT EXISTS analyses_instrument_symbol_idx
    ON analyses ((upper(job->'instrument'->>'symbol')), created_at DESC);

CREATE TABLE IF NOT EXISTS analysis_events (
    analysis_id  text NOT NULL REFERENCES analyses (analysis_id) ON DELETE CASCADE,
    seq          integer NOT NULL,
    event        text NOT NULL,
    ts           timestamptz NOT NULL,
    data         jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (analysis_id, seq)
);

CREATE TABLE IF NOT EXISTS sources (
    analysis_id    text NOT NULL REFERENCES analyses (analysis_id) ON DELETE CASCADE,
    source_id      text NOT NULL,
    url            text NOT NULL,
    title          text NOT NULL,
    publisher      text,
    source_type    text NOT NULL,
    published_at   timestamptz,
    retrieved_at   timestamptz NOT NULL,
    fiscal_period  text,
    content_hash   text,
    record         jsonb NOT NULL,
    PRIMARY KEY (analysis_id, source_id)
);

CREATE INDEX IF NOT EXISTS sources_content_hash_idx ON sources (content_hash);

CREATE TABLE IF NOT EXISTS facts (
    analysis_id  text NOT NULL REFERENCES analyses (analysis_id) ON DELETE CASCADE,
    fact_id      text NOT NULL,
    metric       text NOT NULL,
    value        double precision,
    unit         text NOT NULL,
    period_key   text NOT NULL,
    basis        text NOT NULL,
    source_id    text NOT NULL,
    record       jsonb NOT NULL,
    PRIMARY KEY (analysis_id, fact_id)
);

CREATE TABLE IF NOT EXISTS laya_decisions (
    analysis_id    text NOT NULL REFERENCES analyses (analysis_id) ON DELETE CASCADE,
    decision_id    text NOT NULL,
    stage          text NOT NULL,
    decision_type  text NOT NULL,
    decision       text NOT NULL,
    confidence     double precision NOT NULL,
    record         jsonb NOT NULL,
    PRIMARY KEY (analysis_id, decision_id)
);

CREATE TABLE IF NOT EXISTS calculations (
    analysis_id  text NOT NULL REFERENCES analyses (analysis_id) ON DELETE CASCADE,
    calc_id      text NOT NULL,
    name         text NOT NULL,
    value        double precision,
    unit         text NOT NULL,
    status       text NOT NULL,
    record       jsonb NOT NULL,
    PRIMARY KEY (analysis_id, calc_id)
);

CREATE TABLE IF NOT EXISTS results (
    analysis_id   text PRIMARY KEY REFERENCES analyses (analysis_id) ON DELETE CASCADE,
    status        text NOT NULL,
    completed_at  timestamptz,
    result        jsonb NOT NULL
);

INSERT INTO schema_migrations (version) VALUES (1), (2) ON CONFLICT (version) DO NOTHING;
