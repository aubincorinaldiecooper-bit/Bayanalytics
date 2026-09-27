"""asyncpg-backed ``AnalysisStore`` (AGENT.md sections 7 and 36 "Persistence").

Design:

- ``schema.sql`` is idempotent and applied inside one transaction on every ``start()``.
  ``SCHEMA_VERSION`` is the highest migration this code understands; a database that reports a
  higher version belongs to newer code and the store refuses to start against it.
- Every table keeps a few typed columns for querying plus a ``jsonb`` document holding the
  full Pydantic model (``model_dump(mode="json")``). Reads always rebuild from the document.
- The pool installs ``json``/``jsonb`` codecs on every connection so Python dicts go in and
  come out without manual serialisation.
- The database URL is a secret (section 26). It is never logged: ``redact_dsn`` is the only
  form that may appear in log lines or error messages.

TLS: managed providers set ``?sslmode=require`` (or ``ssl=true``). Railway's ``postgres-ssl``
image serves a self-signed certificate, so ``require`` maps to an ``ssl.SSLContext`` with
``check_hostname=False`` / ``verify_mode=CERT_NONE`` (encrypted, not authenticated), which is
what libpq does for ``require``. ``verify-ca`` / ``verify-full`` verify the chain and need the
provider CA in the trust store or via ``?sslrootcert=/path/ca.pem``.
"""

from __future__ import annotations

import json
import logging
import re
import ssl as ssl_module
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import asyncpg

from bayanalytics.jobs.models import AnalysisJob
from bayanalytics.schemas.calculations import CalculationResult
from bayanalytics.schemas.decisions import LayaDecision
from bayanalytics.schemas.events import AnalysisEvent
from bayanalytics.schemas.evidence import NormalizedFact, SourceRecord
from bayanalytics.schemas.results import AnalysisResult
from bayanalytics.store.memory import interrupted_payload

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"

_TERMINAL_SQL = "('completed', 'failed', 'cancelled')"

SslOption = ssl_module.SSLContext | bool | str

_SSL_MODES = {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
_TRUTHY = {"1", "true", "yes", "on", "require"}
_FALSY = {"0", "false", "no", "off", "disable"}


# --- DSN helpers (pure, unit-tested) -------------------------------------------------------


def redact_dsn(url: str | None) -> str:
    """Return ``url`` with the password replaced by ``***``. Safe for logs."""
    if not url:
        return "<none>"
    try:
        parts = urlsplit(url)
    except ValueError:
        return "<invalid-dsn>"
    if not parts.scheme or not parts.netloc:
        # libpq keyword form ("host=... password=...") or something unrecognised.
        return re.sub(r"(?i)(password|sslpassword)=\S+", r"\1=***", url)
    hostinfo = parts.netloc.rsplit("@", 1)[-1]
    userinfo = ""
    if parts.username is not None:
        userinfo = parts.username
        if parts.password is not None:
            userinfo += ":***"
        userinfo += "@"
    query = parts.query
    if query:
        pairs = parse_qsl(query, keep_blank_values=True)
        pairs = [(k, "***" if k.lower() in {"password", "sslpassword"} else v) for k, v in pairs]
        query = urlencode(pairs)
    return urlunsplit((parts.scheme, userinfo + hostinfo, parts.path, query, parts.fragment))


def parse_ssl_mode(url: str) -> tuple[str, str | None, str | None]:
    """Split TLS hints out of a DSN.

    Returns ``(url_without_ssl_params, sslmode, sslrootcert)``. Recognised query parameters are
    ``sslmode=<libpq mode>`` and ``ssl=true|false`` (Railway / Heroku style). ``sslmode`` wins
    when both are present. Unknown modes raise ``ValueError``.
    """
    parts = urlsplit(url)
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    kept: list[tuple[str, str]] = []
    sslmode: str | None = None
    ssl_flag: str | None = None
    rootcert: str | None = None
    for key, value in pairs:
        lowered = key.lower()
        if lowered == "sslmode":
            sslmode = value.strip().lower()
        elif lowered == "ssl":
            ssl_flag = value.strip().lower()
        elif lowered == "sslrootcert":
            rootcert = value
        else:
            kept.append((key, value))
    mode: str | None = None
    if sslmode is not None:
        if sslmode not in _SSL_MODES:
            raise ValueError(f"unsupported sslmode {sslmode!r}")
        mode = sslmode
    elif ssl_flag is not None:
        if ssl_flag in _TRUTHY:
            mode = "require"
        elif ssl_flag in _FALSY:
            mode = "disable"
        else:
            raise ValueError(f"unsupported ssl flag {ssl_flag!r}")
    clean = urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(kept), parts.fragment))
    return clean, mode, rootcert


def build_ssl(mode: str | None, cafile: str | None = None) -> SslOption | None:
    """Map a libpq ``sslmode`` to what ``asyncpg.create_pool(ssl=...)`` expects.

    ``None`` means "not specified" and lets asyncpg apply its own default (``prefer``).
    """
    if mode is None:
        return None
    mode = mode.strip().lower()
    if mode in _FALSY:
        return False
    if mode in {"allow", "prefer"}:
        return mode  # asyncpg understands the libpq names for opportunistic TLS
    if mode in _TRUTHY:
        # Encrypted but unauthenticated: Railway's postgres-ssl image is self-signed.
        ctx = ssl_module.SSLContext(ssl_module.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl_module.CERT_NONE
        return ctx
    if mode in {"verify-ca", "verify-full"}:
        ctx = ssl_module.create_default_context(cafile=cafile)
        ctx.check_hostname = mode == "verify-full"
        ctx.verify_mode = ssl_module.CERT_REQUIRED
        return ctx
    raise ValueError(f"unsupported sslmode {mode!r}")


def _aware(value: datetime | None) -> datetime | None:
    """asyncpg requires timezone-aware datetimes for ``timestamptz``. Naive means UTC here."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _dump(model: Any) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _json_default(value: Any) -> Any:
    """Encode the few non-JSON types that can appear inside event ``data`` dicts."""
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, set | frozenset | tuple):
        return list(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")


def json_encode(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


async def _init_connection(conn: asyncpg.Connection) -> None:
    for typename in ("json", "jsonb"):
        await conn.set_type_codec(
            typename,
            encoder=json_encode,
            decoder=json.loads,
            schema="pg_catalog",
            format="text",
        )


# --- SQL ------------------------------------------------------------------------------------

_UPSERT_JOB = """
INSERT INTO analyses (
    analysis_id, status, profile, resolved_horizon, query,
    created_at, updated_at, finished_at, error, job
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
ON CONFLICT (analysis_id) DO UPDATE SET
    status = EXCLUDED.status,
    profile = EXCLUDED.profile,
    resolved_horizon = EXCLUDED.resolved_horizon,
    query = EXCLUDED.query,
    updated_at = EXCLUDED.updated_at,
    finished_at = EXCLUDED.finished_at,
    error = EXCLUDED.error,
    job = EXCLUDED.job
"""

_SELECT_JOB = "SELECT job FROM analyses WHERE analysis_id = $1"

_LIST_JOBS = "SELECT job FROM analyses ORDER BY created_at DESC LIMIT $1"

_COUNT_ACTIVE = f"SELECT count(*) FROM analyses WHERE status NOT IN {_TERMINAL_SQL}"

_INSERT_EVENT = """
INSERT INTO analysis_events (analysis_id, seq, event, ts, data)
VALUES ($1, $2, $3, $4, $5)
"""

_LIST_EVENTS = """
SELECT analysis_id, seq, event, ts, data
FROM analysis_events
WHERE analysis_id = $1 AND seq > $2
ORDER BY seq
"""

_UPSERT_SOURCE = """
INSERT INTO sources (
    analysis_id, source_id, url, title, publisher, source_type,
    published_at, retrieved_at, fiscal_period, content_hash, record
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
ON CONFLICT (analysis_id, source_id) DO UPDATE SET
    url = EXCLUDED.url,
    title = EXCLUDED.title,
    publisher = EXCLUDED.publisher,
    source_type = EXCLUDED.source_type,
    published_at = EXCLUDED.published_at,
    retrieved_at = EXCLUDED.retrieved_at,
    fiscal_period = EXCLUDED.fiscal_period,
    content_hash = EXCLUDED.content_hash,
    record = EXCLUDED.record
"""

_UPSERT_FACT = """
INSERT INTO facts (
    analysis_id, fact_id, metric, value, unit, period_key, basis, source_id, record
) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
ON CONFLICT (analysis_id, fact_id) DO UPDATE SET
    metric = EXCLUDED.metric,
    value = EXCLUDED.value,
    unit = EXCLUDED.unit,
    period_key = EXCLUDED.period_key,
    basis = EXCLUDED.basis,
    source_id = EXCLUDED.source_id,
    record = EXCLUDED.record
"""

_UPSERT_DECISION = """
INSERT INTO laya_decisions (
    analysis_id, decision_id, stage, decision_type, decision, confidence, record
) VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (analysis_id, decision_id) DO UPDATE SET
    stage = EXCLUDED.stage,
    decision_type = EXCLUDED.decision_type,
    decision = EXCLUDED.decision,
    confidence = EXCLUDED.confidence,
    record = EXCLUDED.record
"""

_UPSERT_CALCULATION = """
INSERT INTO calculations (
    analysis_id, calc_id, name, value, unit, status, record
) VALUES ($1, $2, $3, $4, $5, $6, $7)
ON CONFLICT (analysis_id, calc_id) DO UPDATE SET
    name = EXCLUDED.name,
    value = EXCLUDED.value,
    unit = EXCLUDED.unit,
    status = EXCLUDED.status,
    record = EXCLUDED.record
"""

_UPSERT_RESULT = """
INSERT INTO results (analysis_id, status, completed_at, result)
VALUES ($1, $2, $3, $4)
ON CONFLICT (analysis_id) DO UPDATE SET
    status = EXCLUDED.status,
    completed_at = EXCLUDED.completed_at,
    result = EXCLUDED.result
"""

_SELECT_RESULT = "SELECT result FROM results WHERE analysis_id = $1"

_SELECT_RECORDS = "SELECT record FROM {table} WHERE analysis_id = $1 ORDER BY {order}"

_MARK_INTERRUPTED = f"""
UPDATE analyses
SET status = 'failed',
    error = $1::jsonb,
    job = job || $2::jsonb,
    finished_at = $3,
    updated_at = $3
WHERE status NOT IN {_TERMINAL_SQL}
RETURNING analysis_id
"""

_HAS_MIGRATIONS = "SELECT to_regclass('schema_migrations') IS NOT NULL"
_MAX_VERSION = "SELECT max(version) FROM schema_migrations"


class SchemaVersionError(RuntimeError):
    """The database schema is newer than this code understands."""


class PostgresStore:
    """``AnalysisStore`` on asyncpg. Call ``start()`` before use and ``close()`` at shutdown."""

    schema_version = SCHEMA_VERSION

    def __init__(
        self,
        database_url: str,
        pool_min: int = 1,
        pool_max: int = 4,
        ssl: str | None = None,
    ) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        self._database_url = database_url
        self._pool_min = max(1, pool_min)
        self._pool_max = max(self._pool_min, pool_max)
        self._ssl_override = ssl
        self._pool: asyncpg.Pool | None = None
        self.applied_version: int | None = None

    @property
    def redacted_dsn(self) -> str:
        return redact_dsn(self._database_url)

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("PostgresStore.start() has not been called")
        return self._pool

    # --- lifecycle --------------------------------------------------------------------

    async def start(self) -> None:
        if self._pool is not None:
            return
        clean_url, mode, rootcert = parse_ssl_mode(self._database_url)
        ssl_option = build_ssl(self._ssl_override or mode, cafile=rootcert)
        kwargs: dict[str, Any] = {}
        if ssl_option is not None:
            kwargs["ssl"] = ssl_option
        try:
            pool = await asyncpg.create_pool(
                clean_url,
                min_size=self._pool_min,
                max_size=self._pool_max,
                init=_init_connection,
                **kwargs,
            )
        except Exception as exc:
            logger.error(
                "postgres pool creation failed dsn=%s error=%s",
                self.redacted_dsn,
                type(exc).__name__,
            )
            raise
        if pool is None:  # pragma: no cover - asyncpg typing artefact
            raise RuntimeError("asyncpg.create_pool returned None")
        try:
            async with pool.acquire() as conn:
                await self._check_version(conn)
                async with conn.transaction():
                    await conn.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
                self.applied_version = await conn.fetchval(_MAX_VERSION)
        except Exception:
            await pool.close()
            raise
        self._pool = pool
        logger.info(
            "postgres store started dsn=%s schema_version=%s pool=%s..%s ssl=%s",
            self.redacted_dsn,
            self.applied_version,
            self._pool_min,
            self._pool_max,
            self._ssl_override or mode or "default",
        )

    async def _check_version(self, conn: asyncpg.Connection) -> None:
        if not await conn.fetchval(_HAS_MIGRATIONS):
            return
        current = await conn.fetchval(_MAX_VERSION)
        if current is not None and current > SCHEMA_VERSION:
            raise SchemaVersionError(
                f"database schema version {current} is newer than supported {SCHEMA_VERSION}"
            )

    async def close(self) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            await pool.close()
            logger.info("postgres store closed dsn=%s", self.redacted_dsn)

    # --- jobs -------------------------------------------------------------------------

    async def _upsert_job(self, job: AnalysisJob) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                _UPSERT_JOB,
                job.analysis_id,
                job.status,
                job.profile,
                job.resolved_horizon,
                job.query,
                _aware(job.created_at),
                _aware(job.updated_at),
                _aware(job.finished_at),
                _dump(job.error) if job.error is not None else None,
                _dump(job),
            )

    async def create_job(self, job: AnalysisJob) -> None:
        await self._upsert_job(job)

    async def update_job(self, job: AnalysisJob) -> None:
        await self._upsert_job(job)

    async def get_job(self, analysis_id: str) -> AnalysisJob | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(_SELECT_JOB, analysis_id)
        return AnalysisJob.model_validate(row["job"]) if row is not None else None

    async def list_jobs(self, limit: int = 50) -> list[AnalysisJob]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(_LIST_JOBS, limit)
        return [AnalysisJob.model_validate(row["job"]) for row in rows]

    async def count_active(self) -> int:
        async with self.pool.acquire() as conn:
            return int(await conn.fetchval(_COUNT_ACTIVE))

    # --- events -----------------------------------------------------------------------

    async def append_event(self, event: AnalysisEvent) -> None:
        try:
            async with self.pool.acquire() as conn:
                await conn.execute(
                    _INSERT_EVENT,
                    event.analysis_id,
                    event.seq,
                    event.event,
                    _aware(event.ts),
                    event.data,
                )
        except asyncpg.UniqueViolationError as exc:
            raise ValueError(
                f"duplicate event seq {event.seq} for analysis {event.analysis_id}"
            ) from exc

    async def list_events(self, analysis_id: str, after_seq: int = 0) -> list[AnalysisEvent]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(_LIST_EVENTS, analysis_id, after_seq)
        return [
            AnalysisEvent(
                event=row["event"],
                analysis_id=row["analysis_id"],
                seq=row["seq"],
                ts=row["ts"],
                data=row["data"] or {},
            )
            for row in rows
        ]

    # --- evidence, decisions, calculations ------------------------------------------

    async def save_sources(self, analysis_id: str, sources: list[SourceRecord]) -> None:
        if not sources:
            return
        rows = [
            (
                analysis_id,
                s.source_id,
                s.url,
                s.title,
                s.publisher,
                s.source_type,
                _aware(s.published_at),
                _aware(s.retrieved_at),
                s.fiscal_period,
                s.content_hash,
                _dump(s),
            )
            for s in sources
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(_UPSERT_SOURCE, rows)

    async def save_facts(self, analysis_id: str, facts: list[NormalizedFact]) -> None:
        if not facts:
            return
        rows = [
            (
                analysis_id,
                f.fact_id,
                f.metric,
                float(f.value),
                f.unit,
                f.period.key(),
                f.basis,
                f.source_id,
                _dump(f),
            )
            for f in facts
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(_UPSERT_FACT, rows)

    async def save_decisions(self, analysis_id: str, decisions: list[LayaDecision]) -> None:
        if not decisions:
            return
        rows = [
            (
                analysis_id,
                d.decision_id,
                d.stage,
                d.decision_type,
                str(d.decision),
                float(d.confidence),
                _dump(d),
            )
            for d in decisions
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(_UPSERT_DECISION, rows)

    async def save_calculations(
        self, analysis_id: str, calculations: list[CalculationResult]
    ) -> None:
        if not calculations:
            return
        rows = [
            (
                analysis_id,
                c.calc_id,
                c.name,
                float(c.value) if c.value is not None else None,
                c.unit,
                c.status,
                _dump(c),
            )
            for c in calculations
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(_UPSERT_CALCULATION, rows)

    async def _records(self, table: str, order: str, analysis_id: str) -> list[dict[str, Any]]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(_SELECT_RECORDS.format(table=table, order=order), analysis_id)
        return [row["record"] for row in rows]

    async def get_sources(self, analysis_id: str) -> list[SourceRecord]:
        records = await self._records("sources", "retrieved_at, source_id", analysis_id)
        return [SourceRecord.model_validate(r) for r in records]

    async def get_facts(self, analysis_id: str) -> list[NormalizedFact]:
        records = await self._records("facts", "fact_id", analysis_id)
        return [NormalizedFact.model_validate(r) for r in records]

    async def get_decisions(self, analysis_id: str) -> list[LayaDecision]:
        records = await self._records("laya_decisions", "decision_id", analysis_id)
        return [LayaDecision.model_validate(r) for r in records]

    async def get_calculations(self, analysis_id: str) -> list[CalculationResult]:
        records = await self._records("calculations", "calc_id", analysis_id)
        return [CalculationResult.model_validate(r) for r in records]

    # --- results ----------------------------------------------------------------------

    async def save_result(self, result: AnalysisResult) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                _UPSERT_RESULT,
                result.analysis_id,
                result.status,
                _aware(result.completed_at),
                _dump(result),
            )

    async def get_result(self, analysis_id: str) -> AnalysisResult | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(_SELECT_RESULT, analysis_id)
        return AnalysisResult.model_validate(row["result"]) if row is not None else None

    # --- startup recovery -------------------------------------------------------------

    async def mark_interrupted(self) -> list[str]:
        """One UPDATE that fails every non-terminal job with INTERRUPTED (section 38)."""
        now = datetime.now(tz=UTC)
        payload = _dump(interrupted_payload())
        patch = {
            "status": "failed",
            "error": payload,
            "finished_at": now.isoformat(),
            "updated_at": now.isoformat(),
        }
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(_MARK_INTERRUPTED, payload, patch, now)
        affected = [row["analysis_id"] for row in rows]
        if affected:
            logger.warning("marked %d analyses INTERRUPTED at startup", len(affected))
        return affected
