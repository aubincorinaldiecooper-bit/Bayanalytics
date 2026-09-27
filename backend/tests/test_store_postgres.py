"""PostgresStore.

The database-backed tests run only when ``BAY_TEST_DATABASE_URL`` points at a reachable
Postgres (they create rows with unique ids and do not clean up). Everything else here is
always on: static validation of ``schema.sql`` and the SQL constants, and the pure DSN helpers.
"""

from __future__ import annotations

import os
import re
import ssl
from collections.abc import AsyncIterator

import pytest

from bayanalytics.config import Settings
from bayanalytics.store import (
    SCHEMA_VERSION,
    InMemoryStore,
    PostgresStore,
    SchemaVersionError,
    build_ssl,
    build_store,
    parse_ssl_mode,
    redact_dsn,
)
from bayanalytics.store import postgres as pg
from bayanalytics.store.migrate import main as migrate_main
from bayanalytics.store.migrate import resolve_database_url
from test_store_memory import (
    check_deep_copy_isolation,
    check_duplicate_seq_rejected,
    check_events,
    check_job_round_trip,
    check_list_jobs_keyset_paging,
    check_mark_interrupted,
    check_records_round_trip,
    check_result_round_trip,
    make_job,
)

TEST_DSN = os.environ.get("BAY_TEST_DATABASE_URL")
requires_postgres = pytest.mark.skipif(not TEST_DSN, reason="BAY_TEST_DATABASE_URL not set")

# --- always-on: schema.sql -------------------------------------------------------------------

_ALLOWED_PREFIXES = ("CREATE TABLE IF NOT EXISTS", "CREATE INDEX IF NOT EXISTS", "INSERT INTO")
_EXPECTED_TABLES = {
    "schema_migrations",
    "analyses",
    "analysis_events",
    "sources",
    "facts",
    "laya_decisions",
    "calculations",
    "results",
}


def schema_statements() -> list[str]:
    text = pg.SCHEMA_PATH.read_text(encoding="utf-8")
    without_comments = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("--")
    )
    return [s.strip() for s in without_comments.split(";") if s.strip()]


def test_schema_sql_is_idempotent() -> None:
    statements = schema_statements()
    assert statements, "schema.sql has no statements"
    for statement in statements:
        upper = " ".join(statement.upper().split())
        assert upper.startswith(_ALLOWED_PREFIXES), statement[:60]
        if upper.startswith("INSERT INTO"):
            assert "ON CONFLICT" in upper, statement[:60]


def test_schema_sql_tables_indexes_and_version() -> None:
    statements = schema_statements()
    tables = {
        m.group(1)
        for s in statements
        if (m := re.match(r"CREATE TABLE IF NOT EXISTS (\w+)", s, re.IGNORECASE))
    }
    assert tables == _EXPECTED_TABLES
    indexes = {
        m.group(1)
        for s in statements
        if (m := re.match(r"CREATE INDEX IF NOT EXISTS \w+ ON (\w+\s*\([^)]*\))", s, re.I))
    }
    normalised = {" ".join(i.lower().split()) for i in indexes}
    assert "analyses (status)" in normalised
    assert "analyses (created_at desc)" in normalised
    assert "sources (content_hash)" in normalised
    inserts = [s for s in statements if s.upper().startswith("INSERT INTO SCHEMA_MIGRATIONS")]
    assert len(inserts) == 1
    versions = [int(v) for v in re.findall(r"VALUES\s*\(\s*(\d+)\s*\)", inserts[0])]
    assert versions == [SCHEMA_VERSION]
    # every child table cascades from analyses
    for statement in statements:
        m = re.match(r"CREATE TABLE IF NOT EXISTS (\w+)", statement, re.IGNORECASE)
        if m and m.group(1) not in {"schema_migrations", "analyses"}:
            assert "REFERENCES analyses (analysis_id) ON DELETE CASCADE" in statement, m.group(1)


def _placeholders(sql: str) -> set[int]:
    return {int(n) for n in re.findall(r"\$(\d+)", sql)}


def test_sql_constants_parameter_counts() -> None:
    """Every INSERT lists as many columns as placeholders; placeholders are 1..n contiguous."""
    inserts = {
        name: value
        for name, value in vars(pg).items()
        if name.startswith("_UPSERT") or name == "_INSERT_EVENT"
    }
    assert len(inserts) == 7
    for name, sql in inserts.items():
        m = re.search(r"INSERT INTO \w+\s*\(([^)]*)\)\s*VALUES\s*\(([^)]*)\)", sql, re.S)
        assert m, name
        columns = [c.strip() for c in m.group(1).split(",") if c.strip()]
        values = [v.strip() for v in m.group(2).split(",") if v.strip()]
        assert len(columns) == len(values), name
        assert _placeholders(sql) == set(range(1, len(columns) + 1)), name
        if "ON CONFLICT" in sql:
            assert "DO UPDATE SET" in sql or "DO NOTHING" in sql, name
            for column in re.findall(r"(\w+) = EXCLUDED\.(\w+)", sql):
                assert column[0] == column[1], name
                assert column[0] in columns, (name, column)
    assert _placeholders(pg._MARK_INTERRUPTED) == {1, 2, 3}
    assert "RETURNING analysis_id" in pg._MARK_INTERRUPTED
    assert "NOT IN ('completed', 'failed', 'cancelled')" in pg._MARK_INTERRUPTED
    assert "job || $2::jsonb" in pg._MARK_INTERRUPTED
    assert _placeholders(pg._LIST_EVENTS) == {1, 2} and "ORDER BY seq" in pg._LIST_EVENTS
    assert _placeholders(pg._SELECT_JOB) == {1}
    assert _placeholders(pg._SELECT_RESULT) == {1}
    assert _placeholders(pg._COUNT_ACTIVE) == set()
    assert _placeholders(pg._LIST_JOBS) == {1}
    assert _placeholders(pg._LIST_JOBS_BEFORE) == {1, 2, 3}
    for sql in (pg._LIST_JOBS, pg._LIST_JOBS_BEFORE):
        assert "ORDER BY created_at DESC, analysis_id DESC" in " ".join(sql.split())


# --- always-on: DSN helpers ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "postgresql://user:s3cret@db.railway.internal:5432/railway?sslmode=require",
            "postgresql://user:***@db.railway.internal:5432/railway?sslmode=require",
        ),
        ("postgres://user@host/db", "postgres://user@host/db"),
        ("postgresql://host:5432/db", "postgresql://host:5432/db"),
        (
            "postgresql://u:p%40ss@[::1]:5433/db?password=leak&ssl=true",
            "postgresql://u:***@[::1]:5433/db?password=%2A%2A%2A&ssl=true",
        ),
        (
            "host=localhost user=bay password=hunter2 dbname=bay",
            "host=localhost user=bay password=*** dbname=bay",
        ),
        ("", "<none>"),
        (None, "<none>"),
    ],
)
def test_redact_dsn(url: str | None, expected: str) -> None:
    redacted = redact_dsn(url)
    assert redacted == expected
    for secret in ("s3cret", "p%40ss", "hunter2", "leak"):
        assert secret not in redacted


def test_parse_ssl_mode() -> None:
    clean, mode, cafile = parse_ssl_mode(
        "postgresql://u:p@h:5432/db?sslmode=require&application_name=bay"
    )
    assert clean == "postgresql://u:p@h:5432/db?application_name=bay"
    assert mode == "require" and cafile is None

    clean, mode, _ = parse_ssl_mode("postgresql://u:p@h/db?ssl=true")
    assert (clean, mode) == ("postgresql://u:p@h/db", "require")

    clean, mode, _ = parse_ssl_mode("postgresql://u:p@h/db?ssl=false")
    assert (clean, mode) == ("postgresql://u:p@h/db", "disable")

    clean, mode, _ = parse_ssl_mode("postgresql://u:p@h/db")
    assert (clean, mode) == ("postgresql://u:p@h/db", None)

    # sslmode wins over ssl=, sslrootcert is captured for verify modes
    clean, mode, cafile = parse_ssl_mode(
        "postgresql://u:p@h/db?ssl=false&sslmode=verify-full&sslrootcert=/etc/ca.pem"
    )
    assert (clean, mode, cafile) == ("postgresql://u:p@h/db", "verify-full", "/etc/ca.pem")

    with pytest.raises(ValueError):
        parse_ssl_mode("postgresql://u:p@h/db?sslmode=bogus")
    with pytest.raises(ValueError):
        parse_ssl_mode("postgresql://u:p@h/db?ssl=maybe")


def test_build_ssl() -> None:
    assert build_ssl(None) is None
    assert build_ssl("disable") is False
    assert build_ssl("prefer") == "prefer"
    assert build_ssl("allow") == "allow"

    require = build_ssl("require")
    assert isinstance(require, ssl.SSLContext)
    assert require.check_hostname is False
    assert require.verify_mode is ssl.CERT_NONE

    verify_ca = build_ssl("verify-ca")
    assert isinstance(verify_ca, ssl.SSLContext)
    assert verify_ca.check_hostname is False
    assert verify_ca.verify_mode is ssl.CERT_REQUIRED

    verify_full = build_ssl("verify-full")
    assert isinstance(verify_full, ssl.SSLContext)
    assert verify_full.check_hostname is True
    assert verify_full.verify_mode is ssl.CERT_REQUIRED

    with pytest.raises(ValueError):
        build_ssl("bogus")


def test_build_store_selects_backend() -> None:
    assert isinstance(build_store(Settings()), InMemoryStore)
    store = build_store(Settings(database_url="postgresql://u:s3cret@h/db?sslmode=require"))
    assert isinstance(store, PostgresStore)
    assert store.schema_version == SCHEMA_VERSION
    assert "s3cret" not in store.redacted_dsn
    assert "s3cret" not in repr(store.redacted_dsn)
    with pytest.raises(RuntimeError):
        _ = store.pool  # not started
    with pytest.raises(ValueError):
        PostgresStore("")


def test_migrate_cli_requires_url(capsys: pytest.CaptureFixture[str]) -> None:
    assert resolve_database_url({}) is None
    assert resolve_database_url({"DATABASE_URL": "a"}) == "a"
    assert resolve_database_url({"DATABASE_URL": "a", "BAY_DATABASE_URL": "b"}) == "b"
    assert migrate_main([], env={}) == 2
    assert "not set" in capsys.readouterr().err


def test_migrate_cli_reports_failure_without_leaking_dsn(
    capsys: pytest.CaptureFixture[str],
) -> None:
    url = "postgresql://bay:s3cret@127.0.0.1:1/bay?sslmode=disable"
    assert migrate_main([], env={"DATABASE_URL": url}) == 1
    captured = capsys.readouterr()
    assert "s3cret" not in captured.out + captured.err
    assert "FAILED" in captured.err
    assert "bay:***@127.0.0.1:1" in captured.out


# --- database-backed ---------------------------------------------------------------------------


@pytest.fixture
async def pg_store() -> AsyncIterator[PostgresStore]:
    assert TEST_DSN
    store = PostgresStore(TEST_DSN, pool_min=1, pool_max=2)
    await store.start()
    try:
        yield store
    finally:
        await store.close()


@requires_postgres
async def test_pg_start_is_idempotent() -> None:
    assert TEST_DSN
    store = PostgresStore(TEST_DSN, pool_min=1, pool_max=1)
    await store.start()
    assert store.applied_version == SCHEMA_VERSION
    await store.close()
    await store.start()  # schema applied again without error
    assert store.applied_version == SCHEMA_VERSION
    await store.close()
    await store.close()  # closing twice is harmless


@requires_postgres
async def test_pg_job_round_trip(pg_store: PostgresStore) -> None:
    await check_job_round_trip(pg_store)


@requires_postgres
async def test_pg_isolation(pg_store: PostgresStore) -> None:
    await check_deep_copy_isolation(pg_store)


@requires_postgres
async def test_pg_list_jobs_keyset_paging(pg_store: PostgresStore) -> None:
    await check_list_jobs_keyset_paging(pg_store)


@requires_postgres
async def test_pg_events(pg_store: PostgresStore) -> None:
    await check_events(pg_store)


@requires_postgres
async def test_pg_duplicate_seq_rejected(pg_store: PostgresStore) -> None:
    await check_duplicate_seq_rejected(pg_store)


@requires_postgres
async def test_pg_result_round_trip(pg_store: PostgresStore) -> None:
    await check_result_round_trip(pg_store)


@requires_postgres
async def test_pg_records_round_trip(pg_store: PostgresStore) -> None:
    await check_records_round_trip(pg_store)


@requires_postgres
async def test_pg_mark_interrupted(pg_store: PostgresStore) -> None:
    await check_mark_interrupted(pg_store)


@requires_postgres
async def test_pg_count_active_and_list(pg_store: PostgresStore) -> None:
    before = await pg_store.count_active()
    job = make_job(status="researching")
    await pg_store.create_job(job)
    assert await pg_store.count_active() == before + 1
    jobs = await pg_store.list_jobs(limit=5)
    assert jobs and jobs[0].analysis_id == job.analysis_id
    job.status = "completed"
    await pg_store.update_job(job)
    assert await pg_store.count_active() == before


@requires_postgres
async def test_pg_refuses_a_database_from_newer_code() -> None:
    import asyncpg

    assert TEST_DSN
    store = PostgresStore(TEST_DSN, pool_min=1, pool_max=1)
    await store.start()
    newer = SCHEMA_VERSION + 1
    clean_url, mode, rootcert = parse_ssl_mode(TEST_DSN)
    ssl_option = build_ssl(mode, cafile=rootcert)
    connect_kwargs = {} if ssl_option is None else {"ssl": ssl_option}
    try:
        await store.pool.execute("INSERT INTO schema_migrations (version) VALUES ($1)", newer)
        await store.close()
        with pytest.raises(SchemaVersionError, match=str(newer)):
            await store.start()
        assert store._pool is None  # nothing left open after the refusal
        with pytest.raises(RuntimeError):
            _ = store.pool
    finally:
        # start() would refuse too, so clean up over a raw connection.
        conn = await asyncpg.connect(clean_url, **connect_kwargs)
        try:
            await conn.execute("DELETE FROM schema_migrations WHERE version = $1", newer)
        finally:
            await conn.close()
        await store.close()
    await store.start()  # back to normal once the newer row is gone
    assert store.applied_version == SCHEMA_VERSION
    await store.close()
