"""Apply ``schema.sql`` to the configured database.

Usage::

    DATABASE_URL=postgresql://... python -m bayanalytics.store.migrate

Reads ``BAY_DATABASE_URL`` first, then ``DATABASE_URL`` (managed providers set the latter).
Prints the redacted DSN and the applied migration version. Exit status is non-zero on any
failure; the raw DSN never appears in the output.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Mapping

from bayanalytics.store.postgres import SCHEMA_VERSION, PostgresStore, redact_dsn


def resolve_database_url(env: Mapping[str, str] | None = None) -> str | None:
    e = os.environ if env is None else env
    return e.get("BAY_DATABASE_URL") or e.get("DATABASE_URL") or None


async def migrate(database_url: str, ssl: str | None = None) -> int:
    store = PostgresStore(database_url, pool_min=1, pool_max=1, ssl=ssl)
    await store.start()
    try:
        return int(store.applied_version or 0)
    finally:
        await store.close()


def _scrub(text: str, url: str) -> str:
    return text.replace(url, redact_dsn(url))


def main(argv: list[str] | None = None, env: Mapping[str, str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    ssl: str | None = None
    if "--ssl" in args:
        idx = args.index("--ssl")
        if idx + 1 >= len(args):
            print("usage: python -m bayanalytics.store.migrate [--ssl MODE]", file=sys.stderr)
            return 2
        ssl = args[idx + 1]
    url = resolve_database_url(env)
    if not url:
        print("migrate: DATABASE_URL (or BAY_DATABASE_URL) is not set", file=sys.stderr)
        return 2
    print(f"migrate: database {redact_dsn(url)} (code schema version {SCHEMA_VERSION})")
    try:
        version = asyncio.run(migrate(url, ssl=ssl))
    except Exception as exc:
        print(
            f"migrate: FAILED {type(exc).__name__}: {_scrub(str(exc), url)}",
            file=sys.stderr,
        )
        return 1
    print(f"migrate: ok, schema version {version}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
