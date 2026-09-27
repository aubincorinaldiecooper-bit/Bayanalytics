"""Command line entry points: serve, migrate, capabilities."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from bayanalytics import __version__
from bayanalytics.config import Settings, set_settings


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from bayanalytics.main import create_app

    settings = Settings.from_env()
    if args.host:
        settings = settings.model_copy(update={"host": args.host})
    if args.port:
        settings = settings.model_copy(update={"port": args.port})
    set_settings(settings)
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        # Open SSE streams must not hold a stop for the length of a Deep synthesis: after this
        # grace the lifespan shuts down and running analyses are reported INTERRUPTED.
        timeout_graceful_shutdown=settings.graceful_shutdown_s,
    )
    return 0


def _migrate(_args: argparse.Namespace) -> int:
    from bayanalytics.store.migrate import main as migrate_main

    return int(migrate_main() or 0)


def _capabilities(_args: argparse.Namespace) -> int:
    from bayanalytics.wiring import build_runtime

    settings = Settings.from_env()
    rt = build_runtime(settings)
    print(json.dumps(rt.capabilities().model_dump(mode="json"), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bayanalytics", description="BayAnalytics backend")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="run the FastAPI backend (loopback by default)")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=_serve)
    migrate = sub.add_parser("migrate", help="apply the Postgres schema")
    migrate.set_defaults(func=_migrate)
    caps = sub.add_parser("capabilities", help="print capabilities without starting the server")
    caps.set_defaults(func=_capabilities)
    args = parser.parse_args(argv)
    result = args.func(args)
    if asyncio.iscoroutine(result):
        result = asyncio.run(result)
    return int(result or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
