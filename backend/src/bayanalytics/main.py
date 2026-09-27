"""FastAPI application factory with lifespan-managed local runtimes (AGENT.md section 36)."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bayanalytics import __version__
from bayanalytics.api.errors import install_error_handlers
from bayanalytics.api.limits import BodyLimitMiddleware
from bayanalytics.api.router import build_router
from bayanalytics.config import Settings, get_settings
from bayanalytics.runtime import Runtime

log = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        rt = runtime
        if rt is None:
            from bayanalytics.wiring import build_runtime

            rt = build_runtime(settings)
        app.state.runtime = rt
        if not settings.is_loopback and not settings.api_key:
            raise RuntimeError(
                "refusing to bind a non-loopback host without BAY_API_KEY "
                "(AGENT.md section 20: never expose the API publicly without authentication)"
            )
        log.info(
            "starting BayAnalytics backend %s with settings %s", __version__, settings.redacted()
        )
        try:
            await rt.start()
        except Exception as exc:
            # One clear line for the operator; uvicorn prints the traceback after it.
            log.error("startup aborted: %s", exc)
            raise
        try:
            yield
        finally:
            await rt.close()

    app = FastAPI(
        title="BayAnalytics",
        version=__version__,
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    app.add_middleware(
        BodyLimitMiddleware,
        default_limit=settings.max_request_body_bytes,
        upload_limit=settings.max_upload_bytes,
        upload_path_suffix="/transcriptions",
    )
    install_error_handlers(app)
    app.include_router(build_router(settings.api_prefix))
    return app


app = None  # created lazily by ``bayanalytics serve``; import ``create_app`` in tests
