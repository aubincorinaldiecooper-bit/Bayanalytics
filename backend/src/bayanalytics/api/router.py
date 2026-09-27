"""Versioned API router (``/api/v1``)."""

from __future__ import annotations

from fastapi import APIRouter

from bayanalytics.api import routes_analyses, routes_system, routes_transcriptions


def build_router(prefix: str = "/api/v1") -> APIRouter:
    router = APIRouter(prefix=prefix)
    router.include_router(routes_system.router)
    router.include_router(routes_analyses.router)
    router.include_router(routes_transcriptions.router)
    return router
