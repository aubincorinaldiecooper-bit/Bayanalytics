"""Health and capabilities (AGENT.md 37.6). Reports capabilities only; never paths or secrets."""

from __future__ import annotations

from fastapi import APIRouter

from bayanalytics import __version__
from bayanalytics.api.deps import RuntimeDep
from bayanalytics.schemas.capabilities import Capabilities, Health

router = APIRouter(tags=["system"])


@router.get("/health", response_model=Health)
async def health(rt: RuntimeDep) -> Health:
    return await rt.health(__version__)


@router.get("/capabilities", response_model=Capabilities)
async def capabilities(rt: RuntimeDep) -> Capabilities:
    return rt.capabilities()
