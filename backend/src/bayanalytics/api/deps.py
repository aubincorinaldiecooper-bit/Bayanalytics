"""FastAPI dependencies."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from bayanalytics.runtime import Runtime


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


RuntimeDep = Annotated[Runtime, Depends(get_runtime)]
