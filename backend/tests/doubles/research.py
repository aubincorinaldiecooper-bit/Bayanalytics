"""Fixture-backed research provider and fetcher (tests only).

Fixture layout (one directory per scenario, e.g. ``tests/fixtures/research/apple``)::

    pages.json     {"fixture": true, "pages": {"<url>": {"status": 200, "content_type": "...",
                    "body_file": "relative/path", "final_url": "...", "paywalled": false}}}
    searches.json  {"fixture": true, "searches": {"<query>": [SearXNG-shaped result, ...],
                    "*": [...]}}
    <body files>   HTML bodies referenced from pages.json (web pages a search returns)

URL keys are matched after ``canonical_url`` so tracking parameters and trailing slashes do
not matter. Query keys are matched case-insensitively with collapsed whitespace; ``"*"`` is
the fallback result list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from bayanalytics.research.extract import extract_page
from bayanalytics.research.fetch import PAYWALL_HEADER, PAYWALL_STATUSES
from bayanalytics.research.provider import (
    EvidenceRecord,
    PageResult,
    ResearchProviderError,
    SearchResult,
)
from bayanalytics.research.searxng import map_result
from bayanalytics.research.sources import canonical_url
from bayanalytics.schemas.common import utcnow

TimeRange = Literal["day", "week", "month", "year"]
PAGES_FILE = "pages.json"
SEARCHES_FILE = "searches.json"


def _normalise_query(query: str) -> str:
    return " ".join(query.lower().split())


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ResearchProviderError(f"fixture file {path} is not valid JSON") from exc
    return payload if isinstance(payload, dict) else {}


class FixtureFetcher:
    """Serve ``PageResult`` objects for URLs listed in ``pages.json``."""

    def __init__(self, fixture_dir: Path | str) -> None:
        self.dir = Path(fixture_dir)
        payload = _load_json(self.dir / PAGES_FILE)
        pages = payload.get("pages") if isinstance(payload.get("pages"), dict) else {}
        self._pages: dict[str, dict[str, Any]] = {
            canonical_url(url): entry for url, entry in pages.items() if isinstance(entry, dict)
        }
        self.calls: list[str] = []

    def has(self, url: str) -> bool:
        return canonical_url(url) in self._pages

    async def open(
        self, url: str, *, ttl_s: float | None = None, accept: str | None = None
    ) -> PageResult:
        self.calls.append(url)
        entry = self._pages.get(canonical_url(url))
        if entry is None:
            raise ResearchProviderError(f"no fixture page for {url}")
        status = int(entry.get("status", 200))
        content_type = entry.get("content_type") or "text/html; charset=utf-8"
        headers: dict[str, str] = {"content-type": content_type}
        body = ""
        if status in PAYWALL_STATUSES or entry.get("paywalled"):
            headers[PAYWALL_HEADER] = "true"
        elif status >= 400:
            raise ResearchProviderError(f"HTTP {status} for {url}")
        else:
            body_file = entry.get("body_file")
            if body_file:
                body = (self.dir / body_file).read_text(encoding="utf-8")
            else:
                body = str(entry.get("body", ""))
        return PageResult(
            url=url,
            final_url=str(entry.get("final_url") or url),
            status=status,
            content_type=content_type,
            body=body,
            fetched_at=utcnow(),
            headers=headers,
        )


class FixtureResearchProvider:
    """``ResearchProvider`` reading searches from ``searches.json`` and pages via FixtureFetcher."""

    def __init__(
        self,
        fixture_dir: Path | str,
        fetcher: FixtureFetcher | None = None,
        *,
        search_configured: bool = True,
        search_error: str | None = None,
    ) -> None:
        self.dir = Path(fixture_dir)
        # The fixture answers searches, so it counts as a configured search backend unless a
        # test says otherwise; ``search_error`` makes every search fail like a backend outage.
        self.search_configured = search_configured
        self.search_error = search_error
        self.fetcher = fetcher or FixtureFetcher(self.dir)
        payload = _load_json(self.dir / SEARCHES_FILE)
        searches = payload.get("searches") if isinstance(payload.get("searches"), dict) else {}
        self._searches: dict[str, list[Any]] = {
            (_normalise_query(q) if q != "*" else "*"): (items if isinstance(items, list) else [])
            for q, items in searches.items()
        }
        self.queries: list[str] = []
        self.closed = False

    async def search(self, query: str) -> list[SearchResult]:
        return await self.search_with(query)

    async def search_with(
        self,
        query: str,
        categories: str | None = None,
        time_range: TimeRange | None = None,
        max_results: int = 10,
    ) -> list[SearchResult]:
        self.queries.append(query)
        if self.search_error is not None:
            raise ResearchProviderError(self.search_error)
        items = self._searches.get(_normalise_query(query))
        if items is None:
            items = self._searches.get("*", [])
        results = [r for r in (map_result(item) for item in items) if r is not None]
        return results[:max_results]

    async def open(self, url: str) -> PageResult:
        return await self.fetcher.open(url)

    async def extract(self, url: str) -> EvidenceRecord:
        return extract_page(await self.open(url))

    async def aclose(self) -> None:
        """Nothing to release; recorded so tests can check the runtime closes its provider."""
        self.closed = True
