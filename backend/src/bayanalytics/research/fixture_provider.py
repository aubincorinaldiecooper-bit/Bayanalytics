"""Fixture-backed research provider and fetcher, plus recorders that capture live responses.

Fixture layout (one directory per scenario, e.g. ``tests/fixtures/research/apple``)::

    pages.json     {"fixture": true, "pages": {"<url>": {"status": 200, "content_type": "...",
                    "body_file": "relative/path", "final_url": "...", "paywalled": false}}}
    searches.json  {"fixture": true, "searches": {"<query>": [SearXNG-shaped result, ...],
                    "*": [...]}}
    <body files>   HTML / JSON / CSV bodies referenced from pages.json

URL keys are matched after ``canonical_url`` so tracking parameters and trailing slashes do
not matter. Query keys are matched case-insensitively with collapsed whitespace; ``"*"`` is
the fallback result list.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from bayanalytics.research.extract import extract_page
from bayanalytics.research.fetch import PAYWALL_HEADER, PAYWALL_STATUSES, PageFetcher
from bayanalytics.research.provider import (
    EvidenceRecord,
    PageResult,
    ResearchProvider,
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

    def __init__(self, fixture_dir: Path | str, fetcher: FixtureFetcher | None = None) -> None:
        self.dir = Path(fixture_dir)
        self.fetcher = fetcher or FixtureFetcher(self.dir)
        payload = _load_json(self.dir / SEARCHES_FILE)
        searches = payload.get("searches") if isinstance(payload.get("searches"), dict) else {}
        self._searches: dict[str, list[Any]] = {
            (_normalise_query(q) if q != "*" else "*"): (items if isinstance(items, list) else [])
            for q, items in searches.items()
        }
        self.queries: list[str] = []

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
        items = self._searches.get(_normalise_query(query))
        if items is None:
            items = self._searches.get("*", [])
        results = [r for r in (map_result(item) for item in items) if r is not None]
        return results[:max_results]

    async def open(self, url: str) -> PageResult:
        return await self.fetcher.open(url)

    async def extract(self, url: str) -> EvidenceRecord:
        return extract_page(await self.open(url))


# --------------------------------------------------------------------------------------
# recording (capture live responses into the fixture layout)
# --------------------------------------------------------------------------------------


class FixtureWriter:
    def __init__(self, out_dir: Path | str) -> None:
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _update(self, file_name: str, key: str, update: dict[str, Any] | list[Any]) -> None:
        path = self.dir / file_name
        payload = _load_json(path)
        payload.setdefault("fixture", True)
        section = "pages" if file_name == PAGES_FILE else "searches"
        bucket = payload.setdefault(section, {})
        bucket[key] = update
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    def record_page(self, url: str, page: PageResult) -> None:
        key = canonical_url(url)
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        ctype = (page.content_type or "").lower()
        ext = "json" if "json" in ctype else "csv" if "csv" in ctype else "html"
        body_dir = self.dir / "recorded"
        body_dir.mkdir(parents=True, exist_ok=True)
        body_file = f"recorded/{digest}.{ext}"
        (self.dir / body_file).write_text(page.body, encoding="utf-8")
        entry: dict[str, Any] = {
            "status": page.status,
            "content_type": page.content_type,
            "final_url": page.final_url,
            "body_file": body_file,
            "recorded_at": page.fetched_at.isoformat(),
        }
        if page.headers.get(PAYWALL_HEADER) == "true":
            entry["paywalled"] = True
        self._update(PAGES_FILE, key, entry)

    def record_search(self, query: str, results: list[SearchResult]) -> None:
        items = [
            {
                "url": r.url,
                "title": r.title,
                "content": r.snippet,
                "engine": r.engine,
                "publishedDate": r.published_at.isoformat() if r.published_at else None,
                "score": r.score,
            }
            for r in results
        ]
        self._update(SEARCHES_FILE, query, items)


class RecordingFetcher:
    def __init__(self, inner: PageFetcher, out_dir: Path | str) -> None:
        self.inner = inner
        self.writer = FixtureWriter(out_dir)

    async def open(
        self, url: str, *, ttl_s: float | None = None, accept: str | None = None
    ) -> PageResult:
        page = await self.inner.open(url, ttl_s=ttl_s, accept=accept)
        self.writer.record_page(url, page)
        return page


class RecordingProvider:
    """Wrap a live provider and write every search/page into ``out_dir`` in fixture layout."""

    def __init__(self, inner: ResearchProvider, out_dir: Path | str) -> None:
        self.inner = inner
        self.writer = FixtureWriter(out_dir)

    async def search(self, query: str) -> list[SearchResult]:
        results = await self.inner.search(query)
        self.writer.record_search(query, results)
        return results

    async def search_with(
        self,
        query: str,
        categories: str | None = None,
        time_range: TimeRange | None = None,
        max_results: int = 10,
    ) -> list[SearchResult]:
        search_with = getattr(self.inner, "search_with", None)
        if callable(search_with):
            results = await search_with(
                query, categories=categories, time_range=time_range, max_results=max_results
            )
        else:
            results = await self.inner.search(query)
        self.writer.record_search(query, results)
        return results

    async def open(self, url: str) -> PageResult:
        page = await self.inner.open(url)
        self.writer.record_page(url, page)
        return page

    async def extract(self, url: str) -> EvidenceRecord:
        return extract_page(await self.open(url))
