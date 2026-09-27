"""SearXNG JSON search client.

Adapted from the ``internetSearchSpec`` tool in GNSIS ``desktop/src/tools/registry.ts``
(Copyright (c) 2026 Aubin Cooper / Sine Studios, MIT License). The pattern carried over is
exactly the SearXNG discovery call: ``GET {base}/search?q=<query>&format=json`` returning
``{"results": [...]}``; a missing base URL is a configuration error, a non-2xx status is a
backend error. Nothing else from GNSIS is imported or depended upon (AGENT.md section 21).

MIT License text: https://opensource.org/license/mit
"""

from __future__ import annotations

import json
from typing import Any, Literal

import httpx

from bayanalytics.research.dates import parse_datetime_lenient
from bayanalytics.research.provider import ResearchProviderError, SearchResult

TimeRange = Literal["day", "week", "month", "year"]


class SearxngSearch:
    """Thin client for a SearXNG instance's JSON API. No API key is involved."""

    def __init__(self, base_url: str | None, http: httpx.AsyncClient, user_agent: str) -> None:
        self._base_url = (base_url or "").strip().rstrip("/")
        self._http = http
        self._user_agent = user_agent

    @property
    def configured(self) -> bool:
        return bool(self._base_url)

    async def search(
        self,
        query: str,
        *,
        categories: str | None = None,
        time_range: TimeRange | None = None,
        language: str | None = "en",
        max_results: int = 10,
    ) -> list[SearchResult]:
        if not query or not query.strip():
            raise ResearchProviderError("search requires a non-empty query")
        if not self._base_url:
            raise ResearchProviderError("search backend not configured")
        params: dict[str, str] = {"q": query.strip(), "format": "json"}
        if categories:
            params["categories"] = categories
        if time_range:
            params["time_range"] = time_range
        if language:
            params["language"] = language
        try:
            response = await self._http.get(
                f"{self._base_url}/search",
                params=params,
                headers={"User-Agent": self._user_agent, "Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise ResearchProviderError(f"search backend unreachable: {exc}") from exc
        if response.status_code < 200 or response.status_code >= 300:
            raise ResearchProviderError(f"search backend -> {response.status_code}")
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise ResearchProviderError("search backend returned invalid JSON") from exc
        if not isinstance(body, dict):
            raise ResearchProviderError("search backend returned an unexpected payload")
        if "error" in body and not body.get("results"):
            raise ResearchProviderError(f"search backend error: {body.get('error')}")
        raw_results = body.get("results") or []
        if not isinstance(raw_results, list):
            raise ResearchProviderError("search backend results are not a list")
        results: list[SearchResult] = []
        for item in raw_results:
            mapped = map_result(item)
            if mapped is not None:
                results.append(mapped)
            if len(results) >= max_results:
                break
        return results


def map_result(item: Any) -> SearchResult | None:
    """Map one SearXNG result object to ``SearchResult``; malformed entries return ``None``."""
    if not isinstance(item, dict):
        return None
    url = item.get("url")
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return None
    title = item.get("title")
    title_text = title.strip() if isinstance(title, str) and title.strip() else url
    snippet = item.get("content")
    engine = item.get("engine")
    score = item.get("score")
    published = item.get("publishedDate") or item.get("published_date")
    engines = item.get("engines")
    metadata: dict[str, Any] = {}
    if isinstance(engines, list):
        metadata["engines"] = [str(e) for e in engines]
    if isinstance(item.get("category"), str):
        metadata["category"] = item["category"]
    return SearchResult(
        url=url,
        title=title_text,
        snippet=snippet.strip() if isinstance(snippet, str) else "",
        engine=str(engine) if isinstance(engine, str) else None,
        published_at=parse_datetime_lenient(published),
        score=float(score) if isinstance(score, int | float) else None,
        metadata=metadata,
    )
