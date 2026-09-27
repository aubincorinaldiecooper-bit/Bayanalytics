"""HTTP ``ResearchProvider`` (SearXNG + Fetcher + extract) and the research stack factory."""

from __future__ import annotations

from typing import Literal

import httpx

from bayanalytics.config import Settings
from bayanalytics.research.edgar import EdgarClient
from bayanalytics.research.extract import extract_page
from bayanalytics.research.fetch import Fetcher
from bayanalytics.research.fixture_provider import FixtureFetcher, FixtureResearchProvider
from bayanalytics.research.prices import StooqPrices
from bayanalytics.research.provider import (
    EvidenceRecord,
    PageResult,
    ResearchProvider,
    SearchResult,
)
from bayanalytics.research.searxng import SearxngSearch

TimeRange = Literal["day", "week", "month", "year"]


class HttpResearchProvider:
    """search -> open/fetch -> extract, nothing more (AGENT.md section 21)."""

    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self._owns_client = http_client is None
        self._http = http_client or httpx.AsyncClient(
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
            timeout=settings.research_fetch_timeout_s,
        )
        self.fetcher = Fetcher(settings, self._http)
        self.searx = SearxngSearch(settings.research_search_url, self._http, settings.user_agent)

    async def search(self, query: str) -> list[SearchResult]:
        return await self.search_with(query)

    async def search_with(
        self,
        query: str,
        categories: str | None = None,
        time_range: TimeRange | None = None,
        max_results: int = 10,
    ) -> list[SearchResult]:
        return await self.searx.search(
            query, categories=categories, time_range=time_range, max_results=max_results
        )

    async def open(self, url: str) -> PageResult:
        return await self.fetcher.open(url)

    async def extract(self, url: str) -> EvidenceRecord:
        page = await self.open(url)
        return extract_page(page)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()


def build_research_stack(
    settings: Settings, http_client: httpx.AsyncClient | None = None
) -> tuple[ResearchProvider, EdgarClient, StooqPrices]:
    """Wire provider + EDGAR + Stooq for ``settings.research_provider`` ("http" or "fixture")."""
    if settings.research_provider == "fixture":
        if settings.research_fixture_dir is None:
            raise ValueError("research_provider=fixture requires research_fixture_dir")
        fetcher = FixtureFetcher(settings.research_fixture_dir)
        provider: ResearchProvider = FixtureResearchProvider(
            settings.research_fixture_dir, fetcher=fetcher
        )
        return provider, EdgarClient(fetcher, settings), StooqPrices(fetcher)
    http_provider = HttpResearchProvider(settings, http_client)
    return (
        http_provider,
        EdgarClient(http_provider.fetcher, settings),
        StooqPrices(http_provider.fetcher),
    )
