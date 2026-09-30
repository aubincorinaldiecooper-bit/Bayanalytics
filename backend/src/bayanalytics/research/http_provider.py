"""HTTP ``ResearchProvider`` (SearXNG + Fetcher + extract) and the research stack factory."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import httpx

from bayanalytics.config import Settings
from bayanalytics.research.edgar import EdgarClient
from bayanalytics.research.extract import extract_page
from bayanalytics.research.fetch import Fetcher, search_backend_hosts
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
    """search -> open/fetch -> extract, nothing more (AGENT.md section 21).

    ``allowed_hosts`` exempts ``host`` / ``host:port`` entries from the fetcher's private-address
    policy; by default it holds only the configured SearXNG backend, which is a legitimate
    LAN deployment (see ``fetch.Fetcher``). Everything else private stays blocked.
    """

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
        *,
        allowed_hosts: Iterable[str] | None = None,
    ) -> None:
        self.settings = settings
        self._owns_client = http_client is None
        # The client default only applies to the SearXNG call; the fetcher always sends with
        # follow_redirects=False and walks redirects itself so every hop is re-validated.
        self._http = http_client or httpx.AsyncClient(
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
            timeout=settings.research_fetch_timeout_s,
        )
        if allowed_hosts is None:
            allowed_hosts = search_backend_hosts(settings.research_search_url)
        self.fetcher = Fetcher(settings, self._http, allowed_hosts=allowed_hosts)
        self.searx = SearxngSearch(settings.research_search_url, self._http, settings.user_agent)

    @property
    def search_configured(self) -> bool:
        """Whether a search backend (``BAY_RESEARCH_SEARCH_URL``) is configured."""
        return self.searx.configured

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
        # extract_page already maps every parser failure to ResearchProviderError("extract_failed").
        return extract_page(page)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()


def build_research_stack(
    settings: Settings, http_client: httpx.AsyncClient | None = None
) -> tuple[ResearchProvider, EdgarClient, StooqPrices]:
    """Wire the HTTP provider + EDGAR + Stooq over one shared fetcher."""
    http_provider = HttpResearchProvider(
        settings,
        http_client,
        allowed_hosts=search_backend_hosts(settings.research_search_url),
    )
    return (
        http_provider,
        EdgarClient(http_provider.fetcher, settings),
        StooqPrices(http_provider.fetcher),
    )
