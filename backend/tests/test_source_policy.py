"""The owner's data-source rule (CLAUDE.md): data comes from web search only.

(a) product code holds no hard-coded URL except loopback defaults;
(b) query templates never steer the search (no ``site:``, no provider names);
(c) ``HttpResearchProvider`` refuses a URL its own search did not return;
(d) the runner only ever opens pages the search returned.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext
from bayanalytics.instruments.base import InstrumentIdentity, ResearchBudget
from bayanalytics.research.http_provider import HttpResearchProvider
from bayanalytics.research.intents import ResearchIntent, build_queries
from bayanalytics.research.provider import ResearchProviderError, SearchResult
from bayanalytics.research.runner import REJECTION_REASONS, ResearchRunner
from doubles import FixtureResearchProvider

SRC = Path(__file__).resolve().parents[1] / "src" / "bayanalytics"
FIXTURES = Path(__file__).parent / "fixtures" / "research" / "apple"
AS_OF = datetime(2026, 9, 26, tzinfo=UTC)

URL_RE = re.compile(r"https?://([A-Za-z0-9][A-Za-z0-9.\-]*|\[[0-9A-Fa-f:]+\])")
ALLOWED_HOSTS: dict[str, str] = {
    "127.0.0.1": "loopback default (local llama-server, local dev CORS origin)",
    "localhost": "loopback default (local dev CORS origin)",
}
PROVIDER_NAMES = (
    "sec.gov",
    "edgar",
    "stooq",
    "yahoo",
    "bloomberg",
    "reuters",
    "nasdaq",
    "macrotrends",
    "marketwatch",
    "seekingalpha",
    "google finance",
)


def test_product_code_has_no_hard_coded_data_urls() -> None:
    offenders: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in URL_RE.finditer(line):
                if match.group(1).lower() not in ALLOWED_HOSTS:
                    offenders.append(f"{path.relative_to(SRC.parent)}:{number}: {match.group(0)}")
    assert offenders == [], "hard-coded URLs in product code:\n" + "\n".join(offenders)


@pytest.mark.parametrize(
    "identity",
    [InstrumentIdentity(symbol="AAPL"), InstrumentIdentity(symbol="BRK.B", exchange="NYSE")],
)
def test_query_templates_never_steer_the_search(identity: InstrumentIdentity) -> None:
    gaps = ["free_cash_flow", "total_debt", "price_history"]
    for intent in ResearchIntent:
        for horizon in ("near_term", "next_cycle", "medium_term", "long_term", "multi_horizon"):
            for planned in build_queries(intent, identity, horizon, AS_OF, gaps):
                assert planned.kind == "search"
                text = (planned.query or "").lower()
                assert "site:" not in text, planned.query
                for name in PROVIDER_NAMES:
                    assert name not in text, (name, planned.query)


def _searx(*urls: str) -> httpx.MockTransport:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        if request.url.host == "searx.test":
            results = [{"url": u, "title": u, "content": "snippet"} for u in urls]
            return httpx.Response(200, json={"results": results})
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, text="<p>page</p>", headers={"content-type": "text/html"})

    transport = httpx.MockTransport(handler)
    transport.requests = requests  # type: ignore[attr-defined]
    return transport


async def test_provider_refuses_a_url_its_search_did_not_return() -> None:
    transport = _searx("https://news.test/a")
    settings = Settings(research_search_url="http://searx.test", research_min_request_interval_s=0)
    async with httpx.AsyncClient(transport=transport) as http:
        provider = HttpResearchProvider(settings, http, allowed_hosts={"searx.test:80"})
        provider.fetcher._allow_private_hosts = True  # the mock hosts do not resolve
        for url in ("https://news.test/a", "https://www.sec.gov/files/company_tickers.json"):
            with pytest.raises(ResearchProviderError) as info:
                await provider.extract(url)
            assert str(info.value).startswith("not_from_search")
            assert getattr(info.value, "reason", None) == "not_from_search"
        assert transport.requests == []  # refused before any request went out
        await provider.search("anything")
        page = await provider.open("https://news.test/a")  # a search hit: allowed
        assert page.status == 200
        with pytest.raises(ResearchProviderError, match="not_from_search"):
            await provider.open("https://news.test/b")
    assert "not_from_search" in REJECTION_REASONS
    assert not any("news.test/b" in r for r in transport.requests)


class _RecordingProvider(FixtureResearchProvider):
    """The fixture provider, recording every search hit and every page opened."""

    def __init__(self) -> None:
        super().__init__(FIXTURES)
        self.returned: set[str] = set()
        self.opened: list[str] = []

    async def search_with(self, query: str, *args: Any, **kwargs: Any) -> list[SearchResult]:
        results = await super().search_with(query, *args, **kwargs)
        self.returned.update(r.url for r in results)
        return results

    async def open(self, url: str) -> Any:
        self.opened.append(url)
        return await super().open(url)


async def test_the_runner_only_opens_pages_the_search_returned() -> None:
    provider = _RecordingProvider()
    budget = ResearchBudget(max_fetch_per_round=10, max_sources=24)
    runner = ResearchRunner(provider, Settings(), budget)
    ctx = AnalysisContext(analysis_id="an_policy", emit=_ignore)
    identity = InstrumentIdentity(symbol="AAPL")
    for intent in ResearchIntent:
        for planned in build_queries(intent, identity, "multi_horizon", AS_OF, ["total_debt"]):
            await runner.execute(planned, identity, AS_OF, ctx)
    assert provider.opened and set(provider.opened) <= provider.returned


async def _ignore(_name: str, _data: dict[str, Any]) -> None:
    return None
