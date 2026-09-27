"""Research-layer hardening: SSRF target policy, hostile HTML, client-safe reasons, cache policy.

No network: every HTTP call goes through ``httpx.MockTransport`` and the DNS resolver is
replaced by a static table so the tests are deterministic and offline.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from bayanalytics.config import Settings
from bayanalytics.context import AnalysisContext
from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.base import InstrumentIdentity, ResearchBudget
from bayanalytics.research import fetch
from bayanalytics.research.extract import (
    MAX_DOM_DEPTH,
    HtmlTooDeepError,
    extract_html,
    extract_page,
)
from bayanalytics.research.fetch import (
    CACHE_MAX_AGE_S,
    MAX_REDIRECTS,
    Fetcher,
    TaggedProviderError,
    address_blocked,
    search_backend_hosts,
)
from bayanalytics.research.http_provider import HttpResearchProvider, build_research_stack
from bayanalytics.research.intents import ResearchIntent, build_queries
from bayanalytics.research.provider import PageResult, ResearchProviderError
from bayanalytics.research.runner import REJECTION_REASONS, ResearchRunner, reject_reason
from bayanalytics.schemas.common import ErrorCode
from doubles import fixture_research_stack

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "research" / "apple"
AS_OF = datetime(2026, 9, 26, tzinfo=UTC)
NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)
IDENTITY = InstrumentIdentity(
    symbol="AAPL", exchange="NASDAQ", name="Apple Inc.", cik="320193", sic="3571"
)
PUBLIC_IP = "93.184.216.34"
RESOLVER_TABLE: dict[str, list[str]] = {
    "public.example": [PUBLIC_IP, "2606:2800:220:1:248:1893:25c8:1946"],
    "news.example": [PUBLIC_IP],
    "searx.example": [PUBLIC_IP],
    "www.reuters.com": [PUBLIC_IP],
    "data.sec.gov": [PUBLIC_IP],
    "www.sec.gov": [PUBLIC_IP],
    "stooq.com": [PUBLIC_IP],
    "intranet.example": ["10.0.0.5"],
    "dual.example": [PUBLIC_IP, "192.168.1.9"],  # one private answer is enough to block
    "meta.example": ["169.254.169.254"],
    "localhost": ["127.0.0.1"],
    "searx.lan": ["192.168.1.20"],
}
PARAGRAPH = (
    "Fixture prose about quarterly results, guidance and the outlook for the business with "
    "enough words to clear the thin-content threshold comfortably."
)
GOOD_HTML = (
    "<html><head><title>Good page</title></head><body><main><article>"
    + "".join(f"<p>{PARAGRAPH} Paragraph {i}.</p>" for i in range(4))
    + "</article></main></body></html>"
)


def deep_html(depth: int) -> str:
    return (
        "<html><body><main>"
        + "<div>" * depth
        + f"<p>{PARAGRAPH}</p>"
        + "</div>" * depth
        + "</main></body></html>"
    )


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "research_contact_email": "dev@example.com",
        "research_min_request_interval_s": 0.0,
        "research_fetch_timeout_s": 5.0,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def mock_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def page(url: str, body: str, ctype: str | None = "text/html; charset=utf-8") -> PageResult:
    return PageResult(
        url=url, final_url=url, status=200, content_type=ctype, body=body, fetched_at=NOW
    )


def json_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.json"))


class RecordingCtx:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.ctx = AnalysisContext(analysis_id="an_hardening", emit=self.sink)

    async def sink(self, name: str, data: dict[str, Any]) -> None:
        self.events.append((name, data))

    def named(self, name: str) -> list[dict[str, Any]]:
        return [data for event, data in self.events if event == name]


@pytest.fixture
def resolver(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Replace DNS with a static table; hosts missing from it do not resolve."""
    table = dict(RESOLVER_TABLE)

    async def fake_resolve(host: str) -> list[str]:
        literal = fetch._literal_address(host)
        return [literal] if literal else list(table.get(host, []))

    monkeypatch.setattr(fetch, "resolve_host", fake_resolve)
    return table


class Recorder:
    """Mock transport that records every request and serves a small routing table."""

    def __init__(self, routes: dict[str, Callable[[httpx.Request], httpx.Response]] | None = None):
        self.requests: list[str] = []
        self.routes = routes or {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        route = self.routes.get(request.url.path)
        if route is not None:
            return route(request)
        return httpx.Response(200, text="<p>ok</p>", headers={"content-type": "text/html"})

    def pages(self) -> list[str]:
        return [u for u in self.requests if not u.endswith("/robots.txt")]


# --------------------------------------------------------------------------------------
# (a) target policy
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8081/props",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/",
        "http://172.16.4.4/",
        "http://192.168.1.1/",
        "http://100.64.0.1/",
        "http://0.0.0.0/",
        "http://[::1]/",
        "http://[::ffff:127.0.0.1]/",
        "http://[fe80::1]/",
        "http://2130706433/",  # decimal 127.0.0.1
        "http://127.1/",
        "http://0x7f000001/",
        "http://localhost:8081/props",
        "http://app.localhost/",
        "http://intranet.example/",
        "http://dual.example/",
        "https://meta.example/",
    ],
)
async def test_open_blocks_private_targets_before_connecting(
    url: str, resolver: dict[str, list[str]]
) -> None:
    recorder = Recorder()
    async with mock_client(recorder) as http:
        with pytest.raises(ResearchProviderError, match="blocked_target") as info:
            await Fetcher(make_settings(), http).open(url)
    assert str(info.value) == "blocked_target"  # keyword only: no address, no URL
    assert isinstance(info.value, TaggedProviderError) and info.value.reason == "blocked_target"
    assert recorder.requests == []  # nothing was sent, not even robots.txt


async def test_open_public_host_works_and_private_allowed_when_opted_in(
    resolver: dict[str, list[str]],
) -> None:
    recorder = Recorder()
    async with mock_client(recorder) as http:
        result = await Fetcher(make_settings(), http).open("https://public.example/page")
        assert result.status == 200 and result.body == "<p>ok</p>"
        assert result.final_url == "https://public.example/page"
        relaxed = Fetcher(make_settings(), http, allow_private_hosts=True)
        local = await relaxed.open("http://127.0.0.1:8081/props")
        assert local.status == 200
    assert "http://127.0.0.1:8081/props" in recorder.requests


async def test_unresolvable_host_is_left_to_the_transport(resolver: dict[str, list[str]]) -> None:
    # No address means nothing to classify; the transport then fails to connect for real.
    recorder = Recorder()
    async with mock_client(recorder) as http:
        result = await Fetcher(make_settings(), http).open("https://nowhere.invalid/x")
    assert result.status == 200


async def test_redirect_to_private_target_is_blocked(resolver: dict[str, list[str]]) -> None:
    targets = {
        "/to-ip": ("http://10.0.0.5/admin", "blocked_target"),
        "/to-name": ("http://intranet.example/", "blocked_target"),
        "/to-metadata": ("http://169.254.169.254/latest/", "blocked_target"),
        "/to-protocol-relative": ("//127.0.0.1:8081/props", "blocked_target"),
        "/to-file": ("file:///etc/passwd", "blocked_target"),
        "/to-loopback6": ("http://[::1]:8081/", "blocked_target"),
        # httpx itself refuses to build the next request for these (InvalidURL); the fetcher
        # turns that into a tagged failure instead of leaking a raw exception.
        "/to-data": ("data:text/html,hello", "fetch_failed"),
        "/to-javascript": ("javascript:alert(1)", "fetch_failed"),
    }
    recorder = Recorder(
        {
            path: (lambda r, loc=loc: httpx.Response(302, headers={"location": loc}))
            for path, (loc, _) in targets.items()
        }
    )
    async with mock_client(recorder) as http:
        fetcher = Fetcher(make_settings(), http)
        for path, (_, expected) in targets.items():
            with pytest.raises(ResearchProviderError) as info:
                await fetcher.open(f"https://public.example{path}")
            assert reject_reason(info.value) == expected, path
            assert str(info.value).startswith(expected), path
    hosts = {httpx.URL(u).host for u in recorder.requests}
    assert hosts == {"public.example"}  # no private host was ever contacted


async def test_public_redirects_are_followed_and_revalidated(
    resolver: dict[str, list[str]],
) -> None:
    def hop(n: int) -> Callable[[httpx.Request], httpx.Response]:
        return lambda r: httpx.Response(301, headers={"location": f"/r{n + 1}"})

    routes: dict[str, Callable[[httpx.Request], httpx.Response]] = {
        f"/r{i}": hop(i) for i in range(MAX_REDIRECTS)
    }
    routes["/cross"] = lambda r: httpx.Response(
        307, headers={"location": "https://news.example/story"}
    )
    recorder = Recorder(routes)
    async with mock_client(recorder) as http:
        fetcher = Fetcher(make_settings(), http)
        result = await fetcher.open("https://public.example/r0")
        assert result.url == "https://public.example/r0"
        assert result.final_url == f"https://public.example/r{MAX_REDIRECTS}"
        assert result.body == "<p>ok</p>"
        cross = await fetcher.open("https://public.example/cross")
        assert cross.final_url == "https://news.example/story"
    assert "https://news.example/story" in recorder.requests


async def test_redirect_loop_stops_after_max_hops(resolver: dict[str, list[str]]) -> None:
    recorder = Recorder({"/loop": lambda r: httpx.Response(302, headers={"location": "/loop"})})
    async with mock_client(recorder) as http:
        with pytest.raises(ResearchProviderError, match="fetch_failed") as info:
            await Fetcher(make_settings(), http).open("https://public.example/loop")
    assert info.value.reason == "fetch_failed"  # type: ignore[attr-defined]
    assert len(recorder.pages()) == MAX_REDIRECTS + 1  # the first request plus five hops


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "data:text/html,hi",
        "ftp://public.example/pub",
        "gopher://public.example/",
        "javascript:alert(1)",
        "http:///no-host",
        "https://",
        "public.example/relative",
    ],
)
async def test_non_http_urls_are_rejected(url: str, resolver: dict[str, list[str]]) -> None:
    recorder = Recorder()
    async with mock_client(recorder) as http:
        with pytest.raises(ResearchProviderError, match="blocked_target"):
            await Fetcher(make_settings(), http).open(url)
    assert recorder.requests == []


async def test_configured_search_backend_on_lan_is_allowed(
    resolver: dict[str, list[str]],
) -> None:
    recorder = Recorder()
    settings = make_settings(research_search_url="http://searx.lan:8080/")
    assert search_backend_hosts(settings.research_search_url) == {"searx.lan:8080"}
    async with mock_client(recorder) as http:
        provider = HttpResearchProvider(settings, http)
        assert provider.fetcher.allowed_hosts == {"searx.lan:8080"}
        result = await provider.open("http://searx.lan:8080/search?q=x&format=json")
        assert result.status == 200
        with pytest.raises(ResearchProviderError, match="blocked_target"):
            await provider.open("http://searx.lan:9090/")  # same host, other port
        with pytest.raises(ResearchProviderError, match="blocked_target"):
            await provider.open("http://192.168.1.20:8080/")  # the address itself is not listed
        with pytest.raises(ResearchProviderError, match="blocked_target"):
            await provider.open("http://192.168.1.21/")
        stack_provider, _, _ = build_research_stack(settings, http)
        assert isinstance(stack_provider, HttpResearchProvider)
        assert stack_provider.fetcher.allowed_hosts == {"searx.lan:8080"}
        literal = make_settings(research_search_url="http://192.168.1.20:8080")
        by_ip = HttpResearchProvider(literal, http)
        assert (await by_ip.open("http://192.168.1.20:8080/search")).status == 200
        with pytest.raises(ResearchProviderError, match="blocked_target"):
            await by_ip.open("http://192.168.1.20:8081/")
    assert search_backend_hosts(None) == frozenset()
    assert search_backend_hosts("  ") == frozenset()
    assert search_backend_hosts("ftp://searx.lan") == frozenset()
    assert search_backend_hosts("https://searx.example") == {"searx.example:443"}


@pytest.mark.parametrize(
    ("address", "blocked"),
    [
        ("127.0.0.1", True),
        ("10.0.0.5", True),
        ("172.16.0.1", True),
        ("192.168.1.1", True),
        ("169.254.169.254", True),
        ("100.64.0.1", True),
        ("192.0.0.1", True),
        ("198.18.0.1", True),
        ("0.0.0.0", True),
        ("224.0.0.1", True),
        ("255.255.255.255", True),
        ("::1", True),
        ("::", True),
        ("fe80::1", True),
        ("fc00::1", True),
        ("ff02::1", True),
        ("::ffff:10.0.0.5", True),
        ("2002:0a00:0005::", True),  # 6to4 embedding 10.0.0.5
        ("2001:db8::1", True),
        ("not an address", True),
        ("93.184.216.34", False),
        ("8.8.8.8", False),
        ("2606:4700::1111", False),
    ],
)
def test_address_blocked_table(address: str, blocked: bool) -> None:
    assert address_blocked(address) is blocked


async def test_resolver_timeout_fails_closed() -> None:
    async def slow(host: str) -> list[str]:
        import asyncio

        await asyncio.sleep(10)
        return [PUBLIC_IP]

    recorder = Recorder()
    async with mock_client(recorder) as http:
        fetcher = Fetcher(make_settings(research_fetch_timeout_s=0.05), http, resolver=slow)
        with pytest.raises(ResearchProviderError, match="timeout"):
            await fetcher.open("https://public.example/page")
    assert recorder.requests == []


# --------------------------------------------------------------------------------------
# (c) hostile HTML
# --------------------------------------------------------------------------------------


def test_deep_html_is_rejected_by_the_tree_builder_not_the_interpreter() -> None:
    with pytest.raises(HtmlTooDeepError):
        extract_html("https://news.example/deep", deep_html(5_000), NOW)
    with pytest.raises(ResearchProviderError, match="extract_failed") as info:
        extract_page(page("https://news.example/deep", deep_html(5_000)))
    assert str(info.value) == "extract_failed"
    assert isinstance(info.value.__cause__, HtmlTooDeepError)


def test_html_below_the_depth_cap_still_extracts() -> None:
    record = extract_html("https://news.example/nested", deep_html(MAX_DOM_DEPTH - 50), NOW)
    assert PARAGRAPH in record.text
    assert record.metadata["paragraphs"] == 1
    good = extract_page(page("https://news.example/good", GOOD_HTML))
    assert good.title == "Good page" and good.metadata["paragraphs"] == 4


def test_deeply_nested_jsonld_does_not_crash_extraction() -> None:
    html = (
        '<html><head><script type="application/ld+json">'
        + "[" * 200_000
        + "]" * 200_000
        + f"</script></head><body><p>{PARAGRAPH}</p></body></html>"
    )
    record = extract_html("https://news.example/jsonld", html, NOW)
    assert record.published_at is None and PARAGRAPH in record.text


async def test_provider_extract_maps_deep_page_to_extract_failed(
    resolver: dict[str, list[str]],
) -> None:
    recorder = Recorder(
        {
            "/deep": lambda r: httpx.Response(
                200, text=deep_html(5_000), headers={"content-type": "text/html"}
            )
        }
    )
    async with mock_client(recorder) as http:
        provider = HttpResearchProvider(make_settings(research_search_url=None), http)
        with pytest.raises(ResearchProviderError, match="extract_failed") as info:
            await provider.extract("https://news.example/deep")
    assert str(info.value) == "extract_failed"


def _searx_payload(*urls: str) -> dict[str, Any]:
    return {
        "results": [
            {"url": url, "title": f"[Fixture] {url.rsplit('/', 1)[-1]}", "content": "snippet"}
            for url in urls
        ]
    }


@pytest.fixture
def fixture_stack():
    settings = Settings(
        research_contact_email="dev@example.com", research_min_request_interval_s=0.0
    )
    return settings, fixture_research_stack(settings, FIXTURE_DIR)


async def test_runner_rejects_deep_page_with_extract_failed_and_continues(
    resolver: dict[str, list[str]], fixture_stack
) -> None:
    _, (_, edgar, prices) = fixture_stack
    recorder = Recorder(
        {
            "/search": lambda r: httpx.Response(
                200,
                json=_searx_payload(
                    "https://news.example/deep",
                    "https://news.example/good",
                    "https://news.example/boom",
                ),
            ),
            "/deep": lambda r: httpx.Response(
                200, text=deep_html(5_000), headers={"content-type": "text/html"}
            ),
            "/good": lambda r: httpx.Response(
                200, text=GOOD_HTML, headers={"content-type": "text/html"}
            ),
            "/boom": lambda r: httpx.Response(500, text="Traceback: secret internals"),
        }
    )
    settings = make_settings(research_search_url="https://searx.example")
    async with mock_client(recorder) as http:
        provider = HttpResearchProvider(settings, http)
        runner = ResearchRunner(provider, edgar, prices, settings, ResearchBudget())
        rec = RecordingCtx()
        planned = build_queries(ResearchIntent.retrieve_recent_news, IDENTITY, "near_term", AS_OF)[
            0
        ]
        result = await runner.execute(planned, IDENTITY, AS_OF, rec.ctx)
        assert [s.url for s in result.sources] == ["https://news.example/good"]
        assert sorted(reason for _, reason in result.rejected) == ["extract_failed", "fetch_failed"]
        deep = next(s for s, reason in result.rejected if reason == "extract_failed")
        assert deep.url == "https://news.example/deep" and deep.rejected_reason == "extract_failed"
        assert runner.stats.sources_rejected == 2 and runner.stats.sources_fetched == 1
        events = rec.named("research.source_rejected")
        assert sorted(e["reason"] for e in events) == ["extract_failed", "fetch_failed"]
        for event in events:
            assert event["reason"] in REJECTION_REASONS
            assert "Traceback" not in json.dumps(event) and "secret" not in json.dumps(event)
        # The analysis goes on: the next query still runs on the same runner.
        more = await runner.execute(
            build_queries(ResearchIntent.retrieve_price_history, IDENTITY, "near_term", AS_OF)[0],
            IDENTITY,
            AS_OF,
            rec.ctx,
        )
        assert more.price_series is not None and runner.stats.termination_reason is None


# --------------------------------------------------------------------------------------
# (d) client-safe reasons
# --------------------------------------------------------------------------------------


def test_reject_reason_maps_to_fixed_keywords() -> None:
    assert reject_reason(TaggedProviderError("blocked_target")) == "blocked_target"
    assert reject_reason(TaggedProviderError("timeout", "fetch failed for http://x: slow")) == (
        "timeout"
    )
    assert reject_reason(TaggedProviderError("extract_failed")) == "extract_failed"
    assert reject_reason(TaggedProviderError("robots_disallowed", "robots.txt disallows x")) == (
        "robots_disallowed"
    )
    assert reject_reason(TaggedProviderError("something_new", "detail")) == "fetch_failed"
    assert reject_reason(ResearchProviderError("no fixture page for https://x/y")) == "fetch_failed"
    assert reject_reason(ResearchProviderError("HTTP 500 for https://x: Traceback")) == (
        "fetch_failed"
    )
    assert reject_reason(ResearchProviderError("paywalled")) == "paywalled"
    for reason in REJECTION_REASONS:
        assert reason.replace("_", "").isalpha() and reason == reason.lower()


async def test_fetch_errors_carry_keyword_reasons(resolver: dict[str, list[str]]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
        if request.url.path == "/slow":
            raise httpx.ReadTimeout("slow upstream 10.0.0.9")
        if request.url.path == "/down":
            raise httpx.ConnectError("connection refused by 10.0.0.9")
        if request.url.path == "/missing":
            return httpx.Response(404, text="nope")
        return httpx.Response(200, text="ok")

    async with mock_client(handler) as http:
        fetcher = Fetcher(make_settings(), http)
        expectations = {
            "/slow": "timeout",
            "/down": "fetch_failed",
            "/missing": "fetch_failed",
            "/private/x": "robots_disallowed",
        }
        for path, expected in expectations.items():
            with pytest.raises(ResearchProviderError) as info:
                await fetcher.open(f"https://public.example{path}")
            assert reject_reason(info.value) == expected, path
            assert str(info.value).startswith(expected)


async def test_research_unavailable_details_carry_no_urls_or_exception_text(
    fixture_stack,
) -> None:
    settings, (provider, edgar, prices) = fixture_stack
    runner = ResearchRunner(provider, edgar, prices, settings, ResearchBudget())
    rec = RecordingCtx()
    missing = InstrumentIdentity(symbol="NOPE", name="Nowhere Corp", cik="999999")
    planned = build_queries(ResearchIntent.retrieve_latest_filing, missing, "near_term", AS_OF)[0]
    with pytest.raises(AnalysisError) as info:
        await runner.execute(planned, missing, AS_OF, rec.ctx)
    assert info.value.code is ErrorCode.RESEARCH_UNAVAILABLE
    assert info.value.details == {
        "kind": "edgar_submissions",
        "stage": "edgar_submissions",
        "reason": "fetch_failed",
    }
    assert "http" not in json.dumps(info.value.details).lower()


async def test_price_failure_reason_is_a_keyword(fixture_stack) -> None:
    settings, (provider, edgar, prices) = fixture_stack
    runner = ResearchRunner(provider, edgar, prices, settings, ResearchBudget())
    rec = RecordingCtx()
    planned = build_queries(ResearchIntent.retrieve_price_history, IDENTITY, "near_term", AS_OF)[
        0
    ].model_copy(update={"params": {"symbol": "zzzz.us", "days": 30}})
    result = await runner.execute(planned, IDENTITY, AS_OF, rec.ctx)
    assert [reason for _, reason in result.rejected] == ["fetch_failed"]
    assert [e["reason"] for e in rec.named("research.source_rejected")] == ["fetch_failed"]


# --------------------------------------------------------------------------------------
# (e) cache policy
# --------------------------------------------------------------------------------------


async def test_bodies_are_cached_only_for_allow_listed_hosts(
    tmp_path: Path, resolver: dict[str, list[str]]
) -> None:
    served = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        served["n"] += 1
        if request.url.host == "data.sec.gov":
            return httpx.Response(
                200, json={"cik": 320193}, headers={"content-type": "application/json"}
            )
        return httpx.Response(
            200,
            text=f"<html><body><p>{PARAGRAPH}</p></body></html>",
            headers={"content-type": "text/html"},
        )

    settings = make_settings(research_cache_dir=tmp_path)
    async with mock_client(handler) as http:
        fetcher = Fetcher(settings, http)
        journalism = "https://www.reuters.com/markets/fixture-story"
        first = await fetcher.open(journalism)
        second = await fetcher.open(journalism)
        assert first.from_cache is False and second.from_cache is False
        assert served["n"] == 2
        assert json_files(tmp_path) == []  # journalism body never touches disk
        structured = "https://data.sec.gov/submissions/CIK0000320193.json"
        fresh = await fetcher.open(structured, ttl_s=3600)
        cached = await fetcher.open(structured, ttl_s=3600)
        assert fresh.from_cache is False and cached.from_cache is True
        assert cached.body == fresh.body and served["n"] == 3
        files = json_files(tmp_path)
        assert len(files) == 1
        payload = json.loads(files[0].read_text())
        assert payload["url"] == structured and payload["ttl_s"] == 3600
        assert payload["page"]["body"] == fresh.body
        assert "expires_at" in payload
        # Operators may widen the allow-list explicitly.
        custom = Fetcher(settings, http, cache_body_hosts={"www.reuters.com"})
        await custom.open(journalism)
        assert (await custom.open(journalism)).from_cache is True


async def test_cache_write_sweeps_expired_files(
    tmp_path: Path, resolver: dict[str, list[str]]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(
            200, text="Date,Close\n2026-09-25,1\n", headers={"content-type": "text/csv"}
        )

    stale = tmp_path / "stale.json"
    stale_tmp = tmp_path / "stale.tmp"
    fresh = tmp_path / "fresh.json"
    unrelated = tmp_path / "notes.txt"
    for path in (stale, stale_tmp, fresh, unrelated):
        path.write_text("{}")
    expired = time.time() - CACHE_MAX_AGE_S - 3600
    for path in (stale, stale_tmp, unrelated):
        os.utime(path, (expired, expired))
    async with mock_client(handler) as http:
        fetcher = Fetcher(make_settings(research_cache_dir=tmp_path), http)
        result = await fetcher.open("https://stooq.com/q/d/l/?s=aapl.us&i=d", ttl_s=3600)
    assert result.status == 200
    assert not stale.exists() and not stale_tmp.exists()
    assert fresh.exists() and unrelated.exists()  # recent entries and foreign files survive
    written = [p for p in json_files(tmp_path) if p != fresh]
    assert len(written) == 1


async def test_cache_read_unlinks_entries_expired_for_the_caller(
    tmp_path: Path, resolver: dict[str, list[str]]
) -> None:
    served = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        served["n"] += 1
        return httpx.Response(
            200, json={"n": served["n"]}, headers={"content-type": "application/json"}
        )

    url = "https://www.sec.gov/files/company_tickers.json"
    async with mock_client(handler) as http:
        fetcher = Fetcher(make_settings(research_cache_dir=tmp_path), http)
        await fetcher.open(url, ttl_s=3600)
        entry = json_files(tmp_path)[0]
        payload = json.loads(entry.read_text())
        payload["page"]["fetched_at"] = "2020-01-01T00:00:00Z"
        entry.write_text(json.dumps(payload))
        refetched = await fetcher.open(url, ttl_s=3600)
    assert refetched.from_cache is False and served["n"] == 2
    assert json.loads(entry.read_text())["page"]["fetched_at"] != "2020-01-01T00:00:00Z"
