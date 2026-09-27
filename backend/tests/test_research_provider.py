"""Provider-layer tests: SearXNG client, Fetcher, extract, sources, dedup. No network."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path

import httpx
import pytest

from bayanalytics.config import Settings
from bayanalytics.research.dates import parse_datetime_lenient
from bayanalytics.research.dedup import Deduplicator
from bayanalytics.research.extract import (
    EXCERPT_CHARS,
    MAX_TEXT_CHARS,
    extract_csv,
    extract_html,
    extract_json,
    extract_page,
    registrable_domain,
)
from bayanalytics.research.fetch import (
    MAX_BODY_BYTES,
    PAYWALL_HEADER,
    TRUNCATED_HEADER,
    Fetcher,
    throttle_key,
)
from bayanalytics.research.provider import PageResult, ResearchProviderError
from bayanalytics.research.searxng import SearxngSearch
from bayanalytics.research.sources import (
    canonical_url,
    classify_freshness,
    classify_source,
    source_record_from_evidence,
)

AS_OF = datetime(2026, 9, 26, tzinfo=UTC)
NOW = datetime(2026, 9, 26, 12, 0, tzinfo=UTC)

SAMPLE_HTML = """<!doctype html>
<html lang="en"><head>
<title>Page title - Site</title>
<meta property="og:title" content="Fixture headline\u0007 with control char">
<meta property="og:site_name" content="Fixture Site">
<meta property="article:published_time" content="2026-09-20T10:00:00Z">
<script>var x = "script text must not appear";</script>
<style>.a{color:red}</style>
</head><body>
<nav><a href="/a">Nav link one</a> <a href="/b">Nav link two</a></nav>
<header>Header text that is chrome</header>
<main><article>
<h1>Fixture headline</h1>
<p>First paragraph of the fixture body with enough words to count as real text. \x01\x02</p>
<p>Second paragraph continues the fixture story with more words so the container wins the score.</p>
<p>Third paragraph has a <a href="/x">link</a> inside but is mostly prose about fixture matters.</p>
</article></main>
<aside><p>Sidebar promotional paragraph that should be dropped from the main text.</p></aside>
<footer><p>Footer text about the fixture site.</p></footer>
</body></html>
"""


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "research_contact_email": "dev@example.com",
        "research_min_request_interval_s": 0.0,
        "research_fetch_timeout_s": 5.0,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def mock_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


# --------------------------------------------------------------------------------------
# searxng
# --------------------------------------------------------------------------------------


async def test_searxng_maps_results_and_sends_params() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["params"] = dict(request.url.params)
        captured["ua"] = request.headers.get("user-agent")
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/a",
                        "title": "A title",
                        "content": "A snippet",
                        "engine": "duckduckgo",
                        "publishedDate": "2026-09-20T10:00:00Z",
                        "score": 3.5,
                        "engines": ["duckduckgo", "bing"],
                    },
                    {"url": "ftp://bad", "title": "skipped"},
                    "garbage",
                    {"url": "https://example.com/b", "title": "", "publishedDate": "not a date"},
                ]
            },
        )

    async with mock_client(handler) as http:
        client = SearxngSearch("https://searx.local/", http, "BayAnalytics/0.1 (dev@example.com)")
        results = await client.search(
            "apple earnings", categories="news", time_range="month", max_results=10
        )
    assert captured["url"].startswith("https://searx.local/search?")  # type: ignore[union-attr]
    assert captured["params"] == {
        "q": "apple earnings",
        "format": "json",
        "categories": "news",
        "time_range": "month",
        "language": "en",
    }
    assert captured["ua"] == "BayAnalytics/0.1 (dev@example.com)"
    assert [r.url for r in results] == ["https://example.com/a", "https://example.com/b"]
    first = results[0]
    assert first.title == "A title"
    assert first.snippet == "A snippet"
    assert first.engine == "duckduckgo"
    assert first.published_at == datetime(2026, 9, 20, 10, 0, tzinfo=UTC)
    assert first.score == 3.5
    assert first.metadata["engines"] == ["duckduckgo", "bing"]
    assert results[1].title == "https://example.com/b"
    assert results[1].published_at is None


async def test_searxng_errors() -> None:
    async with mock_client(lambda r: httpx.Response(503, text="down")) as http:
        client = SearxngSearch("https://searx.local", http, "ua")
        with pytest.raises(ResearchProviderError, match="503"):
            await client.search("x")
        unconfigured = SearxngSearch(None, http, "ua")
        with pytest.raises(ResearchProviderError, match="not configured"):
            await unconfigured.search("x")
        with pytest.raises(ResearchProviderError):
            await client.search("   ")
    async with mock_client(lambda r: httpx.Response(200, text="not json")) as http:
        client = SearxngSearch("https://searx.local", http, "ua")
        with pytest.raises(ResearchProviderError, match="invalid JSON"):
            await client.search("x")


async def test_searxng_max_results() -> None:
    payload = {"results": [{"url": f"https://e.com/{i}", "title": str(i)} for i in range(20)]}
    async with mock_client(lambda r: httpx.Response(200, json=payload)) as http:
        client = SearxngSearch("https://searx.local", http, "ua")
        assert len(await client.search("x", max_results=5)) == 5


# --------------------------------------------------------------------------------------
# fetcher
# --------------------------------------------------------------------------------------


def _robots_ok(request: httpx.Request) -> httpx.Response | None:
    if request.url.path == "/robots.txt":
        return httpx.Response(200, text="User-agent: *\nDisallow: /private\n")
    return None


async def test_fetcher_applies_user_agent_and_rate_limit() -> None:
    stamps: list[tuple[str, float]] = []
    agents: set[str] = set()

    def handler(request: httpx.Request) -> httpx.Response:
        stamps.append((request.url.host, time.monotonic()))
        agents.add(request.headers.get("user-agent", ""))
        robots = _robots_ok(request)
        return robots or httpx.Response(200, text="<html><body><p>ok</p></body></html>")

    settings = make_settings(research_min_request_interval_s=0.05)
    async with mock_client(handler) as http:
        fetcher = Fetcher(settings, http)
        await fetcher.open("https://example.com/one")
        await fetcher.open("https://example.com/two")
        await fetcher.open("https://example.com/three")
    assert agents == {"BayAnalytics/0.1 (dev@example.com)"}
    example = [t for host, t in stamps if host == "example.com"]
    assert len(example) == 4  # robots + 3 pages
    for earlier, later in pairwise(example):
        assert later - earlier >= 0.05 * 0.9


async def test_fetcher_sec_hosts_never_exceed_ten_per_second() -> None:
    stamps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        stamps.append(time.monotonic())
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, json={"ok": True})

    settings = make_settings(research_min_request_interval_s=0.0)
    async with mock_client(handler) as http:
        fetcher = Fetcher(settings, http)
        assert fetcher.interval_for("data.sec.gov") >= 0.11
        assert fetcher.interval_for("www.sec.gov") >= 0.11
        assert throttle_key("www.sec.gov") == throttle_key("data.sec.gov") == "sec.gov"
        await fetcher.open("https://data.sec.gov/submissions/CIK0000320193.json")
        await fetcher.open("https://www.sec.gov/files/company_tickers.json")
    # robots(data) -> page(data) -> robots(www) -> page(www): all share one SEC gate
    for earlier, later in pairwise(stamps):
        assert later - earlier >= 0.11 * 0.9


async def test_fetcher_honours_robots_and_allows_when_robots_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "broken.example":
            if request.url.path == "/robots.txt":
                raise httpx.ConnectError("no robots")
            return httpx.Response(200, text="fine")
        return _robots_ok(request) or httpx.Response(200, text="fine")

    async with mock_client(handler) as http:
        fetcher = Fetcher(make_settings(), http)
        with pytest.raises(ResearchProviderError, match="robots"):
            await fetcher.open("https://example.com/private/report")
        page = await fetcher.open("https://example.com/public/report")
        assert page.status == 200 and page.body == "fine"
        page = await fetcher.open("https://broken.example/anything")
        assert page.body == "fine"


async def test_fetcher_caps_body_at_two_mib() -> None:
    big = b"x" * (MAX_BODY_BYTES + 500_000)

    def handler(request: httpx.Request) -> httpx.Response:
        return _robots_ok(request) or httpx.Response(
            200, content=big, headers={"content-type": "text/plain; charset=utf-8"}
        )

    async with mock_client(handler) as http:
        page = await Fetcher(make_settings(), http).open("https://example.com/big")
    assert len(page.body) == MAX_BODY_BYTES
    assert page.headers[TRUNCATED_HEADER] == "true"


async def test_fetcher_cache_hit_sets_from_cache(tmp_path: Path) -> None:
    calls = {"pages": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls["pages"] += 1
        return httpx.Response(200, text="<p>cached body</p>", headers={"content-type": "text/html"})

    settings = make_settings(research_cache_dir=tmp_path)
    async with mock_client(handler) as http:
        fetcher = Fetcher(settings, http)
        # Bodies are only cached for allow-listed structured/public-domain hosts
        # (fetch.CACHE_BODY_HOSTS); journalism hosts are never body-cached.
        url = "https://stooq.com/page"
        first = await fetcher.open(url)
        second = await fetcher.open(url)
        third = await fetcher.open(url, ttl_s=0)
    assert first.from_cache is False
    assert second.from_cache is True and second.body == "<p>cached body</p>"
    assert third.from_cache is False
    assert calls["pages"] == 2
    key = hashlib.sha256(url.encode()).hexdigest()
    assert (tmp_path / f"{key}.json").exists()
    stored = json.loads((tmp_path / f"{key}.json").read_text())
    assert stored["page"]["body"] == "<p>cached body</p>"


async def test_fetcher_marks_paywalls_and_does_not_bypass() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/403":
            return httpx.Response(403, text="<html>subscribe</html>")
        if request.url.path == "/marker":
            body = (
                '<html><script type="application/ld+json">{"isAccessibleForFree": false}'
                "</script><p>teaser</p></html>"
            )
            return httpx.Response(200, text=body, headers={"content-type": "text/html"})
        if request.url.path == "/missing":
            return httpx.Response(404, text="nope")
        return httpx.Response(200, text="<p>free</p>", headers={"content-type": "text/html"})

    async with mock_client(handler) as http:
        fetcher = Fetcher(make_settings(), http)
        blocked = await fetcher.open("https://paywall.example/403")
        assert blocked.status == 403 and blocked.body == ""
        assert blocked.headers[PAYWALL_HEADER] == "true"
        marker = await fetcher.open("https://paywall.example/marker")
        assert marker.status == 200 and marker.body == ""
        assert marker.headers[PAYWALL_HEADER] == "true"
        free = await fetcher.open("https://paywall.example/free")
        assert PAYWALL_HEADER not in free.headers and free.body == "<p>free</p>"
        with pytest.raises(ResearchProviderError, match="HTTP 404"):
            await fetcher.open("https://paywall.example/missing")
    record = extract_page(blocked)
    assert record.metadata["paywalled"] is True and record.text == ""


async def test_fetcher_retries_once_on_connect_error() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, text="second try")

    async with mock_client(handler) as http:
        page = await Fetcher(make_settings(), http).open("https://flaky.example/x")
    assert page.body == "second try" and attempts["n"] == 2

    def always_fails(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        raise httpx.ReadTimeout("slow")

    async with mock_client(always_fails) as http:
        with pytest.raises(ResearchProviderError, match="fetch failed"):
            await Fetcher(make_settings(), http).open("https://flaky.example/y")


async def test_fetcher_rejects_unsupported_urls() -> None:
    async with mock_client(lambda r: httpx.Response(200)) as http:
        with pytest.raises(ResearchProviderError):
            await Fetcher(make_settings(), http).open("file:///etc/passwd")


# --------------------------------------------------------------------------------------
# extract
# --------------------------------------------------------------------------------------


def test_extract_html_metadata_text_and_hash() -> None:
    record = extract_html("https://www.fixture-site.com/story?utm_source=x", SAMPLE_HTML, NOW)
    assert record.title == "Fixture headline with control char"
    assert record.publisher == "Fixture Site"
    assert record.published_at == datetime(2026, 9, 20, 10, 0, tzinfo=UTC)
    assert record.language == "en"
    assert record.extraction_method == "html_readability_v1"
    assert "First paragraph of the fixture body" in record.text
    assert "Third paragraph has a link inside" in record.text
    for chrome in ("Nav link", "Header text", "Sidebar promotional", "Footer text", "script text"):
        assert chrome not in record.text
    assert "\x01" not in record.text and "\x02" not in record.text
    normalised = " ".join(record.text.split())
    assert record.content_hash == hashlib.sha256(normalised.encode()).hexdigest()
    assert record.excerpt == normalised[:EXCERPT_CHARS].rstrip()
    assert record.retrieved_at == NOW
    assert record.metadata["paragraphs"] == 3


def test_extract_html_fallbacks_jsonld_time_and_domain() -> None:
    html = """<html><head><title>T</title>
    <script type="application/ld+json">
    {"@type":"NewsArticle","datePublished":"2026-09-10T08:00:00+02:00"}</script>
    </head><body><div><p>%s</p></div></body></html>""" % ("prose " * 30)
    record = extract_html("https://news.example.co.uk/x", html, NOW)
    assert record.publisher == "example.co.uk"
    assert record.published_at == datetime(2026, 9, 10, 6, 0, tzinfo=UTC)
    assert record.title == "T"

    html_time = (
        '<html><body><article><time datetime="2026-09-01">Sep 1</time><p>%s</p>'
        "</article></body></html>" % ("words " * 30)
    )
    record = extract_html("https://www.example.com/y", html_time, NOW)
    assert record.published_at == datetime(2026, 9, 1, tzinfo=UTC)
    assert record.publisher == "example.com"

    undated = extract_html(
        "https://example.com/z", "<html><body><p>%s</p></body></html>" % ("w " * 40), NOW
    )
    assert undated.published_at is None


def test_extract_html_caps_text_and_excerpt() -> None:
    paragraphs = "".join(
        f"<p>Paragraph {i} of a very long fixture page with filler words.</p>" for i in range(2000)
    )
    record = extract_html(
        "https://example.com/long", f"<html><body><main>{paragraphs}</main></body></html>", NOW
    )
    assert len(record.text) == MAX_TEXT_CHARS
    assert len(record.excerpt) <= EXCERPT_CHARS


def test_extract_html_without_paragraph_tags_falls_back_to_body_text() -> None:
    html = (
        "<html><body><div>Some text lives directly in a div without paragraph tags at all."
        "</div></body></html>"
    )
    record = extract_html("https://example.com/div", html, NOW)
    assert "lives directly in a div" in record.text


def test_extract_json_and_csv() -> None:
    record = extract_json("https://data.sec.gov/x/CIK1.json", '{"a": 1, "b": [1, 2]}', NOW)
    assert record.extraction_method == "json"
    assert record.structured == {"a": 1, "b": [1, 2]}
    assert record.text == "" and record.excerpt == ""
    assert record.publisher == "sec.gov"
    listed = extract_json("https://example.com/list.json", "[1, 2]", NOW)
    assert listed.structured == {"items": [1, 2]}
    csv_record = extract_csv(
        "https://stooq.com/q/d/l/?s=aapl.us&i=d",
        "Date,Open,High,Low,Close,Volume\n2026-09-24,1,2,0.5,1.5,100\n\n2026-09-25,1.5,2,1,1.8,120\n",
        NOW,
    )
    assert csv_record.extraction_method == "csv"
    assert csv_record.structured["columns"] == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert csv_record.structured["row_count"] == 2
    assert csv_record.structured["rows"][1][4] == "1.8"


def test_extract_page_dispatch() -> None:
    def page(url: str, body: str, ctype: str | None) -> PageResult:
        return PageResult(
            url=url, final_url=url, status=200, content_type=ctype, body=body, fetched_at=NOW
        )

    assert (
        extract_page(page("https://a/x", '{"k": 1}', "application/json")).extraction_method
        == "json"
    )
    assert (
        extract_page(page("https://a/x.csv", "Date,Close\n2026-01-01,1\n", None)).extraction_method
        == "csv"
    )
    assert extract_page(
        page("https://a/x", SAMPLE_HTML, "text/html; charset=utf-8")
    ).extraction_method.startswith("html")
    text_record = extract_page(page("https://a/x", "plain words " * 10, "text/plain"))
    assert text_record.extraction_method == "text"
    sniffed = extract_page(page("https://a/x", '{"sniffed": true}', None))
    assert sniffed.extraction_method == "json"


def test_registrable_domain() -> None:
    assert registrable_domain("www.cnbc.com") == "cnbc.com"
    assert registrable_domain("investor.apple.com") == "apple.com"
    assert registrable_domain("news.bbc.co.uk") == "bbc.co.uk"
    assert registrable_domain("localhost") == "localhost"


def test_parse_datetime_lenient_formats() -> None:
    assert parse_datetime_lenient("2026-09-20T10:00:00Z") == datetime(2026, 9, 20, 10, tzinfo=UTC)
    assert parse_datetime_lenient("2026-09-20 10:00:00") == datetime(2026, 9, 20, 10, tzinfo=UTC)
    assert parse_datetime_lenient("2026-09-20") == datetime(2026, 9, 20, tzinfo=UTC)
    assert parse_datetime_lenient("Sat, 20 Sep 2026 10:00:00 GMT") == datetime(
        2026, 9, 20, 10, tzinfo=UTC
    )
    assert parse_datetime_lenient("September 20, 2026") == datetime(2026, 9, 20, tzinfo=UTC)
    assert parse_datetime_lenient("20 Sep 2026") == datetime(2026, 9, 20, tzinfo=UTC)
    assert parse_datetime_lenient("2026-09-20T12:00:00+02:00") == datetime(
        2026, 9, 20, 10, tzinfo=UTC
    )
    assert parse_datetime_lenient("nonsense") is None
    assert parse_datetime_lenient("") is None
    assert parse_datetime_lenient(None) is None


# --------------------------------------------------------------------------------------
# sources / dedup
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("url", "source_type", "publisher", "redistribution"),
    [
        (
            "https://www.sec.gov/Archives/edgar/data/320193/x.htm",
            "regulatory_filing",
            "SEC EDGAR",
            "allowed",
        ),
        (
            "https://data.sec.gov/submissions/CIK0000320193.json",
            "regulatory_filing",
            "SEC EDGAR",
            "allowed",
        ),
        ("https://stooq.com/q/d/l/?s=aapl.us&i=d", "market_data", "Stooq", "metadata_only"),
        (
            "https://www.reuters.com/markets/x",
            "financial_journalism",
            "reuters.com",
            "metadata_only",
        ),
        (
            "https://www.bloomberg.com/news/x",
            "financial_journalism",
            "bloomberg.com",
            "metadata_only",
        ),
        ("https://www.wsj.com/articles/x", "financial_journalism", "wsj.com", "metadata_only"),
        ("https://www.ft.com/content/x", "financial_journalism", "ft.com", "metadata_only"),
        ("https://www.cnbc.com/2026/x", "financial_journalism", "cnbc.com", "metadata_only"),
        ("https://www.barrons.com/x", "financial_journalism", "barrons.com", "metadata_only"),
        (
            "https://www.marketwatch.com/story/x",
            "financial_journalism",
            "marketwatch.com",
            "metadata_only",
        ),
        ("https://www.nytimes.com/x", "financial_journalism", "nytimes.com", "metadata_only"),
        (
            "https://seekingalpha.com/article/x",
            "secondary_commentary",
            "seekingalpha.com",
            "metadata_only",
        ),
        ("https://www.fool.com/investing/x", "secondary_commentary", "fool.com", "metadata_only"),
        (
            "https://www.investing.com/news/x",
            "secondary_commentary",
            "investing.com",
            "metadata_only",
        ),
        ("https://www.benzinga.com/x", "secondary_commentary", "benzinga.com", "metadata_only"),
        ("https://www.zacks.com/x", "secondary_commentary", "zacks.com", "metadata_only"),
        (
            "https://www.businesswire.com/news/home/x",
            "earnings_release",
            "businesswire.com",
            "metadata_only",
        ),
        (
            "https://www.prnewswire.com/news-releases/x",
            "earnings_release",
            "prnewswire.com",
            "metadata_only",
        ),
        (
            "https://www.globenewswire.com/news-release/x",
            "earnings_release",
            "globenewswire.com",
            "metadata_only",
        ),
        ("https://investor.apple.com/x", "investor_relations", "apple.com", "metadata_only"),
        ("https://ir.example.com/x", "investor_relations", "example.com", "metadata_only"),
        (
            "https://www.example.com/investors/reports",
            "investor_relations",
            "example.com",
            "metadata_only",
        ),
        ("https://randomblog.net/post", "unverified_web", "randomblog.net", "unknown"),
    ],
)
def test_classify_source_table(
    url: str, source_type: str, publisher: str, redistribution: str
) -> None:
    got_type, got_publisher, got_redistribution, note = classify_source(url)
    assert got_type == source_type
    assert got_publisher == publisher
    assert got_redistribution == redistribution
    if source_type == "regulatory_filing":
        assert note and "US government work" in note
    if source_type == "market_data":
        assert note == "Stooq terms: personal use, verify before redistribution"
    if source_type == "unverified_web":
        assert note is None


def test_classify_source_uses_given_publisher_for_unknown_hosts() -> None:
    assert classify_source("https://randomblog.net/post", "Random Blog")[1] == "Random Blog"
    assert classify_source("https://www.sec.gov/x", "ignored")[1] == "SEC EDGAR"


def test_canonical_url() -> None:
    assert (
        canonical_url(
            "HTTPS://WWW.Example.com:443/Path/?utm_source=a&b=2&a=1&fbclid=x&gclid=y#frag"
        )
        == "https://www.example.com/Path?a=1&b=2"
    )
    assert canonical_url("https://example.com/") == "https://example.com"
    assert canonical_url("https://example.com") == "https://example.com"
    assert canonical_url("https://stooq.com/q/d/l/?s=^spx&i=d") == canonical_url(
        "https://stooq.com/q/d/l/?i=d&s=%5Espx"
    )


def test_classify_freshness() -> None:
    assert classify_freshness(None, AS_OF) == "unknown"
    assert classify_freshness(AS_OF - timedelta(days=1), AS_OF) == "current"
    assert classify_freshness(AS_OF - timedelta(days=3), AS_OF) == "current"
    assert classify_freshness(AS_OF - timedelta(days=10), AS_OF) == "recent"
    assert classify_freshness(AS_OF - timedelta(days=31), AS_OF) == "stale"
    assert classify_freshness(datetime(2026, 9, 25), AS_OF) == "current"  # naive treated as UTC


def test_source_record_from_evidence_fills_provenance() -> None:
    record = extract_html("https://www.cnbc.com/2026/09/18/x.html?utm_source=n", SAMPLE_HTML, NOW)
    source = source_record_from_evidence(record, "AAPL", "retrieve_recent_news", AS_OF)
    assert source.source_id.startswith("src_")
    assert source.source_type == "financial_journalism"
    assert source.publisher == "Fixture Site"  # og:site_name wins over the bare domain
    assert source.symbol == "AAPL"
    assert source.research_intent == "retrieve_recent_news"
    assert source.freshness == "recent"
    assert source.redistribution == "metadata_only"
    assert source.excerpt == record.excerpt and len(source.excerpt) <= EXCERPT_CHARS
    assert source.content_hash == record.content_hash
    assert source.extraction_method == "html_readability_v1"
    assert source.metadata["canonical_url"] == "https://www.cnbc.com/2026/09/18/x.html"
    assert source.public_view()["is_primary"] is False


def test_deduplicator_counts_url_hash_and_title_duplicates() -> None:
    dedup = Deduplicator()
    assert dedup.seen("https://example.com/a") is False
    assert dedup.seen("https://example.com/a/?utm_source=x#top") is True
    assert dedup.known_url("https://EXAMPLE.com/a") is True
    digest = hashlib.sha256(b"text").hexdigest()
    assert dedup.seen(digest) is False
    assert dedup.seen(digest) is True
    assert (
        dedup.seen("https://example.com/b", title="Apple Reports Third Quarter Results!") is False
    )
    assert dedup.seen("https://example.com/c", title="apple reports third-quarter results") is True
    assert dedup.seen("https://example.com/d", title="short") is False
    assert dedup.duplicates == 3
    record = extract_html("https://example.com/e", SAMPLE_HTML, NOW)
    assert dedup.seen_record(record) is False
    twin = extract_html("https://mirror.example.com/e", SAMPLE_HTML, NOW)
    assert dedup.seen_record(twin) is True  # same hash and title: counted once
    assert dedup.duplicates == 4
