"""Polite HTTP page fetcher used by the HTTP research provider, EDGAR and Stooq clients.

Responsibilities (AGENT.md sections 21 and 26):

* identifies itself with the settings user agent (SEC EDGAR requires contact information),
* keeps a per-host minimum interval between requests and never exceeds the SEC fair-access
  limit (10 requests per second) on ``sec.gov`` / ``data.sec.gov``,
* honours ``robots.txt`` (cached per host, unavailable robots -> allow),
* streams bodies and cuts them at 2 MiB so a filing can never be stored whole by accident,
* retries once on connect/timeout errors,
* optionally caches pages on disk keyed by sha256(url) with a TTL,
* never bypasses paywalls: HTTP 401/402/403 and bodies with obvious paywall markers come back
  as an empty ``PageResult`` flagged with the ``x-bay-paywalled`` header.

Every failure is raised as ``ResearchProviderError``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Awaitable, Callable
from datetime import timedelta
from pathlib import Path
from typing import Protocol
from urllib import robotparser
from urllib.parse import urlsplit

import httpx

from bayanalytics.config import Settings
from bayanalytics.research.provider import PageResult, ResearchProviderError
from bayanalytics.schemas.common import utcnow

log = logging.getLogger(__name__)

MAX_BODY_BYTES = 2 * 1024 * 1024
DEFAULT_TTL_S = 6 * 3600.0
EDGAR_TICKERS_TTL_S = 24 * 3600.0
PRICE_CSV_TTL_S = 3600.0
SEC_HOSTS: frozenset[str] = frozenset({"sec.gov", "www.sec.gov", "data.sec.gov", "efts.sec.gov"})
SEC_MIN_INTERVAL_S = 0.11  # SEC fair-access policy: at most 10 requests per second
PAYWALL_HEADER = "x-bay-paywalled"
TRUNCATED_HEADER = "x-bay-truncated"
PAYWALL_STATUSES: frozenset[int] = frozenset({401, 402, 403})
PAYWALL_MARKERS: tuple[str, ...] = (
    '"isaccessibleforfree":false',
    '"isaccessibleforfree": false',
    'id="paywall"',
    'class="paywall',
    "data-paywall",
    "subscribe to continue reading",
    "subscribe to read",
    "subscription required",
    "subscribers only",
    "this article is for subscribers",
)
DEFAULT_ACCEPT = (
    "text/html,application/xhtml+xml,application/json,text/csv,text/plain;q=0.9,*/*;q=0.5"
)
_CHARSET_RE = re.compile(rb'<meta[^>]+charset=["\']?\s*([a-zA-Z0-9_\-]+)', re.IGNORECASE)


class PageFetcher(Protocol):
    """What EDGAR/Stooq clients and the HTTP provider need from a fetcher."""

    async def open(
        self, url: str, *, ttl_s: float | None = None, accept: str | None = None
    ) -> PageResult: ...


class _HostThrottle:
    """Per-host minimum interval between requests, cooperative across concurrent callers."""

    def __init__(
        self,
        clock: Callable[[], float],
        sleep: Callable[[float], Awaitable[None]],
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, float] = {}

    async def wait(self, key: str, interval_s: float) -> None:
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            last = self._last.get(key)
            if last is not None and interval_s > 0:
                delay = last + interval_s - self._clock()
                if delay > 0:
                    await self._sleep(delay)
            self._last[key] = self._clock()


def throttle_key(host: str) -> str:
    host = host.lower()
    return "sec.gov" if host in SEC_HOSTS or host.endswith(".sec.gov") else host


class Fetcher:
    def __init__(
        self,
        settings: Settings,
        http: httpx.AsyncClient | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._owns_client = http is None
        self._http = http or httpx.AsyncClient(follow_redirects=True)
        self._user_agent = settings.user_agent
        self._timeout = settings.research_fetch_timeout_s
        self._interval = max(0.0, settings.research_min_request_interval_s)
        self._throttle = _HostThrottle(clock, sleep)
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}
        self._cache_dir: Path | None = settings.research_cache_dir

    @property
    def user_agent(self) -> str:
        return self._user_agent

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    # -- public ---------------------------------------------------------------------------

    async def open(
        self, url: str, *, ttl_s: float | None = None, accept: str | None = None
    ) -> PageResult:
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.netloc:
            raise ResearchProviderError(f"unsupported url: {url}")
        ttl = DEFAULT_TTL_S if ttl_s is None else ttl_s
        cached = self._cache_read(url, ttl)
        if cached is not None:
            return cached
        if not await self._robots_allow(parts.scheme, parts.netloc, url):
            raise ResearchProviderError(f"robots.txt disallows {url}")
        page = await self._fetch_with_retry(url, accept)
        if 200 <= page.status < 300 and page.headers.get(PAYWALL_HEADER) != "true":
            self._cache_write(url, page)
        return page

    # -- throttling -----------------------------------------------------------------------

    def interval_for(self, host: str) -> float:
        if throttle_key(host) == "sec.gov":
            return max(SEC_MIN_INTERVAL_S, self._interval)
        return self._interval

    async def _throttled_get(self, url: str, headers: dict[str, str]) -> httpx.Response:
        host = urlsplit(url).netloc
        await self._throttle.wait(throttle_key(host), self.interval_for(host))
        request = self._http.build_request("GET", url, headers=headers, timeout=self._timeout)
        return await self._http.send(request, stream=True, follow_redirects=True)

    # -- fetching -------------------------------------------------------------------------

    async def _fetch_with_retry(self, url: str, accept: str | None) -> PageResult:
        headers = {"User-Agent": self._user_agent, "Accept": accept or DEFAULT_ACCEPT}
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return await self._fetch_once(url, headers)
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                last_error = exc
                log.debug("fetch %s attempt %d failed: %s", url, attempt + 1, exc)
                continue
            except httpx.HTTPError as exc:
                raise ResearchProviderError(f"fetch failed for {url}: {exc}") from exc
        raise ResearchProviderError(f"fetch failed for {url}: {last_error}") from last_error

    async def _fetch_once(self, url: str, headers: dict[str, str]) -> PageResult:
        response = await self._throttled_get(url, headers)
        try:
            raw, truncated = await _read_capped(response, MAX_BODY_BYTES)
        finally:
            await response.aclose()
        status = response.status_code
        content_type = response.headers.get("content-type")
        out_headers = {k.lower(): v for k, v in response.headers.items() if k.lower() in _KEEP}
        if truncated:
            out_headers[TRUNCATED_HEADER] = "true"
        fetched_at = utcnow()
        final_url = str(response.url)
        if status in PAYWALL_STATUSES:
            out_headers[PAYWALL_HEADER] = "true"
            return PageResult(
                url=url,
                final_url=final_url,
                status=status,
                content_type=content_type,
                body="",
                fetched_at=fetched_at,
                headers=out_headers,
            )
        if status >= 400:
            raise ResearchProviderError(f"HTTP {status} for {url}")
        body = _decode(raw, response, content_type)
        if _looks_paywalled(body, content_type):
            out_headers[PAYWALL_HEADER] = "true"
            body = ""
        return PageResult(
            url=url,
            final_url=final_url,
            status=status,
            content_type=content_type,
            body=body,
            fetched_at=fetched_at,
            headers=out_headers,
        )

    # -- robots ---------------------------------------------------------------------------

    async def _robots_allow(self, scheme: str, netloc: str, url: str) -> bool:
        host = netloc.lower()
        if host not in self._robots:
            self._robots[host] = await self._load_robots(scheme, host)
        parser = self._robots[host]
        if parser is None:
            return True
        try:
            return parser.can_fetch(self._user_agent.split("/")[0] or "*", url)
        except Exception:  # robotparser is permissive; never let it break a fetch
            return True

    async def _load_robots(self, scheme: str, host: str) -> robotparser.RobotFileParser | None:
        robots_url = f"{scheme}://{host}/robots.txt"
        try:
            headers = {"User-Agent": self._user_agent, "Accept": "text/plain"}
            response = await self._throttled_get(robots_url, headers)
            try:
                raw, _ = await _read_capped(response, 256 * 1024)
            finally:
                await response.aclose()
            if response.status_code != 200:
                log.debug("robots.txt for %s -> %s, allowing", host, response.status_code)
                return None
            parser = robotparser.RobotFileParser()
            parser.parse(raw.decode("utf-8", errors="replace").splitlines())
            return parser
        except httpx.HTTPError as exc:
            log.debug("robots.txt for %s unavailable (%s), allowing", host, exc)
            return None

    # -- disk cache -----------------------------------------------------------------------

    def _cache_path(self, url: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json"

    def _cache_read(self, url: str, ttl_s: float) -> PageResult | None:
        path = self._cache_path(url)
        if path is None or ttl_s <= 0 or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            page = PageResult.model_validate(payload["page"])
        except (OSError, ValueError, KeyError):
            return None
        if page.fetched_at + timedelta(seconds=ttl_s) < utcnow():
            return None
        page.from_cache = True
        return page

    def _cache_write(self, url: str, page: PageResult) -> None:
        path = self._cache_path(url)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"url": url, "page": page.model_dump(mode="json")}),
                encoding="utf-8",
            )
            tmp.replace(path)
        except OSError as exc:
            log.debug("cache write failed for %s: %s", url, exc)


_KEEP = {"content-type", "last-modified", "etag", "content-length", "date", "cache-control"}


async def _read_capped(response: httpx.Response, limit: int) -> tuple[bytes, bool]:
    """Read at most ``limit`` bytes; ``truncated`` is conservative (true once the cap is hit)."""
    chunks: list[bytes] = []
    total = 0
    truncated = False
    async for chunk in response.aiter_bytes():
        if not chunk:
            continue
        remaining = limit - total
        if len(chunk) >= remaining:
            chunks.append(chunk[:remaining])
            total = limit
            truncated = True
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks), truncated


def _decode(raw: bytes, response: httpx.Response, content_type: str | None) -> str:
    encoding = response.charset_encoding
    if not encoding and content_type and "html" in content_type:
        match = _CHARSET_RE.search(raw[:4096])
        if match:
            encoding = match.group(1).decode("ascii", errors="ignore")
    for candidate in (encoding, "utf-8"):
        if not candidate:
            continue
        try:
            return raw.decode(candidate, errors="replace")
        except LookupError:
            continue
    return raw.decode("utf-8", errors="replace")


def _looks_paywalled(body: str, content_type: str | None) -> bool:
    if not body or (content_type and "html" not in content_type.lower()):
        return False
    lowered = body.lower()
    return any(marker in lowered for marker in PAYWALL_MARKERS)
