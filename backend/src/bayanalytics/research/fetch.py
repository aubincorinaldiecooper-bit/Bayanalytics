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

Target policy (SSRF hardening). Search hits and redirects are attacker-influenced, so before
any connection the fetcher checks that the URL is ``http``/``https`` and that every address
the host resolves to is globally routable: loopback, link-local (cloud metadata), RFC 1918,
CGNAT, multicast, reserved and unspecified addresses are refused with the fixed reason
``blocked_target`` (never echoing the address). Redirects are followed manually, at most
``MAX_REDIRECTS`` hops, and every ``Location`` is re-validated the same way. Hosts named in
``allowed_hosts`` (``host`` or ``host:port``, e.g. a SearXNG instance on a LAN address) are
exempt; ``allow_private_hosts=True`` disables the policy entirely for local development and
tests. Residual risk: the check resolves the name itself and the transport resolves it again
when connecting, so a DNS-rebinding attacker who answers with a public address first and a
private one a moment later can still reach a private target for one request; closing that
gap needs address pinning inside the transport. A name that does not resolve at all is left
to the transport, which then fails to connect.

Cache policy (data rights). The on-disk cache is transient scratch space, not an archive:
full bodies are written only for structured / public-domain hosts (``CACHE_BODY_HOSTS``:
SEC EDGAR and Stooq); pages from journalism and other third-party hosts are never stored.
Every write also sweeps the directory (bounded scan) and unlinks files older than
``CACHE_MAX_AGE_S``, and reads unlink entries that have expired for the caller's TTL.

Every failure is raised as ``ResearchProviderError``; failures the fetcher classifies carry a
fixed, client-safe ``reason`` keyword (``TaggedProviderError``). Raw transport text stays in
the exception message and logs and never reaches clients.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import re
import socket
import time
from collections.abc import Awaitable, Callable, Iterable
from datetime import timedelta
from pathlib import Path
from typing import Protocol
from urllib import robotparser
from urllib.parse import SplitResult, urlsplit

import httpx

from bayanalytics.config import Settings
from bayanalytics.research.provider import PageResult, ResearchProviderError
from bayanalytics.schemas.common import utcnow

log = logging.getLogger(__name__)

MAX_BODY_BYTES = 2 * 1024 * 1024
MAX_REDIRECTS = 5
DEFAULT_TTL_S = 6 * 3600.0
EDGAR_TICKERS_TTL_S = 24 * 3600.0
PRICE_CSV_TTL_S = 3600.0
SEC_HOSTS: frozenset[str] = frozenset({"sec.gov", "www.sec.gov", "data.sec.gov", "efts.sec.gov"})
SEC_MIN_INTERVAL_S = 0.11  # SEC fair-access policy: at most 10 requests per second
# Hosts whose bodies may be stored on disk: US government works (EDGAR) and Stooq CSVs that
# the price client only keeps as metadata + derived series. Everything else is metadata-only.
CACHE_BODY_HOSTS: frozenset[str] = frozenset(SEC_HOSTS | {"stooq.com"})
CACHE_MAX_AGE_S = EDGAR_TICKERS_TTL_S  # longest TTL any caller uses; bounds on-disk retention
CACHE_SWEEP_LIMIT = 256  # directory entries examined per sweep
PAYWALL_HEADER = "x-bay-paywalled"
TRUNCATED_HEADER = "x-bay-truncated"
PAYWALL_STATUSES: frozenset[int] = frozenset({401, 402, 403})
REDIRECT_STATUSES: frozenset[int] = frozenset({301, 302, 303, 307, 308})
ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})
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
_LEGACY_IPV4_RE = re.compile(r"^[0-9a-fx.]+$", re.IGNORECASE)

# Fixed reason keywords (safe to show to clients). The runner emits only these.
REASON_BLOCKED = "blocked_target"
REASON_FETCH_FAILED = "fetch_failed"
REASON_TIMEOUT = "timeout"
REASON_ROBOTS = "robots_disallowed"
REASON_EXTRACT_FAILED = "extract_failed"

Resolver = Callable[[str], Awaitable[list[str]]]


class TaggedProviderError(ResearchProviderError):
    """``ResearchProviderError`` carrying a fixed ``reason`` keyword.

    ``reason`` is the only part clients may see (runner events, error details); ``detail``
    is for the exception text and logs.
    """

    def __init__(self, reason: str, detail: str | None = None) -> None:
        super().__init__(reason if detail is None else f"{reason}: {detail}")
        self.reason = reason


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


# --------------------------------------------------------------------------------------
# target policy
# --------------------------------------------------------------------------------------


def _literal_address(host: str) -> str | None:
    """Return the address for a literal IP host (including legacy ``127.1`` / decimal forms)."""
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass
    if _LEGACY_IPV4_RE.match(host):
        try:
            return socket.inet_ntoa(socket.inet_aton(host))
        except OSError:
            return None
    return None


async def resolve_host(host: str) -> list[str]:
    """Resolve ``host`` to its addresses off the event loop; ``[]`` when it does not resolve."""
    literal = _literal_address(host)
    if literal is not None:
        return [literal]
    try:
        infos = await asyncio.to_thread(socket.getaddrinfo, host, None, type=socket.SOCK_STREAM)
    except OSError:  # socket.gaierror and friends: NXDOMAIN, no resolver, ...
        return []
    return sorted({str(info[4][0]) for info in infos})


def address_blocked(text: str) -> bool:
    """True unless ``text`` is a globally routable unicast address."""
    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        return True  # refuse anything the resolver hands back that we cannot classify
    candidates: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = [addr]
    for attribute in ("ipv4_mapped", "sixtofour"):
        embedded = getattr(addr, attribute, None)
        if embedded is not None:
            candidates.append(embedded)
    return any(
        not c.is_global
        or c.is_loopback
        or c.is_link_local
        or c.is_private
        or c.is_multicast
        or c.is_reserved
        or c.is_unspecified
        for c in candidates
    )


def split_checked(url: str) -> SplitResult:
    """Split ``url`` and enforce the scheme/host policy; raises ``blocked_target`` otherwise."""
    try:
        parts = urlsplit(url)
        host = parts.hostname
        parts.port  # noqa: B018  # validates the port syntax
    except ValueError:
        raise TaggedProviderError(REASON_BLOCKED) from None
    if parts.scheme not in ALLOWED_SCHEMES or not host or not parts.netloc:
        raise TaggedProviderError(REASON_BLOCKED)
    return parts


def host_key(parts: SplitResult) -> tuple[str, str]:
    """``(host, host:port)`` for allow-list matching, with the scheme's default port."""
    host = (parts.hostname or "").rstrip(".").lower()
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return host, f"{host}:{port}"


def search_backend_hosts(base_url: str | None) -> frozenset[str]:
    """Allow-list entry (``host:port``) for the configured SearXNG base URL, if any."""
    if not base_url or not base_url.strip():
        return frozenset()
    try:
        parts = urlsplit(base_url.strip())
        if parts.scheme not in ALLOWED_SCHEMES or not parts.hostname:
            return frozenset()
        return frozenset({host_key(parts)[1]})
    except ValueError:
        return frozenset()


class Fetcher:
    def __init__(
        self,
        settings: Settings,
        http: httpx.AsyncClient | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        allowed_hosts: Iterable[str] | None = None,
        allow_private_hosts: bool = False,
        cache_body_hosts: Iterable[str] | None = None,
        resolver: Resolver | None = None,
    ) -> None:
        self._settings = settings
        self._owns_client = http is None
        self._http = http or httpx.AsyncClient(follow_redirects=False)
        self._user_agent = settings.user_agent
        self._timeout = settings.research_fetch_timeout_s
        self._interval = max(0.0, settings.research_min_request_interval_s)
        self._throttle = _HostThrottle(clock, sleep)
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}
        self._cache_dir: Path | None = settings.research_cache_dir
        self._allowed_hosts = frozenset(h.lower() for h in (allowed_hosts or ()))
        self._allow_private_hosts = allow_private_hosts
        self._cache_body_hosts = (
            CACHE_BODY_HOSTS if cache_body_hosts is None else frozenset(cache_body_hosts)
        )
        self._resolver = resolver

    @property
    def user_agent(self) -> str:
        return self._user_agent

    @property
    def allowed_hosts(self) -> frozenset[str]:
        return self._allowed_hosts

    async def aclose(self) -> None:
        if self._owns_client:
            await self._http.aclose()

    # -- public ---------------------------------------------------------------------------

    async def open(
        self, url: str, *, ttl_s: float | None = None, accept: str | None = None
    ) -> PageResult:
        parts = split_checked(url)  # scheme + literal-host policy before anything else
        ttl = DEFAULT_TTL_S if ttl_s is None else ttl_s
        cached = self._cache_read(url, ttl)
        if cached is not None:
            return cached
        # Resolution-based policy runs inside _throttled_get for every request (robots, page
        # and each redirect hop), so a host resolving to a private address is refused before
        # its robots.txt is even asked for.
        if not await self._robots_allow(parts.scheme, parts.netloc, url):
            raise TaggedProviderError(REASON_ROBOTS, f"robots.txt disallows {url}")
        page = await self._fetch_with_retry(url, accept)
        if 200 <= page.status < 300 and page.headers.get(PAYWALL_HEADER) != "true":
            self._cache_write(url, page, ttl)
        return page

    # -- target policy --------------------------------------------------------------------

    async def _assert_target_allowed(self, parts: SplitResult) -> None:
        if self._allow_private_hosts:
            return
        host, host_port = host_key(parts)
        if host in self._allowed_hosts or host_port in self._allowed_hosts:
            return
        if host == "localhost" or host.endswith(".localhost"):
            raise TaggedProviderError(REASON_BLOCKED)
        resolver = self._resolver or resolve_host
        try:
            addresses = await asyncio.wait_for(resolver(host), timeout=self._timeout)
        except TimeoutError:
            raise TaggedProviderError(REASON_TIMEOUT, f"resolving {host} timed out") from None
        if any(address_blocked(address) for address in addresses):
            log.warning("refusing to fetch %s: host resolves to a non-global address", host)
            raise TaggedProviderError(REASON_BLOCKED)

    # -- throttling -----------------------------------------------------------------------

    def interval_for(self, host: str) -> float:
        if throttle_key(host) == "sec.gov":
            return max(SEC_MIN_INTERVAL_S, self._interval)
        return self._interval

    async def _throttled_get(self, url: str, headers: dict[str, str]) -> httpx.Response:
        parts = split_checked(url)
        await self._assert_target_allowed(parts)
        host = parts.netloc
        await self._throttle.wait(throttle_key(host), self.interval_for(host))
        try:
            request = self._http.build_request("GET", url, headers=headers, timeout=self._timeout)
            # Redirects are followed manually (see _fetch_once) so every hop is re-validated.
            # httpx still parses a redirect's Location to fill ``response.next_request`` and
            # raises InvalidURL (not an HTTPError) for e.g. ``data:`` targets.
            return await self._http.send(request, stream=True, follow_redirects=False)
        except httpx.InvalidURL as exc:
            raise TaggedProviderError(
                REASON_FETCH_FAILED, f"invalid url or redirect for {url}: {exc}"
            ) from exc

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
                raise TaggedProviderError(
                    REASON_FETCH_FAILED, f"fetch failed for {url}: {exc}"
                ) from exc
        reason = REASON_TIMEOUT if isinstance(last_error, httpx.TimeoutException) else None
        raise TaggedProviderError(
            reason or REASON_FETCH_FAILED, f"fetch failed for {url}: {last_error}"
        ) from last_error

    async def _fetch_once(self, url: str, headers: dict[str, str]) -> PageResult:
        current = url
        response: httpx.Response | None = None
        for _hop in range(MAX_REDIRECTS + 1):
            response = await self._throttled_get(current, headers)
            location = response.headers.get("location")
            if response.status_code not in REDIRECT_STATUSES or not location:
                break
            await response.aclose()
            response = None
            current = _redirect_target(current, location)
        if response is None:
            raise TaggedProviderError(REASON_FETCH_FAILED, f"too many redirects for {url}")
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
        final_url = current
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
            raise TaggedProviderError(REASON_FETCH_FAILED, f"HTTP {status} for {url}")
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

    # -- disk cache (transient; see the module docstring) ---------------------------------

    def _cache_path(self, url: str) -> Path | None:
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json"

    def _body_cacheable(self, url: str) -> bool:
        host = (urlsplit(url).hostname or "").rstrip(".").lower()
        return host in self._cache_body_hosts

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
            _unlink_quietly(path)
            return None
        page.from_cache = True
        return page

    def _cache_write(self, url: str, page: PageResult, ttl_s: float) -> None:
        path = self._cache_path(url)
        if path is None:
            return
        if not self._body_cacheable(url):
            log.debug("not caching body for %s (host not allow-listed)", url)
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            expires_at = page.fetched_at + timedelta(seconds=ttl_s)
            tmp.write_text(
                json.dumps(
                    {
                        "url": url,
                        "ttl_s": ttl_s,
                        "expires_at": expires_at.isoformat(),
                        "page": page.model_dump(mode="json"),
                    }
                ),
                encoding="utf-8",
            )
            tmp.replace(path)
        except OSError as exc:
            log.debug("cache write failed for %s: %s", url, exc)
        self._sweep_expired(path.parent)

    @staticmethod
    def _sweep_expired(directory: Path, now: float | None = None) -> None:
        """Unlink cache files older than ``CACHE_MAX_AGE_S`` (bounded scan, stat only)."""
        now = time.time() if now is None else now
        try:
            with os.scandir(directory) as entries:
                for index, entry in enumerate(entries):
                    if index >= CACHE_SWEEP_LIMIT:
                        break
                    if not entry.name.endswith((".json", ".tmp")):
                        continue
                    try:
                        if now - entry.stat().st_mtime > CACHE_MAX_AGE_S:
                            os.unlink(entry.path)
                    except OSError:
                        continue
        except OSError as exc:
            log.debug("cache sweep of %s failed: %s", directory, exc)


_KEEP = {"content-type", "last-modified", "etag", "content-length", "date", "cache-control"}


def _redirect_target(current: str, location: str) -> str:
    try:
        target = str(httpx.URL(current).join(location))
    except (httpx.InvalidURL, ValueError) as exc:
        raise TaggedProviderError(REASON_FETCH_FAILED, f"invalid redirect from {current}") from exc
    split_checked(target)  # scheme/host policy on every hop; resolution follows on the GET
    return target


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


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
