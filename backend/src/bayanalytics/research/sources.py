"""Source classification, URL canonicalisation and provenance records.

The classification table (AGENT.md section 22 hierarchy, section 26 data rights) is a plain
module constant so new publishers are a one-line addition. Classification is host/path based
and never trusts page content for the source type. The table only labels pages a web search
returned (type, publisher, redistribution terms); it never chooses what to fetch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from bayanalytics.research.dates import ensure_utc
from bayanalytics.research.extract import excerpt_of, registrable_domain
from bayanalytics.research.provider import EvidenceRecord
from bayanalytics.schemas.common import Freshness, Redistribution, SourceType, stable_id
from bayanalytics.schemas.evidence import SourceRecord

CURRENT_MAX_DAYS = 3
RECENT_MAX_DAYS = 30

_TRACKING_PARAMS = frozenset(
    {
        "fbclid",
        "gclid",
        "dclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "igshid",
        "ref",
        "ref_src",
        "cmpid",
        "yptr",
        "ncid",
        "s_kwcid",
    }
)


@dataclass(frozen=True, slots=True)
class SourceRule:
    """One row of the classification table."""

    source_type: SourceType
    redistribution: Redistribution
    hosts: tuple[str, ...] = ()  # registrable-domain suffix match ("reuters.com")
    host_prefixes: tuple[str, ...] = ()  # subdomain prefix match ("investor.")
    path_markers: tuple[str, ...] = ()  # substring of the lower-cased path
    publisher: str | None = None  # fixed publisher name, else derived from the host
    terms_note: str | None = None


SOURCE_RULES: tuple[SourceRule, ...] = (
    SourceRule(
        source_type="regulatory_filing",
        redistribution="allowed",
        hosts=("sec.gov",),
        publisher="SEC EDGAR",
        terms_note="US government work; SEC fair-access policy applies to retrieval",
    ),
    SourceRule(
        source_type="financial_journalism",
        redistribution="metadata_only",
        hosts=(
            "reuters.com",
            "bloomberg.com",
            "wsj.com",
            "ft.com",
            "cnbc.com",
            "barrons.com",
            "marketwatch.com",
            "nytimes.com",
        ),
        terms_note="Copyrighted journalism: store metadata, short excerpt and link only",
    ),
    SourceRule(
        source_type="secondary_commentary",
        redistribution="metadata_only",
        hosts=("seekingalpha.com", "fool.com", "investing.com", "benzinga.com", "zacks.com"),
        terms_note="Commentary site: store metadata, short excerpt and link only",
    ),
    SourceRule(
        source_type="earnings_release",
        redistribution="metadata_only",
        hosts=("businesswire.com", "prnewswire.com", "globenewswire.com"),
        terms_note="Press-release wire: issuer content, store excerpt and link",
    ),
    SourceRule(
        source_type="investor_relations",
        redistribution="metadata_only",
        host_prefixes=("investor.", "investors.", "ir."),
        path_markers=("/investor", "/investors"),
        terms_note="Issuer investor-relations material",
    ),
)

UNKNOWN_RULE = SourceRule(source_type="unverified_web", redistribution="unknown")


def _host_matches(host: str, suffix: str) -> bool:
    return host == suffix or host.endswith("." + suffix)


def classify_source(
    url: str, publisher: str | None = None
) -> tuple[SourceType, str | None, Redistribution, str | None]:
    """Return ``(source_type, publisher_name, redistribution, terms_note)`` for a URL."""
    parts = urlsplit(url)
    host = parts.netloc.lower().split("@")[-1].split(":")[0]
    path = parts.path.lower()
    for rule in SOURCE_RULES:
        matched = any(_host_matches(host, suffix) for suffix in rule.hosts)
        if not matched and rule.host_prefixes:
            matched = any(host.startswith(prefix) for prefix in rule.host_prefixes)
        if not matched and rule.path_markers:
            matched = any(marker in path for marker in rule.path_markers)
        if matched:
            name = rule.publisher or publisher or registrable_domain(host) or None
            return rule.source_type, name, rule.redistribution, rule.terms_note
    name = publisher or registrable_domain(host) or None
    return UNKNOWN_RULE.source_type, name, UNKNOWN_RULE.redistribution, UNKNOWN_RULE.terms_note


def domain_of(url: str) -> str:
    """Registrable domain of a URL (``www.cnbc.com`` -> ``cnbc.com``); ``""`` when it has none."""
    if not url:
        return ""
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return ""
    return registrable_domain(host) if host else ""


def web_pages(sources: list[SourceRecord]) -> list[SourceRecord]:
    """Kept sources with extracted text: the web pages an assessment can rest on."""
    return [
        s for s in sources if not s.rejected_reason and int(s.metadata.get("text_chars") or 0) > 0
    ]


def canonical_url(url: str) -> str:
    """Stable key for one page: lower-cased host, no tracking params, fragment or trailing /."""
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "https").lower()
    host = parts.netloc.lower()
    if (scheme == "https" and host.endswith(":443")) or (scheme == "http" and host.endswith(":80")):
        host = host.rsplit(":", 1)[0]
    params = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
    ]
    params.sort()
    query = urlencode(params, doseq=True)
    path = parts.path.rstrip("/") if parts.path not in ("", "/") else ""
    return urlunsplit((scheme, host, path, query, ""))


def classify_freshness(published_at: datetime | None, as_of: datetime) -> Freshness:
    if published_at is None:
        return "unknown"
    age = ensure_utc(as_of) - ensure_utc(published_at)
    if age <= timedelta(days=CURRENT_MAX_DAYS):
        return "current"
    if age <= timedelta(days=RECENT_MAX_DAYS):
        return "recent"
    return "stale"


def source_record_from_evidence(
    record: EvidenceRecord,
    symbol: str | None,
    intent: str | None,
    as_of: datetime,
    *,
    fiscal_period: str | None = None,
) -> SourceRecord:
    """Provenance for one extracted page (AGENT.md section 5). Stores only an excerpt."""
    source_type, publisher, redistribution, note = classify_source(
        record.final_url or record.url, record.publisher
    )
    metadata = {
        "final_url": record.final_url,
        "canonical_url": canonical_url(record.final_url or record.url),
        "language": record.language,
        "text_chars": len(record.text),
    }
    for key in ("paywalled", "truncated", "from_cache", "status"):
        if key in record.metadata:
            metadata[key] = record.metadata[key]
    return SourceRecord(
        source_id=stable_id("src", canonical_url(record.final_url or record.url)),
        url=record.final_url or record.url,
        title=record.title,
        publisher=publisher,
        source_type=source_type,
        published_at=record.published_at,
        retrieved_at=record.retrieved_at,
        symbol=symbol,
        fiscal_period=fiscal_period,
        excerpt=record.excerpt or excerpt_of(record.text),
        content_hash=record.content_hash,
        extraction_method=record.extraction_method,
        freshness=classify_freshness(record.published_at, as_of),
        redistribution=redistribution,
        terms_note=note,
        research_intent=intent,
        metadata=metadata,
    )
