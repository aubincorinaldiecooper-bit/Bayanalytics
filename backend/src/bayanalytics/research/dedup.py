"""Duplicate detection across a research run (AGENT.md section 21: deduplicate before rank).

Three keys are tracked: exact content hash, canonical URL and a normalised title prefix so
syndicated copies and tracking-parameter variants of one story are counted once.
"""

from __future__ import annotations

import re

from bayanalytics.research.provider import EvidenceRecord
from bayanalytics.research.sources import canonical_url

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
TITLE_PREFIX_CHARS = 80


def normalise_title(title: str) -> str:
    return _NON_ALNUM.sub("", title.lower())[:TITLE_PREFIX_CHARS]


class Deduplicator:
    def __init__(self) -> None:
        self._hashes: set[str] = set()
        self._urls: set[str] = set()
        self._titles: set[str] = set()
        self.duplicates = 0

    def seen(self, url_or_hash: str, *, title: str | None = None) -> bool:
        """Register a URL or a sha256 content hash; True when it (or its title) was seen."""
        value = url_or_hash.strip()
        duplicate = False
        if _HEX64.match(value):
            if value in self._hashes:
                duplicate = True
            self._hashes.add(value)
        elif value:
            key = canonical_url(value)
            if key in self._urls:
                duplicate = True
            self._urls.add(key)
        if title:
            normalised = normalise_title(title)
            if len(normalised) >= 12:
                if normalised in self._titles:
                    duplicate = True
                self._titles.add(normalised)
        if duplicate:
            self.duplicates += 1
        return duplicate

    def known_url(self, url: str) -> bool:
        """True when the canonical form of ``url`` was registered before (no side effects)."""
        return canonical_url(url) in self._urls

    def seen_record(self, record: EvidenceRecord, *, check_url: bool = True) -> bool:
        """Check hash, canonical URL and title of one evidence record; counts at most once."""
        url_dup = self.seen(record.final_url or record.url) if check_url else False
        hash_dup = False
        if record.content_hash and record.text:
            hash_dup = self.seen(record.content_hash)
        title_dup = self.seen("", title=record.title)
        total = int(url_dup) + int(hash_dup) + int(title_dup)
        if total > 1:
            self.duplicates -= total - 1
        return total > 0

    @property
    def unique_count(self) -> int:
        return len(self._urls)
