"""Instrument identification (AGENT.md sections 23, 24 "Ambiguous instrument").

Pure functions over the SEC company-ticker list. The query is scanned for cashtags, upper-case
ticker tokens and company names; anything that maps to more than one company, or to none,
raises ``AMBIGUOUS_INSTRUMENT`` with candidates so the client can ask the user.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.base import InstrumentCandidate, InstrumentIdentity
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.requests import InstrumentRef

_SUFFIXES = {
    "inc",
    "inc.",
    "corp",
    "corp.",
    "corporation",
    "co",
    "co.",
    "company",
    "ltd",
    "ltd.",
    "limited",
    "plc",
    "holdings",
    "holding",
    "group",
    "the",
    "incorporated",
    "n.v.",
    "nv",
    "sa",
    "s.a.",
    "ag",
    "se",
    "lp",
    "l.p.",
    "trust",
}

# Common upper-case tokens that are also tickers; never treated as tickers unless cashtagged.
_TICKER_STOPLIST = {
    "A",
    "AI",
    "ALL",
    "AM",
    "AN",
    "AND",
    "ARE",
    "AS",
    "AT",
    "BE",
    "BIG",
    "BY",
    "CEO",
    "CFO",
    "DO",
    "EPS",
    "EV",
    "FCF",
    "FOR",
    "GO",
    "HAS",
    "IN",
    "IPO",
    "IS",
    "IT",
    "ITS",
    "ME",
    "MY",
    "NEW",
    "NOW",
    "OF",
    "ON",
    "OR",
    "PE",
    "PS",
    "SO",
    "TO",
    "TTM",
    "UP",
    "US",
    "USA",
    "USD",
    "VS",
    "YOY",
    "QOQ",
    "GAAP",
    "ETF",
    "SEC",
    "Q",
    "FY",
}

# Well-known informal names -> normalized SEC titles (lower-case, suffix-stripped).
_ALIASES: dict[str, str] = {
    "google": "alphabet",
    "facebook": "meta platforms",
    "meta": "meta platforms",
    "square": "block",
    "berkshire": "berkshire hathaway",
    "jpmorgan": "jpmorgan chase",
    "jp morgan": "jpmorgan chase",
    "chase": "jpmorgan chase",
    "coke": "coca cola",
    "coca-cola": "coca cola",
    "p&g": "procter gamble",
    "procter and gamble": "procter gamble",
    "j&j": "johnson johnson",
    "johnson and johnson": "johnson johnson",
    "amex": "american express",
    "exxon": "exxon mobil",
    "exxonmobil": "exxon mobil",
    "goldman": "goldman sachs",
    "ibm": "international business machines",
    "walmart": "walmart",
    "mcdonalds": "mcdonalds",
    "mcdonald's": "mcdonalds",
    "amd": "advanced micro devices",
    "ge": "general electric",
    "tsmc": "taiwan semiconductor manufacturing",
    "bofa": "bank of america",
    "citi": "citigroup",
    "lilly": "eli lilly",
    "eli lilly": "eli lilly",
    "nvidia": "nvidia",
    "apple": "apple",
}

_WORD = re.compile(r"[A-Za-z0-9&.'\-]+")
_CASHTAG = re.compile(r"\$([A-Za-z]{1,5}(?:[.\-][A-Za-z])?)\b")
_UPPER_TOKEN = re.compile(r"\b([A-Z]{2,5}(?:[.\-][A-Z])?)\b")


@dataclass(frozen=True)
class TickerRow:
    cik: str
    ticker: str
    title: str
    exchange: str | None = None

    @classmethod
    def from_edgar(cls, row: dict) -> TickerRow:
        cik = row.get("cik_str", row.get("cik", ""))
        return cls(
            cik=str(cik).zfill(10) if cik != "" else "",
            ticker=str(row.get("ticker", "")).upper(),
            title=str(row.get("title", row.get("name", ""))),
            exchange=row.get("exchange"),
        )


def normalize_name(name: str) -> str:
    words = [w.strip(".,'\"()").lower() for w in _WORD.findall(name)]
    kept = [w.replace("&", " ").replace("'", "") for w in words if w and w not in _SUFFIXES]
    joined = " ".join(" ".join(kept).split())
    return joined.replace("-", " ").replace(".", "").strip()


def normalize_ticker(symbol: str) -> str:
    return symbol.strip().upper().replace(".", "-")


def _ngrams(words: list[str], max_n: int = 4) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for n in range(max_n, 0, -1):
        for i in range(0, len(words) - n + 1):
            out.append((i, i + n, " ".join(words[i : i + n])))
    return out


class InstrumentResolver:
    """Resolves a query (and optional explicit instrument) against a ticker list."""

    def __init__(self, rows: list[dict] | list[TickerRow]) -> None:
        self.rows: list[TickerRow] = [
            r if isinstance(r, TickerRow) else TickerRow.from_edgar(r) for r in rows
        ]
        self._by_ticker: dict[str, TickerRow] = {}
        self._by_name: dict[str, list[TickerRow]] = {}
        for row in self.rows:
            if not row.ticker:
                continue
            self._by_ticker.setdefault(normalize_ticker(row.ticker), row)
            self._by_name.setdefault(normalize_name(row.title), []).append(row)
        self._names = list(self._by_name)

    # -- public --------------------------------------------------------------------------
    def resolve(self, query: str, instrument: InstrumentRef | None = None) -> InstrumentIdentity:
        if instrument is not None:
            return self._resolve_ref(instrument)
        cashtags = [normalize_ticker(m) for m in _CASHTAG.findall(query)]
        if cashtags:
            return self._resolve_tickers(cashtags, "cashtag", query)
        upper = [
            normalize_ticker(tok)
            for tok in _UPPER_TOKEN.findall(query)
            if tok not in _TICKER_STOPLIST and normalize_ticker(tok) in self._by_ticker
        ]
        name_hits = self._name_matches(query)
        if upper and not name_hits:
            return self._resolve_tickers(upper, "ticker_token", query)
        if name_hits:
            ciks = {row.cik for row, _span in name_hits}
            if upper:
                ciks |= {self._by_ticker[t].cik for t in upper}
            if len(ciks) > 1:
                raise self._ambiguous(
                    query,
                    [self._candidate(row) for row, _ in name_hits]
                    + [self._candidate(self._by_ticker[t]) for t in upper],
                    reason="multiple_companies",
                )
            rows = [row for row, _ in name_hits]
            best = self._prefer_primary_class(rows)
            return self._identity(best, "name_match", confidence=0.9, siblings=rows)
        raise self._ambiguous(query, self._fuzzy_candidates(query), reason="no_match")

    # -- internals -----------------------------------------------------------------------
    def _resolve_ref(self, ref: InstrumentRef) -> InstrumentIdentity:
        row = self._by_ticker.get(normalize_ticker(ref.symbol))
        if row is None:
            raise AnalysisError(
                ErrorCode.AMBIGUOUS_INSTRUMENT,
                f"The symbol {ref.symbol} is not a listed US public equity we can resolve.",
                details={"reason": "unknown_symbol", "symbol": ref.symbol, "candidates": []},
            )
        identity = self._identity(row, "instrument_ref", confidence=1.0)
        if ref.exchange:
            identity.exchange = ref.exchange.upper()
        return identity

    def _resolve_tickers(self, tickers: list[str], method: str, query: str) -> InstrumentIdentity:
        rows = [self._by_ticker[t] for t in dict.fromkeys(tickers) if t in self._by_ticker]
        if not rows:
            raise self._ambiguous(query, self._fuzzy_candidates(query), reason="unknown_symbol")
        if len({r.cik for r in rows}) > 1:
            raise self._ambiguous(
                query, [self._candidate(r) for r in rows], reason="multiple_companies"
            )
        return self._identity(rows[0], method, confidence=0.98, siblings=rows)

    def _name_matches(self, query: str) -> list[tuple[TickerRow, tuple[int, int]]]:
        words = [w.strip(".,'\"()?!:;").lower() for w in _WORD.findall(query)]
        words = [w for w in words if w]
        hits: list[tuple[TickerRow, tuple[int, int]]] = []
        taken: list[tuple[int, int]] = []
        for start, end, phrase in _ngrams(words):
            if any(s < end and start < e for s, e in taken):
                continue
            key = _ALIASES.get(phrase, phrase)
            key = normalize_name(key)
            rows = self._by_name.get(key)
            if not rows and len(phrase.split()) >= 2:
                rows = self._by_name.get(normalize_name(phrase))
            if rows:
                for row in rows:
                    hits.append((row, (start, end)))
                taken.append((start, end))
        return hits

    @staticmethod
    def _prefer_primary_class(rows: list[TickerRow]) -> TickerRow:
        # Same company, several share classes (GOOGL/GOOG): prefer the shortest/plain ticker.
        return sorted(rows, key=lambda r: (len(r.ticker), r.ticker))[0]

    def _identity(
        self,
        row: TickerRow,
        method: str,
        confidence: float,
        siblings: list[TickerRow] | None = None,
    ) -> InstrumentIdentity:
        aliases = sorted({r.ticker for r in (siblings or []) if r.ticker != row.ticker})
        return InstrumentIdentity(
            symbol=row.ticker,
            exchange=row.exchange,
            name=row.title,
            cik=row.cik or None,
            aliases=aliases,
            confidence=confidence,
            resolution_method=method,
        )

    @staticmethod
    def _candidate(row: TickerRow) -> InstrumentCandidate:
        return InstrumentCandidate(
            symbol=row.ticker, exchange=row.exchange, name=row.title, cik=row.cik or None
        )

    def _fuzzy_candidates(self, query: str, limit: int = 3) -> list[InstrumentCandidate]:
        words = [w.strip(".,'\"()?!:;").lower() for w in _WORD.findall(query)]
        out: list[InstrumentCandidate] = []
        seen: set[str] = set()
        for _start, _end, phrase in _ngrams([w for w in words if w], max_n=3):
            for name in difflib.get_close_matches(
                normalize_name(phrase), self._names, n=2, cutoff=0.86
            ):
                for row in self._by_name[name]:
                    if row.cik not in seen:
                        seen.add(row.cik)
                        out.append(self._candidate(row))
            if len(out) >= limit:
                break
        return out[:limit]

    @staticmethod
    def _ambiguous(query: str, candidates: list[InstrumentCandidate], reason: str) -> AnalysisError:
        unique: dict[str, InstrumentCandidate] = {}
        for cand in candidates:
            unique.setdefault(cand.symbol, cand)
        message = (
            "Which company did you mean?"
            if unique
            else "No public company could be identified in the request."
        )
        return AnalysisError(
            ErrorCode.AMBIGUOUS_INSTRUMENT,
            message,
            details={
                "reason": reason,
                "candidates": [c.model_dump() for c in unique.values()],
            },
        )
