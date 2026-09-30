"""Instrument identification (AGENT.md sections 23, 24 "Ambiguous instrument").

Research is web search only, so there is no company directory, and nothing in this module
touches the network. A ticker the analyst gave always wins, in this order:

1. an explicit ``InstrumentRef`` (its symbol, with the exchange as given);
2. cashtags in the question (``$AAPL``, ``$BRK.B``);
3. ticker-like upper-case tokens (1-5 letters, optional one-letter class suffix such as
   ``BRK.B``) that are not common upper-case words (``_TICKER_STOPLIST``).

Several distinct symbols raise ``AMBIGUOUS_INSTRUMENT`` (reason ``multiple_tickers``) with the
symbols as candidates; none raises it with reason ``ticker_required``.

A question that names a company instead ("Assess apple.") is resolved by the analysis, not
here: ``company_phrase`` takes the company phrase out of the question, the analyzer runs one
topic-only web search (``name_query``), ``ticker_candidates`` collects the tickers the result
titles and snippets name next to a company name, ranked by how many distinct websites name
them, and Laya chooses among those candidates (``EquityAnalyzer.identify``). A name is never
guessed: it is only what the search results wrote next to the ticker.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.base import InstrumentCandidate, InstrumentIdentity
from bayanalytics.research.sources import domain_of
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.requests import InstrumentRef

TICKER_REQUIRED_MESSAGE = "Include the company's stock ticker in your question, for example $AAPL."
MULTIPLE_TICKERS_MESSAGE = "Which company did you mean?"
MAX_NAME_CANDIDATES = 5
MAX_PHRASE_WORDS = 5

# Upper-case words that look like tickers but, in a question, almost never are. A cashtag
# always wins over this list ($ALL, $NOW, $AI are tickers when written that way).
_TICKER_STOPLIST: frozenset[str] = frozenset(
    {
        # articles, pronouns, conjunctions, prepositions, question words
        "A", "AN", "AND", "ARE", "AS", "AT", "BE", "BUT", "BY", "CAN", "DO", "DOES", "FOR",
        "HAS", "HOW", "I", "IF", "IN", "IS", "IT", "ITS", "ME", "MY", "NO", "NOT", "OF", "ON",
        "OR", "OUR", "SO", "THE", "TO", "UP", "VS", "WE", "WHAT", "WHEN", "WHO",
        "WHY", "WILL", "YOU", "ALL", "ANY", "NEW", "NOW", "OUT", "OK", "BIG", "GO", "AM",
        # finance and reporting vocabulary
        "AI", "ADR", "API", "ARPU", "ARR", "ASP", "ATH", "BPS", "CAGR", "CAPEX", "CEO", "CFO",
        "COO", "CPI", "CTO", "DCF", "EBIT", "EPS", "ESG", "ETF", "EV", "FCF", "FED", "FOMC",
        "FX", "FY", "GAAP", "GDP", "IFRS", "IPO", "IR", "KPI", "LTM", "MOM", "NAV", "OTC",
        "PE", "PEG", "PR", "PS", "Q", "QOQ", "QTD", "ROA", "ROE", "ROI", "ROIC", "SEC", "TAM",
        "TTM", "YOY", "YTD", "BUY", "SELL", "HOLD", "LONG", "SHORT", "PUT", "CALL",
        # places, currencies, legal forms
        "US", "USA", "UK", "EU", "U.S", "U.K", "E.U", "USD", "EUR", "GBP", "JPY", "CNY", "CAD",
        "INC", "LTD", "PLC", "LLC", "CORP", "CO", "AG", "SA", "SE", "NV", "LP",
    }
)  # fmt: skip

_CASHTAG = re.compile(r"\$([A-Za-z]{1,5}(?:[.\-][A-Za-z])?)\b")
_TICKER = re.compile(r"[A-Z]{1,5}(?:[.\-][A-Z])?")
_EXPLICIT_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9.\-]{0,15}")
_LEADING = "\"'([{<"
_TRAILING = "\"')]}>.,;:!?"
_APOSTROPHES = str.maketrans({"\u2019": "'", "\u02bc": "'", "\uff07": "'"})


def symbol_key(symbol: str) -> str:
    """Comparison key for one symbol: ``BRK-B`` and ``BRK.B`` are the same share class."""
    return symbol.strip().upper().replace("-", ".")


def cashtags(query: str) -> list[str]:
    """``$aapl`` / ``$BRK.B`` in the order written, upper-cased."""
    return [m.upper() for m in _CASHTAG.findall(query)]


def ticker_tokens(query: str) -> list[str]:
    """Whitespace-separated tokens that are a ticker as a whole once surrounding punctuation
    and a possessive ``'s`` are removed (``AAPL's`` -> ``AAPL``; ``S&P`` and ``P/E`` are
    not tickers), minus the stoplist."""
    out: list[str] = []
    for raw in query.translate(_APOSTROPHES).split():
        token = raw.lstrip(_LEADING).rstrip(_TRAILING)
        if token.endswith("'s") or token.endswith("'S"):
            token = token[:-2]
        if _TICKER.fullmatch(token) and token not in _TICKER_STOPLIST:
            out.append(token)
    return out


class InstrumentResolver:
    """Resolves a query (and optional explicit instrument) to the ticker the analyst gave."""

    def resolve(self, query: str, instrument: InstrumentRef | None = None) -> InstrumentIdentity:
        if instrument is not None:
            return self._resolve_ref(instrument)
        tagged = cashtags(query)
        if tagged:
            return self._resolve_symbols(tagged, "cashtag", confidence=0.98)
        tokens = ticker_tokens(query)
        if tokens:
            return self._resolve_symbols(tokens, "ticker_token", confidence=0.9)
        raise AnalysisError(
            ErrorCode.AMBIGUOUS_INSTRUMENT,
            TICKER_REQUIRED_MESSAGE,
            details={"reason": "ticker_required", "candidates": []},
        )

    @staticmethod
    def _resolve_ref(ref: InstrumentRef) -> InstrumentIdentity:
        symbol = ref.symbol.strip().upper()
        if not _EXPLICIT_SYMBOL.fullmatch(symbol):
            raise AnalysisError(
                ErrorCode.AMBIGUOUS_INSTRUMENT,
                TICKER_REQUIRED_MESSAGE,
                details={"reason": "invalid_symbol", "candidates": []},
            )
        exchange = ref.exchange.strip().upper() if ref.exchange and ref.exchange.strip() else None
        return InstrumentIdentity(
            symbol=symbol,
            exchange=exchange,
            name="",
            confidence=1.0,
            resolution_method="instrument_ref",
        )

    @staticmethod
    def _resolve_symbols(symbols: list[str], method: str, confidence: float) -> InstrumentIdentity:
        distinct: dict[str, str] = {}
        for symbol in symbols:
            distinct.setdefault(symbol_key(symbol), symbol)
        if len(distinct) > 1:
            raise AnalysisError(
                ErrorCode.AMBIGUOUS_INSTRUMENT,
                MULTIPLE_TICKERS_MESSAGE,
                details={
                    "reason": "multiple_tickers",
                    "candidates": [
                        InstrumentCandidate(symbol=s, name="").model_dump()
                        for s in distinct.values()
                    ],
                },
            )
        (symbol,) = distinct.values()
        return InstrumentIdentity(
            symbol=symbol,
            name="",
            confidence=confidence,
            resolution_method=method,
        )


# ---- company names (the analyzer resolves them through web search and Laya) ---------------

# Words around a company name in a question: requests, question words, finance vocabulary and
# time words. What is left, in order, is the company phrase.
_PHRASE_STOPWORDS: frozenset[str] = frozenset(
    """
    a about above after against all also am an analyse analysis analyze and annual any are as
    assess assessment at be been before being below between bullish bearish buy by can cheap
    check company compare could current did do does doing done down during earnings evaluate
    expensive financial financials for from future give go going good growth guidance had has
    have how i if in into invest investing investment is it its last latest like long look
    looking looks margin margins market me medium month months more most my near news next now
    of on or our outlook over overvalued performance performing please price prices profit
    profitability prospects quarter quarterly recent recently report reports results return
    returns revenue review risk risks sales sell share shares short should show shows stock
    stocks tell term than that the their them then there these think thoughts this those
    through to today trend trends undervalued up valuation value view views vs want was we week
    weeks were what whats when where which who why will with worth would year years you your
    """.split()
)
_PHRASE_TOKEN = re.compile(r"&|[A-Za-z0-9][A-Za-z0-9&.'\-]*")
_EXCHANGES = (
    r"NASDAQ|Nasdaq|NasdaqGS|NasdaqGM|NasdaqCM|NYSE|NYSEARCA|NYSE American|NYSE Arca|AMEX|OTC|"
    r"TSX|LSE"
)
_SYMBOL = r"[A-Z]{1,5}(?:[.\-][A-Z])?"
_NAME_BEFORE = re.compile(
    rf"([A-Z][\w&.,'\-]*(?:\s+(?:&\s+)?[A-Z][\w&.,'\-]*){{0,5}})\s*"
    rf"\((?:({_EXCHANGES})\s*:\s*)?({_SYMBOL})\)"
)
_EXCHANGE_PREFIX = re.compile(rf"\b({_EXCHANGES})\s*:\s*({_SYMBOL})\b")
_SYMBOL_STOCK = re.compile(rf"\b({_SYMBOL})\s+(?:stock|shares|share price)\b")
_NOT_SYMBOLS = frozenset(
    {"NASDAQ", "NYSE", "NYSEARCA", "AMEX", "OTC", "TSX", "LSE", "ETF", "NONE", "INC", "LTD"}
)


def company_phrase(query: str) -> str | None:
    """The company phrase of a question without a ticker ("Assess apple." -> ``apple``,
    "How is Johnson & Johnson doing?" -> ``Johnson & Johnson``): the first run of words
    that are not request, question, finance or time words, at most ``MAX_PHRASE_WORDS``.
    ``None`` when nothing is left."""
    runs: list[list[str]] = [[]]
    for raw in _PHRASE_TOKEN.findall(query.translate(_APOSTROPHES)):
        token = raw.rstrip(".'-")
        if token.endswith("'s") or token.endswith("'S"):
            token = token[:-2]
        if not token or token.lower() in _PHRASE_STOPWORDS or token.isdigit():
            if runs[-1]:
                runs.append([])
            continue
        runs[-1].append(token)
    words = next((run for run in runs if run), [])
    phrase = " ".join(words[:MAX_PHRASE_WORDS]).strip(" &")
    return phrase or None


def name_query(phrase: str) -> str:
    """The one topic-only search that looks a company phrase up (no ``site:``, no provider)."""
    return f'"{phrase}" stock ticker symbol'


class NameCandidate(BaseModel):
    """A ticker the search results named, with the name written next to it (as found)."""

    symbol: str
    name: str = ""
    exchange: str | None = None
    domains: list[str] = Field(default_factory=list)
    mentions: int = 0

    def as_candidate(self) -> InstrumentCandidate:
        return InstrumentCandidate(
            symbol=self.symbol, exchange=self.exchange, name=self.name, score=len(self.domains)
        )


def _clean_name(text: str) -> str:
    words = text.split()
    while words and words[0].lower().strip(",.") in _PHRASE_STOPWORDS:
        words.pop(0)
    return " ".join(words).strip(" ,-|:")[:80]


def ticker_candidates(results: list[Any]) -> list[NameCandidate]:
    """Tickers named in search result titles and snippets ("Apple Inc. (AAPL)", "NASDAQ:
    AAPL", "AAPL stock"), with the most frequent name written next to each, ranked by the
    number of distinct websites naming them, then by mentions, then by first appearance."""
    found: dict[str, NameCandidate] = {}
    names: dict[str, dict[str, int]] = {}
    exchanges: dict[str, dict[str, int]] = {}
    order: list[str] = []
    for result in results:
        domain = domain_of(getattr(result, "url", "") or "")
        text = f"{getattr(result, 'title', '')} \n {getattr(result, 'snippet', '')}"
        mentions: list[tuple[str, str, str | None]] = []
        for match in _NAME_BEFORE.finditer(text):
            mentions.append((match.group(3), _clean_name(match.group(1)), match.group(2)))
        for match in _EXCHANGE_PREFIX.finditer(text):
            mentions.append((match.group(2), "", match.group(1)))
        for match in _SYMBOL_STOCK.finditer(text):
            mentions.append((match.group(1), "", None))
        for symbol, name, exchange in mentions:
            key = symbol_key(symbol)
            if key in _NOT_SYMBOLS or symbol in _TICKER_STOPLIST:
                continue
            candidate = found.get(key)
            if candidate is None:
                candidate = found[key] = NameCandidate(symbol=symbol)
                order.append(key)
            candidate.mentions += 1
            if domain and domain not in candidate.domains:
                candidate.domains.append(domain)
            if name and name.upper() != symbol:
                names.setdefault(key, {})[name] = names.get(key, {}).get(name, 0) + 1
            if exchange:
                upper = exchange.upper()
                exchanges.setdefault(key, {})[upper] = exchanges.get(key, {}).get(upper, 0) + 1
    for key, candidate in found.items():
        if names.get(key):
            counted = names[key]
            candidate.name = max(counted, key=lambda n: (counted[n], -list(counted).index(n)))
        if exchanges.get(key):
            counted = exchanges[key]
            candidate.exchange = max(counted, key=lambda e: (counted[e], -list(counted).index(e)))
    return sorted(
        found.values(),
        key=lambda c: (-len(c.domains), -c.mentions, order.index(symbol_key(c.symbol))),
    )
