"""Instrument identification (AGENT.md sections 23, 24 "Ambiguous instrument").

Research is web search only, so there is no company directory to look a name up in and nothing
here touches the network. The instrument is the ticker the analyst gave, in this order:

1. an explicit ``InstrumentRef`` (its symbol, with the exchange as given);
2. cashtags in the question (``$AAPL``, ``$BRK.B``);
3. ticker-like upper-case tokens (1-5 letters, optional one-letter class suffix such as
   ``BRK.B``) that are not common upper-case words (``_TICKER_STOPLIST``).

Several distinct symbols raise ``AMBIGUOUS_INSTRUMENT`` (reason ``multiple_tickers``) with the
symbols as candidates; none raises it with reason ``ticker_required`` and asks for the ticker.
A company name is never guessed: the identity's ``name`` stays empty and research queries use
the symbol.
"""

from __future__ import annotations

import re

from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.base import InstrumentCandidate, InstrumentIdentity
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.requests import InstrumentRef

TICKER_REQUIRED_MESSAGE = "Include the company's stock ticker in your question, for example $AAPL."
MULTIPLE_TICKERS_MESSAGE = "Which company did you mean?"

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
