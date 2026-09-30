"""Instrument identification from the ticker in the request (no directory, no network)."""

from __future__ import annotations

import pytest

from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.identity import TICKER_REQUIRED_MESSAGE, InstrumentResolver
from bayanalytics.schemas.common import ErrorCode
from bayanalytics.schemas.requests import InstrumentRef


@pytest.fixture
def resolver() -> InstrumentResolver:
    return InstrumentResolver()


@pytest.mark.parametrize(
    ("query", "symbol", "method"),
    [
        ("Assess $AAPL.", "AAPL", "cashtag"),
        ("what about $aapl's margins?", "AAPL", "cashtag"),
        ("Is $BRK.B cheap?", "BRK.B", "cashtag"),
        ("$AAPL vs SPY over a year", "AAPL", "cashtag"),  # a cashtag wins over bare tokens
        ("What might happen to NVDA next quarter?", "NVDA", "ticker_token"),
        ("How are AAPL's margins holding up?", "AAPL", "ticker_token"),
        ("Is BRK.B a buy for the long term?", "BRK.B", "ticker_token"),
        ("How did (MSFT) do vs the S&P 500 in Q3?", "MSFT", "ticker_token"),
        ("Should I trust the CEO of F on EPS and P/E?", "F", "ticker_token"),
    ],
)
def test_resolves_the_ticker_in_the_question(
    resolver: InstrumentResolver, query: str, symbol: str, method: str
) -> None:
    identity = resolver.resolve(query)
    assert identity.symbol == symbol and identity.resolution_method == method
    assert identity.name == "" and identity.cik is None and identity.sector is None


def test_explicit_instrument_ref_wins(resolver: InstrumentResolver) -> None:
    identity = resolver.resolve(
        "Assess $MSFT and Apple", InstrumentRef(symbol="aapl", exchange="nasdaq")
    )
    assert identity.symbol == "AAPL" and identity.exchange == "NASDAQ"
    assert identity.resolution_method == "instrument_ref" and identity.confidence == 1.0
    with pytest.raises(AnalysisError) as info:
        resolver.resolve("x", InstrumentRef(symbol="apple inc"))
    assert info.value.details["reason"] == "invalid_symbol"


@pytest.mark.parametrize(
    "query", ["Compare AAPL and MSFT.", "Is $AAPL or $MSFT better?", "AAPL/MSFT: AAPL vs GOOG"]
)
def test_several_tickers_are_ambiguous(resolver: InstrumentResolver, query: str) -> None:
    with pytest.raises(AnalysisError) as info:
        resolver.resolve(query)
    error = info.value
    assert error.code is ErrorCode.AMBIGUOUS_INSTRUMENT
    assert error.details["reason"] == "multiple_tickers"
    assert len(error.details["candidates"]) >= 2
    assert all(c["name"] == "" for c in error.details["candidates"])


def test_the_same_ticker_twice_is_not_ambiguous(resolver: InstrumentResolver) -> None:
    assert resolver.resolve("BRK.B or BRK-B? $BRK.B").symbol == "BRK.B"
    assert resolver.resolve("AAPL, AAPL and AAPL again").symbol == "AAPL"


@pytest.mark.parametrize(
    "query",
    ["Assess Apple.", "How is the CEO doing on EPS and FCF in the U.S.?", "what about S&P?", ""],
)
def test_no_ticker_asks_for_one(resolver: InstrumentResolver, query: str) -> None:
    with pytest.raises(AnalysisError) as info:
        resolver.resolve(query)
    error = info.value
    assert error.code is ErrorCode.AMBIGUOUS_INSTRUMENT
    assert error.details == {"reason": "ticker_required", "candidates": []}
    assert error.message == TICKER_REQUIRED_MESSAGE and "$AAPL" in error.message
