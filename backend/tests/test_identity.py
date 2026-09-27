from __future__ import annotations

import pytest

from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.identity import InstrumentResolver, normalize_name
from bayanalytics.schemas.requests import InstrumentRef

ROWS = [
    {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc.", "exchange": "NASDAQ"},
    {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP", "exchange": "NASDAQ"},
    {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP", "exchange": "NASDAQ"},
    {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc.", "exchange": "NASDAQ"},
    {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc.", "exchange": "NASDAQ"},
    {"cik_str": 1594805, "ticker": "SHOP", "title": "SHOPIFY INC.", "exchange": "NASDAQ"},
    {
        "cik_str": 6201,
        "ticker": "AAL",
        "title": "American Airlines Group Inc.",
        "exchange": "NASDAQ",
    },
    {"cik_str": 4962, "ticker": "AXP", "title": "AMERICAN EXPRESS CO", "exchange": "NYSE"},
    {"cik_str": 1067983, "ticker": "BRK-B", "title": "BERKSHIRE HATHAWAY INC", "exchange": "NYSE"},
    {"cik_str": 1326801, "ticker": "META", "title": "Meta Platforms, Inc.", "exchange": "NASDAQ"},
]


@pytest.fixture
def resolver() -> InstrumentResolver:
    return InstrumentResolver(ROWS)


def test_normalize_name() -> None:
    assert normalize_name("Apple Inc.") == "apple"
    assert normalize_name("MICROSOFT CORP") == "microsoft"
    assert normalize_name("Meta Platforms, Inc.") == "meta platforms"
    assert normalize_name("BERKSHIRE HATHAWAY INC") == "berkshire hathaway"


@pytest.mark.parametrize(
    ("query", "symbol", "method"),
    [
        ("Assess Apple.", "AAPL", "name_match"),
        ("What changed at Apple this quarter?", "AAPL", "name_match"),
        ("What might happen to Shopify over the next 12 months?", "SHOP", "name_match"),
        ("Is Microsoft expensive relative to its history?", "MSFT", "name_match"),
        ("Assess NVDA", "NVDA", "ticker_token"),
        ("How is $msft doing?", "MSFT", "cashtag"),
        ("Assess Google.", "GOOG", "name_match"),
        ("Is Facebook growing?", "META", "name_match"),
        ("Assess Berkshire Hathaway", "BRK-B", "name_match"),
        ("What is the outlook for American Express?", "AXP", "name_match"),
    ],
)
def test_resolve_from_query(
    resolver: InstrumentResolver, query: str, symbol: str, method: str
) -> None:
    identity = resolver.resolve(query)
    assert identity.symbol == symbol
    assert identity.resolution_method == method
    assert identity.cik and len(identity.cik) == 10


def test_share_classes_are_aliases_not_ambiguity(resolver: InstrumentResolver) -> None:
    identity = resolver.resolve("Assess Alphabet.")
    assert identity.symbol == "GOOG" and identity.aliases == ["GOOGL"]


def test_explicit_instrument_ref(resolver: InstrumentResolver) -> None:
    identity = resolver.resolve("Assess it.", InstrumentRef(symbol="brk.b", exchange="nyse"))
    assert identity.symbol == "BRK-B" and identity.exchange == "NYSE"
    assert identity.resolution_method == "instrument_ref" and identity.confidence == 1.0
    with pytest.raises(AnalysisError) as exc:
        resolver.resolve("Assess it.", InstrumentRef(symbol="ZZZZ"))
    assert exc.value.code == "AMBIGUOUS_INSTRUMENT"
    assert exc.value.details["reason"] == "unknown_symbol"


def test_two_companies_is_ambiguous(resolver: InstrumentResolver) -> None:
    with pytest.raises(AnalysisError) as exc:
        resolver.resolve("Compare Apple and Microsoft.")
    assert exc.value.code == "AMBIGUOUS_INSTRUMENT"
    symbols = {c["symbol"] for c in exc.value.details["candidates"]}
    assert symbols == {"AAPL", "MSFT"}
    assert exc.value.message == "Which company did you mean?"


def test_no_match_gives_fuzzy_candidates(resolver: InstrumentResolver) -> None:
    with pytest.raises(AnalysisError) as exc:
        resolver.resolve("Assess Aple.")
    assert exc.value.details["reason"] == "no_match"
    assert [c["symbol"] for c in exc.value.details["candidates"]] == ["AAPL"]
    with pytest.raises(AnalysisError) as exc2:
        resolver.resolve("Tell me about the weather.")
    assert exc2.value.details["candidates"] == []
    assert exc2.value.http_status == 422


def test_stoplist_prevents_false_ticker(resolver: InstrumentResolver) -> None:
    # "AI" and "CEO" are upper-case tokens but never tickers; the name resolves instead.
    identity = resolver.resolve("Is the AI CEO of American Airlines confident?")
    assert identity.symbol == "AAL"
