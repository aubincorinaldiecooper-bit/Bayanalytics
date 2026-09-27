from __future__ import annotations

import pytest

from bayanalytics.errors import AnalysisError
from bayanalytics.instruments.identity import (
    InstrumentResolver,
    normalize_name,
    possessive_base,
    query_words,
)
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
        ("Assess Google.", "GOOGL", "name_match"),
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
    assert identity.symbol == "GOOGL" and identity.aliases == ["GOOG"]


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


def test_stoplist_blocks_real_tickers_that_are_common_words() -> None:
    rows = [
        *ROWS,
        {"cik_str": 1373715, "ticker": "NOW", "title": "ServiceNow, Inc.", "exchange": "NYSE"},
        {"cik_str": 1577526, "ticker": "AI", "title": "C3.ai, Inc.", "exchange": "NYSE"},
        {"cik_str": 1385157, "ticker": "CEO", "title": "CNOOC Ltd", "exchange": "NYSE"},
    ]
    resolver = InstrumentResolver(rows)
    # NOW, AI and CEO are listed tickers, but as bare upper-case words they must not resolve.
    identity = resolver.resolve("Is the AI CEO of American Airlines confident NOW?")
    assert identity.symbol == "AAL" and identity.resolution_method == "name_match"
    assert resolver.resolve("How is $NOW doing?").symbol == "NOW"  # a cashtag still works
    assert resolver.resolve("Assess $ai").symbol == "AI"


def test_resolver_has_no_directory_completeness_flag() -> None:
    # The live SEC directory is the only ticker source: there is no partial-seed mode left.
    resolver = InstrumentResolver(ROWS)
    assert not hasattr(resolver, "directory_complete")
    assert len(resolver.rows) == len(ROWS)


# ------------------------------------------------------------------ possessives

# Issuers whose own names end in "'s", plus a listed ticker that is a stoplisted word.
POSSESSIVE_ROWS = [
    *ROWS,
    {"cik_str": 63908, "ticker": "MCD", "title": "McDonald's Corp", "exchange": "NYSE"},
    {"cik_str": 1059556, "ticker": "MCO", "title": "MOODY'S CORP /DE/", "exchange": "NYSE"},
    {"cik_str": 1373715, "ticker": "NOW", "title": "ServiceNow, Inc.", "exchange": "NYSE"},
]
# Contrived issuers whose names equal a contraction's base ("what", "it"): stripping "'s" from
# "What's" / "It's" would match them and make every such question ambiguous.
CONTRACTION_ROWS = [
    *ROWS,
    {"cik_str": 9000001, "ticker": "WHTH", "title": "What Holdings Inc.", "exchange": "NYSE"},
    {"cik_str": 9000002, "ticker": "ITHD", "title": "IT HOLDINGS INC", "exchange": "NYSE"},
]


@pytest.fixture
def possessive_resolver() -> InstrumentResolver:
    return InstrumentResolver(POSSESSIVE_ROWS)


@pytest.mark.parametrize(
    ("query", "symbol", "method"),
    [
        # straight and curly possessives on names
        ("Is Apple's dividend safe?", "AAPL", "name_match"),
        ("What is Apple's P/E?", "AAPL", "name_match"),
        ("Assess Apple's valuation", "AAPL", "name_match"),
        ("Is Apple\u2019s dividend safe?", "AAPL", "name_match"),
        ("Evaluate Microsoft's margins", "MSFT", "name_match"),
        ("Evaluate Microsoft\u2019s margins", "MSFT", "name_match"),
        ("How strong is Berkshire Hathaway's cash position?", "BRK-B", "name_match"),
        ("What is American Express's outlook?", "AXP", "name_match"),
        ("Is Google's growth durable?", "GOOGL", "name_match"),  # alias, then possessive
        # plural possessive: the trailing apostrophe goes, the name's own "s" stays
        ("Are Meta Platforms' margins improving?", "META", "name_match"),
        ("Are Meta Platforms\u2019 margins improving?", "META", "name_match"),
        # straight and curly possessives on tickers and cashtags
        ("What is AAPL's P/E?", "AAPL", "ticker_token"),
        ("What is AAPL\u2019s P/E?", "AAPL", "ticker_token"),
        ("How is $msft's growth?", "MSFT", "cashtag"),
        ("How is $MSFT\u2019s growth?", "MSFT", "cashtag"),
        # names that contain "'s" still match as written, straight or curly
        ("Are McDonald's margins improving?", "MCD", "name_match"),
        ("Are McDonald\u2019s margins improving?", "MCD", "name_match"),
        ("Is Moody's growing?", "MCO", "name_match"),
    ],
)
def test_possessives_resolve_like_the_bare_name(
    possessive_resolver: InstrumentResolver, query: str, symbol: str, method: str
) -> None:
    identity = possessive_resolver.resolve(query)
    assert identity.symbol == symbol
    assert identity.resolution_method == method


@pytest.mark.parametrize(
    "query",
    ["What's Apple's P/E?", "What\u2019s Apple\u2019s P/E?", "It's time to assess Apple's margins"],
)
def test_contractions_are_not_stripped_into_false_matches(query: str) -> None:
    identity = InstrumentResolver(CONTRACTION_ROWS).resolve(query)
    assert identity.symbol == "AAPL" and identity.resolution_method == "name_match"


def test_possessives_keep_the_stoplist_generic_prefixes_and_ambiguity(
    possessive_resolver: InstrumentResolver,
) -> None:
    # a stoplisted ticker does not become one because of its possessive; a cashtag still does
    assert possessive_resolver.resolve("Is NOW's valuation stretched at Shopify?").symbol == "SHOP"
    assert possessive_resolver.resolve("Is $NOW's valuation stretched?").symbol == "NOW"
    # a generic leading word stays generic: no silent pick between AAL and AXP
    with pytest.raises(AnalysisError) as generic:
        possessive_resolver.resolve("Is American's growth durable?")
    assert generic.value.details["reason"] == "no_match"
    assert generic.value.details["candidates"] == []
    # two companies named possessively are still ambiguous
    with pytest.raises(AnalysisError) as two:
        possessive_resolver.resolve("Compare Apple's and Microsoft's margins.")
    assert two.value.details["reason"] == "multiple_companies"
    assert {c["symbol"] for c in two.value.details["candidates"]} == {"AAPL", "MSFT"}
    # a misspelt possessive still offers the closest company, not a resolution
    with pytest.raises(AnalysisError) as typo:
        possessive_resolver.resolve("Assess Aple's valuation")
    assert typo.value.details["reason"] == "no_match"
    assert [c["symbol"] for c in typo.value.details["candidates"]] == ["AAPL"]


def test_query_words_and_possessive_base() -> None:
    assert query_words("Is Apple\u2019s P/E high? Meta Platforms' too.") == [
        "is",
        "apple's",
        "p",
        "e",
        "high",
        "meta",
        "platforms",
        "too",
    ]
    assert possessive_base("apple's") == "apple"
    assert possessive_base("hathaway's") == "hathaway"
    for kept in ("what's", "it's", "let's", "a's", "apple", "platforms"):
        assert possessive_base(kept) == kept
