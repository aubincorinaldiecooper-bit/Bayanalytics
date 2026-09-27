"""Test doubles: the only place a stand-in for a product component may live.

* ``RuleLaya`` answers Laya questions from a handful of state keys (no model, no subprocess);
* ``ScriptedSpark`` streams a scripted, sectioned text (no llama-server) and answers the
  structured pass-1 request with a scripted interpretation looked up by question text (it
  does not understand language);
* ``FixedTranscriber`` returns a fixed transcript (no whisper.cpp);
* ``FixtureResearchProvider`` / ``FixtureFetcher`` serve the synthetic fixture directories.

Every number a double reports is measured with the double's own deterministic tokenizer
(``doubles.tokens``) or its own clock, or it is ``None``. ``fixture_research_stack`` wires the
fixture provider through the real ``EdgarClient`` and ``StooqPrices``.
"""

from __future__ import annotations

from pathlib import Path

from bayanalytics.config import Settings
from bayanalytics.research.edgar import EdgarClient
from bayanalytics.research.prices import StooqPrices

from .laya import LayaCall, RuleLaya
from .research import FixtureFetcher, FixtureResearchProvider
from .spark import ScriptedSpark, scripted_text
from .tokens import TOKEN_RE, count_many, count_tokens
from .whisper import FixedTranscriber

__all__ = [
    "TOKEN_RE",
    "FixedTranscriber",
    "FixtureFetcher",
    "FixtureResearchProvider",
    "LayaCall",
    "RuleLaya",
    "ScriptedSpark",
    "count_many",
    "count_tokens",
    "fixture_research_stack",
    "scripted_text",
]


def fixture_research_stack(
    settings: Settings, fixture_dir: Path | str
) -> tuple[FixtureResearchProvider, EdgarClient, StooqPrices]:
    """The research stack over one fixture directory: the fixture provider plus the real EDGAR
    and Stooq clients reading through the same ``FixtureFetcher``."""
    fetcher = FixtureFetcher(fixture_dir)
    provider = FixtureResearchProvider(fixture_dir, fetcher=fetcher)
    return provider, EdgarClient(fetcher, settings), StooqPrices(fetcher)
