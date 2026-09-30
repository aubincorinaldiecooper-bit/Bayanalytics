"""Test doubles: the only place a stand-in for a product component may live.

* ``RuleLaya`` answers Laya questions from a handful of state keys (no model, no subprocess);
* ``ScriptedSpark`` streams a scripted, sectioned text (no llama-server) and answers the
  structured pass-1 request with a scripted interpretation looked up by question text (it
  does not understand language);
* ``FixedTranscriber`` returns a fixed transcript (no whisper.cpp);
* ``FixtureResearchProvider`` / ``FixtureFetcher`` serve the synthetic fixture directories.

Every number a double reports is measured with the double's own deterministic tokenizer
(``doubles.tokens``) or its own clock, or it is ``None``. ``fixture_research_stack`` builds the
fixture provider (search hits and web pages only) the runtime takes as its research provider.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bayanalytics.config import Settings

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
    settings: Settings, fixture_dir: Path | str, **provider_options: Any
) -> FixtureResearchProvider:
    """The research provider over one fixture directory (search hits and web pages only).
    ``provider_options`` go to ``FixtureResearchProvider`` (``search_configured``,
    ``search_error``)."""
    return FixtureResearchProvider(fixture_dir, **provider_options)
