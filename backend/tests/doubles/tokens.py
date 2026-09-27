"""The doubles' deterministic tokenizer: one token per word or punctuation mark.

Every token figure a double reports is *measured* with this function over the exact text it
handled (the same rule ``stub_laya.mjs`` applies in the worker), never a characters-per-token
estimate. Tests that need a ``TokenCounter`` build their own with the same regex.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(str(text)))


def count_many(texts: Iterable[str]) -> list[int]:
    return [count_tokens(t) for t in texts]
