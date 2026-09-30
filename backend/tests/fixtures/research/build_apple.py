"""Regenerate the synthetic "apple" research fixture set deterministically (tests only).

Usage (from ``backend/``)::

    .venv/bin/python tests/fixtures/research/build_apple.py [tests/fixtures/research/apple]

``tests/test_research_runner.py`` loads this file with ``importlib`` and checks that the
generated files are byte-identical to the committed fixture.

Everything written here is FIXTURE DATA: search hits and web pages about a fixture company
(every number and article is invented; HTML pages carry ``<meta name="bay-fixture">`` and the
JSON files ``"fixture": true``). Research is web search only, so the fixture is exactly what a
search engine would return: result lists (searches.json) and the pages behind them
(pages.json -> news/*.html, hand-written).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# --------------------------------------------------------------------------------------
# pages.json / searches.json
# --------------------------------------------------------------------------------------

NEWS = [
    # url, body file, search publishedDate (None -> only the page carries the date), snippet
    (
        "https://www.cnbc.com/2026/09/18/fixture-apple-quarter-preview.html",
        "news/cnbc_quarter_preview.html",
        "2026-09-18T12:30:00Z",
        "Fixture preview of the September quarter: what the synthetic numbers imply.",
    ),
    (
        "https://www.cnbc.com/2026/09/18/fixture-apple-quarter-preview.html?utm_source=newsletter&utm_medium=email",
        "news/cnbc_quarter_preview.html",
        "2026-09-18T12:30:00Z",
        "Same fixture preview reached through a tracking link (canonical-URL duplicate).",
    ),
    (
        "https://www.marketwatch.com/story/fixture-apple-outlook-2026-09-12",
        "news/marketwatch_outlook.html",
        None,
        "Fixture outlook column: guidance framing for the next cycle.",
    ),
    (
        "https://www.businesswire.com/news/home/20260730001234/en/Fixture-Apple-Reports-Third-Quarter-Results",
        "news/businesswire_q3_release.html",
        "2026-07-30T20:05:00Z",
        "Fixture earnings release for the June quarter (synthetic figures).",
    ),
    (
        "https://www.fool.com/investing/2026/09/20/fixture-apple-what-to-watch/",
        "news/fool_what_to_watch.html",
        "2026-09-20",
        "Fixture commentary: three things to watch in the synthetic quarter.",
    ),
    (
        "https://www.reuters.com/technology/fixture-apple-october-event-2026-10-02/",
        "news/reuters_after_as_of.html",
        None,
        "Fixture article dated after the evaluation as_of; must be rejected by the leakage guard.",
    ),
    (
        "https://investor.example-fixture.com/news/short-notice",
        "news/ir_thin_notice.html",
        "2026-09-15",
        "Fixture investor-relations notice that is too short to be evidence.",
    ),
    (
        "https://www.prnewswire.com/news-releases/fixture-apple-reports-third-quarter-results-301234567.html",
        "news/prnewswire_q3_syndicated.html",
        "2026-07-30T20:07:00Z",
        "Syndicated copy of the fixture earnings release (content-hash duplicate).",
    ),
]

# Older coverage from two further websites, served only by the catch-all search ("*"): an
# analysis frozen before the summer still finds dated pages from more than one site.
OLDER_NEWS = [
    (
        "https://www.barrons.com/articles/fixture-apple-march-quarter-review-2026-05-04",
        "news/barrons_march_quarter_review.html",
        "2026-05-04T13:00:00Z",
        "Fixture review of the March quarter (synthetic figures).",
    ),
    (
        "https://www.zacks.com/stock/news/fixture-apple-june-quarter-preview-2026-06-15",
        "news/zacks_june_quarter_preview.html",
        "2026-06-15T10:00:00Z",
        "Fixture preview of the June quarter (synthetic figures).",
    ),
]


def build_pages() -> dict:
    pages: dict[str, dict] = {
        "https://www.wsj.com/fixture/paywalled-apple-story": {
            "status": 403,
            "content_type": "text/html",
            "paywalled": True,
        },
    }
    seen: set[str] = set()
    for url, body_file, _, _ in NEWS + OLDER_NEWS:
        if body_file in seen and "utm_" in url:
            continue
        seen.add(body_file)
        pages[url] = {
            "status": 200,
            "content_type": "text/html; charset=utf-8",
            "body_file": body_file,
        }
    return {"fixture": True, "pages": pages}


def _search_hits(news: list[tuple[str, str, str | None, str]]) -> list[dict]:
    results = []
    for url, _, published, snippet in news:
        item = {
            "url": url,
            "title": _title_for(url),
            "content": snippet,
            "engine": "fixture",
            "score": 1.0,
        }
        if published:
            item["publishedDate"] = published
        results.append(item)
    return results


def build_searches() -> dict:
    results = _search_hits(NEWS)
    return {
        "fixture": True,
        "searches": {
            '"AAPL" earnings OR guidance OR outlook': results,
            '"AAPL" stock 2025 results': results[2:5],
            "*": results + _search_hits(OLDER_NEWS),
        },
    }


def _title_for(url: str) -> str:
    if "cnbc" in url:
        return "[Fixture] Apple quarter preview: what the synthetic numbers imply"
    if "marketwatch" in url:
        return "[Fixture] Opinion: the outlook for the next cycle"
    if "businesswire" in url or "prnewswire" in url:
        return "[Fixture] Apple Reports Third Quarter Results"
    if "fool" in url:
        return "[Fixture] 3 things to watch this quarter"
    if "reuters" in url:
        return "[Fixture] October event recap"
    if "barrons" in url:
        return "[Fixture] The March quarter in review"
    if "zacks" in url:
        return "[Fixture] What to expect from the June quarter"
    return "[Fixture] Investor notice"


def build(out_dir: Path) -> list[Path]:
    written: list[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (("pages.json", build_pages()), ("searches.json", build_searches())):
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    default = Path(__file__).resolve().parent / "apple"
    out_dir = Path(args[0]) if args else default
    for path in build(out_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
