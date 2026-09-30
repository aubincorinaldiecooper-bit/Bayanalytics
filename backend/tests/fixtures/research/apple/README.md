# Research fixture: "apple" (SYNTHETIC)

Every file in this directory is fixture data: what a web search would return for the ticker
AAPL (search result lists) and the web pages behind those results. Every number and article
is invented. JSON files carry `"fixture": true`, HTML pages carry
`<meta name="bay-fixture" content="true">`.

Regenerate the generated files (pages.json, searches.json) with:

    cd backend && .venv/bin/python tests/fixtures/research/build_apple.py

(the generator lives next to this directory, under `tests/`, never in the product package;
`tests/test_research_runner.py` checks the committed files match its output byte for byte).

Layout: `pages.json` (URL -> body file), `searches.json` (query -> SearXNG-shaped results,
`"*"` is the catch-all), `news/` (synthetic articles).
