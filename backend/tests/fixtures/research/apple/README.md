# Research fixture: "apple" (SYNTHETIC)

Every file in this directory is fixture data. The identity (Apple Inc., CIK 320193, AAPL) is
real because identity resolution is real; every number, filing accession, price bar and
article is invented. JSON files carry `"fixture": true`, HTML pages carry
`<meta name="bay-fixture" content="true">`.

Regenerate the generated files (CSVs, EDGAR JSON, pages.json, searches.json) with:

    cd backend && .venv/bin/python -m bayanalytics.research.fixtures_build

Layout: `pages.json` (URL -> body file), `searches.json` (query -> SearXNG-shaped results),
`edgar/` (submissions, companyfacts, company_tickers, tiny primary documents), `prices/`
(Stooq-shaped CSVs, 520 business days ending 2026-09-25), `news/` (synthetic articles).
