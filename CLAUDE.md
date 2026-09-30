# BayAnalytics — rules for anyone changing this repository

## Data comes from web search only (owner's rule, not negotiable)

The owner decides where data comes from. The backend aggregates what a **web search** returns.
It never picks its own sources.

- The backend may contact only:
  1. the search engine the operator configured (`BAY_RESEARCH_SEARCH_URL`);
  2. the pages those searches returned, including their redirects and their `robots.txt`.
- Never add a direct connection to a data provider, API, feed or website. That means no SEC
  EDGAR API, no price or market-data API, no news API, no "known good" URLs, and no fallback
  source when search fails or returns nothing.
- Never steer the search toward a chosen site: no `site:` operators, no provider names in
  query templates.
- If data is missing, the analysis says so. It never fills the gap from somewhere else.
- A new source of any kind needs the owner's explicit approval **before** any code is written.
  Ask; don't assume that a spec line like "historical prices" or "filings" approves a provider.
- `HttpResearchProvider` refuses to open any URL its own search did not return, and
  `tests/test_source_policy.py` fails the build if product code gains a hard-coded data URL.
  Do not weaken either one.

Outbound traffic that is not data is limited to model downloads from Hugging Face at startup.
Third-party usage reporting is off (`procenv.FORCED_ENV`, `ORT_DISABLE_TELEMETRY=1` in the image).
Keep it off.

## How changes ship (owner's rule, not negotiable)

- Finish the whole change first, then run the checks once. When asked to remove something, remove
  all of it (code, wiring, config, labels, docs, tests) in one pass before checking.
- Once a change has passed its checks, **leave it alone**: no follow-up tidying, relabelling or
  "one more fix" before it ships, and no re-running checks on code that already passed. Ship
  exactly what passed.
- Leftovers noticed after checks pass are reported to the owner, not fixed on the spot. They go
  in a later change only if the owner asks.

## Other standing rules

- The browser never receives `BAY_API_KEY`. There is no temporary password-based auth.
- No fabricated market data, fake analysis progress or fake browser/agent activity. Every event
  is emitted when the backend actually does the step.
- Never expose raw chain-of-thought.
- Third-party text (page excerpts, search snippets) is sent to clients only when the source's terms
  allow redistribution.
