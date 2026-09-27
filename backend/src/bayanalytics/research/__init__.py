"""Research layer (AGENT.md section 21): bounded intents -> deterministic queries -> provider.

Modules:
    provider         fixed protocol: SearchResult, PageResult, EvidenceRecord, ResearchProvider
    searxng          SearxngSearch (adapted from GNSIS under MIT)
    fetch            Fetcher: user agent, per-host throttle, SEC 10 req/s, robots, 2 MiB cap, cache
    extract          extract_html / extract_json / extract_csv / extract_page -> EvidenceRecord
    sources          classify_source, canonical_url, source_record_from_evidence
    dedup            Deduplicator (content hash, canonical URL, title prefix)
    edgar            EdgarClient (company_tickers, submissions, company_facts)
    prices           StooqPrices, select_benchmarks, to_stooq_symbol
    intents          ResearchIntent, PlannedQuery, build_queries, seed_plan, gap_to_intent
    runner           ResearchRunner.execute(PlannedQuery) -> RoundResult
    http_provider    HttpResearchProvider, build_research_stack(settings)

Test doubles (fixture-backed provider, synthetic Apple data) live under ``tests/doubles`` and
``tests/fixtures``; nothing in this package serves canned data.
"""
