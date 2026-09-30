"""Research layer (AGENT.md section 21): bounded intents -> deterministic web searches -> pages.

Data comes from web search only (see the repository's CLAUDE.md): the backend contacts the
configured search engine and the pages its searches returned, never a source it picks itself.

Modules:
    provider         fixed protocol: SearchResult, PageResult, EvidenceRecord, ResearchProvider
    searxng          SearxngSearch (adapted from GNSIS under MIT)
    fetch            Fetcher: user agent, per-host throttle, robots, 2 MiB cap, SSRF policy, cache
    extract          extract_html / extract_json / extract_csv / extract_page -> EvidenceRecord
    sources          classify_source, canonical_url, source_record_from_evidence, domain_of
    dedup            Deduplicator (content hash, canonical URL, title prefix)
    intents          ResearchIntent, PlannedQuery, build_queries, seed_plan, gap_to_intent
    runner           ResearchRunner.execute(PlannedQuery) -> RoundResult
    http_provider    HttpResearchProvider (refuses URLs its search did not return),
                     build_research_stack(settings)

Test doubles (fixture-backed provider, synthetic Apple data) live under ``tests/doubles`` and
``tests/fixtures``; nothing in this package serves canned data.
"""
