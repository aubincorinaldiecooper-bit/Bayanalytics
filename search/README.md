# Web search service (SearXNG)

This is BayAnalytics' only data source. It was migrated from CLIPIT (`Dockerfile.searxng` and
`searxng/settings.yml`) with the same settings: JSON results on, rate limiter off, Brave Videos
removed.

On Railway it runs as the `searxng` service in the `bayanalytics` project:

- builder: Dockerfile, root directory `/search`
- health check: `/healthz`
- no public domain; the backend reaches it over the private network with
  `BAY_RESEARCH_SEARCH_URL=http://${{searxng.RAILWAY_PRIVATE_DOMAIN}}:8080`
- no variables: `start.sh` generates a random `SEARXNG_SECRET` inside the container at every start,
  so no secret is stored in Railway or the repository (set `SEARXNG_SECRET` only to pin one)

To run it locally:

```sh
docker build -t bay-searxng search
docker run --rm -p 8888:8080 bay-searxng
# then: BAY_RESEARCH_SEARCH_URL=http://127.0.0.1:8888
```

Do not add engines that are data-provider APIs, and do not point the backend at any other data
source. See `CLAUDE.md` at the repository root.
