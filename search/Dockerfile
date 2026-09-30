# Web search for BayAnalytics research: SearXNG, migrated from CLIPIT (Dockerfile.searxng +
# searxng/settings.yml). Private to the Railway project; the backend reaches it at
# http://searxng.railway.internal:8080 through BAY_RESEARCH_SEARCH_URL.
FROM searxng/searxng:latest

COPY settings.yml /etc/searxng/settings.yml
COPY start.sh /usr/local/bin/bay-searxng-start
RUN chmod 0755 /usr/local/bin/bay-searxng-start

ENV SEARXNG_SETTINGS_PATH=/etc/searxng/settings.yml
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/bay-searxng-start"]
