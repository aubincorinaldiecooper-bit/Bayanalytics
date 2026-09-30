#!/bin/sh
# SearXNG needs a secret key. When SEARXNG_SECRET is not set, generate a random one inside the
# container at every start, so no secret is stored in Railway, the repository or any log. The
# key only signs short-lived values (the limiter and image proxy are off), so a new key per
# restart is harmless.
set -eu
if [ -z "${SEARXNG_SECRET:-}" ]; then
  SEARXNG_SECRET="$(od -An -N32 -tx1 /dev/urandom 2>/dev/null | tr -d ' \n' || true)"
  if [ -z "$SEARXNG_SECRET" ]; then
    SEARXNG_SECRET="$(python3 -c 'import secrets; print(secrets.token_hex(32))' 2>/dev/null || true)"
  fi
  if [ -z "$SEARXNG_SECRET" ]; then
    echo "bay-searxng: could not generate SEARXNG_SECRET" >&2
    exit 1
  fi
  export SEARXNG_SECRET
fi
exec /usr/local/searxng/entrypoint.sh "$@"
