#!/usr/bin/env bash
# uniinfer LIVE testsuite runner.
#
# Tiers (cheap -> deep):
#   smoke   — is it alive?   (CLI + proxy: version, a completion, a stream, embeddings)
#   details — is it correct? (reasoning control, stream shape, error handling, embeddings)
#   perf    — is it fast?    (tok/s, TTFT, latency, context scaling)
#
# The unit/mocked tests live in uniinfer/tests/ (pytest, no network). This
# testsuite/ is the LIVE tier: it talks to a real proxy + provider, so it needs
# a running proxy and (for non-open providers) a key.
#
# Usage:
#   ./testsuite/run.sh              # all tiers
#   ./testsuite/run.sh smoke        # one tier
#   PROXY_URL=http://127.0.0.1:8013 PROXY_AUTH=test23@test34 ./testsuite/run.sh
#
# Env (all optional — defaults target a local proxy + amp ollama + mistral CLI):
#   PROXY_URL     proxy base URL          (default http://127.0.0.1:8013)
#   PROXY_AUTH    proxy bearer token      (open providers like ollama need none)
#   MODEL         proxy chat model        (default ollama@qwen3.5:0.8b)
#   EMBED_MODEL   proxy embedding model   (default ollama@nomic-embed-text-v2-moe)
#   PROVIDER      CLI provider            (default mistral)
#   CLI_MODEL     CLI model               (default mistral-small-latest)
set -uo pipefail
TIER="${1:-all}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
fail=0

smoke() {
  echo "############ SMOKE  (alive?) ############"
  echo "-- CLI --"
  PROVIDER="${PROVIDER:-mistral}" MODEL="${CLI_MODEL:-mistral-small-latest}" ./scripts/test_cli.sh || fail=1
  echo "-- proxy --"
  PROXY_URL="${PROXY_URL:-http://127.0.0.1:8013}" PROXY_AUTH="${PROXY_AUTH:-}" \
    CHAT_MODEL="${MODEL:-ollama@qwen3.5:0.8b}" EMBED_MODEL="${EMBED_MODEL:-ollama@nomic-embed-text-v2-moe}" \
    ./scripts/test_proxy.sh || fail=1
}

details() {
  echo "############ DETAILS  (correct?) ############"
  PROXY_URL="${PROXY_URL:-http://127.0.0.1:8013}" PROXY_AUTH="${PROXY_AUTH:-}" \
    MODEL="${MODEL:-ollama@qwen3.5:0.8b}" EMBED_MODEL="${EMBED_MODEL:-ollama@nomic-embed-text-v2-moe}" \
    uv run python testsuite/details.py || fail=1
}

perf() {
  echo "############ PERF  (fast?) ############"
  PROXY_URL="${PROXY_URL:-http://127.0.0.1:8013}" PROXY_AUTH="${PROXY_AUTH:-}" \
    MODEL="${MODEL:-ollama@qwen3.5:0.8b}" \
    uv run python testsuite/perf.py || fail=1
}

case "$TIER" in
  smoke)   smoke ;;
  details) details ;;
  perf)    perf ;;
  all)     smoke; echo; details; echo; perf ;;
  *) echo "usage: $0 [smoke|details|perf|all]"; exit 2 ;;
esac

echo ""
if [ "$fail" -eq 0 ]; then echo "TESTSUITE: PASS"; else echo "TESTSUITE: FAIL"; fi
exit $fail
