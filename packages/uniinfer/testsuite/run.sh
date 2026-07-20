#!/usr/bin/env bash
# uniinfer LIVE testsuite runner.
#
# Tiers (cheap -> deep), all provider-agnostic (the point of uniinfer):
#   smoke   — is it alive?   (CLI + proxy: version, a completion, a stream, embeddings)
#   details — is it correct? (reasoning control, stream shape, error handling, embeddings)
#   perf    — is it fast?    (tok/s, TTFT, latency, context scaling)
#   matrix  — capability probe + matrix for a provider (needs PROVIDER or MODELS)
#   models  — direct LLM validation (alive/tool_use/thinking) for a provider
#
# The unit/mocked tests live in uniinfer/tests/ (pytest, no network). This
# testsuite/ is the LIVE tier: it talks to a real proxy + provider, so it needs
# a running proxy and (for non-open providers) a key.
#
# Usage:
#   ./testsuite/run.sh                 # all tiers (smoke+details+perf)
#   ./testsuite/run.sh smoke            # one tier
#   ./testsuite/run.sh matrix           # needs PROVIDER or MODELS
#   PROXY_URL=https://localhost:8123 PROXY_AUTH=<PROXY_AUTH> \
#     PROVIDER=tu MAX=2 ./testsuite/run.sh matrix
#
# Env (all optional — defaults target a local proxy on :8123):
#   PROXY_URL     proxy base URL          (default https://localhost:8123)
#   PROXY_AUTH    proxy bearer token      (open providers like ollama need none)
#   PROXY_KEY     bearer token alias      (read if PROXY_AUTH unset)
#   MODEL         proxy chat model        (default ollama@qwen3.5:0.8b)
#   EMBED_MODEL   proxy embedding model   (default ollama@nomic-embed-text-v2-moe)
#   PROVIDER      provider for matrix/models tiers
#   MODELS        provider@model,... for matrix/models tiers
#   MAX           max models for the matrix tier (default 3)
#   PROVIDER/CLI_MODEL (CLI smoke provider/model)
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

matrix() {
  echo "############ MATRIX  (capability matrix, any provider) ############"
  if [ -z "${PROVIDER:-}" ] && [ -z "${MODELS:-}" ]; then
    echo "  set PROVIDER (e.g. tu) or MODELS (provider@model,...) to use the matrix tier"
    fail=1; return
  fi
  if [ -n "${MODELS:-}" ]; then
    uv run python testsuite/test_via_proxy.py --models "$MODELS" --max "${MAX:-3}" || fail=1
  else
    uv run python testsuite/test_via_proxy.py --provider "$PROVIDER" --max "${MAX:-3}" || fail=1
  fi
}

models() {
  echo "############ MODELS  (direct validation, any provider) ############"
  if [ -z "${PROVIDER:-}" ] && [ -z "${MODELS:-}" ]; then
    echo "  set PROVIDER or MODELS to use the models tier"
    fail=1; return
  fi
  if [ -n "${MODELS:-}" ]; then
    uv run python testsuite/test_models.py --models "$MODELS" || fail=1
  else
    uv run python testsuite/test_models.py --provider "$PROVIDER" || fail=1
  fi
}

bench() {
  echo "############ BENCH  (OpenAI-compatible probe + tok/s) ############"
  if [ -z "${BASEURL:-}" ]; then
    echo "  set BASEURL (e.g. https://localhost:8123/v1) and MODEL to use the bench tier"
    fail=1; return
  fi
  uv run python testsuite/bench_openai.py \
    --base-url "$BASEURL" ${BEARER:+--bearer "$BEARER"} ${MODEL:+--model "$MODEL"} \
    ${REASONING:+--reasoning "$REASONING"} ${BENCH_ARGS:-} || fail=1
}

case "$TIER" in
  smoke)   smoke ;;
  details) details ;;
  perf)    perf ;;
  matrix)  matrix ;;
  models)  models ;;
  bench)   bench ;;
  all)     smoke; echo; details; echo; perf ;;
  *) echo "usage: $0 [smoke|details|perf|matrix|models|bench|all]"; exit 2 ;;
esac

echo ""
if [ "$fail" -eq 0 ]; then echo "TESTSUITE: PASS"; else echo "TESTSUITE: FAIL"; fi
exit $fail
