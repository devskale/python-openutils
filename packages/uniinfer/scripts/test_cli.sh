#!/usr/bin/env bash
# Smoke-test the uniinfer CLI: version, list-providers, a real completion, and
# --no-think (the reasoning_effort="none" path). Exits non-zero on any failure.
#
# Env:
#   PROVIDER  provider id (default: mistral — needs a credgoo key)
#   MODEL     model id (default: mistral-small-latest)
#   CMD       the uniinfer invocation (default: uv run uniinfer)
set -uo pipefail

PROVIDER="${PROVIDER:-mistral}"
MODEL="${MODEL:-mistral-small-latest}"
CMD="${CMD:-uv run uniinfer}"

# Run from the package root (where pyproject.toml / .venv live).
cd "$(dirname "$0")/.."

pass=0; fail=0
ok()  { echo "  ✓ $1"; pass=$((pass+1)); }
bad() { echo "  ✗ $1"; fail=$((fail+1)); }

echo "CLI: $CMD   provider: $PROVIDER@$MODEL"
echo ""

# 1. version
v=$($CMD --version 2>&1 | tail -1)
echo "$v" | grep -q "uniinfer" && ok "version -> $v" || bad "version"

# 2. list-providers (no creds needed; exercises ProviderFactory)
$CMD --list-providers >/dev/null 2>&1 && ok "list-providers" || bad "list-providers"

# 3. real completion — output must contain the requested token
out=$($CMD -p "$PROVIDER" -m "$MODEL" -q "Reply with exactly: OK" -t 15 2>&1)
echo "$out" | grep -qi "OK" && ok "completion ($PROVIDER@$MODEL)" || bad "completion (no 'OK' in output)"

# 4. --no-think — the reasoning_effort="none" code path must not error
$CMD -p "$PROVIDER" -m "$MODEL" -q "hi" --no-think -t 15 >/dev/null 2>&1 \
  && ok "--no-think (reasoning_effort=none)" || bad "--no-think"

echo ""
echo "Result: $pass passed, $fail failed"
exit $fail
