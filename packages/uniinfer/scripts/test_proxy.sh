#!/usr/bin/env bash
# Smoke-test a uniinfer proxy: version, non-stream chat, STREAM chat (content
# MUST flow), embeddings. Exits non-zero on any failure.
#
# Env:
#   PROXY_URL    base URL (default: the amd production proxy)
#   PROXY_AUTH   bearer token (required)
#   CHAT_MODEL   provider@model for chat (default: an amp ollama chat model)
#   EMBED_MODEL  provider@model for embeddings (default: an amp ollama embed model)
set -uo pipefail

PROXY_URL="${PROXY_URL:-https://amd1.mooo.com:8123}"
PROXY_AUTH="${PROXY_AUTH:?PROXY_AUTH (bearer token) required}"
CHAT_MODEL="${CHAT_MODEL:-ollama@qwen3.5:0.8b}"
EMBED_MODEL="${EMBED_MODEL:-ollama@nomic-embed-text-v2-moe}"

pass=0; fail=0
ok()  { echo "  ✓ $1"; pass=$((pass+1)); }
bad() { echo "  ✗ $1"; fail=$((fail+1)); }

echo "Proxy: $PROXY_URL"
echo "Chat:  $CHAT_MODEL   Embed: $EMBED_MODEL"
echo ""

# 1. version
v=$(curl -sk "$PROXY_URL/v1/system/version" \
  | python3 -c 'import sys,json;print(json.load(sys.stdin).get("version",""))' 2>/dev/null)
[ -n "$v" ] && ok "version endpoint -> $v" || bad "version endpoint"

# 2. non-stream chat — response must carry content. reasoning_effort=none keeps
#    thinking models from burning the small max_tokens budget on reasoning.
body=$(curl -sk -X POST "$PROXY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" -H "Authorization: Bearer $PROXY_AUTH" \
  -d "{\"model\":\"$CHAT_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with: OK\"}],\"max_tokens\":20,\"reasoning_effort\":\"none\"}")
content=$(printf '%s' "$body" | python3 -c '
import sys,json
try:
    d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"] or "")
except Exception: print("")' 2>/dev/null)
[ -n "$content" ] && ok "non-stream chat -> \"$content\"" || bad "non-stream chat (no content)"

# 3. STREAM chat — at least one chunk must carry a non-empty content delta.
#    reasoning_effort=none (thinking models otherwise eat the token budget).
#    (This is the regression guard: a stream of only role/finish chunks is broken.)
stream=$(curl -sk -N -X POST "$PROXY_URL/v1/chat/completions" \
  -H "Content-Type: application/json" -H "Authorization: Bearer $PROXY_AUTH" \
  -d "{\"model\":\"$CHAT_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Count: 1 2 3\"}],\"max_tokens\":30,\"stream\":true,\"reasoning_effort\":\"none\"}")
flowed=$(printf '%s' "$stream" | python3 -c '
import sys,json
n=0
for line in sys.stdin:
    line=line.strip()
    if not line.startswith("data: ") or line=="data: [DONE]": continue
    try:
        d=json.loads(line[6:]); c=d.get("choices",[{}])[0].get("delta",{}).get("content")
        if c: n+=1
    except Exception: pass
print(n)')
done_present=$(printf '%s' "$stream" | grep -c '^\[DONE\]\|^data: \[DONE\]' || true)
if [ "${flowed:-0}" -gt 0 ]; then
  ok "stream chat -> $flowed content chunk(s); [DONE]=$done_present"
else
  bad "stream chat -> 0 content chunks (REGRESSION: content not flowing)"
fi

# 4. embeddings (skip if EMBED_MODEL empty)
if [ -n "$EMBED_MODEL" ]; then
  ebody=$(curl -sk -X POST "$PROXY_URL/v1/embeddings" \
    -H "Content-Type: application/json" -H "Authorization: Bearer $PROXY_AUTH" \
    -d "{\"model\":\"$EMBED_MODEL\",\"input\":[\"hello\"]}")
  edim=$(printf '%s' "$ebody" | python3 -c '
import sys,json
try:
    d=json.load(sys.stdin); print(len(d["data"][0]["embedding"]))
except Exception: print("")' 2>/dev/null)
  [ -n "$edim" ] && ok "embeddings -> dim=$edim" || bad "embeddings (no vector)"
fi

echo ""
echo "Result: $pass passed, $fail failed"
exit $fail
