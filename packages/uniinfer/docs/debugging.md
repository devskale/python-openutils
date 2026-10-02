# Debugging provider requests end-to-end

When a client (pi, scripts) sees empty answers, 400s or timeouts through the
proxy, debug against **captured real payloads** — never against reconstructions
from pi session files (session format ≠ wire format; `toolCall` lives in
`content` there, in `tool_calls` on the wire — a reconstructed request has
burned whole debug cycles).

## Capture-first recipe

1. **Start the capture proxy** (`scripts/debug-capture-proxy.py`):
   `python3 scripts/debug-capture-proxy.py &` — listens on 127.0.0.1:8199,
   relays verbatim to `https://uniinfer.skale.dev` (arg 1 = other upstream),
   logs every request/response pair to `/tmp/captured.jsonl`.
2. **Point the client at it**: in pi's `~/.pi/agent/models.json` set
   `providers.unii.baseUrl = http://127.0.0.1:8199/v1` (`authHeader: true`
   passes the real token through — do NOT hardcode tokens in the proxy).
   **Restore the URL when done** — a stale capture target fails every later
   request with connection-refused.
3. **Reproduce** with the real client + session (`pi --session <id> -p "hi"`).
   The captured `req` is the exact wire payload.
4. **Bisect against the live gateway**: cut the captured `messages` array in
   half, send both halves with the normalizer under test applied, walk down to
   the single message/shape that flips OK ↔ broken. Verify the fix against the
   captured payload, not a hand-built approximation.

## Shape normalization (strict gateways)

`normalize_tool_history` + `normalize_tool_call_ids` (openai_compatible.py)
repair the transcript shapes strict gateways reject; providers opt in per flag
(`STRICT_TOOL_HISTORY`, `NORMALIZE_TOOL_CALL_IDS`). The rejected shapes and the
rewrite rules are documented at `normalize_tool_history` and in
[issues.md](issues.md) §7.

## Empty completions (throttling shadow)

`finish=stop` with zero content/tokens is never a legitimate answer. kilo's
free tier answers 200 + empty when the per-IP budget is spent (and some
replicas do it at random — same request 10×: 3 empty). The proxy replays such
attempts on a fresh connection (`EMPTY_COMPLETION_RETRIES`) and surfaces a real
429 with `retry_after` once the budget is spent.
