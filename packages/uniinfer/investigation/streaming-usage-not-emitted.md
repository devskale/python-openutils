# Bug: streaming `/v1/chat/completions` never emits `usage` — clients see `?` for context %, zeroed token stats

> **Status: FIXED** (0.6.13) — both layers patched; regression tests added
> (`test_router_stream_emits_terminal_usage_chunk`,
> `test_router_stream_omits_usage_when_not_requested`).

**Severity:** high — breaks context-window accounting and cost display for every streaming consumer (pi, and any OpenAI-compatible client that relies on `stream_options.include_usage`).
**Provider affected:** TU (vLLM) — and any backend whose usage arrives on the final SSE chunk. Confirmed for `tu@glm-5.2-744b-preview`.
**Non-streaming:** unaffected (usage is plumbed through).

## Summary

A streaming `/v1/chat/completions` response from the proxy **never contains a `usage` block**. Every chunk the client receives has `usage` omitted, so the client falls back to `0` for all token counts. Non-streaming responses carry `usage` correctly.

Concrete downstream symptom (pi): after auto-compaction the context-percentage slot shows `?/131k (auto)` and **stays** `?`, because pi-core refuses to estimate context size post-compaction and requires a real `usage` from an assistant message *after* the compaction boundary. Since the proxy never delivers one, the condition never clears. The token stats (`↑↓RW`), cache-hit rate (`CH`), and cost (`$`) are also all `0` for the whole session.

## Evidence

Session log from a real `tu@glm-5.2-744b-preview` session (`~/.pi/agent/sessions/.../2026-07-17...jsonl`), 116 assistant messages from `amd-local`:

```
=== ALL amd-local usage values ===
   2 stop=aborted  in=0 out=0 total=0
  11 stop=stop     in=0 out=0 total=0
 102 stop=toolUse  in=0 out=0 total=0
```

All 116 responses, zero usage. The same session's `zai` messages carry real usage (`input:515, output:202, cacheRead:106048, totalTokens:106765`), so this is proxy/provider-specific, not a client-side parse bug.

## Root cause — two layers

### Layer 1 (provider): `TUProvider.astream_complete` discards upstream usage

`uniinfer/providers/tu.py`, `astream_complete`:

```python
yield ChatCompletionResponse(
    message=message,
    provider=self._CREDGOO_SERVICE,
    model=data.get("model", request.model),
    usage={},                          # <-- always empty dict
    raw_response=data,                 # <-- data DOES contain data["usage"] on the final chunk
    finish_reason=finish_reason,
    thinking=reasoning_content,
)
```

Every yielded chunk sets `usage={}`. The upstream vLLM usage (which arrives in `data["usage"]` on the final chunk, standard OpenAI streaming shape) is captured into `raw_response=data` but **never copied onto `response.usage`**. The non-streaming `acomplete` does this correctly (`usage=data.get("usage", {})`); the streaming path was just never wired up.

### Layer 2 (proxy SSE): the streaming router never emits a `usage` chunk even if the provider supplied one

`uniinfer/proxy_services/streaming.py`, `format_chunk_to_openai` and `astream_response_generator`:

- `format_chunk_to_openai` builds each SSE chunk as `{id, object, created, model, choices}` — **no top-level `usage` key is ever set**, regardless of what `response.usage` holds.
- The generator does capture usage into a local `_stats_usage` (for the stats recorder) but **never emits it to the client**.
- The OpenAI streaming contract for usage is: when the client sends `stream_options: {include_usage: true}`, the server emits a **final chunk** with `choices: []` and a top-level `usage` object. The proxy never does this, so even if Layer 1 were fixed, clients requesting usage would still see nothing.

Layer 2 is the reason **every** streaming backend (not just TU) drops usage — but in practice most other providers are consumed non-streaming or the client tolerates the loss, so TU is where it bites.

## Why the client can't recover

pi-core (`getContextUsage`, `agent-session.js`) post-compaction:

```js
const contextTokens = calculateContextTokens(assistant.usage);  // = totalTokens || input+output+cacheRead+cacheWrite
if (contextTokens > 0) { hasPostCompactionUsage = true; break; }
// …
if (!hasPostCompactionUsage) return { tokens: null, contextWindow, percent: null };  // -> footer renders "?"
```

With `totalTokens: 0` this never fires → `percent: null` → `?`. Nothing in the client can invent the number; pi deliberately trusts the provider's usage after compaction rather than guessing. Same logic zeroes `↑↓RW`, `CH`, and `$`.

## Expected behavior

1. `TUProvider.astream_complete` (and any other provider that drops usage in the stream) must forward `data.get("usage", {})` onto the yielded `ChatCompletionResponse.usage`, matching `acomplete`.
2. The proxy streaming router must honor `stream_options: {include_usage: true}`:
   - emit a terminal chunk of shape `{"id","object":"chat.completion.chunk","created","model","choices":[],"usage":{...}}` immediately before `data: [DONE]`, **when the client requested it**;
   - and/or always attach `usage` to the final `finish_reason` chunk (some clients only look there).
3. The `usage` object should be normalized to the OpenAI shape the non-streaming path already produces (`prompt_tokens`, `completion_tokens`, `total_tokens`, optional `prompt_tokens_details`/`completion_tokens_details`), not a bare `{}`.

## Minimal fix sketch

**Layer 1** — `uniinfer/providers/tu.py`, in the `astream_complete` yield:

```python
yield ChatCompletionResponse(
    message=message,
    provider=self._CREDGOO_SERVICE,
    model=data.get("model", request.model),
    usage=data.get("usage") or {},     # was: usage={}
    raw_response=data,
    finish_reason=finish_reason,
    thinking=reasoning_content,
)
```

**Layer 2** — `uniinfer/proxy_services/streaming.py`:
- in `format_chunk_to_openai`, copy `response.usage` onto `chunk_data["usage"]` when present;
- in `astream_response_generator`, track the last seen usage (it already collects `_stats_usage` from `raw_response`); after the final `finish_reason` chunk, if the request asked for `include_usage`, emit one more chunk with `choices: []` + that `usage`, before `data: [DONE]`.

The `stream_options` passthrough already reaches the backend (OpenAI passthrough, commit `d83e461`), so vLLM is already being *asked* for usage — the proxy just isn't forwarding the answer.

## Repro

```bash
# Against the running proxy (amd-1, port 8124). Non-stream works; stream drops usage.
# Non-streaming — observe usage present:
curl -s http://localhost:8124/v1/chat/completions \
  -H "Authorization: Bearer $PROXY_KEY" -H "Content-Type: application/json" \
  -d '{"model":"tu@glm-5.2-744b-preview","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"stream":false}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin).get("usage"))'
# -> {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...}

# Streaming — observe no chunk carries usage:
curl -N -s http://localhost:8124/v1/chat/completions \
  -H "Authorization: Bearer $PROXY_KEY" -H "Content-Type: application/json" \
  -d '{"model":"tu@glm-5.2-744b-preview","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"stream":true,"stream_options":{"include_usage":true}}' \
  | grep -o '"usage":[^}]*}' | head
# -> (no output)
```

## Files

- `uniinfer/providers/tu.py` — `astream_complete` (Layer 1, the provider-specific drop)
- `uniinfer/proxy_services/streaming.py` — `format_chunk_to_openai`, `astream_response_generator` (Layer 2, the SSE never emits usage)
- `uniinfer/tests/test_usage_nested.py` — existing coverage only asserts non-streaming usage; add a streaming-usage assertion
