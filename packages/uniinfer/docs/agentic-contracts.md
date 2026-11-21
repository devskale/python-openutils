# Agentic Contracts — what an agent may rely on

Machine-stable promises of the uniioai proxy. Every contract here is
versioned with the proxy (`/v1` → `version`), changes are additive; breaking
changes get a new surface path, never a silent mutation. Canonical live
surfaces: [`/llms.txt`](/llms.txt) · [`/v1`](/v1) · [`/guide.md`](/guide.md).

## 1. Discovery contract

| Step | Surface | Promise |
|---|---|---|
| 1 | `GET /llms.txt` | Markdown index (llmstxt.org v2): endpoint map, model-id form, realtime-STT protocol inline |
| 2 | `GET /v1` | JSON manifest: `links` (every surface), `auth` (bearer + WS + own-key), `model_id_format_note`, `version` |
| 3 | `GET /v1/providers` | enabled instances, `model_count`, `live_models_url` per alias — no credentials leaked |
| 4 | `GET /v1/models` | full catalog (no auth); `/v1/models/{alias}` live listing (auth) |

`Link` headers: `rel="describedby"` → `/llms.txt`, `rel="service-desc"` →
`/openapi.json`, `rel="help"` → `/guide.md` (HTTP surfaces only — WS contracts
live in llms.txt/this doc because OpenAPI cannot express WebSockets).

## 2. Identity contract

- Model ids are **strict** `alias@model`: bare names → 400, no fallback guessing.
- The **first `@` binds the alias** — model ids may themselves contain `@`.
- Alias = provider instance: built-in (`mistral`, `tu`, …) or fleet alias
  (`hab` → dgx gateway). `/v1/providers` is the authoritative alias list.

## 3. Auth contract

| Mode | How | Semantics |
|---|---|---|
| **Pooled** | `Authorization: Bearer <operator token>` | unlocks every provider on pooled keys; allowlist + revocation + provider scoping |
| **WS** | `?token=` query param (browsers cannot set WS headers) or Authorization header | same tokens |
| **Own key** | `X-Provider-Key: <upstream key>` on ANY provider | no meta-bearer needed for own-key-only requests; rides the caller's upstream quota |

Invariants: a bearer that IS sent must be valid (own keys never launder a bad
bearer); keyless surfaces are exactly the ones marked `no auth` in discovery.

## 4. Surface contracts

### Chat — `POST /v1/chat/completions`
OpenAI dialect (choices/delta/tool_calls), SSE stream ends with `data: [DONE]`,
keep-alive comments during reasoning phases, transparent 429 (HTTP, not
SSE-embedded) when nothing was committed yet. Auto-routed requests carry
`x-routed-model`.

### Responses — `POST /v1/responses`
Normalized onto EVERY chat provider (uniinfer is the normalization layer):
stateless v1 = `input` (string/items incl. function round-trips),
`instructions`, `max_output_tokens`, `temperature`, `stream`, client-side
function tools. SSE events: `response.created` → `output_item.added` →
`output_text.delta`* → `output_item.done` → `response.completed` (usage with
`output_tokens_details.reasoning_tokens`). Mid-stream upstream failures →
`response.failed` (never a broken stream). Not translated → actionable 400
(`previous_response_id`, background, server-side tools). Native opt-in:
`x-native-responses: true` on base_url instances → verbatim passthrough,
else 501.

### Realtime STT — `WS /v1/audio/transcriptions/stream`
Query: `model`, `token`, `language`. Binary frames = raw int16 PCM 16 kHz
mono; **empty binary frame = end-of-audio**. Server JSON:
`{"type":"partial","partial":true,"text":…}` live,
`{"type":"final","partial":false,"text":…}` after EOF. Close codes: 4400
bad request · 4401 auth · 4404 unknown instance · 4502 upstream. ~300 ms
lead-in silence improves first-word recognition.

### Systemone — `POST /v1/systemone`
Decision dialect: `{model, state, questions}` → `{answers}` (choice/estimate
with confidence + probabilities). Catalog type is authority; payload shape is
fallback. Extraction (`fields`) needs a native instance → 400, not 502.

## 5. Reliability contract

- **429**: always carries `Retry-After` when the upstream provides one;
  quota-parked models drop out of auto-route chains live.
- **Auto-routing** (`auto@selector`): transparent fallback to the next
  candidate on 429/dead-upstream — response reports the model that served
  (`model` field + `x-routed-model`).
- **Empty completions** (200 with no content/tools/thinking): retried once,
  then a real 502 — backup chains can route around.
- **Streams**: idle timeout + heartbeats keep long reasoning phases alive;
  terminal usage chunk (`choices:[]`) closes accounting.

## 6. Versioning

`/v1.version` mirrors the package version. Contracts are additive: fields may
appear, never disappear or change meaning. New dialect capabilities land as
new surfaces or explicit opt-ins (e.g. `x-native-responses`), so a client
written against today's contracts keeps working.
