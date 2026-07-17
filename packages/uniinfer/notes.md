# Notes

Findings and discoveries that don't fit elsewhere. Dated, append-only.

---

## 2026-07-17 — OpenCode/Zen: a richer model catalog exists at pi.dev

### Background

`opencode`'s own `GET https://opencode.ai/zen/v1/models` returns a **bare**
catalog — 7 models, fields `id`/`object`/`created`/`owned_by` only. No pricing,
no context window, no max tokens, no reasoning flag, no modalities. Our
`OpenCodeProvider.list_models` uses this endpoint and guesses "free" from the
model id (`-free` suffix, `big-pickle`).

### The pi.dev catalog

While investigating how **pi** (the coding agent) knows the full Zen model list,
its source revealed a richer, public catalog:

```
GET https://pi.dev/api/models/providers/opencode      # no key needed
```

Returns a **dict keyed by model id** — 54 entries, each with rich metadata:

```json
{
  "big-pickle": {
    "id": "big-pickle",
    "name": "Big Pickle",
    "api": "openai-completions",
    "provider": "opencode",
    "baseUrl": "https://opencode.ai/zen/v1",
    "reasoning": true,
    "input": ["text"],
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
    "compat": {"supportsStore": false, "supportsDeveloperRole": false, "maxTokensField": "max_tokens"},
    "contextWindow": 200000,
    "maxTokens": 32000
  },
  ...
}
```

Fields pi.dev gives us that `/v1/models` doesn't: `name`, `api`, `reasoning`,
`input[]` (modalities), `cost{input,output,cacheRead,cacheWrite}`,
`contextWindow`, `maxTokens`, `compat{}`, `thinkingLevelMap?`.

### How pi uses it

pi's `withRemoteCatalog()` (`remote-catalog-provider.ts`) fetches the catalog,
caches it in `~/.pi/agent/models-store.json` with a **4-hour TTL**
(`REMOTE_CATALOG_REFRESH_INTERVAL_MS = 4 * 60 * 60 * 1000`), and merges it over
the built-in models. Default base URL: `https://pi.dev`.

### The `api` field — which protocol each model speaks

This is the most useful field. The 54 models split across four protocols:

| `api` | count | reachable via our `opencode` provider? |
|---|---|---|
| `openai-completions` | 19 | ✅ yes — DeepSeek, GPT, Qwen, GLM, MiniMax, Kimi, Grok + 6 free |
| `anthropic-messages` | 13 | ❌ no — Claude; needs `https://opencode.ai/zen` (Anthropic API) |
| `google-generative-ai` | 3 | ❌ no — Gemini; Google API |
| `openai-responses` | 19 | ❌ no — GPT responses API |

So our OpenAI-compatible provider can only ever serve the 19
`openai-completions` models. The other 35 (Claude/Gemini/responses-API) would
each need their own provider subclass targeting the right base URL + protocol.

### The 6 free models (cost.input == 0, authoritative — not name-guessed)

`big-pickle`, `deepseek-v4-flash-free`, `hy3-free`, `mimo-v2.5-free`,
`nemotron-3-ultra-free`, `north-mini-code-free`.

### Decision: keep `/v1/models`

We are **not** switching to pi.dev. Rationale:

- `/v1/models` is the provider's **own** API — the canonical source. pi.dev is
  a third-party overlay that could change or disappear without notice.
- Our catalog is regenerated daily (04:00 UTC) by the proxy; the bare metadata
  is acceptable and the free-model heuristic (`-free`/`big-pickle`) has held.
- No new external dependency / cache / TTL to maintain.

If we later want richer metadata (pricing, context window, reasoning flag) in
the opencode catalog, pi.dev is the place to pull it from — filter to
`api == "openai-completions"` and map `cost`/`contextWindow`/`maxTokens`/
`reasoning`/`input` onto `ModelInfo`. Until then, this note records the option.

### If we ever add Claude/Gemini via OpenCode

The `api` field + `baseUrl` tell us exactly how:

- Claude (`anthropic-messages`, 13 models) → `https://opencode.ai/zen`, Anthropic
  messages protocol. Would need an `OpenCodeAnthropicProvider` subclass.
- Gemini (`google-generative-ai`, 3 models) → Google generative-ai protocol.
- GPT responses-API (`openai-responses`, 19 models) → OpenAI responses API.

Each is a separate provider subclass; the OpenAI-compatible `opencode` provider
only covers `openai-completions`.
