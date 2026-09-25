# Integrating UniInfer

Two ways to use UniInfer: as a **Python module** (in-process, no server) or via
the **uniioai proxy** (OpenAI-compatible HTTP API). Same models, same
`provider@model` routing, same credgoo key resolution.

> **Model id convention (everywhere):** `provider@model` — e.g.
> `groq@openai/gpt-oss-20b`, `ollama@qwen3.5:0.8b`, `tu@glm-5.2-744b-preview`,
> `opencode@deepseek-v4-flash-free`.
> The proxy splits on the **first** `@`. A bare id or a `:` separator will not
> route. See [AGENTS.md](../AGENTS.md) "Ollama provider" for the Ollama specifics.

---

## 1. Python module integration

Install (from the repo, editable):

```bash
cd python-openutils/packages/uniinfer
uv sync                       # or: uv sync --extra all   (all provider deps)
```

### Keys via credgoo

Providers are authenticated through [credgoo](https://github.com/devskale/python-openutils)
(`uv run credgoo <service>` to retrieve a key). `get_completion` resolves the
key automatically from the credgoo service matching the provider name:

```python
from credgoo import get_api_key
api_key = get_api_key(service="groq")        # credgoo service == provider name
```

### One-shot completion (sync + async)

```python
from uniinfer.completion import Target

# sync
resp = Target("groq@openai/gpt-oss-20b", api_key).complete(
    [{"role": "user", "content": "Say hello in one word."}],
    temperature=0.7,
    max_tokens=4096,            # thinking models need >> 1–2k
)
print(resp.message.content)

# async
import asyncio
resp = asyncio.run(Target("mistral@mistral-medium-latest", api_key).acomplete(
    [{"role": "user", "content": "Hello"}],
))
```

### Free models (OpenCode/Zen)

OpenCode (the `opencode` provider) routes to many models and offers several
for free (id ends in `-free`, plus `big-pickle`). Same `provider@model` form:

```python
# free reasoning model — give it token room (it reasons before answering)
resp = Target("opencode@deepseek-v4-flash-free", api_key).complete(
    [{"role": "user", "content": "Reply with exactly: OK"}],
    max_tokens=256,
)
print(resp.message.content)   # → OK  (resp.thinking holds the reasoning)
```

### Streaming

```python
from uniinfer.completion import Target
import asyncio

async def main():
    async for chunk in Target("tu@qwen-3.6-35b", api_key).astream_complete(
        [{"role": "user", "content": "Count to five."}],
    ):
        # Target yields raw ChatCompletionResponse chunks
        if chunk.message.content:
            print(chunk.message.content, end="", flush=True)

asyncio.run(main())
```

### Tool calling

```python
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather in a location",
        "parameters": {"type": "object",
                       "properties": {"location": {"type": "string"}},
                       "required": ["location"]},
    },
}]
resp = Target("groq@openai/gpt-oss-20b", api_key).complete(
    [{"role": "user", "content": "Weather in Paris?"}],
    tools=tools, tool_choice="auto",
)
print(resp.message.tool_calls)   # [{ "function": { "name": "get_weather", ... } }]
```

### Thinking control (Qwen3.x / GLM-5.x)

Thinking models reason before answering — powerful, but it costs tokens and
latency. Disable it for speed or deterministic output.

**Proxy / Python — OpenAI-standard `reasoning_effort`** (recommended; works
across backends, where `"minimal"` disables reasoning):

```bash
# disable reasoning, OpenAI-style
curl -s $PROXY/chat/completions -H "Authorization: Bearer $KEY" \
  -d '{"model":"tu@qwen-3.6-35b","reasoning_effort":"minimal",
        "messages":[{"role":"user","content":"What is 7*6?"}]}'
# "low" | "medium" | "high" keep reasoning on (with varying effort)
```
```python
Target(...).complete(..., reasoning_effort="none")   # "none"/"minimal" disable; or "low"/"medium"/"high"
```

**Backend-specific knobs** (the proxy maps `reasoning_effort` to these; use them
directly only if you need to bypass the mapping):

```bash
# Ollama: reasoning_effort="none" disables reasoning (the legacy `think:false`
# field still works via a deprecation shim -> reasoning_effort="none")
curl -s $PROXY/chat/completions -H "Authorization: Bearer $KEY" \
  -d '{"model":"ollama@qwen3.5:0.8b","reasoning_effort":"none",
        "messages":[{"role":"user","content":"What is 7*6?"}]}'
# vLLM (tu): chat_template_kwargs
curl -s $PROXY/chat/completions -H "Authorization: Bearer $KEY" \
  -d '{"model":"tu@qwen-3.6-35b",
        "messages":[{"role":"user","content":"hi"}],
        "chat_template_kwargs":{"enable_thinking":false}}'
```

**CLI — `--no-think`** (sets `reasoning_effort="none"`; works across backends):

```bash
uv run uniinfer -p tu -m qwen-3.6-35b --no-think -q "Summarise in one sentence: ..."
```

> Reasoning (when present) is returned as `message.reasoning_content` (non-stream)
> or `delta.reasoning_content` (stream). `--no-think` / `reasoning_effort:"none"`
> work across backends; the legacy `think` field is deprecated (shimmed to
> `reasoning_effort="none"`).

### List models + embeddings

```python
from uniinfer import ProviderFactory
models = ProviderFactory.get_provider_class("ollama").list_models(base_url=OLLAMA_URL)

from uniinfer.provider_access import get_embeddings
vec = get_embeddings(input_texts=["hello"], provider_model_string="ollama@nomic-embed-text")
```

### Capability probe (programmatic)

```python
from uniinfer.capabilities import ProbeTarget, run_capabilities, format_report
import asyncio

report = asyncio.run(run_capabilities(
    ProbeTarget(provider_model="ollama@qwen3.5:0.8b", api_key=key, base_url=url),
    perf=True, save=True,        # save -> _probe_results.json + models.json `probed`
))
print(format_report(report))
```

### Capability probe (CLI)

```bash
# single model (saves by default; --perf adds throughput/context/rate probes)
uv run uniinfer --capabilities -p ollama -m qwen3.5:0.8b --perf

# many models across providers — credgoo keys resolved per-provider internally
uv run uniinfer --capabilities \
  --models ollama@qwen3.5:0.8b groq@llama-3.3-70b-versatile gemini@models/gemini-2.5-flash
# options: --perf · --probes chat,tool_calling,image,thinking_on,thinking_off · --no-save
```

The dashboard at `/capabilities` renders whatever `--capabilities` (or
`/v1/system/capabilities?…&save=true`) writes.

---

## 2. uniioai API (the proxy)

OpenAI-compatible HTTP front. Same `provider@model` ids. Runs on `amd` as a
systemd service — **discover the serving port/exposure from the live config, never
hardcode it:** `ssh amd 'systemctl cat uniioai-proxy | grep ExecStart'` (backend
port) and `ssh amd 'sudo nginx -T | grep -B3 -A8 "server_name.*uniinfer"'`
(public TLS exposure).

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1` | GET | progressive discovery manifest (no auth) |
| `/v1/providers` | GET | enabled provider instances, catalog-only (no auth) |
| `/v1/chat/completions` | POST | chat (stream + non-stream, tools, thinking) |
| `/v1/embeddings` | POST | embeddings |
| `/v1/models` | GET | catalog (with `speed` + `probed` fields when present; `fields=` projection) |
| `/v1/models/{provider}` | GET | live models for a provider/instance (auth) |
| `/v1/images/generations` | POST | image gen |
| `/v1/system/version` | GET | health / version |
| `/v1/system/smoke` | POST | reachability smoke (`?providers=tu`) |
| `/v1/system/capabilities` | GET | capability matrix `?model=provider@x[&save=true]` |
| `/v1/system/rate-limits` | GET | learned AIMD rate-limit state |

### Agent discovery

Agents can discover the API progressively. The first two steps are public,
cached, local reads — they make no upstream/model calls and expose no secrets:

```bash
BASE=https://uniinfer.skale.dev

# 1. machine-readable manifest (also available from / with Accept: application/json)
curl -fsS "$BASE/v1"

# 2. provider/fleet-instance names only
curl -fsS "$BASE/v1/providers"

# 3. lightweight model metadata for one provider
curl -fsS "$BASE/v1/models?provider=tu&fields=id,type,context_window,capabilities"

# 4. live models for an operator-specific fleet instance (requires Bearer auth)
curl -fsS "$BASE/v1/models/zenfg" -H "Authorization: Bearer $KEY"
```

`/v1` links the OpenAPI document and guide; discovery responses include
`Link: </openapi.json>; rel="service-desc"`. `/v1/providers` reports the enabled
`alias`, whether it is builtin/custom, its underlying provider for custom
aliases, cached model count, default model, and follow-up model URLs. It never
exposes `base_url`, credgoo service names, or credentials. `fields=` works on
both the public catalog and authenticated `/v1/models/{instance}` listings and
keeps `id`, `object`, and `provider`; other requested fields must be in the
documented field allowlist.

### Authentication

Production keyed routes use an **operator-issued gateway token** as Bearer auth:

```http
Authorization: Bearer <UNIINFER_GATEWAY_TOKEN>
```

The token looks like `u…@…`, but both halves are **random gateway identity
material**, not credgoo credentials and not provider API keys. The gateway first
checks its SHA-256 hash against the allowlist; when the request reaches a keyed
provider, it resolves the real upstream key server-side through its own credgoo
configuration. A minted token therefore cannot disclose an upstream key.

Discovery is public (`/v1`, `/v1/providers`, `/v1/models`). Keyed chat/embeddings/
images/audio and live instance model listings require the Bearer token. Keyless
local instances can remain tokenless according to their instance config. In the
curl snippets below, `$KEY` means a currently issued gateway token.

#### Operator token minting

Run this on the serving box (amd), from the uniinfer package checkout:

```bash
ssh amd
cd /home/ubuntu/code/python-openutils/packages/uniinfer

# Capture stdout (the token) while leaving metadata on stderr.
# The command line contains no secret, so it is safe in shell history.
TOKEN="$(.venv/bin/python scripts/unii-token.py mint \
  --name my-agent \
  --ttl 30d \
  --provider tu)"

# For the curl snippets in this guide, use the captured token locally:
KEY="$TOKEN"
unset TOKEN
```

Output split:

- **stderr** — non-secret record: name, hash prefix, expiry, provider scope
- **stdout** — the plaintext token, printed exactly once

Install the minted token directly in the client's secret store / environment
variable. Do not write it to a repository, ticket, chat message, log, or shell
history. When you no longer need it in the current shell, run `unset KEY`.

Useful forms:

```bash
# 30 days (default), all providers
.venv/bin/python scripts/unii-token.py mint --name habit

# 7 days, only the exact `tu` provider alias
.venv/bin/python scripts/unii-token.py mint --name ci --ttl 7d --provider tu

# Multiple exact aliases: repeat the flag or use commas
.venv/bin/python scripts/unii-token.py mint --name fleet-agent \
  --ttl 4w --provider tu --provider zenfg,dgemma

# Non-expiring token — use sparingly and revoke when no longer needed
.venv/bin/python scripts/unii-token.py mint --name long-lived --ttl never
```

| Option | Meaning |
|---|---|
| `--name` | Unique active token name (`A-Z`, `a-z`, `0-9`, `_`, `.`, `-`) |
| `--ttl` | `30d` by default; accepts `<number>s|m|h|d|w` or `never` |
| `--provider` | Optional exact provider/instance alias allowlist. Omit it for all providers. |

The provider scope applies to every auth-checked route, including chat,
embeddings, images, audio, capabilities, smoke, SystemOne, and live model
listings. A token scoped to `groq` cannot list or call `tu`.

#### Inspect, rotate, and revoke

```bash
# No plaintext tokens are shown
.venv/bin/python scripts/unii-token.py list

# Remove one token immediately (no proxy restart)
.venv/bin/python scripts/unii-token.py revoke --name ci

# Remove an entry by its exact SHA-256 digest
.venv/bin/python scripts/unii-token.py revoke --hash <sha256-token-hash>

# Drop already-expired metadata entries and allowlist hashes
.venv/bin/python scripts/unii-token.py prune
```

Rotation without a client outage:

1. Mint a new versioned name, e.g. `my-agent-2026-09`.
2. Distribute that token to the client.
3. Revoke the old name.

Active names are unique, so rotate to a new name rather than trying to mint the
same active name twice.

A small client smoke test (this performs a live provider model listing):

```bash
BASE=https://uniinfer.skale.dev
curl -fsS "$BASE/v1/models/tu?fields=id" \
  -H "Authorization: Bearer $TOKEN" >/dev/null && echo "token works"
```

#### Storage and gateway behavior

On amd the registry consists of two `0600` files:

- `~/.config/uniinfer/auth_tokens.allow` — one `sha256(token)` per line; the
  hot-reload source of truth for possession
- `~/.config/uniinfer/auth_tokens.meta.json` — name, created/expiry timestamps,
  and optional provider scopes; also hot-reloaded

Alternate paths can be supplied with `--allowlist` / `--metadata` or via
`UNIINFER_AUTH_TOKENS_FILE` / `UNIINFER_AUTH_TOKENS_META_FILE`.

The CLI never writes a plaintext token to disk. Expiry and provider scopes are
enforced on every bearer request. Malformed metadata fails closed for bearer
requests while keyless routes stay available. Removing a hash revokes access
immediately; no service restart is needed.

Legacy allowlist-only tokens without metadata remain valid until their hash is
removed. They are not shown by `list` because they carry no registry metadata.

### Chat (OpenAI-shaped)

```bash
curl -s https://localhost:8123/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{
    "model": "groq@openai/gpt-oss-20b",
    "messages": [{"role":"user","content":"Reply with one word: ready"}],
    "max_tokens": 4096
  }'
```

Stream: `"stream": true` → SSE `data: {...}` chunks, terminated by `data: [DONE]`.

### Capability matrix (probe what a model can do)

```bash
curl -s "https://localhost:8123/v1/system/capabilities?model=groq@openai/gpt-oss-20b&save=true" \
  -H "Authorization: Bearer $KEY" | jq '.summary, .profile'
```

Returns `{profile, results[], summary}` — each probe `pass|fail|skip|error`.
`&save=true` persists to `_probe_results.json` + the models.json `probed` field
(visible in the **Capabilities** dashboard at `/capabilities`).

### Python (requests / openai SDK)

```python
from openai import OpenAI
client = OpenAI(base_url="https://localhost:8123/v1", api_key=os.environ["UNIINFER_GATEWAY_TOKEN"])
r = client.chat.completions.create(
    model="mistral@mistral-medium-latest",
    messages=[{"role": "user", "content": "Hello"}],
)
print(r.choices[0].message.content)
```

---

## Reference

- [AGENTS.md](../AGENTS.md) — contributor rules, provider implementation, footguns
- [docs/providers.md](providers.md) — full provider index (base URLs, defaults)
- [docs/models.md](models.md) — model catalog & metadata richness
- Web UI: `/webdemo` (chat) · `/perf` (speed) · `/capabilities` (probe matrix) · `/guide` (this page)
