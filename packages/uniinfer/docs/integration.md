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

OpenAI-compatible HTTP front. Same `provider@model` ids. Default dev port
`8124` (nginx TLS on `8123`); production on `amd`.

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/chat/completions` | POST | chat (stream + non-stream, tools, thinking) |
| `/v1/embeddings` | POST | embeddings |
| `/v1/models` | GET | catalog (with `speed` + `probed` fields when present) |
| `/v1/images/generations` | POST | image gen |
| `/v1/system/version` | GET | health / version |
| `/v1/system/smoke` | POST | reachability smoke (`?providers=tu`) |
| `/v1/system/capabilities` | GET | capability matrix `?model=provider@x[&save=true]` |
| `/v1/system/rate-limits` | GET | learned AIMD rate-limit state |

### Auth

Bearer token = the proxy `PROXY_KEY` (credgoo combined `bearer@encryption`):

```bash
export KEY="$PROXY_KEY"          # credgoo combined token (bearer@encryption) — never commit the real value
curl -s https://amd1.mooo.com:8123/v1/system/version -H "Authorization: Bearer $KEY"
# {"version":"0.5.44"}
```

> Ollama requests **bypass** proxy auth locally; all other providers require it.

### Chat (OpenAI-shaped)

```bash
curl -s https://amd1.mooo.com:8123/v1/chat/completions \
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
curl -s "https://amd1.mooo.com:8123/v1/system/capabilities?model=groq@openai/gpt-oss-20b&save=true" \
  -H "Authorization: Bearer $KEY" | jq '.summary, .profile'
```

Returns `{profile, results[], summary}` — each probe `pass|fail|skip|error`.
`&save=true` persists to `_probe_results.json` + the models.json `probed` field
(visible in the **Capabilities** dashboard at `/capabilities`).

### Python (requests / openai SDK)

```python
from openai import OpenAI
client = OpenAI(base_url="https://amd1.mooo.com:8123/v1", api_key=os.environ["PROXY_KEY"])   # your PROXY_KEY (bearer@encryption)
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
