# Integrating UniInfer

Two ways to use UniInfer: as a **Python module** (in-process, no server) or via
the **uniioai proxy** (OpenAI-compatible HTTP API). Same models, same
`provider@model` routing, same credgoo key resolution.

> **Model id convention (everywhere):** `provider@model` — e.g.
> `groq@openai/gpt-oss-20b`, `ollama@qwen3.5:0.8b`, `tu@glm-5.2-744b-preview`.
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
from uniinfer.uniioai import get_completion, aget_completion

# sync
resp = get_completion(
    messages=[{"role": "user", "content": "Say hello in one word."}],
    provider_model_string="groq@openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=4096,            # thinking models need >> 1–2k
)
print(resp.message.content)

# async
import asyncio
resp = asyncio.run(aget_completion(
    messages=[{"role": "user", "content": "Hello"}],
    provider_model_string="mistral@mistral-medium-latest",
))
```

### Streaming

```python
from uniinfer.uniioai import astream_completion
import asyncio

async def main():
    async for chunk in astream_completion(
        messages=[{"role": "user", "content": "Count to five."}],
        provider_model_string="tu@qwen-3.6-35b",
    ):
        delta = chunk.get("choices", [{}])[0].get("delta", {})
        if delta.get("content"):
            print(delta["content"], end="", flush=True)

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
resp = aget_completion_sync_like = get_completion(
    messages=[{"role": "user", "content": "Weather in Paris?"}],
    provider_model_string="groq@openai/gpt-oss-20b",
    tools=tools, tool_choice="auto",
)
print(resp.message.tool_calls)   # [{ "function": { "name": "get_weather", ... } }]
```

### Thinking control (Qwen3.x / GLM-5.x)

```python
# vLLM backends (tu): chat_template_kwargs
get_completion(..., chat_template_kwargs={"enable_thinking": False})

# Ollama: native `think` field (provider-direct; see capabilities runner)
provider = ProviderFactory.get_provider("ollama", api_key=key, base_url=url)
req = ChatCompletionRequest(messages=[...], model="qwen3.5:0.8b", streaming=False)
resp = await provider.acomplete(req, think=False)
```

> ⚠️ The CLI `--no-think` flag targets **vLLM** only (`enable_thinking`), not
> Ollama's `think`.

### List models + embeddings

```python
from uniinfer import ProviderFactory
models = ProviderFactory.get_provider_class("ollama").list_models(base_url=OLLAMA_URL)

from uniinfer.uniioai import get_embeddings
vec = get_embeddings(input_texts=["hello"], provider_model_string="ollama@nomic-embed-text")
```

### Capability probe (programmatic)

```python
from uniinfer.capabilities import Target, run_capabilities, format_report
import asyncio

report = asyncio.run(run_capabilities(
    Target(provider_model="ollama@qwen3.5:0.8b", api_key=key, base_url=url),
    perf=True, save=True,        # save -> _probe_results.json + models.json `probed`
))
print(format_report(report))
```

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
export KEY="test23@test34"          # your PROXY_KEY
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
client = OpenAI(base_url="https://amd1.mooo.com:8123/v1", api_key="test23@test34")
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
