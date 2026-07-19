# llminvoke

Shared LLM invocation layer for **kontext.one** — bridges
[uniinfer](https://github.com/devskale/python-openutils/tree/main/packages/uniinfer)
(the inference library) + [credgoo](https://github.com/devskale/python-openutils/tree/main/packages/credgoo)
(the API key manager) behind one small interface.

## Install

```bash
pip install git+https://github.com/devskale/python-openutils.git#subdirectory=packages/llminvoke
```

uv workspace (`[tool.uv.sources]`):
```toml
llminvoke = { git = "https://github.com/devskale/python-openutils.git", subdirectory = "packages/llminvoke" }
```

## Interface — three levels

```python
from llminvoke import call_llm, invoke_llm, stream_llm
```

### `call_llm` — full invocation (retry + extraction) → `str`

```python
# simple
text = call_llm("Du bist ein Prüfer…", model="qwen-3.6-35b", provider="tu")

# with system prompt
text = call_llm(prompt=user_text, system_prompt="Du bist ein Experte", model=…, provider=…)

# with multipart messages (image)
text = call_llm(messages=[sys_msg, user_with_image], model=…, provider=…)

# with retry
text = call_llm(prompt=…, model=…, provider=…, max_attempts=3)

# passthrough kwargs (e.g. chat_template_kwargs for thinking-model control)
text = call_llm(prompt=…, model=…, provider=…, chat_template_kwargs={"enable_thinking": False})
```

Handles: credgoo key → provider → request → `.complete()` → `extract_response_text()`
(thinking-model aware). Retries on error/empty with exponential backoff (2s, doubled).

### `invoke_llm` — one-shot, raw response

```python
response = invoke_llm(model=…, provider=…, messages=[…])
# response.message.content, response.usage, response.raw_response, etc.
```

For consumers that need the raw `ChatCompletionResponse` (usage data, error
classification, circuit-breaker decisions). No retry, no extraction.

### `stream_llm` — streaming chunks → `Iterator[str]`

```python
for chunk_text in stream_llm(prompt=…, model=…, provider=…):
    print(chunk_text, end="", flush=True)
```

Yields chunk texts as they arrive. One-shot (no retry). For verbose/progressive
output during development.
