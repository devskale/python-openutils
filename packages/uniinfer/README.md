# UniInfer

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Unified Python interface for **LLM chat completions, embeddings, TTS, and STT** across 20+ providers with automatic API key management and an OpenAI-compatible proxy server.

## Quick Start

```bash
# Into a venv (fastest)
uv pip install -r https://skale.dev/uniinfer

# Standalone CLI (no venv needed)
uv tool install "uniinfer @ git+https://github.com/devskale/python-openutils.git#subdirectory=packages/uniinfer"
```

```python
from uniinfer import ProviderFactory, ChatMessage, ChatCompletionRequest

provider = ProviderFactory.get_provider("openai")
response = provider.complete(ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="Hello!")],
    model="gpt-4",
))
print(response.message.content)
```

## Features

- Single API for 20+ LLM providers
- Secure API key management via credgoo
- Streaming, embeddings, TTS, STT
- Automatic fallback strategies
- OpenAI-compatible FastAPI proxy server
- CLI tool for quick interactions

## Installation

**As a dependency** in `pyproject.toml`:
```toml
[project]
dependencies = ["uniinfer"]

[tool.uv.sources]
uniinfer = { git = "https://github.com/devskale/python-openutils.git", subdirectory = "packages/uniinfer" }
```

**From source**:
```bash
git clone https://github.com/devskale/python-openutils.git
cd python-openutils/packages/uniinfer && uv sync
```

Requires Python 3.9+. Provider-specific packages: `uv sync --extra anthropic`.

## Usage

### Streaming

```python
provider = ProviderFactory.get_provider("anthropic")
for chunk in provider.stream_complete(ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="Tell me a story")],
    model="claude-3-sonnet-20240229",
    streaming=True,
)):
    print(chunk.message.content, end="", flush=True)
```

### Embeddings

```python
from uniinfer import EmbeddingProviderFactory, EmbeddingRequest

provider = EmbeddingProviderFactory.get_provider("ollama")
response = provider.embed(EmbeddingRequest(
    input=["Hello world", "Machine learning"],
    model="nomic-embed-text:latest",
))
```

### Fallback Strategies

```python
from uniinfer import FallbackStrategy

strategy = FallbackStrategy(provider_names=["openai", "anthropic", "ollama"])
response, provider_name = strategy.complete(request)
```

## CLI

```bash
uniinfer -p openai -q "Hello" -m gpt-4           # chat
uniinfer -p anthropic -m claude-3-opus-20240229   # interactive
uniinfer -p openai --list-models                   # list models
uniinfer --new-models 7                            # models added in last 7 days
uniinfer --deprecated-models                       # deprecated models
uniinfer -p openrouter --speedtest -m google/gemma-4-26b-a4b-it:free  # benchmark
```

## Supported Providers

**20 providers, 979 models.** See [docs/providers.md](docs/providers.md) for full details.

| Provider | Chat | Embed | TTS | STT | Free | Models |
|----------|:----:|:-----:|:---:|:---:|:----:|-------:|
| OpenRouter | ✅ | — | — | — | ✅ | 337 |
| NVIDIA NGC | ✅ | — | — | — | — | 120 |
| OpenAI | ✅ | ✅ | ✅ | — | — | 112 |
| Mistral | ✅ | — | — | — | — | 69 |
| Cloudflare | ✅ | — | — | — | — | 58 |
| Pollinations | ✅ | — | — | — | ✅ | 57 |
| Gemini | ✅ | — | — | — | — | 55 |
| Arli AI | ✅ | — | — | — | — | 52 |
| StepFun | ✅ | — | — | — | — | 35 |
| Groq | ✅ | — | — | — | — | 16 |
| Chutes | ✅ | — | — | — | — | 13 |
| TU Wien | ✅ | ✅ | ✅ | ✅ | — | 9 |
| Moonshot | ✅ | — | — | — | — | 9 |
| Upstage | ✅ | — | — | — | — | 8 |
| Z.AI | ✅ | — | — | — | ✅ | 7 |
| Z.AI Code | ✅ | — | — | — | — | 7 |
| SambaNova | ✅ | — | — | — | — | 6 |
| Ollama | ✅ | ✅ | — | — | ✅ | 5 |
| OpenAI TTS | — | — | ✅ | — | — | 2 |
| AI21 | ✅ | — | — | — | — | 2 |
| Anthropic | ✅ | — | — | — | — | — |
| Cohere | ✅ | — | — | — | — | — |
| HuggingFace | ✅ | — | — | — | — | — |
| InternLM | ✅ | — | — | — | — | — |
| MiniMax | ✅ | — | — | — | — | — |

Model counts from `models.json` — regenerated daily at 04:00 UTC. Free = provider offers free-tier models.

## API Server

OpenAI-compatible FastAPI proxy:

```bash
uv run uvicorn uniinfer.uniioai_proxy:app --host 0.0.0.0 --port 8123
```

### Authentication

Bearer token in `Authorization` header. Two formats:

1. **Direct API key**: `Bearer YOUR_PROVIDER_API_KEY`
2. **Credgoo token** (multi-provider): `Bearer CREDGOO_BEARER@ENCRYPTION_KEY`

Falls back to `CREDGOO_BEARER_TOKEN` and `CREDGOO_ENCRYPTION_KEY` env vars if no header.

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completions |
| `/v1/embeddings` | POST | Embeddings |
| `/v1/models` | GET | All models (cached) |
| `/v1/catalog` | GET | Raw nested catalog (`?providers=openai,gemini`, `&download=1`) — public, no auth |
| `/v1/models/{provider}` | GET | Live provider models |
| `/v1/models/new?days=7` | GET | Recently added models |
| `/v1/models/deprecated` | GET | Deprecated models |
| `/v1/images/generations` | POST | Image generation |
| `/v1/audio/speech` | POST | TTS |
| `/v1/audio/transcriptions` | POST | STT |
| `/v1/system/version` | GET | Package version |

### Security

- Rate limiting: chat 100/min, embeddings 200/min, media 50/min (configurable via env)
- Auth enforced on `/v1/chat/completions`, `/v1/embeddings`, `/v1/images/*`, `/v1/audio/*`
- Ollama bypasses auth for local dev
- Configure: `UNIINFER_RATE_LIMIT_CHAT`, `UNIINFER_RATE_LIMIT_EMBEDDINGS`, `UNIINFER_RATE_LIMIT_MEDIA`

### Example Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8123/v1",
    api_key="YOUR_CREDGOO_BEARER@ENCRYPTION",
)

response = client.chat.completions.create(
    model="tu@glm-4.7-355b",
    messages=[{"role": "user", "content": "Hello!"}],
)
```

> Model IDs are `provider@model` format for the proxy (e.g. `tu@glm-4.7-355b`).

## Configuration

API keys are managed by credgoo:

```bash
credgoo set openai YOUR_API_KEY
credgoo list
```

Override with env vars for testing: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`.

## Development

```bash
uv sync                              # install deps
uv run pytest                        # run tests
uv run pytest -k "test_name"         # single test
uv run black . && uv run isort . && uv run ruff check . --fix  # format + lint
```

See [AGENTS.md](AGENTS.md) for contributor rules, [ARCHITECTURE.md](ARCHITECTURE.md) for proxy layout, and [docs/models.md](docs/models.md) for the model catalog.

## Contributing

1. Fork → feature branch → tests → all tests pass → format → commit → PR
2. See [AGENTS.md](AGENTS.md) for provider implementation patterns

## License

MIT
