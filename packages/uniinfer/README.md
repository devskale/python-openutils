# UniInfer

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/uniinfer.svg)](https://pypi.org/project/uniinfer/)
[![PyPI Version](https://img.shields.io/pypi/v/uniinfer.svg)](https://pypi.org/project/uniinfer/)

UniInfer provides a consistent Python interface for **LLM chat completions and text embeddings** across multiple providers with seamless API key management and OpenAI-compatible endpoints.

## Features

- 🚀 Single API for 20+ LLM providers (OpenAI, Anthropic, Google Gemini, Mistral, etc.)
- 🔑 Secure API key management with credgoo integration
- ⚡ Real-time streaming support
- 📊 Embedding support for semantic search
- 🗣️ Text-to-Speech (TTS) & Speech-to-Text (STT) support
- 🔄 Automatic fallback strategies
- 📋 Model discovery and management
- 🌐 OpenAI-compatible FastAPI proxy server

## Installation

```bash
# Install
uv venv
uv pip install -r https://skale.dev/uniinfer
```

### Requirements

- Python 3.7+
- credgoo (for API key management)
- Provider-specific packages (installed via extras)

## Quick Start

### Basic Chat Completion

```python
from uniinfer import ProviderFactory, ChatMessage, ChatCompletionRequest

# Get provider (API key retrieved automatically via credgoo)
provider = ProviderFactory.get_provider("openai")

# Create request
request = ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="Hello, how are you?")],
    model="gpt-4",
    temperature=0.7
)

# Get response
response = provider.complete(request)
print(response.message.content)
```

### Streaming Chat Completion

```python
from uniinfer import ProviderFactory, ChatMessage, ChatCompletionRequest

provider = ProviderFactory.get_provider("anthropic")

request = ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="Tell me a story")],
    model="claude-3-sonnet-20240229",
    streaming=True
)

# Stream response
for chunk in provider.stream_complete(request):
    print(chunk.message.content, end="", flush=True)
```

### Text Embeddings

```python
from uniinfer import EmbeddingProviderFactory, EmbeddingRequest

# Get embedding provider (API keys managed automatically)
provider = EmbeddingProviderFactory.get_provider("ollama")

# Create embedding request
request = EmbeddingRequest(
    input=["Hello world", "How are you?", "Machine learning is awesome"],
    model="nomic-embed-text:latest"
)

# Get embeddings
response = provider.embed(request)

# Process results
print(f"Generated {len(response.data)} embeddings")
for i, embedding_data in enumerate(response.data):
    embedding = embedding_data['embedding']
    print(f"Text {i+1}: {len(embedding)} dimensions")
    print(f"First 5 values: {embedding[:5]}")
```

## API Documentation

### Core Classes

#### ChatMessage

```python
from uniinfer import ChatMessage

message = ChatMessage(
    role="user",  # "user", "assistant", "system"
    content="Your message here"
)
```

#### ChatCompletionRequest

```python
from uniinfer import ChatCompletionRequest, ChatMessage

request = ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="Hello")],
    model="gpt-4",
    temperature=0.7,      # 0.0-2.0, higher = more creative
    max_tokens=1000       # Maximum tokens to generate
)
```

#### ChatCompletionResponse

```python
response = provider.complete(request)
print(response.message.content)      # Response text
print(response.message.role)         # "assistant"
print(response.usage.total_tokens)   # Token usage
print(response.model)                 # Model used
```

### Fallback Strategies

```python
from uniinfer import FallbackStrategy, ChatMessage, ChatCompletionRequest

# Create fallback strategy
strategy = FallbackStrategy(
    provider_names=["openai", "anthropic", "ollama"]
)

request = ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="Hello")],
    model="gpt-4"
)

response, provider_name = strategy.complete(request)
print(provider_name, response.message.content)
```

## CLI Usage

UniInfer provides a comprehensive command-line interface:

```bash
# Basic chat
uv run uniinfer -p openai -q "Hello, how are you?" -m gpt-4

# Interactive mode
uv run uniinfer -p anthropic -m claude-3-opus-20240229

# List available models
uv run uniinfer -p openai --list-models

# Embeddings
uv run uniinfer -p ollama --embed --embed-text "Hello world" --model nomic-embed-text:latest

# Text-to-Speech
uv run uniinfer -p tu --tts --tts-text "Hello world" --model kokoro

# Speech-to-Text
uv run uniinfer -p tu --stt --audio-file speech.mp3 --model whisper-large

# With streaming
uv run uniinfer -p openai -q "Tell me a story" -m gpt-4 --stream
```

## API Server

UniInfer includes an OpenAI-compatible FastAPI server:

```bash
# Install server dependencies
uv pip install uniinfer[api]

# Start server
uv run uvicorn uniinfer.uniioai_proxy:app --host 0.0.0.0 --port 8123
```

### Authentication

The API server uses the Bearer token scheme in the `Authorization` header. You can provide the token in two formats:

1. **Direct API Key**: If you are using a single provider, you can pass the provider's API key directly.

   ```
   Authorization: Bearer YOUR_PROVIDER_API_KEY
   ```

2. **Credgoo Token**: To use multiple providers managed by `credgoo`, pass a combined token in the format `bearer_token@encryption_key`.
   ```
   Authorization: Bearer YOUR_CREDGOO_BEARER@YOUR_ENCRYPTION_KEY
   ```
   You can retrieve these values from your environment variables (`CREDGOO_BEARER_TOKEN` and `CREDGOO_ENCRYPTION_KEY`) or your credgoo configuration.

If no token is provided in the header, the server will attempt to use the `CREDGOO_BEARER_TOKEN` and `CREDGOO_ENCRYPTION_KEY` environment variables as a fallback.

### Endpoints

- `POST /v1/chat/completions` - OpenAI-compatible chat completions
- `POST /v1/embeddings` - OpenAI-compatible embeddings
- `GET /v1/models` - List available models
- `POST /v1/audio/speech` - Text-to-Speech
- `POST /v1/audio/transcriptions` - Speech-to-Text

### Security Features

The API server includes built-in security features for production deployments:

1.  **Rate Limiting**: Protects your API from abuse.
    - **Chat Completions**: 100 requests/minute (default)
    - **Embeddings**: 200 requests/minute (default)
    - **Media Generation**: 50 requests/minute (default)
    - **Headers**: Responses include `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` headers.
    - **Configuration**: Configurable via environment variables.

2.  **Authentication**: Enforces strict token validation.
    - **Protected Endpoints**: `/v1/chat/completions`, `/v1/embeddings` (except Ollama), `/v1/images/*`, `/v1/audio/*`.
    - **Public Endpoints**: `/v1/models`, `/webdemo`, `/` (root).
    - **Ollama Bypass**: Local Ollama instances are accessible without authentication for easier local development.

### How to Setup Security Features

To configure security settings for your deployment:

**1. Configure Rate Limits**

Set the following environment variables to customize rate limits:

```bash
export UNIINFER_RATE_LIMIT_CHAT="50/minute"
export UNIINFER_RATE_LIMIT_EMBEDDINGS="100/minute"
export UNIINFER_RATE_LIMIT_MEDIA="10/minute"
```

**2. Configure Authentication**

Ensure your clients send the correct Bearer token in the `Authorization` header:

```http
Authorization: Bearer YOUR_PROVIDER_API_KEY
# OR for multi-provider access via credgoo:
Authorization: Bearer YOUR_CREDGOO_TOKEN@YOUR_ENCRYPTION_KEY
```

**3. Verify Security**

You can verify your security configuration using the included test script:

```bash
uv run python uniinfer/tests/verify_live_proxy.py
```

### Example Client Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8123/v1",
    api_key="dummy-key"  # UniInfer uses credgoo for actual auth
)

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

## Configuration

### API Key Management

UniInfer uses credgoo for secure API key management:

```bash
# Set API key
credgoo set openai YOUR_API_KEY

# List stored keys
credgoo list

# Remove key
credgoo remove openai
```

### Environment Variables

Optional environment variables:

```bash
# Override default API key for testing
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
```

## Supported Providers

| Provider                                             | Chat Models                | Embedding Models       | Streaming |
| ---------------------------------------------------- | -------------------------- | ---------------------- | --------- |
| OpenAI                                               | GPT-4, GPT-3.5             | text-embedding-ada-002 | ✅        |
| Anthropic                                            | Claude 3 Opus/Sonnet/Haiku | -                      | ✅        |
| Mistral                                              | Mistral Large/Small        | mistral-embed          | ✅        |
| Google Gemini                                        | Gemini Pro/Flash           | text-embedding-004     | ✅        |
| Ollama                                               | Llama2, Mistral, etc.      | nomic-embed-text, jina | ✅        |

### Gemini Async Support

**Async Support**: ✅ Gemini provider supports both sync and async operations (`complete()`, `stream_complete()`, `acomplete()`, `astream_complete()`).

**Native Async**: Uses `genai.AsyncClient` which is natively async for optimal performance.

```python
# Example usage
from uniinfer import GeminiProvider, ChatMessage, ChatCompletionRequest
import asyncio

provider = GeminiProvider(api_key="your-gemini-api-key")

# Async completion
async def async_example():
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Hello!")],
        model="gemini-1.5-flash"
    )
    response = await provider.acomplete(request)
    print(response.message.content)
    await provider.close()

# Async streaming
async def async_stream_example():
    request = ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="Tell me a story")],
        model="gemini-1.5-flash"
    )
    async for chunk in provider.astream_complete(request):
        print(chunk.message.content, end="", flush=True)
    await provider.close()

# Run async examples
asyncio.run(async_example())
asyncio.run(async_stream_example())

# List available models
models = GeminiProvider.list_models()
print(f"Available models: {len(models)}")
for i, model in enumerate(models):
    print(f"  {i+1}. {model.get('id')}")
```
| OpenRouter                                           | 60+ models                 | Various                | ✅        |
| HuggingFace                                          | Llama, Mistral             | sentence-transformers  | ✅        |
| Cohere                                               | Command R+                 | embed-english-v3.0     | ✅        |
| Groq                                                 | Llama 3.1                  | -                      | ✅        |
| AI21                                                 | Jamba 1.5                  | -                      | ✅        |
| Moonshot                                             | Kimi                       | -                      | ✅        |
| Arli AI                                              | Qwen 2.5, Llama 3.1        | -                      | ✅        |
| Sambanova                                            | Llama 3.1                  | -                      | ✅        |
| Upstage                                              | Solar                      | -                      | ✅        |
| NGC                                                  | Llama 3.1                  | -                      | ✅        |
| Cloudflare                                           | Llama 3.1                  | -                      | ✅        |
| Bigmodel                                             | GLM-4                      | -                      | ✅        |
| [Tu AI](https://github.com/TU-Wien-dataLAB/aqueduct) | Various                    | -                      | ✅        |
| Chutes                                               | Various                    | -                      | ✅        |
| [Pollinations](https://pollinations.ai/)             | 30+ models (GPT, Claude, Gemini, etc.) | -                      | ✅        |

### Pollinations Support

Pollinations.ai provides access to 30+ generative AI models (OpenAI GPT-5, Anthropic Claude 4.5, Google Gemini 3, Mistral, etc.) via a unified OpenAI-compatible API.

- **GitHub**: [https://github.com/pollinations/pollinations/](https://github.com/pollinations/pollinations/)
- **Documentation**: [https://pollinations.ai/docs](https://pollinations.ai/docs)
- **API Reference**: [https://enter.pollinations.ai/api/docs](https://enter.pollinations.ai/api/docs)
- **Get API Key**: [https://enter.pollinations.ai](https://enter.pollinations.ai)

**Authentication Required**: Pollinations requires an API key (Secret Keys for server-side, Publishable Keys for client-side with rate limits). Keys are available at [enter.pollinations.ai](https://enter.pollinations.ai).

**Async Support**: ✅ Pollinations provider supports both sync and async operations (`complete()`, `stream_complete()`, `acomplete()`, `astream_complete()`).

```python
# Example usage
from uniinfer import PollinationsProvider, ChatMessage, ChatCompletionRequest

provider = PollinationsProvider(api_key="your-pollinations-api-key")

# Sync completion
request = ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="Hello!")],
    model="openai"
)
response = provider.complete(request)
print(response.message.content)

# Async completion
import asyncio
async def async_example():
    async for chunk in provider.astream_complete(request):
        print(chunk.message.content, end="", flush=True)
asyncio.run(async_example())

# List available models
models = PollinationsProvider.list_models()
print(f"Available models: {len(models)}")
```

## Troubleshooting

### Common Issues

**ImportError: No module named 'anthropic'**

```bash
# Install the required extra
uv pip install uniinfer[anthropic]
```

**API key not found**

```bash
# Check if key is stored
credgoo list

# Set the missing key
credgoo set PROVIDER_NAME YOUR_API_KEY
```

**Connection timeout**

```python
# Increase timeout in request
request = ChatCompletionRequest(
    messages=[...],
    model="gpt-4",
    timeout=60  # seconds
)
```

**Model not found**

```bash
# List available models for the provider
uv run uniinfer -p PROVIDER_NAME --list-models
```

## Development

### Running Tests

```bash
# Install development dependencies
uv pip install -e ".[all]"

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=uniinfer --cov-report=term-mit
```

### Code Formatting

```bash
uv run black .
uv run isort .
uv run ruff check . --fix
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`uv run pytest`)
5. Format code (`uv run black . && uv run isort . && uv run ruff check . --fix`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Submit a pull request

### Adding New Providers

#### Dev Guide Index

- [OpenAI-Compatible Provider Guide](#openai-compatible-provider-guide)
- [Anthropic-Compatible Provider Guide](#anthropic-compatible-provider-guide)

#### OpenAI-Compatible Provider Guide

Use this path when the provider exposes OpenAI-style chat endpoints.

1. Create `uniinfer/providers/<provider_name>.py`
2. Inherit from `OpenAICompatibleChatProvider`
3. Set constants:
   - `BASE_URL`
   - `PROVIDER_ID`
   - `ERROR_PROVIDER_NAME`
   - `DEFAULT_MODEL`
   - `CREDGOO_SERVICE` (if key is managed via credgoo)
4. Override only provider-specific behavior:
   - `list_models()` when provider model listing differs
   - headers/default params hooks if needed
5. Export provider in `uniinfer/providers/__init__.py`
6. Register provider in `uniinfer/__init__.py` via `ProviderFactory.register_provider(...)`
7. Add tests in `uniinfer/tests/` (sync, async, streaming, list-models, error mapping)

Minimal template:

```python
from typing import Optional
from .openai_compatible import OpenAICompatibleChatProvider


class NewProvider(OpenAICompatibleChatProvider):
    BASE_URL = "https://api.example.com/v1"
    PROVIDER_ID = "newprovider"
    ERROR_PROVIDER_NAME = "NewProvider"
    DEFAULT_MODEL = "example-model"
    CREDGOO_SERVICE = "newprovider"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=base_url or self.BASE_URL)
```

#### Anthropic-Compatible Provider Guide

Use this path when the provider exposes Anthropic-style `messages` APIs.

1. Create `uniinfer/providers/<provider_name>.py`
2. Inherit from `AnthropicCompatibleProvider`
3. Set constants:
   - `BASE_URL`
   - `PROVIDER_ID`
   - `ERROR_PROVIDER_NAME`
   - `DEFAULT_MODEL`
   - `CREDGOO_SERVICE` (if key is managed via credgoo)
4. Override provider-specific behavior where needed:
   - `list_models()` for providers that do not implement Anthropic model list
   - `_default_headers()` or class header hooks for vendor headers
5. Export provider in `uniinfer/providers/__init__.py`
6. Register provider in `uniinfer/__init__.py` via `ProviderFactory.register_provider(...)`
7. Add tests in `uniinfer/tests/` (sync, async, streaming, list-models fallback, error mapping)

Minimal template:

```python
from typing import Optional
from .anthropic_compatible import AnthropicCompatibleProvider


class NewAnthropicProvider(AnthropicCompatibleProvider):
    BASE_URL = "https://api.example.com/anthropic"
    PROVIDER_ID = "newanthropic"
    ERROR_PROVIDER_NAME = "NewAnthropic"
    DEFAULT_MODEL = "example-anthropic-model"
    CREDGOO_SERVICE = "newanthropic"

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(api_key=api_key, base_url=base_url or self.BASE_URL)
```

See [AGENTS.md](AGENTS.md) for contributor rules and test expectations.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/devskale/python-openutils)
- [PyPI Package](https://pypi.org/project/uniinfer/)
- [Issues](https://github.com/devskale/python-openutils/issues)
