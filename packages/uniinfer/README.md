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

### Basic Installation

```bash
pip install uniinfer credgoo
```

### Install with Extras

Install specific provider support to reduce dependencies:

```bash
# Install all provider support
pip install uniinfer[all] credgoo

# Install specific providers
pip install uniinfer[gemini,anthropic,mistral,cohere] credgoo

# Install API server dependencies
pip install uniinfer[api] credgoo
```

### Install from Source

```bash
# Install via pip
pip install "git+https://github.com/devskale/python-openutils#subdirectory=packages/uniinfer"

# Or clone and install
git clone https://github.com/devskale/python-openutils.git
cd python-openutils/packages/uniinfer
pip install -e ".[all]"
```

### Install with uv

```bash
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
    max_tokens=1000,      # Maximum tokens to generate
    top_p=1.0,           # 0.0-1.0, nucleus sampling
    presence_penalty=0.0, # -2.0 to 2.0
    frequency_penalty=0.0 # -2.0 to 2.0
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
from uniinfer import ProviderFactory, FallbackStrategy, ChatMessage, ChatCompletionRequest

# Create fallback strategy
strategy = FallbackStrategy(
    providers=["openai", "anthropic", "ollama"],
    models=["gpt-4", "claude-3-opus-20240229", "llama2"]
)

# Get provider with fallback
provider = ProviderFactory.get_provider_with_strategy(strategy)

request = ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="Hello")],
    model="gpt-4"
)

response = provider.complete(request)
```

## CLI Usage

UniInfer provides a comprehensive command-line interface:

```bash
# Basic chat
uniinfer -p openai -q "Hello, how are you?" -m gpt-4

# Interactive mode
uniinfer -p anthropic -m claude-3-opus-20240229

# List available models
uniinfer -p openai --list-models

# Embeddings
uniinfer -p ollama --embed --embed-text "Hello world" --model nomic-embed-text:latest

# Text-to-Speech
uniinfer -p tu --tts --tts-text "Hello world" --model kokoro

# Speech-to-Text
uniinfer -p tu --stt --audio-file speech.mp3 --model whisper-large

# With streaming
uniinfer -p openai -q "Tell me a story" -m gpt-4 --stream
```

## API Server

UniInfer includes an OpenAI-compatible FastAPI server:

```bash
# Install server dependencies
pip install uniinfer[api]

# Start server
uvicorn uniinfer.uniioai_proxy:app --host 0.0.0.0 --port 8123
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

| Provider      | Chat Models                | Embedding Models       | Streaming |
| ------------- | -------------------------- | ---------------------- | --------- |
| OpenAI        | GPT-4, GPT-3.5             | text-embedding-ada-002 | ✅        |
| Anthropic     | Claude 3 Opus/Sonnet/Haiku | -                      | ✅        |
| Mistral       | Mistral Large/Small        | mistral-embed          | ✅        |
| Google Gemini | Gemini Pro/Flash           | text-embedding-004     | ✅        |
| Ollama        | Llama2, Mistral, etc.      | nomic-embed-text, jina | ✅        |
| OpenRouter    | 60+ models                 | Various                | ✅        |
| HuggingFace   | Llama, Mistral             | sentence-transformers  | ✅        |
| Cohere        | Command R+                 | embed-english-v3.0     | ✅        |
| Groq          | Llama 3.1                  | -                      | ✅        |
| AI21          | Jamba 1.5                  | -                      | ✅        |
| Moonshot      | Kimi                       | -                      | ✅        |
| Arli AI       | Qwen 2.5, Llama 3.1        | -                      | ✅        |
| Sambanova     | Llama 3.1                  | -                      | ✅        |
| Upstage       | Solar                      | -                      | ✅        |
| NGC           | Llama 3.1                  | -                      | ✅        |
| Cloudflare    | Llama 3.1                  | -                      | ✅        |
| Bigmodel      | GLM-4                      | -                      | ✅        |
| Tu AI         | Various                    | -                      | ✅        |
| Chutes        | Various                    | -                      | ✅        |
| Pollinations  | Free OpenAI-compatible     | -                      | ✅        |
| StepFun       | Various                    | -                      | ✅        |
| InternLM      | InternLM 2.5               | -                      | ✅        |

## Troubleshooting

### Common Issues

**ImportError: No module named 'anthropic'**

```bash
# Install the required extra
pip install uniinfer[anthropic]
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
uniinfer -p PROVIDER_NAME --list-models
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[all]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=uniinfer --cov-report=term-mit
```

### Code Formatting

```bash
black .
isort .
ruff check . --fix
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Format code (`black . && isort . && ruff check . --fix`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Submit a pull request

### Adding New Providers

To add a new provider:

1. Create a new file in `uniinfer/providers/`
2. Inherit from the appropriate provider base class (`ChatProvider`, `EmbeddingProvider`, etc.)
3. Implement required methods (`complete()`, `stream_complete()`, `list_models()`)
4. Register the provider in `uniinfer/__init__.py`
5. Add provider-specific dependencies to `setup.py`
6. Add tests in `uniinfer/tests/`

See [AGENTS.md](AGENTS.md) for detailed development guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/devskale/python-openutils)
- [PyPI Package](https://pypi.org/project/uniinfer/)
- [Issues](https://github.com/devskale/python-openutils/issues)
