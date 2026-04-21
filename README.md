# Python Open Utils

[GitHub](https://github.com/devskale/python-openutils) · [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Stars](https://img.shields.io/github/stars/devskale/python-openutils)](https://github.com/devskale/python-openutils/stargazers)
[![GitHub Issues](https://img.shields.io/github/issues/devskale/python-openutils)](https://github.com/devskale/python-openutils/issues)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A collection of powerful Python utilities for secure credential management and unified LLM inference.

## Install

```bash
# Short URL (needs active venv)
uv pip install -r https://skale.dev/credgoo
uv pip install -r https://skale.dev/uniinfer

# Standalone tool (no venv needed)
uv tool install "credgoo @ git+https://github.com/devskale/python-openutils.git#subdirectory=packages/credgoo"
uv tool install "uniinfer @ git+https://github.com/devskale/python-openutils.git#subdirectory=packages/uniinfer"

# As dependency in pyproject.toml
[project]
dependencies = ["uniinfer", "credgoo"]

[tool.uv.sources]
uniinfer = { git = "https://github.com/devskale/python-openutils.git", subdirectory = "packages/uniinfer" }
credgoo = { git = "https://github.com/devskale/python-openutils.git", subdirectory = "packages/credgoo" }
```

## Packages

### [Credgoo](packages/credgoo/) `v0.1.10`

**The Secure Credentials Manager for Everywhere.**

Securely manage your credentials in Google Sheets with encrypted Apps Script integration and access them seamlessly across all your environments.

- **Use Everywhere**: Unified access for local, dev, and production.
- **Secure Google Sheets Integration**: Encrypted Apps Script backend.
- **End-to-End Encryption**: Secure transmission and storage.
- **Smart Local Caching**: Performance optimized with restrictive permissions.
- **Centralized Control**: Single source of truth for all your secrets.

### [UniInfer](packages/uniinfer/) `v0.5.15`

OpenAI-compatible proxy server providing unified access to 500+ models from 20+ providers with multi-modal support.

- OpenAI-compatible FastAPI proxy — drop-in replacement for OpenAI API
- 500+ models across 20+ providers (OpenAI, Anthropic, Google Gemini, Mistral, etc.)
- Multi-modal support: LLM, Image, Text-to-Speech (TTS), and Transcription
- Secure API key management with credgoo integration
- Real-time streaming support
- Embedding support for semantic search
- Automatic fallback strategies

## Quick Start

```bash
# Clone and set up
git clone https://github.com/devskale/python-openutils.git
cd python-openutils

# Install each package (creates venv, resolves deps from lockfile)
cd packages/credgoo && uv sync && cd ../..
cd packages/uniinfer && uv sync && cd ../..
```

### Credgoo — Get an API Key in 3 Lines

```python
from credgoo import get_api_key

key = get_api_key("openai")
print(key[:8] + "...")  # sk-proj-...
```

First time? Set up your credentials:

```bash
cd packages/credgoo
uv run credgoo --setup
```

### UniInfer — Chat with Any Provider

```python
from uniinfer import ProviderFactory, ChatMessage, ChatCompletionRequest

provider = ProviderFactory.get_provider("openai")
request = ChatCompletionRequest(
    messages=[ChatMessage(role="user", content="Hello!")],
    model="gpt-4",
)
response = provider.complete(request)
print(response.message.content)
```

## Development

```bash
# Sync deps (creates venv + installs from uv.lock)
cd packages/credgoo && uv sync
cd packages/uniinfer && uv sync

# Run tests
cd packages/credgoo && uv run pytest
cd packages/uniinfer && uv run pytest

# Add optional provider extras
cd packages/uniinfer && uv sync --extra anthropic --extra gemini
# or install all at once
cd packages/uniinfer && uv sync --extra all
```

## Documentation

- [Credgoo Documentation](packages/credgoo/README.md)
- [UniInfer Documentation](packages/uniinfer/README.md)

## License

MIT License — see individual package directories for specific license details.
