# AGENTS.md - UniInfer Development Guide

This file provides guidelines for agentic coding agents operating in the uniinfer repository.

## Project Overview

UniInfer is a unified LLM inference interface for Python providing a consistent API across 20+ providers (OpenAI, Anthropic, Mistral, Ollama, etc.). Features include:

- Unified Chat/Embedding/TTS/STT API with streaming and fallback strategies
- CLI tool for quick interactions (`uniinfer_cli.py`)
- OpenAI-compatible FastAPI proxy server (`uniioai_proxy.py`)

## Environment Setup

```bash
cd $PROJECTDIR/python-utils/packages/uniinfer
source .venv/bin/activate  # Use local virtual environment
uv pip install -e ".[all]"    # Install all dependencies
```

## Build/Lint/Test Commands

### Running Tests

```bash
uv run pytest                              # All tests
uv run pytest uniinfer/tests/test_async_functionality.py    # Single file
uv run pytest uniinfer/tests/test_async.py::TestClass::test_method  # Specific test
uv run pytest -v                           # Verbose output
uv run pytest --cov=uniinfer --cov-report=term-mit         # With coverage
```

### Code Formatting

```bash
black .                             # Format code (line-length: 88)
isort .                             # Sort imports (profile: black)
ruff check .                        # Run linter
ruff check . --fix                  # Auto-fix linting issues
```

### Package Distribution

```bash
python setup.py sdist bdist_wheel   # Build package
uv pip install dist/*.whl              # Install built wheel
```

## Code Style Guidelines

### General

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, modules
- Use type hints where appropriate (Python 3.7+)
- Avoid `Any` when possible

### Imports

- Use absolute imports from package root
- Group: stdlib, third-party, local
- Sort with isort (profile: black, multi_line_output: 3)

```python
import json
import requests
from typing import Dict, Any, Iterator, Optional

from ..core import ChatProvider, ChatCompletionRequest, ChatMessage
```

### Naming Conventions

- **Classes**: PascalCase (`ChatCompletionRequest`, `OllamaProvider`)
- **Functions/variables**: snake_case (`complete()`, `api_key`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_TIMEOUT`)
- **Private methods**: leading underscore (`_normalize_base_url()`)

### Type Hints

- Use `Optional[T]` not `Union[T, None]`
- Use `List[T]`, `Dict[K, V]` from typing

```python
def complete(
    self,
    request: ChatCompletionRequest,
    **provider_specific_kwargs
) -> ChatCompletionResponse:
```

### File Structure

- `uniinfer/core.py`: Core classes (ChatMessage, ChatCompletionRequest, etc.)
- `uniinfer/providers/{provider_name}.py`: Provider implementations
- `uniinfer/errors.py`: Error handling
- `uniinfer/factory.py`, `embedding_factory.py`: Factories
- `uniinfer/tests/test_*.py`: Tests

### Error Handling

Use standardized exceptions from `uniinfer/errors.py`:

```python
from uniinfer.errors import (
    UniInferError, ProviderError, AuthenticationError,
    RateLimitError, TimeoutError, InvalidRequestError
)

mapped_error = map_provider_error("provider_name", original_error)
raise mapped_error
```

### Provider Implementation Pattern

1. Create `uniinfer/providers/newprovider.py`
2. Inherit from `ChatProvider` (or `EmbeddingProvider`, `TTSProvider`, `STTProvider`)
3. Implement: `complete()`, `stream_complete()`, `list_models()`
4. Use `map_provider_error()` for error handling
5. Register in `uniinfer/__init__.py` with `ProviderFactory.register_provider()`
6. Add conditional import for optional dependencies
7. Export in `uniinfer/providers/__init__.py`

### Testing

- Place tests in `uniinfer/tests/`
- Name: `test_*.py`
- Use pytest framework
- Mock external API calls with `unittest.mock`
- Test both sync and async methods

### API Compatibility

- Maintain backward compatibility for public APIs
- Use OpenAI-compatible response formats
- Follow OpenAI response structure for embeddings, chat completions, TTS, STT

### Version Updates

When asked to update the version in `setup.py`:

- **Minor version bump** (default): Increment the patch number (e.g., `0.3.5` → `0.3.6`)
- **Major version bump**: Increment the minor number and reset patch to 0 (e.g., `0.3.5` → `0.4.0`)

Update both `setup.py` and `pyproject.toml` if both exist.

### Response Formats

Maintain OpenAI-compatible response structures:

- **Chat Completions**: `message.content`, `message.role`, `usage.prompt_tokens`, `usage.completion_tokens`
- **Embeddings**: `data[].embedding` as a list of floats, `usage.total_tokens`
- **TTS**: Audio in base64 or binary format with `model` and `duration`
- **STT**: `text` field with transcribed content, optionally `duration` and `language`

### Async/Sync Patterns

Providers should implement both sync and async methods where applicable:

```python
async def acomplete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Async version of complete() for concurrent requests."""

def complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
    """Sync wrapper that calls acomplete() in executor."""
    import asyncio
    return asyncio.run(self.acomplete(request))
```
