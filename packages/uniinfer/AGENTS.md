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
# Run all tests
uv run pytest

# Run a single test file
uv run pytest uniinfer/tests/test_auth.py

# Run a specific test class
uv run pytest uniinfer/tests/test_auth.py::TestAuth

# Run a specific test method
uv run pytest uniinfer/tests/test_auth.py::TestAuth::test_token_validation

# Run tests by keyword (function name)
uv run pytest -k "test_token_validation"

# Verbose output with coverage
uv run pytest -v --cov=uniinfer --cov-report=term-missing
```

### Code Formatting

```bash
uv run black .                      # Format code (line-length: 88)
uv run isort .                      # Sort imports (profile: black)
uv run ruff check .                 # Run linter
uv run ruff check . --fix           # Auto-fix linting issues
```

### Package Distribution

```bash
uv run python setup.py sdist bdist_wheel   # Build package
uv pip install dist/*.whl                  # Install built wheel
```

## Writing Tests

Place tests in `uniinfer/tests/` directory. Name files `test_*.py`.

```python
# Example: uniinfer/tests/test_feature.py
import pytest
from unittest.mock import patch, MagicMock

class TestFeature:
    """Test suite for feature X."""
    
    def test_success_case(self):
        """Test the happy path."""
        assert True
    
    @patch('uniinfer.module.external_call')
    def test_with_mock(self, mock_call):
        """Test with mocked external API."""
        mock_call.return_value = {'result': 'success'}
        # test code here
        mock_call.assert_called_once()
    
    def test_error_handling(self):
        """Test error conditions."""
        with pytest.raises(ValueError):
            raise ValueError("expected")
```

Testing best practices:
- Use descriptive test names that explain what's being tested
- Mock external API calls with `unittest.mock`
- Test both sync and async methods
- Test edge cases and error conditions
- Test proxy request limits and security validators (size, message count, auth)
- Use pytest fixtures for shared setup

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
