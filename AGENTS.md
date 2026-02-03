# AGENTS.md - Python Open Utils

This file provides guidelines for agentic coding agents operating in this monorepo.

## Repository Structure

This is a monorepo containing Python utility packages:
- `packages/credgoo/` - Secure API key retrieval from Google Sheets with local caching
- `packages/uniinfer/` - Unified LLM inference interface across 20+ providers

## Environment Setup

```bash
# Initialize all packages with uv (creates venvs, installs deps)
./uvinit.sh -x

# Initialize specific package
./uvinit.sh -x credgoo    # or uniinfer

# Activate environment
source packages/credgoo/.venv/bin/activate  # or uniinfer/.venv/bin/activate

# Install in development mode
cd packages/credgoo && uv pip install -e .
cd packages/uniinfer && uv pip install -e ".[all]"
```

## Build/Lint/Test Commands

```bash
cd packages/uniinfer

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

# Code formatting
uv run black .         # Format code (line-length: 88)
uv run isort .         # Sort imports (profile: black)
uv run ruff check .    # Run linter
uv run ruff check . --fix    # Auto-fix issues

# Package distribution
uv run python setup.py sdist bdist_wheel
uv pip install dist/*.whl
```

## Code Style Guidelines

### General

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, modules
- Use type hints where appropriate
- Python 3.6+ for credgoo, Python 3.7+ for uniinfer
- **IMPORTANT: DO NOT ADD ANY COMMENTS unless asked**

### Imports

Use absolute imports from package root. Group: stdlib, third-party, local.

```python
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from uniinfer.core import ChatProvider, ChatCompletionRequest
```

Run `isort .` on package directory before committing (profile: black, multi_line_output: 3).

### Naming Conventions

- **Classes**: PascalCase (`ChatCompletionRequest`, `OllamaProvider`)
- **Functions/variables**: snake_case (`complete()`, `api_key`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_TIMEOUT`)
- **Private methods**: leading underscore (`_normalize_base_url()`)

### Type Hints

```python
def complete(
    self,
    request: ChatCompletionRequest,
    **kwargs
) -> ChatCompletionResponse:
    
def get_api_key(service: str, cache_dir: Optional[Path] = None) -> Optional[str]:
```

### Error Handling

Use standardized exceptions from `uniinfer/errors.py`:

```python
from uniinfer.errors import (
    UniInferError, ProviderError, AuthenticationError,
    RateLimitError, TimeoutError, InvalidRequestError
)

try:
    result = some_operation()
except Exception as e:
    mapped_error = map_provider_error("provider_name", e)
    raise mapped_error
```

### Security Considerations

Credgoo handles sensitive credentials:
- Never log plaintext API keys or encryption keys
- Use restrictive file permissions (0o600) for cached credentials
- Keys are encrypted before storage using XOR + Base64
- Decrypt only when needed for API calls

### File Structure

**Credgoo:** `credgoo/credgoo.py` (main), `credgoo/__init__.py` (exports)
**UniInfer:** `uniinfer/core.py` (core classes), `uniinfer/providers/{provider_name}.py` (implementations), `uniinfer/factory.py` (factories), `uniinfer/errors.py` (error handling), `uniinfer/tests/test_*.py` (tests)

### Adding Provider Support (UniInfer)

1. Create `uniinfer/providers/newprovider.py`
2. Inherit from base provider class (`ChatProvider`, `EmbeddingProvider`, etc.)
3. Implement required methods (`complete()`, `stream_complete()`, `list_models()`)
4. Use standardized error handling with `map_provider_error()`
5. Register in `uniinfer/__init__.py` with `ProviderFactory.register_provider()`
6. Add provider dependencies to `setup.py` extras_require
7. Export in `uniinfer/providers/__init__.py`

### Version Management

When asked to update version in `setup.py`:
- **Minor bump** (default): Increment patch (0.1.5 → 0.1.6)
- **Major bump**: Increment minor, reset patch (0.1.5 → 0.2.0)

Update both `setup.py` and `pyproject.toml` if both exist.
