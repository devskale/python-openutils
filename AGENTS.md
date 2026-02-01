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

### Running Tests

UniInfer uses pytest. Credgoo has no tests yet.

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
```

### Code Formatting

Apply to package directory being modified:

```bash
cd packages/credgoo  # or packages/uniinfer
uv run black .         # Format code
uv run isort .         # Sort imports
uv run ruff check .    # Run linter
uv run ruff check . --fix    # Auto-fix issues
```

### Package Distribution

```bash
cd packages/credgoo  # or packages/uniinfer
uv run python setup.py sdist bdist_wheel
uv pip install dist/*.whl
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
- Use pytest fixtures for shared setup

## Code Style Guidelines

### General

- Follow PEP 8 style guidelines
- Write docstrings for all functions, classes, modules
- Use type hints where appropriate
- Python 3.6+ for credgoo, Python 3.7+ for uniinfer

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

**Credgoo:**
- `credgoo/credgoo.py` - Main implementation
- `credgoo/__init__.py` - Package exports

**UniInfer:**
- `uniinfer/core.py` - Core classes (ChatMessage, ChatCompletionRequest)
- `uniinfer/providers/{provider_name}.py` - Provider implementations
- `uniinfer/factory.py` - Provider factories
- `uniinfer/errors.py` - Error handling
- `uniinfer/tests/test_*.py` - Tests

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
