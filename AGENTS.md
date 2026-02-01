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

### Testing

UniInfer uses pytest. Credgoo has no tests yet.

```bash
cd packages/uniinfer

# Run all tests
uv run pytest

# Run single test file
uv run pytest uniinfer/tests/test_async_functionality.py

# Run specific test with full path
uv run pytest uniinfer/tests/test_async.py::TestClass::test_method

# Run specific test function by name
uv run pytest -k "test_function_name"

# Verbose output with coverage
uv run pytest -v --cov=uniinfer --cov-report=term-mit
```

### Code Formatting

Apply to package directory being modified:

```bash
black packages/credgoo/    # Format code
isort packages/credgoo/    # Sort imports
ruff check packages/credgoo/    # Run linter
ruff check packages/credgoo/ --fix    # Auto-fix issues
```

### Package Distribution

```bash
cd packages/credgoo  # or packages/uniinfer
python setup.py sdist bdist_wheel
uv pip install dist/*.whl
```

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

Use exception chaining with context:

```python
try:
    result = some_operation()
except Exception as e:
    raise ValueError(f"Failed to process request: {e}")
```

For API errors, use descriptive messages with context.

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
4. Use standardized error handling
5. Register in `uniinfer/__init__.py`
6. Add provider dependencies to `setup.py` extras_require
7. Export in `uniinfer/providers/__init__.py`

### Version Management

When asked to update version in `setup.py`:
- **Minor bump** (default): Increment patch (0.1.5 → 0.1.6)
- **Major bump**: Increment minor, reset patch (0.1.5 → 0.2.0)

### Testing

- Use `unittest.mock` for external API calls
- Write descriptive test names
- Test edge cases and error conditions

No tests exist for credgoo yet - add tests when fixing bugs or adding features.