# AGENTS.md - Credgoo

This file provides guidelines for agentic coding agents operating in the credgoo package.

## Package Overview

Credgoo is a secure credential manager for retrieving API keys from Google Sheets with local caching. It provides both a CLI tool and Python API for accessing credentials securely across environments.

## Environment Setup

```bash
# Initialize with uv (creates venv, installs deps)
uv venv
uv pip install -e .

# Or with pip
pip install -e .

# Activate environment
source .venv/bin/activate
```

## Build/Lint/Test Commands

### Testing

No tests exist yet - add tests when fixing bugs or adding features.

```bash
# Run pytest (when tests are added)
pytest

# Run specific test file
pytest tests/test_credgoo.py

# Run specific test with full path
pytest tests/test_credgoo.py::TestClass::test_method

# Run specific test function by name
pytest -k "test_function_name"

# Verbose output with coverage
pytest -v --cov=credgoo --cov-report=term-mit
```

### Code Formatting

Apply to package directory:

```bash
# Format code
black credgoo/

# Sort imports
isort credgoo/

# Run linter
ruff check credgoo/
ruff check credgoo/ --fix  # Auto-fix issues

# Run all validations together
black credgoo/ && isort credgoo/ && ruff check credgoo/
```

### Package Distribution

```bash
# Build package
python setup.py sdist bdist_wheel

# Install from wheel
pip install dist/*.whl
```

## Code Style Guidelines

### General

- Follow PEP 8 style guidelines
- Write docstrings for all functions
- Use type hints where appropriate (Python 3.6+)
- Security-first approach when handling credentials

### Imports

Group imports: stdlib, third-party, local. Use absolute imports.

```python
import requests
import base64
import json
import os
import sys
import time
from pathlib import Path
from importlib.metadata import version

# Current package imports use relative
from .credgoo import decrypt_key, get_api_key
```

Run `isort credgoo/` before committing.

### Naming Conventions

- **Classes**: PascalCase (e.g., `ChatProvider` - though credgoo is function-oriented)
- **Functions/variables**: snake_case (`get_api_key()`, `api_key`, `cache_dir`)
- **Constants**: UPPER_SNAKE_CASE - not currently used but should be for magic values
- **Private methods/functions**: leading underscore (e.g., `_normalize()`)

### Type Hints

```python
def get_api_key(
    service: str,
    bearer_token: str = None,
    encryption_key: str = None,
    api_url: str = None,
    cache_dir = None,
    no_cache: bool = False
):

from typing import Optional
from pathlib import Path

def decrypt_local_key(encrypted_api_key: str, encryption_key: str) -> Optional[str]:
```

### Error Handling

Use exception chaining with context - but never log plaintext keys:

```python
try:
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    if data.get("status") == "success":
        encrypted_key = data.get("encryptedKey")
        if encrypted_key:
            api_key = decrypt_key(encrypted_key, encryption_key)
            return api_key
except Exception as e:
    print(f"Error processing request: {e}")
    return None
```

### Security Requirements

CRITICAL - Credgoo handles sensitive credentials:

1. **Never log plaintext** API keys or encryption keys
2. **Use restrictive file permissions** (0o600) for all credential files
3. **Encrypt before storage** - keys are encrypted (XOR + Base64) before caching
4. **Decrypt only in memory** - decrypt keys only when needed for API calls
5. **No secrets in code** - never commit hardcoded tokens or keys
6. **Validate inputs** - validate URLs and paths before use
7. **Handle errors gracefully** - return None on errors, don't expose internal state

### File Structure

```
credgoo/
├── credgoo/
│   ├── __init__.py       # Package exports
│   └── credgoo.py        # Main implementation (all functions)
├── example/              # Usage examples
├── appscript/            # Google Apps Script code
├── setup.py              # Package configuration
├── README.md             # User documentation
└── AGENTS.md             # This file
```

### Adding Features

When adding features:

1. Update `credgoo/__init__.py` to export new public functions
2. Update version in `setup.py` (patch bump: 0.1.5 → 0.1.6)
3. Add tests in `tests/` directory
4. Update docstrings and type hints
5. Ensure security requirements are met

### Version Management

When asked to update version in `setup.py`:
- **Minor bump** (default): Increment patch (0.1.5 → 0.1.6)
- **Major bump**: Increment minor, reset patch (0.1.5 → 0.2.0)

### Testing Strategy

- Use `unittest.mock` for external HTTP requests (Google Sheets API)
- Test encryption/decryption cycles with various inputs
- Test file permission handling (0o600)
- Test cache behavior (cache hit, miss, update)
- Test error conditions (invalid keys, network failures)
- Test with various credential configurations

### Common Patterns

**File operations with security**:
```python
try:
    cache_file = cache_dir / 'api_keys.json'
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(data, f, indent=2)
    os.chmod(cache_file, 0o600)  # Critical: owner-only
except Exception as e:
    print(f"Warning: Failed to write cache: {e}")
    return None  # Don't expose error details
```

**Accessing optional config**:
```python
stored_token, stored_key, stored_url = load_credentials(cred_file)
final_token = bearer_token if bearer_token else stored_token
```

**Cache-first approach**:
```python
if not no_cache:
    cached_key = get_cached_api_key(service, key, cache_dir)
    if cached_key:
        return cached_key
api_key = fetch_from_source(service, key)
if api_key and not no_cache:
    cache_api_key(service, api_key, key, cache_dir)
return api_key
```