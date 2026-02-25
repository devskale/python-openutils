# AGENTS.md - Python Open Utils

Monorepo for Python utility packages.

## Packages

- `packages/credgoo/` - Secure API key retrieval from Google Sheets with local caching
- `packages/uniinfer/` - Unified LLM inference interface across 20+ providers

## Environment Setup

```bash
# Initialize all packages (creates venvs, installs deps)
./uvinit.sh -x

# Or initialize specific package
./uvinit.sh -x credgoo    # or uniinfer
```

## Package-Specific Docs

Detailed guidelines for each package are in their respective AGENTS.md:

- **UniInfer**: `packages/uniinfer/AGENTS.md` - provider implementation patterns, API compatibility, async/sync patterns
- **Credgoo**: `packages/credgoo/AGENTS.md` - security requirements, encryption, file permissions

## Quick Commands

### UniInfer
```bash
cd packages/uniinfer
uv run pytest                                    # All tests
uv run pytest uniinfer/tests/test_auth.py        # Single file
uv run pytest -k "test_name"                     # By keyword
uv run pytest path/to/test.py::Class::method    # Single test
uv run black . && uv run ruff check .            # Format + lint
```

### Credgoo
```bash
cd packages/credgoo
pytest                              # All tests
pytest tests/test_credgoo.py         # Single file
pytest -k "test_name"                # By keyword
black credgoo/ && ruff check credgoo/
```

## Shared Code Style

- **PEP 8** - Follow Python style guidelines
- **No comments** unless explicitly requested
- **Docstrings** required for all functions/classes/modules
- **Type hints** where appropriate
- **Imports**: stdlib → third-party → local (use `isort`)
- **Naming**: PascalCase (classes), snake_case (functions/vars), UPPER_SNAKE (constants)

### Imports Example
```python
import json
import requests
from pathlib import Path
from typing import Dict, List, Optional
from uniinfer.core import ChatProvider
```

## Security (Credgoo)

- Never log plaintext API keys or encryption keys
- Use restrictive file permissions (0o600) for cached credentials
- Encrypt keys before storage (XOR + Base64)
- Decrypt only when needed for API calls

## Version Updates

When asked to update version in `setup.py`:
- **Minor bump** (default): Increment patch (0.1.5 → 0.1.6)
- **Major bump**: Increment minor, reset patch (0.1.5 → 0.2.0)

Update both `setup.py` and `pyproject.toml` exist.
 if both