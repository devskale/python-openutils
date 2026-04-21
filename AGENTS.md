# AGENTS.md - Python Open Utils

> **Meta repo**: [kontext.one](https://github.com/devskale/kontext.one) — deploy, release, cross-repo orchestration. See [GUIDE.md](https://github.com/devskale/kontext.one/blob/main/GUIDE.md).

## Repo Map

Run `kontextone map` from the metarepo to see all repos with paths and branch status.

In the **kontext.one meta setup** this repo sits alongside:

| Role | Repo | Local path |
|------|------|-------------|
| **Meta** | kontext.one | `..` |
| **klark0** | klark0 (Next.js) | `../klark0` |
| **utils** | python-utils | `../python-utils` |
| **openutils** | python-openutils | `.` (this repo) |

If `../repos.yml` and `../scripts/` exist, you're in a meta setup. Otherwise this repo is standalone.

**Typical local dev**: klark0 on localhost, utils on pi5, openutils on amd (see `../repos.yml` for servers).

---

Monorepo for Python utility packages.

## Packages

- `packages/credgoo/` - Secure API key retrieval from Google Sheets with local caching
- `packages/uniinfer/` - Unified LLM inference interface across 20+ providers

## Dependency flow

```
credgoo ←── uniinfer (local path within this repo)
  ↑              ↑
  └──────────── python-utils packages (via git URL from GitHub)
```

### Within this repo: use local paths

uniinfer depends on credgoo via `path =` source:
```toml
[tool.uv.sources]
credgoo = { path = "../credgoo", editable = true }
```

### External repos (python-utils): use git URLs

See python-utils/AGENTS.md for the full guide. Pattern:
```toml
[tool.uv.sources]
credgoo = { git = "https://github.com/devskale/python-openutils.git", subdirectory = "packages/credgoo" }
uniinfer = { git = "https://github.com/devskale/python-openutils.git", subdirectory = "packages/uniinfer" }
```

### When to bump versions

After changing credgoo or uniinfer:
1. Commit and push to `main` on python-openutils
2. In python-utils: `cd packages/THAT_PACKAGE && uv lock -U` to pick up the new commit
3. Deploy: `pushto` runs `uvinit.sh` which does `uv sync` from the pinned lockfile

## Environment Setup

```bash
# Sync all packages (creates venvs, installs deps from lockfile)
cd packages/credgoo && uv sync
cd packages/uniinfer && uv sync
```

## Package-Specific Docs

Detailed guidelines for each package are in their respective AGENTS.md:

- **UniInfer**: `packages/uniinfer/AGENTS.md` - provider implementation patterns, API compatibility, async/sync patterns
- **Credgoo**: `packages/credgoo/AGENTS.md` - security requirements, encryption, file permissions

## Quick Commands

### UniInfer
```bash
cd packages/uniinfer
uv sync                                # Install/update deps
uv run pytest                           # All tests
uv run pytest -k "test_name"            # By keyword
```

### Credgoo
```bash
cd packages/credgoo
uv sync                                # Install/update deps
uv run pytest                           # All tests
uv run pytest -k "test_name"            # By keyword
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

When asked to update version in `pyproject.toml`:
- **Minor bump** (default): Increment patch (0.1.5 → 0.1.6)
- **Major bump**: Increment minor, reset patch (0.1.5 → 0.2.0)
 if both