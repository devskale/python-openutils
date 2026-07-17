# AGENTS.md — Credgoo

**Package**: credgoo · **Parent**: [../AGENTS.md](../AGENTS.md)

## Tooling

- Python 3.9+, **uv** for package management (not pip)
- Single dependency: `requests`

## Commands

```bash
cd packages/credgoo && uv sync                    # install deps
cd packages/credgoo && uv run pytest              # all tests
cd packages/credgoo && uv run pytest -k "test_name"  # single test
```

## Public API

One function. Import like this:

```python
from credgoo import get_api_key
```

🚫 **Never** use `from credgoo.credgoo import get_api_key` — that's an internal path.

`get_api_key(service)` returns `str | None`. Always handle `None`.

## Architecture

Two layers:

- **`credgoo/store.py`** — `CredentialStore`, the deep module. Owns the
  credential file (format + v1/v2→v3 migration), backend resolution,
  encryption-key derivation, and the local cache. Data-only interface (no
  printing); testable by injecting a backend.
- **`credgoo/credgoo.py`** — the public adapter `get_api_key()` plus the CLI
  (argparse `main()`) and thin CLI adapters that add prompts/prints and
  delegate to `CredentialStore`.

Backends live in `credgoo/backends/` (`base.py` ABC, `gdrive.py`, `airtable.py`).
`__init__.py` re-exports the public surface, `__main__.py` is the CLI entry.

Flow: `get_api_key()` → `CredentialStore.get()` → cache check → backend
`fetch_key()` → decrypt (XOR + Base64) → cache encrypted locally.

See [CONTEXT.md](CONTEXT.md) for the domain glossary.

## Critical Rules

- ✅ **Always** run `uv run pytest` before committing
- 🚫 **Never** log or print plaintext API keys
- 🚫 **Never** hardcode credentials in source code
- 🚫 **Never** change file permissions away from `0o600` on cache/credential files
- ⚠️ **Ask first** before adding dependencies (this package stays minimal — only `requests`)

## Known Footguns

- `decrypt_key()` strips an 8-char IV from the front — this must match the Apps Script's `robustEncrypt()` IV logic exactly
- Cache file is `api_keys.json` (JSON), credentials file is `credgoo.txt` (also JSON despite the extension) — don't confuse them
- `--setup` requires a TTY; it will fail with "non-interactive mode" in CI/pipes
- `get_api_key()` silently returns `None` if credentials are missing — callers must check

## Common Service Names

`openai`, `groq`, `gemini`, `anthropic`, `mistral`, `openrouter`, `cloudflare`,
`cohere`, `huggingface`, `sambanova`, `moonshot`, `upstage`, `chutes`, `tu`
