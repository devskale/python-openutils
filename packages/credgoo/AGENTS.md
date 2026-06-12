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

```
credgoo/
├── __init__.py          # exports: get_api_key, get_api_key_from_google, decrypt_key, cache_api_key
├── __main__.py          # CLI entry point → credgoo:main
└── credgoo.py           # all logic: fetch, encrypt, decrypt, cache, CLI, interactive setup
```

Flow: cache check → fetch from Apps Script → decrypt (XOR + Base64) → cache encrypted locally.

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
