# AGENTS.md - Credgoo

**Package**: credgoo (python-openutils) · **Repo map**: [../AGENTS.md](../AGENTS.md)

Secure API key manager. Keys live in an encrypted Google Sheet, are fetched over HTTPS, and cached locally at `~/.config/api_keys/`.

## Usage — you only need these two things

### Python: get a key

```python
from credgoo import get_api_key

api_key = get_api_key("openai")
```

That's it. Returns the plaintext key as a string, or `None` if not found.
No arguments besides the service name are needed — credentials are read from the local cache file.

**Always handle `None`:**

```python
api_key = get_api_key("openai")
if not api_key:
    raise RuntimeError("No API key for openai. Run 'credgoo --setup' first.")
```

### CLI: update a cached key

```bash
credgoo SERVICE --update
```

Forces a fresh fetch from Google Sheets and updates the local cache.
Use this when a key has been rotated in the sheet and the old cached one is stale.

```bash
credgoo SERVICE --no-cache    # fetch fresh but don't update cache
credgoo SERVICE                # print cached key (or fetch if not cached)
```

### First-time setup (user does this, not you)

```bash
credgoo --setup
```

Interactive prompt for bearer token + encryption key + Apps Script URL.
After setup, all `get_api_key()` calls work automatically — no config needed in code.

## Import — always use this form

```python
from credgoo import get_api_key
```

**Never** use `from credgoo.credgoo import get_api_key` — that's an internal import path.

## How it works

```
get_api_key("openai")
  1. Check ~/.config/api_keys/api_keys.json for cached key
  2. If cached → decrypt with stored encryption key → return plaintext
  3. If not cached → fetch encrypted key from Google Sheets via Apps Script
  4. Decrypt → cache locally → return plaintext
```

Cache files use `0o600` permissions. Keys are XOR-encrypted before caching.

## Full Python API

```python
from credgoo import get_api_key       # main function — get a key by service name
```

All other functions are internal. You should only ever call `get_api_key()`.

`get_api_key()` signature for reference:
```python
get_api_key(
    service: str,                  # e.g. "openai", "groq", "gemini"
    bearer_token: str = None,      # override stored token (rarely needed)
    encryption_key: str = None,    # override stored key (rarely needed)
    api_url: str = None,           # override Apps Script URL (rarely needed)
    cache_dir: str = None,         # override cache dir (default: ~/.config/api_keys)
    no_cache: bool = False,        # force fresh fetch
) -> str | None
```

## Common service names

`openai`, `groq`, `gemini`, `anthropic`, `mistral`, `openrouter`, `cloudflare`,
`cohere`, `huggingface`, `sambanova`, `moonshot`, `upstage`, `chutes`, `tu`

## Security rules

- **Never** log, print, or commit plaintext API keys
- **Never** hardcode credentials in source code
- Cache files use 0o600 permissions — don't change this

## Setup

```bash
cd packages/credgoo && uv sync
```
