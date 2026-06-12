# Credgoo `v0.1.11`

> Secure API key retrieval from Google Sheets with encrypted local caching.

Keys are stored encrypted in a Google Sheet, fetched over HTTPS, and cached locally at `~/.config/api_keys/`. One function, one CLI, zero config in code.

## Install

**Into a venv:**
```bash
uv pip install -r https://skale.dev/credgoo
```

**Standalone CLI:**
```bash
uv tool install "credgoo @ git+https://github.com/devskale/python-openutils.git#subdirectory=packages/credgoo"
```

**As a dependency:**
```toml
[project]
dependencies = ["credgoo"]

[tool.uv.sources]
credgoo = { git = "https://github.com/devskale/python-openutils.git", subdirectory = "packages/credgoo" }
```

## Quick Start

```bash
credgoo --setup    # first time: enter token + encryption key + Apps Script URL
credgoo openai     # prints the API key
```

```python
from credgoo import get_api_key

api_key = get_api_key("openai")  # str or None
```

## CLI Reference

```bash
credgoo SERVICE              # print key (cached or fresh)
credgoo SERVICE --update     # force fresh fetch + update cache (after key rotation)
credgoo SERVICE --no-cache   # force fresh fetch without caching
credgoo --setup              # interactive first-time setup
credgoo --version
credgoo -v SERVICE           # verbose output
```

## How It Works

```
get_api_key("openai")
  1. Check ~/.config/api_keys/api_keys.json for cached key
  2. If cached → decrypt (XOR) with stored encryption key → return plaintext
  3. If not cached → fetch encrypted key from Google Sheets via Apps Script
  4. Decrypt → cache locally (encrypted) → return plaintext
```

Files on disk:
| File | Purpose |
|------|---------|
| `~/.config/api_keys/credgoo.txt` | Stored credentials (token, encryption key, URL) |
| `~/.config/api_keys/api_keys.json` | Cached encrypted keys per service |

Both use `0o600` (owner-only) permissions.

## Python API

The public API is a single function:

```python
from credgoo import get_api_key

api_key = get_api_key("openai")           # uses stored credentials
api_key = get_api_key("openai", no_cache=True)  # force fresh fetch
```

Returns `str` (plaintext key) or `None`. Always handle the `None` case:

```python
api_key = get_api_key("openai")
if not api_key:
    raise RuntimeError("No API key for openai. Run 'credgoo --setup' first.")
```

Full signature (overrides are rarely needed):

```python
get_api_key(
    service: str,
    bearer_token: str = None,
    encryption_key: str = None,
    api_url: str = None,
    cache_dir: str = None,         # default: ~/.config/api_keys
    no_cache: bool = False,
) -> str | None
```

## Google Apps Script Setup

See [appscript/README.md](appscript/README.md) for setting up the Google Sheet backend that credgoo talks to.

## Examples

```python
# Basic usage
from credgoo import get_api_key
key = get_api_key("openai")

# With Google Gemini
import google.generativeai as genai
from credgoo import get_api_key
genai.configure(api_key=get_api_key("gemini"))
```

## License

MIT
