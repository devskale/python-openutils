# Credgoo

Secure API key manager backed by Google Sheets.

Keys are stored encrypted in a Google Sheet, fetched over HTTPS, and cached locally at `~/.config/api_keys/`.

## Install

```bash
uv add credgoo
```

## Python usage

```python
from credgoo import get_api_key

api_key = get_api_key("openai")  # returns str or None
```

That's the entire API. Call `get_api_key()` with a service name, get the key back.

## CLI usage

```bash
credgoo openai              # print the API key (cached or fresh)
credgoo openai --update     # force fresh fetch + update cache (after key rotation)
credgoo openai --no-cache   # force fresh fetch without updating cache
credgoo --setup             # interactive first-time setup
credgoo --version           # show version
```

## How it works

1. First call fetches the encrypted key from your Google Sheet via Apps Script
2. Decrypts it in memory and caches it locally (`~/.config/api_keys/api_keys.json`)
3. Subsequent calls return from cache — no network needed

## First-time setup

```bash
credgoo --setup
```

Interactive prompt for:
- **Bearer token** — authenticates with your Google Apps Script
- **Encryption key** — decrypts the stored keys
- **Apps Script URL** — your Google Sheet endpoint

After setup, all `get_api_key()` calls work automatically. No config in code needed.

## Security

- Keys are encrypted (XOR + Base64) before caching
- Cache files use `0o600` (owner-only) permissions
- Credentials stored at `~/.config/api_keys/credgoo.txt` (also `0o600`)
- Never log or print plaintext keys

## Google Apps Script setup

See [appscript/README.md](appscript/README.md) for setting up the Google Sheet backend.

## License

MIT
