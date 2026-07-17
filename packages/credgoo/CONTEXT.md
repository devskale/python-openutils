# CONTEXT — credgoo

Domain glossary for credgoo. Terms used in code, docs, and architecture
reviews. Keep this current when concepts are named or sharpened.

## Nouns

- **Service** — a named secret a caller wants, e.g. `openai`, `gemini`,
  `anthropic`. The lookup key for everything else.
- **API Key** — the secret string retrieved for a Service. Always returned
  plaintext to the caller; never logged.
- **Backend** — a pluggable source of API Keys: `gdrive` (Google Apps Script)
  or `airtable`. Mandatory: `setup`, `fetch_key`. Optional capabilities
  (`add_key`/`delete_key`/`clear_key`/`list_keys`/`dedupe_keys`) are declared on
  the ABC with a default that raises `UnsupportedOperation`; a backend
  advertises support by overriding. Ask via `backend.supports(capability)`.
  See `credgoo/backends/`.
- **Credential Store** — the configured entry point (`CredentialStore` in
  `credgoo/store.py`). Owns the credential file (format + v1/v2→v3 migration),
  backend resolution, encryption-key derivation, and the local cache. The deep
  module behind `get_api_key()`. Its interface is data-only; presentation
  (prompts, prints) lives in the CLI layer (`credgoo.py`).
- **Credential file** — `credgoo.txt` (JSON despite the extension). v3 layout:
  `{"default_backend": "airtable", "airtable": {...}, "gdrive": {...}}`.
  Older flat layouts are auto-migrated on load.
- **Cache** — `api_keys.json` in the cache dir. Holds API Keys encrypted with a
  per-backend key (gdrive: stored; airtable: derived from the PAT). Integrity is
  opt-in per backend via `cache_integrity` (airtable: on, with an HMAC tag so a
  PAT rotation invalidates stale cache). The cache itself is backend-agnostic —
  it honours an `integrity` flag, it knows no backend names.

## Verbs (store interface)

`get` · `add` · `delete` · `clear` · `list` · `dedupe` · `set_default` ·
`configured` · `supports(capability)`

## Failure mode

`UnsupportedOperation` — raised by a backend's default optional-method body
  when asked for a capability it doesn't override. The store translates it into
  its data-only contract (bool / None / empty) after logging.
