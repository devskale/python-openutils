# Credgoo Security Model

## What credgoo is

A personal CLI tool that retrieves API keys from cloud backends (Google Sheets, Airtable).
Single user, single machine, ~90 API keys.

## What we're protecting

API keys for services like OpenAI, Anthropic, Groq, etc.
Useful to an attacker, but all revocable from the provider's dashboard.

## Threat model

| Threat | Realistic? | Impact |
|--------|-----------|--------|
| Accidental paste in chat/commit | Yes | One key exposed |
| Casual file browsing by another user | Unlikely (single-user Mac) | All keys |
| Malware on the machine | Possible | All keys + PAT |
| Stolen laptop, powered off | Possible | Everything on disk |
| Stolen laptop, unlocked | Possible | Everything on disk + in memory |
| Targeted attack (nation state) | No | Out of scope |

## How it works

### Backends

Two backends, different security models:

**gdrive (legacy)**
- Apps Script URL + bearer token + encryption key stored in `credgoo.txt`
- Keys fetched from Google Sheets via Apps Script
- Token passed in URL query params (visible in logs)

**airtable (current)**
- Personal Access Token stored in `credgoo.txt`
- Scoped to one base ("credgoo") — read/write/schema only on that base
- Revocable instantly from https://airtable.com/create/tokens
- Encryption key derived from PAT via SHA-256 (not stored separately)
- Cache includes HMAC for integrity validation on PAT rotation

### File layout

```
~/.config/api_keys/
├── credgoo.txt     # Config + PAT, 0o600 (owner read/write only)
└── api_keys.json   # Encrypted cache, 0o600
```

### Cache encryption

- **gdrive**: XOR + Base64 with stored key (legacy, not real encryption)
- **airtable**: XOR + Base64 with key derived from PAT (`sha256(PAT)[:32]`) + HMAC validation
- Not real encryption — obfuscation that raises the bar from "trivial" to "needs effort"
- If the PAT is in the same directory as the cache, the encryption is security theater
- We keep it because: prevents casual `cat` snooping, and it's already built

### Cache invalidation on PAT rotation

```
Old PAT → sha256 → key A → cache encrypted with key A → HMAC matches ✅

PAT rotates → sha256 → key B → try decrypt cache → HMAC mismatch → auto-delete stale entry → fetch fresh from Airtable → re-cache with key B ✅
```

Self-healing. No manual intervention needed.

## Security layers

### What we have

| Layer | What it does |
|-------|-------------|
| `0o600` file permissions | Only your user can read |
| FileVault (macOS) / LUKS (Linux) | Disk encryption — powered off = encrypted |
| Screen lock | Runtime boundary while away |
| PAT scoping | Token only accesses one Airtable base |
| PAT revocation | Instant kill switch from Airtable web UI |
| Cache obfuscation | XOR + Base64 — prevents casual browsing |
| HMAC validation | Detects stale cache after PAT rotation |
| Confirm prompts | Prevents accidental `--add` / `--delete` / `--dedupe` |

### What we DON'T have

| Feature | Why not |
|---------|---------|
| Master password | Forgettable = data loss. PAT is re-generatable, master password is not. |
| AES-256 encryption | Key would be stored next to the data — theater anyway. |
| macOS Keychain storage | `gh` does this. Would be a real improvement. Needs `keyring` dep. See below. |
| Zero-knowledge architecture | Overkill for a personal tool. |

## Comparison with other tools

| Tool | Where token stored | Encryption | Forget = gone? |
|------|-------------------|------------|---------------|
| 1Password | Your head + device | AES-256-GCM | Yes |
| `gh` (macOS) | **macOS Keychain** | System-managed | No (re-generate) |
| `gh` (Linux) | Plaintext file | None | No |
| `aws` CLI | Plaintext file | None | No |
| `docker` | Base64 in config (or credential store) | None | No |
| SSH keys | File, optional passphrase | Asymmetric | Optional |
| **credgoo** | Plaintext file, derived key for cache | XOR + Base64 | No (re-generate PAT) |

## Possible improvements

### Store PAT in macOS Keychain (medium effort, high value)

Same approach as `gh`:
- `credgoo --setup-backend` → PAT goes to Keychain, not `credgoo.txt`
- `credgoo.txt` only has `base_id` and `table_name` (non-secret config)
- Cache encryption becomes real: key derived from PAT, PAT not on disk
- Fallback on Linux: store in file with `0o600` (same as today)

Requires `keyring` dependency. Graceful fallback if not available.

### Multi-device sync

Currently each machine needs `--setup-backend` independently.
Could sync `credgoo.txt` (without PAT) via dotfiles, and store PAT per-device.

### Audit log

Airtable has built-in revision history. Could add `--history <service>` to show when a key was last changed.

## Decision log

1. **XOR over AES**: We chose XOR+Base64 for cache "encryption" because the key is always accessible alongside the data. AES would be theater with the same key management. XOR is simpler and equally (in)effective.

2. **No master password**: A master password is forgettable. If you forget it, encrypted cache is gone. The PAT is re-generatable from Airtable's UI — you can never lose your keys.

3. **Derived key from PAT**: Removed the separate `encryption_key` field from airtable creds. Key is derived from PAT via SHA-256 at runtime. One less secret to manage. Self-healing on PAT rotation via HMAC validation.

4. **gdrive kept as legacy**: The old Google Sheets backend works. Don't break it. New features (HMAC, derived keys, `--list`, `--add`, `--delete`, `--dedupe`) only on airtable.

5. **Confirm prompts over passwords**: Adding `[y/N]` confirm on destructive ops prevents accidents without the risk of a forgotten password locking you out.

## CLI reference

```bash
credgoo openai                       # get key from default backend
credgoo openai --backend airtable    # get key from specific backend
credgoo --add openai sk-xxx          # add/update key (prompts confirm)
credgoo --delete openai              # delete key (prompts confirm)
credgoo --list --backend airtable    # list all service names
credgoo --dedupe --backend airtable  # remove duplicates (prompts confirm)
credgoo --set-backend airtable       # switch default backend
credgoo --setup-backend              # add/configure a backend
credgoo --update openai              # force fresh fetch
```
