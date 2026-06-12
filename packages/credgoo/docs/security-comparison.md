# Password Manager Security Comparison

How the major password/secret managers handle encryption, key storage, and multi-device access.

## At a glance

| Tool | Encryption | Key Derivation | Master secret | Multi-device | Open source |
|------|-----------|----------------|---------------|-------------|-------------|
| **1Password** | AES-256-GCM | PBKDF2-HMAC-SHA256 (650k rounds) + Secret Key | Password + 128-bit Secret Key | Sync via cloud | No |
| **Bitwarden** | AES-256-CBC + HMAC | PBKDF2-SHA256 (600k) or Argon2id | Master password | Sync via cloud | Yes |
| **KeePassXC** | AES-256-CBC + HMAC-SHA256 | Argon2id | Master password (or key file) | Manual (file sync) | Yes |
| **iCloud Keychain** | AES-256-GCM (per-row) | Secure Enclave + device passcode | Device passcode + biometrics | Apple devices only | No |
| **Proton Pass** | AES-256 ( libsodium) | Argon2id + SRP | Master password | Sync via cloud | Yes |
| **credgoo** | XOR + Base64 | SHA-256 (fast) | PAT (stored on disk) | None (per-machine) | Yes |

## How each one works

### 1Password

```
Master Password (in your head)
        +
Secret Key (128-bit, stored on first device only)
        │
        ▼
  PBKDF2-HMAC-SHA256 (650,000 iterations)
        │
        ▼
  HKDF → Stretched Master Key (512-bit)
        │
        ▼
  Decrypts Protected Symmetric Key
        │
        ▼
  AES-256-GCM encrypts/decrypts vault
```

- **Two secrets**: password (memorized) + Secret Key (on device)
- Secret Key never sent to server. Even 1Password can't decrypt your data.
- **Account recovery**: only via team admin (enterprise) or emergency contact
- **Forget password = gone forever** (no recovery for individuals)
- Biometric unlock: stores derived key in OS keychain, not the password

### Bitwarden

```
Master Password + email (salt)
        │
        ▼
  PBKDF2-SHA256 (600,000 iterations) or Argon2id
        │
        ▼
  Master Key (256-bit)
        │
        ▼
  HKDF → Stretched Master Key
        │
        ├──→ Master Password Hash (sent to server for auth, NOT the key)
        │
        └──→ Decrypts Generated Symmetric Key
              │
              ▼
           AES-256-CBC + HMAC-SHA256 encrypts vault
```

- **One secret**: master password
- Server gets a hash, not the key. Zero-knowledge.
- Open source — anyone can audit
- Self-hostable (Vaultwarden for lightweight self-hosting)
- **Forget password = gone forever** (unless org admin recovers)
- Supports passkeys, SSO, YubiKey for 2FA

### KeePassXC

```
Master Password (or key file, or both)
        │
        ▼
  Argon2id (memory-hard, configurable rounds + memory)
        │
        ▼
  AES-256-CBC + HMAC-SHA256 encrypts the .kdbx database file
```

- **Offline only** — no cloud, no server, no sync
- Database is a single encrypted file (.kdbx)
- Key file option: a file acts as second factor (like SSH key)
- Multi-device: you sync the .kdbx file yourself (Nextcloud, Syncthing, USB)
- **Forget password = gone forever**
- No network attack surface — the database never leaves your machine
- Open source, audited

### iCloud Keychain

```
Device passcode / Touch ID / Face ID
        │
        ▼
  Secure Enclave (hardware chip)
        │
        ├──→ Table key (metadata, cached for fast search)
        └──→ Per-row key (secret value, always goes through Secure Enclave)
              │
              ▼
           AES-256-GCM per keychain item
```

- **No master password** — uses device passcode + biometrics
- Hardware-backed: keys live in Secure Enclave, never in software
- Per-item encryption: each password has its own key
- Synced via iCloud (end-to-end encrypted)
- **Apple devices only** (Mac, iPhone, iPad, Watch, Vision Pro)
- Security tied to: device passcode strength + Find My (remote wipe)
- If passcode removed, `WhenPasscodeSet` items are destroyed

### Proton Pass

```
Master Password
        │
        ▼
  Argon2id + SRP (Secure Remote Password)
        │
        ▼
  Encrypts vault keys
        │
        ▼
  AES-256 via libsodium
```

- Same model as Bitwarden but with Proton's privacy infrastructure
- SRP protocol: password never sent to server, even during login
- Open source, audited
- **Forget password = gone forever**
- Built by Proton (same team as Proton Mail)

## Key concepts explained

### Key Derivation Functions (KDF)

| KDF | Speed | Purpose | Used by |
|-----|-------|---------|---------|
| **PBKDF2** | Configurable (100k-650k iterations) | Stretch password into key, slow down brute force | 1Password, Bitwarden |
| **Argon2id** | Memory-hard (uses 64MB+ RAM) | Resistant to GPU/ASIC attacks | KeePassXC, Bitwarden (optional), Proton Pass |
| **SHA-256** | Fast (single pass) | NOT a KDF — designed for speed, not security | credgoo |
| **HKDF** | Fast | Derive multiple keys from one secret | 1Password, Bitwarden |

**Why speed matters**: A slow KDF means an attacker with your encrypted vault can only try ~100 passwords/second instead of millions. Argon2id is best because it also consumes memory, making GPU attacks impractical.

### Encryption algorithms

| Algorithm | Type | Authenticated? | Used by |
|-----------|------|---------------|---------|
| **AES-256-GCM** | Symmetric | Yes (built-in) | 1Password, iCloud Keychain |
| **AES-256-CBC + HMAC** | Symmetric | Yes (separate HMAC) | Bitwarden, KeePassXC |
| **XOR + Base64** | Obfuscation | No | credgoo |

Authenticated encryption (GCM or CBC+HMAC) detects tampering. If one bit flips in the ciphertext, decryption fails. Our XOR approach silently produces garbage.

### Multi-device sync

| Approach | How | Security tradeoff |
|----------|-----|-------------------|
| **Cloud sync** (1Password, Bitwarden) | Encrypted vault uploaded, decrypted locally | Server never sees plaintext. But server is an attack surface. |
| **iCloud sync** (Apple) | E2E encrypted via Secure Enclave | Apple devices only. Most seamless. |
| **File sync** (KeePassXC) | You sync the .kdbx file yourself | No server attack surface. But you must manage sync and conflicts. |
| **No sync** (credgoo) | Per-machine setup | Simplest. But each machine needs setup. |

## What this means for credgoo

### What we could learn from them

| From | Lesson | Applicable? |
|------|--------|------------|
| 1Password | Two secrets (password + device key) | Overkill for our use case |
| Bitwarden | Open source, self-hostable | Already open source. Self-hosting = Airtable |
| KeePassXC | Offline-first, single encrypted file | Could work — but we want cloud access |
| iCloud Keychain | Hardware-backed (Secure Enclave) | Could store PAT in Keychain on Mac |
| All of them | **Slow KDF** (PBKDF2/Argon2) | Only matters if we add a master password |

### What we should NOT do

- **Master password**: All password managers have the "forget = gone" problem. Our PAT is re-generatable. That's strictly better for a personal tool.
- **AES-256**: The key would be derived from PAT or stored next to data. No real gain over XOR for our threat model.
- **Cloud sync**: credgoo.txt already lives on each machine. Airtable is the sync layer.

### The one real improvement

**Store PAT in macOS Keychain** (like `gh` does).

```
Mac:    PAT in Keychain → hardware-backed → not on filesystem → cache encryption becomes real
Linux:  PAT in file 0o600 → same as today
```

This moves credgoo from "same as aws/docker" to "same as gh" on Mac. The only remaining gap vs password managers would be the XOR encryption, which doesn't matter if the key is in Keychain.

## TL;DR hierarchy

```
                    MORE SECURE
                        ▲
    iCloud Keychain    │  (hardware-backed, per-item encryption)
    1Password          │  (two secrets, AES-256-GCM, 650k PBKDF2)
    Bitwarden          │  (open source, AES-256, Argon2id)
    KeePassXC          │  (offline, AES-256, Argon2id)
    ───────────────────┼───────────────────────────────────────
    gh (macOS)         │  (Keychain-backed, but plaintext token)
    credgoo + Keychain │  (Keychain-backed PAT, XOR cache)
    ───────────────────┼───────────────────────────────────────
    credgoo (current)  │  (PAT in file, XOR cache, 0o600)
    gh (Linux)         │  (plaintext token in file)
    aws CLI            │  (plaintext secret key in file)
    docker             │  (base64 password in config)
                        ▼
                    MORE CONVENIENT
```

We're already in the "industry standard for CLI tools" tier. Keychain storage would bump us up one level on Mac.
