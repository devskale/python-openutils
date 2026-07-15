"""CredentialStore — the deep module behind credgoo's public interface.

Owns the credential file (its format and the v1/v2 -> v3 migration), backend
resolution, encryption-key derivation, and the local API-key cache. The CLI
layer in ``credgoo.py`` constructs a store per operation and delegates; it owns
presentation (prompts, prints) only.

The interface is data-only: methods return data or bool and never print, so the
store is testable through its own seam — inject a backend, assert on returned
values, never touch ``requests`` or stdout.
"""

import base64
import hashlib
import hmac as _hmac
import json
import logging
import os
import time
from pathlib import Path

from .backends import BACKENDS

logger = logging.getLogger("credgoo")


# ---- Encryption (for the local cache) ----

def encrypt_local_key(api_key, encryption_key):
    """Encrypt the API key for local caching using XOR and Base64."""
    try:
        key_bytes = encryption_key.encode('utf-8')
        key_len = len(key_bytes)
        encrypted_bytes = bytearray(
            char_code ^ key_bytes[i % key_len]
            for i, char_code in enumerate(api_key.encode('utf-8'))
        )
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    except Exception as e:
        logger.error("Local encryption error: %s", e)
        return None


def decrypt_local_key(encrypted_api_key, encryption_key):
    """Decrypt the locally cached API key using XOR."""
    try:
        encrypted_bytes = base64.b64decode(encrypted_api_key)
        key_bytes = encryption_key.encode('utf-8')
        key_len = len(key_bytes)
        decrypted_bytes = bytearray(
            byte_val ^ key_bytes[i % key_len]
            for i, byte_val in enumerate(encrypted_bytes)
        )
        return decrypted_bytes.decode('utf-8')
    except Exception as e:
        logger.error("Local decryption error: %s", e)
        return None


def _compute_hmac(plaintext, key):
    """Compute HMAC-SHA256 of plaintext using key."""
    return _hmac.new(key.encode('utf-8'), plaintext.encode('utf-8'), hashlib.sha256).hexdigest()


def _cache_key_name(backend_name, service):
    """Namespaced cache key to avoid collisions across backends."""
    return f"{backend_name}:{service}"


# ---- Local cache ----

def cache_api_key(backend_name, service, api_key, encryption_key, cache_dir, *, integrity=False):
    """Store encrypted API key in a namespaced cache file.

    *backend_name* only namespaces the entry. When *integrity* is true, store an
    HMAC tag so staleness (e.g. credential rotation) is detectable on read. The
    caller sets *integrity* from the backend's cache_integrity policy — the
    cache itself knows no backend names.
    """
    if not encryption_key:
        logger.warning("Cannot cache API key without an encryption key.")
        return

    encrypted_key_for_cache = encrypt_local_key(api_key, encryption_key)
    if not encrypted_key_for_cache:
        logger.warning("Failed to encrypt API key for caching.")
        return

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_data = {
            "service": service,
            "api_key": encrypted_key_for_cache,
            "timestamp": str(int(time.time()))
        }
        if integrity:
            cache_data["hmac"] = _compute_hmac(api_key, encryption_key)

        cache_file = cache_dir / 'api_keys.json'

        existing_cache = {}
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    existing_cache = json.load(f)
            except json.JSONDecodeError:
                existing_cache = {}

        key_name = _cache_key_name(backend_name, service)
        existing_cache[key_name] = cache_data

        with open(cache_file, 'w') as f:
            json.dump(existing_cache, f, indent=2)

        os.chmod(cache_file, 0o600)
        logger.info("API key for %s cached (encrypted)", key_name)
    except Exception as e:
        logger.warning("Failed to cache API key: %s", e)


def get_cached_api_key(backend_name, service, encryption_key, cache_dir, *, integrity=False):
    """Retrieve and decrypt API key for a specific service from cache.

    When *integrity* is true, validate the stored HMAC tag and invalidate the
    entry on mismatch (e.g. credential rotation). Set from the backend's
    cache_integrity policy.
    """
    if not encryption_key:
        logger.warning("Cannot decrypt cached key without an encryption key.")
        return None

    cache_file = cache_dir / 'api_keys.json'

    if not cache_file.exists():
        return None

    key_name = _cache_key_name(backend_name, service)

    try:
        with open(cache_file, 'r') as f:
            cache = json.load(f)

        if key_name not in cache:
            return None

        entry = cache[key_name]
        encrypted_cached_key = entry.get("api_key")
        if not encrypted_cached_key:
            logger.warning("No 'api_key' field in cache for %s.", key_name)
            return None

        decrypted_key = decrypt_local_key(encrypted_cached_key, encryption_key)
        if not decrypted_key:
            return None

        if integrity:
            stored_hmac = entry.get("hmac")
            expected_hmac = _compute_hmac(decrypted_key, encryption_key)
            if stored_hmac != expected_hmac:
                logger.info("Cache HMAC mismatch for %s (stale cache after PAT rotation?). Invalidating.", key_name)
                cache.pop(key_name, None)
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(cache, f, indent=2)
                    os.chmod(cache_file, 0o600)
                except Exception:
                    pass
                return None

        logger.info("Using cached key for %s", key_name)
        return decrypted_key

    except Exception as e:
        logger.warning("Failed to read cached API key: %s", e)

    return None


def _delete_cached_key(backend_name, service, cache_dir):
    """Remove a single entry from the cache file."""
    cache_file = cache_dir / 'api_keys.json'
    if not cache_file.exists():
        return
    try:
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        key_name = _cache_key_name(backend_name, service)
        if key_name in cache:
            del cache[key_name]
            with open(cache_file, 'w') as f:
                json.dump(cache, f, indent=2)
            os.chmod(cache_file, 0o600)
    except Exception:
        pass


# ---- Credential file (format + migration) ----

def store_credentials(creds, cred_file):
    """Store credentials dict to file."""
    try:
        cred_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cred_file, 'w') as f:
            json.dump(creds, f, indent=2)
        os.chmod(cred_file, 0o600)
        logger.info("Credentials saved to %s", cred_file)
    except Exception as e:
        logger.warning("Failed to store credentials: %s", e)


def load_credentials(cred_file):
    """Load credentials as a dict. Returns empty dict if missing.
    Auto-migrates old formats:
      - v1 (flat gdrive): {"url": "...", "token": "...", "encryption_key": "..."}
      - v2 (single backend): {"backend": "gdrive", "url": "...", ...}
      -> v3 (multi-backend): {"default_backend": "gdrive", "gdrive": {...}}
    """
    try:
        if not cred_file.exists():
            return {}
        with open(cred_file, 'r') as f:
            creds = json.load(f)

        if not isinstance(creds, dict):
            return {}

        # Already v3
        if "default_backend" in creds:
            return creds

        # v2: flat with "backend" key
        if "backend" in creds:
            backend_name = creds.pop("backend")
            migrated = {"default_backend": backend_name, backend_name: dict(creds)}
            store_credentials(migrated, cred_file)
            return migrated

        # v1: flat gdrive (has url + token, no backend key)
        if "url" in creds or "token" in creds:
            migrated = {"default_backend": "gdrive", "gdrive": dict(creds)}
            store_credentials(migrated, cred_file)
            return migrated

        return creds
    except Exception as e:
        logger.warning("Failed to load credentials: %s", e)
        return {}


def save_backend_creds(backend_name, backend_data, cache_dir):
    """Merge a single backend's creds into credgoo.txt, setting default if absent."""
    cred_file = Path(cache_dir) / 'credgoo.txt'
    existing = load_credentials(cred_file)
    existing[backend_name] = backend_data
    if "default_backend" not in existing:
        existing["default_backend"] = backend_name
    store_credentials(existing, cred_file)


# ---- Backend resolution ----

def _resolve_backend(creds, backend_name=None):
    """Return (backend_name, backend_instance, backend_creds) or (None, None, None)."""
    if not backend_name:
        backend_name = creds.get("default_backend")
    if not backend_name:
        return None, None, None

    cls = BACKENDS.get(backend_name)
    if not cls:
        logger.error("Unknown backend: %s", backend_name)
        return None, None, None

    backend_creds = creds.get(backend_name, {})
    return backend_name, cls(), backend_creds


# ---- The deep module ----

class CredentialStore:
    """The configured credential store.

    One place that knows where the credential file lives, how to migrate it,
    how to resolve a backend, and how to derive the cache encryption key.
    Constructed per operation: ``__init__`` loads the credential file, resolves
    the (default or overridden) backend, and derives the encryption key once.

    The interface is data-only — methods return data or bool and never print.
    The CLI layer is responsible for prompts and human-readable output.
    """

    def __init__(self, cache_dir, backend_name=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cred_file = self.cache_dir / 'credgoo.txt'
        self.creds = load_credentials(self.cred_file)
        self.backend_name, self.backend, self.backend_creds = _resolve_backend(self.creds, backend_name)
        self.enc_key = self.backend.cache_key(self.backend_creds) if self.backend else None

    @property
    def configured(self):
        """True if a backend was resolved from the stored credentials."""
        return self.backend is not None

    def supports(self, capability):
        """True if the resolved backend offers *capability* (e.g. 'add_key').

        Delegates to the backend's own capability predicate — the contract is
        declared on the ABC, so this is an override check, not a method probe.
        """
        if not self.backend:
            logger.error("No backend configured. Run 'credgoo --setup-backend'.")
            return False
        if not self.backend.supports(capability):
            logger.error("Backend '%s' does not support %s.", self.backend_name, capability)
            return False
        return True

    def get(self, service, no_cache=False):
        """Plaintext key for *service*, or None. Cache-aware."""
        if not self.backend:
            logger.error("No backend configured. Run 'credgoo --setup-backend'.")
            return None
        if not no_cache and self.enc_key:
            cached = get_cached_api_key(
                self.backend_name, service, self.enc_key, self.cache_dir,
                integrity=self.backend.cache_integrity)
            if cached:
                return cached
        api_key = self.backend.fetch_key(service, self.backend_creds)
        if api_key and not no_cache and self.enc_key:
            cache_api_key(self.backend_name, service, api_key, self.enc_key, self.cache_dir,
                          integrity=self.backend.cache_integrity)
        return api_key

    def add(self, service, api_key):
        """Add/update a key in the backend and prime the cache. Returns bool."""
        if not self.supports('add_key'):
            return False
        ok = self.backend.add_key(service, api_key, self.backend_creds)
        if ok and self.enc_key:
            cache_api_key(self.backend_name, service, api_key, self.enc_key, self.cache_dir,
                          integrity=self.backend.cache_integrity)
        return ok

    def delete(self, service):
        """Delete a key from the backend and drop its cache entry."""
        if not self.supports('delete_key'):
            return False
        ok = self.backend.delete_key(service, self.backend_creds)
        if ok:
            _delete_cached_key(self.backend_name, service, self.cache_dir)
        return ok

    def clear(self, service):
        """Blank a key in the backend and drop its cache entry."""
        if not self.supports('clear_key'):
            return False
        ok = self.backend.clear_key(service, self.backend_creds)
        if ok:
            _delete_cached_key(self.backend_name, service, self.cache_dir)
        return ok

    def list(self):
        """All service names in the backend. Returns list[str] (empty if unsupported)."""
        if not self.supports('list_keys'):
            return []
        return self.backend.list_keys(self.backend_creds)

    def dedupe(self):
        """Remove duplicate services, keeping the first. Returns (kept, removed)."""
        if not self.supports('dedupe_keys'):
            return 0, 0
        return self.backend.dedupe_keys(self.backend_creds)

    def set_default(self, name):
        """Set the default backend (must already be configured). Returns bool."""
        if name not in BACKENDS:
            logger.error("Unknown backend: %s", name)
            return False
        if name not in self.creds:
            logger.error("Backend '%s' not configured yet. Run 'credgoo --setup-backend'.", name)
            return False
        self.creds["default_backend"] = name
        store_credentials(self.creds, self.cred_file)
        return True
