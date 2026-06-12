import hashlib
import hmac as _hmac
import logging
import base64
import argparse
import os
import sys
import json
import time
from pathlib import Path
from importlib.metadata import version

logger = logging.getLogger("credgoo")


# ---- Encryption (for local cache) ----

def encrypt_local_key(api_key, encryption_key):
    """Encrypt the API key for local caching using XOR and Base64."""
    try:
        encrypted_bytes = bytearray()
        key_bytes = encryption_key.encode('utf-8')
        key_len = len(key_bytes)
        for i, char_code in enumerate(api_key.encode('utf-8')):
            key_char_code = key_bytes[i % key_len]
            encrypted_bytes.append(char_code ^ key_char_code)
        return base64.b64encode(encrypted_bytes).decode('utf-8')
    except Exception as e:
        logger.error("Local encryption error: %s", e)
        return None


def decrypt_local_key(encrypted_api_key, encryption_key):
    """Decrypt the locally cached API key using XOR."""
    try:
        encrypted_bytes = base64.b64decode(encrypted_api_key)
        decrypted_bytes = bytearray()
        key_bytes = encryption_key.encode('utf-8')
        key_len = len(key_bytes)
        for i, byte_val in enumerate(encrypted_bytes):
            key_char_code = key_bytes[i % key_len]
            decrypted_bytes.append(byte_val ^ key_char_code)
        return decrypted_bytes.decode('utf-8')
    except Exception as e:
        logger.error("Local decryption error: %s", e)
        return None


# ---- Caching ----

def _cache_key_name(backend_name, service):
    """Namespaced cache key to avoid collisions across backends."""
    return f"{backend_name}:{service}"


def _compute_hmac(plaintext, key):
    """Compute HMAC-SHA256 of plaintext using key."""
    return _hmac.new(key.encode('utf-8'), plaintext.encode('utf-8'), hashlib.sha256).hexdigest()


def _derive_key_from_pat(pat):
    """Derive a 32-char encryption key from an Airtable PAT via SHA-256."""
    return hashlib.sha256(pat.encode('utf-8')).hexdigest()[:32]


def _get_enc_key(backend_name, backend_creds):
    """Get the encryption key for a backend.
    gdrive: stored encryption_key (legacy)
    airtable: derived from PAT at runtime
    """
    enc = backend_creds.get("encryption_key")
    if enc:
        return enc
    if backend_name == "airtable":
        pat = backend_creds.get("airtable_token")
        if pat:
            return _derive_key_from_pat(pat)
    return None


def _is_hmac_backend(backend_name):
    """Whether this backend uses HMAC validation (airtable) or legacy cache (gdrive)."""
    return backend_name == "airtable"


def cache_api_key(backend_name, service, api_key, encryption_key, cache_dir):
    """Store encrypted API key in service-specific cache file."""
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
        if _is_hmac_backend(backend_name):
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


def get_cached_api_key(backend_name, service, encryption_key, cache_dir):
    """Retrieve and decrypt API key for specific service from cache.
    For HMAC backends: validates integrity, invalidates stale cache on PAT rotation.
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

        if _is_hmac_backend(backend_name):
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


# ---- Credential storage ----

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
      → v3 (multi-backend): {"default_backend": "gdrive", "gdrive": {...}}
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


# ---- Backend resolution ----

def _resolve_backend(creds, backend_name=None):
    """Return (backend_name, backend_instance, backend_creds) or (None, None, None)."""
    from .backends import BACKENDS

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


# ---- Main public API ----

def get_api_key(service, cache_dir=None, no_cache=False, backend_name=None):
    """
    Get API key for a service. Auto-detects backend from stored credentials.
    Use *backend_name* to override the default backend for this call.
    Returns plaintext key (str) or None.
    """
    if cache_dir is None:
        cache_dir = Path.home() / '.config' / 'api_keys'
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cred_file = cache_dir / 'credgoo.txt'
    creds = load_credentials(cred_file)

    resolved_name, backend, backend_creds = _resolve_backend(creds, backend_name)
    if not backend:
        logger.error("No backend configured. Run 'credgoo --setup-backend'.")
        return None

    enc_key = _get_enc_key(resolved_name, backend_creds)

    if not no_cache and enc_key:
        cached_key = get_cached_api_key(resolved_name, service, enc_key, cache_dir)
        if cached_key:
            return cached_key

    api_key = backend.fetch_key(service, backend_creds)
    if api_key and not no_cache and enc_key:
        cache_api_key(resolved_name, service, api_key, enc_key, cache_dir)
    return api_key


# ---- Setup ----

def _confirm(message):
    """Prompt for confirmation. Returns True if user confirms."""
    if not sys.stdin.isatty():
        return True
    resp = input(f"{message} [y/N] ").strip().lower()
    return resp in ('y', 'yes')


def _generate_encryption_key():
    import secrets as _secrets
    import string
    return ''.join(_secrets.choice(string.ascii_letters + string.digits) for _ in range(32))


def setup_backend(cache_dir):
    """Interactive: pick a backend, run its setup. Preserves existing backends."""
    from .backends import BACKENDS, BACKEND_LABELS

    cache_dir.mkdir(parents=True, exist_ok=True)
    cred_file = cache_dir / 'credgoo.txt'
    existing = load_credentials(cred_file)

    print("\ncredgoo: Backend setup\n")

    configured = [k for k in BACKENDS if k in existing]
    if configured:
        print("Already configured: " + ", ".join(configured))
        print()

    print("Choose a backend to set up:\n")

    names = list(BACKENDS.keys())
    for i, name in enumerate(names, 1):
        label = BACKEND_LABELS.get(name, name)
        marker = " (reconfigure)" if name in existing else ""
        print(f"  {i}. {label}{marker}")
    print()

    while True:
        choice = input(f"Enter choice (1-{len(names)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            break
        print("Invalid choice.")

    chosen = names[int(choice) - 1]
    backend = BACKENDS[chosen]()

    # Inject helpers backends can use
    backend._store_creds = lambda data: _save_backend_creds(chosen, data, cache_dir)
    backend._generate_encryption_key = _generate_encryption_key
    backend._cred_file = cred_file

    ok = backend.setup(cache_dir)
    if ok and "default_backend" not in existing:
        existing["default_backend"] = chosen
        store_credentials(existing, cred_file)
    return ok


def _save_backend_creds(backend_name, backend_data, cache_dir):
    """Save a single backend's creds into the multi-backend credgoo.txt."""
    cred_file = cache_dir / 'credgoo.txt'
    existing = load_credentials(cred_file)
    existing[backend_name] = backend_data
    if "default_backend" not in existing:
        existing["default_backend"] = backend_name
    store_credentials(existing, cred_file)


def set_default_backend(backend_name, cache_dir):
    """Change the default backend."""
    from .backends import BACKENDS

    cache_dir.mkdir(parents=True, exist_ok=True)
    cred_file = cache_dir / 'credgoo.txt'
    existing = load_credentials(cred_file)

    if backend_name not in BACKENDS:
        print(f"Unknown backend: {backend_name}", file=sys.stderr)
        print(f"Available: {', '.join(BACKENDS.keys())}", file=sys.stderr)
        return False

    if backend_name not in existing:
        print(f"Backend '{backend_name}' not configured yet. Run 'credgoo --setup-backend' first.", file=sys.stderr)
        return False

    existing["default_backend"] = backend_name
    store_credentials(existing, cred_file)
    print(f"Default backend set to: {backend_name}")
    return True


def add_key(service, api_key, cache_dir=None, backend_name=None):
    """Add or update an API key in the active backend."""
    if cache_dir is None:
        cache_dir = Path.home() / '.config' / 'api_keys'
    else:
        cache_dir = Path(cache_dir)

    cred_file = cache_dir / 'credgoo.txt'
    creds = load_credentials(cred_file)

    resolved_name, backend, backend_creds = _resolve_backend(creds, backend_name)
    if not backend:
        logger.error("No backend configured. Run 'credgoo --setup-backend'.")
        return False

    if not hasattr(backend, 'add_key'):
        logger.error("Backend '%s' does not support adding keys via CLI.", resolved_name)
        return False

    if not _confirm(f"Add key for '{service}' to {resolved_name}?"):
        print("Aborted.", file=sys.stderr)
        return False

    ok = backend.add_key(service, api_key, backend_creds)
    if ok:
        enc_key = _get_enc_key(resolved_name, backend_creds)
        if enc_key:
            cache_api_key(resolved_name, service, api_key, enc_key, cache_dir)
        print(f"credgoo: Key for '{service}' saved to {resolved_name}.", file=sys.stderr)
    else:
        print(f"credgoo: Failed to save key for '{service}'.", file=sys.stderr)
    return ok


def list_keys(cache_dir=None, backend_name=None):
    """List all service names from the active backend."""
    if cache_dir is None:
        cache_dir = Path.home() / '.config' / 'api_keys'
    else:
        cache_dir = Path(cache_dir)

    cred_file = cache_dir / 'credgoo.txt'
    creds = load_credentials(cred_file)

    resolved_name, backend, backend_creds = _resolve_backend(creds, backend_name)
    if not backend:
        logger.error("No backend configured. Run 'credgoo --setup-backend'.")
        return False

    if not hasattr(backend, 'list_keys'):
        logger.error("Backend '%s' does not support listing keys.", resolved_name)
        return False

    services = backend.list_keys(backend_creds)
    for svc in services:
        print(svc)
    print(f"\nTotal: {len(services)} keys ({resolved_name})", file=sys.stderr)
    return True


def dedupe_keys(cache_dir=None, backend_name=None):
    """Remove duplicate service entries from the active backend."""
    if cache_dir is None:
        cache_dir = Path.home() / '.config' / 'api_keys'
    else:
        cache_dir = Path(cache_dir)

    cred_file = cache_dir / 'credgoo.txt'
    creds = load_credentials(cred_file)

    resolved_name, backend, backend_creds = _resolve_backend(creds, backend_name)
    if not backend:
        logger.error("No backend configured. Run 'credgoo --setup-backend'.")
        return False

    if not hasattr(backend, 'dedupe_keys'):
        logger.error("Backend '%s' does not support deduplication.", resolved_name)
        return False

    if not _confirm(f"Remove duplicate services from {resolved_name}?"):
        print("Aborted.", file=sys.stderr)
        return False

    kept, removed = backend.dedupe_keys(backend_creds)
    print(f"credgoo: {kept} unique, {removed} duplicates removed ({resolved_name})", file=sys.stderr)
    return True


def delete_key(service, cache_dir=None, backend_name=None):
    """Delete a key from the active backend and clear its cache."""
    if cache_dir is None:
        cache_dir = Path.home() / '.config' / 'api_keys'
    else:
        cache_dir = Path(cache_dir)

    cred_file = cache_dir / 'credgoo.txt'
    creds = load_credentials(cred_file)

    resolved_name, backend, backend_creds = _resolve_backend(creds, backend_name)
    if not backend:
        logger.error("No backend configured. Run 'credgoo --setup-backend'.")
        return False

    if not hasattr(backend, 'delete_key'):
        logger.error("Backend '%s' does not support deleting keys.", resolved_name)
        return False

    if not _confirm(f"Delete key for '{service}' from {resolved_name}?"):
        print("Aborted.", file=sys.stderr)
        return False

    ok = backend.delete_key(service, backend_creds)
    if ok:
        enc_key = _get_enc_key(resolved_name, backend_creds)
        if enc_key:
            _delete_cached_key(resolved_name, service, cache_dir)
        print(f"credgoo: Key for '{service}' deleted from {resolved_name}.", file=sys.stderr)
    else:
        print(f"credgoo: Failed to delete key for '{service}'.", file=sys.stderr)
    return ok


def clear_key(service, cache_dir=None, backend_name=None):
    """Blank a key and add cleared timestamp. Removes from cache."""
    if cache_dir is None:
        cache_dir = Path.home() / '.config' / 'api_keys'
    else:
        cache_dir = Path(cache_dir)

    cred_file = cache_dir / 'credgoo.txt'
    creds = load_credentials(cred_file)

    resolved_name, backend, backend_creds = _resolve_backend(creds, backend_name)
    if not backend:
        logger.error("No backend configured. Run 'credgoo --setup-backend'.")
        return False

    if not hasattr(backend, 'clear_key'):
        logger.error("Backend '%s' does not support clearing keys.", resolved_name)
        return False

    if not _confirm(f"Clear key for '{service}' in {resolved_name}?"):
        print("Aborted.", file=sys.stderr)
        return False

    ok = backend.clear_key(service, backend_creds)
    if ok:
        _delete_cached_key(resolved_name, service, cache_dir)
        print(f"credgoo: Key for '{service}' cleared in {resolved_name}.", file=sys.stderr)
    else:
        print(f"credgoo: Failed to clear key for '{service}'.", file=sys.stderr)
    return ok


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


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser(
        description="Retrieve API keys securely from your chosen backend")
    parser.add_argument(
        "service", nargs="?", help="Service name to retrieve the API key for")
    parser.add_argument("--setup-backend", action="store_true",
                        help="Interactive backend setup (choose + configure)")
    parser.add_argument("--set-backend", metavar="NAME",
                        help="Change the default backend (e.g. gdrive, airtable)")
    parser.add_argument("--backend", metavar="NAME",
                        help="Use this backend for this request only")
    parser.add_argument(
        "--cache-dir", help="Directory to store cached API keys (default: ~/.config/api_keys/)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Bypass cache and force retrieval from source")
    parser.add_argument("--update", action="store_true",
                        help="Force fresh fetch and update cache")
    parser.add_argument('--add', nargs=2, metavar=('SERVICE', 'KEY'),
                        help='Add or update a key in the active backend')
    parser.add_argument('--list', action='store_true',
                        help='List all service names in the active backend')
    parser.add_argument('--dedupe', action='store_true',
                        help='Remove duplicate service entries from the active backend')
    parser.add_argument('--clear', metavar='SERVICE',
                        help='Blank a key and add cleared timestamp')
    parser.add_argument('--delete', metavar='SERVICE',
                        help='Delete a key from the active backend')
    parser.add_argument('--version', action='version',
                        version='%(prog)s ' + version('credgoo'),
                        help="Show program's version number and exit")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format='%(message)s', stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, format='%(message)s', stream=sys.stderr)

    if not args.setup_backend and not args.service and not args.set_backend and not args.add and not args.list and not args.dedupe and not args.clear and not args.delete:
        parser.error("the following arguments are required: service")

    cache_dir = Path(args.cache_dir) if args.cache_dir else (Path.home() / '.config' / 'api_keys')

    if args.set_backend:
        return 0 if set_default_backend(args.set_backend, cache_dir) else 1

    if args.setup_backend:
        if not sys.stdin.isatty():
            print("Error: Backend setup requires a TTY.", file=sys.stderr)
            return 1
        return 0 if setup_backend(cache_dir) else 1

    if args.add:
        return 0 if add_key(args.add[0], args.add[1], cache_dir=cache_dir, backend_name=args.backend) else 1

    if args.list:
        return 0 if list_keys(cache_dir=cache_dir, backend_name=args.backend) else 1

    if args.dedupe:
        return 0 if dedupe_keys(cache_dir=cache_dir, backend_name=args.backend) else 1

    if args.clear:
        return 0 if clear_key(args.clear, cache_dir=cache_dir, backend_name=args.backend) else 1

    if args.delete:
        return 0 if delete_key(args.delete, cache_dir=cache_dir, backend_name=args.backend) else 1

    force_no_cache = args.no_cache or args.update

    current_cached_key = None
    if args.update:
        current_cached_key = get_api_key(args.service, cache_dir=cache_dir, no_cache=False, backend_name=args.backend)
        if current_cached_key:
            print(f"credgoo: Found cached key for {args.service}.", file=sys.stderr)
        else:
            print(f"credgoo: No cached key found for {args.service}.", file=sys.stderr)

    api_key = get_api_key(args.service, cache_dir=cache_dir, no_cache=force_no_cache, backend_name=args.backend)

    if args.update and api_key and current_cached_key:
        if api_key != current_cached_key:
            print(f"credgoo: Online key for {args.service} has changed. Updating cache.", file=sys.stderr)
        else:
            print(f"credgoo: Online key for {args.service} is up to date.", file=sys.stderr)
    elif args.update and api_key and not current_cached_key:
        print(f"credgoo: Fetched online key for {args.service}. No previous cache.", file=sys.stderr)
    elif args.update and not api_key:
        print(f"credgoo: Failed to fetch online key for {args.service}.", file=sys.stderr)

    if api_key:
        print(api_key)
        return 0
    else:
        print("credgoo: Failed to retrieve API key", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
