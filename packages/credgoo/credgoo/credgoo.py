"""credgoo — retrieve API keys from a pluggable backend.

This module is the CLI and public-adapter layer. The deep module behind it is
:class:`credgoo.store.CredentialStore`, which owns the credential file, backend
resolution, and the local cache. Functions here either adapt the store onto the
public :func:`get_api_key` contract or add presentation (prompts, prints) for
the command-line interface.
"""

import argparse
import logging
import sys
from pathlib import Path
from importlib.metadata import version

from .store import (
    CredentialStore,
    load_credentials,
    save_backend_creds,
    store_credentials,
)

logger = logging.getLogger("credgoo")


def _resolve_cache_dir(cache_dir):
    """Default the cache dir to ~/.config/api_keys unless overridden."""
    return Path(cache_dir) if cache_dir else (Path.home() / '.config' / 'api_keys')


# ---- Main public API ----

def get_api_key(service, cache_dir=None, no_cache=False, backend_name=None,
                bearer_token=None, encryption_key=None, api_url=None):
    """
    Get API key for a service. Auto-detects backend from stored credentials.
    Use *backend_name* to override the default backend for this call.
    Returns plaintext key (str) or None.

    Legacy params (*bearer_token*, *encryption_key*, *api_url*) are accepted
    for backward compatibility but ignored — credentials are resolved from
    the configured backend.
    """
    if any(v is not None for v in (bearer_token, encryption_key, api_url)):
        logger.debug(
            "get_api_key() called with legacy params (bearer_token, encryption_key, api_url). "
            "These are ignored; credentials are resolved from the configured backend.")
    store = CredentialStore(_resolve_cache_dir(cache_dir), backend_name)
    return store.get(service, no_cache=no_cache)


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
    backend._store_creds = lambda data: save_backend_creds(chosen, data, cache_dir)
    backend._generate_encryption_key = _generate_encryption_key
    backend._cred_file = cred_file

    ok = backend.setup(cache_dir)
    if ok:
        # Reload to pick up any creds saved by backend.setup() via _store_creds
        existing = load_credentials(cred_file)
        if "default_backend" not in existing:
            existing["default_backend"] = chosen
            store_credentials(existing, cred_file)
    return ok


# ---- CLI adapters: presentation (prompt + print) + delegation to CredentialStore ----

def set_default_backend(backend_name, cache_dir):
    """Change the default backend."""
    store = CredentialStore(cache_dir)
    return store.set_default(backend_name)


def add_key(service, api_key, cache_dir=None, backend_name=None):
    """Add or update an API key in the active backend."""
    store = CredentialStore(_resolve_cache_dir(cache_dir), backend_name)
    if not store.supports('add_key'):
        return False
    if not _confirm(f"Add key for '{service}' to {store.backend_name}?"):
        print("Aborted.", file=sys.stderr)
        return False
    ok = store.add(service, api_key)
    if ok:
        print(f"credgoo: Key for '{service}' saved to {store.backend_name}.", file=sys.stderr)
    else:
        print(f"credgoo: Failed to save key for '{service}'.", file=sys.stderr)
    return ok


def list_keys(cache_dir=None, backend_name=None):
    """List all service names from the active backend."""
    store = CredentialStore(_resolve_cache_dir(cache_dir), backend_name)
    if not store.supports('list_keys'):
        return False
    services = store.list()
    for svc in services:
        print(svc)
    print(f"\nTotal: {len(services)} keys ({store.backend_name})", file=sys.stderr)
    return True


def dedupe_keys(cache_dir=None, backend_name=None):
    """Remove duplicate service entries from the active backend."""
    store = CredentialStore(_resolve_cache_dir(cache_dir), backend_name)
    if not store.supports('dedupe_keys'):
        return False
    if not _confirm(f"Remove duplicate services from {store.backend_name}?"):
        print("Aborted.", file=sys.stderr)
        return False
    kept, removed = store.dedupe()
    print(f"credgoo: {kept} unique, {removed} duplicates removed ({store.backend_name})", file=sys.stderr)
    return True


def delete_key(service, cache_dir=None, backend_name=None):
    """Delete a key from the active backend and clear its cache."""
    store = CredentialStore(_resolve_cache_dir(cache_dir), backend_name)
    if not store.supports('delete_key'):
        return False
    if not _confirm(f"Delete key for '{service}' from {store.backend_name}?"):
        print("Aborted.", file=sys.stderr)
        return False
    ok = store.delete(service)
    if ok:
        print(f"credgoo: Key for '{service}' deleted from {store.backend_name}.", file=sys.stderr)
    else:
        print(f"credgoo: Failed to delete key for '{service}'.", file=sys.stderr)
    return ok


def clear_key(service, cache_dir=None, backend_name=None):
    """Blank a key and add cleared timestamp. Removes from cache."""
    store = CredentialStore(_resolve_cache_dir(cache_dir), backend_name)
    if not store.supports('clear_key'):
        return False
    if not _confirm(f"Clear key for '{service}' in {store.backend_name}?"):
        print("Aborted.", file=sys.stderr)
        return False
    ok = store.clear(service)
    if ok:
        print(f"credgoo: Key for '{service}' cleared in {store.backend_name}.", file=sys.stderr)
    else:
        print(f"credgoo: Failed to clear key for '{service}'.", file=sys.stderr)
    return ok


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
