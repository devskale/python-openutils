import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config.instances import instance_requires_api_key
from .errors import AuthenticationError
from .provider_access import get_provider_api_key
from .proxy_services.token_registry import (
    TokenRegistryError,
    check_token_constraints,
    load_metadata_for_auth,
    metadata_path_for,
)

logger = logging.getLogger(__name__)

# Initialize the security scheme
security = HTTPBearer(auto_error=False)


# ── issued-token allowlist (hot-reloaded) ──────────────────────────────
# UNIINFER_AUTH_TOKENS_FILE: one sha256(token) hex per line, '#' comments.
# Unset/missing file → allowlist disabled (legacy behavior). Present → ONLY
# these tokens pass verify_provider_access; everything else is a fast 401 and
# feeds the auth-ban counter. Editing the file revokes/grants instantly
# (mtime reload, no restart) — the mechanism behind token rotation.
_tokens_cache: tuple[float, frozenset] | None = None  # (mtime, hashes)
_token_metadata_cache: tuple[str, float, dict] | None = None  # (path, mtime, doc)


def _token_metadata() -> dict:
    """Load hot-reloaded token metadata, or an empty document when unusable."""
    global _token_metadata_cache
    allowlist_str = os.getenv("UNIINFER_AUTH_TOKENS_FILE", "").strip()
    if not allowlist_str:
        return {}
    path_str = os.getenv("UNIINFER_AUTH_TOKENS_META_FILE", "").strip()
    path = Path(path_str) if path_str else metadata_path_for(Path(allowlist_str))
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if (
        _token_metadata_cache is not None
        and _token_metadata_cache[0] == str(path)
        and _token_metadata_cache[1] == mtime
    ):
        return _token_metadata_cache[2]
    try:
        document = load_metadata_for_auth(path)
    except RuntimeError as e:
        logger.warning("Cannot read token metadata %s: %s", path, e)
        document = {"version": 1, "tokens": {}}
    _token_metadata_cache = (str(path), mtime, document)
    return document


def _allowed_token_hashes() -> frozenset | None:
    """Current allowlist as sha256-hex set, or None when not configured."""
    global _tokens_cache
    path_str = os.getenv("UNIINFER_AUTH_TOKENS_FILE", "").strip()
    if not path_str:
        return None
    path = Path(path_str)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _tokens_cache = None
        return None
    if _tokens_cache is not None and _tokens_cache[0] == mtime:
        return _tokens_cache[1]
    try:
        hashes = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                hashes.add(line.lower())
        _tokens_cache = (mtime, frozenset(hashes))
        logger.info("Token allowlist loaded: %d entries from %s", len(hashes), path)
        return _tokens_cache[1]
    except OSError as e:
        logger.warning("Cannot read token allowlist %s: %s", path, e)
        return None


def clear_token_caches() -> None:
    """Clear test/operator-controlled auth file caches."""
    global _tokens_cache, _token_metadata_cache
    _tokens_cache = None
    _token_metadata_cache = None


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def validate_proxy_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """Validate the bearer token provided to the proxy.

    Pass-through: the token must simply be PRESENT. It can be either a credgoo
    combined token ('bearer@encryption') or — historically — a direct provider
    key. Bare direct keys are rejected in verify_provider_access (see there);
    this layer only guards against missing credentials.

    Args:
        credentials: The HTTPBearer credentials from the request.

    Returns:
        str: The validated token string.

    Raises:
        HTTPException: 401 if authentication is missing.
    """
    if not credentials or not credentials.credentials:
        logger.warning("Authentication missing in request")
        raise HTTPException(
            status_code=401,
            detail="Authentication required. Please provide a Bearer token (provider key or credgoo combo).",
        )
    return credentials.credentials


def get_optional_proxy_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[str]:
    """
    Optionally returns the bearer token if provided, without raising an error.

    Args:
        credentials: The HTTPBearer credentials from the request.

    Returns:
        Optional[str]: The token string if present, otherwise None.
    """
    if credentials and credentials.credentials:
        return credentials.credentials
    return None


def verify_provider_access(token: str, provider_name: str) -> str:
    """
    Verifies that the provided token can be used to retrieve an API key for the provider.

    Hard gate (live finding 2026-08-26): bare (non-'@') tokens are rejected for
    keyed providers. A bare string used to slip through as a "direct key" — but
    keyed providers like TU are served from the process-wide POOLED client whose
    Authorization header carries the SERVER's own credentials, so any junk
    bearer got served on our quota once the pool was warm. Consumers must
    present a credgoo combined token ('bearer@encryption'), which resolves
    server-side via credgoo.

    Args:
        token: The bearer token (direct or credgoo combo).
        provider_name: The name of the LLM provider.

    Returns:
        str: The actual provider API key.

    Raises:
        HTTPException: 401 if key retrieval fails.
    """
    try:
        # Issued-token allowlist (when configured): only tokens in the file
        # pass — rotation/revocation is an edit away, no restart.
        # A None token (no Authorization header) skips the allowlist: it is the
        # legitimate path for keyless providers (e.g. pollinations); keyed
        # providers still 401 downstream. Hashing None here crashed with a 500
        # on every keyless request once the allowlist was configured.
        allowed = _allowed_token_hashes()
        if allowed is not None and token:
            token_hash = _token_hash(token)
            if token_hash not in allowed:
                logger.warning(
                    "Rejected bearer token not on the issued list (provider='%s')",
                    provider_name,
                )
                raise AuthenticationError(
                    "Unknown or revoked token. Request a current token from the operator."
                )
            try:
                check_token_constraints(token_hash, provider_name, _token_metadata())
            except TokenRegistryError as e:
                raise AuthenticationError(str(e)) from e
        if token and "@" not in token and instance_requires_api_key(provider_name):
            logger.warning(
                "Rejected bare (non-combo) bearer token for '%s'", provider_name
            )
            raise AuthenticationError(
                "Direct provider keys are not accepted on this gateway; "
                "send a credgoo combined token (bearer@encryption)."
            )
        api_key = get_provider_api_key(token, provider_name)
        if not api_key and instance_requires_api_key(provider_name):
            raise AuthenticationError(
                f"No API key found for provider '{provider_name}'"
            )
        return api_key
    except (ValueError, AuthenticationError) as e:
        logger.error(f"Authentication failed for {provider_name}: {e}")
        raise HTTPException(status_code=401, detail=str(e))
    except Exception:
        logger.exception(f"Unexpected error during authentication for {provider_name}")
        raise HTTPException(status_code=500, detail="Internal authentication error")
