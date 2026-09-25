"""Operator-side registry for issued gateway tokens.

The allowlist remains the hot-reload source of truth for *whether* a bearer
token is valid. This module owns the optional metadata sidecar: stable names,
expiry, and optional provider scopes. It never stores plaintext tokens.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DEFAULT_ALLOWLIST = Path.home() / ".config" / "uniinfer" / "auth_tokens.allow"
DEFAULT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_TTL_PATTERN = re.compile(r"^(\d+)([smhdw])$")
_TTL_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


class TokenRegistryError(RuntimeError):
    """Raised for invalid or conflicting token-registry operations."""


@dataclass(frozen=True)
class TokenPaths:
    """Resolved allowlist and metadata paths."""

    allowlist: Path
    metadata: Path


@dataclass(frozen=True)
class TokenRecord:
    """Metadata record for one issued token."""

    sha256: str
    name: str
    created_at: str
    expires_at: str | None
    providers: list[str] | None
    allowlisted: bool = True


def metadata_path_for(allowlist: Path) -> Path:
    """Default metadata sidecar path next to *allowlist*."""
    return allowlist.with_name(allowlist.stem + ".meta.json")


def resolve_paths(
    allowlist: str | Path | None = None,
    metadata: str | Path | None = None,
) -> TokenPaths:
    """Resolve paths from explicit values or the gateway environment."""
    allow = Path(
        allowlist or os.getenv("UNIINFER_AUTH_TOKENS_FILE") or DEFAULT_ALLOWLIST
    )
    meta = Path(
        metadata
        or os.getenv("UNIINFER_AUTH_TOKENS_META_FILE")
        or metadata_path_for(allow)
    )
    return TokenPaths(allowlist=allow, metadata=meta)


def parse_ttl(value: str) -> timedelta | None:
    """Parse an operator TTL (``30d``, ``12h``, ``1w``) or ``never``."""
    normalized = value.strip().lower()
    if normalized in {"never", "none", "-"}:
        return None
    match = _TTL_PATTERN.match(normalized)
    if not match:
        raise TokenRegistryError(
            "invalid TTL; use <number>s|m|h|d|w (for example 30d) or never"
        )
    return timedelta(seconds=int(match.group(1)) * _TTL_UNITS[match.group(2)])


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, content: str) -> None:
    """Atomically write a 0600 registry file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _load_metadata(path: Path) -> dict[str, Any]:
    """Load the metadata document, or an empty document when absent."""
    if not path.exists():
        return {"version": 1, "tokens": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenRegistryError(f"cannot read token metadata {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("tokens"), dict):
        raise TokenRegistryError(f"invalid token metadata document: {path}")
    return raw


def _save_metadata(path: Path, document: dict[str, Any]) -> None:
    _atomic_write(
        path,
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def read_allowlist(path: Path) -> set[str]:
    """Read lower-case SHA-256 entries, ignoring comments and blanks."""
    if not path.exists():
        return set()
    hashes: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip().lower()
        if entry:
            hashes.add(entry)
    return hashes


def _write_allowlist(path: Path, hashes: set[str]) -> None:
    content = "# uniinfer issued tokens — sha256(token) per line; hot-reloaded\n"
    content += "".join(f"{entry}\n" for entry in sorted(hashes))
    _atomic_write(path, content)


def _records(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return document.setdefault("tokens", {})


def _normalize_providers(providers: list[str] | None) -> list[str] | None:
    if providers is None:
        return None
    normalized: list[str] = []
    for value in providers:
        for provider in value.split(","):
            provider = provider.strip()
            if provider and provider not in normalized:
                normalized.append(provider)
    if not normalized:
        raise TokenRegistryError("provider scope cannot be empty")
    return normalized


def mint_token(
    name: str,
    ttl: str = "30d",
    providers: list[str] | None = None,
    *,
    allowlist: str | Path | None = None,
    metadata: str | Path | None = None,
    now: datetime | None = None,
) -> tuple[str, TokenRecord]:
    """Mint a new gateway token and add its hash to the allowlist.

    Returns the plaintext token exactly once. The metadata file contains only
    the SHA-256 hash and operator metadata.
    """
    if not DEFAULT_NAME_PATTERN.fullmatch(name):
        raise TokenRegistryError(
            "invalid token name; use 1-64 characters from A-Z a-z 0-9 _ . -"
        )
    paths = resolve_paths(allowlist, metadata)
    document = _load_metadata(paths.metadata)
    records = _records(document)
    if any(record.get("name") == name for record in records.values()):
        raise TokenRegistryError(f"active token name already exists: {name}")

    providers_normalized = _normalize_providers(providers)
    delta = parse_ttl(ttl)
    issued_at = now or _utc_now()
    expires_at = None if delta is None else _isoformat(issued_at + delta)
    existing_hashes = read_allowlist(paths.allowlist)

    while True:
        token = f"u{secrets.token_hex(16)}@{secrets.token_hex(16)}"
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        if token_hash not in records and token_hash not in existing_hashes:
            break

    record = {
        "name": name,
        "created_at": _isoformat(issued_at),
        "expires_at": expires_at,
        "providers": providers_normalized,
    }
    records[token_hash] = record
    _save_metadata(paths.metadata, document)

    try:
        _write_allowlist(paths.allowlist, existing_hashes | {token_hash})
    except Exception:
        document["tokens"].pop(token_hash, None)
        _save_metadata(paths.metadata, document)
        raise

    details = TokenRecord(
        sha256=token_hash,
        name=name,
        created_at=record["created_at"],
        expires_at=expires_at,
        providers=providers_normalized,
    )
    return token, details


def list_tokens(
    *,
    allowlist: str | Path | None = None,
    metadata: str | Path | None = None,
) -> list[TokenRecord]:
    """List registry records. ``allowlisted=False`` is reported by callers."""
    paths = resolve_paths(allowlist, metadata)
    document = _load_metadata(paths.metadata)
    allowlisted = read_allowlist(paths.allowlist)
    records: list[TokenRecord] = []
    for token_hash, record in sorted(_records(document).items()):
        records.append(
            TokenRecord(
                sha256=token_hash,
                name=str(record.get("name", "")),
                created_at=str(record.get("created_at", "")),
                expires_at=record.get("expires_at"),
                providers=record.get("providers"),
                allowlisted=token_hash in allowlisted,
            )
        )
    return records


def revoke_token(
    *,
    name: str | None = None,
    token_hash: str | None = None,
    allowlist: str | Path | None = None,
    metadata: str | Path | None = None,
) -> list[TokenRecord]:
    """Revoke matching active tokens by unique name and/or exact SHA-256."""
    if not name and not token_hash:
        raise TokenRegistryError("revoke requires --name or --hash")
    paths = resolve_paths(allowlist, metadata)
    document = _load_metadata(paths.metadata)
    records = _records(document)
    normalized_hash = token_hash.lower() if token_hash else None
    matches = [
        (key, value)
        for key, value in records.items()
        if (name is None or value.get("name") == name)
        and (normalized_hash is None or key == normalized_hash)
    ]
    if not matches:
        raise TokenRegistryError("no matching active token")
    if name and token_hash and len(matches) > 1:
        # This cannot happen for an exact map key, but keeps future schema drift safe.
        raise TokenRegistryError("hash did not identify a unique token")

    revoked: list[TokenRecord] = []
    for key, value in matches:
        revoked.append(
            TokenRecord(
                sha256=key,
                name=str(value.get("name", "")),
                created_at=str(value.get("created_at", "")),
                expires_at=value.get("expires_at"),
                providers=value.get("providers"),
            )
        )
        records.pop(key, None)

    allowlisted = read_allowlist(paths.allowlist)
    remaining = allowlisted - {record.sha256 for record in revoked}
    # Revoke at the gateway first (allowlist), then clean metadata.
    _write_allowlist(paths.allowlist, remaining)
    _save_metadata(paths.metadata, document)
    return revoked


def prune_expired(
    *,
    allowlist: str | Path | None = None,
    metadata: str | Path | None = None,
    now: datetime | None = None,
) -> list[TokenRecord]:
    """Remove expired hashes from the allowlist and metadata."""
    paths = resolve_paths(allowlist, metadata)
    document = _load_metadata(paths.metadata)
    records = _records(document)
    current = now or _utc_now()
    expired: list[TokenRecord] = []
    for key, value in list(records.items()):
        expires_at = value.get("expires_at")
        if not expires_at:
            continue
        try:
            deadline = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        except ValueError:
            continue
        if deadline <= current:
            expired.append(
                TokenRecord(
                    sha256=key,
                    name=str(value.get("name", "")),
                    created_at=str(value.get("created_at", "")),
                    expires_at=str(expires_at),
                    providers=value.get("providers"),
                )
            )
            records.pop(key, None)
    if expired:
        allowlisted = read_allowlist(paths.allowlist)
        _write_allowlist(
            paths.allowlist, allowlisted - {record.sha256 for record in expired}
        )
        _save_metadata(paths.metadata, document)
    return expired


def check_token_constraints(
    token_hash: str, provider_name: str, metadata: dict[str, Any]
) -> None:
    """Enforce optional metadata constraints for an already allowlisted token.

    Missing metadata means a legacy unconstrained token. A marked metadata load
    failure fails closed for that bearer token without affecting keyless routes.
    """
    if metadata.get("error"):
        raise TokenRegistryError(
            "Token metadata unavailable; token rejected until the operator repairs it."
        )
    record = metadata.get("tokens", {}).get(token_hash)
    if not record:
        return
    expires_at = record.get("expires_at")
    if expires_at:
        deadline = datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        if datetime.now(timezone.utc) >= deadline:
            raise TokenRegistryError(
                "Token expired. Request a current token from the operator."
            )
    providers = record.get("providers")
    if providers is not None and provider_name not in providers:
        raise TokenRegistryError(
            "Token is not authorized for this provider. Request a scoped token."
        )


def load_metadata_for_auth(path: Path) -> dict[str, Any]:
    """Load metadata for the hot request path and mark load failures.

    The allowlist remains authoritative for possession. Marking (rather than
    raising) lets auth fail closed only for bearer-token constraint checks while
    keyless routes continue operating.
    """
    if not path.exists():
        return {"version": 1, "tokens": {}}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"version": 1, "tokens": {}, "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(raw, dict) or not isinstance(raw.get("tokens"), dict):
        return {
            "version": 1,
            "tokens": {},
            "error": f"invalid metadata document: {path}",
        }
    return raw
