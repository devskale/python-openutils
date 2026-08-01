"""Provider-instances overlay: named, configurable fleet endpoints.

``load_instances()`` merges the built-in registry (the source of truth for *what
exists*, so new releases propagate drift-free) with an optional gitignored JSON
overlay carrying ``enabled`` flags + per-instance overrides + custom aliases.
``resolve_instance()`` is the one alias -> :class:`InstanceSpec` seam used by
completion (``Target``), embeddings, the models router, and the CLI — replacing
the scattered ``extra_params`` / ``provider_name == 'ollama'`` special-cases.

See ``docs/provider-instances-design.md`` for the full design.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from uniinfer.factory import ProviderFactory

logger = logging.getLogger(__name__)

DEFAULT_FILENAME = "provider_instances.json"

# Overlay entry fields that map 1:1 onto InstanceSpec attributes.
_ENTRY_FIELDS = (
    "provider",
    "base_url",
    "credgoo_service",
    "requires_api_key",
    "enabled",
    "default_model",
)


def instance_file_path() -> str:
    """Resolve the overlay path: ``$UNIINFER_INSTANCES_FILE`` else ``<cwd>/provider_instances.json``."""
    return os.environ.get("UNIINFER_INSTANCES_FILE") or os.path.join(
        os.getcwd(), DEFAULT_FILENAME
    )


@dataclass(frozen=True)
class InstanceSpec:
    """A resolved named instance — the output of :func:`resolve_instance`.

    Attributes:
        alias: The addressable name (the ``provider`` side of ``alias@model``).
        provider: The underlying factory class key. Equals ``alias`` for built-ins.
        base_url: Endpoint URL, or None to use the class default.
        credgoo_service: Credgoo key service, or None to resolve per rules.
        requires_api_key: Whether completions need an API key (anonymous = False).
        enabled: False hides the instance from listing/dispatch.
        default_model: Default model id, or None.
        is_builtin: True if the alias names a registry provider (overridable, not deletable).
    """

    alias: str
    provider: str
    base_url: Optional[str] = None
    credgoo_service: Optional[str] = None
    requires_api_key: bool = True
    enabled: bool = True
    default_model: Optional[str] = None
    is_builtin: bool = False


def _spec_from_class(alias: str, provider: str, is_builtin: bool) -> InstanceSpec:
    """Build a spec by reading the underlying provider class's identity attrs."""
    cls = ProviderFactory.get_provider_class(provider)
    return InstanceSpec(
        alias=alias,
        provider=provider,
        base_url=getattr(cls, "BASE_URL", None) or None,
        credgoo_service=getattr(cls, "CREDGOO_SERVICE", None),
        requires_api_key=bool(getattr(cls, "REQUIRES_API_KEY", True)),
        default_model=getattr(cls, "DEFAULT_MODEL", None),
        is_builtin=is_builtin,
    )


def _builtin_spec(name: str) -> InstanceSpec:
    """A built-in's spec. Lazy providers (e.g. gemini) get safe defaults without
    forcing their heavy SDK import just to enumerate the registry.

    base_url is deliberately None: a built-in uses its own class default, and
    Target only forwards a base_url when the file *overrides* it (forwarding the
    class's own BASE_URL would break providers whose __init__ rejects base_url).
    """
    if ProviderFactory.is_lazy(name):
        return InstanceSpec(alias=name, provider=name, is_builtin=True)
    return replace(_spec_from_class(name, name, is_builtin=True), base_url=None)


def _apply_entry(spec: InstanceSpec, entry: Any) -> InstanceSpec:
    """Overlay a file entry's declared fields onto a spec, keeping non-None values."""
    if not isinstance(entry, dict):
        raise ValueError(f"instance entry must be an object, got {type(entry).__name__}")
    overrides = {k: entry[k] for k in _ENTRY_FIELDS if k in entry}
    return replace(spec, **overrides)


def load_instances(path: Optional[str] = None) -> dict[str, InstanceSpec]:
    """Load + merge: registry built-ins overlaid by the JSON file.

    Args:
        path: Explicit overlay file. None resolves via :func:`instance_file_path`.

    Returns:
        ``alias -> InstanceSpec`` for every built-in (enabled by default) plus
        every custom alias declared in the file.

    Raises:
        ValueError: if the file is malformed JSON, an entry is not an object, a
            custom alias omits ``provider``, or references an unregistered
            provider. (Missing file is NOT an error — built-ins only.)
    """
    registered = set(ProviderFactory.list_providers())
    merged: dict[str, InstanceSpec] = {name: _builtin_spec(name) for name in registered}

    file_path = Path(path) if path else Path(instance_file_path())
    if not file_path.exists():
        return merged

    try:
        raw = json.loads(file_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"instances file {file_path} is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"instances file {file_path} must be a JSON object")

    for alias, entry in raw.items():
        if alias in merged:
            merged[alias] = _apply_entry(merged[alias], entry)
            continue
        provider = entry.get("provider") if isinstance(entry, dict) else None
        if not provider:
            raise ValueError(
                f"custom instance '{alias}' must declare a 'provider' (underlying class key)"
            )
        if provider not in registered:
            raise ValueError(
                f"instance '{alias}' references unknown provider '{provider}' "
                f"(not one of: {', '.join(sorted(registered))})"
            )
        spec = _apply_entry(_spec_from_class(alias, provider, is_builtin=False), entry)
        if spec.credgoo_service is None:
            spec = replace(spec, credgoo_service=alias)  # custom default: own alias
        merged[alias] = spec

    return merged


# --------------------------------------------------------------------------- #
# Hot-path cached loader (mtime-check + graceful-degrade — Q12)
# --------------------------------------------------------------------------- #
_CACHE: dict[str, tuple[float, "dict[str, InstanceSpec]"]] = {}


def clear_instances_cache() -> None:
    """Drop the cached overlay (test helper / forced refresh)."""
    _CACHE.clear()


def get_instances(path: Optional[str] = None) -> dict[str, InstanceSpec]:
    """Return the merged overlay, cached by file mtime.

    Re-reads + re-validates only when the file's mtime changes. On a reload
    failure (malformed edit mid-write) it serves the *last known-good* merge
    and warns — never takes the caller down. A failure with nothing cached
    (first boot) propagates, so misconfiguration is caught loudly at startup.
    """
    p = str(Path(path) if path else Path(instance_file_path()))
    try:
        mtime = os.path.getmtime(p) if Path(p).exists() else -1.0
    except OSError:
        mtime = -1.0

    cached = _CACHE.get(p)
    if cached and cached[0] == mtime:
        return cached[1]

    try:
        merged = load_instances(path=p)
    except ValueError:
        if cached is not None:
            logger.warning(
                "instances overlay reload failed for %s; serving last-good", p
            )
            return cached[1]
        raise  # nothing cached -> fail fast at boot

    _CACHE[p] = (mtime, merged)
    return merged


def resolve_instance(
    alias: str,
    instances: Optional[dict[str, InstanceSpec]] = None,
    path: Optional[str] = None,
) -> InstanceSpec:
    """Return the merged spec for an alias (built-in or custom).

    Args:
        alias: The instance name.
        instances: A pre-loaded merge (skips re-reading the file).
        path: Overlay path when ``instances`` is None.

    Raises:
        ValueError: if the alias is unknown or disabled.
    """
    table = instances if instances is not None else get_instances(path=path)
    spec = table.get(alias)
    if spec is None:
        raise ValueError(
            f"unknown provider instance '{alias}' "
            f"(known: {', '.join(sorted(table))})"
        )
    return spec


def instance_requires_api_key(alias: str) -> bool:
    """Whether dispatching to *alias* needs an API key.

    The single flag that replaces the hardcoded ``provider_name == 'ollama'``
    keyless special-case: any instance (local vLLM, LM Studio, localhost Ollama)
    with ``requires_api_key: false`` is auth-optional. Unknown aliases default
    to True (safe — demand a key).
    """
    try:
        return resolve_instance(alias).requires_api_key
    except Exception:
        return True
