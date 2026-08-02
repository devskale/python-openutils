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
from dataclasses import dataclass, field, replace
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
    # Open, forward-adoptable access metadata: {keytype, models, budget, …}.
    # keytype = free-form label; models = optional selective spec (list today,
    # filter object later). Code reads what it knows; the rest is preserved.
    access: dict = field(default_factory=dict)


def _spec_from_class(alias: str, provider: str, is_builtin: bool) -> InstanceSpec:
    """Build a spec by reading the underlying provider class's identity attrs.

    base_url is deliberately None: passing a class's own BASE_URL back to its
    __init__/list_models breaks providers that reject base_url (mistral,
    pollinations, openrouter, …). Only a *file-declared* base_url (custom alias
    or built-in override) sets it; otherwise the provider self-defaults.
    """
    cls = ProviderFactory.get_provider_class(provider)
    return InstanceSpec(
        alias=alias,
        provider=provider,
        base_url=None,
        credgoo_service=getattr(cls, "CREDGOO_SERVICE", None),
        requires_api_key=bool(getattr(cls, "REQUIRES_API_KEY", True)),
        default_model=getattr(cls, "DEFAULT_MODEL", None),
        is_builtin=is_builtin,
    )


def _builtin_spec(name: str) -> InstanceSpec:
    """A built-in's spec. Lazy providers (e.g. gemini) get safe defaults without
    forcing their heavy SDK import just to enumerate the registry."""
    if ProviderFactory.is_lazy(name):
        return InstanceSpec(alias=name, provider=name, is_builtin=True)
    return _spec_from_class(name, name, is_builtin=True)


def _apply_entry(spec: InstanceSpec, entry: Any) -> InstanceSpec:
    """Overlay a file entry's declared fields onto a spec, keeping non-None values."""
    if not isinstance(entry, dict):
        raise ValueError(f"instance entry must be an object, got {type(entry).__name__}")
    overrides = {k: entry[k] for k in _ENTRY_FIELDS if k in entry}
    if isinstance(entry.get("access"), dict):
        overrides["access"] = entry["access"]
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


# --------------------------------------------------------------------------- #
# Overlay management primitives (add / remove / enable / reset / show)
# Pure file ops — the CLI flags and the smart add-probe are shells over these.
# --------------------------------------------------------------------------- #
def _overlay_path(path: Optional[str] = None) -> Path:
    return Path(path) if path else Path(instance_file_path())


def read_overlay(path: Optional[str] = None) -> dict[str, Any]:
    """Read the raw overlay JSON (empty dict if the file is missing)."""
    p = _overlay_path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"overlay file {p} is not valid JSON: {e}") from e
    if not isinstance(raw, dict):
        raise ValueError(f"overlay file {p} must be a JSON object")
    return raw


def write_overlay(data: dict[str, Any], path: Optional[str] = None) -> Path:
    """Write the overlay JSON (creates parent dirs). Returns the path."""
    p = _overlay_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    clear_instances_cache()
    return p


def upsert_instance(
    alias: str,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
    credgoo_service: Optional[str] = None,
    requires_api_key: Optional[bool] = None,
    enabled: Optional[bool] = None,
    default_model: Optional[str] = None,
    access: Optional[dict] = None,
    path: Optional[str] = None,
) -> dict[str, Any]:
    """Add or update an instance entry in the overlay; returns the written entry."""
    registered = set(ProviderFactory.list_providers())
    overlay = read_overlay(path)
    entry = dict(overlay.get(alias, {}))
    for k, v in (
        ("provider", provider),
        ("base_url", base_url),
        ("credgoo_service", credgoo_service),
        ("requires_api_key", requires_api_key),
        ("enabled", enabled),
        ("default_model", default_model),
    ):
        if v is not None:
            entry[k] = v
    if access:
        entry["access"] = {**(entry.get("access") or {}), **access}
    is_builtin = alias in registered
    if not is_builtin:
        if "provider" not in entry:
            raise ValueError(
                f"custom instance '{alias}' must declare a 'provider'"
            )
        if entry["provider"] not in registered:
            raise ValueError(
                f"instance '{alias}' references unknown provider '{entry['provider']}'"
            )
    overlay[alias] = entry
    write_overlay(overlay, path)
    return entry


def remove_instance(alias: str, path: Optional[str] = None) -> bool:
    """Delete a custom alias. Built-ins refuse (route to disable/reset)."""
    if alias in ProviderFactory.list_providers():
        raise ValueError(
            f"'{alias}' is a built-in provider — use --disable-provider {alias} "
            f"(turn off) or --reset-provider {alias} (revert overrides)"
        )
    overlay = read_overlay(path)
    if alias not in overlay:
        raise ValueError(f"no custom instance '{alias}' to remove")
    del overlay[alias]
    write_overlay(overlay, path)
    return True


def set_instance_enabled(alias: str, enabled: bool, path: Optional[str] = None) -> dict[str, Any]:
    """Toggle an instance (built-in or custom) on/off via the overlay."""
    return upsert_instance(alias, enabled=enabled, path=path)


def reset_instance(alias: str, path: Optional[str] = None) -> bool:
    """Drop any overlay entry for *alias* (reverts a built-in to its registry default)."""
    overlay = read_overlay(path)
    if alias not in overlay:
        return False
    del overlay[alias]
    write_overlay(overlay, path)
    return True


def show_instance(alias: str, path: Optional[str] = None) -> InstanceSpec:
    """Return the resolved spec for *alias*."""
    return resolve_instance(alias, path=path)


def alias_serve_decision(age, ttl, inflight):
    """Decide how /v1/models/{alias} serves a custom alias.

    Args:
        age: Seconds since the alias's catalog entry was last refreshed, or
            None if it has never been cached.
        ttl: Freshness window in seconds.
        inflight: Whether a background refresh for this alias is already running.

    Returns:
        ``"fetch_sync"`` (no cache -> populate on first hit),
        ``"serve_cached"`` (fresh, or stale-but-refreshing), or
        ``"serve_cached_and_refresh"`` (stale and idle -> serve stale + refresh).
    """
    if age is None:
        return "fetch_sync"
    if age > ttl and not inflight:
        return "serve_cached_and_refresh"
    return "serve_cached"


def merge_custom_aliases(
    builtins_result: dict[str, Any],
    existing_providers: dict[str, Any],
    aliases: dict[str, InstanceSpec],
) -> dict[str, Any]:
    """Merge custom-alias catalog entries into a fresh built-in regenerate.

    The daily ``generate_models.py`` rebuilds the catalog from built-ins only;
    without this, custom fleet members would be wiped every refresh. Custom
    aliases (``is_builtin`` False) are carried over from the existing catalog
    so they persist until refreshed lazily via ``/v1/models/{alias}``. Built-in
    keys are never overwritten by stale existing copies (the fresh fetch wins).
    """
    merged = dict(builtins_result)
    for alias, spec in aliases.items():
        if spec.is_builtin:
            continue
        if alias in existing_providers and alias not in merged:
            merged[alias] = existing_providers[alias]
    return merged


# --------------------------------------------------------------------------- #
# Per-instance model overrides (layered rules: defaults < pattern < per-model)
# --------------------------------------------------------------------------- #
def selective_ids(spec: InstanceSpec) -> Optional[set]:
    """The restrictive model-id set for an instance, or None (= all models).

    A string-list (selective ids) or a dict (keys = ids) restricts; a list of
    match/set pattern rules, or an absent ``access.models``, does not.
    """
    models = (spec.access or {}).get("models")
    if isinstance(models, list) and all(isinstance(x, str) for x in models):
        return set(models)
    if isinstance(models, dict):
        return set(models.keys())
    return None


def instance_allows(spec: InstanceSpec, model) -> bool:
    """True if an instance's access rules permit this model.

    Two independent filters (both must pass when set):
    - ``access.models`` selective list/dict (see :func:`selective_ids`) — explicit ids.
    - ``access.only`` — an access tag (``"free"`` / ``"paid"``) to keep; auto-tracks
      the provider's current free/paid set, so a no-budget/free key instance stays
      correct as the catalog changes (no stale enumeration).

    ``model`` may be a ModelInfo or a plain dict.
    """
    acc = spec.access or {}
    allowed = selective_ids(spec)
    if allowed is not None:
        mid = model.get("id") if isinstance(model, dict) else getattr(model, "id", None)
        if mid not in allowed:
            return False
    only = acc.get("only")
    if only:
        macc = model.get("access") if isinstance(model, dict) else getattr(model, "access", None)
        if macc != only:
            return False
    return True


def apply_model_overrides(spec: InstanceSpec, model: dict) -> dict:
    """Return a copy of ``model`` with this instance's access overrides applied.

    Precedence (most-specific wins): per-model dict > ``match``/``set`` pattern
    rules > ``models_defaults`` > the catalog entry as-is.

    ``match`` matches if it's a substring of the model id OR equals the model's
    ``access`` tag (e.g. ``"(TRIAL)"`` or ``"free"``).
    """
    acc = spec.access or {}
    out = dict(model)
    for k, v in (acc.get("models_defaults") or {}).items():
        out[k] = v
    models = acc.get("models")
    if isinstance(models, dict):
        ov = models.get(out.get("id"))
        if isinstance(ov, dict):
            out.update(ov)
    elif isinstance(models, list):
        for rule in models:
            if not (isinstance(rule, dict) and "match" in rule and "set" in rule):
                continue
            m = rule["match"]
            mid = out.get("id") or ""
            if (isinstance(m, str) and (m in mid or out.get("access") == m)):
                out.update(rule["set"] or {})
    return out
