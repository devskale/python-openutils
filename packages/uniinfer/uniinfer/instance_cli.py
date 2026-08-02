"""CLI helper for the provider-instance fleet (``--add-provider`` & friends).

A thin shell over the pure primitives in ``uniinfer.config.instances`` plus the
smart add-probe (reachability + anonymous-detect + credgoo key store). See
``docs/provider-instances-design.md``.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx

from uniinfer.config.instances import (
    DEFAULT_FILENAME,
    instance_file_path,
    read_overlay,
    remove_instance,
    reset_instance,
    set_instance_enabled,
    show_instance,
    upsert_instance,
    write_overlay,
)
from uniinfer.factory import ProviderFactory


# --------------------------------------------------------------------------- #
# smart add-probe
# --------------------------------------------------------------------------- #
def _probe(base_url: str, api_key: Optional[str]) -> tuple[bool, bool]:
    """Probe ``{base_url}/models``. Returns (reachable, anonymous).

    anonymous = reachable WITHOUT a key (-> requires_api_key false).
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", headers=headers, timeout=10.0)
        return r.status_code == 200, (r.status_code == 200 and not api_key)
    except Exception:
        return False, False


def _store_key_in_credgoo(service: str, key: str) -> bool:
    """Best-effort store a key under a credgoo service (no interactive confirm)."""
    try:
        from credgoo.store import CredentialStore, _resolve_cache_dir  # type: ignore

        store = CredentialStore(_resolve_cache_dir(None))
        if not store.supports("add_key"):
            return False
        return bool(store.add(service, key))
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# verb handlers
# --------------------------------------------------------------------------- #
def _do_init() -> None:
    path = instance_file_path()
    if os.path.exists(path):
        print(f"{path} already exists — leaving it untouched.")
        return
    write_overlay({}, path=path)
    print(f"Wrote {path} (empty object — all built-ins active, no overrides).")
    print("Add a fleet member with:")
    print("  uniinfer --add-provider <alias> --provider-class <class> --base-url <url>")
    print("<class> is a registered provider key, e.g. openai-compat, ollama, tu, zai.")
    print("Schema & examples: docs/provider-instances-design.md")


def _do_add(args: Any) -> None:
    alias = args.add_provider
    provider: Optional[str] = args.provider_class
    base_url: Optional[str] = args.base_url
    credgoo_service: Optional[str] = args.credgoo_service
    requires_api_key: Optional[bool] = False if args.no_key else None

    # key handling first (so the probe can use it to distinguish anonymous)
    api_key_for_probe: Optional[str] = None
    if args.key:
        svc = credgoo_service or alias
        if _store_key_in_credgoo(svc, args.key):
            print(f"Stored key in credgoo under '{svc}'.")
            api_key_for_probe = args.key
        else:
            print(f"Warning: could not store key in credgoo '{svc}' — continuing.")

    # smart probe (unless --no-verify or no base_url)
    if base_url and not args.no_verify:
        reachable, anonymous = _probe(base_url, api_key_for_probe)
        if reachable:
            print(f"Probe OK: {base_url}/models reachable" + (" (anonymous — keyless)" if anonymous else ""))
            if anonymous and requires_api_key is None:
                requires_api_key = False
            if provider is None:
                provider = _infer_provider(base_url)
        else:
            print(f"Warning: {base_url}/models not reachable — saved anyway (use --no-verify to silence).")

    if provider is None:
        raise SystemExit(
            "error: --provider is required (or pass --base-url without --no-verify so it can be inferred)."
        )

    entry = upsert_instance(
        alias,
        provider=provider,
        base_url=base_url,
        credgoo_service=credgoo_service,
        requires_api_key=requires_api_key,
        access=_access_from_args(args),
    )
    print(f"Saved instance '{alias}': {json.dumps(entry)}")
    print(f"Use it as: {alias}@<model>")


def _access_from_args(args: Any) -> Optional[dict]:
    """Build an access dict from --keytype/--only (None if neither set)."""
    access: dict = {}
    if getattr(args, "keytype", None):
        access["keytype"] = args.keytype
    if getattr(args, "only", None):
        access["only"] = args.only
    return access or None


def _do_tag(args: Any) -> None:
    """Tag an existing instance's access (--keytype / --only)."""
    alias = args.tag
    access = _access_from_args(args)
    if not access:
        raise SystemExit("error: --tag requires --keytype and/or --only")
    entry = upsert_instance(alias, access=access)
    print(f"Tagged '{alias}' access: {json.dumps(entry.get('access'))}")


def _infer_provider(base_url: str) -> str:
    """Guess the underlying class from the URL shape."""
    if ":11434" in base_url or "/api/" in base_url:
        return "ollama"
    return "openai-compat"


def _do_remove(args: Any) -> None:
    if remove_instance(args.remove_provider):
        print(f"Removed custom instance '{args.remove_provider}'.")


def _do_enable(args: Any, enabled: bool) -> None:
    set_instance_enabled(args.enable_provider or args.disable_provider, enabled)
    alias = args.enable_provider or args.disable_provider
    state = "enabled" if enabled else "disabled"
    print(f"{state.capitalize()} '{alias}'.")


def _do_reset(args: Any) -> None:
    if reset_instance(args.reset_provider):
        print(f"Reset '{args.reset_provider}' to registry defaults.")
    else:
        print(f"'{args.reset_provider}' had no overrides to reset.")


def _do_show(args: Any) -> None:
    spec = show_instance(args.show_provider)
    kind = "built-in" if spec.is_builtin else "custom"
    print(f"{spec.alias}  ({kind}, provider={spec.provider})")
    print(f"  base_url:         {spec.base_url}")
    print(f"  credgoo_service:  {spec.credgoo_service}")
    print(f"  requires_api_key: {spec.requires_api_key}")
    print(f"  enabled:          {spec.enabled}")
    print(f"  default_model:    {spec.default_model}")
    print(f"  access:           {spec.access or '-'}")


def manage(args: Any) -> None:
    """Dispatch the provider-management flags."""
    import sys

    try:
        if args.init:
            _do_init()
        elif args.add_provider:
            _do_add(args)
        elif args.remove_provider:
            _do_remove(args)
        elif args.enable_provider:
            _do_enable(args, True)
        elif args.disable_provider:
            _do_enable(args, False)
        elif args.reset_provider:
            _do_reset(args)
        elif getattr(args, "tag", None):
            _do_tag(args)
        elif args.show_provider:
            _do_show(args)
    except (ValueError, SystemExit) as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
