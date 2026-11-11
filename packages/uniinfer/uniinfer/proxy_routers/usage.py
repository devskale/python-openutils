"""Usage router: per-key quota attribution for statuslines and scripts.

GET /v1/usage — read-only reporting across providers and upstream keys:

- ``providers``: one entry per configured/active provider. Each carries a
  ``pool`` (per-upstream-key usage from the instances overlay — pool key
  labels resolve via credgoo service -> sha256, never key material),
  aggregated ``used``/``limits`` (n_keys x documented per-key limits) and
  ``worst`` (highest-utilized key+window). ``mine`` is the caller's own
  attribution (pool + caller-supplied X-Provider-Key keys).
- ``windows``: legacy aggregate (primary provider, per-model weighted) —
  kept verbatim for the existing statusline consumer during migration.

Everything is config-driven: providers and limits come from
provider_limits.json, pool membership from the deployed instances overlay
(provider_instances.json), the primary provider defaults to 'tu'
(UNIINFER_USAGE_PRIMARY_PROVIDER). No provider is hardcoded.

Same bearer auth as the chat API; unknown or revoked tokens get 401.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from uniinfer.auth import _allowed_token_hashes, _token_metadata, validate_proxy_token
from uniinfer.config.providers import (
    get_all_provider_limits,
    get_quota_source,
    get_request_multipliers,
    get_window_limits,
)
from uniinfer.proxy_services.key_usage import (
    bucket_end,
    bucket_start,
    get_key_usage,
    token_hash,
    upstream_key_hash,
)

_WINDOW_KINDS = ("minute", "hour", "day")


def _primary_provider() -> str:
    """The provider the legacy ``windows`` aggregate reports (env-configurable)."""
    return os.getenv("UNIINFER_USAGE_PRIMARY_PROVIDER", "tu").strip() or "tu"


def _pool_labels(provider: str) -> dict[str, str]:
    """Map upstream-key-hash -> label for a provider's pool (instances overlay).

    Pool membership is deploy config: every overlay instance whose spec
    resolves to the provider (entity chain: spec.provider == provider or
    alias == provider) contributes its credgoo service's key under its alias
    name. Keys that cannot be resolved (credgoo failure) are skipped — an
    unresolved key is not verifiably in the pool.
    """
    from uniinfer.config.instances import load_instances

    labels: dict[str, str] = {}
    for alias, spec in load_instances().items():
        entity = getattr(spec, "entity", None) or spec.provider or alias
        if entity != provider and alias != provider:
            continue
        service = spec.credgoo_service or alias
        try:
            from credgoo import get_api_key

            key = get_api_key(service)
        except Exception:  # noqa: BLE001 — label resolution must never 500
            continue
        if key:
            labels[upstream_key_hash(key)] = alias
    return labels


def _display_key(name: str) -> str:
    """Truncated key name for the response — never a secret."""
    return name if len(name) <= 8 else name[:8] + "…"


def create_usage_router() -> APIRouter:
    router = APIRouter()

    @router.get("/v1/usage")
    async def usage(
        api_bearer_token: str = Depends(validate_proxy_token),
    ) -> JSONResponse:
        th = token_hash(api_bearer_token)
        allowed = _allowed_token_hashes()
        if allowed is not None and th not in allowed:
            raise HTTPException(status_code=401, detail="Unknown or revoked token.")
        meta: dict[str, Any] = _token_metadata().get("tokens", {}).get(th, {})
        collector = get_key_usage()
        all_limits = get_all_provider_limits()
        now = time.time()

        def _window_reset(kind: str) -> str:
            return (
                datetime.fromtimestamp(
                    bucket_end(kind, bucket_start(kind, int(now))), timezone.utc
                )
                .isoformat()
                .replace("+00:00", "Z")
            )

        primary = _primary_provider()
        providers_cfg = {primary} | set(all_limits) | collector.active_providers()

        providers: dict[str, Any] = {}
        for prov in sorted(providers_cfg):
            multipliers = get_request_multipliers(prov)
            limits_cfg = all_limits.get(prov, {})
            per_key_limits = get_window_limits(prov)
            labels = _pool_labels(prov)
            pool_raw = collector.provider_pool_view(prov, multipliers, now=now)

            keys: list[dict[str, Any]] = []
            upstream_mode = get_quota_source(prov) == "upstream-headers"
            for kh, label in sorted(labels.items(), key=lambda kv: kv[1]):
                used = pool_raw.get(kh, {})
                entry: dict[str, Any] = {
                    "key": label,
                    "used": {kind: used.get(kind, 0.0) for kind in _WINDOW_KINDS},
                    # Per-key truth: this key's fill vs ITS OWN quota. The
                    # statusline shows the key the current model rides on
                    # (model prefix = alias = key) — no pool aggregates.
                    "limits": dict(per_key_limits),
                }
                if upstream_mode:
                    from uniinfer.proxy_services import upstream_quota
                    from datetime import timedelta

                    truth: dict[str, Any] = {}
                    for model, e in upstream_quota.view(kh).items():
                        dims = {}
                        for dim in ("requests", "tokens"):
                            d = e.get(dim)
                            if not d:
                                continue
                            reset_at = None
                            if d.get("reset_s") is not None:
                                reset_at = (
                                    datetime.fromtimestamp(
                                        e["ts"] + d["reset_s"], timezone.utc
                                    ).isoformat().replace("+00:00", "Z")
                                )
                            dims[dim] = {
                                "used": d["limit"] - d["remaining"],
                                "limit": d["limit"],
                                "remaining": d["remaining"],
                                **({"reset_at": reset_at} if reset_at else {}),
                            }
                        if dims:
                            truth[model] = {"age_s": int(time.time() - e["ts"]), **dims}
                    if truth:
                        entry["upstream"] = truth
                keys.append(entry)

            mine = collector.usage_view(th, multipliers, provider=prov, now=now)

            providers[prov] = {
                "pool": {"keys": keys},
                "mine": {
                    kind: mine["windows"][kind]["used"] for kind in _WINDOW_KINDS
                },
                "multiplier_note": limits_cfg.get("multiplier_note"),
                "source": limits_cfg.get("source"),
            }

        primary_mults = get_request_multipliers(primary)
        legacy = collector.usage_view(th, primary_mults, provider=primary, now=now)
        windows = {
            kind: {
                "used": legacy["windows"][kind]["used"],
                "limit": get_window_limits(primary).get(kind),
                "reset_at": _window_reset(kind),
            }
            for kind in _WINDOW_KINDS
        }

        payload = {
            "key": _display_key(str(meta.get("name") or "unnamed")),
            "providers": providers,
            "windows": windows,
            "models": legacy["models"],
            "upstream": {"primary_provider": primary},
        }
        return JSONResponse(content=payload)

    return router
