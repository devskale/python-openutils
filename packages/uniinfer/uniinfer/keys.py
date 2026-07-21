"""Classify API keys by access tier: free / balance / paid / invalid / nokey.

Lean probe — calls each provider's ``list_models()`` (a single ``GET /models``,
zero inference tokens) to determine whether the key is valid and, where the
provider exposes it, whether the account has a free tier, a prepaid balance,
or paid per-token billing.

Key tiers
---------
- ``free``    : $0 to use. May still need a key (groq, gemini) but no payment.
- ``balance`` : prepaid credit that depletes; model stops when $0. Needs a key.
- ``paid``    : per-token billing against a card/subscription. Needs a funded key.
- ``invalid`` : key rejected (401/403) or never set.
- ``nokey``   : provider needs no key (ollama local, pollinations anonymous).

The free/balance/paid distinction comes from two signals:

1. Provider-level *universally-free* flag (groq, pollinations, ollama, tu):
   the whole provider is free, so any working key is ``free``.
2. Per-model pricing in the ``list_models`` response (openrouter, kilo, ngc):
   if *any* served model has ``cost.input == 0`` the account can use the free
   tier; if all served models cost money it is ``paid``.

For providers that expose no pricing (mistral, moonshot, stepfun, …) the tier
stays ``""`` (unknown) rather than guessing — a working key with no pricing
info is neither claimed free nor paid.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

from .core import ModelInfo
from .factory import ProviderFactory

log = logging.getLogger(__name__)

# Providers where *every* model is free (forever-free tier, no payment).
# A working key on one of these is always ``free``. Sourced from each
# provider's docs / terms; updated when a provider changes its tier model.
_UNIVERSALLY_FREE = frozenset({"groq", "pollinations", "ollama", "tu", "tu-staging"})

# Providers that need no key at all (anonymous access).
_NO_KEY_REQUIRED = frozenset({"ollama", "pollinations"})

# Credgoo service name overrides where it differs from the provider name.
_CREDGOO_SERVICE_OVERRIDES = {
    "kilo": "kilocode",
    # zai / zai-code / tu / tu-staging already match their provider names
}


@dataclass
class KeyReport:
    """Result of probing one provider's key."""

    provider: str
    credgoo_service: str
    tier: str  # "free" | "balance" | "paid" | "invalid" | "nokey" | ""
    status: str  # "ok" | "no_key" | "auth_error" | "error" | "skipped"
    n_models: int = 0
    n_free_models: int = 0
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "credgoo_service": self.credgoo_service,
            "tier": self.tier,
            "status": self.status,
            "n_models": self.n_models,
            "n_free_models": self.n_free_models,
            "detail": self.detail,
        }


def _credgoo_service(provider: str) -> str:
    """Resolve the credgoo service name for a provider.

    Most providers use their own name as the credgoo service; a few override
    it (kilo -> kilocode). Providers that need no key return "".
    """
    if provider in _NO_KEY_REQUIRED:
        return ""
    return _CREDGOO_SERVICE_OVERRIDES.get(provider, provider)


def _resolve_key(provider: str, credgoo_service: str, encryption_key: Optional[str] = None) -> Optional[str]:
    """Fetch the API key for a provider via credgoo, or None if not set."""
    if not credgoo_service:
        return None
    try:
        from credgoo import get_api_key
        return get_api_key(credgoo_service, encryption_key=encryption_key)
    except Exception as e:  # not configured, backend error, etc.
        log.debug("no key for %s via %s: %s", provider, credgoo_service, e)
        return None


def _probe_one(provider: str, encryption_key: Optional[str] = None) -> KeyReport:
    """Probe a single provider: resolve key, call list_models, classify tier."""
    svc = _credgoo_service(provider)

    # Providers that need no key: probe without one.
    if provider in _NO_KEY_REQUIRED:
        try:
            cls = ProviderFactory.get_provider_class(provider)
            models = cls.list_models()
            return KeyReport(
                provider=provider, credgoo_service="", tier="nokey",
                status="ok", n_models=len(models),
                detail=f"{len(models)} models, no key needed",
            )
        except Exception as e:
            return KeyReport(provider, "", "", "error", detail=str(e)[:120])

    key = _resolve_key(provider, svc, encryption_key)
    if not key:
        return KeyReport(provider, svc, "", "no_key", detail="no key in credgoo")

    cls = ProviderFactory.get_provider_class(provider)
    try:
        models = cls.list_models(api_key=key)
    except Exception as e:
        # Classify the failure: auth vs other.
        from .errors import AuthenticationError
        if isinstance(e, AuthenticationError):
            return KeyReport(provider, svc, "invalid", "auth_error", detail=str(e)[:120])
        # Some providers return 402/429 (balance exhausted) as a generic error.
        msg = str(e).lower()
        if "balance" in msg or "insufficient" in msg or "credit" in msg:
            return KeyReport(provider, svc, "balance", "error", detail=str(e)[:120])
        return KeyReport(provider, svc, "", "error", detail=str(e)[:120])

    n = len(models)
    # Universally-free provider: any working key is free.
    if provider in _UNIVERSALLY_FREE:
        return KeyReport(provider, svc, "free", "ok", n_models=n,
                         detail=f"{n} models, universally-free provider")

    # Derive tier from per-model pricing where available.
    free_models = [m for m in models if _is_free(m)]
    priced_models = [m for m in models if _has_pricing(m)]
    n_free = len(free_models)
    if n_free and n_free == n:
        # All served models are free.
        return KeyReport(provider, svc, "free", "ok", n_models=n, n_free_models=n_free,
                         detail=f"all {n} models free")
    if n_free:
        # Mix of free and paid — account can use the free tier.
        return KeyReport(provider, svc, "free", "ok", n_models=n, n_free_models=n_free,
                         detail=f"{n_free}/{n} models free")
    if priced_models and not n_free:
        # Models have explicit pricing and none are free -> paid.
        return KeyReport(provider, svc, "paid", "ok", n_models=n,
                         detail=f"{n} models, none free")
    # No pricing info on any model — can't classify. Keep "" (unknown).
    return KeyReport(provider, svc, "", "ok", n_models=n,
                     detail=f"{n} models, no pricing info")


def _is_free(m: ModelInfo) -> bool:
    """A model is free if its cost.input is explicitly 0."""
    cost = m.cost
    if not cost:
        return False
    return cost.get("input") == 0


def _has_pricing(m: ModelInfo) -> bool:
    """A model has pricing info if its cost dict carries an input value."""
    cost = m.cost
    if not cost:
        return False
    return cost.get("input") is not None


def probe_keys(
    providers: Optional[list[str]] = None,
    *,
    max_workers: int = 8,
    encryption_key: Optional[str] = None,
) -> list[KeyReport]:
    """Probe keys for all (or given) providers in parallel.

    Args:
        providers: Provider names to probe; None = all registered providers.
        max_workers: Parallelism for the lean /models probes.
        encryption_key: Optional credgoo encryption key.

    Returns:
        List of :class:`KeyReport` sorted by provider name.
    """
    names = providers or sorted(ProviderFactory.list_providers())
    reports: list[KeyReport] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_probe_one, n, encryption_key): n for n in names}
        for f in as_completed(futs):
            try:
                reports.append(f.result())
            except Exception as e:
                name = futs[f]
                reports.append(KeyReport(name, "", "", "error", detail=str(e)[:120]))
    reports.sort(key=lambda r: r.provider)
    return reports


def format_report(reports: list[KeyReport]) -> str:
    """Format key probe results as a fixed-width table."""
    lines = [f"{'PROVIDER':14} {'TIER':8} {'STATUS':10} {'MODELS':>6} {'FREE':>5}  DETAIL"]
    lines.append("-" * 78)
    for r in reports:
        lines.append(
            f"{r.provider:14} {r.tier or '':8} {r.status:10} {r.n_models:>6} "
            f"{r.n_free_models:>5}  {r.detail}"
        )
    return "\n".join(lines)
