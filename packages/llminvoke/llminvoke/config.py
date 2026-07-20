"""Canonical model config — registry loading + resolution.

The deep module behind ``resolve_model``. Loads the shipped registry
(``models.yml``) + optional runtime ``clients.yml``, merges them per the
precedence chain (ADR 0004), filters DSGVO-incompatible backups, and returns a
``ResolvedConfig`` ready for the retry/backup loop.

Precedence (high → low)::

    env var  >  team-settings (DB)  >  clients.yml (runtime)  >  catalog default

This module is intentionally free of LLM-call logic — it only *resolves*. The
retry/backup loop lives in ``__init__.call_llm`` and consumes ``ResolvedConfig``.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover — pyyaml is a declared dep
    yaml = None

__all__ = [
    "ModelRef",
    "RetryPolicy",
    "ResolvedConfig",
    "resolve_model",
    "get_model_info",
    "is_dsgvo_provider",
    "classify_error",
]

_logger = logging.getLogger("llminvoke.config")

# ── paths ──────────────────────────────────────────────────────────────
_PACKAGE_DIR = Path(__file__).parent
_REGISTRY_PATH = _PACKAGE_DIR / "models.yml"

# ── caches ─────────────────────────────────────────────────────────────
_registry_cache: dict[str, Any] | None = None
_clients_cache: dict[str, Any] | None = None
_clients_mtime: float | None = None


# ════════════════════════════════════════════════════════════════════════
# Data shapes
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ModelRef:
    """A provider@model pair — one slot in a resolution chain."""
    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}@{self.model}"

    @classmethod
    def parse(cls, spec: str) -> "ModelRef":
        """Parse ``provider@model`` (model may itself contain ``@``)."""
        if "@" not in spec:
            raise ValueError(f"model spec must be 'provider@model', got: {spec!r}")
        provider, model = spec.split("@", 1)
        return cls(provider=provider.strip(), model=model.strip())


@dataclass(frozen=True)
class RetryPolicy:
    """How a single model is retried before escalating to the next backup."""
    attempts: int = 3
    backoff: str = "exponential"       # exponential | fixed
    base_delay: float = 2.0            # seconds
    max_delay: float = 30.0
    honor_retry_after: bool = True

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Compute the sleep before ``attempt`` (0-indexed). 0 = no wait."""
        if retry_after is not None and self.honor_retry_after:
            return min(retry_after, self.max_delay)
        if attempt <= 0:
            return 0.0
        if self.backoff == "exponential":
            return min(self.base_delay * (2 ** attempt), self.max_delay)
        return min(self.base_delay, self.max_delay)  # fixed


@dataclass
class ResolvedConfig:
    """Everything the retry/backup loop needs, DSGVO-filtered at resolve time.

    ``chain`` is ``[primary] + backups`` — already filtered so a DSGVO-bound
    client never has a non-DSGVO backup. The loop walks it in order.
    """
    primary: ModelRef
    backups: list[ModelRef] = field(default_factory=list)
    temperature: float = 0.7
    max_tokens: int = 4096
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    dsgvo_required: bool = False

    @property
    def chain(self) -> list[ModelRef]:
        """Ordered: primary first, then backups."""
        return [self.primary, *self.backups]

    @property
    def provider(self) -> str:
        """Convenience — the primary provider (for create_provider)."""
        return self.primary.provider

    @property
    def model(self) -> str:
        """Convenience — the primary model."""
        return self.primary.model


# ════════════════════════════════════════════════════════════════════════
# Registry + clients loading
# ════════════════════════════════════════════════════════════════════════

def _load_registry() -> dict[str, Any]:
    """Load + cache the shipped models.yml (catalog + default profile)."""
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache
    if yaml is None:
        raise ImportError("pyyaml is required: uv add pyyaml")
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        _registry_cache = yaml.safe_load(f) or {}
    return _registry_cache


def _clients_yml_path() -> Path | None:
    """Locate the runtime clients.yml (env override > data dir > none)."""
    explicit = os.environ.get("KONTEXT_CLIENTS_YML", "").strip()
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    data_dir = os.environ.get("KONTEXT_DATA_DIR", "").strip()
    if data_dir:
        p = Path(data_dir) / "clients.yml"
        return p if p.is_file() else None
    return None


def _load_clients() -> dict[str, Any]:
    """Load + cache the runtime clients.yml, reloading on mtime change.

    The clients file is editable without redeploy, so we watch its mtime and
    re-read when it changes (cheap stat per resolve).
    """
    global _clients_cache, _clients_mtime
    path = _clients_yml_path()
    if path is None:
        return {}
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if _clients_cache is not None and _clients_mtime == mtime:
        return _clients_cache
    if yaml is None:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            _clients_cache = yaml.safe_load(f) or {}
        _clients_mtime = mtime
    except (OSError, yaml.YAMLError) as exc:
        _logger.warning("Failed to load clients.yml (%s): %s", path, exc)
        _clients_cache = {}
    return _clients_cache


def reload_config() -> None:
    """Force a reload of both registry + clients caches."""
    global _registry_cache, _clients_cache, _clients_mtime
    _registry_cache = None
    _clients_cache = None
    _clients_mtime = None


# ── catalog queries ────────────────────────────────────────────────────

def get_model_info(spec: str) -> dict[str, Any]:
    """Look up a model's metadata from the catalog (context_window, etc.).

    ``spec`` is ``provider@model``. Returns ``{}`` if unknown (best-effort —
    callers fall back to conservative defaults).
    """
    models = _load_registry().get("models", {})
    # try exact key, then the model-only suffix
    if spec in models:
        return models[spec]
    if "@" in spec:
        _, model = spec.split("@", 1)
        return models.get(model, {})
    return {}


def is_dsgvo_provider(provider: str) -> bool:
    """Whether a provider is DSGVO-approved (from the registry)."""
    providers = _load_registry().get("providers", {})
    return bool(providers.get(provider, {}).get("dsgvo", False))


# ════════════════════════════════════════════════════════════════════════
# Resolution
# ════════════════════════════════════════════════════════════════════════

def _parse_retry(raw: dict | None, fallback: RetryPolicy) -> RetryPolicy:
    """Build a RetryPolicy from a config dict, inheriting unset fields."""
    if not raw:
        return fallback
    return RetryPolicy(
        attempts=int(raw.get("attempts", fallback.attempts)),
        backoff=str(raw.get("backoff", fallback.backoff)),
        base_delay=float(raw.get("base_delay", fallback.base_delay)),
        max_delay=float(raw.get("max_delay", fallback.max_delay)),
        honor_retry_after=bool(raw.get("honor_retry_after", fallback.honor_retry_after)),
    )


def _filter_dsgvo(refs: list[ModelRef], dsgvo_required: bool) -> list[ModelRef]:
    """Drop non-DSGVO providers when the client is DSGVO-bound."""
    if not dsgvo_required:
        return refs
    kept = [r for r in refs if is_dsgvo_provider(r.provider)]
    if len(kept) < len(refs):
        dropped = [str(r) for r in refs if r not in kept]
        _logger.info("DSGVO filter dropped non-DSGVO backups: %s", ", ".join(dropped))
    return kept


def resolve_model(
    package: str | None = None,
    client: str | None = None,
    task: str | None = None,
    env_prefix: str | None = None,
) -> ResolvedConfig:
    """Resolve the effective model config per the ADR 0004 precedence chain.

    Args:
        package: package name (pdf2md, strukt2meta, ...) — applies package defaults.
        client: client/deal id — applies per-client overrides from clients.yml.
            Defaults to ``$KONTEXT_CLIENT`` env, then ``"default"``.
        task: task-type within a package (e.g. strukt2meta's ``kriterien``).

    Returns a DSGVO-filtered ``ResolvedConfig`` ready for the retry/backup loop.
    Env overrides (``<PREFIX>_MODEL`` etc.) are applied by the caller, not here.
    """
    registry = _load_registry()
    default = registry.get("default", {})
    packages = registry.get("packages", {})

    # ── start from catalog default ──
    primary_spec = default.get("model", "tu@qwen-3.6-35b")
    temperature = float(default.get("temperature", 0.7))
    max_tokens = int(default.get("max_tokens", 4096))
    retry = _parse_retry(default.get("retry"), RetryPolicy())
    backups_raw: list[str] = list(default.get("backups", []))
    dsgvo_required = False

    # ── layer 1: package defaults (engineering tuning) ──
    pkg_cfg = packages.get(package or "", {})
    if pkg_cfg:
        primary_spec = pkg_cfg.get("model", primary_spec)
        temperature = float(pkg_cfg.get("temperature", temperature))
        max_tokens = int(pkg_cfg.get("max_tokens", max_tokens))
        backups_raw = list(pkg_cfg.get("backups", backups_raw))
        retry = _parse_retry(pkg_cfg.get("retry"), retry)
        # task-type override within package
        if task:
            task_cfg = pkg_cfg.get("tasks", {}).get(task, {})
            primary_spec = task_cfg.get("model", primary_spec)
            temperature = float(task_cfg.get("temperature", temperature))
            max_tokens = int(task_cfg.get("max_tokens", max_tokens))

    # ── layer 2: client overrides (business choice, from clients.yml) ──
    client_id = client or os.environ.get("KONTEXT_CLIENT", "").strip() or "default"
    clients = _load_clients()
    client_cfg = clients.get(client_id, {})
    if client_cfg:
        primary_spec = client_cfg.get("model", primary_spec)
        temperature = float(client_cfg.get("temperature", temperature))
        max_tokens = int(client_cfg.get("max_tokens", max_tokens))
        backups_raw = list(client_cfg.get("backups", backups_raw))
        retry = _parse_retry(client_cfg.get("retry"), retry)
        dsgvo_required = bool(client_cfg.get("dsgvo_required", dsgvo_required))

    # ── build refs + DSGVO-filter backups ──
    primary = ModelRef.parse(primary_spec)
    backup_refs = [ModelRef.parse(s) for s in backups_raw]
    backup_refs = _filter_dsgvo(backup_refs, dsgvo_required)

    # ── layer 3 (highest): env override — primary only (Q12) ──
    # Backups still flow from config so a container pinning its primary
    # doesn't lose fallback protection.
    if env_prefix:
        ep = os.environ.get(f"{env_prefix}_PROVIDER", "").strip()
        em = os.environ.get(f"{env_prefix}_MODEL", "").strip()
        if ep and em:
            primary = ModelRef(provider=ep, model=em)

    return ResolvedConfig(
        primary=primary,
        backups=backup_refs,
        temperature=temperature,
        max_tokens=max_tokens,
        retry=retry,
        dsgvo_required=dsgvo_required,
    )


# ════════════════════════════════════════════════════════════════════════
# Error classification (for the retry loop)
# ════════════════════════════════════════════════════════════════════════

# Permanent errors escalate to the next backup immediately (no retry).
PERMANENT_ERRORS = frozenset({"auth_error", "context_window_exceeded", "not_found", "bad_request"})


def classify_error(error: BaseException) -> str:
    """Classify an exception into a short tag for retry decisions.

    Inspects the message + class name + ``__cause__`` chain. Mirrors agentos'
    ``circuit_breaker.classify_error`` taxonomy so the two tiers agree.
    """
    parts: list[str] = []
    cause: BaseException | None = error
    while cause is not None:
        parts.append(str(cause).lower())
        parts.append(type(cause).__name__.lower())
        cause = cause.__cause__
    combined = " ".join(parts)

    if any(k in combined for k in ("context window", "token limit", "max_tokens", "too long")):
        return "context_window_exceeded"
    if any(k in combined for k in ("429", "rate limit", "too many requests")):
        return "rate_limited"
    if any(k in combined for k in ("timeout", "timed out", "deadline")):
        return "timeout"
    if any(k in combined for k in ("connection", "network", "dns", "refused", "unreachable", "eof")):
        return "network_error"
    if any(k in combined for k in ("401", "403", "unauthorized", "forbidden", "authentication", "api key")):
        return "auth_error"
    if any(k in combined for k in ("404", "not found", "model not")):
        return "not_found"
    if any(k in combined for k in ("400", "bad request", "invalid")):
        return "bad_request"
    if any(k in combined for k in ("500", "502", "503", "504", "internal", "server error")):
        return "server_error"
    if "providererror" in combined:
        return "provider_error"
    return type(error).__name__.lower()


def _extract_retry_after(error: BaseException) -> float | None:
    """Best-effort: pull a retry-after value (seconds) from an error."""
    # direct attribute (some HTTP libs expose it)
    for attr in ("retry_after", "retryAfter"):
        val = getattr(error, attr, None)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    # parse from message: "retry after 5" / "retry-after: 5"
    msg = str(error).lower()
    m = re.search(r"retry[- ]after[:\s]+(\d+(?:\.\d+)?)", msg)
    if m:
        return float(m.group(1))
    return None


# ════════════════════════════════════════════════════════════════════════
# Alarm emission (structured log — the worker endpoint reads these)
# ════════════════════════════════════════════════════════════════════════

def emit_alarm(
    severity: str,
    provider: str,
    model: str,
    error_type: str = "",
    message: str = "",
    *,
    package: str | None = None,
    client: str | None = None,
) -> None:
    """Emit a structured alarm log entry (JSON).

    ``severity`` is ``"alarm"`` (empty/failure) — the worker's alarm endpoint
    aggregates these for healthcheck/UI surfacing (ADR 0004 §6).
    """
    entry = {
        "ts": time.time(),
        "severity": severity,
        "provider": provider,
        "model": model,
        "error_type": error_type,
        "message": message[:500],
        "package": package,
        "client": client,
    }
    _logger.warning("ALARM %s", json.dumps(entry, ensure_ascii=False))
