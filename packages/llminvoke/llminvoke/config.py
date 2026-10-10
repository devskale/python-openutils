"""Canonical model config — registry loading + resolution.

The deep module behind ``resolve_model``. Loads the shipped catalog
(``models.yml``, packaged) + the optional box overlay (``<box>/models.yml``,
gitignored, mtime-hot-reload), deep-merges them per the models.yml-hierarchy
contract (issue llminvoke-models-hierarchy; ADR 0004 amendment in progress),
filters DSGVO-incompatible backups, and returns a ``ResolvedConfig`` ready
for the retry/backup loop.

Precedence (high → low)::

    env (caller prefix)  >  box overlay (default/package/task)  >  catalog (default/package/task)

The overlay carries the box's CHOICE (base_url, bearer, model, task routing);
engineering sampling (temperature, max_tokens, backups, retry) lives in the
catalog — a true overlay may still override it deliberately (deep-merge),
but a LEGACY ``clients.yml`` (loaded via ``KONTEXT_CLIENTS_YML`` during the
WP4 transition) has its sampling keys stripped + warned: they are
known-drifted duplicates (audit 2026-10-09).

Merge semantics: dicts deep-merge (overlay wins), lists/scalars replace
wholesale. Only the ``default`` + ``packages`` subtrees merge — ``providers``
(picker list, overlay-only) and ``models`` (catalog-only) never mix.

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

try:  # credgoo resolves `credgoo:<service>` bearer refs (a declared dep)
    from credgoo import get_api_key as _credgoo_get_api_key
except ImportError:  # pragma: no cover
    _credgoo_get_api_key = None

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
_overlay_cache: dict[str, Any] | None = None
_overlay_mtime: float | None = None
_overlay_legacy: bool = False   # True while a clients.yml feeds the overlay (WP4 transition)


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
    request_kwargs: dict = field(default_factory=dict)  # e.g. chat_template_kwargs
    # OpenAI-compatible endpoint (the gateway). None = use the provider's own endpoint.
    base_url: str | None = None
    # Lokaler OpenAI-kompatibler Server (z.B. vLLM auf der Box): model-id BARE
    # senden — das provider@model-format ist das UNII-GATEWAY-routing, ein lokaler
    # vllm kennt nur die nackte model-id (dgx 2026-09-22). clients.yml: bare_model: true
    bare_model: bool = False
    bearer: str | None = None          # resolved key (credgoo/env/inline), not the ref

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


def _box_path() -> tuple[Path, bool] | None:
    """Locate the box overlay: ``(path, legacy)`` or None.

    ``KONTEXT_MODELS_YML`` is the true overlay; ``KONTEXT_CLIENTS_YML`` (and a
    ``clients.yml`` in KONTEXT_DATA_DIR) loads as a LEGACY overlay with
    sampling keys stripped — the WP4 migration bridge, removable once every
    box carries a models.yml.
    """
    explicit_new = os.environ.get("KONTEXT_MODELS_YML", "").strip()
    if explicit_new:
        p = Path(explicit_new)
        return (p, False) if p.is_file() else None
    explicit_legacy = os.environ.get("KONTEXT_CLIENTS_YML", "").strip()
    if explicit_legacy:
        p = Path(explicit_legacy)
        return (p, True) if p.is_file() else None
    data_dir = os.environ.get("KONTEXT_DATA_DIR", "").strip()
    if data_dir:
        for name, legacy in (("models.yml", False), ("clients.yml", True)):
            p = Path(data_dir) / name
            if p.is_file():
                return p, legacy
    return None


# Sampling keys a legacy clients.yml must NOT carry into the merge — the
# catalog owns them (audit 2026-10-09: max_tokens 64000 vs 10000 drift etc.).
# Choice keys (model/base_url/bearer/thinking/bare_model/dsgvo_required)
# stay honored from legacy files; true overlays may set anything.
_LEGACY_STRIP_KEYS = frozenset({"temperature", "max_tokens", "backups", "retry"})


def _sanitize_legacy(cfg: dict[str, Any]) -> dict[str, Any]:
    """Strip drift-prone sampling keys from a legacy overlay, warn once."""
    stripped: list[str] = []

    def _walk(node: dict[str, Any], path: str) -> None:
        for k, v in list(node.items()):
            here = f"{path}.{k}" if path else k
            if k in _LEGACY_STRIP_KEYS:
                del node[k]
                stripped.append(here)
            elif isinstance(v, dict):
                _walk(v, here)

    _walk(cfg, "")
    if stripped:
        _logger.warning(
            "legacy clients.yml as overlay: sampling keys ignored (catalog owns "
            "them; write a <box>/models.yml to own sampling deliberately): %s",
            ", ".join(sorted(stripped)),
        )
    return cfg


def _load_overlay() -> dict[str, Any]:
    """Load + cache the box overlay, reloading on mtime change.

    The overlay is editable without redeploy — cheap stat per resolve, re-read
    on mtime change (unchanged behavior inherited from the clients.yml loader).
    """
    global _overlay_cache, _overlay_mtime, _overlay_legacy
    located = _box_path()
    if located is None:
        return {}
    path, legacy = located
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}
    if _overlay_cache is not None and _overlay_mtime == mtime:
        return _overlay_cache
    if yaml is None:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if legacy:
            cfg = _sanitize_legacy(cfg)
        _overlay_cache = cfg
        _overlay_mtime = mtime
        _overlay_legacy = legacy
    except (OSError, yaml.YAMLError) as exc:
        _logger.warning("Failed to load overlay (%s): %s", path, exc)
        _overlay_cache = {}
    return _overlay_cache


def reload_config() -> None:
    """Force a reload of both registry + overlay caches."""
    global _registry_cache, _overlay_cache, _overlay_mtime, _overlay_legacy
    _registry_cache = None
    _overlay_cache = None
    _overlay_mtime = None
    _overlay_legacy = False


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


# bearer ref forms: credgoo:<service> | ${ENV}/$ENV | inline string
_ENV_REF_RE = re.compile(r"^\$\{?([A-Z_][A-Z0-9_]*)\}?$")


def _resolve_bearer(ref: str | None) -> str | None:
    """Resolve a bearer ref to the actual key: credgoo:svc / ${ENV} / inline.

    None/empty → None. credgoo caches lookups internally. A `credgoo:` ref when
    credgoo isn't installed, or a missing key, raises (fail loud — no silent
    anonymous calls).
    """
    if not ref:
        return None
    ref = ref.strip()
    if ref.startswith("credgoo:"):
        svc = ref[len("credgoo:"):].strip()
        if not _credgoo_get_api_key:
            raise RuntimeError(f"bearer ref {ref!r} needs credgoo, which isn't installed")
        key = _credgoo_get_api_key(svc)
        if not key:
            raise RuntimeError(f"credgoo returned no key for service {svc!r}")
        return key
    m = _ENV_REF_RE.match(ref)
    if m:
        return os.environ.get(m.group(1))
    return ref  # inline literal


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge overlay into base: dicts merge recursively (overlay wins),
    lists and scalars replace wholesale — a backup chain is replaced, never
    concatenated (contract: issue llminvoke-models-hierarchy)."""
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _chain(cfg: dict[str, Any], package: str | None, task: str | None) -> dict[str, Any]:
    """Flatten one file's default ← package ← task into a single dict
    (task most specific wins). The ``tasks`` container itself never leaks out."""
    out: dict[str, Any] = dict(cfg.get("default", {}))
    if package:
        pkg = cfg.get("packages", {}).get(package, {})
        out.update({k: v for k, v in pkg.items() if k != "tasks"})
        if task:
            out.update(pkg.get("tasks", {}).get(task, {}))
    return out


def resolve_model(
    package: str | None = None,
    client: str | None = None,
    task: str | None = None,
    env_prefix: str | None = None,
) -> ResolvedConfig:
    """Resolve the effective model config per the models.yml hierarchy.

    Layers (low → high): catalog (default/package/task) ← box overlay
    (default/package/task — the overlay beats the catalog at ANY level,
    deep-merged), then the caller's env prefix pins the primary on top.
    Team settings (DB) sit above the overlay in klark0's TS port; llminvoke
    itself sees files + env only.

    Args:
        package: package name (pdf2md, strukt2meta, ...) — applies package defaults.
        client: DEPRECATED — the per-client business layer (ADR 0004) died with
            the clients.yml; accepted for signature compatibility, ignored.
        task: task-type within a package (e.g. strukt2meta's ``kriterien``).

    Returns a DSGVO-filtered ``ResolvedConfig`` ready for the retry/backup loop.
    """
    catalog = _chain(_load_registry(), package, task)
    overlay = _chain(_load_overlay(), package, task)
    eff = _deep_merge(catalog, overlay)

    primary_spec = eff.get("model", "tu@qwen-3.6-35b")
    temperature = float(eff.get("temperature", 0.7))
    max_tokens = int(eff.get("max_tokens", 4096))
    retry = _parse_retry(eff.get("retry"), RetryPolicy())
    backups_raw: list[str] = list(eff.get("backups", []))
    dsgvo_required = bool(eff.get("dsgvo_required", False))

    # ── endpoint triple: env is the fallback, the overlay overrides ──
    base_url = eff.get("base_url") or (os.environ.get("OPENAI_BASE_URL", "").strip() or None)
    bare_model = bool(eff.get("bare_model", False))
    bearer_ref: str | None = eff.get("bearer") or (
        os.environ.get("OPENAI_API_KEY", "").strip() or None
    )
    thinking: str | bool | None = eff.get("thinking")     # off|on|none|low|medium|high (mapped per model at return)
    task_kwargs: dict = dict(eff.get("request_kwargs") or {})

    # ── build refs + DSGVO-filter backups ──
    primary = ModelRef.parse(primary_spec)
    backup_refs = [ModelRef.parse(s) for s in backups_raw]
    backup_refs = _filter_dsgvo(backup_refs, dsgvo_required)

    # ── env pin (highest): primary only — backups still flow from config so a
    # container pinning its primary doesn't lose fallback protection. ──
    if env_prefix:
        ep = os.environ.get(f"{env_prefix}_PROVIDER", "").strip()
        em = os.environ.get(f"{env_prefix}_MODEL", "").strip()
        if ep and em:
            primary = ModelRef(provider=ep, model=em)

    # ── map `thinking` → the provider's reasoning knob (per model) ──
    # qwen-3.x: chat_template_kwargs.enable_thinking (off=False). reasoning_effort
    # models (o-series-style): minimal/low/medium/high. 'on'/None = default.
    # 'off' AND 'none' both mean "disable reasoning" (uniinfer REASONING_OFF) and
    # both map to the chat-template knob — NOT reasoning_effort: aqueduct-backed
    # TU models 400 on reasoning_effort (UnsupportedParamsError, verified live
    # 2026-09-09), while chat_template_kwargs.enable_thinking=false is honored
    # by the same endpoint (verified same day).
    if thinking is not None:
        thinking = "on" if isinstance(thinking, bool) and thinking else \
                   "off" if isinstance(thinking, bool) else str(thinking).strip().lower()
        if thinking in ("off", "none"):
            ctk = dict(task_kwargs.get("chat_template_kwargs") or {})
            ctk.setdefault("enable_thinking", False)
            task_kwargs["chat_template_kwargs"] = ctk
        elif thinking in ("minimal", "low", "medium", "high"):
            task_kwargs["reasoning_effort"] = thinking

    return ResolvedConfig(
        primary=primary,
        backups=backup_refs,
        temperature=temperature,
        max_tokens=max_tokens,
        retry=retry,
        dsgvo_required=dsgvo_required,
        request_kwargs=task_kwargs,
        base_url=base_url,
        bare_model=bare_model,
        bearer=_resolve_bearer(bearer_ref),
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
