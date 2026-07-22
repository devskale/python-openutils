"""Export accessible uniinfer models into a pi-compatible ``models.json`` entry.

The uniioai proxy is a *metaprovider* in pi terms: one ``baseUrl``/``apiKey``
pointing at the proxy, routing to many models via the ``provider@model`` id
notation (pi's id is what the proxy splits on the first ``@``).

This module builds such an entry from the catalog, mapping ``ModelInfo``
metadata onto pi's model schema, and merges it into ``~/.pi/agent/models.json``
(with a backup).

pi model schema (the fields pi reads):

    {
      "id": "tu@qwen-3.6-35b",            # provider@model — proxy routing
      "name": "Qwen 3.6 35B (TU)",         # human-readable
      "reasoning": true,                   # supports thinking/reasoning
      "input": ["text", "image"],          # modalities
      "contextWindow": 131072,             # input context
      "maxTokens": 32768,                  # max output tokens
      "compat": {"maxTokensField": "max_tokens"},
      "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    }
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_PI_MODELS_JSON = Path.home() / ".pi" / "agent" / "models.json"

# pi provider name -> uniinfer provider id (for models.json backfill).
# Entries whose id contains '@' carry the provider inline and don't need this.
_PI_PROVIDER_MAP = {
    "zen.fg": "opencode",
    "opencode": "opencode",
    "tu-aqueduct": "tu",
    "zai": "zai",
    "uart": "ollama",
}


def parse_pi_entries(pi_data: dict) -> list[tuple[str, str, dict]]:
    """Extract (uniinfer_provider, model_id, metadata) from pi models.json data.

    metadata carries the curated fields pi has: contextWindow, input modalities,
    reasoning. The provider comes from the '@' in the id if present, else from
    the pi provider name via ``_PI_PROVIDER_MAP``.
    """
    out = []
    for pname, pdata in (pi_data.get("providers") or {}).items():
        for m in pdata.get("models") or []:
            mid = m.get("id", "")
            if "@" in mid:
                prov, model_id = mid.split("@", 1)
            else:
                prov = _PI_PROVIDER_MAP.get(pname)
                model_id = mid
            if not prov:
                continue
            caps = {}
            if m.get("reasoning"):
                caps["reasoning"] = True
            if "image" in (m.get("input") or []):
                caps["vision"] = True
            meta = {}
            if m.get("contextWindow"):
                meta["context_window"] = m["contextWindow"]
            if caps:
                meta["capabilities"] = caps
            inp = m.get("input")
            if inp:
                meta["modalities"] = {"input": inp, "output": ["text"]}
            if meta:
                out.append((prov, model_id, meta))
    return out


def import_pi_metadata(
    pi_path: Path = DEFAULT_PI_MODELS_JSON,
    catalog_path: Optional[Path] = None,
) -> dict:
    """Import curated metadata from pi's models.json into the catalog.

    Writes a ``model_overrides.json`` (keyed by model id) that the runtime merge
    already applies, AND backfills the live ``models.json`` immediately so the
    change is visible without a full regeneration.

    Returns a summary dict: {entries, backfilled, skipped, overrides_path}.
    """
    from uniinfer.proxy_services.models_registry import Catalog

    catalog = Catalog(path=str(catalog_path) if catalog_path else None)

    pi_data = json.loads(pi_path.read_text())
    entries = parse_pi_entries(pi_data)
    result = catalog.backfill_fields(entries)
    return {
        "entries": len(entries),
        "backfilled": result["backfilled"],
        "skipped": result["skipped"],
        "overrides_path": str(catalog.overrides_path),
    }

# Reasoning-model heuristic: ids/names that indicate a thinking/reasoning model.
# Used when capabilities.reasoning isn't set explicitly. Conservative — only
# high-confidence patterns.
_REASONING_PATTERNS = re.compile(
    r"(?:^|[-/_])(?:r1|o1|o3|thinking|reason|kimi-k|kimi-k2|kimi-k3|qwen3|"
    r"glm-5|glm-4\.5-flash|glm-4\.7-flash|nemotron|deepseek-r|"
    r"step-3\.7|reasoning)(?:$|[-/_0-9.])",
    re.IGNORECASE,
)

# Default max output tokens when the catalog doesn't carry it. Generous for
# reasoning models (they burn budget on thinking) — matches the proxy default.
_DEFAULT_MAX_TOKENS = 32768


@dataclass
class PiModel:
    """A pi model entry ready for serialization."""

    id: str
    name: str
    reasoning: bool
    input: list[str]
    context_window: int
    max_tokens: int
    cost: dict
    compat: dict

    def to_dict(self) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "reasoning": self.reasoning,
            "input": self.input,
            "contextWindow": self.context_window,
            "maxTokens": self.max_tokens,
            "cost": self.cost,
            "compat": self.compat,
        }
        return d


def _is_reasoning(model_id: str, capabilities: Optional[dict]) -> bool:
    """Determine if a model supports reasoning/thinking."""
    if capabilities and capabilities.get("reasoning"):
        return True
    return bool(_REASONING_PATTERNS.search(model_id))


def _modalities(model: dict) -> list[str]:
    """Extract input modalities (text/image) from a catalog model dict."""
    mods = (model.get("modalities") or {}).get("input") or ["text"]
    caps = model.get("capabilities") or {}
    out = ["text"]
    if "image" in mods or caps.get("vision"):
        out.append("image")
    return out


def _pi_cost(cost: Optional[dict]) -> dict:
    """Map a catalog cost dict onto pi's {input, output, cacheRead, cacheWrite}.

    Catalog cost is per-million tokens (openrouter/kilo convention). pi wants
    per-token rates. Free models → all zeros.
    """
    zero = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    if not cost:
        return zero
    inp = cost.get("input")
    if inp is None:
        return zero
    if inp == 0:
        return zero
    # Convert per-million → per-token.
    return {
        "input": inp / 1_000_000,
        "output": (cost.get("output") or 0) / 1_000_000,
        "cacheRead": (cost.get("cache_read") or 0) / 1_000_000,
        "cacheWrite": (cost.get("cache_write") or 0) / 1_000_000,
    }


def _prettify(provider: str, model_id: str) -> str:
    """Build a human-readable model name from provider + id."""
    # Use the leaf of a slashed id (openrouter's 'vendor/model') and title-case.
    leaf = model_id.split("/")[-1]
    # Strip common free/trial suffixes for the display name.
    leaf = re.sub(r":free$|-free$|-free$", "", leaf, flags=re.IGNORECASE)
    leaf = re.sub(r"[-_]", " ", leaf).strip()
    leaf = " ".join(w.capitalize() if w.islower() else w for w in leaf.split())
    prov = provider.upper() if len(provider) <= 4 else provider.capitalize()
    return f"{leaf} ({prov})"


def catalog_model_to_pi(provider: str, model: dict) -> PiModel:
    """Convert one catalog model dict (nested under a provider) to a PiModel."""
    mid = model.get("id", "")
    caps = model.get("capabilities")
    ctx = model.get("context_window") or model.get("contextWindow")
    return PiModel(
        id=f"{provider}@{mid}",
        name=model.get("name") or _prettify(provider, mid),
        reasoning=_is_reasoning(mid, caps),
        input=_modalities(model),
        context_window=ctx if isinstance(ctx, int) and ctx > 0 else 131072,
        max_tokens=_DEFAULT_MAX_TOKENS,
        cost=_pi_cost(model.get("cost")),
        compat={"maxTokensField": "max_tokens"},
    )


def accessible_models(
    catalog: dict,
    *,
    access_filter: str = "free",
    providers: Optional[list[str]] = None,
) -> list[tuple[str, dict]]:
    """Return (provider, model) pairs from the catalog that are accessible.

    Args:
        catalog: nested catalog dict (from ``Catalog().read_nested()``).
        access_filter: "free" (only free/usable models), "all" (everything),
            or "" (only models with no access info).
        providers: restrict to these provider ids; None = all.

    A model counts as free if any of:
        - its ``access`` field is "free" (explicitly stamped), or
        - its ``access`` is unstamped AND ``cost.input == 0`` (free by pricing —
          catches openrouter/kilo/ngc/opencode even before the access sweep), or
        - its provider is universally free (groq, tu, pollinations, …) with no
          pricing data.
    """
    from .keys import _UNIVERSALLY_FREE

    out = []
    for pid, pdata in (catalog.get("providers") or {}).items():
        if providers and pid not in providers:
            continue
        universally_free = pid in _UNIVERSALLY_FREE
        for m in pdata.get("models") or []:
            acc = m.get("access", "")
            cost = m.get("cost") or {}
            is_free = (
                acc == "free"
                or (not acc and cost.get("input") == 0)
                or (not acc and universally_free and not cost)
            )
            if access_filter == "free" and not is_free:
                continue
            if access_filter == "" and acc:
                continue
            out.append((pid, m))
    return out


def build_provider_entry(
    name: str, base_url: str, api_key: str, models: list[PiModel]
) -> dict:
    """Build a pi provider entry (the metaprovider block)."""
    return {
        "baseUrl": base_url.rstrip("/"),
        "apiKey": api_key,
        "api": "openai-completions",
        "authHeader": True,
        "models": [m.to_dict() for m in models],
    }


def merge_into_pi_models_json(
    entry: dict,
    provider_name: str,
    path: Path = DEFAULT_PI_MODELS_JSON,
) -> Path:
    """Merge a provider entry into pi's models.json, backing up first.

    Returns the backup path. Creates the file if it doesn't exist.
    """
    data = {"providers": {}}
    if path.exists():
        backup = path.with_suffix(".json.bak")
        shutil.copy2(path, backup)
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {"providers": {}}
    data.setdefault("providers", {})[provider_name] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    return path.with_suffix(".json.bak") if path.exists() else path


def parse_selection(spec: str, n: int) -> list[int]:
    """Parse a selection spec like '1,3-5,9' or 'all' into 0-based indices.

    Args:
        spec: comma/range string, or 'all'.
        n: total number of items.
    """
    spec = spec.strip().lower()
    if spec in ("all", "a", "*"):
        return list(range(n))
    idx = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                for i in range(int(lo), int(hi) + 1):
                    if 1 <= i <= n:
                        idx.add(i - 1)
            except ValueError:
                continue
        elif part.isdigit():
            i = int(part)
            if 1 <= i <= n:
                idx.add(i - 1)
    return sorted(idx)
