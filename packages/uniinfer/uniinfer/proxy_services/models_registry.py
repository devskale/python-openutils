import os
import sys
import json
import time
import fcntl
import asyncio
import logging
import dataclasses
from typing import Any
from contextlib import contextmanager

from starlette.concurrency import run_in_threadpool
import subprocess

logger = logging.getLogger("uniioai_proxy")

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREDEFINED_MODELS = [
    "mistral@mistral-tiny-latest",
    "ollama@qwen2.5:3b",
    "openrouter@google/gemma-3-12b-it:free",
    "arli@Mistral-Nemo-12B-Instruct-2407",
    "internlm@internlm3-latest",
    "stepfun@step-1-flash",
    "upstage@solar-mini-250401",
    "zai@glm-4.5-flash",
    "zai-code@glm-4.5",
    "ngc@google/gemma-3-27b-it",
    "cohere@command-r",
    "moonshot@kimi-latest",
    "groq@llama3-8b-8192",
    "gemini@models/gemma-3-27b-it",
    "chutes@Qwen/Qwen3-235B-A22B",
    "pollinations@grok",
]


def get_refetch_interval_seconds() -> float:
    raw = os.getenv("REFETCHTIME", "24")
    try:
        hours = float(raw)
        if hours <= 0:
            raise ValueError
    except ValueError:
        logger.warning("Invalid REFETCHTIME value '%s', defaulting to 24 hours", raw)
        hours = 24.0
    return hours * 3600.0


def models_file_is_stale() -> bool:
    models_json = os.path.join(PACKAGE_ROOT, "models", "models.json")
    if not os.path.exists(models_json):
        return True
    age_seconds = time.time() - os.path.getmtime(models_json)
    return age_seconds > get_refetch_interval_seconds()


_refresh_lock = asyncio.Lock()


async def refresh_models_file() -> dict[str, Any]:
    """Re-generate models.json (single-flight: only one refresh at a time).

    The lock stops concurrent /v1/models requests (and the manual refresh
    endpoint) from each spawning their own generate_models.py subprocess, which
    would stack memory on small hosts. generate_models.py also holds its own
    flock, so even an overlap with the systemd timer is a harmless no-op.
    """
    if _refresh_lock.locked():
        return {"status": "skipped", "message": "A models refresh is already in progress"}
    async with _refresh_lock:
        scripts_dir = os.path.join(PACKAGE_ROOT, "scripts")
        if not os.path.exists(scripts_dir):
            scripts_dir = os.path.join(os.path.dirname(PACKAGE_ROOT), "scripts")

        result = await run_in_threadpool(
            subprocess.run,
            [sys.executable, os.path.join(scripts_dir, "generate_models.py")],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            logger.error("generate_models.py failed: %s", result.stderr)
            return {"status": "error", "message": result.stderr}
        return {"status": "success", "message": "Models updated successfully"}


async def ensure_fresh_models_file() -> None:
    if not models_file_is_stale():
        return
    if _refresh_lock.locked():
        # A refresh is already running (on-demand or timer); don't stack spawns.
        return
    try:
        await refresh_models_file()
    except Exception as e:
        logger.warning("Failed to refresh stale models file: %s", e)


class Catalog:
    """Owns the models.json catalog and its sidecar files.

    Single writer for models.json, model_overrides.json, type_overrides.json,
    and _model_history.json. All writes are atomic (tmp + rename) and
    serialized via flock(LOCK_EX) to prevent lost updates when the generator
    (full rebuild), the live-fetch path (upsert_provider), and the pi-import
    path (backfill_fields) run concurrently.

    Reads (read_nested, list_resolved) intentionally skip the lock: they read
    the atomically-renamed file, which is always internally consistent.

    Construct with a custom ``path`` only for tests; production callers should
    use the default (PACKAGE_ROOT/models/models.json).
    """

    def __init__(self, path: str | None = None) -> None:
        self._path = path or os.path.join(PACKAGE_ROOT, "models", "models.json")
        self._dir = os.path.dirname(self._path)

    @property
    def overrides_path(self) -> str:
        return os.path.join(self._dir, "model_overrides.json")

    @property
    def _type_overrides_path(self) -> str:
        return os.path.join(self._dir, "type_overrides.json")

    @property
    def _history_path(self) -> str:
        return os.path.join(self._dir, "_model_history.json")

    @property
    def _stale_path(self) -> str:
        return os.path.join(self._dir, "_stale_models.json")

    # ------------------------------------------------------------------ #
    # internal loaders                                                   #
    # ------------------------------------------------------------------ #

    def _load(self) -> dict | None:
        """Load models.json; return None if missing or corrupt."""
        if not os.path.exists(self._path):
            return None
        try:
            with open(self._path) as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error reading models.json: %s", e)
            return None

    def _load_overrides(self) -> dict:
        """Load model_overrides.json (full doc shape: {_meta?, models})."""
        if not os.path.exists(self.overrides_path):
            return {"models": {}}
        try:
            with open(self.overrides_path) as f:
                return json.load(f)
        except Exception:
            return {"models": {}}

    def _load_type_overrides_doc(self) -> dict:
        """Load type_overrides.json as the full {_meta, models} document."""
        if not os.path.exists(self._type_overrides_path):
            return {
                "_meta": {"description": "Curated model type assignments."},
                "models": {},
            }
        try:
            with open(self._type_overrides_path) as f:
                return json.load(f)
        except Exception:
            return {
                "_meta": {"description": "Curated model type assignments."},
                "models": {},
            }

    def _load_history(self) -> dict[str, dict]:
        """Load _model_history.json, migrating legacy string values to objects."""
        if not os.path.exists(self._history_path):
            return {}
        try:
            with open(self._history_path) as f:
                raw = json.load(f)
        except Exception:
            return {}
        migrated: dict[str, dict] = {}
        for key, val in raw.items():
            if isinstance(val, str):
                migrated[key] = {"first_seen": val, "last_seen": val}
            elif isinstance(val, dict) and "first_seen" in val:
                if "last_seen" not in val:
                    val["last_seen"] = val["first_seen"]
                migrated[key] = val
        return migrated

    # ------------------------------------------------------------------ #
    # internal writers                                                   #
    # ------------------------------------------------------------------ #

    def _atomic_write(
        self, path: str, data, indent: int | None = None, sort_keys: bool = False
    ) -> None:
        """Write JSON atomically: stage to <path>.tmp, then os.replace."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        kwargs: dict[str, Any] = {"ensure_ascii": False}
        if indent:
            kwargs["indent"] = indent
        else:
            kwargs["separators"] = (",", ":")
        if sort_keys:
            kwargs["sort_keys"] = True
        with open(tmp, "w") as f:
            json.dump(data, f, **kwargs)
        os.replace(tmp, path)

    @contextmanager
    def _write_lock(self):
        """flock(LOCK_EX) around read-modify-write to serialize concurrent writers."""
        lock_path = os.path.join(self._dir, ".catalog.lock")
        f = open(lock_path, "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()

    # ------------------------------------------------------------------ #
    # write operations                                                   #
    # ------------------------------------------------------------------ #

    def write_all(self, catalog: dict) -> None:
        """Full rebuild: replace models.json with a freshly-built catalog dict.

        The generator calls this after assembling the complete catalog. The
        flock prevents a concurrent upsert_provider / backfill_fields from
        overwriting the full rebuild with a stale single-provider snapshot.
        """
        with self._write_lock():
            self._atomic_write(self._path, catalog)

    def upsert_provider(self, provider_id: str, models_list) -> None:
        """Read-modify-write a single provider's models into models.json.

        Called after a live API fetch so subsequent /v1/models calls get fresh
        data for that provider without re-hitting the API. Preserves
        first_seen/last_seen from existing cache entries and _model_history.json.
        """
        from datetime import datetime, timezone

        from uniinfer.core import ModelInfo

        if not models_list:
            return

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with self._write_lock():
            data = self._load() or {}
            history = self._load_history()

            existing_models: dict[str, dict] = {}
            for m in data.get("providers", {}).get(provider_id, {}).get("models", []):
                existing_models[m["id"]] = m

            # Serialize ModelInfo → dict (skip None fields and raw payload).
            model_dicts = []
            for m in models_list:
                if isinstance(m, ModelInfo):
                    d = {"id": m.id}
                    for field in dataclasses.fields(m):
                        val = getattr(m, field.name)
                        if val is None or field.name in ("id", "raw"):
                            continue
                        d[field.name] = val
                    model_dicts.append(d)
                else:
                    model_dicts.append({"id": str(m)})

            # Preserve first_seen (existing cache → history → today).
            for m in model_dicts:
                mid = m["id"]
                old = existing_models.get(mid, {})
                if old.get("first_seen"):
                    m["first_seen"] = old["first_seen"]
                else:
                    hentry = history.get(f"{provider_id}/{mid}")
                    if hentry:
                        m["first_seen"] = hentry["first_seen"]
                    else:
                        m["first_seen"] = today
                m["last_seen"] = today

            data.setdefault("_meta", {})
            data.setdefault("providers", {})
            data["_meta"]["generated"] = datetime.now(timezone.utc).isoformat()
            data["providers"][provider_id] = {
                "provider_class": "",
                "kind": "chat",
                "models": model_dicts,
            }

            total = sum(len(p.get("models", [])) for p in data["providers"].values())
            data["_meta"]["total_models"] = total
            data["_meta"]["total_providers"] = len(data["providers"])

            self._atomic_write(self._path, data)

        logger.info(
            "Updated cache for provider %s: %d models", provider_id, len(model_dicts)
        )

    def backfill_fields(self, entries: list[tuple[str, str, dict]]) -> dict:
        """Read-modify-write: fill metadata gaps from pi imports.

        ``entries`` is a list of ``(provider_id, model_id, meta_dict)``. Only
        gaps are filled — existing values are never overwritten. Writes both
        model_overrides.json (persistent runtime merge) and models.json
        (immediate visibility).

        Returns ``{backfilled, skipped}``.
        """
        with self._write_lock():
            data = self._load() or {"providers": {}}
            overrides = self._load_overrides()
            overrides.setdefault(
                "_meta",
                {
                    "description": (
                        "Curated metadata (context, modalities, capabilities). "
                        "Merged at runtime into models.json. "
                        "Sourced from pi's models.json."
                    )
                },
            )

            backfilled = 0
            skipped = 0
            for prov, model_id, meta in entries:
                cat_models = {
                    m.get("id"): m
                    for m in (
                        data.get("providers", {}).get(prov, {}).get("models") or []
                    )
                }
                cm = cat_models.get(model_id)
                applied = False
                if cm is not None:
                    for k, v in meta.items():
                        if not cm.get(k):
                            cm[k] = v
                            applied = True
                ov = overrides.setdefault("models", {}).setdefault(model_id, {})
                for k, v in meta.items():
                    ov.setdefault(k, v)
                if applied:
                    backfilled += 1
                else:
                    skipped += 1

            self._atomic_write(self.overrides_path, overrides, indent=2)
            if data.get("providers"):
                self._atomic_write(self._path, data)

        return {"backfilled": backfilled, "skipped": skipped}

    # ------------------------------------------------------------------ #
    # read operations                                                    #
    # ------------------------------------------------------------------ #

    def read_nested(self, provider_filter: str | None = None) -> dict:
        """Load the raw nested catalog, optionally filtered by provider(s).

        Returns the native nested shape:
        ``{"_meta": {...}, "providers": {pid: {provider_class, kind, models}}}``.
        Meta totals are recomputed to reflect the filtered subset.
        """
        data = self._load()
        if data is None:
            return {
                "_meta": {"generated": None, "total_models": 0, "total_providers": 0},
                "providers": {},
            }

        providers = data.get("providers", {})
        if provider_filter:
            wanted = {p.strip() for p in provider_filter.split(",") if p.strip()}
            providers = {k: v for k, v in providers.items() if k in wanted}

        total = sum(len(p.get("models", [])) for p in providers.values())
        meta = dict(data.get("_meta", {}))
        meta["total_models"] = total
        meta["total_providers"] = len(providers)
        meta["filtered"] = bool(provider_filter)
        return {"_meta": meta, "providers": providers}

    def list_resolved(self) -> list[dict]:
        """Build a flat OpenAI-compatible model list from models.json.

        Each model gets a resolved ``type`` (override > type_overrides > stored >
        derive) and a ``freshness`` field ('fresh' if last_seen == generated
        date, else 'stale' with days_since_seen).
        """
        from datetime import datetime

        from uniinfer.core import ModelInfo

        data = self._load()
        if data is None:
            return [
                {"id": m, "object": "model", "owned_by": "skaledev"}
                for m in PREDEFINED_MODELS
            ]

        type_overrides = self._load_type_overrides_doc().get("models", {})
        model_overrides = self._load_overrides().get("models", {})

        generated_date = data.get("_meta", {}).get("generated", "")[:10]
        result: list[dict] = []
        for provider_id, provider_data in data.get("providers", {}).items():
            for model in provider_data.get("models", []):
                override = model_overrides.get(model["id"], {})
                entry = {
                    "id": model["id"],
                    "object": "model",
                    "owned_by": model.get("owned_by", "skaledev"),
                    "provider": provider_id,
                }
                # Resolve type: overrides > type_overrides > stored > derive.
                model_type = None
                if override.get("type"):
                    model_type = override["type"]
                elif model["id"] in type_overrides:
                    model_type = type_overrides[model["id"]]
                elif model.get("type") and model["type"] != "chat":
                    model_type = model["type"]
                if not model_type:
                    mi = ModelInfo(id=model["id"], modalities=model.get("modalities"))
                    model_type = mi.derive_type()
                entry["type"] = model_type
                # Fields: override > models.json > skip.
                for field in (
                    "context_window", "max_output", "dimensions", "cost",
                    "capabilities", "modalities", "first_seen",
                    "deprecation_date", "deprecation_replacement", "status",
                    "release_date", "knowledge_cutoff", "name", "speed",
                    "access",
                ):
                    val = override.get(field) if field in override else model.get(field)
                    if val is not None:
                        entry[field] = val
                # Freshness: last_seen vs models.json generated date.
                last_seen = model.get("last_seen") or generated_date
                entry["last_seen"] = last_seen
                if last_seen == generated_date:
                    entry["freshness"] = "fresh"
                    entry["days_since_seen"] = 0
                else:
                    try:
                        days = (
                            datetime.strptime(generated_date, "%Y-%m-%d")
                            - datetime.strptime(last_seen, "%Y-%m-%d")
                        ).days
                    except Exception:
                        days = 0
                    entry["freshness"] = "stale"
                    entry["days_since_seen"] = days
                result.append(entry)
        return result

    # ------------------------------------------------------------------ #
    # history                                                            #
    # ------------------------------------------------------------------ #

    def load_history(self) -> dict[str, dict]:
        """Load _model_history.json (migrated to object format)."""
        return self._load_history()

    def save_history(self, history: dict[str, dict]) -> None:
        """Persist _model_history.json (atomic, sorted keys)."""
        with self._write_lock():
            self._atomic_write(
                self._history_path, history, indent=2, sort_keys=True
            )

    # ------------------------------------------------------------------ #
    # overrides CRUD                                                     #
    # ------------------------------------------------------------------ #

    def read_overrides(self) -> dict:
        """Load model_overrides.json (full doc: {_meta?, models})."""
        return self._load_overrides()

    def save_override(self, model_id: str, fields: dict) -> None:
        """Save fields for a model into model_overrides.json.

        Also updates type_overrides.json if 'type' is among the fields.
        """
        from datetime import datetime, timezone

        with self._write_lock():
            data = self._load_overrides()
            data.setdefault("models", {}).setdefault(model_id, {}).update(fields)
            data["models"][model_id]["_updated"] = datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            self._atomic_write(self.overrides_path, data, indent=2)

            if "type" in fields:
                td = self._load_type_overrides_doc()
                td["models"][model_id] = fields["type"]
                self._atomic_write(self._type_overrides_path, td, indent=2)

        logger.info("Saved override for %s: %s", model_id, list(fields.keys()))

    def delete_override(self, model_id: str) -> bool:
        """Delete all overrides for a model. Returns True if anything was deleted."""
        with self._write_lock():
            data = self._load_overrides()
            if model_id not in data.get("models", {}):
                return False
            del data["models"][model_id]
            self._atomic_write(self.overrides_path, data, indent=2)
        return True

    # ------------------------------------------------------------------ #
    # stale models (generator-owned file, read-only here)                #
    # ------------------------------------------------------------------ #

    def read_stale_models(self) -> list[dict]:
        """Load stale models from _stale_models.json (written by generate_models.py)."""
        if not os.path.exists(self._stale_path):
            return []
        try:
            with open(self._stale_path) as f:
                return json.load(f)
        except Exception:
            return []


def parse_models_file() -> list[str]:
    """Deprecated: models.txt is no longer used. Returns predefined fallback."""
    return PREDEFINED_MODELS
