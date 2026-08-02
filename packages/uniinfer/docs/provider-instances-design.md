# Provider Instances — named, configurable fleet endpoints

Status: **design (grilled & locked)** · Branch: `feat/uniinfer-named-instances`

One declarative, gitignored file drives named instances for *any* provider type:
multiple Ollama endpoints, a fleet of OpenAI-compatible/vLLM servers, combined
Z.AI, etc. The class registry stays the source of truth for *what exists*; the
file is an **overlay** of `enabled` flags + per-instance overrides + custom
aliases.

## Addressing

- Bare alias, split on first `@`: `ollama-home@gemma3:1b`, `vllm-local@Qwen3-35B`,
  `zai-local@glm-5.2`. An alias **is** a provider name to the registry.

## Config file

- Path: `UNIINFER_INSTANCES_FILE` env, else `<cwd>/provider_instances.json`
  (mirrors the `.env` convention). Gitignored; ship a `*.example`.
- Holds **no raw keys** — references credgoo *service names*.

Schema (all fields optional except `provider` on a custom alias):

```jsonc
{
  "ollama-home": { "provider": "ollama", "base_url": "http://localhost:11434" },
  "ollama-amp":  { "provider": "ollama", "base_url": "https://amp1.mooo.com:11444", "enabled": true },
  "vllm-local":  { "provider": "openai-compat", "base_url": "http://localhost:8000/v1", "requires_api_key": false },
  "vllm-prod":   { "provider": "openai-compat", "base_url": "https://vllm.internal/v1", "credgoo_service": "vllm-prod" },
  "zai-local":   { "provider": "zai", "base_url": "http://localhost:9000/v4" },
  "ollama":      { "base_url": "https://amp1.mooo.com:11444" }   // overrides built-in
}
```

Field defaults (when omitted): inherit the underlying class attrs
(`BASE_URL`, `CREDGOO_SERVICE`, `REQUIRES_API_KEY`) / `PROVIDER_CONFIGS`.

## Overlay semantics

- Registry = source of truth for *discovery* → new-release providers propagate
  automatically (no drift, no merge chore).
- File = `enabled` flags + overrides + custom aliases. Override-on-collision
  (a file entry named like a built-in overrides it; e.g. redefine `ollama`'s
  base_url). Replaces today's `extra_params` hack + `provider_name == "ollama"`
  special-cases.
- Out-of-box (no file): all built-ins enabled; ollama defaults to
  `localhost:11434` (the sane universal default; your file carries amp1).
- Errors: **malformed JSON / unregistered `provider` ref → raise at boot**
  (fail fast). Hot-reload parse failure → **keep last-good + warn** (never take
  the proxy down).

## Key resolution (per instance)

- `credgoo_service`: per-instance; default = the alias name for custom
  instances (each fleet member its own key namespace), overridable to share.
  Resolution order: instance field → class `CREDGOO_SERVICE` → alias name.
- `requires_api_key`: per-instance; default = class `REQUIRES_API_KEY`.
  Anonymous instances (local vLLM, LM Studio, Ollama) declare `false`; the
  proxy's auth-bypass reads this instead of a hardcoded `ollama` check.

## Generic class

- New `OpenAICompatProvider` (subclass of `OpenAICompatibleChatProvider`):
  identity (`BASE_URL`/`PROVIDER_ID`/`CREDGOO_SERVICE`) comes **entirely from
  the instance config** — no hardcoded values. Registered under `openai-compat`.
  `list_models()` hits `{base_url}/models` (vLLM / Ollama-compat / LM Studio /
  any `/v1`). Fleet members point `provider` at it.
- Concrete subclasses (`groq`, `kilo`, `openrouter`, `mistral`, …) stay — they
  encode provider-specific *reasoning dialects* the generic deliberately omits.

## Reload (runtime)

- mtime-check per resolution: one `os.stat`; re-parse + re-validate if newer,
  then swap. Graceful-degrade on reload error (keep last-good, log loudly).

## Model listing (per instance)

- Catalog keyed by **alias** (`models.json["ollama-home"]` ≠ `["ollama-amp"]`).
- `/v1/models/{alias}` resolves via the shared `resolve_instance()` and lists
  through the underlying class's `list_models` with the instance's base_url/key.
  The raw list then passes through **`Catalog.resolve_for_instance(alias, models)`**
  — provider-level + per-model overrides (keyed by the instance's *underlying
  provider*, not the alias), then the instance's selective/only access filter.
  Both the live path and the SWR-cached alias response route through it, so they
  cannot drift (the cached path previously filtered but skipped overrides).
- Disabled aliases skipped everywhere.
- **Hybrid refresh:** built-ins on the daily timer (unchanged); custom aliases
  lazy + stale-while-revalidate (reuse `ensure_fresh_models_file` machinery),
  driven by a per-alias `last_refreshed` timestamp.

## CLI (flat-flag CRUD family, consistent with existing `--list-providers`)

- `uniinfer init` — write a commented full-registry template (create-if-missing).
- `--add-provider <alias> --provider <class> --base-url … [--key sk-…] [--credgoo-service …] [--no-key] [--no-verify] [--keytype LABEL] [--only TIER]`
  - **Probe by default**: `{base_url}/models` → infer type, verify reachability,
    auto-detect anonymous (`requires_api_key: false`), pre-fetch model list.
    Warn-not-block on failure; `--no-verify` for headless/CI.
  - `--key` stores via credgoo under the alias (overridable); explicit `--key`
    bypasses credgoo's interactive `_confirm`. Prompt-fallback honors it.
  - `--keytype` / `--only` stamp the instance's `access` (keytype label + tier
    filter free/granted/paid) at creation.
- `--remove-provider <alias>` — **smart safe-remove**: custom aliases delete
  (confirm + `--force`); built-in names refuse and steer to `--disable-provider`
  / `--reset-provider` (never a half-registered state).
- `--enable-provider` / `--disable-provider <alias>` — toggle.
- `--reset-provider <alias>` — drop overrides, revert to registry defaults.
- `--tag <alias> --keytype LABEL --only TIER` — tag an existing instance's
  access (incl. built-ins via overlay); merges, so partial updates don't clobber.
- `--show-provider <alias>` (prints the resolved access tags too).
- TUI deferred.

## Embeddings

In scope — `resolve_instance()` drives `EmbeddingProviderFactory` too
(`ollama-home@nomic-embed-text`).

## The deep seam (the "testable design advancement")

```
resolve_instance(alias) -> InstanceSpec(alias, provider, base_url,
                                        credgoo_service, requires_api_key,
                                        enabled, default_model, is_builtin)
```

One pure, unit-testable function owns alias → resolved spec. Used by completion
(`Target`), embeddings, the models router, the CLI, and key resolution —
replacing the scattered `extra_params`/special-case logic. The mtime-reload
loader wraps `load_instances()` and is independently unit-testable.

## Phasing (vertical slice first, on the real seams)

- **L1 core:** instances loader + mtime-reload + `resolve_instance()` + factory
  aliasing + generic `OpenAICompatProvider` + Target/provider_access seam +
  delete ollama special-cases → *instances route at runtime.*
- **L2 listing:** catalog keyed by alias + `/v1/models/{alias}` + hybrid refresh.
- **L3 CLI:** `init` + smart add/remove + enable/disable/reset/show.
- First slice: L1 + minimal CLI (add/list/remove with probe+write), end-to-end
  testable live; then layer L2 + rest of L3. Tests first at each step.

## Key management & accessibility (Phase 2.5 — visibility)

Each instance carries an **open `access` dict** (forward-adoptable — new access
semantics are new keys, no schema migration):

```jsonc
"access": {
  "keytype": "GLM Coding Pro",     // free-form label (any name)
  "models": ["glm-4.5-flash"]      // OPTIONAL — present = selective access
}
```

- **`keytype`** — free-form display label ("GLM Coding Pro", "Trial", "Free
  tier", "anonymous"…). Not an enum.
- **`models`** — *optional* selective spec. Present (a list) → the instance can
  only reach those models. Absent → all of the provider's models. (Filter
  objects like `{"match":"free"}` / `{"access":"free"}` are reserved for later —
  the open dict absorbs them without a migration.)
- **free / paid** — stays **per-model** (`ModelInfo.access`), surfaced per
  instance as a `free/all/paid` count on the Dashboard fleet panel.

### First-cut tags (the "few important ones")

```jsonc
"zai-code":  { "provider": "zai-code", "credgoo_service": "zai-code",
               "access": { "keytype": "GLM Coding Pro" } },
"zai-trial": { "provider": "zai", "credgoo_service": "zai-trial",
               "access": { "keytype": "Trial", "models": ["glm-4.5-flash"] } },
"opencode":  { "provider": "opencode", "access": { "keytype": "Paid + free models" } },
"arli":      { "provider": "arli",     "access": { "keytype": "Free tier" } },
"kilo":      { "provider": "kilo",     "access": { "keytype": "Free access" } },
"pollinations": { "provider": "pollinations", "access": { "keytype": "anonymous" } }
```

Each provider's `free/all/paid` split is computed from the catalog
(`ModelInfo.access`), gated by the instance's selective `models` list when set.

### What this is NOT (yet)

- **No routing / auto-selection** — addressing `zai@model` still resolves one
  instance explicitly. Free-preference routing + fallback are a later layer.
- **No budgets / per-period metering** — `access` can already *store*
  `budget`/`period`/`ratelimit` (forward-adoptable), but nothing acts on them
  yet. Budget tracking is a future layer.

The dashboard fleet panel is the first place this surfaces: `keytype` +
`free/all/paid` + `scope` per instance.
