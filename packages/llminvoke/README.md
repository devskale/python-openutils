# llminvoke

Shared LLM invocation layer for kontext.one. Bridges uniinfer (inference) + credgoo (API keys) behind a config-aware interface with retry, backup chains, and DSGVO enforcement.

## Quick start

```python
from llminvoke import call_llm, stream_llm, resolve_model

# Recommended: resolve from package (catalog + clients.yml + env)
text = call_llm("Summarize this", package="pdf2md")

# Streaming (before-first-token retry + backup; after first token, no backup)
for chunk in stream_llm("Write a haiku", package="md2blank"):
    print(chunk, end="")

# Explicit model/provider (backward compat)
text = call_llm("prompt", model="qwen-3.6-35b", provider="tu")

# Pre-resolve and inspect
cfg = resolve_model(package="strukt2meta", task="kriterien")
print(cfg.primary, cfg.backups, cfg.temperature, cfg.retry)
text = call_llm("prompt", config=cfg)
```

## Three invocation levels

| function | returns | retry | backup | use when |
|---|---|---|---|---|
| `call_llm` | `str` | ✅ backoff | ✅ chain | **default** — most callers |
| `stream_llm` | `Iterator[str]` | before first token | before first token | streaming output |
| `invoke_llm` | raw response | ❌ | ❌ | escape hatch (agentos chain/breaker) |
| `create_provider` | `ChatProvider` | — | — | raw provider (tool-calling, custom loops) |

## Config resolution (ADR 0004)

Model config — which model, params, backups, retry, DSGVO — resolves through `resolve_model` per a strict precedence chain:

```
env var  >  team-settings (DB)  >  clients.yml (runtime)  >  catalog default
```

- **`models.yml`** (ships with the package): the catalog — providers (DSGVO-flagged), models (context windows, capabilities), global default profile, per-package/task engineering defaults.
- **`clients.yml`** (runtime, server-side): client→provider mapping. Editable without redeploy. The backend-only source.
- **Team settings** (klark0 DB): override for app-driven jobs.
- **Env**: pins primary only (`PDF2MD_VLM_MODEL` etc.); backups still flow from config.

```python
cfg = resolve_model(package="pdf2md", client="default", task="assess")
# → ResolvedConfig(
#     primary=ModelRef("tu", "qwen-3.6-35b"),
#     backups=[...],               # DSGVO-filtered
#     temperature=0.2,
#     max_tokens=4096,
#     retry=RetryPolicy(attempts=3, backoff="exponential", ...),
#     dsgvo_required=False,
#   )
```

### DSGVO enforcement

A client declaring `dsgvo_required: true` gets its backup chain filtered at resolve time — non-DSGVO providers are silently dropped. The loop never sees them.

### Backend-only operation

CLI/backend-only runs pick their client via `KONTEXT_CLIENT` env (default `"default"`). No app/DB required.

## Retry + fail-over model

```
transient (429/timeout/network) → retry same model with backoff
                                   3 attempts · exponential 2s→4s→8s · cap 30s · honor Retry-After
                                 → exhausted → escalate to backup
permanent (auth/context-exceeded) → escalate immediately
fail_fast=True (opt-in param)     → skip retry, immediate escalation
empty response                    → treated as failure → backup
```

429 means "slow down" — retry with delay (lowers rate), not "switch models".

## Alarms

Empty responses and hard failures emit structured alarms (`emit_alarm`) — the worker surfaces recent alarms via an endpoint for healthcheck/UI.

## Migration status

| package | resolution | notes |
|---|---|---|
| **pdf2md** | `call_llm(package="pdf2md", env_prefix="PDF2MD_VLM")` | assess, vlm, nail_vlm, visual_verify all migrated |
| **strukt2meta** | `call_llm(package="strukt2meta", task=task_type)` | manual retry loop absorbed; verbose streaming kept (raw thinking display) |
| **md2blank** | `stream_llm(package="md2blank")` | CLI model/provider still override |
| **agentos** | keeps its own resolver+breaker | wraps `invoke_llm`/`stream_llm` one-shot; reads context windows from catalog |

## Adding a provider (e.g. wwhb)

1. Add a row to `providers:` in `models.yml` (with `dsgvo: true/false`)
2. Add model entries to `models:` if new
3. Map the client in `clients.yml` or team settings

No package code changes. Deploy openutils to ship the catalog edit.

## See also

- [ADR 0003](../../../docs/adr/0003-shared-llm-invocation-layer.md) — the invocation seam (call_llm / invoke_llm / stream_llm)
- [ADR 0004](../../../docs/adr/0004-canonical-model-config.md) — canonical model config (registry + resolve_model + retry/backup)
