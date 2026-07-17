# AGENTS.md - UniInfer

**Package**: uniinfer (python-openutils) · **Repo map**: [../AGENTS.md](../AGENTS.md)

## Stack

- Python 3.9+, **uv** for package management (not pip)
- Formatting: black (line-length 88), isort (profile: black), ruff
- Tests: pytest with unittest.mock
- Proxy: FastAPI + uvicorn

## Commands

```bash
uv sync                                    # install deps from lockfile
uv sync --extra all                        # install all optional provider deps

uv run pytest                              # all tests
uv run pytest uniinfer/tests/test_auth.py  # single file
uv run pytest -k "test_token_validation"   # by keyword
uv run pytest -v --cov=uniinfer --cov-report=term-missing  # with coverage

uv run black .                             # format
uv run isort .                             # sort imports
uv run ruff check . --fix                  # lint + auto-fix

uv build                                   # build package
uv run python3 scripts/generate_models.py  # regenerate models.json
```

## Deployment (Production)

Proxy runs as **systemd service** on `amd-1`, port **8124**.

```bash
sudo systemctl status uniioai-proxy
sudo systemctl restart uniioai-proxy
sudo journalctl -u uniioai-proxy -f        # live logs
curl -s http://localhost:8124/v1/system/version  # health check
```

After pushing to `main`: `cd /home/ubuntu/code/python-openutils && git pull && cd packages/uniinfer && uv sync && sudo systemctl restart uniioai-proxy`

| Key | Value |
|-----|-------|
| Service file | `/etc/systemd/system/uniioai-proxy.service` |
| Working dir | `/home/ubuntu/code/python-openutils/packages/uniinfer` |
| Binary | `.venv/bin/python .venv/bin/uniioai-proxy --port 8124` |
| Config | `.env` (`PROXY_KEY`, `PROXYHOST`, `PROXY_PORT`) — gitignored |
| Models refresh | `uniioai-models-refresh.timer` daily at 04:00 UTC |

> **Naming:** the `uniioai-proxy` **command/service** runs the `uniinfer.proxy_app:main` **module** (renamed from `uniioai_proxy.py`; the command + logger name `uniioai_proxy` are intentionally kept). See [ARCHITECTURE.md](ARCHITECTURE.md#naming).

## Proxy Auth Token

Proxy requires a **credgoo combined token** (`bearer@encryption`) as Bearer auth, stored in `.env` as `PROXY_KEY`.

- **Format**: `<credgoo_bearer>@<credgoo_encryption_key>` — `@` separator triggers credgoo resolution
- **In test code**: `headers={"Authorization": f"Bearer {os.getenv('PROXY_KEY')}"}`
- **No token?** Assert `status_code in [200, 401, 500]` to pass in both authed and unauthed environments

## Docs

| File | Owns |
|------|------|
| `README.md` | User-facing usage, installation, provider table |
| `ARCHITECTURE.md` | Proxy router/service/schema layout |
| `AGENTS.md` | This file — contributor rules |
| `docs/models.md` | Model catalog, types, metadata richness |
| `docs/providers.md` | Full provider index with base URLs, defaults |
| `docs/integration.md` | How to integrate: Python module + uniioai proxy API |

## Code Style

- PEP 8, enforced by black + isort + ruff (see config in files)
- **No comments** unless explicitly requested
- Docstrings required for all public functions/classes/modules (Google style)
- Type hints on all public APIs
- Imports: absolute from package root, grouped stdlib → third-party → local
- Error handling: use `map_provider_error()` from `uniinfer/errors.py`

## Testing

- Place in `uniinfer/tests/`, name `test_*.py`
- Mock external API calls with `unittest.mock`
- Test sync + async + streaming + list-models + error mapping for providers
- Test proxy request limits and security validators (size, message count, auth)

## Model Catalog

- `list_models()` returns `list[ModelInfo]` — `ModelInfo.__str__` returns `id`, `__eq__` matches strings
- Fix model types in `uniinfer/models/type_overrides.json`
- Regenerate: `uv run python3 scripts/generate_models.py`
- Auto-managed: `_model_history.json` (first_seen), `_speed_results.json` (speed tests)

## Provider Implementation

### Key modules (where things live)

- `uniinfer/completion.py` — **`Target`**: the deep completion-dispatch module. Owns parse → instantiate → request-build → dispatch → access-recording behind `complete / stream_complete / acomplete / astream_complete`. This is the one home for "reach a model and complete"; the old `get/stream/aget/astream_completion` helpers were removed.
- `uniinfer/provider_access.py` — proxy helpers: credgoo key resolution (`get_provider_api_key`), embeddings (`get_embeddings`), model listing.
- `uniinfer/proxy_app.py` — the FastAPI app + `main()` entry point (renamed from `uniioai_proxy.py`).
- `uniinfer/core.py` — data classes: `ChatCompletionRequest` (with typed `reasoning_effort: Literal["none","minimal","low","medium","high"]`), `ChatProvider`, `ModelInfo`.
- `uniinfer/ratelimit.py` — adaptive per-(provider,model) AIMD limiter (TU).
- Provider limits: `uniinfer/config/provider_limits.json` → baked into `PROVIDER_CONFIGS[*].free_tier_limits`, served at `/v1/system/provider-limits`.
- Live testsuite: `testsuite/run.sh` (smoke/details/perf) + `scripts/test_cli.sh`, `scripts/test_proxy.sh`.

### Adding a New Provider

1. Create `uniinfer/providers/<name>.py`
2. Inherit from `ChatProvider`, `OpenAICompatibleChatProvider`, or `AnthropicCompatibleProvider`
3. Set constants: `BASE_URL`, `PROVIDER_ID`, `ERROR_PROVIDER_NAME`, `DEFAULT_MODEL`, `CREDGOO_SERVICE`
4. Implement: `complete()`, `stream_complete()`, `list_models()` (+ async variants)
5. Export in `uniinfer/providers/__init__.py`
6. Register in `uniinfer/__init__.py` via `ProviderFactory.register_provider()`
7. Add conditional import for optional deps
8. Add tests (sync, async, streaming, list-models, error mapping)

### OpenAI-Compatible Flow

Inherit `OpenAICompatibleChatProvider` → override only `list_models()` and header hooks if non-standard.

### Anthropic-Compatible Flow

Inherit `AnthropicCompatibleProvider` → override `list_models()` fallback if no Anthropic model-list endpoint.

### Ollama provider — name & model id (common footgun)

Setting up Ollama in uniinfer trips people up on **two things**: the provider
name and how the model id is spelled.

- **Provider name**: `ollama` (registered in `ProviderFactory`, `uniinfer/__init__.py`). Implementation: `uniinfer/providers/ollama.py` — a **custom** `ChatProvider` using Ollama's native `/api/chat` + `/api/tags`, *not* the OpenAI-compatible subclass.
- **Model id = `ollama@<model>`**: everything after the `@` is the **literal Ollama model id** (version tag included), e.g. `ollama@gemma3:1b`, `ollama@qwen3.5:0.8b`. The proxy / `Target` / `parse_provider_model` split on the **first** `@` to get `{provider, model}`.
- **Won't route**: a bare id (`gemma3:1b`) or wrong separator (`ollama:gemma3:1b`) → `ValueError: Invalid provider_model format. Expected 'provider@modelname'`.
- **Thinking**: reasoning is controlled by `reasoning_effort` on the request (`none`/`minimal` disable it). Ollama maps that to its native `think` field internally; callers no longer pass a `think=` kwarg. The CLI `--no-think` sets `reasoning_effort="none"` (works across backends, not just vLLM).
- **Auth**: Ollama bypasses proxy auth locally (see Known Footguns).

## Boundaries

- ✅ Always run tests + lint before committing
- ✅ Use `map_provider_error()` for all provider error handling
- ✅ Maintain OpenAI-compatible response formats
- ⚠️ Ask before modifying `core.py` public API (backward compat)
- ⚠️ Ask before adding new dependencies
- 🚫 Never commit real tokens or `.env` files
- 🚫 Never remove a failing test without explicit approval
- 🚫 Never modify auto-generated files (`_model_history.json`, `_speed_results.json`) by hand

## Version Updates

- **Minor** (default): increment patch (`0.3.5` → `0.3.6`)
- **Major**: increment minor, reset patch (`0.3.5` → `0.4.0`)

## Adaptive Rate Limiting

TU enforces a **self-tuning AIMD** rate limit (`uniinfer/ratelimit.py`),
keyed per `(provider, model). It does **not** trust the provider's advertised
limit (TU says "25/min" but the real ceiling is far lower, especially for
heavy models like `glm-5.2-744b-preview`).

- On a 429 it halves the estimate, applies an exponential cooldown, and
  **retries** the call (so clients rarely see a 429).
- On success (after a stable period) it nudges the estimate up.
- **Daily re-challenge**: restores the ceiling and probes higher, so a silent
  25 → 100 → 1000/min upgrade is discovered automatically.
- Learned limits persist to `_rate_limits.json` (gitignored) across restarts.
- Production TU (`tu`) and TU Staging (`tu-staging`) have **separate**
  limiters (distinct backends).
- **Observability**: `GET /v1/system/rate-limits` returns the live per-(provider,
  model) state — learned rpm, ceiling, active cooldown, 429 streak, last
  re-challenge. Public/read-only.

Env vars (all optional):

| Var | Default | Meaning |
|-----|---------|---------|
| `UNIINFER_RATE_LIMIT_DEFAULT` | 25 | initial rpm estimate |
| `UNIINFER_RATE_LIMIT_MIN` | 0.5 | floor (never fully stalls) |
| `UNIINFER_RATE_LIMIT_CEILING` | 1000 | hard cap additive-increase won't pass |
| `UNIINFER_RATE_LIMIT_WINDOW` | 60 | sliding-window size (seconds) |
| `UNIINFER_RATE_LIMIT_RECHALLENGE_HOURS` | 24 | re-challenge interval |
| `UNIINFER_RATE_LIMIT_PERSIST` | `_rate_limits.json` | state file (empty = disable) |

The config.json `tu_rate_limit_per_minute` still seeds the initial estimate.

## Known Footguns

- **Thinking models need `max_tokens` ≫ 1–2k** — Qwen3.x / GLM-5.x / Claude extended-thinking consume the token budget on reasoning *before* the visible answer. A too-low cap yields empty / truncated output that looks like a model bug. Defaults in the smoke router (`4096`), the Anthropic provider (`8192`), and the CLI speedtest (`4096`) are set for this reason; the proxy chat path defaults to `32768`. Always pass a generous `max_tokens` when exercising thinking models.
- `PROXY_KEY` format is `bearer@encryption` — the `@` is the delimiter, not part of either value
- `ModelInfo` equality matches strings — `model_info == "gpt-4"` works but is easy to miss
- Ollama models are addressed as `ollama@<model>` (split on first `@`); a bare id or `:` separator won't route — see "Ollama provider" above. Ollama also bypasses proxy auth — don't assume all endpoints require auth in tests
- Provider metadata richness varies widely — see `docs/models.md` for the matrix
- Reasoning models (TU/vLLM) may return `reasoning_content` in the assistant message
