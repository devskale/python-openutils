# Changelog

All notable changes to **uniinfer** are documented in this file.
Versions follow [Semantic Versioning](https://semver.org/); this file
adheres to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.6.15] - 2026-07-21

### Fixed

- **TU `_prepare_payload` passthrough crash** — 0.6.14 referenced
  `self.EXTRA_FORWARD_DENY` which TU (a direct `ChatProvider` subclass, not
  `OpenAICompatibleChatProvider`) did not define, causing `AttributeError` on
  every TU request. Adds `EXTRA_FORWARD_DENY` to `TUProvider` and makes the
  passthrough failsafe via `getattr(..., frozenset())`.

## [0.6.14] - 2026-07-21

### Fixed

- **TU now forwards `stream_options` (and other OpenAI passthrough params) to
  vLLM.** TU's `_prepare_payload` built a curated payload and dropped the
  proxy's OpenAI passthrough extras — so `stream_options.include_usage` never
  reached vLLM, meaning vLLM never emitted the terminal usage chunk that
  0.6.13's proxy fix was waiting for. Streaming usage now flows end-to-end.
  Mirrors the base `OpenAICompatibleChatProvider._build_payload` passthrough
  (respects `EXTRA_FORWARD_DENY`).

## [0.6.13] - 2026-07-21

### Fixed

- **Streaming usage never emitted** — the proxy's streaming `/v1/chat/completions`
  now honors `stream_options.include_usage` and emits a terminal `choices:[]` +
  `usage` chunk before `[DONE]`. Fixes context-% display and token stats
  (`↑↓RW`, `CH`, `$`) for streaming consumers (pi, OpenAI-compatible clients).
  Three-layer fix:
  - `TUProvider.astream_complete`: forward upstream `usage` instead of hardcoding `{}`;
    forward the terminal `choices:[]` usage chunk instead of skipping it.
  - `OpenAICompatibleChatProvider.astream_complete`: forward the terminal
    `choices:[]` usage chunk (was skipped by the `if not choices: continue` gate).
  - Proxy `streaming.py`: attach `usage` to chunks via `format_chunk_to_openai`;
    emit the terminal usage chunk when `include_usage` was requested.

## [0.6.12] - 2026-07-21

### Fixed
- **Vision was broken through uniinfer for all OpenAI-compatible providers.**
  `OpenAICompatibleChatProvider._flatten_messages` silently stripped `image_url`
  parts from multimodal content, collapsing list content to a text-only string —
  so vision models never received the image and hallucinated. Confirmed against
  `kilo@stepfun/step-3.7-flash:free` (raw API returned "sylents" 5/5; via uniinfer
  the image was dropped and the model confabulated). Added an opt-in
  `PRESERVE_MULTIMODAL: bool = False` class flag: when True, list content
  (including image_url parts) is forwarded as-is; when False (default), the
  legacy flatten-to-string behaviour is preserved for text-only backends.
  Enabled on `KiloProvider` and `OpenCodeProvider` (both gateways forward
  native OpenAI multimodal content to vision-capable upstreams).

## [0.6.10] - 2026-07-19

### Added
- **Structured-output probe** (`probe_structured_output`) — sends
  `response_format: {type: json_schema, json_schema: {...}}` and validates the
  response parses as JSON and conforms to the schema (required keys + declared
  types). Closes the gap exposed by Requesty's 244-model structured-output
  matrix: models that *declare* `structured_output` but don't honor it. Skips
  when the catalog declares no `structured_output` capability or the backend
  400s on `response_format`. No `jsonschema` dependency — lightweight structural
  validation only.
- **Tool-calling probe upgrade** — `probe_tool_calling` is no longer boolean
  ("did any tool fire?"). It now scores three facets, matching the ecosystem
  (LLMToolCallingTester / DeepEval ToolCorrectness): **selection** (precision —
  offer 3 tools, ask for one by name, the *correct* tool must fire),
  **parameters** (the called tool's args parse as JSON and carry the required
  key), and **negative** (with tools available but a plain prompt, the model
  must *not* call a tool — catches over-eager callers). Status is `pass` only
  if all three pass; `fail` carries evidence naming the failing facet.
- New fixtures: `tools_multi.json` (3-tool set), `person_schema.json`.
- `structured_output` now added to the declared-capabilities list (was tracked
  in the catalog but not surfaced to probe skip-logic).

## [0.6.9] - 2026-07-19

### Changed
- **Kilo Gateway: credgoo auto-resolution for free models.** `KiloProvider`
  now resolves a key from credgoo (`kilocode` service) in both `__init__` and
  `list_models` when no key is passed. Free models still work anonymously if no
  key is stored, but authenticated free-model requests get higher rate limits
  than the 200 req/hr anonymous tier. Paid models already needed the key; this
  just makes free models use it too when available.

## [0.6.8] - 2026-07-17

### Added
- **Kilo Gateway provider** (`kilo`) — the Kilo AI Gateway
  (OpenAI/OpenRouter-compatible at `https://api.kilo.ai/api/gateway`) aggregates
  300+ models from 60+ providers (Anthropic, OpenAI, Google, xAI, DeepSeek, Qwen,
  NVIDIA, …). The `/models` endpoint is public (no auth) and returns rich
  metadata (pricing, context window, max tokens, modalities, capabilities).
  **Free models work anonymously** — 12+ free models usable with no key at all,
  rate-limited to 200 req/hr per IP (e.g. `tencent/hy3:free`,
  `nvidia/nemotron-3-ultra-550b-a55b:free`, `cohere/north-mini-code:free`,
  `poolside/laguna-m.1:free`). Paid models need a Kilo Gateway API key, stored
  via `credgoo --add kilocode <key>` (credgoo service is `kilocode`). Use as
  `kilo@<model>`; model ids are `provider/model-name` (e.g.
  `kilo@anthropic/claude-sonnet-4.6`). The `kilo-auto/*` tiers
  (`frontier`/`balanced`/`free`/`small`/`efficient`) route server-side. Note:
  `nvidia/*:free` endpoints log prompts/outputs for NVIDIA service improvement —
  do not send confidential data. Tests + docs/providers.md + README.

### Changed
- `OpenAICompatibleChatProvider`: new `REQUIRES_API_KEY` class flag (default
  `True`). When `False`, `acomplete`/`astream_complete` skip the api-key guard,
  so gateways with an anonymous free tier (Kilo) can serve free models without a
  key. No behaviour change for existing providers.
- CLI `_resolve_credgoo_service` now honors a provider's `CREDGOO_SERVICE`
  attribute (was hard-coded to the provider id), so the kilo gateway resolves its
  key from the `kilocode` credgoo service.

## [0.6.7] - 2026-07-17

### Added
- **OpenCode/Zen provider** (`opencode`) — the OpenCode model router
  (OpenAI-compatible at `opencode.ai/zen/v1`) aggregates DeepSeek, GPT, Gemini,
  Qwen, GLM, MiniMax, Kimi, and more. Free models surfaced automatically:
  `deepseek-v4-flash-free`, `big-pickle`, `mimo-v2.5-free`, `hy3-free`,
  `nemotron-3-ultra-free`, `north-mini-code-free` (marked cost 0 in the catalog).
  Use as `opencode@<model>`; pass the Zen key as the Bearer token (or store it
  via `credgoo --add opencode <key>` on the airtable backend for seamless
  resolution). Note: many free models reason — use a generous `max_tokens`.
  Claude models on OpenCode use the Anthropic-messages API (not this OpenAI
  provider). Tests + docs/providers.md + provider_limits entry.

## [0.6.6] - 2026-07-17

### Added
- **OpenAI passthrough** — the proxy forwards unmapped OpenAI params
  (`top_p`, `response_format`, `seed`, `stream_options`, `logprobs`, …) verbatim
  to OpenAI-compatible backends instead of dropping them. New OpenAI features
  reach backends without a per-field code change: the input schema captures
  extras (`extra="allow"`) → flow through `Target` /
  `ChatCompletionRequest.extra` → the OpenAI-compatible provider payload. JSON
  mode (`response_format`), sampling (`top_p`/`seed`), and stream usage
  (`stream_options`) now work against supporting backends (e.g. Mistral JSON mode
  returns clean JSON, no fences). A per-provider `EXTRA_FORWARD_DENY` safety
  valve strips any field that 400s a backend (empty by default). Compat tests + a
  live JSON-mode check.

  This future-proofs the OpenAI-compatible claim: the earlier hard-breaks
  (`developer` role, trailing-assistant prefill) were symptoms of a fixed-field
  whitelist; passthrough makes the proxy absorb new OpenAI features automatically
  instead of one-off patching each.

## [0.6.5] - 2026-07-17

### Fixed
- **Trailing assistant message (prefill) no longer 400s on Mistral.** A
  trailing assistant message is the OpenAI-compatible prefill/continuation
  pattern (force raw JSON, no markdown fences). Mistral requires `prefix=True`
  on it (400 otherwise), but `prefix` isn't an OpenAI field so no client can set
  it. Regression tests cover the prefix logic; live details check validates the
  prefill against Mistral.

### Changed (generalized)
- **OpenAI-compatibility layer is now declarative, not one-off.** A trailing
  assistant prefill is handled by a base `PREFILL_FLAG` on
  `OpenAICompatibleChatProvider`: backends that need a continuation flag declare
  it (`MistralProvider.PREFILL_FLAG = "prefix"`); the base `_flatten_messages`
  applies it to the last assistant turn. Backends that accept a trailing
  assistant natively (Anthropic, Gemini, vLLM, Ollama, …) leave it `None`. The
  `developer`-role normalization stays at the proxy (universal scope).
  Documented in ARCHITECTURE.md ("OpenAI compatibility layer") so the next
  backend quirk lands in the right place.

## [0.6.4] - 2026-07-17

### Fixed
- **`role: "developer"` no longer 422s.** OpenAI's newer `developer` role
  (system instructions for reasoning models; emitted by the official SDKs and
  `@ai-sdk/openai`) was forwarded verbatim and rejected by backends that don't
  accept it (Mistral 422, etc.). The proxy now accepts the full OpenAI role set
  and collapses `developer` → `system` for backends without it (functionally
  equivalent, universally accepted), while preserving it for providers whose API
  accepts it natively (`openai`, `openrouter`). Regression tests cover both
  paths; live details check validates against Mistral.

## [0.6.3] - 2026-07-17

### Fixed
- **Non-streaming `/v1/chat/completions` no longer 400s on nested usage.**
  Providers like Mistral (and OpenAI reasoning models) return
  `prompt_tokens_details` / `completion_tokens_details` as objects; the schemas
  typed `usage: dict[str, int]`, which rejected those objects and made the proxy
  400 its own successful upstream result on non-streaming calls (streaming was
  unaffected). Modeled `CompletionUsage` (+ `PromptTokensDetails` /
  `CompletionTokensDetails`) per the OpenAI spec. Regression tests added.

### Changed
- `uv lock -U`: openai 2.46, anthropic 0.117, fastapi 0.139.2, mistralai 2.7,
  google-genai 2.12.1, huggingface-hub 1.24, others.
- Docs: `ARCHITECTURE.md` gains a crystal-clear **Naming** table
  (`uniioai_proxy.py` → `proxy_app.py`; the `uniioai-proxy` command + logger
  name are intentionally kept); rename notes in `AGENTS.md` and `docs/issues.md`.

## [0.6.2] - 2026-07-17

### Added
- **Unified web app at `/`** — a cohesive, production-grade single-page app
  split into **Chat** (provider/model selectors, streaming + markdown, reasoning
  aside, stop/abort), **Dashboard** (Usage from `/v1/system/stats`, Provider
  limits from `/v1/system/provider-limits`, links to the dedicated dashboards),
  and **Settings** (proxy base URL, API key with remember toggle, default
  temperature/max_tokens/reasoning_effort/system prompt, refresh-models).
  Self-contained (inline CSS/JS, no build), dark theme, responsive. Replaces the
  old root JSON; the legacy `/webdemo`, `/perf`, `/capabilities`, `/guide` pages
  remain.

## [0.6.1] - 2026-07-17

### Added
- **Provider free-tier limits** are now data-driven and queryable:
  `uniinfer/config/provider_limits.json` (24 providers; rpm/rpd/tpm/tpd/monthly/
  concurrent; per-model overrides like gemini-3.5-flash=20 RPD) is loaded into
  `PROVIDER_CONFIGS[*].free_tier_limits`. New `GET /v1/system/provider-limits`
  joins documented limits with live usage (24h/7d) + utilization %;
  `GET /v1/system/provider-limits.html` serves a vanilla-JS dashboard.
- **Live testsuite** (`testsuite/`): tiered smoke → details → perf runner
  (`run.sh`), covering streaming, thinking (on/off), tools, 3-turn (turn-based)
  context carry, error handling, and honest throughput (completion_tokens incl.
  thinking, isolated from prefill). `scripts/test_cli.sh` + `test_proxy.sh` for
  smoke.

### Fixed
- Rate-limit tests no longer hit a live amp ollama behind nginx auth with a
  nonexistent model — they mock the completion path and use fake model sentinels
  (no real model encoded; models are ephemeral).

## [0.6.0] - 2026-07-16

Two architectural deepenings. Each turns a concern smeared across many call
sites into one deep module behind a small interface. Net −614 lines.

### Changed
- **Completion dispatch is now a deep `Target` module**
  (`uniinfer/completion.py`). The six duplicated parse → instantiate →
  request-build → dispatch → access-recording sequences that lived in
  `uniioai.py` and `capabilities` collapse into one `Target` exposing
  `complete / stream_complete / acomplete / astream_complete`. Callers (proxy
  routers, CLI, capabilities, examples) migrate to it. Stream paths yield raw
  `ChatCompletionResponse`; OpenAI/SSE shaping stays at the proxy seam
  (`streaming.py`), fixing the prior sync/async yield-shape asymmetry.
- **Thinking-control translation moved into providers.** A typed
  `reasoning_effort: Literal["none","minimal","low","medium","high"]` is the
  single primary intent on `ChatCompletionRequest`; each reasoning-capable
  provider maps it to its own dialect (ollama `think`, vLLM
  `chat_template_kwargs`, Z.AI `thinking`). Contract: `none`/`minimal`
  disables reasoning everywhere; an explicit `chat_template_kwargs` escape
  hatch wins. The old `if reasoning_effort == "minimal"` translation blocks in
  the router, CLI, and capabilities are gone.
- **`parse_provider_model` is now shared** (`uniinfer.completion`); the HTTP
  adapter translates `ValueError → HTTPException(400)`.
- **Capabilities' config dataclass renamed `Target → ProbeTarget`**
  (`capabilities.Target` remains as a back-compat alias). Its three ollama
  bypass sites collapse onto a non-recording `Target(record_access=False)`.
- `--no-think` now sets `reasoning_effort="none"` (was
  `chat_template_kwargs`). The deprecated HTTP `think` field is shimmed to
  `reasoning_effort="none"` at the proxy boundary.

### Added
- `uniinfer.completion.Target`, `parse_provider_model`, `REASONING_OFF`.
- `CONTEXT.md` domain glossary.
- Tests: `test_completion_target.py` (20), `test_reasoning_effort_contract.py`
  (28) — the thinking-control contract had zero coverage before.

### Removed (breaking)
- `ChatCompletionRequest.enable_thinking` and `.thinking_budget` (both dead —
  no caller set them; vLLM silently ignores the top-level field).
- `uniinfer.uniioai.get_completion / stream_completion / aget_completion /
  astream_completion / format_chunk_to_openai` (moved into `Target` /
  `streaming.py`).
- Ollama's provider-specific `think=` kwarg (now read off the request as
  `reasoning_effort`).

## [0.5.40] - 2026-07-15

### Added
- **CLI thinking-control flags**: `uniinfer --no-think` sets
  `chat_template_kwargs.enable_thinking=false` (for Qwen3.x / GLM-5.x style
  vLLM-served models — handy for A/B comparing thinking vs non-thinking), and
  `--chat-template-kwargs JSON` forwards an arbitrary object to the backend
  chat template (overrides `--no-think`).

## [0.5.39] - 2026-07-15

### Added
- **Adaptive per-model rate limiting for TU.** Replaced the fixed
  25 req/min throttle in the TU provider with a self-tuning AIMD
  (Additive-Increase / Multiplicative-Decrease) controller
  (`uniinfer/ratelimit.py`). TU advertises "25/min" but the real effective
  limit is far lower (especially for heavy models such as
  `glm-5.2-744b-preview`, ~5/min) and changes without notice. The limiter:
  - enforces a sliding-window cap at the *estimated* safe rate per
    `(provider, model)`, with even temporal spacing;
  - on HTTP 429 infers the real limit from the burst that tripped it and
    halves the estimate, applying an exponential cooldown; the failed call
    is retried with backoff instead of erroring straight to the client;
  - nudges the estimate upward on success once a 429-free period elapses;
  - **re-challenges daily**: restores the ceiling and probes higher so a
    silent 25 -> 100 -> 1000/min upgrade is discovered automatically;
  - persists learned limits to `_rate_limits.json` across restarts.
  Configurable via `UNIINFER_RATE_LIMIT_*` env vars.
- **`GET /v1/system/rate-limits`** observability endpoint: returns the live
  per-(provider, model) learned rate, ceiling, active cooldown, 429 streak
  and last re-challenge time so the tracked limits are readable in prod.

### Fixed
- Re-entrant locking in the rate limiter (`RLock`) so persistence writes
  from within `on_429`/`on_success`/`reset` no longer deadlock.
- **TU Staging isolation**: `TUStagingProvider` (a distinct backend,
  `aqueduct-staging`) now resolves its own rate limiter (`tu-staging`)
  instead of sharing production TU's learned limit.
- `_KeyState.from_dict` tolerates corrupt/partial `_rate_limits.json`
  values (falls back to defaults instead of crashing at load).

## [0.5.34] - 2026-06-24

### Fixed
- **Leaked tool-call XML in `reasoning_content`.** The TU Aqueduct vLLM
  `glm47` tool parser intermittently leaks its native
  `<tool_call><arg_key>…<arg_value>…</tool_call>` XML into the
  **thinking** stream, not only into `content`. The GLM leak interceptor
  now also processes `reasoning_content`, stripping leaked XML before it
  reaches OpenAI-compatible clients (pi, klark0, …). Confirmed by
  cross-referencing `logs/tu_raw_chat.log` with pi session logs, which
  showed reasoning streams ending in `</tool_call>` despite clean
  `finish_reason` (no preemption).
- **Stream tail dropped when `finish_reason` is sent.** The leak-repair
  rolling tail buffer (~64 chars) was only flushed in the unreachable
  `if not sent_finish_reason` branch, so any response whose upstream
  signalled `finish_reason` lost its final ≤64 chars of content (and now
  reasoning). Both the content and reasoning tails are flushed before
  emitting the `finish_reason` chunk in both the dict and object code
  paths.

## [0.5.33] - 2026-06-23

### Added
- Public `/v1/catalog` endpoint with provider filtering and optional
  download parameter.

## [0.5.32] - 2026-06-20

### Added
- **GLM-5.x tool-call leak repair** (`GlmLeakInterceptor`): detects
  leaked XML tool-call format in streamed `content` and reconstructs it
  as structured OpenAI `tool_calls` deltas.
- `chat_template_kwargs` passthrough for provider-specific serving
  options (e.g. vLLM tool-parser selection).

## [0.5.21] - 2026-04-21

### Changed
- Internal maintenance release.

## [0.5.20] - 2026-04-21

### Changed
- Internal maintenance release.

## [0.5.19] - 2026-04-21

### Changed
- Version bump.

## [0.5.18] - 2026-04-18

### Added
- `generate_models` script, `models.json`, Ollama updates.

## [0.5.17] - 2026-04-16

### Added
- `ModelInfo` dataclass — rich model metadata harvested from all
  providers (context window, pricing, capabilities, modalities).

## [0.5.16] - 2026-03-15

### Changed
- Bump `credgoo` 0.1.10 → 0.1.11.

## [0.5.15] - 2026-03-13

### Changed
- Bump to align with credgoo release.

## [0.5.12] - 2026-02-26

### Added
- Credgoo auto-invoke in `GeminiProvider`; env-var fallback removed from
  factories.

## [0.5.4] - 2026-02-12

### Added
- Integrate `reasoning_content` across proxy and webdemo.

## [0.5.3] - 2026-02-11

### Added
- Thinking (`reasoning_content`) support for Z.ai and TU providers.
