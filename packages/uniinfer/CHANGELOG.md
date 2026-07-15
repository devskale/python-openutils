# Changelog

All notable changes to **uniinfer** are documented in this file.
Versions follow [Semantic Versioning](https://semver.org/); this file
adheres to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
