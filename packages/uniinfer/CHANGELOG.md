# Changelog

All notable changes to **uniinfer** are documented in this file.
Versions follow [Semantic Versioning](https://semver.org/); this file
adheres to [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
