# CONTEXT — uniinfer

Domain glossary for uniinfer. Terms used in code, docs, and architecture
reviews. Keep this current when concepts are named or sharpened.

## Nouns

- **Provider** — an LLM backend behind `ChatProvider` (ollama, tu/vLLM, zai,
  openai, anthropic, gemini, …). Registered in `ProviderFactory`. Each owns its
  own dialect of every cross-provider concern.
- **Reasoning effort** — the cross-provider *intent*: how hard a model should
  reason. Typed `Literal["none","minimal","low","medium","high"]`, carried on
  `ChatCompletionRequest.reasoning_effort`. The single primary thinking-control
  intent on the request interface.
- **Thinking dialect** — a provider's native reasoning knob. Owned by the
  provider, translated internally from reasoning effort, never by callers:
  ollama `think` (bool), vLLM `chat_template_kwargs.enable_thinking`,
  Z.AI `thinking` object. Each reasoning-capable provider maps reasoning effort
  to its own dialect.
- **ProbeTarget** — the capability suite's probe-config dataclass
  (`capabilities.Target` is a back-compat alias). Carries `provider_model,
  api_key, base_url, max_tokens, timeout, heavy_perf`. Builds a non-recording
  `Target` (record_access=False) to dispatch, so probes don't pollute usage
  metadata.
- **parse_provider_model** — the one shared `provider@model` split
  (`uniinfer.completion.parse_provider_model`, raises `ValueError`). The HTTP
  seam (`uniioai_proxy.parse_provider_model`) is a thin adapter translating
  that to `HTTPException(400)` plus allowed-provider validation.
- **Target** — the deep completion-dispatch module (`uniinfer.completion.Target`). Binds a
  `provider@model` to a ready provider instance and owns the full parse →
  instantiate → request-build → dispatch → access-recording sequence behind one
  small interface (`complete / stream_complete / acomplete / astream_complete`).
  All four paths build one uniform request (only the `streaming` flag and
  sync/async dispatch differ) and yield raw `ChatCompletionResponse` (stream
  paths). `record_access` (default True) controls model-access tracking;
  diagnostic callers pass False. The single home for "reach a model and
  complete" — formerly six duplicated copies across uniioai.py + capabilities.

## Contract

- **"none / minimal disables reasoning"** — every reasoning-capable provider
  honors `reasoning_effort="none"` (and `"minimal"`) by turning reasoning OFF in
  its own dialect. A caller sets reasoning effort once and trusts the meaning
  across providers. This is the *leverage* of the thinking-control seam.
- **Escape-hatch precedence** — an explicit `chat_template_kwargs` overrides any
  reasoning-effort-derived default ("default unless overridden"). The escape
  hatch wins because it is the reliable, deliberately-set knob.

## Verbs (request thinking surface)

`reasoning_effort` (intent) · `chat_template_kwargs` (escape)

Retired from the request interface: `enable_thinking` (dead — vLLM silently
  ignores the top-level field; the reliable path is
  `chat_template_kwargs.enable_thinking`), `thinking_budget` (only tu read it,
  never set). Ollama `think` no longer rides `**provider_specific_kwargs` — it
  is ollama's dialect of reasoning effort, read off the request.

## Failure mode

- **Deprecated HTTP `think`** — accepted at the proxy boundary as a deprecation
  shim: `{"think": false}` → `reasoning_effort="none"` (only if the caller did
  not already set reasoning effort). Emits a deprecation log; to be removed
  after a release cycle.
