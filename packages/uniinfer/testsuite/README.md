# uniinfer testsuite — live tiers

Live integration tests that talk to a **real proxy + provider** (the unit/mocked
tests live in `uniinfer/tests/`, run with `uv run pytest` — no network). uniinfer
is provider-agnostic by design, so **every testsuite script works against any
provider** — there is no TU-specific or provider-locked test left.

## Shared config

All scripts read the same config (see `testsuite/_proxy_common.py`):

| Env | Default | Meaning |
|-----|---------|---------|
| `PROXY_URL` | `https://localhost:8123` | proxy base URL (built from `PROXYHOST`/`PROXY_PORT`/`PROXY_SCHEME` if unset) |
| `PROXY_AUTH` | _(none)_ | proxy bearer token (also read as `PROXY_KEY`; open providers like ollama need none) |
| `MODEL` | `ollama@qwen3.5:0.8b` | single chat model (details/perf tiers) |
| `EMBED_MODEL` | `ollama@nomic-embed-text-v2-moe` | embedding model (details tier) |
| `PROVIDER` | _(none)_ | provider for the `matrix`/`models` tiers |
| `MODELS` | _(none)_ | `provider@model,...` for the `matrix`/`models` tiers |
| `MAX` | `3` | max models for the `matrix` tier |
| `DOWN_TIMEOUT` | `8` | fail-fast reachability gate (s) — how long to wait on a chat/readiness probe before declaring a model **down**. Kept single-digit so reachability is known within seconds; bump it if your models cold-start slowly |
| `AUTH_REQUIRED_MODEL` | _(none)_ | a key-required model for the details 401 test (else skipped) |

Self-signed HTTPS proxies are accepted (`verify=False`), so the script "just
works" against `localhost:8123` or a local `:8123`.

## Tiers (cheap → deep)

| Tier | Answers | What it checks |
|------|---------|----------------|
| **smoke** | is it alive? | CLI version/list/completion/`--no-think`; proxy version, a non-stream chat, a stream (content must flow), embeddings |
| **details** | is it correct? | reasoning control (on/off + the legacy `think:false` shim), tool calling, stream shape (content + finish + `[DONE]`), **3-turn (turn-based) streaming with context carry**, malformed model → 4xx, embeddings |
| **perf** | is it fast? | generation throughput, TTFT, latency, context scaling |
| **bench** | probe + tok/s | OpenAI-compatible probe + quick/long tok/s for **any** `/v1/chat/completions` endpoint (vLLM, Ollama, OpenAI, uniinfer proxy) — `baseurl` + `bearer` config |
| **matrix** | what can it do? | capability probe + matrix for a provider (`tool_calling`, `structured_output`, `image`, `thinking`) via `/v1/system/capabilities` |
| **models** | does it behave? | direct LLM validation (alive / tool_use / thinking) for a provider via `/v1/chat/completions` |

`smoke`/`details`/`perf` run against a single `MODEL`. `matrix`/`models` run
across a whole provider (auto-discovered from `/v1/models`) or an explicit
`MODELS` list — no provider is hardcoded.

## Run

```bash
# all tiers (default target: a local proxy on :8123 + ollama chat model)
./testsuite/run.sh
./testsuite/run.sh smoke

# capability matrix for any provider (auto-discovers its chat models)
PROXY_URL=https://localhost:8123 PROXY_AUTH=<PROXY_AUTH> \
  PROVIDER=tu ./testsuite/run.sh matrix

# direct LLM validation for an explicit set of models
uv run python testsuite/test_models.py --models tu@qwen-3.6-35b,mistral@mistral-small-latest

# capability matrix for an explicit model
uv run python testsuite/test_via_proxy.py --models tu@qwen-3.6-35b --max 1
```

## Files

- `_proxy_common.py` — shared proxy config + httpx client + result glyphs
- `run.sh` — tier runner (`smoke | details | perf | matrix | models | all`)
- `smoke` → `../scripts/test_cli.sh`, `../scripts/test_proxy.sh`
- `details.py` — correctness (reasoning / tools / streaming / turn-based / errors)
- `perf.py` — throughput / TTFT / latency / context scaling
- `test_via_proxy.py` — capability probe + matrix for any provider (via proxy)
- `test_models.py` — direct LLM validation (alive / tool_use / thinking) for any provider (via proxy)
- `bench_openai.py` — probe + quick/long tok/s for **any** OpenAI-compatible `/v1/chat/completions` endpoint (`BASEURL` + `BEARER` + `MODEL`)
- `bench_realworld.py` — **real ö-Vergaberecht doc bench**. Realistic fictional LV fixtures (`fixtures/docs/realworld/lv_reinigung_*.md`, PII-free) × the query/case set in `fixtures/cases/realworld_lv.jsonl` (Gesamtbetrag / günstigste Position / Kategorien / größte Menge — each with a ground-truth `expected` for the `check` column). Runs **all cases per doc** (one invocation = multiple ö-Vergaberecht tests) across `--reasonings` (nothink/think) × `--streams`. Metrics: `tok/s` (effective, incl prefill), `dec/s` (decode/generation, `completion/(lat−ttft)`, suppressed when the decode phase is <1s), **exact** `reasoning_tokens` (from `completion_tokens_details`; `~` = ÷3.5 estimate fallback), derived `content_tokens`, `ttft`, `time`. Single `_call(stream=)` seam feeds `extract_usage` + `compute_throughput`. Supports `--model` (single) or `--config testsuite/models.json` (batch, multiple models). Output: terminal table + JSONL log.

## Quick reference

```bash
# capability matrix for a provider (auto-discovers its chat models)
PROXY_URL=https://localhost:8123 PROXY_AUTH=<PROXY_AUTH> \
  PROVIDER=tu ./testsuite/run.sh matrix

# generic OpenAI-compatible probe + tok/s (vLLM / Ollama / OpenAI / uniinfer proxy)
BASEURL=https://localhost:8123/v1 BEARER=<PROXY_AUTH> \
  MODEL=tu@qwen-3.6-35b REASONING=none ./testsuite/run.sh bench

# or directly
uv run python testsuite/bench_openai.py --base-url https://localhost:8123/v1 \
  --bearer <PROXY_AUTH> --model tu@qwen-3.6-35b --reasoning none
```

## Thinking-model note

Thinking models (qwen, glm, …) spend the token budget on reasoning before the
visible answer. The tests therefore pair `max_tokens` with the reasoning state
(small when reasoning is off, generous when on) — see `_SMALL`/`_GENEROUS` in
`details.py`. Don't hand-tune `max_tokens` per test.

The `thinking` check in `test_models.py` is auto-skipped (not failed) for models
that don't emit `reasoning_content` — the script discovers thinking capability
empirically rather than from a hardcoded model list.
