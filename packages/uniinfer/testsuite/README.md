# uniinfer testsuite — live tiers

Live integration tests that talk to a **real proxy + provider** (the unit/mocked
tests live in `uniinfer/tests/`, run with `uv run pytest` — no network).

## Tiers (cheap → deep)

| Tier | Answers | What it checks |
|------|---------|----------------|
| **smoke** | is it alive? | CLI version/list/completion/`--no-think`; proxy version, a non-stream chat, a stream (content must flow), embeddings |
| **details** | is it correct? | reasoning control (on/off + the legacy `think:false` shim), tool calling, stream shape (content + finish + `[DONE]`), **3-turn (turn-based) streaming with context carry**, malformed model → 4xx, embeddings |
| **perf** | is it fast? | generation throughput, TTFT, latency, context scaling |

## Run

```bash
# all tiers (default target: a local proxy on :8013 + amp ollama + mistral CLI)
./testsuite/run.sh                 # all
./testsuite/run.sh smoke           # one tier
./testsuite/run.sh details
./testsuite/run.sh perf
```

Point it at another proxy/model via env (all optional):

```bash
PROXY_URL=https://localhost:8123 PROXY_AUTH= \
  MODEL=tu@qwen-3.6-35b ./testsuite/run.sh all
```

| Env | Default | Meaning |
|-----|---------|---------|
| `PROXY_URL` | `http://127.0.0.1:8013` | proxy base URL |
| `PROXY_AUTH` | _(none)_ | proxy bearer (open providers like ollama need none) |
| `MODEL` | `ollama@qwen3.5:0.8b` | proxy chat model |
| `EMBED_MODEL` | `ollama@nomic-embed-text-v2-moe` | proxy embedding model |
| `PROVIDER` / `CLI_MODEL` | `mistral` / `mistral-small-latest` | CLI smoke provider/model |
| `AUTH_REQUIRED_MODEL` | _(none)_ | a key-required model for the details 401 test (else skipped) |

## Token counting (perf)

Total output = `usage.completion_tokens`, which **includes thinking** per the
OpenAI spec (`completion_tokens_details.reasoning_tokens` is a subset, not
additive). Throughput = `completion_tokens / (total − TTFT)` — generation
isolated from prefill (not `tokens/total`, which lets prefill crush short runs;
not chunk-count, since a chunk isn't a token). If the provider itemizes
reasoning tokens, the thinking share is shown; otherwise the report falls back
to TTFT + output tok/s (`completion_tokens` still includes any thinking).

## Files

- `run.sh` — tier runner (`smoke | details | perf | all`)
- `smoke` → `../scripts/test_cli.sh`, `../scripts/test_proxy.sh`
- `details.py` — correctness (reasoning / tools / streaming / turn-based / errors)
- `perf.py` — throughput / TTFT / latency / context scaling
- `test_tu_models.py` — (pre-existing) TU model validation suite

## Thinking-model note

Thinking models (qwen, glm, …) spend the token budget on reasoning before the
visible answer. The tests therefore pair `max_tokens` with the reasoning state
(small when reasoning is off, generous when on) — see `_SMALL`/`_GENEROUS` in
`details.py`. Don't hand-tune `max_tokens` per test.
