# GLM-5.2-preview degenerate-loop investigation (TU Aqueduct)

**Status:** root cause isolated to the **upstream TU / vLLM serving layer**.
The uniinfer package is **exonerated**. See `ISSUE_REPORT.md` for the upstream
bug report.

## Summary

`glm-5.2-744b-preview` on TU Aqueduct intermittently collapses mid-generation
into a short-unit repeating-token loop, e.g.:

```
0\n0\n0\n0\n0\n0…
0%0%0%0%0%0%0…
0,0.0.0.0.0.0.0…
: 0\n: 0\n: 0\n: 0\n…
TheThe::::::::::::…
```

This is the exact symptom users see in agent loops (pi): the model emits a few
real tokens then degenerates into garbage until `max_tokens` is hit
(`finish_reason=length`).

## Validated conclusions

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Bug is in the **uniinfer package** | ❌ **Falsified** | `check_model_vs_package.py` — identical prompt sent DIRECT to TU Aqueduct and THROUGH the proxy; both degenerate identically. Path A never touches uniinfer. |
| Driven by **`max_tokens`** being too large | ❌ **Falsified** | `check_max_tokens.py` — at 8k input, `max_tokens=220` loops identically to `max_tokens=50000`. The cap is irrelevant; the loop onset is in the first output tokens. |
| Driven by **input context size** | ⚠️ **Partially** | Context ≥ ~180k tokens degenerates **consistently/reliably** (`check_glm_ctx.py`). Below that it is intermittent. |
| **Transient backend instability** at any size | ✅ **Confirmed** | The same 8k prompt looped 4/4 during one window, then was clean 4/4 ~40 min later. The TU GLM deployment enters degraded episodes that loop even small inputs, then self-recovers. |
| GLM-specific (not TU-wide) | ✅ **Likely** | `qwen-3.5-397b` stays coherent at the same sizes and hard-rejects oversized contexts (`400 context length`). |

### The one stable trigger: large input context

GLM-5.2-preview is advertised at ~500k context but its **stable regime is far
below that**. Reliable degeneration at ≥ ~180k input tokens (≈36% of the
advertised window), every run, both direct and via the proxy.

### Context-window note (why "large" looks different per model)

The `262144`-token limit seen in `logs/tu_raw_chat.log` is **qwen's** (error
body says `Model Group=qwen-3.5-397b`), not GLM's. GLM is served at a larger
window (~500k). This is why qwen "seems fine" on huge threads: it rejects them
with a hard `400` rather than serving a degenerate regime.

## Reproducers (run directly, no uniinfer needed)

All scripts read the TU key via `credgoo.get_api_key("tu")` and hit TU Aqueduct
**directly** (model id `glm-5.2-744b-preview`), bypassing uniinfer entirely.

| Script | Purpose |
|---|---|
| `check_model_vs_package.py` | **Decisive A/B**: identical prompt → TU direct vs uniinfer proxy. Proves model fault. |
| `check_glm_ctx.py` | Scales input context (180k / 230k / 245k) to reproduce the loop and find the threshold. |
| `check_max_tokens.py` | Varies `max_tokens` (220 / 8192 / 50000) at fixed input to falsify the max_tokens hypothesis. |
| `check_glm_scale.py` | Earlier scaling probe (8k–100k); clean when backend healthy. |
| `check_glm_docsum.py` | Large coherent-document summarization probe. |
| `check_glm_tools.py` | Tools + large context probe (the glm47 tool-parser path). |
| `check_glm_live.py` | Quick live sanity check (short vs padded prompt). |

The unit tests in `uniinfer/tests/test_glm_smoke.py` cover the uniinfer-side
**leak-repair / streaming** layer in isolation (14 tests, all green) and prove
the package neither causes nor amplifies the loop.

## Run

```bash
cd packages/uniinfer
uv run python investigation/glm5_loop_repro/check_model_vs_package.py
uv run pytest uniinfer/tests/test_glm_smoke.py -v
```

## Recommended mitigations

1. **Keep GLM input context well under ~150k** (auto-compact in the consumer).
   This eliminates the deterministic failure mode.
2. **Detect-and-retry** for transient episodes: abort a stream that shows
   degenerate repetition and resend — usually succeeds on retry. A tested
   `looks_degenerate()` helper exists in the repro scripts, ready to wire into
   the proxy as an early-abort guard.
3. **Report upstream** — see `ISSUE_REPORT.md`.
