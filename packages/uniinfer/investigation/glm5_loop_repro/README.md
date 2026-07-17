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
| Driven by **input context size** | ⚠️ **Correlation only** | Large context raises the *probability* of degeneration, but is **NOT a deterministic cliff**. The same ~92k config was degenerate **4/4** in one window, clean **0/5** an hour later (`check_glm_ctx.py`, this dir). |
| **Transient backend instability** at any size | ✅ **Dominant factor (confirmed)** | Across every deterministic variable tested (size 8k–230k, repetitive/varied content, single-turn vs 842-message multi-turn, `max_tokens` 220–50000, temp 0.0–0.2), results flip between clean and degenerate **over time** for identical inputs. Same 8k prompt: 4/4 degenerate then 4/4 clean ~40 min later. This fingerprints a serving-layer (preemption / KV-cache restore) instability, not an input property. |
| GLM-specific (not TU-wide) | ✅ **Likely** | `qwen-3.5-397b` stays coherent at the same sizes and hard-rejects oversized contexts (`400 context length`). |

### The dominant factor: transient backend instability

After iteration, **no deterministic client-side trigger was found**. The
degeneration is a transient backend-state phenomenon on TU's GLM-5.2-FP8
(`asc_amd/zai-org/GLM-5.2-FP8`) deployment — consistent with preemption /
KV-cache restore races. Evidence: identical inputs flip between clean and
degenerate across time windows; per-request latency for the *same* payload
varies 10× (8s vs 86s), another fingerprint of scheduling churn.

Large input context raises the **probability** of hitting a bad state (the
first reproductions all happened at ≥180k), so capping context remains the
best practical mitigation — but it is not a clean cliff and cannot be relied
on as a deterministic test.

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
