# Bug Report: GLM-5.2-preview degenerate repeating-token loops

**Endpoint:** `https://aqueduct.ai.datalab.tuwien.ac.at/v1/chat/completions`
**Model id requested:** `glm-5.2-744b-preview`
**Model id served (from response `model` field):** `asc_amd/zai-org/GLM-5.2-FP8`
**Severity:** High — makes the model unusable for long-context workloads.

## Summary

`glm-5.2-744b-preview` intermittently collapses mid-generation into a
short-unit repeating-token loop. A few real tokens are emitted, then the output
degenerates into one of these patterns until `max_tokens` is hit
(`finish_reason=length`):

```
0\n0\n0\n0\n0\n0…
0%0%0%0%0%0%0…
0,0.0.0.0.0.0.0…
: 0\n: 0\n: 0\n: 0\n…
TheThe::::::::::::…
```

This is reproducible with a plain OpenAI-compatible `POST /chat/completions`
(no client-side processing, no proxy). Output captured verbatim from the API
response/stream.

## Two facets of one flaky instability (not two separate bugs)

Across extensive iteration, **no deterministic client-side trigger was found**.
The degeneration is a transient backend-state phenomenon. Large context raises
its *probability* but is **not a clean cliff**; identical inputs flip between
clean and degenerate over time. The two facets below are the same instability
seen at different probability points.

### Facet 1 — High-probability at large input context

At **input ≥ ~180k tokens** degeneration is frequent, but **not 100%
deterministic**. Same ~92k-token prompt, `temperature=0` (greedy),
`max_tokens=400`, 5 back-to-back runs:

- Window A: degenerate **4/4**
- Window B (~1h later): degenerate **0/5** (clean `OK.` every run)

Example degenerate output, 183k-token single user turn, `temperature=0.2`,
`max_tokens=220`, non-streaming:

```
finish: length
content: '0%0%0%0%0%0%0%0%0%0%0%0%0%0%0%0%0%0%0%0%0…'
```

### Facet 2 — Lower-probability at any context size (including 8k)

During degraded backend windows the deployment loops **even at 8k input**,
then self-recovers. Same identical 8k prompt, `temperature=0.2`,
`max_tokens=400`:

- Window A: degenerate **4/4** (`0\n0\n0…`)
- Window B (≈40 min later): clean **4/4** (`OK`)

The failure is therefore **non-deterministic** from a client's perspective.
A secondary fingerprint: per-request latency for the *same* payload varies
~10× (8s vs 86s at 92k tokens), indicating heavy scheduling/preemption churn.

## What was ruled out (deterministic triggers)

Every deterministic input variable was tested and **none** reliably triggers
the loop:

- **`max_tokens`**: at 8k input, `max_tokens=220` loops identically to
  `max_tokens=50000`. Onset is in the first output tokens.
- **`temperature`**: reproduces at `0.2` and `0` (greedy).
- **Prompt content**: repetitive padding, varied prose, and short natural
  prompts ALL flip between clean/degenerate over time. Repetitive input is
  neither necessary nor sufficient (in fact it was the most *robust* content
  type, clean up to 170k during healthy windows).
- **Conversation structure**: single giant user message vs 842-message
  multi-turn thread — both clean during healthy windows.
- **Input size**: NOT a clean cliff — same size clean 0/5 and degenerate 4/4
  in different time windows.

The only consistent correlate is **time / backend state**.

## Most likely upstream root cause

The fingerprint (non-deterministic, time-varying, strikes at any size, 10×
latency variance for identical input) most strongly indicates a
**preemption / KV-cache restore race** on the FP8 vLLM deployment:

1. **Preemption / KV-cache restore races (primary suspect).** When vLLM
   preempts and restores a request (or a restored snapshot is corrupted), the
   model emits garbage from t≈0 — exactly the immediate `0\n0\n0…` onset
   observed. This explains why the same input is sometimes fine and sometimes
   broken, and why large context (more KV state to evict/restore) raises the
   probability.
2. **FP8 quantization degradation at long context.** The served checkpoint is
   `GLM-5.2-FP8`. FP8 KV-cache / weight quantization can amplify at long
   sequence lengths; worth checking whether bad-state frequency tracks
   quantization precision.
3. **glm47 chat template / tool parser** — same component has a related known
   defect (leaks native `<tool_call>` XML into `content`/`reasoning_content`
   during streaming; vLLM #39757, #42400, #36857; vllm-ascend #8154).

## Reproducer (plain curl/Python, no special client)

```bash
KEY="<aqueduct key>"
python - <<'PY'
import httpx
BLOCK=("The migration working group reviewed the staging cluster logs and found "
       "no anomalies; latency p99 held at 142ms, disk at 63 percent, and the "
       "on-call rotation handled two low-severity tickets. Budget code 4729. "
       "Tea was served. ")
prompt = BLOCK*5500 + "\n\nIgnoring everything above, answer in one short sentence: what is 2+2, and what is the capital of Austria?"
r = httpx.post("https://aqueduct.ai.datalab.tuwien.ac.at/v1/chat/completions",
    headers={"Authorization":"Bearer %s"%__import__("os").environ["KEY"],
             "Content-Type":"application/json"},
    json={"model":"glm-5.2-744b-preview",
          "messages":[{"role":"user","content":prompt}],
          "temperature":0.2,"max_tokens":220,"stream":False},
    timeout=400)
d=r.json(); c=d["choices"][0]
print("served model:", d.get("model"))
print("finish:", c.get("finish_reason"))
print("content:", repr((c.get("message",{}) or {}).get("content"))[:300])
PY
```

Expected at this size during a degraded backend window: `finish=length`,
content begins with a few tokens then collapses into a `0` / `%` / `:`
repeating loop. Because the fault is non-deterministic, the same payload may
return clean `OK.` in a healthy window — **retry across several minutes / at
different times** to observe the failure. (We observed 4/4 degenerate → 0/5
clean ~1h later for the identical payload.)

## Ask

1. Investigate **preemption / KV-cache restore races** on the FP8 vLLM
deployment as the primary suspect (non-deterministic onset at t≈0, 10× latency
variance for identical input).
2. Check whether bad-state frequency tracks **FP8 quantization** (try an
   FP16/BF16 checkpoint at the same lengths).
3. Audit the **glm47 chat template** for long-context issues.
4. If feasible, expose server-side degenerate-output detection so clients can
   fail fast instead of consuming a full `max_tokens` of garbage.
