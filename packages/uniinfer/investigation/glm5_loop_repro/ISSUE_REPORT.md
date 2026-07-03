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

## Two distinct failure modes (both reproduced)

### Mode 1 — Deterministic at large input context (the reliable one)

At **input ≥ ~180k tokens** the model degenerates **every run**, regardless of
`max_tokens`, `temperature`, or streaming. GLM-5.2-preview is served at a
large context window (~500k), so this cliff sits at roughly **36% of the
window** — far below any advertised usable length.

Example, 183k-token single user turn, `temperature=0.2`, `max_tokens=220`,
non-streaming:

```
finish: length
content: '0%0%0%0%0%0%0%0%0%0%0%0%0%0%0%0%0%0%0%0%0…'
```

### Mode 2 — Transient episodes at ANY context size (including 8k)

The deployment periodically enters degraded windows where it loops **even at
8k input**, then self-recovers. Same identical 8k prompt, `temperature=0.2`,
`max_tokens=400`:

- Window A: degenerate **4/4** (`0\n0\n0…`)
- Window B (≈40 min later): clean **4/4** (`OK`)

The failure is therefore **non-deterministic** from a client's perspective.

## What was ruled out

- **Not `max_tokens`.** Explicitly falsified: at 8k input, `max_tokens=220`
  loops identically to `max_tokens=50000`. The loop onset is in the first
  output tokens; the cap only controls how long the loop runs.
- **Not `temperature`.** Reproduces at `temperature=0.2` and `temperature=0`
  (greedy).
- **Not the prompt being "repetitive padding".** A short natural prompt is
  clean during healthy windows; the *same* short prompt loops during degraded
  windows. Repetitive input is neither necessary nor sufficient.

## Likely upstream culprits to investigate

1. **FP8 quantization degradation at long context.** The served checkpoint is
   `GLM-5.2-FP8`. FP8 KV-cache / weight quantization is a well-known source
   of quality collapse at long sequence lengths; worth checking whether the
   cliff tracks quantization precision rather than raw position count.
2. **glm47 chat template / tool parser** at long context — the same component
   has a related known defect: it leaks native `<tool_call>` XML into
   `content` / `reasoning_content` during streaming (vLLM #39757, #42400,
   #36857; vllm-ascend #8154). A template/tokenization fault at long context
   is consistent with a hard cliff plus transient episodes.
3. **RoPE / position-embedding scaling** misconfiguration beyond ~180k
   positions.
4. **Preemption / KV-cache restore races** producing the transient
   any-size episodes — the loop output appears at t≈0 (immediate), consistent
   with a broken/restored model state rather than slow drift.

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

Expected at this size: `finish=length`, content begins with a few tokens then
collapses into a `0` / `%` / `:` repeating loop. Reliable every run.

## Ask

1. Confirm the ≥~180k-input cliff and check whether it tracks the **FP8**
   quantization (try an FP16/BF16 checkpoint at the same lengths).
2. Investigate the glm47 template / RoPE scaling at long context.
3. Investigate the transient any-size loop episodes (suspected
   preemption/restart races).
4. If feasible, expose server-side degenerate-output detection so clients can
   fail fast instead of consuming a full `max_tokens` of garbage.
