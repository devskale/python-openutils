# Issues — GLM-5.2-744b-preview on TU Aqueduct (vLLM)

Tracking the known problems with `glm-5.2-744b-preview` (served by TU Aqueduct on
vLLM-Ascend, model `asc_amd/zai-org/GLM-5.2-FP8`) and their partial fixes.

> **Module renames since these notes were written:** `uniioai.py` →
> `provider_access.py` (key/embeddings/listing helpers) and `uniioai_proxy.py`
> → `proxy_app.py` (the FastAPI app). The completion functions
> (`get/stream/aget/astream_completion`) moved to `uniinfer.completion.Target`.
> See [ARCHITECTURE.md](../ARCHITECTURE.md#naming).

Status legend: ✅ done · 🟡 partial / work-in-progress · ⬜ open

---

## ✅ 1. Streaming tool-call leak — interceptor (fixed + deployed)

**Symptom:** During streaming, GLM-5.2's tool calls intermittently leak into
`content` as raw XML instead of structured `tool_calls`. The model genuinely
intends the call (verified against its chat template), but vLLM's streaming
parser fails to intercept it.

Example leaked format (the model's documented native format):
```
...prose...<tool_call>bash<arg_key>command</arg_key><arg_value>echo "ok"</arg_value></tool_call>
```
Often the opening `<tool_call>` tag arrives truncated, sticking the tool name
to preceding prose (e.g. `wherebash<arg_key>...`).

**Root cause (upstream):** vLLM `glm47` tool parser desyncs on GLM-5.2's XML
format during streaming, especially after long content generation. Known bugs:
- vLLM #39757 — streaming truncates tool names
- vLLM #42400 — intermittent parse failures, amplified at long context
- vLLM #36857 — streaming tool args leak as content
- vllm-ascend #8154 — same on Ascend

**Current state — 🟡 interceptor built + deployed but ONE BUG REMAINING:**

`uniinfer/proxy_services/glm_leak_repair.py` detects the leaked XML, suppresses
it from `content`, and reconstructs a structured `tool_calls` entry. Wired into
`uniinfer/proxy_services/streaming.py` (both the dict and object chunk paths).

**Status: ✅ fixed, tested, deployed.** The reconstructed tool-call delta now
includes `"index": 0`, so pi (and any OpenAI-style client) coalesces the
arguments correctly. Verified via `/tmp/test_e2e_stream.py` (asserts
`index == 0` and non-empty args) and `/tmp/test_glm_leak.py` (3 real samples +
2 edge cases). Proxy restarted with the fix.

Earlier symptom (now resolved):
```
Validation failed for tool "bash":
  - command: must have required properties command
Received arguments: {}
```

**The fix** (in `glm_leak_repair.py`): each reconstructed tool call carries
`"index": 0` so streaming-delta coalescing routes the args to the right slot
(pi CHANGELOG #3576 documents this exact coalescing requirement).

---

## ⬜ 2. `thinkingFormat` config (Path A) — did it help?

`~/.pi/agent/models.json` for `tu@glm-5.2-744b-preview` was given:
```json
"compat": {
  "thinkingFormat": "qwen-chat-template",
  "supportsReasoningEffort": false,
  "supportsUsageInStreaming": false
}
```
Intent: make pi send `chat_template_kwargs.enable_thinking` (accepted by TU)
instead of `reasoning_effort` (rejected by TU with `UnsupportedParamsError`).

**Result:** leak still occurred after applying this. So config alone does NOT
fix the streaming tool-call leak. The interceptor (issue #1) is still needed.

**Open question — worth revisiting:** does `qwen-chat-template` actually change
the leak *frequency*, or the reasoning-token cost? Not measured. Could compare:
- count of leaks per N tool-call requests, with vs without the compat block.
- reasoning_content token usage in both configs.

If it has no measurable effect, consider switching `thinkingFormat` to `"zai"`
(pi's documented GLM format, uses `reasoning_effort`) — but TU rejects
`reasoning_effort`, so that would need proxy-side translation.

---

## 🟡 3. `tool_choice="required"` rejected (intentional)

TU's `required` path uses vLLM constrained decoding, which 500s on GLM-5.2's
reasoning/tool parser. We **do not** support it — `_prepare_payload()` in
`uniinfer/providers/tu.py` raises a clear `ValueError` → HTTP 400:
```
tool_choice='required' is not supported by the TU provider
(upstream vLLM constrained-decoding bug on reasoning models).
Use tool_choice='auto', or a named tool_choice to force a specific tool.
```

**This is uniform across all TU models** (incl. qwen-3.5, where it technically
worked) — chosen over a silent rewrite to avoid wrong-tool-call behavior.

**If qwen consumers need `required` back:** gate the rejection on model name
(match `glm-5.2` / a `REASONING_MODELS` set) in `tu.py`. Not done.

---

## ✅ 4. `chat_template_kwargs` passthrough — DONE

`chat_template_kwargs` is now wired through the full stack so callers can
control thinking on Qwen3.x / GLM-5.x via the proxy and the uniinfer CLI/SDK.

**Threaded through:**
- `uniinfer/core.py` — `ChatCompletionRequest(chat_template_kwargs=...)`
- `uniinfer/providers/tu.py` — `_prepare_payload()` forwards it to the payload
- `uniinfer/uniioai.py` — `get_completion`, `stream_completion`,
  `aget_completion`, `astream_completion` all accept and forward it
- `uniinfer/proxy_schemas/chat.py` — `ChatCompletionRequestInput.chat_template_kwargs`
- `uniinfer/proxy_routers/chat.py` — passes `request_input.chat_template_kwargs`
  into both the streaming and non-streaming paths
- `uniinfer/proxy_services/streaming.py` — `astream_response_generator` forwards
  it to `astream_completion`

**Verified end-to-end through the proxy** (`tu@qwen-3.6-35b`, max_tokens 2048):

| Config | reasoning_tokens | completion_tokens |
|---|---|---|
| `chat_template_kwargs.enable_thinking=false` | **0** | 12 |
| default (thinking ON) | 582 | 202 |

**Important correction** (verified live + confirmed by vLLM #35574): the
top-level `enable_thinking` field is **silently IGNORED** by these backends —
it returns 200 but still produces 2504 reasoning tokens. Only
`chat_template_kwargs.enable_thinking` actually works. The proxy schema's
older `think` field is also not honored by TU.

**Usage via the proxy:**
```json
{
  "model": "tu@qwen-3.6-35b",
  "messages": [...],
  "chat_template_kwargs": {"enable_thinking": false}
}
```
**Usage via pi:** set `"compat": {"thinkingFormat": "qwen-chat-template"}` on
the model in `models.json`; pi then emits this field for you.

**Known vLLM trap (vLLM #35574):** top-level `enable_thinking: false` is
accepted but does nothing. Always use `chat_template_kwargs`.

---

## ⬜ 5. `max_tokens` starvation on reasoning models

GLM-5.2 (reasoning model) spends substantial tokens on `reasoning_content`
(observed: 76–1200 tokens) which count toward `max_tokens`. A caller with a
small `max_tokens` (e.g. 256) gets `finish_reason="length"` with **empty
content** — the whole budget went to thinking.

**Mitigations (no code change needed):**
- Raise `max_tokens` (current models.json value: 65536 — fine).
- Disable thinking via issue #4's passthrough when the task doesn't need it.

---

## ⬜ 6. Other unfixed upstream GLM-5.x vLLM instabilities (not ours to fix)

- Intermittent non-streaming 500s / `Connection error`
- vLLM #39757 streaming tool-name truncation (the `�bash` we saw)
- vLLM #36857 streaming args returned in final chunk, not incrementally

These only get fixed at the TU Aqueduct / vLLM layer. Worth reporting upstream
with the evidence collected here.

---

## Reference: relevant files

| File | Role |
|------|------|
| `uniinfer/core.py` | `ChatCompletionRequest` now has `chat_template_kwargs` (issue #4) |
| `uniinfer/providers/tu.py` | TU provider `_prepare_payload()` — `tool_choice` guard (issue #3), forwards `chat_template_kwargs` (issue #4) |
| `uniinfer/uniioai.py` | `get/stream/aget/astream_completion` forward `chat_template_kwargs` (issue #4) |
| `uniinfer/proxy_services/glm_leak_repair.py` | The leak interceptor (issue #1) |
| `uniinfer/proxy_services/streaming.py` | Interceptor wiring + `chat_template_kwargs` passthrough (issues #1, #4) |
| `uniinfer/proxy_schemas/chat.py` | `ChatCompletionRequestInput.chat_template_kwargs` (issue #4) |
| `uniinfer/proxy_routers/chat.py` | Forwards `chat_template_kwargs` to stream + non-stream paths (issue #4) |
| `~/.pi/agent/models.json` | pi model config + `compat` block (issue #2) |
| `/tmp/test_glm_leak.py`, `/tmp/test_e2e_stream.py` | test harnesses (note: in /tmp) |

## Reference: pi findings that informed this

- pi has first-class GLM-5.2 support; `thinkingFormat` options include
  `zai`, `qwen`, `qwen-chat-template`, `chat-template`.
- pi CHANGELOG #3325: same bug class ("thinking model degrades tool-call args
  to `{}`") fixed for Qwen via thinking replay.
- pi CHANGELOG #3576: OpenAI streaming **coalesces tool-call deltas by stable
  `index`** — the root of the empty-args bug in issue #1.
- pi CHANGELOG #3175: hardened Anthropic streaming against malformed tool-call
  JSON with defensive repair — precedent for client-side reconstruction.
