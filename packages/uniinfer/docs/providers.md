# Provider Index

All providers registered in uniinfer. See [Provider Details](#provider-details) for per-provider docs.

## Chat Providers

| ID | Free Contingent | Free Rate Limits | URL |
|----|---------------|-----------------|-----|
| `groq` | All models, forever | 30 RPM, 100K TPD | [console.groq.com](https://console.groq.com/docs/rate-limits) |
| `openrouter` | 25+ `:free` models; deposit ≥$10 → 1000 req/day | 50 req/day (free) / 1000 req/day (with credits) | [openrouter.ai/pricing](https://openrouter.ai/pricing) |
| `ollama` | Self-hosted, all free | — | [ollama.com](https://ollama.com) |
| `pollinations` | All models, no key required; sk_ = no limits | pk_: 1 pollen/hour/IP; sk_: unlimited | [gen.pollinations.ai/docs](https://gen.pollinations.ai/docs) |
| `arli` | Qwen-3.5-27B | 1 req at a time, 12K ctx | [arliai.com/pricing](https://www.arliai.com/pricing?lang=en) |
| `zai` | glm-4.5-flash | — | [z.ai](https://z.ai) |
| `openai` | — | — | [platform.openai.com](https://platform.openai.com) |
| `anthropic` | — | — | [docs.anthropic.com](https://docs.anthropic.com) |
| `gemini` | Flash: 1,500 RPD; Pro: 50 RPD (trial) | Flash: 15 RPM / 1M TPM | [ai.google.dev/pricing](https://ai.google.dev/pricing) |
| `mistral` | — | — | [docs.mistral.ai](https://docs.mistral.ai) |
| `cohere` | — | — | [docs.cohere.com](https://docs.cohere.com) |
| `huggingface` | — | — | [huggingface.co](https://huggingface.co) |
| `cloudflare` | — | — | [developers.cloudflare.com](https://developers.cloudflare.com) |
| `sambanova` | — | — | [sambanova.ai](https://sambanova.ai) |
| `moonshot` | — | — | [platform.moonshot.cn](https://platform.moonshot.cn) |
| `stepfun` | — | — | [platform.stepfun.com](https://platform.stepfun.com) |\n| `upstage` | — | — | [upstage.ai](https://upstage.ai) |
| `internlm` | — | — | [internlm.ai](https://internlm.ai) |
| `minimax` | — | — | [minimaxi.com](https://minimaxi.com) |\n| `chutes` | — | — | [chutes.ai](https://chutes.ai) |\n| `zai-code` | — | — | [z.ai](https://z.ai) |\n| `ngc` | — | — | [build.nvidia.com](https://build.nvidia.com) |\n| `ai21` | — | — | [ai21.com](https://ai21.com) |\n| `tu` | — | — | — |\n| `tu-staging` | — | — | — |

## Embedding Providers

| ID | Free | URL |
|----|------|-----|
| `ollama` | ✅ | [ollama.com](https://ollama.com) |
| `tu` | — | — |

## TTS / STT Providers

| ID | Kind | Free | URL |
|----|------|------|-----|
| `tu` | TTS | — | — |
| `tu` | STT | — | — |

---

## Provider Details

### groq — Groq *(info: 2026-06-03)*

Ultra-fast LPU (Language Processing Unit) chip inference. Custom SDK, not OpenAI-compatible base.

- **API docs**: [console.groq.com/docs](https://console.groq.com/docs)
- **Rate limits**: [console.groq.com/docs/rate-limits](https://console.groq.com/docs/rate-limits)
- **Get key**: [console.groq.com/keys](https://console.groq.com/keys) (free, no credit card)
- **SDK**: `groq` (pip)
- **Free tier**: ✅ Forever free, no credit card required
  - All models available on free tier
  - Limits are per-minute (RPM/TPM) and per-day (RPD/TPD) — whichever you hit first
  - Rate limit headers included in every response (`x-ratelimit-remaining-tokens`, etc.)
  - 429 response with `retry-after` header when exceeded
- **Free tier rate limits** (per org, whichever cap hit first):

  | Model | RPM | TPM | TPD |
  |-------|-----|-----|-----|
  | `llama-3.3-70b-versatile` | 30 | 12K | 100K |
  | `llama-4-scout-17b-16e-instruct` | 30 | 30K | 500K |

  Full limits: [console.groq.com/docs/rate-limits](https://console.groq.com/docs/rate-limits)

- **Reasoning**: ✅ `reasoning_content` support (DeepSeek R1 distill models)
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Implementation**: Custom (`ChatProvider`), uses `groq.Groq` / `groq.AsyncGroq` SDK clients

### arli — Arli AI *(info: 2026-06-03)*

OpenAI-compatible inference with "derestricted" models (uncensored fine-tunes).

- **API docs**: [www.arliai.com](https://www.arliai.com)
- **Pricing**: [www.arliai.com/pricing](https://www.arliai.com/pricing?lang=en)
- **Free tier**: ✅ One free model with unlimited tokens/requests
  - `Qwen-3.5-27B-Derestricted` — 12K context, 1 request at a time, delayed response
  - 5 trial requests per 2 days for all other models
- **Reasoning**: ✅ (select models)
- **Vision**: ✅ (select models)
- **Tools**: ✅ (OpenAI-compatible)
- **Streaming**: ✅
- **Implementation**: `OpenAICompatibleChatProvider`, custom `list_models()` from `/v1/models/textgen-models`

### openrouter — OpenRouter *(info: 2026-06-03)*

Unified API router for 400+ models from 60+ providers. OpenAI-compatible.

- **API docs**: [openrouter.ai/docs](https://openrouter.ai/docs)
- **Pricing**: [openrouter.ai/pricing](https://openrouter.ai/pricing)
- **Models browser**: [openrouter.ai/models](https://openrouter.ai/models)
- **Free tier**: ✅ 25+ free models (`:free` suffix), no credit card required
  - Free: 50 req/day, 20 RPM
  - With ≥$10 credits: free model limit jumps to **1,000 req/day**
  - Credits are a deposit (5.5% fee), deducted per-token at provider pass-through prices
  - Credits unlock the full 400+ paid model catalog
- **Notable free models**:

  | Model | Context | Notes |
  |-------|---------|-------|
  | `google/gemma-4-31b-it:free` | 262K | Vision + tools |
  | `openai/gpt-oss-120b:free` | 131K | Tools |
  | `meta-llama/llama-3.3-70b-instruct:free` | 131K | Tools |
  | `qwen/qwen3-coder:free` | 1M | Tools |
  | `z-ai/glm-4.5-air:free` | 131K | Tools |
  | `openrouter/free` | 200K | Auto-selects random free model |

  Full list: [openrouter.ai/openrouter/free](https://openrouter.ai/openrouter/free)
- **Reasoning**: ✅ (select models)
- **Vision**: ✅ (select models)
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Implementation**: `OpenAICompatibleChatProvider`

### pollinations — Pollinations *(info: 2026-06-03)*

Multi-modal generation platform (text, image, video, audio) with an OpenAI-compatible endpoint. No key needed for basic use.

- **API docs**: [gen.pollinations.ai/docs](https://gen.pollinations.ai/docs)
- **API reference**: [pollinations-ai.com/api.html](https://pollinations-ai.com/api.html)
- **Get key**: [enter.pollinations.ai](https://enter.pollinations.ai/) (free, no credit card)
- **Free tier**: ✅ Fully free, no key required for basic requests
  - **No key**: Works out of the box, best-effort rate limits
  - **Publishable key** (`pk_`): 1 pollen/hour per IP+key (client-side, demos)
  - **Secret key** (`sk_`): **No rate limits** (server-side, recommended)
  - Two key types: publishable (`pk_`) for client-side, secret (`sk_`) for server-side
- **Modalities**: Text (GPT-5, Claude, Gemini, DeepSeek V3.2, Qwen3-Coder), Image (Flux, GPT Image), Video (Seedance, Veo), Audio (TTS/STT)
- **Reasoning**: ✅ (select models)
- **Vision**: ✅ (select models)
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Implementation**: `OpenAICompatibleChatProvider`, custom `list_models()` tries multiple endpoints

### gemini — Google Gemini *(info: 2026-06-03)*

Google's flagship LLM with native SDK (`google-genai`). Not OpenAI-compatible base — custom implementation. Also offers an [OpenAI compatibility layer](https://ai.google.dev/gemini-api/docs/openai).

- **API docs**: [ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs)
- **Pricing**: [ai.google.dev/pricing](https://ai.google.dev/pricing)
- **Rate limits**: [ai.google.dev/gemini-api/docs/rate-limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- **Get key**: [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (free, no credit card)
- **SDK**: `google-genai` (pip)
- **Free tier**: ✅ Most generous free tier among major providers
  - **No credit card**, no expiration, permanent free tier
  - Google may use free-tier requests for model improvement (paid tier does not)
- **Free tier rate limits**:

  | Model | RPM | RPD | TPM |
  |-------|-----|-----|-----|
  | `gemini-2.5-flash` | 15 | 1,500 | 1M |
  | `gemini-2.5-flash-lite` | 30 | 1,500 | 1M |
  | `gemini-2.5-pro` | 5 | 50 | 1M (trial-only) |
  | `gemma-3` | 30 | 1,500 | 1M |

  Full limits: [ai.google.dev/pricing](https://ai.google.dev/pricing)
- **Reasoning/Thinking**: ✅ native thinking support (`thought` parts in response)
- **Vision**: ✅ multimodal input (images, audio, video, PDFs)
- **Tools**: ✅ function calling + tool config (AUTO/ANY/NONE mode)
- **Streaming**: ✅ native async streaming
- **Extra capabilities**: Grounding with Google Search, code execution, long context (up to 1M), embeddings, JSON mode, Live API
- **Implementation**: Custom `ChatProvider`, uses `google.genai.Client` / `client.aio` for async

<!-- remaining providers TBD -->
