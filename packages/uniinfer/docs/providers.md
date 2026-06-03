# Provider Index

All providers registered in uniinfer. See [Provider Details](#provider-details) for per-provider docs.

## Chat Providers

| ID | Free Contingent | Free Rate Limits | URL |
|----|---------------|-----------------|-----|
| `groq` | All models, forever | 30 RPM, 100K TPD | [console.groq.com](https://console.groq.com/docs/rate-limits) |
| `openrouter` | Free models available | Per-model | [openrouter.ai](https://openrouter.ai/models) |
| `ollama` | Self-hosted, all free | — | [ollama.com](https://ollama.com) |
| `pollinations` | All models, no key | — | [pollinations.ai](https://pollinations.ai) |
n| `arli` | Qwen-3.5-27B | 1 req at a time, 12K ctx | [arliai.com/pricing](https://www.arliai.com/pricing?lang=en) |
| `zai` | glm-4.5-flash | — | [z.ai](https://z.ai) |
| `openai` | — | — | [platform.openai.com](https://platform.openai.com) |
| `anthropic` | — | — | [docs.anthropic.com](https://docs.anthropic.com) |
| `gemini` | — | — | [ai.google.dev](https://ai.google.dev) |
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

<!-- remaining providers TBD -->
