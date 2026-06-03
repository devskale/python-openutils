# Provider Index

All providers registered in uniinfer, with configuration and capability details.

## Chat Providers

| ID | Provider | Base URL | Credgoo | Free | Conditional |
|----|----------|----------|---------|------|------------|
| `openai` | OpenAI | `https://api.openai.com/v1` | `openai` | — | — |
| `anthropic` | Anthropic | `https://api.anthropic.com` | `anthropic` | — | — |
| `gemini` | Google Gemini | — | `gemini` | — | `google-genai` |
| `mistral` | Mistral AI | `https://api.mistral.ai/v1` | `mistral` | — | — |
| `groq` | Groq | — | `groq` | ✅ | `groq` |
| `cohere` | Cohere | — | `cohere` | — | `cohere` |
| `openrouter` | OpenRouter | `https://openrouter.ai/api/v1` | `openrouter` | ✅ | — |
| `ollama` | Ollama | `localhost:11434` | — | ✅ | — |
| `huggingface` | HuggingFace | — | `huggingface` | — | `huggingface-hub` |
| `cloudflare` | Cloudflare AI Gateway | — | `cloudflare` | — | — |
| `sambanova` | SambaNova | `https://api.sambanova.ai/v1` | `sambanova` | — | — |
| `pollinations` | Pollinations | `https://gen.pollinations.ai/v1` | — | ✅ | — |
| `arli` | Arli AI | `https://api.arliai.com/v1` | `arli` | — | — |
| `moonshot` | Moonshot AI | `https://api.moonshot.cn/v1` | `moonshot` | — | `moonshot` |
| `stepfun` | StepFun | `https://api.stepfun.com/v1` | `stepfun` | — | — |
| `upstage` | Upstage | `https://api.upstage.ai/v1/solar` | `upstage` | — | — |
| `internlm` | InternLM | `https://chat.intern-ai.org.cn/api/v1` | `internlm` | — | — |
| `minimax` | MiniMax | `https://api.minimax.io/anthropic` | `minimax` | — | — |
| `chutes` | Chutes AI | `https://llm.chutes.ai/v1` | `chutes` | — | — |
| `zai` | Z.AI | `https://api.z.ai/api/paas/v4` | `zai` | ✅ | `zai-sdk` |
| `zai-code` | Z.AI Code | `https://api.z.ai/api/coding/paas/v4` | `zai-code` | — | `zai-sdk` |
| `ngc` | NVIDIA NGC | `https://integrate.api.nvidia.com/v1` | `ngc` | — | — |
| `ai21` | AI21 Labs | — | `ai21` | — | `ai21` |
| `tu` | TU Wien | `https://aqueduct.ai.datalab.tuwien.ac.at/v1` | `tu` | — | — |
| `tu-staging` | TU Wien (staging) | — | `tu` | — | — |

## Embedding Providers

| ID | Provider | Free | Notes |
|----|----------|------|-------|
| `ollama` | Ollama | ✅ | Self-hosted |
| `tu` | TU Wien | — | |

## TTS / STT Providers

| ID | Provider | Kind | Free | Notes |
|----|----------|------|------|-------|
| `tu` | TU Wien | TTS | — | Kokoro voices |
| `tu` | TU Wien | STT | — | Whisper models |

---

## Provider Details

### groq — Groq

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

<!-- remaining providers TBD -->
