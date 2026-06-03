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

Ultra-fast LPU (Language Processing Unit) inference. Custom SDK, not OpenAI-compatible base.

- **API docs**: [console.groq.com/docs](https://console.groq.com/docs)
- **Get key**: [console.groq.com/keys](https://console.groq.com/keys)
- **SDK**: `groq` (pip)
- **Free tier**: ✅ Yes — forever free, no credit card required
  - 30 RPM / 6000 TPM (varies by model)
  - 1000 requests/day
  - All models available
  - Rate-limited but no total token cap
- **Notable free models**: Llama 3.3 70B, Llama 4 Scout/Maverick, DeepSeek R1 Distill 70B, Gemma 2 9B, Mixtral 8x7B
- **Reasoning**: ✅ `reasoning_content` support (R1 models)
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Implementation**: Custom (`ChatProvider`), uses `groq.Groq` / `groq.AsyncGroq` SDK clients

<!-- remaining providers TBD -->
