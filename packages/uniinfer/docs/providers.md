# Provider Index

All providers registered in uniinfer, with configuration and capability details.

## Chat Providers

| ID | Provider | Base URL | Default Model | Credgoo | Conditional | Notes |
|----|----------|----------|---------------|---------|------------|-------|
| `openai` | OpenAI | `https://api.openai.com/v1` | `gpt-3.5-turbo` | `openai` | — | OpenAICompatible base |
| `anthropic` | Anthropic | `https://api.anthropic.com` | `claude-3-sonnet-20240229` | `anthropic` | — | AnthropicCompatible base |
| `gemini` | Google Gemini | — | — | `gemini` | `google-genai` | Custom impl |
| `mistral` | Mistral AI | `https://api.mistral.ai/v1` | — | `mistral` | — | OpenAICompatible base |
| `groq` | Groq | — | — | `groq` | `groq` | Custom impl (groq SDK) |
| `cohere` | Cohere | — | — | `cohere` | `cohere` | Custom impl |
| `openrouter` | OpenRouter | `https://openrouter.ai/api/v1` | `moonshotai/moonlight-16b-a3b-instruct:free` | `openrouter` | — | OpenAICompatible base, free models |
| `ollama` | Ollama | `localhost:11434` | — | — | — | Self-hosted, no key needed |
| `huggingface` | HuggingFace | — | — | `huggingface` | `huggingface-hub` | OpenAICompatible base |
| `cloudflare` | Cloudflare AI Gateway | — | — | `cloudflare` | — | OpenAICompatible base |
| `sambanova` | SambaNova | `https://api.sambanova.ai/v1` | `Meta-Llama-3.1-8B-Instruct` | `sambanova` | — | OpenAICompatible base |
| `pollinations` | Pollinations | `https://gen.pollinations.ai/v1` | `openai` | — | — | Free, no key needed |
| `arli` | Arli AI | `https://api.arliai.com/v1` | `Mistral-Nemo-12B-Instruct-2407` | `arli` | — | OpenAICompatible base |
| `moonshot` | Moonshot AI | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` | `moonshot` | `moonshot` | Custom impl |
| `stepfun` | StepFun | `https://api.stepfun.com/v1` | `step-1-8k` | `stepfun` | — | OpenAICompatible base |
| `upstage` | Upstage | `https://api.upstage.ai/v1/solar` | `solar-pro` | `upstage` | — | OpenAICompatible base |
| `internlm` | InternLM | `https://chat.intern-ai.org.cn/api/v1` | `internlm3-latest` | `internlm` | — | OpenAICompatible base |
| `minimax` | MiniMax | `https://api.minimax.io/anthropic` | `MiniMax-M2.1` | `minimax` | — | AnthropicCompatible base |
| `chutes` | Chutes AI | `https://llm.chutes.ai/v1` | `deepseek-ai/DeepSeek-V3` | `chutes` | — | OpenAICompatible base |
| `zai` | Z.AI | `https://api.z.ai/api/paas/v4` | `glm-4.7` | `zai` | `zai-sdk` | Custom impl (zai SDK) |
| `zai-code` | Z.AI Code | `https://api.z.ai/api/coding/paas/v4` | `glm-4.5` | `zai-code` | `zai-sdk` | Coding-focused endpoint |
| `ngc` | NVIDIA NGC | `https://integrate.api.nvidia.com/v1` | — | `ngc` | — | OpenAICompatible base |
| `ai21` | AI21 Labs | — | — | `ai21` | `ai21` | Custom impl |
| `tu` | TU Wien | `https://aqueduct.ai.datalab.tuwien.ac.at/v1` | — | `tu` | — | Custom impl |
| `tu-staging` | TU Wien (staging) | — | — | `tu` | — | Staging endpoint |

## Embedding Providers

| ID | Provider | Default Model | Notes |
|----|----------|---------------|-------|
| `ollama` | Ollama | — | Self-hosted |
| `tu` | TU Wien | — | |

## TTS / STT Providers

| ID | Provider | Kind | Notes |
|----|----------|------|-------|
| `tu` | TU Wien | TTS | Kokoro voices |
| `tu` | TU Wien | STT | Whisper models |

## Legend

- **Conditional**: Provider only registers if its Python SDK is installed (`uv sync --extra all` to get all)
- **Credgoo**: Service name used to fetch the API key from credgoo
- **Free**: No API key required

## Provider Implementation Patterns

| Pattern | Base Class | Providers |
|---------|------------|-----------|
| OpenAI-compatible | `OpenAICompatibleChatProvider` | openai, mistral, openrouter, sambanova, arli, stepfun, upstage, internlm, chutes, ngc, huggingface, cloudflare |
| Anthropic-compatible | `AnthropicCompatibleProvider` | anthropic, minimax |
| Custom SDK | `ChatProvider` | groq, cohere, moonshot, gemini, ai21, tu, zai, zai-code |
