# Provider Index

## Chat Providers

| ID | Free Contingent | Free Rate Limits | URL |
|----|----------------|-----------------|-----|
| `groq` | All models, forever free | 30 RPM, 100K TPD (70B) | [console.groq.com](https://console.groq.com/docs/rate-limits) |
| `openrouter` | 25+ `:free` models; ≥$10 deposit → 1000 req/day | Free: 50 req/day; w/ credits: 1000/day | [openrouter.ai/pricing](https://openrouter.ai/pricing) |
| `pollinations` | All models, no key needed | `pk_`: 1 pollen/hr/IP; `sk_`: unlimited | [gen.pollinations.ai/docs](https://gen.pollinations.ai/docs) |
| `ollama` | Self-hosted: unlimited; Cloud: light usage free | Local: unlimited; Cloud Free: 1 concurrent | [ollama.com/pricing](https://ollama.com/pricing) |
| `arli` | Qwen-3.5-27B-Derestricted, unlimited | 1 concurrent req, 12K ctx | [arliai.com/pricing](https://www.arliai.com/pricing?lang=en) |
| `gemini` | Flash 2.x: 1,500 RPD; **3.5-flash: 20 RPD** ⚠️; Pro: 50 RPD (trial) | Flash: 10-15 RPM / 1M TPM | [ai.google.dev/pricing](https://ai.google.dev/pricing) |
| `zai` | `glm-4.5-flash` + `glm-4.7-flash` + `glm-4.6v-flash` permanent | Free Flash: no explicit limit (fair use) | [docs.z.ai](https://docs.z.ai) |
| `mistral` | All models (Experiment mode); ~1B tokens/mo | 2 RPM all models | [docs.mistral.ai](https://docs.mistral.ai/admin/user-management-finops/tier) |
| `cohere` | Trial: 5 RPM, 100K calls/mo, all models | 5 RPM (Trial); 500 RPM (Production) | [docs.cohere.com/docs/rate-limits](https://docs.cohere.com/docs/rate-limits) |
| `huggingface` | Serverless: free, ~100s req/hr, models **<10GB** | Few hundred req/hr; PRO $9/mo: higher | [huggingface.co/docs/inference-providers](https://huggingface.co/docs/inference-providers) |
| `cloudflare` | 10K Neurons/day (~20–60 req for 70B) | Varies by model size | [developers.cloudflare.com/workers-ai/platform/pricing](https://developers.cloudflare.com/workers-ai/platform/pricing) |
| `sambanova` | $5 credits (3mo) + ongoing 20 RPM / 200K TPD | Free: 20 RPM; Dev: 240 RPM (70B) | [docs.sambanova.ai](https://docs.sambanova.ai/docs/en/models/rate-limits) |
| `ngc` | Developer Program: ~40 RPM, 100+ models | ~40 RPM (varies per model) | [build.nvidia.com](https://build.nvidia.com) |
| `moonshot` | Trial credits + ~3 RPM / 40K TPM | Free: ~3 RPM; Paid: 60 RPM | [platform.moonshot.ai/docs/pricing](https://platform.moonshot.ai/docs/pricing) |
| `stepfun` | V0: 10 RPM / 5M TPM (no payment) | V0: 10 RPM; V1 ($15): 1K RPM | [platform.stepfun.ai/docs/en/guides/pricing/details](https://platform.stepfun.ai/docs/en/guides/pricing/details) |
| `internlm` | V0: 10 RPM / 5M TPM (no payment) | V0: 10 RPM; V1 ($15): 1K RPM | [chat.intern-ai.org.cn](https://chat.intern-ai.org.cn) |
| `upstage` | `solar-pro-3:free` (MoE 102B/12B active) | Varies by model | [upstage.ai](https://upstage.ai) |
| `minimax` | Trial credits; M3 open-weight imminent | Sub: $20–120/mo; or self-host M3 | [minimax.io](https://minimax.io) |
| `ai21` | Trial credits only | Contact sales for limits | [ai21.com](https://ai21.com) |
| `chutes` | ❌ No free tier; pay-per-token only | Varies by model | [chutes.ai/pricing](https://chutes.ai/pricing) |
| `openai` | $5 credits (3mo); GPT-5 needs paid tier | Free: 3 RPM / 200 RPD | [platform.openai.com/api/docs/pricing](https://platform.openai.com/api/docs/pricing) |
| `anthropic` | ~$5 credits (signup); no recurring free | Tier 1: 50 RPM / varies by model | [docs.anthropic.com/en/api/rate-limits](https://docs.anthropic.com/en/api/rate-limits) |
| `opencode` | Free models: `deepseek-v4-flash-free`, `big-pickle`, `mimo-v2.5-free`, `hy3-free`, `nemotron-3-ultra-free`, `north-mini-code-free` | Router (aggregates many models); many free models reason | [opencode.ai/zen](https://opencode.ai/zen) |
| `kilo` | 12+ free models usable **anonymously** (`tencent/hy3:free`, `nvidia/nemotron-3-ultra-550b-a55b:free`, `cohere/north-mini-code:free`, `poolside/laguna-*:free`, …) | Free: 200 req/hr per IP (anonymous); paid: per-key | [kilo.ai/gateway](https://kilo.ai/gateway) |

## Embedding Providers

| ID | Free | URL |
|----|------|-----|
| `ollama` | ✅ | [ollama.com](https://ollama.com) |

## TTS / STT Providers

| ID | Kind | Free | URL |
|----|------|------|-----|
| `cloudflare` | TTS/STT | ✅ (Whisper, Aura) | [developers.cloudflare.com/workers-ai](https://developers.cloudflare.com/workers-ai) |
| `stepfun` | TTS/ASR | Paid | [platform.stepfun.ai/docs](https://platform.stepfun.ai/docs) |

---

## Provider Details

### groq — Groq

Ultra-fast LPU inference. Forever free, no CC. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [console.groq.com/docs](https://console.groq.com/docs) · [get key](https://console.groq.com/keys) (free, no CC)
- **Free**: all models, rate limit headers in every response
- **Implementation**: Custom `ChatProvider`, uses `groq` SDK

### arli — Arli AI

OpenAI-compatible "derestricted" (uncensored) fine-tunes. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [arliai.com](https://www.arliai.com) · [pricing](https://www.arliai.com/pricing?lang=en)
- **Free**: `Qwen-3.5-27B-Derestricted` — 12K ctx, 1 concurrent req. 5 trial requests / 2 days for other models
- **Implementation**: `OpenAICompatibleChatProvider`

### openrouter — OpenRouter

Unified API for 400+ models from 60+ providers. Zero markup on paid models. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [openrouter.ai/docs](https://openrouter.ai/docs) · [pricing](https://openrouter.ai/pricing)
- **Free**: 25+ `:free` models. Notable: `google/gemma-4-31b-it:free` (262K ctx), `qwen/qwen3-coder:free` (1M ctx), `meta-llama/llama-3.3-70b-instruct:free`
- **Implementation**: `OpenAICompatibleChatProvider`

### pollinations — Pollinations

Multi-modal (text, image, video, audio). No key needed. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [gen.pollinations.ai/docs](https://gen.pollinations.ai/docs) · [get key](https://enter.pollinations.ai/) (free)
- **Free**: fully free. `pk_` (publishable): 1 pollen/hr/IP. `sk_` (secret): **no rate limits**
- **Implementation**: `OpenAICompatibleChatProvider`

### kilo — Kilo Gateway

Unified OpenAI/OpenRouter-compatible API for 300+ models from 60+ providers (Anthropic, OpenAI, Google, xAI, DeepSeek, Qwen, NVIDIA, …). Free models work **without a key**. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [kilo.ai/docs/gateway](https://kilo.ai/docs/gateway) · [models](https://kilo.ai/docs/gateway/models-and-providers) · [get key](https://kilo.ai/) (paid models only)
- **Base URL**: `https://api.kilo.ai/api/gateway` · **Credgoo service**: `kilocode`
- **Free**: 12+ models usable anonymously — 200 req/hr per IP. Notable: `tencent/hy3:free` (295B MoE, 262K ctx), `nvidia/nemotron-3-ultra-550b-a55b:free` (1M ctx), `cohere/north-mini-code:free`, `poolside/laguna-m.1:free`, `stepfun/step-3.7-flash:free`, `kwaipilot/kat-coder-pro-v2.5:free`.
- **Auto tiers**: `kilo-auto/frontier|balanced|free|small|efficient` — server-side routing; underlying model can change.
- **⚠️ NVIDIA free endpoints** (`nvidia/*:free`): trial use only — prompts/outputs are logged by NVIDIA for service improvement. Do not send confidential data.
- **Implementation**: `OpenAICompatibleChatProvider` (`REQUIRES_API_KEY = False` — free models work without a key)

### gemini — Google Gemini

Most generous free tier among major providers. `google-genai` SDK (not OpenAI-compatible base). · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs) · [get key](https://aistudio.google.com/apikey) (free, no CC)
- **Free**: permanent, no CC. Google may use free-tier requests for model improvement.

  | Model | RPM | RPD | TPM | Notes |
  |-------|-----|-----|-----|-------|
  | `gemini-2.5-flash` | 15 | 1,500 | 1M | Best value |
  | `gemini-3.5-flash` | 10 | **20** | 1M | ⚠️ Only 20 RPD! |

  **⚠️ Critical**: `gemini-3.5-flash` has only **20 RPD** — confirmed by live 429 errors. Other Flash models have 1,500 RPD.

- **Rate limit introspection**:
  ```python
  except RateLimitError as e:
      print(e.quota_metric)  # "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
      print(e.quota_limit)   # 20
  ```
- **Implementation**: Custom `ChatProvider`, `google.genai.Client`

### openai — OpenAI

Reference OpenAI-compatible protocol. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [platform.openai.com/api/docs](https://platform.openai.com/api/docs) · [get key](https://platform.openai.com/api-keys)
- **Free**: $5 credits (3 months). GPT-5+ requires paid Tier 1 ($5 top-up: 500 RPM)
- **Implementation**: `OpenAICompatibleChatProvider`

### anthropic — Anthropic (Claude)

Custom Messages API. Phone verification required. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [docs.anthropic.com](https://docs.anthropic.com/en/docs/intro) · [get key](https://console.anthropic.com)
- **Free**: ~$5 signup credits. No recurring free tier.
- **Tier 1** ($5): 50 RPM · Prompt caching doesn't count toward rate limits
- **Implementation**: `OpenAICompatibleChatProvider`

### mistral — Mistral AI

Paris-based EU lab. GDPR/data residency. Open-weight models. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [docs.mistral.ai](https://docs.mistral.ai) · [get key](https://console.mistral.ai)
- **Free** ("Experiment"): all models, 2 RPM, ~1B tokens/month
- **Implementation**: `OpenAICompatibleChatProvider`

### huggingface — HuggingFace

ML hub. Serverless (free), Endpoints (dedicated GPU), Inference Providers (gateway). · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [huggingface.co/docs/inference-providers](https://huggingface.co/docs/inference-providers) · [get key](https://huggingface.co/settings/tokens)
- **Free** (Serverless): ~few hundred req/hr, models **<10GB**. Cold starts: 10–30s.
- **Implementation**: Custom `ChatProvider`, `huggingface_hub.InferenceClient`

### cloudflare — Cloudflare Workers AI

Edge inference at 300+ locations. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [developers.cloudflare.com/workers-ai](https://developers.cloudflare.com/workers-ai) · [get key](https://dash.cloudflare.com)
- **Free**: 10,000 Neurons/day. Also: embeddings, image gen, TTS/STT
- **Implementation**: Custom `ChatProvider`, REST to Workers AI gateway

### cohere — Cohere

Enterprise-focused. Best-in-class RAG. · 🧠 ✅ · 👁️ ❌ · 🔧 ✅ · 📡 ✅

- **Docs**: [docs.cohere.com](https://docs.cohere.com) · [get key](https://dashboard.cohere.com)
- **Free** (Trial): 5 RPM, 100K calls/month, all models
- **Implementation**: Custom `ChatProvider`

### ngc — NVIDIA NIM

100+ models on NVIDIA GPUs. OpenAI-compatible. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [build.nvidia.com](https://build.nvidia.com) (NVIDIA Developer Program, phone verification)
- **Free** (Developer): ~40 RPM, 100+ models. Downloadable NIM containers (free dev on ≤16 GPUs)
- **Implementation**: `OpenAICompatibleChatProvider`

### zai — Zhipu AI (智谱AI)

Chinese lab, GLM family (MoE). Strong coding & agent capabilities. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [docs.z.ai](https://docs.z.ai) · [get key](https://open.bigmodel.cn) (free)
- **Free**: `glm-4.5-flash`, `glm-4.7-flash` (203K ctx), `glm-4.6v-flash` (vision) — permanently free, no RPM limit
- **Implementation**: Custom `ChatProvider`

### sambanova — SambaNova

Custom RDU hardware (not GPU). Fast time-to-first-token. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [docs.sambanova.ai](https://docs.sambanova.ai) · [get key](https://cloud.sambanova.ai)
- **Free**: $5 credits (3 months) + ongoing 20 RPM / 200K TPD
- **Implementation**: `OpenAICompatibleChatProvider`

### moonshot — Moonshot AI / Kimi (月之暗面)

Long context pioneer (128K+). · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [platform.moonshot.ai/docs](https://platform.moonshot.ai/docs) · [get key](https://platform.moonshot.ai) (phone verification)
- **Free**: trial credits + ~3 RPM. Also free on Cloudflare, OpenRouter
- **Implementation**: `OpenAICompatibleChatProvider`

### chutes — Chutes AI

Serverless GPU on TEE/Bittensor decentralized infrastructure. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [chutes.ai](https://chutes.ai) · [pricing](https://chutes.ai/pricing)
- **Free**: ❌ None. Pay-per-token only.
- **Implementation**: `OpenAICompatibleChatProvider`

### ollama — Ollama

Most popular local LLM runner. 1000+ models. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [ollama.com](https://ollama.com)
- **Provider id**: `ollama` — registered in `ProviderFactory`, implemented in `uniinfer/providers/ollama.py` as a **custom** `ChatProvider` (Ollama's native `/api/chat` + `/api/tags`, *not* the OpenAI-compatible subclass).
- **Model id format**: `ollama@<ollama-model-name>` — the part after `@` is the **literal Ollama model id** (version tag included), exactly as `ollama list` / `/api/tags` reports it. Examples: `ollama@gemma3:1b`, `ollama@qwen3.5:0.8b`, `ollama@llama3.1:8b`.
- **Routing**: the proxy / `get_completion()` split the string on the **first** `@` → `{provider, model}`. A bare id (`gemma3:1b`) or wrong separator (`ollama:gemma3:1b`) raises `ValueError` and **will not route** — it must be `provider@model`.
- **Thinking**: Ollama returns reasoning in `message.thinking`; the provider exposes it as `response.thinking`. Toggle via the native `think` field, e.g. `provider_specific_kwargs={"think": False}`. ⚠️ The CLI `--no-think` flag targets **vLLM** (`chat_template_kwargs.enable_thinking`), *not* Ollama's `think`.
- **Self-hosted**: completely free, unlimited. Native API at `http://localhost:11434` (`/api/chat`, `/api/tags`); an OpenAI-compatible shim also exists at `/v1`. No API key locally.
- **Cloud** (optional): Free (light usage) / Pro $20/mo / Max $100/mo
- **Implementation**: Custom `ChatProvider` (local HTTP API)

### stepfun — StepFun (阶跃星辰)

Chinese lab. Strong multimodal, aggressive pricing. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [platform.stepfun.ai/docs](https://platform.stepfun.ai/docs) · [get key](https://platform.stepfun.ai) (phone verification)
- **Free** (V0): 5 concurrency, 10 RPM, 5M TPM
- **Implementation**: `OpenAICompatibleChatProvider`

### internlm — InternLM (上海人工智能实验室)

Chinese lab (Shanghai AI Lab). Open-weight bilingual CN/EN. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [chat.intern-ai.org.cn](https://chat.intern-ai.org.cn)
- **Free** (V0): identical tier ladder as StepFun
- **Implementation**: `OpenAICompatibleChatProvider`

### upstage — Upstage AI

Korean lab. Solar MoE. Strong RAG. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [docs.upstage.ai](https://docs.upstage.ai) · [get key](https://console.upstage.ai)
- **Free**: `solar-pro-3:free` (MoE 102B / 12B active)
- **Implementation**: `OpenAICompatibleChatProvider`

### minimax — MiniMax

Chinese lab. **1M token context**. Anthropic-compatible endpoint (unique). · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [platform.minimax.io/docs](https://platform.minimax.io/docs/api-reference/api-overview) · [get key](https://platform.minimax.io)
- **Free**: trial credits only. M3 open-weight imminent. Also free on OpenRouter.
- **Implementation**: `AnthropicCompatibleProvider` (not OpenAI!)

### ai21 — AI21 Labs

Israeli lab. SSM + Transformer hybrid (Jamba). Enterprise-focused. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs**: [ai21.com](https://ai21.com) · [get key](https://platform.ai21.com)
- **Free**: trial credits only
- **Implementation**: Custom `ChatProvider`, `ai21.AI21Client` SDK

### TU Wien

Austrian academic provider. Chat + Embed + TTS + STT. · 🧠 ✅ · 👁️ ❌ · 🔧 ❌ · 📡 ❌

- **Implementation**: Custom `ChatProvider`
- **Base URL**: `https://aqueduct.ai.datalab.tuwien.ac.at/v1` (prod), `…/aqueduct-staging…/v1` (staging)
- **Gateway**: TU runs [Aqueduct](https://tu-wien-datalab.github.io/aqueduct/) — a self-hosted OpenAI-compatible relay, not a model host. Requests are proxied to vLLM backends.

#### Model aliases vs. underlying model ids

`/v1/models` returns **aliases** only (`owned_by: aqueduct`), e.g. `qwen-3.6-35b`, `glm-5.2-744b-preview`, `gemma-4-e2b-it`. The alias→upstream mapping is configured in the Aqueduct admin UI and is **not exposed by the public API**. Per the [Aqueduct models docs](https://tu-wien-datalab.github.io/aqueduct/user-guide/models/), "the underlying model of an alias can change" — aliases are intentionally stable names, not repo identifiers.

To recover the actual underlying model id (e.g. to find the Hugging Face repo), issue a chat completion and read the `model` field in the response — vLLM echoes back the real served model path:

```bash
KEY=$(uv run credgoo tu)
curl -s "https://aqueduct.ai.datalab.tuwien.ac.at/v1/chat/completions" \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"qwen-3.6-35b","messages":[{"role":"user","content":"hi"}],"max_tokens":5,"chat_template_kwargs":{"enable_thinking":false}}'
# → response.model = "llm-d-qwen36/QuantTrio/Qwen3.6-35B-A3B-AWQ"
#   system_fingerprint = "vllm-0.21.0-..."
```

Decoding the returned id:

| Part | Meaning |
|------|---------|
| `llm-d-qwen36` | vLLM deployment/served-name prefix (internal to the Aqueduct host) |
| `QuantTrio/Qwen3.6-35B-A3B-AWQ` | the **Hugging Face repo path** → https://huggingface.co/QuantTrio/Qwen3.6-35B-A3B-AWQ |

Suffixes seen on TU-served models: `A3B` = 3B activated params (MoE, 35B total), `AWQ` = 4-bit quantized.

> ⚠️ The `model` echo is vLLM behaviour, not an Aqueduct API guarantee. If a backend is ever swapped to a non-vLLM host, this trick stops working — fall back to asking the model to self-identify, or contact the TU admin.

### zai-code — Z.AI Code

Coding-specific endpoint from Zhipu AI.

- **Implementation**: Inherits `ZAIBaseProvider`
