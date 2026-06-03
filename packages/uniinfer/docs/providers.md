# Provider Index

All providers registered in uniinfer. See [Provider Details](#provider-details) for per-provider docs.

## Chat Providers

| ID | Free Contingent | Free Rate Limits | URL |
|----|---------------|-----------------|-----|
| `groq` | All models, forever | 30 RPM, 100K TPD | [console.groq.com](https://console.groq.com/docs/rate-limits) |
| `openrouter` | 25+ `:free` models; deposit ≥$10 → 1000 req/day | 50 req/day (free) / 1000 req/day (with credits) | [openrouter.ai/pricing](https://openrouter.ai/pricing) |
| `ollama` | Self-hosted: unlimited; Cloud: light usage free | Local: unlimited; Cloud Free: 1 concurrent | [ollama.com/pricing](https://ollama.com/pricing) |
| `pollinations` | All models, no key required; sk_ = no limits | pk_: 1 pollen/hour/IP; sk_: unlimited | [gen.pollinations.ai/docs](https://gen.pollinations.ai/docs) |
| `arli` | Qwen-3.5-27B | 1 req at a time, 12K ctx | [arliai.com/pricing](https://www.arliai.com/pricing?lang=en) |
| `zai` | `glm-4.5-flash` + `glm-4.7-flash` free (permanent) | Free Flash: no explicit limit (fair use); Trial: 5 RPM / 5M tokens | [docs.z.ai](https://docs.z.ai) |
| `openai` | $5 credits (3mo); GPT-5 needs paid tier | Free: 3 RPM / 200 RPD; Tier 1: 500 RPM | [platform.openai.com/api/docs/pricing](https://platform.openai.com/api/docs/pricing) |
| `anthropic` | ~$5 credits (signup); no recurring free | Tier 1: 50 RPM / varies by model | [docs.anthropic.com/en/api/rate-limits](https://docs.anthropic.com/en/api/rate-limits) |
| `gemini` | Flash: 1,500 RPD; Gemma 4: free, 30 RPM | Flash: 15 RPM / 1M TPM; Gemma: 30 RPM / 1M TPM | [ai.google.dev/pricing](https://ai.google.dev/pricing) |
| `mistral` | All models (Experiment mode); 2 RPM, ~1B tokens/mo | Free: 2 RPM; Tier 1+: scales with spend | [docs.mistral.ai/admin/user-management-finops/tier](https://docs.mistral.ai/admin/user-management-finops/tier) |
| `cohere` | Trial: 5 RPM, 100K calls/mo (all models) | Free: 5 RPM; Paid: 500 RPM+ | [docs.cohere.com/docs/rate-limits](https://docs.cohere.com/docs/rate-limits) |
| `huggingface` | Serverless: free, ~100s req/hr, **models <10GB only** | Free: few hundred req/hr; PRO $9/mo: much higher | [huggingface.co/docs/inference-providers](https://huggingface.co/docs/inference-providers) |
| `cloudflare` | 10K Neurons/day (free); $0.011/1K Neurons paid | Varies by model; ~50+ req/day for 8B; ~20+ for 70B | [developers.cloudflare.com/workers-ai/platform/pricing](https://developers.cloudflare.com/workers-ai/platform/pricing) |
| `sambanova` | $5 credits (3mo) + ongoing free tier (20 RPM) | Free: 20 RPM / 200K TPD; Dev: up to 240 RPM | [docs.sambanova.ai/docs/en/models/rate-limits](https://docs.sambanova.ai/docs/en/models/rate-limits) |
| `moonshot` | Trial credits + ~3 RPM / 40K TPM (all models) | Free: ~3 RPM; Paid: 60 RPM | [platform.moonshot.ai/docs/pricing](https://platform.moonshot.ai/docs/pricing) |
| `stepfun` | V0: 10 RPM / 5M TPM (free) | V0: 10 RPM; V1 ($15): 1K RPM | [platform.stepfun.ai/docs/en/guides/pricing/details](https://platform.stepfun.ai/docs/en/guides/pricing/details) |
| `upstage` | `solar-pro-3:free` available | Varies by model | [upstage.ai](https://upstage.ai) |
| `internlm` | V0: 10 RPM / 5M TPM (free) | V0: 10 RPM; V1 ($15): 1K RPM | [chat.intern-ai.org.cn](https://chat.intern-ai.org.cn) |
| `minimax` | Trial credits; M3 open-weight imminent | Subscription: $20–120/mo; OR self-host M3 | [minimax.io](https://minimax.io) |
| `chutes` | ❌ No confirmed free tier; pay-per-token only | Varies by model | [chutes.ai/pricing](https://chutes.ai/pricing) |
| `zai-code` | — | — | [z.ai](https://z.ai) |
| `ngc` | Developer Program: ~40 RPM, 100+ models | Varies by model (shown in UI); ~40 RPM typical | [build.nvidia.com](https://build.nvidia.com) |
| `ai21` | Trial credits only | Contact sales for limits | [ai21.com](https://ai21.com) |
| `opencode` ⬇️ | Experiment: ~1B tok/mo, 60 RPM, 500K TPM (no CC) | Pay-as-you-go ($20); Go $10/mo | [opencode.ai/zen](https://opencode.ai/zen) |
| `tu` | — | — | — |
| `tu-staging` | — | — | — |

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
  | **`gemma-4-31b-it`** | **30** | **1,500** | **1M** (vision + tools) |
  | **`gemma-4-26b-a4b-it`** | **30** | **1,500** | **1M** (MoE, 3.8B active/token) |

  Full limits: [ai.google.dev/pricing](https://ai.google.dev/pricing)
- **Reasoning/Thinking**: ✅ native thinking support (`thought` parts in response)
- **Vision**: ✅ multimodal input (images, audio, video, PDFs)
- **Tools**: ✅ function calling + tool config (AUTO/ANY/NONE mode)
- **Streaming**: ✅ native async streaming
- **Extra capabilities**: Grounding with Google Search, code execution, long context (up to 1M), embeddings, JSON mode, Live API
- **Implementation**: Custom `ChatProvider`, uses `google.genai.Client` / `client.aio` for async

### openai — OpenAI *(info: 2026-06-03)*

The original LLM API. Reference implementation for the OpenAI-compatible protocol.

- **API docs**: [platform.openai.com/api/docs](https://platform.openai.com/api/docs)
- **Pricing**: [platform.openai.com/api/docs/pricing](https://platform.openai.com/api/docs/pricing)
- **Rate limits**: [platform.openai.com/api/docs/rate-limits](https://platform.openai.com/api/docs/rate-limits) (tier-based, pay-as-you-go required for full access)
- **Get key**: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
- **SDK**: `openai` (pip)
- **Free tier**: ✅ $5 free credits on signup, valid 3 months
  - No credit card needed to get credits
  - GPT-5/GPT-5.5 **not available** on free tier — requires Tier 1 (paid top-up ≥$5)
- **Free tier rate limits**:

  | Model | RPM | TPM | RPD |
  |-------|-----|-----|-----|
  | `gpt-4o` | 3 | 30K | 200 |
  | `gpt-4o-mini` | 3 | 40K | 200 |
  | `gpt-5` / `gpt-5.5` | ❌ unavailable | — | — |

  Tier 1 (paid, $5+ top-up): GPT-5.5 = 500 RPM / 500K TPM; GPT-5.4 Mini = 500 RPM / 500K TPM
- **Reasoning**: ✅ (`o1`, `o1-mini`, reasoning_effort param)
- **Vision**: ✅ (GPT-4o, GPT-5 series)
- **Tools**: ✅ function calling, tool use, web search, file search, computer use, MCP
- **Streaming**: ✅
- **Extra capabilities**: Responses API, Batch API (50% discount), prompt caching, Agents SDK, real-time voice, Codex
- **Implementation**: `OpenAICompatibleChatProvider`

### anthropic — Anthropic (Claude) *(info: 2026-06-03)*

Anthropic's Claude API. Custom protocol (Messages API), not OpenAI-compatible base.

- **API docs**: [docs.anthropic.com/en/docs/intro](https://docs.anthropic.com/en/docs/intro)
- **API reference**: [docs.anthropic.com/en/api/overview](https://docs.anthropic.com/en/api/overview)
- **Rate limits**: [docs.anthropic.com/en/api/rate-limits](https://docs.anthropic.com/en/api/rate-limits)
- **Get key**: [console.anthropic.com](https://console.anthropic.com) (phone verification required)
- **SDK**: `anthropic` (pip)
- **Free tier**: ✅ ~$5 free credits on signup, no credit card
  - Phone number verification required for API access
  - Credits work across all Claude models (Haiku, Sonnet, Opus)
  - No fixed expiration stated — lasts until spent
  - **No recurring free tier** — once credits are used, it's pay-as-you-go only
- **Tier system** (auto-advances with spend):

  | Tier | Required Spend | Monthly Limit |
  |------|---------------|---------------|
  | Free | $0 (signup bonus) | ~$5 credits |
  | Tier 1 | $5 deposit | $500/month |
  | Tier 2 | $40 cumulative | $500/month |
  | Tier 3 | $200 cumulative | $1,000/month |
  | Tier 4 | $400 cumulative | $200K/month |

- **Rate limits** (per model class, Tier 1+):

  | Model Class | RPM | ITPM | OTPM |
  |------------|-----|------|------|
  | Claude Sonnet 4.x | 50 | 30K | 8K |
  | Claude Haiku 4.5 | 50 | 50K | 10K |
  | Claude Opus 4.x | 50 | 500K | 80K |

  Full limits: [docs.anthropic.com/en/api/rate-limits](https://docs.anthropic.com/en/api/rate-limits)
- **Reasoning/Extended thinking**: ✅ `thinking` budget parameter
- **Vision**: ✅ (all models)
- **Tools**: ✅ function calling, tool use, MCP connectors, computer use
- **Streaming**: ✅ (SSE)
- **Extra capabilities**: Prompt caching (cached input tokens don't count toward rate limits!), Message Batches API (50% discount), Managed Agents, Fast Mode (Opus), PDF/image input
- **Implementation**: `OpenAICompatibleChatProvider`

### mistral — Mistral AI *(info: 2026-06-03)*

European AI lab (Paris). OpenAI-compatible API. Strong GDPR/EU data residency story.

- **API docs**: [docs.mistral.ai](https://docs.mistral.ai)
- **Pricing**: [mistral.ai/products](https://mistral.ai/products)
- **Rate limits / tiers**: [docs.mistral.ai/admin/user-management-finops/tier](https://docs.mistral.ai/admin/user-management-finops/tier)
- **Get key**: [console.mistral.ai](https://console.mistral.ai)
- **SDK**: `mistralai` (pip)
- **Free tier** ("Experiment" mode): ✅ All models, no credit card
  - **2 RPM**, ~1B tokens/month
  - Access to all models including Large, Codestral, Pixtral
  - Intended for evaluation and prototyping
- **Paid tiers** (Scale plan, auto-advance by cumulative billing):

  | Tier | Cumulative Billing | Notes |
  |------|-------------------|-------|
  | Tier 1 | $0 (upgrade to Scale) | Auto on plan upgrade |
  | Tier 2 | >$20 | Auto |
  | Tier 3 | >$100 | Auto |
  | Tier 4 | >$500 | Auto |
  | Higher | >$2,000 | Contact support |

- **Model lineup**:

  | Model | Input ($/M) | Output ($/M) | Context |
  |-------|-----------|-------------|---------|
  | `mistral-large-3` | $2.00 | $6.00 | 128K |
  | `mistral-medium-3` | $1.00 | $3.00 | 128K |
  | `mistral-small-3.1` | $0.20 | $0.60 | 128K |
  | Codestral | code pricing | code pricing | — |
  | Pixtral | vision pricing | vision pricing | — |

- **Reasoning**: ✅ (select models)
- **Vision**: ✅ (Pixtral, Large 3)
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Extra**: Open-weight models available for self-hosting (Mistral 7B, Mixtral 8x7B/8x22B, Apache 2.0), native EU hosting/GDPR compliance, La Plateforme console with Vibe IDE
- **Implementation**: `OpenAICompatibleChatProvider`

### huggingface — HuggingFace *(info: 2026-06-03)*

The ML hub's inference API. Three products: Serverless (free), Endpoints (dedicated GPU), Inference Providers (unified gateway to 15+ partners). Custom SDK (`huggingface_hub`), not OpenAI-compatible base.

- **API docs**: [huggingface.co/docs/inference-providers](https://huggingface.co/docs/inference-providers)
- **Pricing / products overview**: [huggingface.co/docs/inference-providers/pricing](https://huggingface.co/docs/inference-providers/pricing)
- **Get key**: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (free, no CC for basic token)
- **SDK**: `huggingface_hub` (pip) — `InferenceClient` / `AsyncInferenceClient`
- **Free tier** (Serverless Inference API): ✅ Free, no credit card
  - **~few hundred requests per hour**, models under **~10GB** (roughly ≤8B params, or quantized small models)
  - 100K+ models available from the Hub (but large ones won't load on free tier)
  - Cold starts on less popular models: 10-30 seconds
  - Best for: prototyping, small LLMs (7B-8B), embeddings, classification
  - Not great for: full-size 70B+ LLMs, high-volume, latency-critical workloads
- **PRO plan ($9/month)** raises limits significantly:
  - Higher Serverless rate limits
  - 25 min/day H200 ZeroGPU compute (vs ~3-5 min free)
  - 1TB private + 10TB public storage
  - 2M monthly Inference Provider credits
- **Inference Providers** (unified gateway): Routes to Groq, Together AI, Fireworks, Replicate, Cerebras, SambaNova, and 10+ others. Pass-through pricing, OpenAI-compatible endpoint.
- **Inference Endpoints** (dedicated GPU): $0.03 CPU/hr → $6.00 H100/hr, scale-to-zero
- **Reasoning**: ✅ (select models)
- **Vision**: ✅ (select models)
- **Tools**: ✅ function calling (select models)
- **Streaming**: ✅
- **Implementation**: Custom `ChatProvider`, uses `huggingface_hub.InferenceClient`

### cloudflare — Cloudflare Workers AI *(info: 2026-06-03)*

Edge inference on Cloudflare's global network. Runs models at the edge (300+ locations). Custom protocol with OpenAI-compatible mode available.

- **API docs**: [developers.cloudflare.com/workers-ai](https://developers.cloudflare.com/workers-ai)
- **Pricing**: [developers.cloudflare.com/workers-ai/platform/pricing](https://developers.cloudflare.com/workers-ai/platform/pricing)
- **Models catalog**: [developers.cloudflare.com/workers-ai/models](https://developers.cloudflare.com/workers-ai/models)
- **Get key**: [dash.cloudflare.com](https://dash.cloudflare.com) (free account, phone verification required)
- **Free tier** (Workers Free plan): ✅ **10,000 Neurons/day**
  - Neurons = GPU compute unit (varies by model size and token count)
  - No credit card required for free plan
  - Phone verification required; must opt into data training for free tier
  - Paid: $0.011/1K Neurons above free allocation
- **What 10K Neurons buys you** (rough daily capacity):

  | Model | ~Requests/day* |
  |-------|-------------|
  | `@cf/meta/llama-3.2-1b-instruct` | ~200–500 |
  | `@cf/meta/llama-3.1-8b-instruct-fp8` | ~50–150 |
  | `@cf/google/gemma-4-26b-a4b-it` | ~30–80 |
  | `@cf/openai/gpt-oss-120b` | ~15–40 |
  | `@cf/meta/llama-3.3-70b-instruct` | ~20–60 |

  \*Depends on prompt/response length. Shorter = more requests.
- **Notable free models**: Llama 3.x (1B/3B/8B/70B), Gemma 3/4, Qwen3-30B-A3B, GPT-OSS (20B/120B), DeepSeek R1 Distill 32B, Kimi K2.5/K2.6, GLM-4.7-Flash, Nemotron 120B, Granite 4.0 Micro
- **Also includes**: Embeddings (BGE, Qwen), Image generation (Flux), TTS/STT (Whisper, Aura, Melotts), Translation
- **Reasoning**: ✅ (DeepSeek R1 Distill, select models)
- **Vision**: ✅ (Llama 3.2 11B Vision, select models)
- **Tools**: ✅ function calling (beta)
- **Streaming**: ✅
- **Extra**: OpenAI-compatible endpoint, batch API (beta), runs at edge (low latency for users near CF locations)
- **Implementation**: Custom `ChatProvider`, REST API to Workers AI gateway

### cohere — Cohere *(info: 2026-06-03)*

Enterprise-focused LLM API. Strong RAG and tool-use story (Command R series). Custom protocol, not OpenAI-compatible base.

- **API docs**: [docs.cohere.com](https://docs.cohere.com)
- **Pricing**: [cohere.com/pricing](https://cohere.com/pricing)
- **Rate limits**: [docs.cohere.com/docs/rate-limits](https://docs.cohere.com/docs/rate-limits)
- **Get key**: [dashboard.cohere.com](https://dashboard.cohere.com) (free trial key)
- **SDK**: `cohere` (pip)
- **Free tier** (Trial): ✅ Permanent, no credit card
  - **5 RPM**, **100K calls/month**
  - All models available (Command R+, Command R, Embed, Rerank)
  - No SLA on trial tier
  - Suitable for prototyping only — 100K/mo ≈ 3,300/day at ~2 req/min avg
- **Production tier** (add credit card): Auto-upgrades
  - 500 RPM default (can be increased)
  - Pay-per-token, no monthly minimum
  - 99.5% SLA
- **Model pricing**:

  | Model | Input ($/M) | Output ($/M) | Context |
  |-------|-----------|-------------|---------|
  | `command-r-plus` | $2.50 | $10.00 | 128K |
  | `command-r` | $0.15 | $0.60 | 128K |
  | `embed-v3` | $0.10 | — (input only) | — |
  | `rerank-v3` | $1.00/1K queries | — | — |

- **Reasoning**: ✅ (Command R+)
- **Vision**: ❌ (text-only models)
- **Tools**: ✅ native tool use / function calling (core strength)
- **Streaming**: ✅
- **Extra**: Best-in-class RAG support (citations), Embed v3, Rerank v3, Command A+ (open weights), enterprise focus (SOC 2, GDPR)
- **Implementation**: Custom `ChatProvider`

### ngc — NVIDIA NIM *(info: 2026-06-03)*

NVIDIA's inference platform (NVIDIA Inference Microservices). OpenAI-compatible API with 100+ models running on NVIDIA GPUs. Three usage modes: hosted API, downloadable containers, enterprise.

- **API docs / catalog**: [build.nvidia.com](https://build.nvidia.com)
- **API reference**: [build.nvidia.com/explore/discover](https://build.nvidia.com/explore/discover)
- **Get key**: [build.nvidia.com](https://build.nvidia.com) (NVIDIA Developer Program, phone verification required)
- **Endpoint**: `https://integrate.api.nvidia.com/v1` (OpenAI-compatible)
- **Free tier** (Developer Program): ✅ Rate-limited prototyping access
  - **~40 RPM** (varies by model — limits shown in build.nvidia.com UI per model)
  - No credit card required; phone verification needed
  - All catalog models available (Llama, Nemotron, DeepSeek, Qwen, Kimi, GLM, Mistral, Phi, Gemma, etc.)
  - Intended for development/testing only — production use requires NVIDIA AI Enterprise license
- **Production** (NVIDIA AI Enterprise):
  - From **$4,500/GPU/year** or ~$1/GPU-hour in cloud
  - Self-hosted NIM containers on your infrastructure
  - Pricing is per-GPU, not per-model or per-token
- **Notable models**:

  | Model | Notes |
  |-------|-------|
  | `nvidia/nemotron-3-super-49b-v1.5` | NVIDIA's own 49B model |
  | `meta/llama-3.3-70b-instruct` | Llama 3.3 70B |
  | `deepseek-ai/deepseek-r1` | Reasoning model |
  | `qwen/qwen2.5-coder-32b` | Code specialist |
  | `moonshotai/kimi-k2.5` / `kimi-k2.6` | Latest Kimi |
  | `z-ai/glm-4.7-flash` | Z.AI GLM |
  | `google/gemma-4-26b-a4b-it` | MoE Gemma 4 |

- **Reasoning**: ✅ (DeepSeek R1, select models)
- **Vision**: ✅ (select models)
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Extra**: Downloadable NIM containers (free dev/test on up to 16 GPUs), strong GPU-optimized performance, Nemotron series (NVIDIA's own models)
- **Implementation**: `OpenAICompatibleChatProvider`

### zai — Zhipu AI (Z.ai) *(info: 2026-06-03)*

Chinese AI lab (智谱AI). Creator of the GLM model family — MoE architecture, strong coding & agent capabilities. Custom protocol.

- **API docs**: [docs.z.ai](https://docs.z.ai)
- **Pricing / model guide**: [docs.z.ai/guides/llm/glm-4.5](https://docs.z.ai/guides/llm/glm-4.5)
- **Get key**: [open.bigmodel.cn](https://open.bigmodel.cn) (free registration)
- **SDK**: custom (built into uniinfer provider)
- **Free tier** ✅ Two permanently free models:
  - **`glm-4.5-flash`** — General-purpose lightweight, fully free
  - **`glm-4.7-flash`** — 203K context, fully free
  - **`glm-4.6v-flash`** — Vision model, fully free
  - No credit card required; register and get API key
  - Not trial-limited — genuinely free for all registered users, **no explicit RPM/RPD limit** (fair use policy)
  - General API trial (separate): **5M tokens**, **5 RPM**
- **Paid models** (API or Coding Plan subscription):

  | Model | Input ($/M) | Output ($/M) | Context |
  |-------|-----------|-------------|---------|
  | `glm-5` | $1.00 | $3.20 | 203K |
  | `glm-5.1` | $1.40 | $4.40 | 203K |
  | `glm-5-turbo` | $1.20 | $4.00 | 203K |
  | `glm-4.7` | $0.60 | $2.20 | 205K |
  | `glm-4.5-air` | $0.20 | $1.10 | 131K |
  | `glm-4.7-flashx` | $0.07 | $0.40 | — |

- **Coding Plan** (subscription, billed quarterly):
  - Lite ~$10/mo: GLM-5.1, 5-Turbo, 4.7, 4.6, 4.5-Air
  - Pro ~$30/mo: +GLM-5 (flagship)
  - Max ~$80/mo: 4x Pro usage
- **Reasoning**: ✅
- **Vision**: ✅ (GLM-4.6V, GLM-5V-Turbo, GLM-OCR)
- **Tools**: ✅ function calling, web search, web reader (free MCP tools included)
- **Streaming**: ✅
- **Extra**: MoE architecture, native Chinese+English, competitive pricing vs Western providers (GLM-5 ≈ 3–5× cheaper than Claude Sonnet), also available free on OpenRouter (`z-ai/glm-4.5-air:free`)
- **Implementation**: Custom `ChatProvider`

### sambanova — SambaNova *(info: 2026-06-03)*

Enterprise AI inference on custom RDU hardware (SN40L Reconfigurable Dataflow Unit). OpenAI-compatible API. Known for fast inference on large open-source models.

- **API docs**: [docs.sambanova.ai](https://docs.sambanova.ai)
- **Rate limits**: [docs.sambanova.ai/docs/en/models/rate-limits](https://docs.sambanova.ai/docs/en/models/rate-limits)
- **Get key**: [cloud.sambanova.ai](https://cloud.sambanova.ai) (free signup)
- **Endpoint**: `https://api.sambanova.ai/v1` (OpenAI-compatible)
- **Free tier** ✅ Two-part free offering:
  - **$5 signup credits** (expires after 3 months)
  - **Ongoing Free Tier** (no payment method) — rate-limited but permanent
  - No credit card required for free tier
- **Free tier rate limits**:

  | Model | RPM | RPD | TPD |
  |-------|-----|-----|-----|
  | `Meta-Llama-3.3-70B-Instruct` | 20 | 20 | 200K |
  | `DeepSeek-V3.1` | 20 | 20 | 200K |
  | `gpt-oss-120b` | 20 | 20 | 200K |
  | `Llama-4-Maverick-17B-128E` *(preview)* | 20 | 20 | 200K |

- **Developer tier** (add payment method) — much higher:

  | Model | RPM | RPD |
  |-------|-----|-----|
  | `Meta-Llama-3.3-70B-Instruct` | **240** | **48K** |
  | `DeepSeek-V3.1` / `gpt-oss-120b` / `MiniMax-M2.7` | 60 | 12K |

- **Reasoning**: ✅ (DeepSeek V3.x)
- **Vision**: ✅ (select models)
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Extra**: Custom RDU hardware (not GPU — purpose-built for LLM inference), fast time-to-first-token, preview models available for evaluation, enterprise/on-prem options
- **Implementation**: `OpenAICompatibleChatProvider`

### moonshot — Moonshot AI (Kimi) *(info: 2026-06-03)*

Chinese AI lab (月之暗面). Creator of the **Kimi** model family — known for long context windows and strong coding capabilities. OpenAI-compatible API.

- **API docs**: [platform.moonshot.ai/docs](https://platform.moonshot.ai/docs)
- **Pricing**: [platform.moonshot.ai/docs/pricing](https://platform.moonshot.ai/docs/pricing)
- **Get key**: [platform.moonshot.ai](https://platform.moonshot.ai) (phone verification required)
- **Endpoint**: `https://api.moonshot.cn/v1` (OpenAI-compatible)
- **SDK**: `openai` (pip — uses OpenAI SDK with custom base_url)
- **Free tier** ✅ Trial credits + rate-limited ongoing access
  - **Trial credits** on signup (amount varies, no credit card)
  - **~3 RPM**, **~40K TPM** on free tier
  - All models available including 128K context (`moonshot-v1-128k`)
  - Phone verification required; no regional restriction on API endpoint
  - Interface primarily in Chinese (browser translation works fine)
- **Paid tier** (add payment method):

  | Metric | Free | Paid (Base) |
  |--------|------|-------------|
  | RPM | ~3 | 60 |
  | TPM | ~40K | 500K+ |
  | Models | 8K/32K/128K context | 8K/32K/128K context |

- **Model lineup**:

  | Model | Context | Notes |
  |-------|---------|-------|
  | `kimi-k2.6` | 256K+ | Latest, multimodal, coding-focused |
  | `kimi-k2.5` | 256K | Trillion-parameter class |
  | `moonshot-v1-128k` | 128K | Classic Kimi v1 |
  | `moonshot-v1-32k` | 32K | Standard |
  | `moonshot-v1-8k` | 8K | Lightweight |

- **Reasoning**: ✅
- **Vision**: ✅ (K2.x series)
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Extra**: Famous for **long context** (pioneered 128K+ in production), strong coding agent performance (Kimi Code CLI), also available free on Cloudflare Workers AI and OpenRouter (`moonshotai/kimi-k2.6:free`)
- **Implementation**: `OpenAICompatibleChatProvider`

### chutes — Chutes AI *(info: 2026-06-03)*

Serverless GPU inference gateway for open-source models. Runs on TEE (Trusted Execution Environment) / Bittensor decentralized infrastructure. OpenAI-compatible API.

- **Website**: [chutes.ai](https://chutes.ai)
- **Pricing**: [chutes.ai/pricing](https://chutes.ai/pricing)
- **Get key**: [chutes.ai](https://chutes.ai) (signup required)
- **Endpoint**: `https://llm.chutes.ai/v1` (OpenAI-compatible)
- **Free tier**: ❌ No confirmed free tier or trial credits
  - Pay-per-token pricing only, no subscription
  - May have had signup credits in the past — not currently advertised
  - Supports both USD and TAO (Bittensor crypto) payments
- **Model catalog**: Open-source models (Llama, Qwen, Gemma, DeepSeek, Mistral, etc.) on TEE GPUs
- **Private deployments**: Deploy your own model on confidential GPU ($1.80/hr+, deploy fee from $5.40)
- **Reasoning**: ✅ (select models)
- **Vision**: ✅ (select models)
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Extra**: TEE/secure compute option, decentralized infrastructure, "permanently hot" models ready to scale, Chutes Search (AI-powered search), Fictio (custom experiences)
- **Implementation**: `OpenAICompatibleChatProvider`

### ollama — Ollama *(info: 2026-06-03)*

The most popular local LLM runner. Self-hosted models on your own hardware — completely free and unlimited. Also offers **Ollama Cloud** (hosted) as a paid option.

- **Website / docs**: [ollama.com](https://ollama.com)
- **Cloud pricing**: [ollama.com/pricing](https://ollama.com/pricing)
- **Self-hosted**: ✅ **Completely free**, unlimited usage, no API key needed
  - Run any model (Llama, Qwen, Gemma, DeepSeek, Mistral, 1000+ models) locally
  - Only cost is your own hardware/electricity
  - OpenAI-compatible API at `http://localhost:11434/v1`
- **Ollama Cloud** (hosted, paid):
  - **Free tier**: Light usage, 1 concurrent model, session limits reset every 5 hours
  - **Pro**: $20/mo, 3 concurrent models, ~50x more usage than free
  - **Max**: $100/mo, 10 concurrent models, ~250x free usage
  - Usage levels vary by model size (Level 1 = light like GPT-OSS-20B → Level 4 = heavy like DeepSeek-V4-Pro)
- **Reasoning**: ✅ (select models)
- **Vision**: ✅ (select models)
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Extra**: Largest model library of any provider (1000+), local-first philosophy, same CLI/API for local and cloud, model upload to cloud (Pro+), GPU-time-based billing (not token-based)
- **Implementation**: Custom `ChatProvider` (local HTTP API)

### stepfun — StepFun (阶跃星辰) *(info: 2026-06-03)*

Chinese AI lab. Creator of the **Step** model family — strong multimodal reasoning with aggressive pricing. OpenAI-compatible.

- **API docs**: [platform.stepfun.ai/docs](https://platform.stepfun.ai/docs)
- **Pricing & rate limits**: [platform.stepfun.ai/docs/en/guides/pricing/details](https://platform.stepfun.ai/docs/en/guides/pricing/details)
- **Get key**: [platform.stepfun.ai](https://platform.stepfun.ai) (phone verification required)
- **Endpoint**: `https://api.stepfun.com/v1` (OpenAI-compatible)
- **Free tier** (V0): ✅ No payment required
  - **5 concurrency, 10 RPM, 5M TPM**
  - Generous for a free tier — enough for light development
- **Paid tiers** (auto-advance by cumulative top-up):

  | Tier | Cumulative Top-Up | Concurrency | RPM | TPM |
  |------|-------------------|-------------|-----|-----|
  | V0 | $0 | 5 | 10 | 5M |
  | V1 | $15 | 100 | 1K | 20M |
  | V2 | $70 | 200 | 5K | 30M |
  | V3 | $300 | 400 | 10K | 40M |
  | V4 | $700 | 1K | 20K | 50M |
  | V5 | $1,500 | 10K | 200K | 100M |

- **Model pricing**:

  | Model | Input ($/M, cache miss) | Input (cache hit) | Output ($/M) |
  |-------|----------------------|------------------|---------------|
  | `step-3.7-flash` | $0.20 | $0.04 | $1.15 |
  | `step-3.5-flash` | $0.10 | $0.02 | $0.30 |

- **Reasoning**: ✅
- **Vision**: ✅ (multimodal models)
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Extra**: TTS/ASR models available (stepaudio, step-tts), prompt caching with 80% discount on cache hits, Chinese + English
- **Implementation**: `OpenAICompatibleChatProvider`

### upstage — Upstage AI *(info: 2026-06-03)*

Korean AI company. Creator of the **Solar** model family — known for efficient MoE architecture and strong RAG capabilities. OpenAI-compatible.

- **Website**: [upstage.ai](https://upstage.ai)
- **API docs**: [docs.upstage.ai](https://docs.upstage.ai)
- **Get key**: [console.upstage.ai](https://console.upstage.ai) (free signup)
- **Endpoint**: `https://api.upstage.ai/v1/solar` (OpenAI-compatible)
- **Free tier**: ✅ Free models available
  - **`solar-pro-3`** available as free variant (MoE, 102B total / 12B active params)
  - Rate limits on free tier: varies by model
- **Model lineup**:

  | Model | Notes |
  |-------|-------|
  | `solar-pro-3` / `solar-pro-3:free` | MoE flagship, 102B/12B active |
  | `solar-mini` | Lightweight |
  | `solar-embedding` | Embeddings |
  | `solar-1-pass` | RAG / document parsing |

- **Pricing** (paid): Competitive per-token rates
- **Reasoning**: ✅
- **Vision**: ✅
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Extra**: Strong RAG story (document parsing, embeddings, passage retrieval), Korean lab with global focus, enterprise customers
- **Implementation**: `OpenAICompatibleChatProvider`

### internlm — InternLM (上海人工智能实验室) *(info: 2026-06-03)*

Chinese AI lab (Shanghai AI Lab). Creator of the **InternLM** model family — open-weight models with strong bilingual (Chinese/English) capabilities. OpenAI-compatible.

- **Website**: [internlm.ai](https://internlm.ai)
- **API docs**: [chat.intern-ai.org.cn](https://chat.intern-ai.org.cn)
- **Get key**: [chat.intern-ai.org.cn](https://chat.intern-ai.org.cn) (free registration)
- **Endpoint**: `https://chat.intern-ai.org.cn/api/v1` (OpenAI-compatible)
- **Free tier** (V0): ✅ No payment required
  - **5 concurrency, 10 RPM, 5M TPM**
  - Same generous V0 as StepFun — enough for light development
- **Paid tiers** (auto-advance by cumulative top-up):

  | Tier | Cumulative Top-Up | Concurrency | RPM | TPM |
  |------|-------------------|-------------|-----|-----|
  | V0 | $0 | 5 | 10 | 5M |
  | V1 | $15 | 100 | 1K | 20M |
  | V2 | $70 | 200 | 5K | 30M |
  | V3 | $300 | 400 | 10K | 40M |
  | V4 | $700 | 1K | 20K | 50M |
  | V5 | $1,500 | 10K | 200K | 100M |

- **Model lineup**:

  | Model | Notes |
  |-------|-------|
  | `internlm3-latest` / `internlm3` | Flagship, bilingual CN/EN |
  | `internlm2.5` | Previous gen |
  | `InternVL2` | Vision-language model |

- **Reasoning**: ✅
- **Vision**: ✅ (InternVL2 series)
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Extra**: Open-weight models (self-host option), strong Chinese language + math/coding benchmarks, InternVL vision models, same tier ladder as StepFun
- **Implementation**: `OpenAICompatibleChatProvider`

### minimax — MiniMax *(info: 2026-06-03)*

Chinese AI lab. Creator of **MiniMax M-series** models — known for ultra-long context (1M tokens), frontier-grade performance at low prices. Anthropic-compatible API.

- **Website**: [minimax.io](https://minimax.io)
- **API docs**: [platform.minimax.io/docs](https://platform.minimax.io/docs/api-reference/api-overview)
- **Get key**: [platform.minimax.io](https://platform.minimax.io) (free registration)
- **Endpoint**: `https://api.minimax.io/v1` (Anthropic-compatible)
- **SDK**: `ai21` SDK or Anthropic-compatible
- **Free tier**: ⚠️ Limited / trial credits only
  - No permanent free tier documented
  - Signup credits may be available (check dashboard — changes frequently)
  - **M3 is open-weight** (promised release imminent) — self-host for truly free access
  - Also available free on **OpenRouter** (`minimax/minimax-m2.5:free`)
- **Subscription plans** (hosted API):

  | Plan | Price | Tokens/month |
  |------|------|-------------|
  | Plus | $20/mo | ~1.7B |
  | Max | $50/mo | ~5.1B |
  | Ultra | $120/mo | ~9.8B |

- **Model lineup**:

  | Model | Context | Notes |
  |-------|---------|-------|
  | `MiniMax-M3` | **1M tokens** | Latest, open-weight (imminent), multimodal |
  | `MiniMax-M2.5` | Large | SOTA productivity, free on OpenRouter |
  | `MiniMax-M2.1` | — | Previous gen |

- **Reasoning**: ✅
- **Vision**: ✅ (M3)
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Extra**: **1M token context window** (among the largest available), aggressive pricing in the Chinese LLM price war, Anthropic-compatible endpoint (unique)
- **Implementation**: `AnthropicCompatibleProvider` (not OpenAI!)

### ai21 — AI21 Labs *(info: 2026-06-03)*

Israeli AI lab. Creator of the **Jamba** model family — SSM (State Space Model) + Transformer hybrid architecture. Known for efficient long-context processing. Custom SDK (`ai21`).

- **Website**: [ai21.com](https://ai21.com)
- **Product page**: [ai21.com/jamba](https://ai21.com/jamba)
- **Get key**: [platform.ai21.com](https://platform.ai21.com) (free signup)
- **SDK**: `ai21` (pip)
- **Free tier**: ⚠️ Trial credits only
  - Signup credits available for new accounts
  - No permanent free tier advertised
  - Enterprise-focused pricing (contact sales)
- **Model lineup**:

  | Model | Architecture | Notes |
  |-------|------------|-------|
  | `jamba-1.5` / `jamba-1.5-large` | SSM + Transformer hybrid | Flagship, efficient long context |
  | `jamba-1.5-mini` | SSM + Transformer | Smaller/faster |
  | Maestro 2 | — | Knowledge agent platform |

- **Reasoning**: ✅
- **Vision**: ✅
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Extra**: **SSM architecture** (more efficient than pure Transformer for long sequences), enterprise focus (finance, healthcare, defense, manufacturing), Maestro knowledge agent product
- **Implementation**: Custom `ChatProvider`, uses `ai21.AI21Client` SDK

### opencode — OpenCode Zen *(info: 2026-06-03)* ⬇️ *not yet in uniinfer*

**OpenCode** is an open-source coding agent (160K GitHub stars). **Zen** is their curated model marketplace — tested & benchmarked specifically for coding agents, with zero markup on model pricing. Not (yet) a uniinfer provider.

- **Website / docs**: [opencode.ai/zen](https://opencode.ai/zen) · [open-code.ai/docs/zen](https://open-code.ai/docs/zen)
- **GitHub**: [anomalyco/opencode](https://github.com/anomalyco/opencode) — 160K+ stars
- **Get key**: [opencode.ai/auth](https://opencode.ai/auth) (free signup)
- **Endpoint**: OpenAI-compatible (`https://api.opencode.ai/v1` or via OpenCode client)
- **Free tier** (Experiment plan): ✅ No credit card required
  - **~1B tokens/month total**
  - **Per-model: 1 RPS (60 RPM), 500K TPM, 1B tokens/month**
  - Requires **phone verification**
  - Requires **opting into data training** (your prompts may be used to improve models)
  - Free models are available "as long as OpenCode makes them available" — no hard guarantees
- **Paid options**:
  - **Pay-as-you-go**: Add $20 balance (+$1.23 card fee), auto-topup at $5, set monthly spend limits
  - **Go plan**: $10/mo for 3 coding models (GLM-5, Kimi K2.5, MiniMax M2.5)
- **Free model lineup** (changes frequently):

  | Model | Source | Notes |
  |-------|--------|-------|
  | `minimax-m2.5-free` | MiniMax | Free on Zen |
  | `deepseek-v4-flash-free` | DeepSeek | Limited-time free |
  | `grok-code-fast-1-free` | xAI | Limited-time free |
  | `mistral-medium-3` | Mistral | Via Experiment plan |
  | `glm-5-free` | Zhipu AI | Coding model |
  | `kimi-k2.5-free` / `kimi-k2.6-free` | Moonshot/Kimi | Long-context coding |

- **Reasoning**: ✅ (select models)
- **Vision**: ✅ (select models)
- **Tools**: ✅ function calling
- **Streaming**: ✅
- **Extra**: **Purpose-built for coding agents** (not general chat), all models benchmarked by OpenCode team, US-hosted with zero-retention policy (exceptions apply), works with any agent not just OpenCode, transparent pricing (zero markup), 12 language UI support
- **Caveats**: Free tier = data training opt-in required; free models can be removed anytime; rate limits are soft ("determined by server traffic")
- **Implementation**: ❌ Not yet in uniinfer — OpenAI-compatible endpoint, could add as `OpenAICompatibleChatProvider`

<!-- remaining providers TBD -->
