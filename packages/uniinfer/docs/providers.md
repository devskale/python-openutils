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
| `openai` | $5 credits (3mo); GPT-5 needs paid tier | Free: 3 RPM / 200 RPD; Tier 1: 500 RPM | [platform.openai.com/api/docs/pricing](https://platform.openai.com/api/docs/pricing) |
| `anthropic` | ~$5 credits (signup); no recurring free | Tier 1: 50 RPM / varies by model | [docs.anthropic.com/en/api/rate-limits](https://docs.anthropic.com/en/api/rate-limits) |
| `gemini` | Flash: 1,500 RPD; Gemma 4: free, 30 RPM | Flash: 15 RPM / 1M TPM; Gemma: 30 RPM / 1M TPM | [ai.google.dev/pricing](https://ai.google.dev/pricing) |
| `mistral` | All models (Experiment mode); 2 RPM, ~1B tokens/mo | Free: 2 RPM; Tier 1+: scales with spend | [docs.mistral.ai/admin/user-management-finops/tier](https://docs.mistral.ai/admin/user-management-finops/tier) |
| `cohere` | — | — | [docs.cohere.com](https://docs.cohere.com) |
| `huggingface` | Serverless: free, ~100s req/hr (<10B models) | Free: few hundred req/hr; PRO $9/mo: much higher | [huggingface.co/docs/inference-providers](https://huggingface.co/docs/inference-providers) |
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
  - **~few hundred requests per hour**, models under ~10B parameters
  - 100K+ models available from the Hub
  - Cold starts on less popular models: 10-30 seconds
  - Best for: prototyping, small LLMs (7B-8B), embeddings, classification
  - Not great for: 70B+ LLMs, high-volume, latency-critical workloads
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

<!-- remaining providers TBD -->
