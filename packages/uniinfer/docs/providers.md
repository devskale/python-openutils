# Provider Index

All providers registered in uniinfer. See [Provider Details](#provider-details) below.

## Chat Providers

| ID | Free Contingent | Free Rate Limits | URL |
|----|----------------|-----------------|-----|
| `groq` | All models, forever free | 30 RPM, 100K TPD (70B) | [console.groq.com](https://console.groq.com/docs/rate-limits) |
| `openrouter` | 25+ `:free` models; ≥$10 deposit → 1000 req/day | Free: 50 req/day; w/ credits: 1000/day | [openrouter.ai/pricing](https://openrouter.ai/pricing) |
| `pollinations` | All models, no key needed; `sk_` = no limits | `pk_`: 1 pollen/hr/IP; `sk_`: unlimited | [gen.pollinations.ai/docs](https://gen.pollinations.ai/docs) |
| `ollama` | Self-hosted: unlimited; Cloud: light usage free | Local: unlimited; Cloud Free: 1 concurrent | [ollama.com/pricing](https://ollama.com/pricing) |
| `arli` | Qwen-3.5-27B-Derestricted, unlimited | 1 concurrent req, 12K ctx | [arliai.com/pricing](https://www.arliai.com/pricing?lang=en) |
| `gemini` | Flash: 1,500 RPD; **3.5-flash: 20 RPD** ⚠️; Gemma 4: 30 RPM | Flash: 15 RPM / 1M TPM | [ai.google.dev/pricing](https://ai.google.dev/pricing) |
| `zai` | `glm-4.5-flash` + `glm-4.7-flash` + `glm-4.6v-flash` permanent | Free Flash: no explicit limit (fair use) | [docs.z.ai](https://docs.z.ai) |
| `mistral` | All models (Experiment mode); ~1B tokens/mo | 2 RPM all models | [docs.mistral.ai](https://docs.mistral.ai/admin/user-management-finops/tier) |
| `cohere` | Trial: 5 RPM, 100K calls/mo, all models | 5 RPM (Trial); 500 RPM (Production) | [docs.cohere.com/docs/rate-limits](https://docs.cohere.com/docs/rate-limits) |
| `huggingface` | Serverless: free, ~100s req/hr, **models <10GB** | Few hundred req/hr; PRO $9/mo: higher | [huggingface.co/docs/inference-providers](https://huggingface.co/docs/inference-providers) |
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
| `opencode` ⬇️ | Experiment: ~1B tok/mo, 60 RPM, 500K TPM | Pay-go ($20); Go plan $10/mo | [opencode.ai/zen](https://opencode.ai/zen) |
| `openai` | $5 credits (3mo); GPT-5 needs paid tier | Free: 3 RPM / 200 RPD | [platform.openai.com/api/docs/pricing](https://platform.openai.com/api/docs/pricing) |
| `anthropic` | ~$5 credits (signup); no recurring free | Tier 1: 50 RPM / varies by model | [docs.anthropic.com/en/api/rate-limits](https://docs.anthropic.com/en/api/rate-limits) |

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

### groq — Groq *(info: 2026-06-03)*

Ultra-fast LPU inference. Forever free, no CC. · `groq` SDK · 🧠 ✅ · 👁️ ❌ · 🔧 ✅ · 📡 ✅

- **Docs / Key**: [console.groq.com/docs](https://console.groq.com/docs) · [get key](https://console.groq.com/keys) (free, no CC)
- **Free tier** — all models, rate limit headers in every response:

  | Model | RPM | TPM | TPD |
  |-------|-----|-----|-----|
  | `llama-3.3-70b-versatile` | 30 | 12K | 100K |
  | `llama-4-scout-17b-16e-instruct` | 30 | 30K | 500K |

  Full limits: [console.groq.com/docs/rate-limits](https://console.groq.com/docs/rate-limits)
- **Implementation**: Custom `ChatProvider`, uses `groq` SDK

### arli — Arli AI *(info: 2026-06-03)*

OpenAI-compatible "derestricted" (uncensored) fine-tunes. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Key**: [arliai.com](https://www.arliai.com) · [pricing](https://www.arliai.com/pricing?lang=en)
- **Free tier** — one free model, unlimited tokens:
  - `Qwen-3.5-27B-Derestricted` — 12K ctx, 1 concurrent req (delayed response)
  - 5 trial requests / 2 days for all other models
- **Implementation**: `OpenAICompatibleChatProvider`

### openrouter — OpenRouter *(info: 2026-06-03)*

Unified API for 400+ models from 60+ providers. Zero markup on paid models. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Pricing**: [openrouter.ai/docs](https://openrouter.ai/docs) · [pricing](https://openrouter.ai/pricing) · [models](https://openrouter.ai/models)
- **Free tier** — 25+ `:free` models, no CC. Free: 50 req/day. With ≥$10 credits: **1,000 req/day** (credits are a deposit, 5.5% fee, deducted per-token).

  Notable free: `google/gemma-4-31b-it:free` (262K ctx), `qwen/qwen3-coder:free` (1M ctx), `meta-llama/llama-3.3-70b-instruct:free`, `openrouter/free` (auto-selects random free).
- **Implementation**: `OpenAICompatibleChatProvider`

### pollinations — Pollinations *(info: 2026-06-03)*

Multi-modal platform (text, image, video, audio). No key needed for basic use. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Key**: [gen.pollinations.ai/docs](https://gen.pollinations.ai/docs) · [get key](https://enter.pollinations.ai/) (free)
- **Free tier** — fully free, no key required. Two key types:
  - `pk_` (publishable): 1 pollen/hr per IP+key — client-side / demos
  - `sk_` (secret): **no rate limits** — server-side / production
- **Models**: GPT-5, Claude, Gemini, DeepSeek, Qwen, Flux (image), Seedance/Veo (video)
- **Implementation**: `OpenAICompatibleChatProvider`

### gemini — Google Gemini *(info: 2026-06-09)*

Most generous free tier among major providers — but per-model RPD varies significantly. `google-genai` SDK (not OpenAI-compatible base; [compatibility layer available](https://ai.google.dev/gemini-api/docs/openai)). · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Pricing / Key**: [ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs) · [pricing](https://ai.google.dev/pricing) · [get key](https://aistudio.google.com/apikey) (free, no CC)
- **Free tier** — permanent, no CC, no expiration. Google may use free-tier requests for model improvement.

  | Model | RPM | RPD | TPM | Notes |
  |-------|-----|-----|-----|-------|
  | `gemini-2.5-flash` | 15 | 1,500 | 1M | Best value free model |
  | `gemini-2.5-flash-lite` | 30 | 1,500 | 1M | Faster, cheaper |
  | `gemini-3-flash-preview` | 10 | 1,500 | 1M | Preview, thinking model |
  | `gemini-3.1-flash-lite-preview` | 15 | 1,500 | 1M | Preview, very fast |
  | `gemini-3.5-flash` | 10 | **20** | 1M | ⚠️ Only 20 RPD! Thinking model |
  | `gemini-2.5-pro` | 5 | 50 (trial only) | 1M | Trial only, very limited |
  | `gemma-4-31b-it` | 30 | 1,500 | 1M | Open-weight |
  | `gemma-4-26b-a4b-it` (MoE 3.8B active) | 30 | 1,500 | 1M | Open-weight MoE |

  **⚠️ Critical**: `gemini-3.5-flash` has only **20 RPD** (requests per day) on the free tier — dramatically lower than other Gemini models (1,500 RPD). This is not a typo. Easy to exhaust during development/testing.

- **Extras**: Grounding (Google Search), code execution, 1M ctx, Live API, native thinking
- **Implementation**: Custom `ChatProvider`, `google.genai.Client`

### openai — OpenAI *(info: 2026-06-03)*

The original LLM API. Reference OpenAI-compatible protocol. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Pricing / Key**: [platform.openai.com/api/docs](https://platform.openai.com/api/docs) · [pricing](https://platform.openai.com/api/docs/pricing) · [get key](https://platform.openai.com/api-keys)
- **Free tier** — $5 credits on signup (3 months). GPT-5+ requires paid Tier 1.

  | Model | RPM | TPM | RPD |
  |-------|-----|-----|-----|
  | `gpt-4o` | 3 | 30K | 200 |
  | `gpt-4o-mini` | 3 | 40K | 200 |
  | `gpt-5` / `gpt-5.5` | ❌ unavailable on free tier | | |

  Tier 1 ($5 top-up): GPT-5.5 = 500 RPM / 500K TPM
- **Extras**: Responses API, Batch API (50% off), prompt caching, Agents SDK, Codex, MCP
- **Implementation**: `OpenAICompatibleChatProvider`

### anthropic — Anthropic (Claude) *(info: 2026-06-03)*

Custom Messages API (not OpenAI-compatible base). Phone verification required. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Pricing / Key**: [docs.anthropic.com](https://docs.anthropic.com/en/docs/intro) · [rate limits](https://docs.anthropic.com/en/api/rate-limits) · [get key](https://console.anthropic.com)
- **Free tier** — ~$5 signup credits. **No recurring free tier** — once spent, pay-as-you-go only.
- **Tier system** (auto-advances with cumulative spend):

  | Tier | Required Spend | Monthly Cap |
  |------|---------------|-------------|
  | Free | $0 (signup bonus) | ~$5 |
  | Tier 1 | $5 | $500/mo |
  | Tier 2 | $40 | $500/mo |
  | Tier 3 | $200 | $1,000/mo |
  | Tier 4 | $400 | $200K/mo |

- **Rate limits** (Tier 1+): Sonnet 4.x = 50 RPM / 30K ITPM · Opus 4.x = 50 RPM / 500K ITPM
- **Extras**: Prompt caching (doesn't count toward rate limits!), Batch API (50% off), computer use, MCP
- **Implementation**: `OpenAICompatibleChatProvider`

### mistral — Mistral AI *(info: 2026-06-03)*

Paris-based EU lab. Strong GDPR/data residency story. Open-weight models for self-host. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Pricing / Key**: [docs.mistral.ai](https://docs.mistral.ai) · [tiers](https://docs.mistral.ai/admin/user-management-finops/tier) · [get key](https://console.mistral.ai)
- **Free tier** ("Experiment" mode) — all models, no CC: **2 RPM**, ~1B tokens/month
- **Paid** (Scale plan, auto-advance by billing):

  | Model | Input $/M | Output $/M | Ctx |
  |-------|----------|-----------|-----|
  | `mistral-large-3` | $2.00 | $6.00 | 128K |
  | `mistral-medium-3` | $1.00 | $3.00 | 128K |
  | `mistral-small-3.1` | $0.20 | $0.60 | 128K |

- **Implementation**: `OpenAICompatibleChatProvider`

### huggingface — HuggingFace *(info: 2026-06-03)*

The ML hub. Three products: Serverless (free), Endpoints (dedicated GPU), Inference Providers (gateway to 15+ partners). `huggingface_hub` SDK. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Pricing / Key**: [huggingface.co/docs/inference-providers](https://huggingface.co/docs/inference-providers) · [pricing](https://huggingface.co/docs/inference-providers/pricing) · [get key](https://huggingface.co/settings/tokens)
- **Free tier** (Serverless) — ~few hundred req/hr, models **<10GB** (~≤8B params or quantized small). 100K+ models. Cold starts: 10–30s on less popular models.
- **PRO** ($9/mo): higher Serverless limits, 25 min/day H200 ZeroGPU, 2M Inference Provider credits
- **Implementation**: Custom `ChatProvider`, `huggingface_hub.InferenceClient`

### cloudflare — Cloudflare Workers AI *(info: 2026-06-03)*

Edge inference at 300+ locations. OpenAI-compatible mode available. Phone verification required. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Pricing / Key**: [developers.cloudflare.com/workers-ai](https://developers.cloudflare.com/workers-ai) · [pricing](https://developers.cloudflare.com/workers-ai/platform/pricing) · [get key](https://dash.cloudflare.com)
- **Free tier** — **10,000 Neurons/day** (Neurons = GPU compute unit). No CC. Must opt into data training.

  | Model | ~Requests/day |
  |-------|--------------|
  | `llama-3.2-1b` | ~200–500 |
  | `llama-3.1-8b-fp8` | ~50–150 |
  | `gemma-4-26b-a4b` | ~30–80 |
  | `gpt-oss-120b` | ~15–40 |
  | `llama-3.3-70b` | ~20–60 |

  *\*Shorter prompts = more requests. Paid: $0.011/1K Neurons.*
- **Also includes**: Embeddings, image gen (Flux), TTS/STT (Whisper, Aura)
- **Implementation**: Custom `ChatProvider`, REST to Workers AI gateway

### cohere — Cohere *(info: 2026-06-03)*

Enterprise-focused. Best-in-class RAG (citations, rerank). Custom protocol. · 🧠 ✅ · 👁️ ❌ · 🔧 ✅ · 📡 ✅

- **Docs / Pricing / Key**: [docs.cohere.com](https://docs.cohere.com) · [pricing](https://cohere.com/pricing) · [get key](https://dashboard.cohere.com)
- **Free tier** (Trial) — permanent, no CC: **5 RPM**, **100K calls/month**, all models
- **Production** (add CC): 500 RPM, pay-per-token, 99.5% SLA

  | Model | Input $/M | Output $/M | Ctx |
  |-------|----------|-----------|-----|
  | `command-r-plus` | $2.50 | $10.00 | 128K |
  | `command-r` | $0.15 | $0.60 | 128K |

- **Extras**: Embed v3, Rerank v3, Command A+ (open weights), SOC 2, GDPR
- **Implementation**: Custom `ChatProvider`

### ngc — NVIDIA NIM *(info: 2026-06-03)*

100+ models on NVIDIA GPUs. OpenAI-compatible endpoint. Three modes: hosted API, containers (free dev), enterprise ($4.5K/GPU/yr). · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Catalog / Key**: [build.nvidia.com](https://build.nvidia.com) (NVIDIA Developer Program, phone verification)
- **Free tier** (Developer) — **~40 RPM** (varies per model, shown in UI). 100+ models. No CC.
- **Notable models**: Nemotron 49B, Llama 3.3 70B, DeepSeek R1, Qwen 2.5 Coder 32B, Kimi K2.5/K2.6, GLM 4.7
- **Extras**: Downloadable NIM containers (free dev on ≤16 GPUs), GPU-optimized inference
- **Implementation**: `OpenAICompatibleChatProvider`

### zai — Zhipu AI (智谱AI) *(info: 2026-06-03)*

Chinese lab, GLM family (MoE). Strong coding & agent capabilities. Custom protocol. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Key**: [docs.z.ai](https://docs.z.ai) · [get key](https://open.bigmodel.cn) (free registration)
- **Free tier** — three permanently free models, **no explicit RPM limit** (fair use):
  - `glm-4.5-flash` · `glm-4.7-flash` (203K ctx) · `glm-4.6v-flash` (vision)
  - General API trial (separate): 5M tokens, 5 RPM
- **Paid models**:

  | Model | Input $/M | Output $/M | Ctx |
  |-------|----------|-----------|-----|
  | `glm-5` | $1.00 | $3.20 | 203K |
  | `glm-4.7` | $0.60 | $2.20 | 205K |
  | `glm-4.5-air` | $0.20 | $1.10 | 131K |

- **Coding Plan** (subscription): Lite ~$10/mo · Pro ~$30/mo · Max ~$80/mo
- **Extras**: Native Chinese+English, GLM-5 ≈ 3–5× cheaper than Claude Sonnet, web search + web reader (free MCP tools)
- **Implementation**: Custom `ChatProvider`

### sambanova — SambaNova *(info: 2026-06-03)*

Custom RDU hardware (not GPU — purpose-built for LLM inference). Fast time-to-first-token. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Key**: [docs.sambanova.ai](https://docs.sambanova.ai) · [get key](https://cloud.sambanova.ai) · endpoint: `https://api.sambanova.ai/v1`
- **Free tier** — two parts:
  - **$5 signup credits** (expires 3 months)
  - **Ongoing free**: 20 RPM / 200K TPD — permanent, no CC

  | Model | Free RPM | Dev RPM (w/ CC) |
  |-------|----------|----------------|
  | `Meta-Llama-3.3-70B-Instruct` | 20 | **240** |
  | `DeepSeek-V3.1` / `gpt-oss-120b` | 20 | 60 |

- **Implementation**: `OpenAICompatibleChatProvider`

### moonshot — Moonshot AI / Kimi (月之暗面) *(info: 2026-06-03)*

Long context pioneer (128K+ in production). Strong coding agent (Kimi Code CLI). · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Key**: [platform.moonshot.ai/docs](https://platform.moonshot.ai/docs) · [get key](https://platform.moonshot.ai) (phone verification) · endpoint: `https://api.moonshot.cn/v1`
- **Free tier** — trial credits + **~3 RPM / ~40K TPM**. All models incl. 128K ctx.
- **Paid**: 60 RPM / 500K+ TPM
- **Model lineup**:

  | Model | Ctx | Notes |
  |-------|-----|-------|
  | `kimi-k2.6` | 256K+ | Latest, multimodal, coding |
  | `kimi-k2.5` | 256K | Trillion-parameter class |
  | `moonshot-v1-128k` | 128K | Classic Kimi v1 |

- **Also free on**: Cloudflare Workers AI, OpenRouter (`moonshotai/kimi-k2.6:free`)
- **Implementation**: `OpenAICompatibleChatProvider`

### chutes — Chutes AI *(info: 2026-06-03)*

Serverless GPU inference on TEE / Bittensor decentralized infrastructure. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Website / Key**: [chutes.ai](https://chutes.ai) · [pricing](https://chutes.ai/pricing) · endpoint: `https://llm.chutes.ai/v1`
- **Free tier**: ❌ None. Pay-per-token only (USD or TAO crypto). May have had signup credits in the past.
- **Extras**: TEE/secure compute, private model deployment ($1.80/hr+), Chutes Search
- **Implementation**: `OpenAICompatibleChatProvider`

### ollama — Ollama *(info: 2026-06-03)*

Most popular local LLM runner. 1000+ models, unlimited & free on your own hardware. Also Ollama Cloud (hosted). · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Website**: [ollama.com](https://ollama.com) · [cloud pricing](https://ollama.com/pricing)
- **Self-hosted**: ✅ Completely free, unlimited. OpenAI-compatible at `http://localhost:11434/v1`. No API key.
- **Cloud** (hosted, optional):
  - **Free**: light usage, 1 concurrent model, sessions reset every 5h
  - **Pro**: $20/mo (3 concurrent, ~50× usage)
  - **Max**: $100/mo (10 concurrent, ~250× usage)
- **Implementation**: Custom `ChatProvider` (local HTTP API)

### stepfun — StepFun (阶跃星辰) *(info: 2026-06-03)*

Chinese lab, Step model family. Strong multimodal reasoning, aggressive pricing. TTS/ASR models too. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Key**: [platform.stepfun.ai/docs](https://platform.stepfun.ai/docs) · [pricing](https://platform.stepfun.ai/docs/en/guides/pricing/details) · [get key](https://platform.stepfun.ai) (phone verification) · endpoint: `https://api.stepfun.com/v1`
- **Free tier** (V0): **5 concurrency, 10 RPM, 5M TPM**
- **Paid tiers** (auto-advance by cumulative top-up):

  | Tier | Top-Up | Concurrency | RPM | TPM |
  |------|--------|-------------|-----|-----|
  | V0 | $0 | 5 | 10 | 5M |
  | V1 | $15 | 100 | 1K | 20M |
  | V2 | $70 | 200 | 5K | 30M |
  | V3 | $300 | 400 | 10K | 40M |
  | V4 | $700 | 1K | 20K | 50M |
  | V5 | $1,500 | 10K | 200K | 100M |

- **Model pricing**: `step-3.7-flash` $0.20/$1.15 · `step-3.5-flash` $0.10/$0.30 (input/output per M). Prompt caching: 80% discount.
- **Implementation**: `OpenAICompatibleChatProvider`

### internlm — InternLM (上海人工智能实验室) *(info: 2026-06-03)*

Chinese lab (Shanghai AI Lab). Open-weight bilingual CN/EN models. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Key**: [chat.intern-ai.org.cn](https://chat.intern-ai.org.cn) · endpoint: `https://chat.intern-ai.org.cn/api/v1`
- **Free tier** (V0): **5 concurrency, 10 RPM, 5M TPM** — identical tier ladder as StepFun (see above).
- **Models**: `internlm3-latest` (flagship), `InternVL2` (vision), `internlm2.5` (prev gen)
- **Extras**: Open-weight (self-host), strong CN math/coding benchmarks
- **Implementation**: `OpenAICompatibleChatProvider`

### upstage — Upstage AI *(info: 2026-06-03)*

Korean lab. Solar MoE models. Strong RAG (document parsing, embeddings, retrieval). · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Key**: [upstage.ai](https://upstage.ai) · [docs](https://docs.upstage.ai) · [get key](https://console.upstage.ai) · endpoint: `https://api.upstage.ai/v1/solar`
- **Free tier**: `solar-pro-3:free` — MoE 102B total / 12B active params
- **Models**: `solar-pro-3`, `solar-mini`, `solar-embedding`, `solar-1-pass` (RAG)
- **Implementation**: `OpenAICompatibleChatProvider`

### minimax — MiniMax *(info: 2026-06-03)*

Chinese lab. **1M token context window** — among the largest available. Anthropic-compatible endpoint (unique). · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Key**: [platform.minimax.io/docs](https://platform.minimax.io/docs/api-reference/api-overview) · [get key](https://platform.minimax.io) · endpoint: `https://api.minimax.io/anthropic`
- **Free tier**: ⚠️ Trial credits only (no permanent free tier). **M3 open-weight release imminent** — self-host for free. Also free on OpenRouter (`minimax/minimax-m2.5:free`).
- **Subscriptions**: Plus $20/mo (~1.7B tok) · Max $50/mo (~5.1B) · Ultra $120/mo (~9.8B)
- **Models**: `MiniMax-M3` (1M ctx, open-weight imminent) · `MiniMax-M2.5` (SOTA) · `MiniMax-M2.1`
- **Implementation**: `AnthropicCompatibleProvider` (not OpenAI!)

### ai21 — AI21 Labs *(info: 2026-06-03)*

Israeli lab. **SSM + Transformer hybrid** (Jamba) — more efficient than pure Transformer for long sequences. Enterprise-focused. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Website / Key**: [ai21.com](https://ai21.com) · [get key](https://platform.ai21.com) · SDK: `ai21` (pip)
- **Free tier**: ⚠️ Trial credits only. No permanent free tier. Enterprise pricing (contact sales).
- **Models**: `jamba-1.5` / `jamba-1.5-large` (flagship) · `jamba-1.5-mini` · Maestro 2 (knowledge agent platform)
- **Implementation**: Custom `ChatProvider`, `ai21.AI21Client` SDK

### opencode — OpenCode Zen *(info: 2026-06-03)* ⬇️ *not yet in uniinfer*

Open-source coding agent (160K GitHub ⭐). Zen = curated, benchmarked models specifically for coding agents. Zero markup. · 🧠 ✅ · 👁️ ✅ · 🔧 ✅ · 📡 ✅

- **Docs / Key**: [opencode.ai/zen](https://opencode.ai/zen) · [get key](https://opencode.ai/auth) · [GitHub](https://github.com/anomalyco/opencode)
- **Free tier** (Experiment plan) — no CC: **~1B tokens/month**, **60 RPM**, **500K TPM** per model. Requires phone verification + data training opt-in. Free models available "as long as OpenCode makes them available" (no hard guarantees).
- **Paid**: Pay-as-you-go ($20 balance, auto-topup at $5) · Go plan $10/mo (GLM-5, Kimi K2.5, MiniMax M2.5)
- **Free models**: `minimax-m2.5`, `deepseek-v4-flash` (limited-time), `grok-code-fast-1` (limited-time), `mistral-medium-3`, `glm-5`, `kimi-k2.5`/`k2.6`
- **Extras**: Purpose-built for coding agents (not general chat), US-hosted, zero-retention policy (w/ exceptions)
- **Caveats**: Data training opt-in required; free models can disappear; soft rate limits
- **Implementation**: ❌ Not yet in uniinfer — OpenAI-compatible, would be `OpenAICompatibleChatProvider`
