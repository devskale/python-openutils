# UniInfer Smoketest Results — 2026-05-26

**Version:** 0.5.27  
**Proxy:** localhost:8124

## Provider Summary

| Provider | Chat | Embed | Image | TTS | STT | Status |
|---|---|---|---|---|---|---|
| **ai21** | — | — | — | — | — | ❌ 403 access denied (no plan) |
| **anthropic** | 0 models listed | — | — | — | — | ❌ no models returned |
| **arli** | 1 model (Qwen3.5-27B-Derestricted) | — | — | — | — | ❌ 429 free plan limit (2 concurrent) |
| **chutes** | — | — | — | — | — | ❌ $0 balance, quota exceeded |
| **cloudflare** | 20/20 ✅ | — | — | — | — | ✅ **fixed: messages API, chat.completion parsing** |
| **gemini** | 5/8 ✅ | — | — | — | — | ✅ (3 hit rate limits on rapid fire) |
| **groq** | 6/6 ✅ | — | — | — | — | ✅ |
| **internlm** | 7/7 ✅ | — | — | — | — | ✅ |
| **mistral** | 8/8 ✅ | — | — | — | — | ✅ |
| **moonshot** | 6/6 ✅ | — | — | — | — | ✅ **fixed: model_defaults.json temp=1 for kimi models** |
| **ngc** | 5/6 ✅ | — | — | — | — | ✅ (qwen3-30b-a3b → 404) |
| **ollama** | 1/1 ✅ | 5/5 ✅ | — | — | — | ✅ |
| **openai** | — | — | — | — | — | ⏭️ not tested (credential rotation risk) |
| **openrouter** | 4/8 ✅ | — | — | — | — | ✅ (rate limits on popular free models) |
| **pollinations** | 7/10 ✅ | — | 6/8 ✅ | — | — | ✅ **fixed: migrated to gen.pollinations.ai/v1, image model types** |
| **sambanova** | — | — | — | — | — | 💳 payment required |
| **stepfun** | — | — | — | — | — | 💳 402 insufficient credit |
| **tu** | 3/3 ✅ | 1/1 ✅ | 1/1 ✅ | 1/1 ✅ | 1/1 ✅ | ✅ |
| **tu-staging** | — | — | — | — | — | ❌ auth rejected (staging key expired) |
| **upstage** | — | — | — | — | — | 💳 API key suspended, no credit |
| **zai** | — | — | — | — | — | ❌ 429 error 1113 (insufficient balance) |
| **zai-code** | 7/7 ✅ | — | — | — | — | ✅ |

## Fixes Applied

### 1. Cloudflare — switch to messages API (`0.5.25`)
- Provider used legacy `prompt` API → switched to `messages` API
- Added parsing for both simple (`result.response`) and OpenAI-compatible (`result.choices`) responses
- Extract `reasoning_content` from thinking models (gpt-oss-120b, kimi-k2.5, etc.)
- Fixed proxy schema to allow `@` in model IDs (e.g. `cloudflare@@cf/model`)

### 2. Model defaults config (`0.5.26`)
- Added `model_defaults.json` for per-model runtime config (no code changes needed)
- Set `temperature: 1.0` for moonshot `kimi-k2.5` and `kimi-k2.6` (API requirement)
- OpenAICompatibleChatProvider loads and applies defaults in `_build_payload`

### 3. Pollinations migration (`0.5.27`)
- Switched from `text.pollinations.ai/openai` (deprecated) to `gen.pollinations.ai/v1`
- Model type derived from `output_modalities` (image/chat/video/audio)
- Removed stale `allowed_models` whitelist for images (server validates)
- Image model list fetched from `gen.pollinations.ai/v1/models` with auth

## Auth/Plan Issues (need manual resolution)
- **ai21**: 403 — no plan access
- **anthropic**: empty model list — key issue?
- **chutes**: $0 balance
- **sambanova**: needs payment method
- **stepfun**: insufficient credit
- **tu-staging**: staging key expired
- **upstage**: API key suspended, no credit
- **zai**: error 1113 insufficient balance (zai-code plan works fine)
