# Add DeepInfra & TogetherAI Providers

## DeepInfra

> ⚠️ **NOT free tier.** DeepInfra requires a credit card or pre-pay to use any services.
> Per their pricing page: "You have to add a card or pre-pay or you won't be able to use our services."
> Source: https://deepinfra.com/pricing

- [ ] Create `uniinfer/providers/deepinfra.py`
  - OpenAI-compatible API (`https://api.deepinfra.com/v1/openai`)
  - **Requires credit card or pre-pay** (not free)
  - Hosts open-weight models: Llama, Mixtral, Qwen, etc.
  - Supports streaming, function calling
  - Use `openai_compatible` pattern or `OpenAI` SDK with custom `base_url`
- [ ] Register in `uniinfer/__init__.py` (`ProviderFactory.register_provider("deepinfra", DeepInfraProvider)`)
- [ ] Add to `__all__` list
- [ ] Add async support (`acomplete`, `astream_complete`)
- [ ] Add tests in `tests/providers/test_deepinfra.py`

**Docs:** https://deepinfra.com/docs

## TogetherAI

> ⚠️ **NOT free tier.** Together AI requires a minimum $5 credit card purchase to access the platform.
> Per their docs: "Together AI does not currently offer free trials. Access to the Together platform
> requires a minimum $5 credit purchase."
> Source: https://docs.together.ai/docs/billing-credits

- [ ] Create `uniinfer/providers/togetherai.py`
  - OpenAI-compatible API (`https://api.together.xyz/v1`)
  - **Requires credit card + $5 minimum credit purchase** (not free)
  - Hosts 100+ open-source models
  - Supports streaming, tool calling, JSON mode
  - Use `openai_compatible` pattern or `OpenAI` SDK with custom `base_url`
- [ ] Register in `uniinfer/__init__.py` (`ProviderFactory.register_provider("together", TogetherAIProvider)`)
- [ ] Add to `__all__` list
- [ ] Add async support (`acomplete`, `astream_complete`)
- [ ] Add tests in `tests/providers/test_togetherai.py`

**Docs:** https://docs.together.ai

## Notes

- Both providers use OpenAI-compatible endpoints → can leverage the existing `openai_compatible.py` pattern or inherit from `OpenAICompatibleProvider`
- Both DeepInfra and TogetherAI require credit card/payment — **neither is free**
- Truly free-tier LLM API providers (no credit card): NVIDIA NIM, Groq, Cloudflare, Google Gemini
- ArliAI (`providers/arli.py`) already exists in uniinfer — paid plans only (Starter+), not a free tier, but cheap & unrestricted with no logging
- See `uniinfer/skills/add-provider/SKILL.md` for implementation blueprint
