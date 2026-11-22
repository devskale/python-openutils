"""ChatProvider interface contract — one parametrized test per guarantee.

Replaces 34 per-provider "method exists / is async" assertions with the
contract itself: every registered ChatProvider must implement the async
completion interface with the right callable shapes. Depth over breadth —
per-provider BEHAVIOR lives in the provider-specific test files.
"""
import importlib
import inspect

import pytest

PROVIDERS = {
    "openai": ("uniinfer.providers.openai", "OpenAIProvider", {"api_key": "k"}),
    "mistral": ("uniinfer.providers.mistral", "MistralProvider", {"api_key": "k"}),
    "ollama": ("uniinfer.providers.ollama", "OllamaProvider", {}),
    "gemini": ("uniinfer.providers.gemini", "GeminiProvider", {"api_key": "k"}),
    "pollinations": (
        "uniinfer.providers.pollinations",
        "PollinationsProvider",
        {},
    ),
}

_OPTIONAL = {
    "anthropic": ("uniinfer.providers.anthropic", "AnthropicProvider", {"api_key": "k"}),
}
for _name, (_mod, _cls, _kw) in _OPTIONAL.items():
    if importlib.util.find_spec(_name) is not None:
        PROVIDERS[_name] = (_mod, _cls, _kw)


def _make(alias):
    mod_name, cls_name, kwargs = PROVIDERS[alias]
    cls = getattr(importlib.import_module(mod_name), cls_name)
    return cls(**kwargs)


@pytest.mark.parametrize("alias", sorted(PROVIDERS))
class TestChatProviderContract:
    def test_async_completion_interface(self, alias):
        """acomplete: coroutine · astream_complete: async generator · both
        exist on every provider — the dispatch core depends on these shapes."""
        p = _make(alias)
        assert inspect.iscoroutinefunction(p.acomplete), "acomplete must be a coroutine function"
        assert inspect.isasyncgenfunction(p.astream_complete), "astream_complete must be an async generator"

    def test_sync_completion_interface(self, alias):
        p = _make(alias)
        assert callable(p.complete)
        assert callable(p.stream_complete)

    def test_list_models_classmethod(self, alias):
        mod_name, cls_name, _ = PROVIDERS[alias]
        cls = getattr(importlib.import_module(mod_name), cls_name)
        assert callable(getattr(cls, "list_models", None))

    def test_is_chat_provider(self, alias):
        """The dispatch core instantiates through the ChatProvider seam —
        every provider must sit under it (no duck-typing strays)."""
        from uniinfer.core import ChatProvider

        assert isinstance(_make(alias), ChatProvider)
