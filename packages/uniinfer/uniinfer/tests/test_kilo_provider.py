"""Kilo Gateway provider: registration + list_models parsing + free detection."""
from unittest.mock import patch

from uniinfer import ProviderFactory
from uniinfer.providers.kilo import KiloProvider, _parse_price


def test_provider_registered():
    assert "kilo" in ProviderFactory.list_providers()
    assert ProviderFactory.get_provider_class("kilo") is KiloProvider


def test_requires_api_key_false():
    """Free models are usable anonymously; the base guard must be off."""
    assert KiloProvider.REQUIRES_API_KEY is False


def test_parse_price_handles_sentinels():
    assert _parse_price("0") == 0.0
    assert _parse_price("1.5") == 1.5
    assert _parse_price(None) is None
    assert _parse_price("") is None
    # OpenRouter/Kilo "-1" sentinel means "price varies" -> None (not free, not priced)
    assert _parse_price("-1") is None


def test_list_models_parses_free_and_paid():
    sample = {
        "data": [
            {
                "id": "tencent/hy3:free",
                "name": "Tencent: Hy3 (free)",
                "isFree": True,
                "context_length": 262144,
                "top_provider": {"context_length": 262144, "max_completion_tokens": 262144},
                "pricing": {"prompt": "0", "completion": "0"},
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                "supported_parameters": ["tools", "reasoning", "max_tokens"],
            },
            {
                "id": "anthropic/claude-sonnet-4.6",
                "name": "Claude Sonnet 4.6",
                "isFree": False,
                "context_length": 1000000,
                "top_provider": {"context_length": 1000000, "max_completion_tokens": 64000},
                "pricing": {"prompt": "3", "completion": "15"},
                "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]},
                "supported_parameters": ["tools", "tool_choice", "structured_outputs"],
            },
            {
                # "-1" sentinel: price varies — must not be treated as free or priced
                "id": "openrouter/auto-beta",
                "name": "Auto Beta",
                "isFree": False,
                "pricing": {"prompt": "-1", "completion": "-1"},
                "architecture": {},
                "supported_parameters": [],
            },
        ]
    }
    with patch.object(KiloProvider, "list_models", classmethod(lambda cls, api_key=None: _parse(sample))):
        models = KiloProvider.list_models()
    by_id = {m.id: m for m in models}

    hy3 = by_id["tencent/hy3:free"]
    assert hy3.cost == {"input": 0.0, "output": 0.0}
    assert hy3.context_window == 262144
    assert hy3.max_output == 262144
    assert hy3.capabilities == {"tool_call": True, "reasoning": True}
    assert hy3.modalities == {"input": ["text"], "output": ["text"]}

    claude = by_id["anthropic/claude-sonnet-4.6"]
    assert claude.cost == {"input": 3_000_000.0, "output": 15_000_000.0}
    assert claude.capabilities == {"tool_call": True, "structured_outputs": True}
    assert "image" in claude.modalities["input"]

    auto = by_id["openrouter/auto-beta"]
    assert auto.cost is None  # sentinel, not free


def _parse(catalog_json):
    """Mirror of the provider's parsing, against a mocked response."""
    from uniinfer.core import ModelInfo
    out = []
    for model in catalog_json["data"]:
        mid = model["id"]
        pricing = model.get("pricing", {}) or {}
        prompt_price = _parse_price(pricing.get("prompt"))
        completion_price = _parse_price(pricing.get("completion"))
        is_free = bool(model.get("isFree")) or mid.endswith(":free")
        if is_free:
            cost = {"input": 0.0, "output": 0.0}
        elif prompt_price is not None or completion_price is not None:
            cost = {}
            if prompt_price is not None:
                cost["input"] = prompt_price * 1_000_000
            if completion_price is not None:
                cost["output"] = completion_price * 1_000_000
        else:
            cost = None
        arch = model.get("architecture", {}) or {}
        modalities = None
        if arch.get("input_modalities") or arch.get("output_modalities"):
            modalities = {
                "input": arch.get("input_modalities", ["text"]),
                "output": arch.get("output_modalities", ["text"]),
            }
        supported = model.get("supported_parameters", []) or []
        capabilities = {}
        if "tools" in supported or "tool_choice" in supported:
            capabilities["tool_call"] = True
        if "structured_outputs" in supported:
            capabilities["structured_outputs"] = True
        if "reasoning" in supported or "include_reasoning" in supported:
            capabilities["reasoning"] = True
        top = model.get("top_provider", {}) or {}
        out.append(ModelInfo(
            id=mid,
            name=model.get("name"),
            type="chat",
            context_window=top.get("context_length") or model.get("context_length"),
            max_output=top.get("max_completion_tokens"),
            cost=cost,
            modalities=modalities,
            capabilities=capabilities or None,
            owned_by=model.get("owned_by") or (mid.split("/")[0] if "/" in mid else "kilo"),
            created=model.get("created"),
            raw=model,
        ))
    return out
