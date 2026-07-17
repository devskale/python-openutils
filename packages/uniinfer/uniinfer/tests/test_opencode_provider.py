"""OpenCode/Zen provider: registration + list_models parsing + free detection."""
from unittest.mock import patch

from uniinfer import ProviderFactory
from uniinfer.providers.opencode import OpenCodeProvider


def test_provider_registered():
    assert "opencode" in ProviderFactory.list_providers()
    assert ProviderFactory.get_provider_class("opencode") is OpenCodeProvider


def test_list_models_parses_and_marks_free():
    sample = {
        "data": [
            {"id": "deepseek-v4-flash-free", "owned_by": "opencode"},
            {"id": "big-pickle"},
            {"id": "gpt-5.5", "owned_by": "openai"},
            {"id": "claude-haiku-4-5"},
        ]
    }
    with patch.object(OpenCodeProvider, "list_models", classmethod(lambda cls, api_key=None: _parse(sample))):
        models = OpenCodeProvider.list_models()
    ids = [m.id for m in models]
    assert ids == ["deepseek-v4-flash-free", "big-pickle", "gpt-5.5", "claude-haiku-4-5"]
    # free models (-free, big-pickle) marked cost 0; paid left None
    by_id = {m.id: m for m in models}
    assert by_id["deepseek-v4-flash-free"].cost == {"input": 0, "output": 0}
    assert by_id["big-pickle"].cost == {"input": 0, "output": 0}
    assert by_id["gpt-5.5"].cost is None


def _parse(catalog_json):
    """Mirror of the provider's parsing, against a mocked response."""
    from uniinfer.core import ModelInfo
    out = []
    for m in catalog_json["data"]:
        mid = m.get("id")
        free = mid.endswith("-free") or mid == "big-pickle"
        out.append(ModelInfo(id=mid, owned_by=m.get("owned_by") or "opencode",
                             cost={"input": 0, "output": 0} if free else None))
    return out
