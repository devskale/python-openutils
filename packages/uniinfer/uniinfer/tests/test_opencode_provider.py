"""OpenCode/Zen provider: registration + list_models parsing + free detection."""
from unittest.mock import patch
import requests

from uniinfer import ProviderFactory
from uniinfer.providers.opencode import OpenCodeProvider
from uniinfer.core import ModelInfo


def test_provider_registered():
    assert "opencode" in ProviderFactory.list_providers()
    assert ProviderFactory.get_provider_class("opencode") is OpenCodeProvider


def test_list_models_parses_and_marks_free():
    """Test that list_models correctly parses pi.dev catalog and marks free models."""
    # Mock pi.dev catalog response (simplified)
    sample_catalog = {
        "deepseek-v4-flash-free": {
            "id": "deepseek-v4-flash-free",
            "name": "DeepSeek V4 Flash Free",
            "contextWindow": 200000,
            "maxTokens": 128000,
            "cost": {"input": 0, "output": 0},
            "input": ["text"],
        },
        "big-pickle": {
            "id": "big-pickle",
            "name": "Big Pickle",
            "contextWindow": 200000,
            "maxTokens": 32000,
            "cost": {"input": 0, "output": 0},
            "input": ["text"],
        },
        "gpt-5.5": {
            "id": "gpt-5.5",
            "name": "GPT-5.5",
            "contextWindow": 1050000,
            "maxTokens": 128000,
            "cost": {"input": 5, "output": 30},
            "input": ["text", "image"],
        },
        "claude-haiku-4-5": {
            "id": "claude-haiku-4-5",
            "name": "Claude Haiku 4.5",
            "contextWindow": 200000,
            "maxTokens": 64000,
            "cost": {"input": 1, "output": 5},
            "input": ["text", "image"],
        },
    }

    with patch.object(requests, "get") as mock_get:
        mock_get.return_value.json.return_value = sample_catalog
        mock_get.return_value.raise_for_status.return_value = None

        models = OpenCodeProvider.list_models()

    ids = [m.id for m in models]
    assert ids == ["deepseek-v4-flash-free", "big-pickle", "gpt-5.5", "claude-haiku-4-5"]

    # Free models (-free, big-pickle) marked cost 0; paid left as-is from catalog
    by_id = {m.id: m for m in models}
    assert by_id["deepseek-v4-flash-free"].cost == {"input": 0, "output": 0}
    assert by_id["big-pickle"].cost == {"input": 0, "output": 0}
    assert by_id["gpt-5.5"].cost == {"input": 5, "output": 30}
    assert by_id["claude-haiku-4-5"].cost == {"input": 1, "output": 5}

    # Context windows should be populated from pi.dev
    assert by_id["deepseek-v4-flash-free"].context_window == 200000
    assert by_id["gpt-5.5"].context_window == 1050000
    assert by_id["claude-haiku-4-5"].context_window == 200000
