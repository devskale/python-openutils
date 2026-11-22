"""Multimodal-Durchreichung für openai-compat Instanzen (Issue
hab-qwen-image-multimodal).

Historie: VLM-Traffic lief bis 2026-10-22 über tu@ (TU-Provider preserved
multimodal explizit). Der Default-Wechsel zu hab@ (openai-compat) killte
Bilder still — das OCR-Hint im System-Prompt tarnte die Halluzination als
plausible Outputs. Fix: OpenAICompatProvider (pure passthrough für
vLLM/Ollama/LM Studio) preserves per Default; das per-Instance-Flag bleibt
als Escape-Hatch für String-only-Backends (false erzwingt den Legacy-Flatten).
"""
import json

import pytest

from uniinfer.config.instances import (
    clear_instances_cache,
    resolve_instance,
    upsert_instance,
)
from uniinfer.core import ChatMessage
from uniinfer.providers.openai_compatible import OpenAICompatibleChatProvider
from uniinfer.providers.openai_compat import OpenAICompatProvider


@pytest.fixture
def overlay_env(tmp_path, monkeypatch):
    clear_instances_cache()
    path = tmp_path / "provider_instances.json"
    monkeypatch.setenv("UNIINFER_INSTANCES_FILE", str(path))
    yield path
    clear_instances_cache()


def _image_messages():
    return [
        ChatMessage(
            role="user",
            content=[
                {"type": "text", "text": "Was ist auf dem Bild?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                },
            ],
        )
    ]


# --------------------------------------------------------------------------- #
# Klassen-Defaults
# --------------------------------------------------------------------------- #
def test_openai_compat_preserves_by_default():
    """Der Vorfall: hab@ hatte kein Overlay-Flag — der Klassen-Default musste
    das Bild schon durchreichen (vorher: stiller Flatten, Halluzination)."""
    assert OpenAICompatProvider.PRESERVE_MULTIMODAL is True


def test_base_class_keeps_legacy_flatten_default():
    """Konkrete Third-Party-Provider (groq, mistral, …) erben die Basis —
    ihr auditiertes Verhalten bleibt unangetastet."""
    assert OpenAICompatibleChatProvider.PRESERVE_MULTIMODAL is False


# --------------------------------------------------------------------------- #
# Overlay → Spec → Target → Provider (Escape-Hatch in beide Richtungen)
# --------------------------------------------------------------------------- #
def test_overlay_entry_sets_flag(overlay_env):
    upsert_instance(
        "hab",
        provider="openai-compat",
        base_url="http://dgx:3001/v1",
        preserve_multimodal=True,
    )
    assert resolve_instance("hab").preserve_multimodal is True


def test_overlay_entry_without_flag_defaults_none(overlay_env):
    upsert_instance("hab", provider="openai-compat", base_url="http://dgx:3001/v1")
    assert resolve_instance("hab").preserve_multimodal is None


def _target_via_overlay(overlay_env, monkeypatch):
    import uniinfer.completion as comp

    monkeypatch.setattr(
        comp.ProviderFactory,
        "get_provider",
        lambda name, **kwargs: OpenAICompatProvider(**kwargs),
    )
    return comp.Target("hab@qwen3.6-35b")


def test_incident_regression_hab_without_flag(overlay_env, monkeypatch):
    """Live-Kette des Vorfalls: Overlay OHNE preserve_multimodal (der Ist-
    Zustand auf amd), Dispatch über Target auf den echten openai-compat —
    das image_url-Part muss durchreichen."""
    overlay_env.write_text(
        json.dumps(
            {"hab": {"provider": "openai-compat", "base_url": "http://dgx:3001/v1"}}
        )
    )
    target = _target_via_overlay(overlay_env, monkeypatch)
    assert target.provider.PRESERVE_MULTIMODAL is True
    out = target.provider._flatten_messages(_image_messages())
    assert any(part.get("type") == "image_url" for part in out[0]["content"])


def test_escape_hatch_false_restores_flatten(overlay_env, monkeypatch):
    """String-only-Backend: explizites false erzwingt den Legacy-Flatten
    trotz Klassen-Default True."""
    overlay_env.write_text(
        json.dumps(
            {
                "hab": {
                    "provider": "openai-compat",
                    "base_url": "http://dgx:3001/v1",
                    "preserve_multimodal": False,
                }
            }
        )
    )
    target = _target_via_overlay(overlay_env, monkeypatch)
    assert target.provider.PRESERVE_MULTIMODAL is False
    out = target.provider._flatten_messages(_image_messages())
    assert out[0]["content"] == "Was ist auf dem Bild?"


# --------------------------------------------------------------------------- #
# Provider-Verhalten: _flatten_messages
# --------------------------------------------------------------------------- #
def test_flatten_keeps_image_parts_when_flagged():
    p = OpenAICompatibleChatProvider(api_key="k", PRESERVE_MULTIMODAL=True)
    out = p._flatten_messages(_image_messages())
    content = out[0]["content"]
    assert isinstance(content, list)
    assert any(part.get("type") == "image_url" for part in content)


def test_flatten_drops_image_parts_by_default():
    p = OpenAICompatibleChatProvider(api_key="k")
    out = p._flatten_messages(_image_messages())
    assert out[0]["content"] == "Was ist auf dem Bild?"
