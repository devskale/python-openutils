"""Kein hartkodierter Modell-Default im Resolver (Owner 2026-11-16: „alles via config").

Fehlt der model-Key in jeder Config-Schicht → fail loud (ValueError),
KEIN stiller Code-Fallback mehr. Der frühere Default verdeckte kaputte
Configs und driftete (tu@qwen-3.6-35b-Falle).
"""
import os
import textwrap

import pytest
import yaml


def _write(tmp_path, cfg):
    p = tmp_path / "models.yml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return str(p)


def test_missing_model_everywhere_raises(monkeypatch, tmp_path):
    """Registry ohne default.model + leerer Overlay → ValueError, kein Fallback."""
    import llminvoke.config as cfg_mod

    box = _write(tmp_path, {"default": {"backups": []}})  # auch ohne model
    monkeypatch.setenv("KONTEXT_MODELS_YML", box)
    monkeypatch.setattr(cfg_mod, "_registry_cache", {"default": {}, "models": {}}, raising=False)
    cfg_mod.reload_config()
    monkeypatch.setattr(cfg_mod, "_registry_cache", {"default": {}, "models": {}})
    try:
        with pytest.raises(ValueError, match="kein model-Key"):
            cfg_mod.resolve_model()
    finally:
        cfg_mod.reload_config()


def test_model_from_config_resolves(monkeypatch, tmp_path):
    """Gegenprobe: model via Config → ResolvedConfig.primary korrekt."""
    import llminvoke.config as cfg_mod

    box = _write(tmp_path, {"default": {"model": "hab@qwen3.6-35b", "backups": ["tu@qwen-3.8-27b"]}})
    monkeypatch.setenv("KONTEXT_MODELS_YML", box)
    monkeypatch.setattr(cfg_mod, "_registry_cache", {"default": {}, "models": {}}, raising=False)
    cfg_mod.reload_config()
    monkeypatch.setattr(cfg_mod, "_registry_cache", {"default": {}, "models": {}})
    try:
        r = cfg_mod.resolve_model()
        assert r.primary.model == "qwen3.6-35b"
        assert [b.model for b in r.backups] == ["qwen-3.8-27b"]
    finally:
        cfg_mod.reload_config()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
