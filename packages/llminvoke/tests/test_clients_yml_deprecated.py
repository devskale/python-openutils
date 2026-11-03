"""clients.yml DEPRECATED: Endpoint-Only — Routing-Macht entzogen.

Issue llminvoke-clients-yml-abschaltung (User 2026-11-03: „deprecate
clients.yml bitte"): Eine verbliebene clients.yml wird auf die ENDPOINT-
Domain reduziert (base_url/bearer/bare_model) — model/backups/packages/
thinking/sampling werden gedroppt + geloggt. Unmigrierte Boxen (tu)
behalten ihren Endpoint (kein Ausfall), aber die REGISTRY routet alles
(User 2026-11-02: „Modus-Defaults per Registry, ohne overrides").

Pi5-Beweis (Issue): resolve-Matrix 8/8 identisch zwischen clients.yml
(IST, endpoint-reduziert) und Box-models.yml (SOLL).
"""

import pytest

from llminvoke import config as cfg


REGISTRY = {
    "providers": {"hab": {"dsgvo": True}, "tu": {"dsgvo": True}},
    "default": {"model": "hab@qwen3.6-35b"},
    "packages": {
        "strukt2meta": {
            "dsgvo_required": True,
            "tasks": {"kriterien": {"model": "tu@glm-5.3-744b-experimental"}},
        }
    },
}

LEGACY_FULL = {
    "default": {
        "base_url": "https://uniinfer.skale.dev/v1",
        "bearer": "t-test",
        "model": "tu@qwen-3.6-35b-vllm",          # Routing → gestrichen
        "backups": ["tu@qwen-3.8-27b"],            # Routing → gestrichen
        "temperature": 0.99,                        # Sampling → gestrichen
    },
    "packages": {                                    # Task-Profile → gestrichen
        "strukt2meta": {"model": "tu@qwen-3.6-35b-vllm"},
        "agentos": {"tasks": {"queryRoute": {"thinking": "off"}}},
    },
}


def _write_yaml(tmp_path, name, data):
    import yaml

    p = tmp_path / name
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(cfg, "_overlay_cache", None)
    monkeypatch.setattr(cfg, "_overlay_mtime", None)
    monkeypatch.setattr(cfg, "_registry_cache", None)
    monkeypatch.delenv("KONTEXT_MODELS_YML", raising=False)
    monkeypatch.delenv("KONTEXT_CLIENTS_YML", raising=False)
    reg = _write_yaml(tmp_path, "models.yml", REGISTRY)
    ov = _write_yaml(tmp_path, "clients.yml", LEGACY_FULL)
    monkeypatch.setattr(cfg, "_REGISTRY_PATH", reg)
    monkeypatch.setenv("KONTEXT_CLIENTS_YML", ov)
    return reg, ov


def test_endpoint_survives(_fresh_caches):
    r = cfg.resolve_model(package="pdf2md")
    assert r.base_url == "https://uniinfer.skale.dev/v1"
    assert r.bearer == "t-test"


def test_routing_revoked_registry_owns_it(_fresh_caches):
    """Blankett-Modell + Backups + Paket-Overrides sind WEG: die Registry
    routet (Default hab, Task-Pin GLM) — auch für Pakete, die das alte
    clients.yml explizit auf vllm pinnte."""
    r = cfg.resolve_model(package="pdf2md")
    assert r.primary.model == "qwen3.6-35b" and r.primary.provider == "hab"
    assert [str(b) for b in r.backups] == []  # LEGACY_FULL.backups gestrichen
    k = cfg.resolve_model(package="strukt2meta", task="kriterien")
    assert k.primary.model == "glm-5.3-744b-experimental"  # Task-Pin unblockiert


def test_deprecation_warning_names_the_drops(_fresh_caches, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="llminvoke.config"):
        cfg.resolve_model(package="agentos", task="queryRoute")
    txt = caplog.text
    assert "DEPRECATED" in txt and "clients.yml" in txt
    assert "default.model" in txt and "packages" in txt


def test_box_models_yml_keeps_full_power(_fresh_caches, tmp_path, monkeypatch):
    """Die Box-models.yml bleibt der ECHTE Overlay: darf weiterhin bewusst
    Routing setzen (Not-Schalter) — nur clients.yml ist entmachtet."""
    box = _write_yaml(
        tmp_path,
        "box.yml",
        {"default": {"base_url": "https://uniinfer.skale.dev/v1", "bearer": "t", "model": "tu@qwen-3.6-35b-vllm"}},
    )
    monkeypatch.setenv("KONTEXT_MODELS_YML", box)
    r = cfg.resolve_model(package="pdf2md")
    assert r.primary.model == "qwen-3.6-35b-vllm"
    assert r.base_url == "https://uniinfer.skale.dev/v1"


def test_models_yml_takes_priority_over_clients(_fresh_caches, tmp_path, monkeypatch):
    box = _write_yaml(tmp_path, "box2.yml", {"default": {"base_url": "https://box/v1", "bearer": "b"}})
    monkeypatch.setenv("KONTEXT_MODELS_YML", box)
    r = cfg.resolve_model(package="strukt2meta", task="kriterien")
    assert r.primary.model == "glm-5.3-744b-experimental"
    assert r.base_url == "https://box/v1"
