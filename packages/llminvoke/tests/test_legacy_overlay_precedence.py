"""Legacy-Overlay-Präzedenz: Katalog-Task-Modell schlägt Overlay-Blankett.

User 2026-11-02: Modus-Defaults leben in der REGISTRY (strukt2meta:
AAB/kriterien → Grossmodell glm, B-Docs → DSGVO-Default) — „per default
einstellbar, ohne overrides". Ein Legacy-clients.yml-Blankett (default.model)
darf diese Task-Entscheidung nicht mehr fällen; explizite Overlay-Paket/Task-
Blöcke gewinnen weiterhin (Ops-Schnellschalter), ebenso base_url/bearer.
"""

import pytest

from llminvoke import config as cfg


REGISTRY = {
    "default": {"model": "hab@qwen3.6-35b"},
    "packages": {
        "strukt2meta": {
            "dsgvo_required": True,
            "tasks": {"kriterien": {"model": "tu@glm-5.3-744b-experimental"}},
        }
    },
}

LEGACY_BLANKET = {
    "default": {
        "base_url": "https://uniinfer.skale.dev/v1",
        "bearer": "t-test",
        "model": "hab@qwen3.6-35b",
        "backups": ["tu@qwen-3.8-27b"],
    }
}


def _write_yaml(tmp_path, name, data):
    import yaml

    p = tmp_path / name
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch, tmp_path, request):
    import yaml as _yaml

    monkeypatch.setattr(cfg, "_overlay_cache", None)
    monkeypatch.setattr(cfg, "_overlay_mtime", None)
    monkeypatch.setattr(cfg, "_overlay_legacy", False)
    monkeypatch.setattr(cfg, "_registry_cache", None)

    reg = getattr(request, "param", None) or _write_yaml(
        tmp_path, "models.yml", REGISTRY
    )
    ov = getattr(request, "overlay", None)
    if ov is None:
        ov = _write_yaml(tmp_path, "clients.yml", LEGACY_BLANKET)
    monkeypatch.setattr(cfg, "_REGISTRY_PATH", reg)
    monkeypatch.setenv("KONTEXT_CLIENTS_YML", ov)
    return reg, ov


def test_catalog_task_model_beats_legacy_blanket(_fresh_caches):
    r = cfg.resolve_model(package="strukt2meta", task="kriterien")
    assert r.primary.model == "glm-5.3-744b-experimental"
    assert r.primary.provider == "tu"
    # Ops-Domain bleibt vom Overlay: Proxy-Routing + Failover des Blanketts
    # wurde bewusst entfernt (passt nicht zum Task-Modell), base_url bleibt.
    assert r.base_url == "https://uniinfer.skale.dev/v1"


def test_legacy_blanket_still_fills_untasked_packages(_fresh_caches):
    r = cfg.resolve_model(package="pdf2md")  # kein Katalog-Task-Modell
    assert r.primary.model == "qwen3.6-35b"  # Blankett greift wie bisher


def test_explicit_overlay_task_block_still_wins(_fresh_caches, tmp_path):
    import yaml as _yaml

    ov = tmp_path / "clients_specific.yml"
    ov.write_text(
        _yaml.safe_dump(
            {
                "default": LEGACY_BLANKET["default"],
                "packages": {
                    "strukt2meta": {"tasks": {"kriterien": {"model": "tu@other-big"}}}
                },
            }
        ),
        encoding="utf-8",
    )
    import os

    os.environ["KONTEXT_CLIENTS_YML"] = str(ov)
    cfg._overlay_cache = None
    cfg._overlay_mtime = None
    cfg._overlay_legacy = False

    r = cfg.resolve_model(package="strukt2meta", task="kriterien")
    assert r.primary.model == "other-big"  # Ops-Override schlägt Katalog
