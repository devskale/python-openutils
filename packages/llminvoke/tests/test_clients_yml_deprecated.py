"""clients.yml als Overlay: DEPRECATED — Guards bleiben, Warnung wandert.

Issue llminvoke-clients-yml-abschaltung (2026-11-03): pi5 trägt eine
Box-models.yml (voller Overlay, nur Endpoint-Domain). Noch nicht migrierte
Boxen (tu) behalten auf clients.yml das EXAKTE 0.3.5-Verhalten
(sampling-strip + Blankett-Präzedenz-Schutz) — nur die Warnung ist jetzt
eine DEPRECATED-Migrations-Aufforderung. Der komplette Legacy-Zweig stirbt
in einem letzten Commit, sobald tu migriert ist.

Pi5-Beweis (Issue): resolve-Matrix 8/8 identisch zwischen clients.yml
(IST, 0.3.5-Verhalten) und Box-models.yml (SOLL) — das Box-File trägt NUR
base_url/bearer, Routing/Sampling steht komplett in der Registry.
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
    monkeypatch.setattr(cfg, "_overlay_cache", None)
    monkeypatch.setattr(cfg, "_overlay_mtime", None)
    monkeypatch.setattr(cfg, "_overlay_deprecated", False)
    monkeypatch.setattr(cfg, "_registry_cache", None)
    monkeypatch.delenv("KONTEXT_MODELS_YML", raising=False)
    monkeypatch.delenv("KONTEXT_CLIENTS_YML", raising=False)

    reg = _write_yaml(tmp_path, "models.yml", REGISTRY)
    ov = getattr(request, "overlay", None) or _write_yaml(
        tmp_path, "clients.yml", LEGACY_BLANKET
    )
    monkeypatch.setattr(cfg, "_REGISTRY_PATH", reg)
    monkeypatch.setenv("KONTEXT_CLIENTS_YML", ov)
    return reg, ov


def test_deprecated_blanket_does_not_beat_catalog_task(_fresh_caches):
    """0.3.5-Schutz UNVERÄNDERT (tu-Sicherheit): Katalog-Task-Modell schlägt
    das clients.yml-Blankett; Ops-Domain (base_url) bleibt vom Overlay."""
    r = cfg.resolve_model(package="strukt2meta", task="kriterien")
    assert r.primary.model == "glm-5.3-744b-experimental"
    assert r.primary.provider == "tu"
    assert r.base_url == "https://uniinfer.skale.dev/v1"


def test_deprecated_blanket_still_fills_untasked_packages(_fresh_caches):
    r = cfg.resolve_model(package="pdf2md")  # kein Katalog-Task-Modell
    assert r.primary.model == "qwen3.6-35b"  # Blankett greift wie bisher


def test_deprecated_clients_yml_warns_migration(_fresh_caches, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="llminvoke.config"):
        cfg.resolve_model(package="pdf2md")
    assert any(
        "DEPRECATED" in rec.message and "clients.yml" in rec.message
        for rec in caplog.records
    )


def test_box_models_yml_without_blanket_lets_catalog_own_routing(_fresh_caches, tmp_path, monkeypatch):
    """Das Pi5-Zielbild: Box-File trägt NUR Endpoint → Registry routet
    komplett (inkl. GLM-Großmodell-Task)."""
    box = _write_yaml(
        tmp_path, "box.yml", {"default": {"base_url": "https://uniinfer.skale.dev/v1", "bearer": "t"}}
    )
    monkeypatch.setenv("KONTEXT_MODELS_YML", box)
    r = cfg.resolve_model(package="strukt2meta", task="kriterien")
    assert r.primary.model == "glm-5.3-744b-experimental"
    assert r.base_url == "https://uniinfer.skale.dev/v1"


def test_models_yml_takes_priority_over_clients(_fresh_caches, tmp_path, monkeypatch):
    """Beide Pfade gesetzt → KONTEXT_MODELS_YML gewinnt (_box_path-Reihenfolge)."""
    box = _write_yaml(
        tmp_path, "box2.yml", {"default": {"base_url": "https://box/v1", "bearer": "b"}}
    )
    monkeypatch.setenv("KONTEXT_MODELS_YML", box)
    r = cfg.resolve_model(package="strukt2meta", task="kriterien")
    assert r.primary.model == "glm-5.3-744b-experimental"
    assert r.base_url == "https://box/v1"
