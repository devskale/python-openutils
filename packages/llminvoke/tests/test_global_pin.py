"""LLMINVOKE_PIN_*: processweiter Primary-Pin (2026-10-22).

User-Forderung: CLI -m soll den GANZEN Lauf pinnen (Standard-Pfad-Steps,
Gather, Synthesis, FAP) — ein Haken in der Auflösungsschicht statt je Callsite.
Task-Profile (no-think, temperature) und Backups bleiben unberührt."""

import os

import pytest


@pytest.fixture
def _pin(monkeypatch):
    def set_(provider: str, model: str):
        monkeypatch.setenv("LLMINVOKE_PIN_PROVIDER", provider)
        monkeypatch.setenv("LLMINVOKE_PIN_MODEL", model)
    return set_


def test_pin_uebersteuert_default_primary(_pin):
    from llminvoke.config import resolve_model

    _pin("hab", "qwen3.6-35b")
    cfg = resolve_model(package="agentos")
    assert str(cfg.primary) == "hab@qwen3.6-35b", cfg.primary


def test_pin_uebersteuert_env_prefix(_pin, monkeypatch):
    from llminvoke.config import resolve_model

    monkeypatch.setenv("X_PROVIDER", "tu")
    monkeypatch.setenv("X_MODEL", "qwen-3.8-27b")
    _pin("hab", "qwen3.6-35b")
    cfg = resolve_model(package="agentos", env_prefix="X")
    assert str(cfg.primary) == "hab@qwen3.6-35b"


def test_task_request_kwargs_bleiben(_pin):
    from llminvoke.config import resolve_model

    _pin("hab", "qwen3.6-35b")
    cfg = resolve_model(package="agentos", task="stepRoute")
    # no-think-Profile des Tasks bleibt trotz Pin
    kw = cfg.request_kwargs.get("chat_template_kwargs") or {}
    assert kw.get("enable_thinking") is False


def test_halber_pin_wirkt_nicht(monkeypatch):
    from llminvoke.config import resolve_model

    monkeypatch.setenv("LLMINVOKE_PIN_PROVIDER", "hab")
    monkeypatch.delenv("LLMINVOKE_PIN_MODEL", raising=False)
    cfg = resolve_model(package="agentos")
    assert str(cfg.primary).startswith("tu@")  # Default unangetastet
