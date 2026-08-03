"""Endpoint triple (base_url + bearer) cascade + bearer-ref resolution.

Covers the Phase-1 llm-gateway contract: resolve_model surfaces base_url + bearer
through env-fallback → default → package → task, and resolves bearer refs
(credgoo:svc / ${ENV} / inline). See clients.yml.example + issue
uniinfer-proxy-as-llm-gateway.
"""
import pytest

from llminvoke import config


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    for k in ("KONTEXT_CLIENTS_YML", "OPENAI_BASE_URL", "OPENAI_API_KEY", "TEST_BEARER"):
        monkeypatch.delenv(k, raising=False)
    config.reload_config()
    yield
    config.reload_config()


def _clients(monkeypatch, tmp_path, data):
    import yaml
    p = tmp_path / "clients.yml"
    p.write_text(yaml.safe_dump(data))
    monkeypatch.setenv("KONTEXT_CLIENTS_YML", str(p))
    config.reload_config()


def test_env_fallback_when_no_clients_yml(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    r = config.resolve_model()
    assert r.base_url == "https://env/v1"
    assert r.bearer == "sk-env"


def test_default_base_url_and_env_bearer_ref(monkeypatch, tmp_path):
    monkeypatch.setenv("TEST_BEARER", "sk-fromenv")
    _clients(monkeypatch, tmp_path, {
        "default": {"base_url": "https://u/v1", "bearer": "${TEST_BEARER}", "model": "tu@qwen-3.6-35b"},
    })
    r = config.resolve_model()
    assert r.base_url == "https://u/v1"
    assert r.bearer == "sk-fromenv"


def test_package_overrides_full_triple(monkeypatch, tmp_path):
    _clients(monkeypatch, tmp_path, {
        "default": {"base_url": "https://d/v1", "bearer": "sk-d"},
        "packages": {"pdf2md": {"base_url": "http://p/v1", "bearer": "sk-p", "model": "tu@vlm"}},
    })
    r = config.resolve_model(package="pdf2md")
    assert r.base_url == "http://p/v1"
    assert r.bearer == "sk-p"
    assert r.model == "vlm"


def test_task_overrides_package(monkeypatch, tmp_path):
    _clients(monkeypatch, tmp_path, {
        "default": {"base_url": "https://d/v1", "bearer": "sk-d"},
        "packages": {"pdf2md": {"base_url": "http://p/v1", "bearer": "sk-p",
                                "tasks": {"vlm": {"base_url": "http://t/v1", "bearer": "sk-t"}}}},
    })
    r = config.resolve_model(package="pdf2md", task="vlm")
    assert r.base_url == "http://t/v1"
    assert r.bearer == "sk-t"


def test_credgoo_ref_resolved(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_credgoo_get_api_key", lambda svc: f"key-for-{svc}")
    _clients(monkeypatch, tmp_path, {"default": {"base_url": "https://u/v1", "bearer": "credgoo:uniinfer"}})
    r = config.resolve_model()
    assert r.bearer == "key-for-uniinfer"


def test_credgoo_ref_missing_key_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "_credgoo_get_api_key", lambda svc: None)
    _clients(monkeypatch, tmp_path, {"default": {"bearer": "credgoo:nope"}})
    with pytest.raises(RuntimeError):
        config.resolve_model()


def test_inline_bearer_passthrough(monkeypatch, tmp_path):
    _clients(monkeypatch, tmp_path, {"default": {"bearer": "sk-raw"}})
    r = config.resolve_model()
    assert r.bearer == "sk-raw"
