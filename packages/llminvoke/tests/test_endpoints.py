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


# ── slice 3: gateway routing (invoke_llm) ──────────────────────────────

def test_invoke_llm_gateway_routes_via_openai_with_full_model_id(monkeypatch):
    """base_url set → OpenAI-compatible transport; model id = full provider@model."""
    import llminvoke
    from uniinfer import ChatMessage
    seen = {}

    class _FakeProv:
        def complete(self, request):
            seen["model_id"] = request.model
            return "RAW"

    def _fake_create(provider, *, base_url=None, api_key=None):
        seen["provider"], seen["base_url"], seen["api_key"] = provider, base_url, api_key
        return _FakeProv()

    monkeypatch.setattr(llminvoke, "create_provider", _fake_create)
    llminvoke.invoke_llm(
        model="qwen-3.6-35b", provider="tu",
        messages=[ChatMessage(role="user", content="hi")],
        base_url="https://uniinfer.skale.dev/v1", bearer="sk-test",
    )
    assert seen["provider"] == "openai"                       # gateway transport
    assert seen["base_url"] == "https://uniinfer.skale.dev/v1"
    assert seen["api_key"] == "sk-test"                        # bearer → api_key
    assert seen["model_id"] == "tu@qwen-3.6-35b"               # full provider@model (gotcha)


def test_invoke_llm_legacy_path_without_base_url(monkeypatch):
    """No base_url → named provider + bare model (unchanged legacy path)."""
    import llminvoke
    from uniinfer import ChatMessage
    seen = {}

    class _FakeProv:
        def complete(self, request):
            seen["model_id"] = request.model
            return "RAW"

    def _fake_create(provider, *, base_url=None, api_key=None):
        seen["provider"], seen["base_url"] = provider, base_url
        return _FakeProv()

    monkeypatch.setattr(llminvoke, "create_provider", _fake_create)
    llminvoke.invoke_llm(
        model="qwen", provider="tu",
        messages=[ChatMessage(role="user", content="hi")],
    )
    assert seen["provider"] == "tu"                            # named provider
    assert seen["base_url"] is None
    assert seen["model_id"] == "qwen"                          # bare model


# ── slice 4: thinking knob ────────────────────────────────────────────

def test_thinking_off_maps_to_enable_thinking_false(monkeypatch, tmp_path):
    _clients(monkeypatch, tmp_path, {"default": {"model": "tu@qwen-3.6-35b", "thinking": "off"}})
    r = config.resolve_model()
    assert r.request_kwargs.get("chat_template_kwargs") == {"enable_thinking": False}


def test_thinking_high_maps_to_reasoning_effort(monkeypatch, tmp_path):
    _clients(monkeypatch, tmp_path, {"default": {"model": "tu@qwen-3.6-35b", "thinking": "high"}})
    r = config.resolve_model()
    assert r.request_kwargs.get("reasoning_effort") == "high"


def test_thinking_cascade_task_overrides_package(monkeypatch, tmp_path):
    _clients(monkeypatch, tmp_path, {
        "default": {"model": "tu@qwen-3.6-35b", "thinking": "on"},
        "packages": {"agentos": {"tasks": {"retriever": {"thinking": "off"}}}},
    })
    r = config.resolve_model(package="agentos", task="retriever")
    assert r.request_kwargs.get("chat_template_kwargs") == {"enable_thinking": False}


def test_thinking_yaml_bool_off_coerced(monkeypatch, tmp_path):
    """YAML parses unquoted `off` as False — coerce to the off mapping."""
    p = tmp_path / "c.yml"
    p.write_text("default:\n  model: tu@qwen-3.6-35b\n  thinking: off\n")
    monkeypatch.setenv("KONTEXT_CLIENTS_YML", str(p))
    config.reload_config()
    r = config.resolve_model()
    assert r.request_kwargs.get("chat_template_kwargs") == {"enable_thinking": False}
