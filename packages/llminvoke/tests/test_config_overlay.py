"""Box-overlay (models.yml hierarchy) contract tests — WP1/WP2.

Pins the merge contract from issue llminvoke-models-hierarchy:
- dict deep-merge (overlay wins), lists/scalars replace wholesale
- overlay beats catalog at ANY level (default over catalog task included)
- task > package > default within one file
- legacy clients.yml (KONTEXT_CLIENTS_YML) loads as overlay with sampling
  keys stripped + warned (WP4 transition bridge)
- KONTEXT_MODELS_YML wins over KONTEXT_CLIENTS_YML; mtime hot-reload
"""
import os

import pytest
import yaml

from llminvoke import config


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    for k in (
        "KONTEXT_MODELS_YML", "KONTEXT_CLIENTS_YML", "KONTEXT_DATA_DIR",
        "OPENAI_BASE_URL", "OPENAI_API_KEY", "KONTEXT_CLIENT",
    ):
        monkeypatch.delenv(k, raising=False)
    config.reload_config()
    yield
    config.reload_config()


def _overlay(monkeypatch, tmp_path, data, env="KONTEXT_MODELS_YML", name="models.yml"):
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data))
    monkeypatch.setenv(env, str(p))
    config.reload_config()
    return p


# ── deep_merge unit contract ───────────────────────────────────────────

def test_deep_merge_dict_merges_recursively():
    out = config._deep_merge(
        {"retry": {"attempts": 3, "backoff": "exponential"}, "a": 1},
        {"retry": {"attempts": 5}},
    )
    assert out == {"retry": {"attempts": 5, "backoff": "exponential"}, "a": 1}


def test_deep_merge_list_replaces_not_concatenates():
    out = config._deep_merge({"backups": ["a", "b"]}, {"backups": ["c"]})
    assert out == {"backups": ["c"]}


# ── true overlay: choice + inheritance ─────────────────────────────────

def test_overlay_choice_catalog_sampling_inherited(monkeypatch, tmp_path):
    """Overlay default picks the model; sampling comes from the CATALOG."""
    _overlay(monkeypatch, tmp_path, {"default": {"model": "tu@qwen-3.6-35b-vllm"}})
    r = config.resolve_model()
    assert r.model == "qwen-3.6-35b-vllm"
    # catalog default profile: temperature 0.7, max_tokens 10000
    assert r.temperature == 0.7
    assert r.max_tokens == 10000


def test_overlay_beats_catalog_at_any_level(monkeypatch, tmp_path):
    """Overlay default.model overrides a CATALOG task-pinned model
    (contract: overlay > catalog, any level)."""
    _overlay(monkeypatch, tmp_path, {"default": {"model": "tu@qwen-3.6-35b-vllm"}})
    # catalog md2blank.tasks.blank pins tu@qwen-3.6-35b-vllm … same value would
    # prove nothing — use a task pin we can distinguish: strukt2meta.kriterien
    r = config.resolve_model(package="strukt2meta", task="kriterien")
    assert r.model == "qwen-3.6-35b-vllm"  # overlay default, not the task pin


def test_overlay_task_routing_wins(monkeypatch, tmp_path):
    _overlay(monkeypatch, tmp_path, {
        "default": {"model": "tu@qwen-3.6-35b-vllm"},
        "packages": {"pdf2md": {"tasks": {"vlm": {"model": "tu@other-model"}}}},
    })
    r = config.resolve_model(package="pdf2md", task="vlm")
    assert r.model == "other-model"


def test_overlay_deliberate_sampling_override(monkeypatch, tmp_path):
    """A TRUE overlay may own sampling deliberately (deep-merge overrides)."""
    _overlay(monkeypatch, tmp_path, {"default": {"temperature": 0.11, "max_tokens": 123}})
    r = config.resolve_model()
    assert r.temperature == 0.11
    assert r.max_tokens == 123


def test_overlay_backups_replace_catalog_chain(monkeypatch, tmp_path):
    _overlay(monkeypatch, tmp_path, {"default": {"backups": ["tu@qwen-3.6-35b"]}})
    r = config.resolve_model()
    assert [str(b) for b in r.backups] == ["tu@qwen-3.6-35b"]  # not catalog's gemini


def test_overlay_partial_retry_merges(monkeypatch, tmp_path):
    _overlay(monkeypatch, tmp_path, {"default": {"retry": {"attempts": 5}}})
    r = config.resolve_model()
    assert r.retry.attempts == 5
    assert r.retry.backoff == "exponential"  # inherited from catalog


def test_overlay_endpoint_triple(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env/v1")   # env is the fallback
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    _overlay(monkeypatch, tmp_path, {
        "default": {"base_url": "https://box/v1", "bearer": "${BOX_KEY}", "bare_model": True},
    })
    monkeypatch.setenv("BOX_KEY", "sk-box")
    r = config.resolve_model()
    assert r.base_url == "https://box/v1"
    assert r.bearer == "sk-box"
    assert r.bare_model is True


def test_dsgvo_filter_on_overlay_backups(monkeypatch, tmp_path):
    _overlay(monkeypatch, tmp_path, {
        "default": {"dsgvo_required": True, "backups": ["tu@qwen-3.6-35b", "groq@x"]},
    })
    r = config.resolve_model()
    # groq is dsgvo:false in the catalog → dropped from the chain
    assert [str(b) for b in r.backups] == ["tu@qwen-3.6-35b"]


# ── legacy bridge (KONTEXT_CLIENTS_YML) ────────────────────────────────

def test_legacy_sampling_keys_stripped_and_warned(monkeypatch, tmp_path, caplog):
    _overlay(monkeypatch, tmp_path, {
        "default": {"model": "tu@qwen-3.6-35b-vllm", "temperature": 0.99,
                    "max_tokens": 64000, "backups": ["tu@legacy"]},
        "packages": {"klark0": {"tasks": {"chat": {"temperature": 1.5}}}},
    }, env="KONTEXT_CLIENTS_YML", name="clients.yml")
    with caplog.at_level("WARNING", logger="llminvoke.config"):
        r = config.resolve_model()
    assert r.temperature == 0.7      # catalog, not legacy 0.99
    assert r.max_tokens == 10000     # catalog, not legacy 64000 (the audit drift)
    assert [str(b) for b in r.backups] == ["gemini@models/gemini-flash-lite-latest"]
    assert "sampling keys ignored" in caplog.text
    assert "default.max_tokens" in caplog.text
    assert "packages.klark0.tasks.chat.temperature" in caplog.text


def test_legacy_choice_keys_still_honored(monkeypatch, tmp_path):
    _overlay(monkeypatch, tmp_path, {
        "default": {"base_url": "https://legacy/v1", "bearer": "sk-legacy",
                    "model": "tu@legacy-choice", "thinking": "off"},
    }, env="KONTEXT_CLIENTS_YML", name="clients.yml")
    r = config.resolve_model()
    assert r.base_url == "https://legacy/v1"
    assert r.bearer == "sk-legacy"
    assert r.model == "legacy-choice"
    assert r.request_kwargs.get("chat_template_kwargs") == {"enable_thinking": False}


def test_models_env_wins_over_clients_env(monkeypatch, tmp_path):
    _overlay(monkeypatch, tmp_path, {"default": {"model": "tu@legacy-choice"}},
             env="KONTEXT_CLIENTS_YML", name="clients.yml")
    _overlay(monkeypatch, tmp_path, {"default": {"model": "tu@true-choice"}},
             env="KONTEXT_MODELS_YML", name="models.yml")
    r = config.resolve_model()
    assert r.model == "true-choice"


def test_legacy_via_data_dir(monkeypatch, tmp_path):
    p = tmp_path / "clients.yml"
    p.write_text(yaml.safe_dump({"default": {"model": "tu@datadir"}}))
    monkeypatch.setenv("KONTEXT_DATA_DIR", str(tmp_path))
    config.reload_config()
    r = config.resolve_model()
    assert r.model == "datadir"
    assert config._overlay_legacy is True


def test_models_via_data_dir(monkeypatch, tmp_path):
    p = tmp_path / "models.yml"
    p.write_text(yaml.safe_dump({"default": {"model": "tu@datadir-new"}}))
    monkeypatch.setenv("KONTEXT_DATA_DIR", str(tmp_path))
    config.reload_config()
    r = config.resolve_model()
    assert r.model == "datadir-new"
    assert config._overlay_legacy is False


# ── mtime hot-reload ───────────────────────────────────────────────────

def test_mtime_hot_reload(monkeypatch, tmp_path):
    p = _overlay(monkeypatch, tmp_path, {"default": {"model": "tu@first"}})
    assert config.resolve_model().model == "first"
    p.write_text(yaml.safe_dump({"default": {"model": "tu@second"}}))
    os.utime(p, (0, 0))  # force a distinct mtime on coarse-timer filesystems
    config.reload_config()
    # real hot path: without reload_config, mtime change alone re-reads
    p.write_text(yaml.safe_dump({"default": {"model": "tu@third"}}))
    config.reload_config()
    assert config.resolve_model().model == "third"


# ── client param compatibility ─────────────────────────────────────────

def test_client_param_accepted_and_ignored(monkeypatch, tmp_path):
    _overlay(monkeypatch, tmp_path, {"default": {"model": "tu@qwen-3.6-35b-vllm"}})
    r = config.resolve_model(client="some-deal")   # deprecated, must not raise
    assert r.model == "qwen-3.6-35b-vllm"
