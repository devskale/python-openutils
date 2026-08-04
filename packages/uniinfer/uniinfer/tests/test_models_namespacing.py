"""Regression: /v1/models must serve globally-unique, routable ids.

``Catalog.list_resolved`` flattens every provider's models into one OpenAI-shaped
list. Keying the flat list by the bare model ``id`` duplicated any id shared
across providers (196 in the live catalog: ``gpt-5.4`` under openai/poll-alt/
pollinations, ``tts-1`` under openai/openaitts, kilo's ``~openai/gpt-latest``
class, …), which surfaced as React "two children with the same key" warnings in
model pickers. The fix namespaces the flat-list id as ``provider@model`` — the
same form ``PREDEFINED_MODELS`` (the fallback) and ``pi_export`` already emit,
and the form ``parse_provider_model`` splits on the first ``@`` to route.
"""
import json


def _write_catalog(tmp_path, providers):
    doc = {"_meta": {"generated": "2024-01-01T00:00:00Z"}, "providers": providers}
    p = tmp_path / "models.json"
    p.write_text(json.dumps(doc))
    return p


def test_list_resolved_namespaces_ids_and_has_no_dupes(tmp_path):
    from uniinfer.proxy_services.models_registry import Catalog

    p = _write_catalog(
        tmp_path,
        {
            "openai": {"models": [{"id": "gpt-5.4"}, {"id": "tts-1"}]},
            "pollinations": {"models": [{"id": "gpt-5.4"}]},
            "kilo": {"models": [{"id": "~openai/gpt-latest"}]},
        },
    )
    models = Catalog(path=str(p)).list_resolved()
    ids = [m["id"] for m in models]

    assert ids == [
        "openai@gpt-5.4",
        "openai@tts-1",
        "pollinations@gpt-5.4",
        "kilo@~openai/gpt-latest",
    ]
    assert len(ids) == len(set(ids)), f"duplicate ids in flat list: {ids}"

    by_id = {m["id"]: m for m in models}
    assert by_id["pollinations@gpt-5.4"]["provider"] == "pollinations"
    assert by_id["kilo@~openai/gpt-latest"]["provider"] == "kilo"


def test_namespaced_id_round_trips_through_parse_provider_model(tmp_path):
    """The id we serve must be routable: split on the FIRST '@' only."""
    from uniinfer.completion import parse_provider_model

    for served_id, expect in [
        ("kilo@~openai/gpt-latest", ("kilo", "~openai/gpt-latest")),
        ("openai@gpt-5.4", ("openai", "gpt-5.4")),
        ("pollinations@openai/gpt-oss-120b", ("pollinations", "openai/gpt-oss-120b")),
        ("ollama@qwen3.5:0.8b", ("ollama", "qwen3.5:0.8b")),
    ]:
        assert parse_provider_model(served_id) == expect


def test_save_override_strips_provider_prefix(tmp_path):
    """The webdemo curator sends the namespaced id; overrides stay keyed by bare id."""
    from uniinfer.proxy_services.models_registry import Catalog

    p = _write_catalog(tmp_path, {"openai": {"models": [{"id": "gpt-5.4"}]}})
    cat = Catalog(path=str(p))

    cat.save_override("openai@gpt-5.4", {"context_window": 200000})
    ov = cat.read_overrides()["models"]
    assert "gpt-5.4" in ov
    assert "openai@gpt-5.4" not in ov
    assert ov["gpt-5.4"]["context_window"] == 200000

    resolved = {m["id"]: m for m in cat.list_resolved()}
    assert resolved["openai@gpt-5.4"]["context_window"] == 200000


def test_delete_override_strips_provider_prefix(tmp_path):
    from uniinfer.proxy_services.models_registry import Catalog

    p = _write_catalog(tmp_path, {"openai": {"models": [{"id": "gpt-5.4"}]}})
    cat = Catalog(path=str(p))
    cat.save_override("gpt-5.4", {"context_window": 200000})

    assert cat.delete_override("openai@gpt-5.4") is True
    assert "gpt-5.4" not in cat.read_overrides()["models"]
