"""pi_export: accessible-models selection + catalog->pi mapping."""
from uniinfer.pi_export import accessible_models, catalog_model_to_pi


def _catalog():
    return {"providers": {
        # free by explicit access stamp
        "kilo": {"models": [
            {"id": "stepfun/step-3.7-flash:free", "access": "free",
             "cost": {"input": 0, "output": 0}, "context_window": 262144,
             "capabilities": {"reasoning": True}},
        ]},
        # free by pricing (cost.input == 0, no access stamp)
        "openrouter": {"models": [
            {"id": "free-model", "cost": {"input": 0, "output": 0}},
            {"id": "paid-model", "cost": {"input": 3, "output": 15}},
        ]},
        # universally free (quota-free) provider, no pricing data
        "groq": {"models": [{"id": "llama-free"}]},
        # paid provider with no pricing data -> not accessible
        "anthropic": {"models": [{"id": "claude"}]},
    }}


def test_accessible_models_runs_and_filters_free():
    # Regression: accessible_models used to import a non-existent
    # `_UNIVERSALLY_FREE` from keys.py and raise ImportError.
    pairs = accessible_models(_catalog(), access_filter="free")
    ids = sorted(f"{p}@{m['id']}" for p, m in pairs)
    assert ids == [
        "groq@llama-free",
        "kilo@stepfun/step-3.7-flash:free",
        "openrouter@free-model",
    ]


def test_access_filter_all_returns_everything():
    pairs = accessible_models(_catalog(), access_filter="all")
    assert len(pairs) == 5  # 1 + 2 + 1 + 1


def test_provider_restriction():
    pairs = accessible_models(_catalog(), access_filter="free", providers=["kilo"])
    assert len(pairs) == 1
    assert pairs[0][0] == "kilo"


def test_catalog_model_to_pi_mapping():
    cat = _catalog()["providers"]["kilo"]["models"][0]
    pi = catalog_model_to_pi("kilo", cat)
    d = pi.to_dict()
    assert d["id"] == "kilo@stepfun/step-3.7-flash:free"
    assert d["reasoning"] is True
    assert d["contextWindow"] == 262144
    assert d["cost"] == {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    assert d["compat"] == {"maxTokensField": "max_tokens"}
