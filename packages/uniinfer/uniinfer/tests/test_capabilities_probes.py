"""Unit tests for the upgraded capability probes (tool-calling + structured-output).

These mock the completion Target so no real backend is hit. They cover the
probe *logic* — facet scoring (selection / parameter structure / negative),
JSON-schema structural validation, and skip-logic — not live model behavior
(which lives in the testsuite/ tier).
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from uniinfer.capabilities.core import (
    ProbeTarget,
    _tool_call_args,
    _tool_call_names,
    _type_matches,
    _validate_structured,
    probe_structured_output,
    probe_tool_calling,
)

PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"},
    },
    "required": ["name", "age", "city"],
    "additionalProperties": False,
}


def _msg(tool_calls=None, content=""):
    return SimpleNamespace(message=SimpleNamespace(tool_calls=tool_calls, content=content))


def _tc(name, args):
    return {"function": {"name": name, "arguments": __import__("json").dumps(args)}}


def _target(profile=None, provider_model="ollama@test", max_tokens=256, timeout=30):
    t = ProbeTarget(
        provider_model=provider_model,
        api_key="k",
        base_url=None,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    if profile is not None:
        t.profile = profile
    return t


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class TestToolCallHelpers:
    def test_names_from_dicts(self):
        assert _tool_call_names([_tc("get_weather", {"location": "Paris"})]) == ["get_weather"]

    def test_names_from_objects(self):
        tc = SimpleNamespace(function=SimpleNamespace(name="get_stock_price"))
        assert _tool_call_names([tc]) == ["get_stock_price"]

    def test_names_empty(self):
        assert _tool_call_names([]) == []
        assert _tool_call_names(None) == []

    def test_args_parses_json(self):
        calls = [_tc("get_weather", {"location": "Paris", "unit": "celsius"})]
        assert _tool_call_args(calls, 0) == {"location": "Paris", "unit": "celsius"}

    def test_args_already_dict(self):
        calls = [{"function": {"name": "x", "arguments": {"a": 1}}}]
        assert _tool_call_args(calls, 0) == {"a": 1}

    def test_args_bad_json(self):
        calls = [{"function": {"name": "x", "arguments": "not json{"}}]
        assert _tool_call_args(calls, 0) == {}

    def test_args_index_out_of_range(self):
        assert _tool_call_args([], 0) == {}
        assert _tool_call_args([_tc("x", {})], 5) == {}


# --------------------------------------------------------------------------- #
# structured-output validation
# --------------------------------------------------------------------------- #
class TestValidateStructured:
    def test_pass_valid(self):
        status, parsed = _validate_structured(
            '{"name": "Alice", "age": 30, "city": "Berlin"}', PERSON_SCHEMA
        )
        assert status == "pass"
        assert parsed["name"] == "Alice"

    def test_fail_not_json(self):
        status, parsed = _validate_structured("not json at all", PERSON_SCHEMA)
        assert status.startswith("fail")
        assert parsed == {}

    def test_fail_not_object(self):
        status, _ = _validate_structured('["a", "b"]', PERSON_SCHEMA)
        assert "not an object" in status

    def test_fail_missing_required(self):
        status, _ = _validate_structured('{"name": "Alice"}', PERSON_SCHEMA)
        assert "missing required" in status

    def test_fail_wrong_type(self):
        status, _ = _validate_structured(
            '{"name": "Alice", "age": "thirty", "city": "Berlin"}', PERSON_SCHEMA
        )
        assert "age" in status and "integer" in status


class TestTypeMatches:
    @pytest.mark.parametrize(
        "py_type,json_type,expected",
        [
            ("str", "string", True),
            ("int", "integer", True),
            ("float", "number", True),
            ("int", "number", True),
            ("bool", "boolean", True),
            ("list", "array", True),
            ("dict", "object", True),
            ("str", "integer", False),
            ("int", "string", False),
        ],
    )
    def test_pairs(self, py_type, json_type, expected):
        assert _type_matches(py_type, json_type) is expected


# --------------------------------------------------------------------------- #
# probe_tool_calling — facet scoring
# --------------------------------------------------------------------------- #
class TestProbeToolCalling:
    @pytest.mark.asyncio
    async def test_all_pass(self):
        """Correct tool + required arg present + no call on negative prompt."""
        weather_resp = _msg(tool_calls=[_tc("get_weather", {"location": "Paris"})])
        neg_resp = _msg()
        with patch(
            "uniinfer.capabilities.core._completion_target"
        ) as mt:
            mt.return_value.acomplete = AsyncMock(side_effect=[weather_resp, neg_resp])
            r = await probe_tool_calling(_target())
        assert r.status == "pass"
        assert r.detail["facets"]["selection"] == "pass"
        assert r.detail["facets"]["parameters"] == "pass"
        assert r.detail["facets"]["negative"] == "pass"

    @pytest.mark.asyncio
    async def test_wrong_tool_selected(self):
        """Model calls a tool, but the wrong one — selection fails."""
        weather_resp = _msg(tool_calls=[_tc("get_stock_price", {"ticker": "AAPL"})])
        neg_resp = _msg()
        with patch("uniinfer.capabilities.core._completion_target") as mt:
            mt.return_value.acomplete = AsyncMock(side_effect=[weather_resp, neg_resp])
            r = await probe_tool_calling(_target())
        assert r.status == "fail"
        assert "get_stock_price" in r.detail["facets"]["selection"]
        assert r.detail["facets"]["parameters"] == "skip (wrong tool)"

    @pytest.mark.asyncio
    async def test_correct_tool_missing_param(self):
        """Right tool but no 'location' arg — parameters facet fails."""
        weather_resp = _msg(tool_calls=[_tc("get_weather", {"unit": "celsius"})])
        neg_resp = _msg()
        with patch("uniinfer.capabilities.core._completion_target") as mt:
            mt.return_value.acomplete = AsyncMock(side_effect=[weather_resp, neg_resp])
            r = await probe_tool_calling(_target())
        assert r.status == "fail"
        assert r.detail["facets"]["parameters"].startswith("fail")
        assert "location" in r.detail["facets"]["parameters"]

    @pytest.mark.asyncio
    async def test_no_tool_called(self):
        """Model didn't call any tool — selection fails."""
        weather_resp = _msg(content="I can't check the weather.")
        neg_resp = _msg()
        with patch("uniinfer.capabilities.core._completion_target") as mt:
            mt.return_value.acomplete = AsyncMock(side_effect=[weather_resp, neg_resp])
            r = await probe_tool_calling(_target())
        assert r.status == "fail"
        assert r.detail["facets"]["selection"] == "fail (no tool called)"

    @pytest.mark.asyncio
    async def test_negative_overeager(self):
        """Model calls a tool when it shouldn't — negative facet fails."""
        weather_resp = _msg(tool_calls=[_tc("get_weather", {"location": "Paris"})])
        neg_resp = _msg(tool_calls=[_tc("get_stock_price", {"ticker": "AAPL"})])
        with patch("uniinfer.capabilities.core._completion_target") as mt:
            mt.return_value.acomplete = AsyncMock(side_effect=[weather_resp, neg_resp])
            r = await probe_tool_calling(_target())
        assert r.status == "fail"
        assert r.detail["facets"]["negative"].startswith("fail")

    @pytest.mark.asyncio
    async def test_skip_no_capability(self):
        """Model declares no tools capability — skip."""
        r = await probe_tool_calling(_target(profile={"capabilities": ["completion"]}))
        assert r.status == "skip"
        assert "no tools capability" in r.evidence

    @pytest.mark.asyncio
    async def test_skip_on_unsupported_error(self):
        """Backend 400s with 'does not support tools' — skip, not error."""
        with patch("uniinfer.capabilities.core._completion_target") as mt:
            mt.return_value.acomplete = AsyncMock(
                side_effect=Exception("model does not support tools")
            )
            r = await probe_tool_calling(_target())
        assert r.status == "skip"

    @pytest.mark.asyncio
    async def test_error_on_other_exception(self):
        """Unrelated exception — error, not skip."""
        with patch("uniinfer.capabilities.core._completion_target") as mt:
            mt.return_value.acomplete = AsyncMock(side_effect=RuntimeError("network down"))
            r = await probe_tool_calling(_target())
        assert r.status == "error"


# --------------------------------------------------------------------------- #
# probe_structured_output
# --------------------------------------------------------------------------- #
class TestProbeStructuredOutput:
    @pytest.mark.asyncio
    async def test_pass_valid_json(self):
        resp = _msg(content='{"name": "Alice", "age": 30, "city": "Berlin"}')
        with patch("uniinfer.capabilities.core._completion_target") as mt:
            mt.return_value.acomplete = AsyncMock(return_value=resp)
            r = await probe_structured_output(_target())
        assert r.status == "pass"
        assert r.detail["parsed"]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_fail_not_json(self):
        resp = _msg(content="I can't do JSON.")
        with patch("uniinfer.capabilities.core._completion_target") as mt:
            mt.return_value.acomplete = AsyncMock(return_value=resp)
            r = await probe_structured_output(_target())
        assert r.status == "fail"
        assert "not valid JSON" in r.evidence

    @pytest.mark.asyncio
    async def test_fail_missing_keys(self):
        resp = _msg(content='{"name": "Alice"}')
        with patch("uniinfer.capabilities.core._completion_target") as mt:
            mt.return_value.acomplete = AsyncMock(return_value=resp)
            r = await probe_structured_output(_target())
        assert r.status == "fail"
        assert "missing required" in r.evidence

    @pytest.mark.asyncio
    async def test_fail_empty_response(self):
        resp = _msg(content="")
        with patch("uniinfer.capabilities.core._completion_target") as mt:
            mt.return_value.acomplete = AsyncMock(return_value=resp)
            r = await probe_structured_output(_target())
        assert r.status == "fail"
        assert "empty" in r.evidence

    @pytest.mark.asyncio
    async def test_skip_no_capability(self):
        r = await probe_structured_output(
            _target(profile={"capabilities": ["completion", "tools"]})
        )
        assert r.status == "skip"
        assert "structured_output" in r.evidence

    @pytest.mark.asyncio
    async def test_skip_on_response_format_error(self):
        """Backend rejects response_format — skip, not error."""
        with patch("uniinfer.capabilities.core._completion_target") as mt:
            mt.return_value.acomplete = AsyncMock(
                side_effect=Exception("response_format not supported")
            )
            r = await probe_structured_output(_target())
        assert r.status == "skip"

    @pytest.mark.asyncio
    async def test_error_on_other_exception(self):
        with patch("uniinfer.capabilities.core._completion_target") as mt:
            mt.return_value.acomplete = AsyncMock(side_effect=RuntimeError("timeout"))
            r = await probe_structured_output(_target())
        assert r.status == "error"


# --------------------------------------------------------------------------- #
# verify_tool_call — the cheap empirical truth-check for the catalog
# --------------------------------------------------------------------------- #
from uniinfer.capabilities.core import verify_tool_call, softprobe_catalog


class TestVerifyToolCall:
    """One request, three outcomes: True / False / None (inconclusive)."""

    @pytest.mark.asyncio
    async def test_true_when_tool_call_emitted(self):
        t = ProbeTarget(provider_model="p@m")
        with patch("uniinfer.capabilities.core._complete_quiet",
                   AsyncMock(return_value=_msg(tool_calls=[_tc("get_time", {})]))):
            verdict, ev = await verify_tool_call(t)
        assert verdict is True and "get_time" in ev

    @pytest.mark.asyncio
    async def test_false_when_answers_in_text(self):
        t = ProbeTarget(provider_model="p@m")
        with patch("uniinfer.capabilities.core._complete_quiet",
                   AsyncMock(return_value=_msg(content="It is 3 pm."))):
            verdict, ev = await verify_tool_call(t)
        assert verdict is False and "text" in ev

    @pytest.mark.asyncio
    async def test_none_when_neither_tool_nor_content(self):
        t = ProbeTarget(provider_model="p@m")
        with patch("uniinfer.capabilities.core._complete_quiet",
                   AsyncMock(return_value=_msg())):
            verdict, _ = await verify_tool_call(t)
        assert verdict is None

    @pytest.mark.asyncio
    async def test_none_on_ratelimit_style_error(self):
        t = ProbeTarget(provider_model="p@m")
        with patch("uniinfer.capabilities.core._complete_quiet",
                   AsyncMock(side_effect=RuntimeError("HTTP 429: too many requests"))):
            verdict, _ = await verify_tool_call(t)
        assert verdict is None

    @pytest.mark.asyncio
    async def test_false_when_backend_rejects_tools(self):
        t = ProbeTarget(provider_model="p@m")
        with patch("uniinfer.capabilities.core._complete_quiet",
                   AsyncMock(side_effect=RuntimeError("model does not support tools"))):
            verdict, _ = await verify_tool_call(t)
        assert verdict is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "err",
        [
            "InvalidRequestError: Function calling is not enabled for models/antigravity",
            "InvalidRequestError: This model only supports Interactions API.",
        ],
    )
    async def test_false_on_definitive_provider_refusals(self, err):
        """Provider-worded 400 refusals are False, not inconclusive — the model
        will never emit a tool call on this serving path."""
        t = ProbeTarget(provider_model="p@m")
        with patch("uniinfer.capabilities.core._complete_quiet",
                   AsyncMock(side_effect=RuntimeError(err))):
            verdict, _ = await verify_tool_call(t)
        assert verdict is False


class TestSoftprobeEmpiricalTools:
    """The softprobe writes verified tool_call back via model_overrides."""

    def _catalog_doc(self):
        return {"providers": {"gemini": {"models": [
            {"id": "flash-free", "type": "chat", "cost": {}, "capabilities": {}},
            {"id": "flash-paid", "type": "chat", "cost": {"input": 0.3, "output": 2.5},
             "capabilities": {"tool_call": True}},
            {"id": "embed", "type": "embed", "cost": {}},
        ]}}}

    @pytest.mark.asyncio
    async def test_free_chat_model_verified_and_overridden(self, tmp_path, monkeypatch):
        doc = self._catalog_doc()
        cat_inst = SimpleNamespace(
            read_nested=lambda *a, **k: doc,
            save_override=lambda mid, fields: doc["providers"]["gemini"]["models"]
            .__iter__().__next__() and None,  # placeholder, replaced below
        )
        saved = {}
        cat_inst.save_override = lambda mid, fields: saved.update({mid: fields})
        with patch("uniinfer.proxy_services.models_registry.Catalog",
                   lambda: cat_inst), \
             patch("uniinfer.capabilities.core.Catalog", lambda: cat_inst, create=True), \
             patch("uniinfer.capabilities.core.probe_profile",
                   AsyncMock(return_value=SimpleNamespace(detail={"profile": {}}))), \
             patch("uniinfer.capabilities.core.save_probe_result", lambda r: None), \
             patch("uniinfer.capabilities.core.verify_tool_call",
                   AsyncMock(return_value=(True, "emitted ['get_time']"))):
            summ = await softprobe_catalog(stale_days=None, empirical_tools=True)
        assert saved["flash-free"]["capabilities"] == {
            "tool_call": True, "tool_call_verified": True}
        assert summ["tool_verified_true"] == 1

    @pytest.mark.asyncio
    async def test_paid_model_not_verified_by_default(self):
        doc = self._catalog_doc()
        cat_inst = SimpleNamespace(read_nested=lambda *a, **k: doc,
                                   save_override=lambda mid, fields: None)
        with patch("uniinfer.proxy_services.models_registry.Catalog", lambda: cat_inst), \
             patch("uniinfer.capabilities.core.probe_profile",
                   AsyncMock(return_value=SimpleNamespace(detail={"profile": {}}))), \
             patch("uniinfer.capabilities.core.save_probe_result", lambda r: None), \
             patch("uniinfer.capabilities.core.verify_tool_call",
                   AsyncMock(return_value=(True, "should not be called for paid"))) as vt:
            await softprobe_catalog(stale_days=None, empirical_tools=True)
        # flash-paid is chat+paid → verify must not run; embed is not chat either
        assert vt.await_count == 1  # only flash-free

    @pytest.mark.asyncio
    async def test_inconclusive_keeps_declared_and_writes_nothing(self):
        doc = self._catalog_doc()
        saved = {}
        cat_inst = SimpleNamespace(read_nested=lambda *a, **k: doc,
                                   save_override=lambda mid, fields: saved.update({mid: fields}))
        with patch("uniinfer.proxy_services.models_registry.Catalog", lambda: cat_inst), \
             patch("uniinfer.capabilities.core.probe_profile",
                   AsyncMock(return_value=SimpleNamespace(detail={"profile": {}}))), \
             patch("uniinfer.capabilities.core.save_probe_result", lambda r: None), \
             patch("uniinfer.capabilities.core.verify_tool_call",
                   AsyncMock(return_value=(None, "HTTP 429"))):
            summ = await softprobe_catalog(stale_days=None, empirical_tools=True)
        assert saved == {}  # nothing written
        assert summ["tool_inconclusive"] == 1

    @pytest.mark.asyncio
    async def test_fresh_model_neither_probed_nor_verified(self, tmp_path, monkeypatch):
        """Regression: fresh (tested_at within stale_days) models must skip the
        metadata probe AND the verify — probing them anyway would refresh
        tested_at every run, collapsing the 7-day verify stagger to
        once-per-model-ever (the draft's removed-continue bug)."""
        import json as _json
        from datetime import datetime, timedelta, timezone

        sidecar = tmp_path / "_probe_results.json"
        fresh = (datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
        sidecar.write_text(_json.dumps(
            {"gemini/flash-free": {"tested_at": fresh, "profile": {}}}))
        monkeypatch.setattr("uniinfer.capabilities.core.PROBE_RESULTS_PATH", sidecar)

        doc = self._catalog_doc()
        cat_inst = SimpleNamespace(read_nested=lambda *a, **k: doc,
                                   save_override=lambda mid, fields: None)
        with patch("uniinfer.proxy_services.models_registry.Catalog", lambda: cat_inst), \
             patch("uniinfer.capabilities.core.probe_profile",
                   AsyncMock(return_value=SimpleNamespace(detail={"profile": {}}))) as pp, \
             patch("uniinfer.capabilities.core.save_probe_result", lambda r: None), \
             patch("uniinfer.capabilities.core.verify_tool_call",
                   AsyncMock(return_value=(True, "emitted ['get_time']"))) as vt:
            summ = await softprobe_catalog(stale_days=7, empirical_tools=True)
        # flash-free is fresh → skipped entirely; flash-paid (stale+chat, paid)
        # and embed (stale, not chat) are probed but never verified
        assert pp.await_count == 2 and vt.await_count == 0
        assert summ["skipped"] == 1 and summ["probed"] == 2

    @pytest.mark.asyncio
    async def test_tool_verify_evidence_persisted_to_sidecar(self, tmp_path, monkeypatch):
        """Verdict AND evidence land under tool_verify in _probe_results.json —
        including the inconclusive case (the 403 evidence base for the eventual
        auto-degrade decision)."""
        import json as _json

        sidecar = tmp_path / "_probe_results.json"
        sidecar.write_text(_json.dumps(
            {"gemini/flash-free": {"tested_at": "2026-09-01T00:00:00Z", "profile": {}}}))
        monkeypatch.setattr("uniinfer.capabilities.core.PROBE_RESULTS_PATH", sidecar)

        doc = self._catalog_doc()
        cat_inst = SimpleNamespace(read_nested=lambda *a, **k: doc,
                                   save_override=lambda mid, fields: None)
        with patch("uniinfer.proxy_services.models_registry.Catalog", lambda: cat_inst), \
             patch("uniinfer.capabilities.core.probe_profile",
                   AsyncMock(return_value=SimpleNamespace(detail={"profile": {}}))), \
             patch("uniinfer.capabilities.core.save_probe_result", lambda r: None), \
             patch("uniinfer.capabilities.core.verify_tool_call",
                   AsyncMock(return_value=(None, "HTTP 403: Mistral API key is required"))):
            await softprobe_catalog(stale_days=None, empirical_tools=True)
        data = _json.loads(sidecar.read_text())
        tv = data["gemini/flash-free"]["tool_verify"]
        assert tv["verdict"] == "none" and "403" in tv["evidence"]
        assert tv["tested_at"]  # stamped
        # the pre-existing probe entry survives the merge
        assert data["gemini/flash-free"]["profile"] == {}
