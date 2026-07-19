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
