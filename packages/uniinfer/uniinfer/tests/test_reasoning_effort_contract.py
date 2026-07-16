"""Contract tests for the reasoning-effort thinking-control seam (candidate 1).

Pins the cross-provider contract — "none / minimal disables reasoning" — through
each reasoning-capable provider's own interface, plus escape-hatch precedence and
the deprecated-`think` HTTP shim. See CONTEXT.md.

These are the tests the scattered caller-side translation never had: today there
was zero coverage on reasoning_effort -> native-knob translation.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uniinfer import ChatCompletionRequest, ChatMessage
from uniinfer.core import REASONING_OFF, ReasoningEffort  # noqa: F401  (export check)
from uniinfer.providers.ollama import OllamaProvider
from uniinfer.providers.tu import TUProvider
from uniinfer.providers.zai import _map_reasoning_effort_to_thinking


# --------------------------------------------------------------------------- #
# zai — pure function dialect map
# --------------------------------------------------------------------------- #
class TestZaiReasoningMap:
    @pytest.mark.parametrize("effort", ["none", "minimal"])
    def test_off_levels_disable(self, effort):
        assert _map_reasoning_effort_to_thinking(effort) == {"type": "disabled"}

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_active_levels_enable(self, effort):
        assert _map_reasoning_effort_to_thinking(effort) == {"type": "enabled"}

    @pytest.mark.parametrize("effort", ["off", "disabled", "false", "0"])
    def test_legacy_off_spellings_disable(self, effort):
        assert _map_reasoning_effort_to_thinking(effort) == {"type": "disabled"}

    def test_unset_is_none(self):
        assert _map_reasoning_effort_to_thinking(None) is None


# --------------------------------------------------------------------------- #
# tu — payload building (sync, no I/O): the contract + escape precedence
# --------------------------------------------------------------------------- #
class TestTuPreparePayload:
    def _req(self, **kw):
        base = dict(messages=[ChatMessage(role="user", content="hi")], model="qwen3")
        base.update(kw)
        return ChatCompletionRequest(**base)

    def _payload(self, **kw):
        return TUProvider(api_key="k", supports_reasoning_effort=True)._prepare_payload(
            self._req(**kw)
        )

    @pytest.mark.parametrize("effort", ["none", "minimal"])
    def test_off_levels_disable_via_ctk(self, effort):
        p = self._payload(reasoning_effort=effort)
        assert p["chat_template_kwargs"]["enable_thinking"] is False

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_active_levels_do_not_force_disable(self, effort):
        p = self._payload(reasoning_effort=effort)
        ctk = p.get("chat_template_kwargs", {})
        assert ctk.get("enable_thinking") is not False

    def test_unset_leaves_no_ctk(self):
        p = self._payload()
        assert "chat_template_kwargs" not in p

    def test_escape_hatch_wins_over_off_intent(self):
        """Explicit chat_template_kwargs.enable_thinking=True is not overridden."""
        p = self._payload(
            reasoning_effort="none",
            chat_template_kwargs={"enable_thinking": True},
        )
        assert p["chat_template_kwargs"]["enable_thinking"] is True

    def test_escape_hatch_merged_with_injected_default(self):
        """Other ctk keys are preserved; enable_thinking injected as default."""
        p = self._payload(
            reasoning_effort="none", chat_template_kwargs={"foo": "bar"}
        )
        ctk = p["chat_template_kwargs"]
        assert ctk["foo"] == "bar"
        assert ctk["enable_thinking"] is False

    def test_reasoning_effort_forwarded_when_supported(self):
        p = self._payload(reasoning_effort="high")
        assert p["reasoning_effort"] == "high"

    def test_reasoning_effort_not_forwarded_when_unsupported(self):
        p = TUProvider(api_key="k", supports_reasoning_effort=False)._prepare_payload(
            self._req(reasoning_effort="high")
        )
        assert "reasoning_effort" not in p
        # but the off-contract still applies via ctk for none/minimal
        p2 = TUProvider(api_key="k", supports_reasoning_effort=False)._prepare_payload(
            self._req(reasoning_effort="none")
        )
        assert p2["chat_template_kwargs"]["enable_thinking"] is False


# --------------------------------------------------------------------------- #
# ollama — reasoning_effort -> native `think` flag (pins the asymmetry fix:
# no think= kwarg; the request field drives it for both sync & async paths)
# --------------------------------------------------------------------------- #
class TestOllamaReasoningDialect:
    @staticmethod
    def _run(req):
        provider = OllamaProvider(api_key=None, base_url="http://localhost:11434")

        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "model": req.model,
            "prompt_eval_count": 1,
            "eval_count": 1,
        }
        client_mock = MagicMock()
        client_mock.post = AsyncMock(return_value=resp_mock)

        with patch.object(
            OllamaProvider, "_get_async_client", AsyncMock(return_value=client_mock)
        ):
            asyncio.run(provider.acomplete(req))

        return client_mock.post.call_args.kwargs["json"]

    @pytest.mark.parametrize("effort", ["none", "minimal"])
    def test_off_levels_set_think_false(self, effort):
        payload = self._run(
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")],
                model="qwen3",
                reasoning_effort=effort,
            )
        )
        assert payload["think"] is False

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    def test_active_levels_set_think_true(self, effort):
        payload = self._run(
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")],
                model="qwen3",
                reasoning_effort=effort,
            )
        )
        assert payload["think"] is True

    def test_unset_omits_think(self):
        payload = self._run(
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hi")], model="qwen3"
            )
        )
        assert "think" not in payload


# --------------------------------------------------------------------------- #
# HTTP deprecation shim — legacy `think:false` -> reasoning_effort="none"
# --------------------------------------------------------------------------- #
class TestHttpThinkShim:
    def _client_and_capture(self, monkeypatch):
        from fastapi.testclient import TestClient

        import uniinfer.proxy_routers.chat as chat_mod
        from uniinfer.uniioai_proxy import app

        captured: dict = {}

        async def fake_acomplete(messages, *, reasoning_effort=None, **kw):
            captured["reasoning_effort"] = reasoning_effort
            return SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None, role="assistant"),
                thinking=None,
                finish_reason="stop",
                usage=None,
            )

        fake_target = MagicMock()
        fake_target.acomplete = fake_acomplete
        monkeypatch.setattr(chat_mod, "Target", lambda *a, **k: fake_target)
        monkeypatch.setattr(chat_mod, "verify_provider_access", lambda *a, **k: "k")
        return TestClient(app), captured

    def test_think_false_maps_to_reasoning_none(self, monkeypatch):
        client, captured = self._client_and_capture(monkeypatch)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "ollama@qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "think": False,
            },
        )
        assert resp.status_code == 200
        assert captured.get("reasoning_effort") == "none"

    def test_explicit_reasoning_effort_beats_think_shim(self, monkeypatch):
        client, captured = self._client_and_capture(monkeypatch)
        client.post(
            "/v1/chat/completions",
            json={
                "model": "ollama@qwen3",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "think": False,
                "reasoning_effort": "high",
            },
        )
        # explicit reasoning_effort wins; shim does not clobber it
        assert captured.get("reasoning_effort") == "high"
