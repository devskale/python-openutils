"""Contract tests for ``Target`` — the completion dispatch module (candidate 2).

Pins the deepened interface: parse, provider instantiation (incl. extra_params
precedence), request building, the four dispatch paths, raw yields, and the
record_access side-effect contract. See CONTEXT.md.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uniinfer import ChatMessage
from uniinfer.completion import Target, parse_provider_model
from uniinfer.providers.ollama import OllamaProvider


# --------------------------------------------------------------------------- #
# parse_provider_model — the one shared parse
# --------------------------------------------------------------------------- #
class TestParseProviderModel:
    def test_splits_provider_and_model(self):
        assert parse_provider_model("openai@gpt-4") == ("openai", "gpt-4")

    def test_model_may_contain_at(self):
        # split on first @ only (e.g. ollama@gemma3:1b, or weird model ids)
        assert parse_provider_model("ollama@gemma3:1b") == ("ollama", "gemma3:1b")

    def test_missing_at_raises(self):
        with pytest.raises(ValueError):
            parse_provider_model("gpt-4")

    def test_empty_provider_raises(self):
        with pytest.raises(ValueError):
            parse_provider_model("@gpt-4")

    def test_empty_model_raises(self):
        with pytest.raises(ValueError):
            parse_provider_model("openai@")


# --------------------------------------------------------------------------- #
# Target — parse + instantiation + extra_params precedence
# --------------------------------------------------------------------------- #
class TestTargetInstantiation:
    def test_exposes_provider_and_model_names(self):
        with patch("uniinfer.completion.ProviderFactory") as factory:
            factory.get_provider.return_value = MagicMock(name="prov")
            t = Target("openai@gpt-4", api_key="k")
        assert t.provider_name == "openai"
        assert t.model_name == "gpt-4"

    def test_extra_params_merged_from_config(self):
        with patch("uniinfer.completion.ProviderFactory") as factory:
            factory.get_provider.return_value = MagicMock(name="prov")
            Target("cloudflare@x", api_key="k")
        # cloudflare has extra_params in PROVIDER_CONFIGS -> merged into kwargs
        _, kwargs = factory.get_provider.call_args
        assert "api_key" in kwargs

    def test_explicit_base_url_wins_over_config(self):
        with patch("uniinfer.completion.ProviderFactory") as factory:
            factory.get_provider.return_value = MagicMock(name="prov")
            Target("cloudflare@x", api_key="k", base_url="https://explicit")
        _, kwargs = factory.get_provider.call_args
        assert kwargs["base_url"] == "https://explicit"

    def test_record_access_default_true(self):
        with patch("uniinfer.completion.ProviderFactory") as factory:
            factory.get_provider.return_value = MagicMock(name="prov")
            t = Target("openai@gpt-4", api_key="k")
        assert t._record_access is True


# --------------------------------------------------------------------------- #
# Target — dispatch (parametrized over the four paths)
# --------------------------------------------------------------------------- #
def _mock_provider():
    provider = MagicMock(name="provider")
    provider.complete = MagicMock(return_value=_resp("sync"))
    provider.stream_complete = MagicMock(return_value=iter([_resp("chunk")]))
    provider.acomplete = AsyncMock(return_value=_resp("async"))
    provider.astream_complete = MagicMock(return_value=_aiter([_resp("chunk")]))
    return provider


def _resp(content="hi"):
    r = MagicMock(name="resp")
    r.message = MagicMock()
    r.message.content = content
    r.message.tool_calls = None
    r.thinking = None
    r.finish_reason = "stop"
    r.usage = None
    r.raw_response = None
    return r


def _aiter(chunks):
    async def gen():
        for c in chunks:
            yield c
    return gen()


class TestTargetDispatch:
    def _target(self):
        with patch("uniinfer.completion.ProviderFactory") as factory:
            factory.get_provider.return_value = _mock_provider()
            return Target("openai@gpt-4", api_key="k")

    def test_complete_builds_request_and_dispatches(self):
        t = self._target()
        resp = t.complete([{"role": "user", "content": "hi"}], temperature=0.5, max_tokens=10)
        t.provider.complete.assert_called_once()
        req = t.provider.complete.call_args.args[0]
        assert req.streaming is False
        assert req.temperature == 0.5
        assert req.max_tokens == 10
        assert resp.message.content == "sync"

    def test_stream_complete_sets_streaming_and_yields_raw(self):
        t = self._target()
        chunks = list(t.stream_complete([{"role": "user", "content": "hi"}]))
        req = t.provider.stream_complete.call_args.args[0]
        assert req.streaming is True
        assert len(chunks) == 1  # raw ChatCompletionResponse, not formatted dicts

    def test_acomplete_builds_request_and_dispatches(self):
        t = self._target()
        resp = asyncio.run(
            t.acomplete([{"role": "user", "content": "hi"}], reasoning_effort="none")
        )
        req = t.provider.acomplete.call_args.args[0]
        assert req.streaming is False
        assert req.reasoning_effort == "none"
        assert resp.message.content == "async"

    def test_astream_complete_sets_streaming_and_yields_raw(self):
        t = self._target()
        chunks = asyncio.run(_collect(t.astream_complete([{"role": "user", "content": "hi"}])))
        req = t.provider.astream_complete.call_args.args[0]
        assert req.streaming is True
        assert len(chunks) == 1

    def test_accepts_chatmessage_objects(self):
        t = self._target()
        t.complete([ChatMessage(role="user", content="hi")])
        req = t.provider.complete.call_args.args[0]
        assert req.messages[0].role == "user"


async def _collect(agen):
    out = []
    async for c in agen:
        out.append(c)
    return out


# --------------------------------------------------------------------------- #
# Target — record_access side-effect contract (Q4)
# --------------------------------------------------------------------------- #
class TestTargetRecordAccess:
    def test_records_on_complete_by_default(self):
        with patch("uniinfer.completion.ProviderFactory") as factory, patch(
            "uniinfer.completion.update_model_accessed"
        ) as recorded:
            factory.get_provider.return_value = _mock_provider()
            Target("openai@gpt-4", api_key="k").complete([{"role": "user", "content": "hi"}])
        recorded.assert_called_once_with("gpt-4", "openai")

    def test_does_not_record_when_record_access_false(self):
        with patch("uniinfer.completion.ProviderFactory") as factory, patch(
            "uniinfer.completion.update_model_accessed"
        ) as recorded:
            factory.get_provider.return_value = _mock_provider()
            Target("openai@gpt-4", api_key="k", record_access=False).complete(
                [{"role": "user", "content": "hi"}]
            )
        recorded.assert_not_called()

    def test_records_after_stream_completes(self):
        with patch("uniinfer.completion.ProviderFactory") as factory, patch(
            "uniinfer.completion.update_model_accessed"
        ) as recorded:
            factory.get_provider.return_value = _mock_provider()
            list(
                Target("openai@gpt-4", api_key="k").stream_complete(
                    [{"role": "user", "content": "hi"}]
                )
            )
        recorded.assert_called_once_with("gpt-4", "openai")

    def test_stream_respects_record_access_false(self):
        with patch("uniinfer.completion.ProviderFactory") as factory, patch(
            "uniinfer.completion.update_model_accessed"
        ) as recorded:
            factory.get_provider.return_value = _mock_provider()
            list(
                Target("openai@gpt-4", api_key="k", record_access=False).stream_complete(
                    [{"role": "user", "content": "hi"}]
                )
            )
        recorded.assert_not_called()

    def test_record_failure_does_not_break_completion(self):
        with patch("uniinfer.completion.ProviderFactory") as factory, patch(
            "uniinfer.completion.update_model_accessed",
            side_effect=RuntimeError("disk full"),
        ):
            factory.get_provider.return_value = _mock_provider()
            resp = Target("openai@gpt-4", api_key="k").complete(
                [{"role": "user", "content": "hi"}]
            )
        assert resp.message.content == "sync"  # completion still succeeded


# --------------------------------------------------------------------------- #
# Target — real ollama provider: reasoning_effort flows to the payload
# (integration of candidate 1 + 2: no think= kwarg, request field drives it)
# --------------------------------------------------------------------------- #
class TestTargetOllamaReasoning:
    def test_reasoning_effort_none_drives_ollama_think_false(self):
        provider = OllamaProvider(api_key=None, base_url="http://localhost:11434")
        resp_mock = MagicMock()
        resp_mock.status_code = 200
        resp_mock.json.return_value = {
            "message": {"role": "assistant", "content": "ok"},
            "model": "qwen3",
            "prompt_eval_count": 1,
            "eval_count": 1,
        }
        client_mock = MagicMock()
        client_mock.post = AsyncMock(return_value=resp_mock)
        with patch.object(
            OllamaProvider, "_get_async_client", AsyncMock(return_value=client_mock)
        ), patch("uniinfer.completion.ProviderFactory") as factory:
            factory.get_provider.return_value = provider
            t = Target("ollama@qwen3", record_access=False)
            asyncio.run(t.acomplete([{"role": "user", "content": "hi"}], reasoning_effort="none"))
        sent = client_mock.post.call_args.kwargs["json"]
        assert sent["think"] is False
