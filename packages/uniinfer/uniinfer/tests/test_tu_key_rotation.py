"""
TU provider: credgoo key-rotation recovery on upstream 401.

credgoo-issued TU keys rotate without notice; a long-running process (the amd
proxy) holds the old key — baked into the pooled client's Authorization header
— and every call 401s until restart. On 401 the provider refetches the key from
credgoo, rebuilds the pooled client, and retries. An unchanged key (or an
explicitly injected one) raises immediately: permanent auth error.
"""
import json
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from uniinfer import ChatCompletionRequest, ChatMessage
from uniinfer.errors import AuthenticationError
from uniinfer.providers.tu import TUProvider, _TU_CLIENT_CACHE


def _fake_credgoo(monkeypatch, key_holder: dict):
    """Inject a fake credgoo module serving key_holder['key']."""
    fake = types.ModuleType("credgoo")

    def get_api_key(service):
        return key_holder["key"]

    fake.get_api_key = get_api_key
    monkeypatch.setitem(sys.modules, "credgoo", fake)


def _good_response(content: str = "ok") -> MagicMock:
    payload = {
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "model": "m",
        "usage": {"total_tokens": 2},
    }
    m = MagicMock(spec=httpx.Response)
    m.status_code = 200
    m.json.return_value = payload
    m.text = json.dumps(payload)
    return m


def _unauthorized_response() -> MagicMock:
    m = MagicMock(spec=httpx.Response)
    m.status_code = 401
    m.text = '{"error": {"message": "Authentication Required"}}'
    return m


def _pool_client(post_return=None) -> AsyncMock:
    c = AsyncMock(spec=httpx.AsyncClient)
    c.is_closed = False
    c.post.return_value = post_return
    return c


def _request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="m", messages=[ChatMessage(role="user", content="hi")])


@pytest.fixture(autouse=True)
def clean_pool():
    _TU_CLIENT_CACHE.clear()
    yield
    _TU_CLIENT_CACHE.clear()


@pytest.fixture(autouse=True)
def fast_sleep(monkeypatch):
    monkeypatch.setattr("uniinfer.providers.tu.asyncio.sleep", AsyncMock())


class TestTUKeyRotation:
    """401 → credgoo refetch → pooled client rebuilt with the new key → retry."""

    @pytest.mark.asyncio
    async def test_rotation_recovers_on_fresh_client(self, monkeypatch):
        keys = {"key": "stale-key"}
        _fake_credgoo(monkeypatch, keys)
        monkeypatch.delenv("TU_API_KEY", raising=False)

        provider = TUProvider(api_key=None)  # init fetches stale key from credgoo
        assert provider.api_key == "stale-key"

        mock_a = _pool_client(_unauthorized_response())
        mock_b = _pool_client(_good_response("recovered"))
        _TU_CLIENT_CACHE[provider.base_url] = mock_a
        monkeypatch.setattr(TUProvider, "_new_async_client", lambda self: mock_b)

        keys["key"] = "fresh-key"  # rotate between init and the call

        response = await provider.acomplete(_request())

        assert response.message.content == "recovered"
        assert provider.api_key == "fresh-key"
        assert _TU_CLIENT_CACHE[provider.base_url] is mock_b

    @pytest.mark.asyncio
    async def test_unchanged_key_raises_immediately(self, monkeypatch):
        keys = {"key": "stale-key"}
        _fake_credgoo(monkeypatch, keys)
        monkeypatch.delenv("TU_API_KEY", raising=False)

        provider = TUProvider(api_key=None)
        mock_a = _pool_client(_unauthorized_response())
        _TU_CLIENT_CACHE[provider.base_url] = mock_a

        with pytest.raises(AuthenticationError):
            await provider.acomplete(_request())

        assert mock_a.post.await_count == 1  # no blind retry on the same key

    @pytest.mark.asyncio
    async def test_explicit_key_never_refetched(self, monkeypatch):
        calls = []
        fake = types.ModuleType("credgoo")
        fake.get_api_key = lambda service: calls.append(service) or "should-not-be-used"
        monkeypatch.setitem(sys.modules, "credgoo", fake)
        monkeypatch.delenv("TU_API_KEY", raising=False)

        provider = TUProvider(api_key="explicit-key")  # caller-owned
        mock_a = _pool_client(_unauthorized_response())
        _TU_CLIENT_CACHE[provider.base_url] = mock_a

        with pytest.raises(AuthenticationError):
            await provider.acomplete(_request())

        assert calls == []  # no credgoo round-trip for explicit keys

    def test_new_client_carries_current_key(self, monkeypatch):
        """The rebuilt client must carry the REFRESHED key in its headers."""
        keys = {"key": "fresh-key"}
        _fake_credgoo(monkeypatch, keys)
        monkeypatch.delenv("TU_API_KEY", raising=False)

        provider = TUProvider(api_key=None)
        provider.api_key = "fresh-key"  # simulate post-refresh state

        client = provider._new_async_client()

        assert client.headers["Authorization"] == "Bearer fresh-key"
