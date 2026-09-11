"""Retry-After pass-through: an upstream 429's Retry-After must reach the
caller as an HTTP header (clients like the OpenAI SDK honor it for backoff)."""

import httpx
import pytest

from uniinfer.errors import RateLimitError, map_provider_error


def test_ratelimiterror_carries_retry_after():
    e = RateLimitError("rl", status_code=429, response_body="{}", retry_after=12.0)
    assert e.retry_after == 12.0


def test_ratelimiterror_retry_after_optional():
    e = RateLimitError("rl", status_code=429)
    assert e.retry_after is None


def test_map_provider_error_passes_retry_after():
    e = map_provider_error(
        "tu", Exception("TU API error: 429 - quota"), status_code=429, response_body="{}", retry_after=7.5
    )
    assert isinstance(e, RateLimitError)
    assert e.retry_after == 7.5


def _response(status: int, headers: dict | None = None, text: str = "{}") -> httpx.Response:
    return httpx.Response(status, headers=headers or {}, text=text)


@pytest.mark.asyncio
async def test_post_429_parses_retry_after_header():
    """Upstream 429 with Retry-After → RateLimitError carries the parsed value."""
    from uniinfer.providers.tu import TUProvider, TUTelemetry

    TUTelemetry._instance = None
    try:
        prov = TUProvider(api_key="k")
        req = httpx.Request("POST", "https://x/v1/chat/completions")
        resp = _response(429, {"retry-after": "12"}, '{"error":"quota"}')

        async def post(*a, **kw):
            return resp

        client = MagicMockClient(post)
        with pytest.raises(RateLimitError) as exc:
            await prov._post_with_ratelimit_retry(client, "https://x/v1/chat/completions", {}, "m")
        assert exc.value.retry_after == 12.0
        assert exc.value.status_code == 429
    finally:
        TUTelemetry._instance = None


@pytest.mark.asyncio
async def test_post_429_without_header_retry_after_none():
    from uniinfer.providers.tu import TUProvider, TUTelemetry

    TUTelemetry._instance = None
    try:
        prov = TUProvider(api_key="k")
        resp = _response(429, None, '{"error":"quota"}')

        async def post(*a, **kw):
            return resp

        client = MagicMockClient(post)
        with pytest.raises(RateLimitError) as exc:
            await prov._post_with_ratelimit_retry(client, "https://x/v1/chat/completions", {}, "m")
        assert exc.value.retry_after is None
    finally:
        TUTelemetry._instance = None


class MagicMockClient:
    """Minimal httpx.AsyncClient stand-in: only .post() is used on the 429 path."""

    def __init__(self, post_fn):
        self._post_fn = post_fn

    def post(self, *a, **kw):
        return self._post_fn(*a, **kw)
