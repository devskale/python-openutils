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


# --- TU rate budget + payload audit log + transport-retry knob -----------------

def test_tu_rate_budget_off_by_default(monkeypatch):
    import uniinfer.providers.tu as tu
    monkeypatch.delenv("TU_RATE_LIMIT_PER_MIN", raising=False)
    assert tu._get_tu_rate_bucket() is None


@pytest.mark.asyncio
async def test_tu_rate_budget_throttles_with_retry_after(monkeypatch):
    import uniinfer.providers.tu as tu
    monkeypatch.setenv("TU_RATE_LIMIT_PER_MIN", "2")
    tu._TU_RATE_BUCKET = None; tu._TU_RATE_BUCKET_RATE = 0.0
    await tu._enforce_tu_rate_budget("m")   # token 1
    await tu._enforce_tu_rate_budget("m")   # token 2
    with pytest.raises(RateLimitError) as exc:
        await tu._enforce_tu_rate_budget("m")  # empty -> local 429
    assert exc.value.retry_after is not None and exc.value.retry_after > 0
    assert exc.value.status_code == 429
    tu._TU_RATE_BUCKET = None; tu._TU_RATE_BUCKET_RATE = 0.0
    monkeypatch.delenv("TU_RATE_LIMIT_PER_MIN", raising=False)


def test_log_outgoing_payload_keys_not_content(caplog):
    import logging
    import uniinfer.providers.tu as tu
    with caplog.at_level(logging.INFO, logger="uniinfer.providers.tu"):
        tu._log_outgoing_payload("m", {"model": "m", "messages": [{"role": "user", "content": "GEHEIM"}]}, operation="TEST")
    joined = " ".join(caplog.messages)
    assert "payload keys=[model,messages]" in joined or "keys=[messages,model]" in joined
    assert "GEHEIM" not in joined  # content never logged without UNIINFER_DEBUG_RAW


def test_transport_retries_env_knob(monkeypatch):
    import uniinfer.providers.tu as tu
    monkeypatch.setenv("TU_TRANSPORT_RETRIES", "0")
    prov = tu.TUProvider(api_key="k")

    class C:
        def __init__(self):
            self.calls = 0

        async def post(self, *a, **kw):
            import httpx as _h
            self.calls += 1
            raise _h.ReadTimeout("wedged")

    import httpx as _hx
    from uniinfer.errors import ProviderError
    c = C()
    with pytest.raises(ProviderError):
        import asyncio
        asyncio.run(prov._post_with_ratelimit_retry(c, "https://x/v1/chat/completions", {"model": "m"}, "m"))
    assert c.calls == 1  # 0 retries -> exactly one attempt, no masking
