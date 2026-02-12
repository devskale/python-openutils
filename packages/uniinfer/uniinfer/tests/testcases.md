## Request Limits Verification

**script:** `uniinfer/tests/test_request_limits.py`
**howtorun:** `uv run pytest uniinfer/tests/test_request_limits.py`
**description:** Validates request size limits and payload validation using `TestClient` and mocks.

1. Validates that requests > 10MB return 413 Entity Too Large.
2. Validates that requests with > 500 messages return 422 Unprocessable Entity.
3. Validates that invalid model formats return 422 Unprocessable Entity.
   **result:** All tests passed.

## Live Verification

**script:** `uniinfer/tests/verify_live_proxy.py`
**howtorun:** `uv run python uniinfer/tests/verify_live_proxy.py`
**description:** Starts the proxy server as a subprocess and runs end-to-end verification checks against it using `requests`.

1. Validates that /v1/chat/completions requires auth for protected providers (401).
2. Validates that Ollama requests bypass auth (reach provider).
3. Validates that rate limits are enforced (triggering 429 after limit is exceeded).
4. Verifies presence of X-RateLimit headers on 401 (auth failure) and 429 (rate limit exceeded) responses.
   **result:** All live checks passed.

## Authentication Logic Verification

**script:** `uniinfer/tests/test_auth.py`
**howtorun:** `uv run pytest uniinfer/tests/test_auth.py`
**description:** Unit tests for the authentication module (`uniinfer/auth.py`).

1. Validates `validate_proxy_token` dependency.
2. Validates `get_optional_proxy_token` dependency.
3. Tests token format validation (Bearer scheme).
   **result:** All tests passed.

## Rate Limiting Verification

**script:** `uniinfer/tests/test_rate_limiting.py`
**howtorun:** `uv run pytest uniinfer/tests/test_rate_limiting.py`
**description:** Tests the integration of `slowapi` rate limiting with the proxy server.

1. Sets low limits via environment variables.
2. Verifies that requests exceeding the limit return 429 Too Many Requests.
   **result:** All tests passed.

## Proxy Security Verification

**script:** `uniinfer/tests/test_proxy_security.py`
**howtorun:** `uv run pytest uniinfer/tests/test_proxy_security.py`
**description:** Tests security controls on the proxy endpoints.

1. Verifies that `/v1/chat/completions` requires authentication for non-local providers.
2. Tests response status codes for unauthorized access (401).
   **result:** All tests passed.
