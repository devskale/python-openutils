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
