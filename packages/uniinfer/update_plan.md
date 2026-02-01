# UniInfer Update & Improvement Plan

## Executive Summary

Based on comprehensive code review findings, **UniInfer is currently NOT PRODUCTION-READY**. This plan addresses all critical issues systematically.

**Current Status:**

- Version: 0.4.1 (with inconsistency - **init**.py shows 0.1.0)
- Critical Issues: 10 found
- Test Coverage: ~5% (only 3 test files)
- Security Vulnerabilities: 3 critical, 4 high
- Production Readiness: ❌ NO

**Target Status:**

- Production-ready with async support, comprehensive testing, and security fixes
- Test coverage: 80%+
- All critical vulnerabilities resolved
- Version: 1.0.0

---

## Phases Overview

| Phase       | Duration    | Focus                  | Priority                         | Output |
| ----------- | ----------- | ---------------------- | -------------------------------- | ------ |
| **Phase 1** | 1 week      | Critical Security      | Secure the codebase              |
| **Phase 2** | 2 weeks     | Core Architecture      | Add async, input validation      |
| **Phase 3** | 2 weeks     | Testing & CI           | Comprehensive test suite, CI/CD  |
| **Phase 4** | 2 weeks     | Production Features    | Token tracking, context, caching |
| **Phase 5** | 1 week      | Documentation & Polish | Docs, cleanup, release           |
| **Total**   | **8 weeks** | -                      | -                                |

---

## Phase 1: Critical Security Fixes (Week 1)

### 1.1 Fix Version Inconsistency (Priority: CRITICAL)

**Issue:** `__init__.py` shows `0.1.0`, `setup.py` shows `0.4.1`

**Steps:**

1.45→1. [x] Determine correct version (v0.4.7)
46→2. [x] Update `uniinfer/__init__.py`: Change `__version__ = "0.4.7"`
47→3. [x] Update `setup.py`:

**Files:**

- `uniinfer/__init__.py:108`
- `setup.py:71`

**Time:** 1 hour

**Verification:**

```bash
python -c "import uniinfer; print(uniinfer.__version__)"
# Should output: 0.5.0
```

---

### 1.2 Secure Proxy Server (Priority: CRITICAL)

**Issue:** `uniioai_proxy.py` (47KB) has no rate limiting, auth validation, or DDoS protection

**Steps:**

**1.2.1 Add Rate Limiting**

1. [ ] Install slowapi: `pip install slowapi`
2. [ ] Add rate limiter to imports: `from slowapi import Limiter`
3. [ ] Initialize limiter: `limiter = Limiter(key_func=get_remote_address)`
4. [ ] Add rate limit to `/v1/chat/completions`:
   ```python
   @app.post("/v1/chat/completions")
   @limiter.limit("100/minute")  # 100 requests per minute
   async def chat_completions(request: Request):
   ```
   5.82→5. [x] Add rate limit to `/v1/embeddings`:
   ```python
   @app.post("/v1/embeddings")
   @limiter.limit("200/minute")
   async def embeddings(request: Request):
   ```
   5.88→6. [x] Add rate limit to `/v1/audio/*` endpoints

**Files:**

- `uniinfer/uniioai_proxy.py` (multiple locations)
- `setup.py` (add slowapi dependency)

**Time:** 3 hours

**1.2.2 Add Authentication Validation**

1.99→1. [x] Create auth module: `uniinfer/auth.py`
100→2. [x] Implement token validation:

```python
def validate_auth_token(token: str) -> tuple[bool, str | None]:
    # Check Bearer token format
    # Validate with credgoo if needed
    # Return (is_valid, provider_name)
```

3. [ ] Add auth decorator: `require_auth()`
4. [ ] Apply to all endpoints
5. [ ] Add error response for invalid auth

**Files:**

- `uniinfer/auth.py` (new file)
- `uniinfer/uniioai_proxy.py` (import and use)

**Time:** 2 hours

**1.2.3 Add Request Size Limits**

1. [ ] Add max request size to FastAPI config:
   ```python
   app = FastAPI(
       max_request_size=10 * 1024 * 1024  # 10MB
   )
   ```
2. [ ] Validate message count in request body
3. [ ] Validate model parameter
4. [ ] Return 413 if too large

**Files:**

- `uniinfer/uniioai_proxy.py:1` (app initialization)

**Time:** 1 hour

**1.2.4 Add CORS Configuration**

1. [ ] Import CORSMiddleware: `from fastapi.middleware.cors import CORSMiddleware`
2. [ ] Add CORS middleware:
   ```python
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # Configure for production
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

**Files:**

- `uniinfer/uniioai_proxy.py`

**Time:** 30 minutes

**1.2.5 Add Logging**

1. [x] Install python-json-logger: `pip install python-json-logger`
2. [x] Configure structured logging
3. [x] Add request logging (IP, endpoint, status, duration)
4. [x] Add error logging
5. [x] Configure log levels

**Files:**

- `uniinfer/uniioai_proxy.py` (add logging throughout)

**Time:** 2 hours

---

### 1.3 Add Dependency Lockfile (Priority: HIGH)

**Issue:** No requirements.lock or poetry.lock, supply chain attack risk

**Steps:**

**1.3.1 Migrate to Poetry (Recommended)**

1. [ ] Install poetry: `pip install poetry`
2. [ ] Initialize poetry: `poetry init`
3. [ ] Convert setup.py to pyproject.toml
4. [ ] Move dependencies to pyproject.toml:
   ```toml
   [tool.poetry.dependencies]
   python = "^3.7"
   requests = "^2.25.0"
   fastapi = "^0.100.0"
   # ... etc.
   ```
5. [ ] Move dev dependencies:
   ```toml
   [tool.poetry.group.dev.dependencies]
   pytest = "^7.0"
   black = "^23.0"
   mypy = "^1.0"
   ```
6. [ ] Install dependencies: `poetry install`
7. [ ] Generate lockfile: `poetry lock`
8. [ ] Commit poetry.lock to repo

**Files:**

- `pyproject.toml` (new, replacing setup.py)
- `poetry.lock` (new, commit this)
- `setup.py` (delete after migration)

**Time:** 4 hours

**Alternative (Keep setup.py):**

1. [ ] Install pip-tools: `pip install pip-tools`
2. [ ] Create requirements.txt with pinned versions
3. [ ] Generate lock: `pip-compile requirements.in`
4. [ ] Commit requirements.txt

**Time:** 2 hours

---

### 1.4 Add Security Scanning (Priority: HIGH)

**Issue:** No automated security vulnerability scanning

**Steps:**

**1.4.1 Add pip-audit to CI (See Phase 3)**
**1.4.2 Add Pre-commit Hooks**

1. [ ] Install pre-commit: `pip install pre-commit`
2. [ ] Create `.pre-commit-config.yaml`:
   ```yaml
   repos:
     - repo: https://github.com/PyCQA/bandit
       rev: master
       hooks:
         - id: bandit
           args: [-c, pyproject.toml]
     - repo: https://github.com/psf/black
       rev: 23.12.1
       hooks:
         - id: black
     - repo: https://github.com/pycqa/isort
       rev: 5.12.0
       hooks:
         - id: isort
   ```
3. [ ] Install pre-commit: `pre-commit install`

**Files:**

- `.pre-commit-config.yaml` (new)
- `pyproject.toml` (add bandit config)

**Time:** 2 hours

**Verification:**

```bash
pre-commit run --all-files
```

---

## Phase 2: Core Architecture Improvements (Weeks 2-3)

### 2.1 Add Input Validation (Priority: CRITICAL)

**Issue:** No parameter validation, security vulnerability

**Steps:**

**2.1.1 Create Validation Module**

1. [ ] Create `uniinfer/validation.py`
2. [ ] Implement parameter validators:

   ```python
   from typing import Optional, List
   from pydantic import BaseModel, validator

   class ChatCompletionRequestV2(BaseModel):
       messages: List[ChatMessage]
       model: Optional[str] = None
       temperature: float = 1.0
       max_tokens: Optional[int] = None

       @validator('temperature')
       def validate_temperature(cls, v):
           if not 0.0 <= v <= 2.0:
               raise ValueError('temperature must be between 0.0 and 2.0')
           return v

       @validator('max_tokens')
       def validate_max_tokens(cls, v):
           if v is not None and v <= 0:
               raise ValueError('max_tokens must be positive')
           return v
   ```

3. [ ] Add model name validation
4. [ ] Add message format validation
5. [ ] Add tool schema validation

**Files:**

- `uniinfer/validation.py` (new file)
- `setup.py` or `pyproject.toml` (add pydantic)

**Time:** 6 hours

**2.1.2 Update Core Classes to Use Validation**

1. [ ] Refactor `ChatCompletionRequest` to use Pydantic
2. [ ] Update providers to validate requests
3. [ ] Add validation error handling

**Files:**

- `uniinfer/core.py` (rewrite ChatCompletionRequest)
- `uniinfer/providers/*.py` (add validation calls)

**Time:** 4 hours

---

### 2.2 Add Async Support (Priority: CRITICAL)

**Issue:** No async support, fundamental performance limitation

**Steps:**

**2.2.1 Refactor Base Provider Classes**

1. [ ] Add async abstract methods to `core.py`:

   ```python
   from abc import ABC, abstractmethod
   from typing import AsyncIterator

   class ChatProvider(ABC):
       @abstractmethod
       async def acomplete(
           self,
           request: ChatCompletionRequest,
           **kwargs
       ) -> ChatCompletionResponse:
           pass

       @abstractmethod
       async def astream_complete(
           self,
           request: ChatCompletionRequest,
           **kwargs
       ) -> AsyncIterator[ChatCompletionResponse]:
           pass

       # Keep sync methods as wrappers
       def complete(self, request, **kwargs):
           import asyncio
           return asyncio.run(self.acomplete(request, **kwargs))
   ```

2. [ ] Update all provider base classes

**Files:**

- `uniinfer/core.py` (add async methods)
- `uniinfer/providers/*.py` (implement async in all 27 providers)

**Time:** 20 hours (1 hour per provider × 20, plus core changes)

**2.2.2 Implement Async HTTP Client**

1. [ ] Install httpx: `pip install httpx`
2. [ ] Create async client factory:

   ```python
   import httpx

   class AsyncProviderBase:
       def __init__(self):
           self._client = httpx.AsyncClient(
               timeout=httpx.Timeout(60.0, connect=10.0),
               limits=httpx.Limits(max_connections=100),
           )

       async def close(self):
           await self._client.aclose()
   ```

3. [ ] Add connection pooling
4. [ ] Add keep-alive

**Files:**

- `uniinfer/http_client.py` (new)
- `setup.py` or `pyproject.toml` (add httpx)

**Time:** 4 hours

**2.2.3 Update Provider Implementations**

1. [ ] Rewrite OpenAI provider with async
2. [ ] Rewrite Anthropic provider with async
3. [ ] Rewrite top 5 most-used providers with async
4. [ ] Add TODO comments for remaining providers

**Files:**

- `uniinfer/providers/openai.py`
- `uniinfer/providers/anthropic.py`
- `uniinfer/providers/gemini.py`
- `uniinfer/providers/mistral.py`
- `uniinfer/providers/ollama.py`

**Time:** 10 hours

---

### 2.3 Improve Error Handling (Priority: MEDIUM)

**Issue:** Broad exception catching, poor error messages

**Steps:**

**2.3.1 Refactor Error Handling in Providers**

1. [ ] Create specific exception types in `errors.py`:

   ```python
   class NetworkError(ProviderError):
       pass

   class APIError(ProviderError):
       def __init__(self, message: str, status_code: int, response_body: str):
           super().__init__(message, status_code, response_body)
           self.response_body = response_body  # For debugging
   ```

2. [ ] Replace broad `except Exception` with specific catches
3. [ ] Add request IDs for debugging
4. [ ] Sanitize error messages (remove API keys)

**Files:**

- `uniinfer/errors.py` (add new exception classes)
- `uniinfer/providers/*.py` (update all error handling)

**Time:** 6 hours

**2.3.2 Add Structured Logging**

1. [ ] Install python-json-logger
2. [ ] Configure structured logging
3. [ ] Add logging to all providers
4. [ ] Add logging to core modules

**Files:**

- `uniinfer/logging_config.py` (new)
- `uniinfer/__init__.py` (configure logging)

**Time:** 3 hours

---

## Phase 3: Testing & CI/CD (Weeks 4-5)

### 3.1 Create Comprehensive Test Suite (Priority: CRITICAL)

**Issue:** Only 3 test files, ~5% coverage

**Steps:**

**3.1.1 Set Up Testing Infrastructure**

1. [ ] Update pytest configuration in `pyproject.toml`:
   ```toml
   [tool.pytest.ini_options]
   testpaths = ["tests"]
   python_files = ["test_*.py"]
   python_classes = ["Test*"]
   python_functions = ["test_*"]
   addopts = [
       "--cov=uniinfer",
       "--cov-report=html",
       "--cov-report=term-missing",
       "--cov-fail-under=80",
   ]
   ```
2. [ ] Install pytest-cov: `pip install pytest-cov pytest-mock`

**Files:**

- `pyproject.toml` (add pytest config)

**Time:** 1 hour

**3.1.2 Add Unit Tests for Core**

1. [ ] Create `tests/test_core.py`:

   ```python
   import pytest
   from uniinfer import ChatMessage, ChatCompletionRequest

   class TestChatMessage:
       def test_create_message(self):
           msg = ChatMessage(role="user", content="Hello")
           assert msg.role == "user"
           assert msg.content == "Hello"

       def test_to_dict(self):
           msg = ChatMessage(role="user", content="Hello")
           d = msg.to_dict()
           assert d["role"] == "user"
           assert d["content"] == "Hello"

   class TestChatCompletionRequest:
       def test_temperature_validation(self):
           with pytest.raises(ValueError):
               ChatCompletionRequest(
                   messages=[],
                   temperature=3.0  # Invalid
               )

       def test_max_tokens_validation(self):
           with pytest.raises(ValueError):
                   ChatCompletionRequest(
                       messages=[],
                       max_tokens=-100  # Invalid
                   )
   ```

2. [ ] Add tests for all core classes
3. [ ] Add tests for factory
4. [ ] Add tests for error handling

**Files:**

- `tests/test_core.py` (new)
- `tests/test_factory.py` (new)
- `tests/test_errors.py` (new)

**Time:** 4 hours

**3.1.3 Add Provider Tests**

1. [ ] Create test structure for each provider:
   - `tests/providers/test_openai.py`
   - `tests/providers/test_anthropic.py`
   - `tests/providers/test_mistral.py`
   - `tests/providers/test_ollama.py`
   - etc.
2. [ ] Add test template:

   ```python
   import pytest
   from unittest.mock import Mock, patch
   from uniinfer import OpenAIProvider

   @pytest.fixture
   def mock_openai_response():
       return {
           "id": "chatcmpl-abc",
           "object": "chat.completion",
           "created": 1699000000,
           "model": "gpt-4",
           "choices": [{
               "message": {"role": "assistant", "content": "Test"}
           }],
           "usage": {"prompt_tokens": 10, "completion_tokens": 20}
       }

   class TestOpenAIProvider:
       @patch('uniinfer.providers.openai.requests.post')
       def test_complete_success(self, mock_post, mock_openai_response):
           mock_post.return_value.json.return_value = mock_openai_response
           provider = OpenAIProvider(api_key="test-key")

           request = ChatCompletionRequest(
               messages=[ChatMessage(role="user", content="Hello")]
           )
           response = provider.complete(request)

           assert response.message.content == "Test"
           assert response.usage["total_tokens"] == 30

       @patch('uniinfer.providers.openai.requests.post')
       def test_complete_auth_error(self, mock_post):
           mock_post.return_value.status_code = 401
           provider = OpenAIProvider(api_key="bad-key")

           request = ChatCompletionRequest(
               messages=[ChatMessage(role="user", content="Hello")]
           )
           with pytest.raises(AuthenticationError):
               provider.complete(request)

       @patch('uniinfer.providers.openai.requests.post')
       def test_rate_limit_error(self, mock_post):
           mock_post.return_value.status_code = 429
           provider = OpenAIProvider(api_key="test-key")

           request = ChatCompletionRequest(
               messages=[ChatMessage(role="user", content="Hello")]
           )
           with pytest.raises(RateLimitError):
               provider.complete(request)
   ```

3. [ ] Implement tests for top 5 providers (OpenAI, Anthropic, Mistral, Gemini, Ollama)
4. [ ] Add basic tests for remaining providers (at minimum, one test each)

**Files:**

- `tests/providers/test_openai.py` (new)
- `tests/providers/test_anthropic.py` (new)
- `tests/providers/test_mistral.py` (new)
- `tests/providers/test_gemini.py` (new)
- `tests/providers/test_ollama.py` (new)
- etc.

**Time:** 12 hours (2-3 hours per provider for top 5, 30 min each for rest)

**3.1.4 Add Integration Tests**

1. [ ] Create `tests/integration/` directory
2. [ ] Add integration tests with real API calls (use test API keys)
3. [ ] Add tests for streaming
4. [ ] Add tests for tool calling
5. [ ] Add tests for fallback strategies

**Files:**

- `tests/integration/test_streaming.py` (new)
- `tests/integration/test_tools.py` (new)
- `tests/integration/test_fallbacks.py` (new)

**Time:** 4 hours

**3.1.5 Achieve 80%+ Coverage**

1. [ ] Run coverage: `pytest --cov=uniinfer`
2. [ ] Identify uncovered code
3. [ ] Add tests for uncovered paths
4. [ ] Repeat until 80%+ coverage

**Time:** 6 hours (iterative)

---

### 3.2 Set Up CI/CD Pipeline (Priority: HIGH)

**Issue:** No automated testing or quality checks

**Steps:**

**3.2.1 Create GitHub Actions Workflow**

1. [ ] Create `.github/workflows/ci.yml`:

   ```yaml
   name: CI

   on:
     push:
       branches: [main, develop]
     pull_request:
       branches: [main]

   jobs:
     test:
       runs-on: ubuntu-latest
       strategy:
         matrix:
           python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]

       steps:
         - uses: actions/checkout@v3

         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: ${{ matrix.python-version }}

         - name: Install Poetry
           run: pip install poetry

         - name: Install dependencies
           run: poetry install

         - name: Run tests
           run: poetry run pytest --cov=uniinfer --cov-fail-under=80

         - name: Upload coverage
           uses: codecov/codecov-action@v3

     lint:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3

         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: "3.11"

         - name: Install Poetry
           run: pip install poetry

         - name: Install dependencies
           run: poetry install

         - name: Run Black
           run: poetry run black --check uniinfer/

         - name: Run isort
           run: poetry run isort --check-only uniinfer/

         - name: Run mypy
           run: poetry run mypy uniinfer/

         - name: Run bandit
           run: poetry run bandit -r uniinfer/

     security:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3

         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: "3.11"

         - name: Install Poetry
           run: pip install poetry

         - name: Install dependencies
           run: poetry install

         - name: Run pip-audit
           run: poetry run pip-audit || true
   ```

**Files:**

- `.github/workflows/ci.yml` (new)

**Time:** 2 hours

**3.2.2 Add Coverage Reporting**

1. [ ] Sign up for Codecov
2. [ ] Add Codecov token to repo secrets
3. [ ] Configure coverage reporting in CI

**Time:** 30 minutes

---

## Phase 4: Production Features (Weeks 6-7)

### 4.1 Add Token Tracking (Priority: HIGH)

**Issue:** No token counting or cost estimation

**Steps:**

**4.1.1 Add Token Counter**

1. [ ] Create `uniinfer/token_counter.py`
2. [ ] Implement token counting for different providers:

   ```python
   from tiktoken import encoding_for_model

   def count_tokens(text: str, model: str) -> int:
       try:
           encoding = encoding_for_model(model)
           return len(encoding.encode(text))
       except KeyError:
           # Fallback for unsupported models
           return len(text.split()) * 1.3  # Approximate
   ```

3. [ ] Add support for OpenAI, Anthropic, Google models

**Files:**

- `uniinfer/token_counter.py` (new)
- `setup.py` or `pyproject.toml` (add tiktoken)

**Time:** 4 hours

**4.1.2 Update Responses to Include Token Usage**

1. [ ] Add token counting to `ChatCompletionResponse`
2. [ ] Add `estimated_tokens` field
3. [ ] Add `estimated_cost` field
4. [ ] Calculate costs per provider/model

**Files:**

- `uniinfer/core.py` (update ChatCompletionResponse)

**Time:** 3 hours

**4.1.3 Add Cost Estimation**

1. [ ] Create `uniinfer/pricing.py`
2. [ ] Add pricing data for models:

   ```python
   MODEL_PRICING = {
       "gpt-4": {"input": 0.03, "output": 0.06, "per": 1000},
       "gpt-3.5-turbo": {"input": 0.001, "output": 0.002, "per": 1000},
       "claude-3-opus-20240229": {"input": 0.015, "output": 0.075, "per": 1000},
       # ... etc.
   }

   def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
       if model not in MODEL_PRICING:
           return 0.0
       pricing = MODEL_PRICING[model]
       input_cost = (input_tokens / pricing["per"]) * pricing["input"]
       output_cost = (output_tokens / pricing["per"]) * pricing["output"]
       return input_cost + output_cost
   ```

**Files:**

- `uniinfer/pricing.py` (new)

**Time:** 2 hours

---

### 4.2 Add Context Management (Priority: HIGH)

**Issue:** No context serialization or persistence

**Steps:**

**4.2.1 Create Context Data Structures**

1. [ ] Create `uniinfer/context.py`
2. [ ] Define context types:

   ```python
   from typing import List, Dict, Any
   from dataclasses import dataclass, asdict
   from datetime import datetime

   @dataclass
   class Context:
       system_prompt: str | None = None
       messages: List[Dict[str, Any]]
       metadata: Dict[str, Any] = None

       def to_dict(self) -> Dict:
           return asdict(self)

       @classmethod
       def from_dict(cls, data: Dict) -> 'Context':
           return cls(**data)

       def to_json(self) -> str:
           import json
           return json.dumps(self.to_dict())

       @classmethod
       def from_json(cls, json_str: str) -> 'Context':
           import json
           return cls.from_dict(json.loads(json_str))
   ```

3. [ ] Add timestamp tracking

**Files:**

- `uniinfer/context.py` (new)

**Time:** 3 hours

**4.2.2 Add Context Manager**

1. [ ] Create `uniinfer/context_manager.py`
2. [ ] Implement context persistence:

   ```python
   class ContextManager:
       def __init__(self, storage_backend='memory'):
           self.storage = self._get_storage(storage_backend)
           self.contexts = {}

       def save_context(self, session_id: str, context: Context):
           self.contexts[session_id] = context
           self.storage.save(session_id, context.to_json())

       def load_context(self, session_id: str) -> Context | None:
           if session_id in self.contexts:
               return self.contexts[session_id]
           json_str = self.storage.load(session_id)
           if json_str:
               return Context.from_json(json_str)
           return None
   ```

3. [ ] Add support for different backends (memory, file, redis)

**Files:**

- `uniinfer/context_manager.py` (new)

**Time:** 4 hours

---

### 4.3 Add Caching (Priority: HIGH)

**Issue:** No prompt caching or response caching

**Steps:**

**4.3.1 Create Cache Interface**

1. [ ] Create `uniinfer/cache.py`
2. [ ] Implement caching:

   ```python
   from hashlib import sha256
   import json
   from typing import Optional

   class Cache:
       def __init__(self, backend='memory', ttl=300):
           self.backend = backend
           self.ttl = ttl
           self.cache = {}

       def _generate_key(self, request: ChatCompletionRequest) -> str:
           # Create hash from request
           data = json.dumps({
               "model": request.model,
               "messages": request.messages,
               "temperature": request.temperature,
           }, sort_keys=True)
           return sha256(data.encode()).hexdigest()

       def get(self, request: ChatCompletionRequest) -> Optional[ChatCompletionResponse]:
           key = self._generate_key(request)
           return self.cache.get(key)

       def set(self, request: ChatCompletionRequest, response: ChatCompletionResponse):
           key = self._generate_key(request)
           self.cache[key] = {
               "response": response,
               "timestamp": time.time()
           }

       def cleanup(self):
           # Remove expired entries
           now = time.time()
           expired = [k for k, v in self.cache.items()
                      if now - v["timestamp"] > self.ttl]
           for k in expired:
               del self.cache[k]
   ```

**Files:**

- `uniinfer/cache.py` (new)

**Time:** 3 hours

**4.3.2 Add Provider-Specific Caching**

1. [ ] Add caching flags to provider base class
2. [ ] Implement Anthropic prompt caching
3. [ ] Implement OpenAI caching (if available)
4. [ ] Add cache hit/miss metrics

**Files:**

- `uniinfer/core.py` (add caching to ChatProvider)
- `uniinfer/providers/anthropic.py` (add prompt caching)

**Time:** 4 hours

---

### 4.4 Add Retry Logic (Priority: MEDIUM)

**Issue:** No exponential backoff or retry logic

**Steps:**

**4.4.1 Create Retry Decorator**

1. [ ] Create `uniinfer/retry.py`
2. [ ] Implement retry logic:

   ```python
   import time
   from functools import wraps
   from typing import Callable, Type

   def retry(
       max_retries: int = 3,
       backoff_factor: float = 2.0,
       exceptions: tuple[Type[Exception], ...] = (Exception,)
   ):
       def decorator(func: Callable):
           @wraps(func)
           def wrapper(*args, **kwargs):
               last_exception = None
               for attempt in range(max_retries + 1):
                   try:
                       return func(*args, **kwargs)
                   except exceptions as e:
                       last_exception = e
                       if attempt < max_retries:
                           delay = backoff_factor ** attempt
                           time.sleep(delay)
               raise last_exception
           return wrapper
       return decorator
   ```

**Files:**

- `uniinfer/retry.py` (new)

**Time:** 2 hours

**4.4.2 Apply Retry to Providers**

1. [ ] Add retry decorator to provider methods
2. [ ] Configure retries per provider
3. [ ] Add jitter to prevent thundering herd

**Files:**

- `uniinfer/providers/*.py` (add @retry decorators)

**Time:** 3 hours

---

## Phase 5: Documentation & Polish (Week 8)

### 5.1 Clean Up Repository (Priority: LOW)

**Issue:** Examples directory polluted with 20+ image files

**Steps:**

**5.1.1 Clean Examples Directory**

1. [ ] Remove all image files from `examples/`
2. [ ] Keep only `.py` files
3. [ ] Create `examples/images/` if needed
4. [ ] Move actual example images to subdirectory

**Files:**

- `examples/` (remove image files)

**Time:** 30 minutes

**5.1.2 Split Large Files**

1. [ ] Split `uniinfer_cli.py` into modules:
   - `cli/main.py`
   - `cli/commands.py`
   - `cli/utils.py`
2. [ ] Split `uniioai_proxy.py` into modules:
   - `proxy/server.py`
   - `proxy/endpoints.py`
   - `proxy/middleware.py`
   - `proxy/auth.py`

**Files:**

- `uniinfer/cli/` (new directory)
- `uniinfer/proxy/` (new directory)

**Time:** 6 hours

---

### 5.2 Update Documentation (Priority: MEDIUM)

**Issue:** Version mismatch, missing sections

**Steps:**

**5.2.1 Update README**

1. [ ] Add version badge with correct version
2. [ ] Add production readiness notice
3. [ ] Add migration guide (from 0.4.x to 0.5.0)
4. [ ] Add troubleshooting section
5. [ ] Add examples for new features (context, caching, tokens)
6. [ ] Add security best practices section

**Files:**

- `README.md`

**Time:** 3 hours

**5.2.2 Create Migration Guide**

1. [ ] Create `MIGRATION.md`
2. [ ] Document breaking changes
3. [ ] Provide migration examples

**Files:**

- `MIGRATION.md` (new)

**Time:** 2 hours

**5.2.3 Update AGENTS.md**

1. [ ] Add new development guidelines
2. [ ] Document async patterns
3. [ ] Document validation patterns
4. [ ] Document testing requirements

**Files:**

- `AGENTS.md`

**Time:** 2 hours

---

### 5.3 Prepare for Release (Priority: HIGH)

**Issue:** No structured release process

**Steps:**

**5.3.1 Create Changelog**

1. [ ] Create `CHANGELOG.md`
2. [ ] Add 0.5.0 release notes:

   ```markdown
   # Changelog

   ## [0.5.0] - 2026-03-XX

   ### Added

   - Async support for all providers
   - Input validation with Pydantic
   - Token tracking and cost estimation
   - Context management and serialization
   - Response caching
   - Comprehensive test suite (80%+ coverage)
   - CI/CD pipeline with GitHub Actions
   - Retry logic with exponential backoff

   ### Security

   - Fixed proxy server vulnerabilities (rate limiting, auth validation)
   - Added structured logging
   - Added security scanning (pip-audit, bandit)

   ### Changed

   - Migrated from setup.py to Poetry
   - Improved error handling
   - Added dependency lockfile

   ### Fixed

   - Fixed version inconsistency (0.1.0 vs 0.4.1)
   - Fixed broad exception catching
   - Fixed insecure proxy server
   ```

**Files:**

- `CHANGELOG.md` (new)

**Time:** 1 hour

**5.3.2 Create Release Checklist**

1. [ ] Run all tests: `pytest --cov=uniinfer`
2. [ ] Ensure 80%+ coverage
3. [ ] Run linting: `black . && isort . && mypy uniinfer/ && bandit -r uniinfer/`
4. [ ] Run security scan: `pip-audit`
5. [ ] Update version to 1.0.0
6. [ ] Update CHANGELOG
7. [ ] Create git tag: `git tag v1.0.0`
8. [ ] Build package: `poetry build`
9. [ ] Publish to PyPI: `poetry publish`
10. [ ] Create GitHub release

**Time:** 2 hours

---

## Prioritized Task List

### Critical (Must Do - Weeks 1-3)

- [ ] Fix version inconsistency
- [ ] Secure proxy server (rate limiting, auth, CORS, logging)
- [ ] Add input validation (Pydantic)
- [ ] Add async support (refactor to async/await)
- [ ] Implement comprehensive test suite (80%+ coverage)
- [ ] Set up CI/CD pipeline
- [ ] Add dependency lockfile

### High (Should Do - Weeks 4-6)

- [ ] Add token tracking
- [ ] Add cost estimation
- [ ] Add context management
- [ ] Add caching
- [ ] Add retry logic
- [ ] Improve error handling
- [ ] Add structured logging

### Medium (Nice to Have - Weeks 7-8)

- [ ] Split large monolithic files
- [ ] Clean up examples directory
- [ ] Update documentation
- [ ] Create migration guide
- [ ] Add provider-specific features (prompt caching)

### Low (Polish - Week 8)

- [ ] Add type stubs for better IDE support
- [ ] Add performance benchmarks
- [ ] Add more integration tests
- [ ] Improve example code quality

---

## Estimated Effort

| Phase                        | Tasks        | Total Time                   |
| ---------------------------- | ------------ | ---------------------------- |
| Phase 1: Security            | 5 tasks      | 14 hours                     |
| Phase 2: Architecture        | 3 tasks      | 36 hours                     |
| Phase 3: Testing & CI        | 2 tasks      | 29 hours                     |
| Phase 4: Production Features | 4 tasks      | 25 hours                     |
| Phase 5: Documentation       | 3 tasks      | 17.5 hours                   |
| **TOTAL**                    | **17 tasks** | **121.5 hours (~3.5 weeks)** |

**Note:** This assumes 1 developer working full-time. With multiple developers, can complete faster.

---

## Success Criteria

### Before Calling Production-Ready:

- ✅ All security vulnerabilities resolved
- ✅ Input validation on all parameters
- ✅ Async support for all providers
- ✅ Test coverage 80%+
- ✅ CI/CD pipeline passing
- ✅ Token tracking implemented
- ✅ Cost estimation working
- ✅ Context management functional
- ✅ Caching operational
- ✅ Retry logic in place
- ✅ Documentation updated
- ✅ Version 1.0.0 released

### Phase Completion Gates:

- **Phase 1 Complete**: All security issues fixed, proxy server secured
- **Phase 2 Complete**: All providers async, input validation working
- **Phase 3 Complete**: Test suite passing with 80%+ coverage, CI/CD green
- **Phase 4 Complete**: Token/cost tracking, context, caching, retry all working
- **Phase 5 Complete**: Documentation complete, repository clean, v1.0.0 released

---

## Risk Mitigation

### Potential Risks:

1. **Scope creep** - Too many features for 8 weeks
   - Mitigation: Focus on MVP production features, defer advanced features

2. **Breaking changes** - Async migration may break existing code
   - Mitigation: Keep sync methods as wrappers, provide migration guide

3. **Test coverage difficult** - 80%+ coverage for 27 providers is hard
   - Mitigation: Prioritize top 5 providers, add basic tests for rest

4. **Provider-specific quirks** - Each provider has unique behavior
   - Mitigation: Document known issues, add compatibility layer

5. **Resource constraints** - Limited time/developers
   - Mitigation: Prioritize critical items, defer nice-to-haves

---

## Next Steps

1. **Week 1 (Starting Now):**
   - [ ] Fix version inconsistency (1 hour)
   - [ ] Secure proxy server (8 hours)
   - [ ] Add dependency lockfile (4 hours)
   - [ ] Add security scanning (2 hours)

2. **Week 2-3:**
   - [ ] Add input validation (10 hours)
   - [ ] Add async support (24 hours)
   - [ ] Improve error handling (9 hours)

3. **Week 4-5:**
   - [ ] Create test suite (23 hours)
   - [ ] Set up CI/CD (2.5 hours)

4. **Week 6-7:**
   - [ ] Add token/cost tracking (9 hours)
   - [ ] Add context management (7 hours)
   - [ ] Add caching (7 hours)
   - [ ] Add retry logic (5 hours)

5. **Week 8:**
   - [ ] Clean up repository (6.5 hours)
   - [ ] Update documentation (7 hours)
   - [ ] Prepare and execute v1.0.0 release (3 hours)

---

## Conclusion

This 8-week plan addresses all critical issues found in the code review and transforms UniInfer from **prototype-only** to **production-ready**.

**Key Achievements After Completion:**

- ✅ Security vulnerabilities resolved
- ✅ Production-grade architecture (async, validation, caching)
- ✅ Comprehensive testing (80%+ coverage)
- ✅ Automated quality checks (CI/CD)
- ✅ Production features (tokens, costs, context)
- ✅ Professional documentation
- ✅ v1.0.0 release

**Recommendation:** Execute this plan sequentially, starting with Phase 1 immediately, to make UniInfer production-ready by end of 8-week cycle.
