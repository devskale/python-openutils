# UniInfer v1.0 Production Roadmap

This plan outlines the technical requirements and implementation steps to reach v1.0 (Production Grade). It addresses architectural, security, and feature gaps identified during the v0.4.7 review.

## Implementation Status Tracking

| Category | Requirement | Status |
| :--- | :--- | :--- |
| **Security** | Version Synchronization | ✅ Completed |
| | Proxy Middleware (Auth, Rate Limit) | ✅ Completed |
| | Dependency Locking (`requirements.lock`) | ✅ Completed |
| | Remote Requirement Removal | ⏳ Pending |
| **Architecture** | Async/Await Provider Core | 🔄 In Progress (5/27) |
| | Pydantic Input Validation | ⏳ Pending |
| | Exception Normalization | ⏳ Pending |
| **Quality** | CI/CD (GitHub Actions) | ⏳ Pending |
| | Test Coverage (Target 80%) | 🔄 In Progress (~15%) |
| **Features** | Token/Cost Tracking | ⏳ Pending |
| | Context Serialization | ⏳ Pending |
| | Response Caching | ⏳ Pending |

---

## Phase 1: Security & Supply Chain (Short Term)

### 1.1 Remove Remote Requirement Fetching
**Requirement:** Eliminate external HTTP fetches during package installation to prevent supply chain risks.
**Steps:**
1. Locate any `urllib` or `requests` calls in `pyproject.toml` or initialization scripts that fetch remote dependency lists.
2. Bundle all requirements directly in `pyproject.toml`.
3. Verify with `uv pip compile`.

### 1.2 Automated Security Scanning
**Requirement:** Integrate static analysis into the development workflow.
**Steps:**
1. Configure `bandit` in `pyproject.toml`:
   ```toml
   [tool.bandit]
   exclude_dirs = ["tests", "examples"]
   tests = ["B201", "B301"] # Specific security checks
   ```
2. Add `pip-audit` to the verification script.

---

## Phase 2: Core Architecture (Weeks 1-3)

### 2.1 Async/Await Migration Pattern
**Requirement:** Standardize all 27 providers to support non-blocking I/O.
**Actionable Pattern:**
1. Inherit from `AsyncProviderBase` (to be created in `http_client.py`).
2. Use `httpx.AsyncClient` for all network calls.
3. Implement `acomplete` and `astream_complete`.
4. **Target List:** Priority on Anthropic, OpenAI, and Google (Top 3 traffic).

### 2.2 Pydantic Core Validation
**Requirement:** Replace manual dictionary checks with strict type validation.
**Steps:**
1. Refactor `ChatMessage` and `ChatCompletionRequest` into Pydantic models in `core.py`.
2. Add `@field_validator` for `temperature` (0.0-2.0) and `top_p` (0.0-1.0).
3. Implement `ProviderConfig` model to validate environment variables/keys.

---

## Phase 3: Observability & Production Features (Weeks 4-6)

### 3.1 Token & Cost Instrumentation
**Requirement:** Provide usage metrics compatible with OpenAI's `usage` object.
**Steps:**
1. Integrate `tiktoken` for OpenAI/Mistral and `anthropic`'s tokenizer for Claude.
2. Update `ChatCompletionResponse` to include a required `usage` field.
3. Create a static `pricing.json` and a helper to calculate `estimated_cost` based on the usage object.

### 3.2 Context & Persistence
**Requirement:** Enable conversation state saving/loading.
**Steps:**
1. Implement a `ConversationBuffer` class that handles message windowing (sliding window vs. summary).
2. Add `.to_json()` and `.from_json()` methods to the `Context` class for easy database storage.

---

## Phase 4: Testing & Distribution (Weeks 7-8)

### 4.1 Test Coverage Uplift
**Requirement:** Achieve 80% line coverage.
**Actionable Goal:**
- Unit tests for all 27 providers (mocking network responses).
- Integration tests for the Proxy server using `TestClient`.
- **Command:** `pytest --cov=uniinfer --cov-report=xml`

### 4.2 CI/CD Pipeline
**Steps:**
1. Create `.github/workflows/verify.yml` to run:
   - `ruff check .`
   - `pytest`
   - `pip-audit`
2. Configure branch protection to require passing checks.

---

## Verification Checklist for v1.0
- [ ] `uv run pytest` passes with >80% coverage.
- [ ] No `Exception` is caught without logging or re-raising a specific `UniInferError`.
- [ ] All `complete()` calls have a corresponding `acomplete()` sibling.
- [ ] `examples/` contains only `.py` and `.md` files (no binary blobs).
- [ ] `requirements.lock` is up-to-date with `pyproject.toml`.

