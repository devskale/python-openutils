# Security & Best Practices Review

## UniInfer (Python)

### API Key Management

**Current Implementation:**
```python
# Uses credgoo for secure key management
from uniinfer import ProviderFactory

# API key retrieved automatically via credgoo
provider = ProviderFactory.get_provider("openai")
```

**Strengths:**
- ✅ **Credgoo integration** - Encrypted key storage
- ✅ **No hardcoded keys** - Keys not in source
- ✅ **Environment variable support** - Can override with `OPENAI_API_KEY`, etc.
- ✅ **Provider-specific keys** - Separate keys per provider

**Weaknesses:**
- ⚠️ **Credgoo dependency** - External tool required
- ⚠️ **No key rotation** visible - No automatic refresh
- ⚠️ **No OAuth support** - Only static API keys
- ⚠️ **No key validation** - Keys not validated before use
- ⚠️ **No key scope** - No restrictions on key usage

**API Key Security Score: 6/10**

---

### Input Validation

**Current Implementation:**
```python
# Minimal validation in core classes
class ChatCompletionRequest:
    def __init__(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: float = 1.0,  # No range validation
        max_tokens: Optional[int] = None,  # No validation
        ...
    ):
        self.messages = messages
        self.temperature = temperature  # Could be any value
```

**Issues:**
1. **No parameter validation** - `temperature` could be -100 or 1000
2. **No model validation** - Invalid model names accepted
3. **No message validation** - Malformed messages not checked
4. **No length limits** - No input size validation
5. **No tool schema validation** - Tool definitions not validated

**Input Validation Score: 2/10** (Critical weakness)

---

### Error Handling

**Current Implementation:**
```python
# Broad exception catching (found in multiple places)
try:
    response = requests.post(url, headers=headers, json=data)
except Exception as e:  # Too broad
    status_code = getattr(e.response, 'status_code', None)
    response_body = getattr(e.response, 'text', None)
    raise map_provider_error("Provider", e, ...)
```

**Issues:**
1. **Catches all exceptions** - Even KeyboardInterrupt, SystemExit
2. **No sensitive data redaction** - API keys in error messages possible
3. **No logging** - Print statements used instead
4. **No error categorization** - User must handle all ProviderErrors
5. **Silent failures** possible - Some errors may be swallowed

**Security Risks:**
- API keys could leak in stack traces
- Debug info exposed in error messages
- No audit trail of errors

**Error Handling Score: 3/10**

---

### Dependency Security

**Dependencies Analysis:**
```
Core dependencies:
- requests >= 2.25.0
- openai (no version specified)
- fastapi >= 0.100.0
- pydantic (no version specified)
- python-dotenv >= 1.0.0

Remote requirements:
- https://skale.dev/credgoo (fetched dynamically)
```

**Issues:**
1. **No lockfile** - No pinned versions, vulnerable to supply chain attacks
2. **Unpinned dependencies** - "openai", "pydantic" without versions
3. **Remote requirements** - Fetching from HTTP URL (MITM risk)
4. **No security scanning** - No pip-audit, safety checks in CI/CD
5. **No known vulnerabilities check** visible

**Dependencies Security Score: 3/10**

---

### Network Security

**Current Implementation:**
```python
# Basic HTTP requests with requests library
response = requests.post(
    base_url + "/chat/completions",
    headers=headers,
    json=data,
    timeout=60  # Default timeout
)
```

**Issues:**
1. **No certificate pinning** - Trusts all certificates
2. **No proxy support** - Can't configure HTTP proxy
3. **Timeout is basic** - No connection timeout, only read timeout
4. **No request signing** - For AWS-like providers
5. **No retry with backoff** - Immediate failures
6. **No user agent** - Not customizable

**Network Security Score: 4/10**

---

### Code Security

**Security Issues Found:**

1. **No SQL injection prevention** - Not applicable (no database)
2. **No XSS prevention** - Not applicable (no web output)
3. **No command injection checks** - CLI tool doesn't validate inputs
4. **No path traversal prevention** - Not applicable (no file I/O)
5. ** eval/exec not used** - Good
6. **No hardcoded secrets** - Good

**Code Security Score: 6/10** (Minimal attack surface)

---

### Proxy Server Security

**FastAPI Proxy Implementation:**
```python
# uniioai_proxy.py (47KB)
# OpenAI-compatible proxy server
app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    auth = request.headers.get("Authorization")
    # Basic auth handling
```

**Issues:**
1. **No rate limiting** - Unlimited requests possible
2. **No request validation** - Any request accepted
3. **No IP blocking** - No DDoS protection
4. **No request size limits** - Could be exploited
5. **No CORS configuration** visible
6. **Authentication is basic** - Bearer token only, no JWT
7. **No audit logging** - No request tracking

**Proxy Security Score: 2/10** (Not production-ready)

---

**Overall Security Score: (6 + 2 + 3 + 3 + 4 + 6 + 2) / 7 = 3.7/10**

---

## pi-ai (TypeScript)

### API Key Management

**Current Implementation:**
```typescript
// Environment-based API key detection
export function getEnvApiKey(provider: string): string | undefined {
  const envVar = {
    'openai': 'OPENAI_API_KEY',
    'anthropic': 'ANTHROPIC_API_KEY',
    'google': 'GEMINI_API_KEY',
    // ... etc.
  }[provider];

  return process.env[envVar];
}

// Support for OAuth
const credentials = await loginAnthropic({...});
const { apiKey } = await getOAuthApiKey('anthropic', auth);
```

**Strengths:**
- ✅ **Environment variables** - Keys not in source
- ✅ **Multiple auth methods** - Static keys, OAuth, ADC
- ✅ **5+ OAuth providers** - Anthropic, OpenAI Codex, GitHub Copilot, Google Gemini CLI, Antigravity
- ✅ **Token refresh** - Automatic OAuth token refresh
- ✅ **Browser-safe** - Keys passed explicitly in browser
- ✅ **No hardcoded keys** - Good practice

**Weaknesses:**
- ⚠️ **No key rotation** visible
- ⚠️ **No key validation** before first use
- ⚠️ **No key scope** or restrictions
- ⚠️ **No credential encryption** - Plain environment variables

**API Key Security Score: 8/10**

---

### Input Validation

**Current Implementation:**
```typescript
// Tool validation with AJV and TypeBox
export function validateToolCall(tools: Tool[], toolCall: ToolCall): any {
  const tool = tools.find((t) => t.name === toolCall.name);
  if (!tool) {
    throw new Error(`Tool "${toolCall.name}" not found`);
  }
  return validateToolArguments(tool, toolCall);
}

// AJV validation
const validate = ajv.compile(tool.parameters);
if (validate(args)) {
  return args;
}
// Format validation errors nicely
const errors = validate.errors?.map(...) || "Unknown validation error";
throw new Error(`Validation failed for tool "${tool.name}":\n${errors}`);
```

**Strengths:**
- ✅ **Tool argument validation** - AJV with TypeBox schemas
- ✅ **Type coercion** - Automatic type conversion
- ✅ **Detailed error messages** - Clear validation feedback
- ✅ **Schema-based** - Prevents invalid tool calls
- ✅ **Browser CSP handling** - Graceful degradation when AJV unavailable

**Weaknesses:**
- ⚠️ **No model name validation** - Invalid model IDs accepted
- ⚠️ **No parameter range validation** - temperature could be invalid
- ⚠️ **No message length validation** - Could overflow context
- ⚠️ **No tool schema validation** on registration

**Input Validation Score: 7/10**

---

### Error Handling

**Current Implementation:**
```typescript
// Event-based error handling
interface ErrorEvent {
  type: 'error';
  reason: 'error' | 'aborted';
  error: AssistantMessage; // Contains partial content
}

// Error preservation
const message = await stream.result();
if (message.stopReason === 'error') {
  console.error('Request failed:', message.errorMessage);
  // message.content has partial content
  // message.usage has partial tokens
}

// Abort support
const controller = new AbortController();
setTimeout(() => controller.abort(), 2000);
await stream(model, context, { signal: controller.signal });
```

**Strengths:**
- ✅ **Error events in streaming** - Real-time error detection
- ✅ **Partial content preserved** - Don't lose data on error
- ✅ **Abort signal support** - Clean cancellation
- ✅ **Error reasons** - Clear categorization
- ✅ **Error messages preserved** - Debugging info available
- ✅ **No sensitive data in errors** - Keys not in error messages

**Weaknesses:**
- ⚠️ **No retry logic** - Immediate failure on errors
- ⚠️ **No circuit breaker** - No protection from failing providers
- ⚠️ **No error aggregation** - User must handle individually

**Error Handling Score: 7/10**

---

### Dependency Security

**Dependencies Analysis:**
```json
{
  "dependencies": {
    "@anthropic-ai/sdk": "0.71.2",
    "@aws-sdk/client-bedrock-runtime": "^3.966.0",
    "@google/genai": "1.34.0",
    "@mistralai/mistralai": "1.10.0",
    "@sinclair/typebox": "^0.34.41",
    "ajv": "^8.17.1",
    "ajv-formats": "^3.0.1",
    "openai": "6.10.0",
    // ... 8 more dependencies
  }
}
```

**Security Audit Results (npm audit):**
```
HIGH severity vulnerabilities:
- @aws-sdk/client-bedrock-runtime >= 3.894.0
- @aws-sdk/core >= 3.894.0
- @aws-sdk/client-sso >= 3.894.0
- @aws-sdk/credential-provider-* (multiple)
- @aws-sdk/middleware-* (multiple)
- @aws-sdk/token-providers >= 3.894.0

Root cause: @aws-sdk/xml-builder vulnerability
Fix available: Update to version 3.893.0 (major downgrade)
```

**Issues:**
1. **High-severity vulnerabilities** - Multiple AWS packages vulnerable
2. **No lockfile committed** - package-lock.json not in repo
3. **No security scanning in CI** - No visible npm audit automation
4. **Heavy dependencies** - 18 prod packages increase attack surface
5. **Depends on multiple SDKs** - Each SDK could have vulnerabilities

**Dependencies Security Score: 4/10** (Vulnerabilities present)

---

### Network Security

**Current Implementation:**
```typescript
// Using SDKs for HTTP
import OpenAI from "openai";
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Custom headers support
const model: Model = {
  // ...
  headers?: Record<string, string>;
};

// Environment-based proxy support
// (SDKs respect HTTP_PROXY, HTTPS_PROXY)
```

**Strengths:**
- ✅ **SDK handles HTTPS** - Encrypted connections
- ✅ **Environment proxy support** - SDKs respect HTTP_PROXY
- ✅ **Custom headers** - Can add authentication headers
- ✅ **Timeout support** - Via SDK defaults
- ✅ **Abort signal** - Request cancellation

**Weaknesses:**
- ⚠️ **No certificate pinning** - Trusts all certificates
- ⚠️ **Proxy support indirect** - Depends on SDK behavior
- ⚠️ **No request signing** - For AWS-like providers (SDK handles)
- ⚠️ **No explicit retry config** - Depends on SDK defaults

**Network Security Score: 6/10**

---

### Code Security

**Security Analysis:**

1. **No eval or Function** - ✅ Good (no dynamic code execution)
2. **No dangerous patterns** - ✅ Good
3. **TypeBox + AJV validation** - ✅ Prevents injection
4. **Browser CSP handling** - ✅ Graceful when eval blocked
5. **No hardcoded secrets** - ✅ Good
6. **No SQL injection vectors** - ✅ Good (no database)

**Potential Issues:**
- ⚠️ **`any` types in some places** - Could bypass type safety
- ⚠️ **Dynamic tool calls** - Executes user-defined functions (intentional)
- ⚠️ **No sanitization** of user messages before sending to LLM (intentional)

**Code Security Score: 7/10**

---

### Browser Security

**Browser Extension Support:**
```typescript
// CSP-aware validation
const isBrowserExtension = typeof globalThis !== "undefined" &&
  globalThis.chrome?.runtime?.id !== undefined;

// Skip AJV in browser (CSP restriction)
if (!ajv || isBrowserExtension) {
  return toolCall.arguments; // Trust LLM output
}

// Security warning in README
// "Exposing API keys in frontend code is dangerous"
```

**Strengths:**
- ✅ **CSP handling** - Works with Manifest V3 restrictions
- ✅ **Security warnings** - Documentation warns about frontend key exposure
- ✅ **Explicit API key in browser** - No auto-detection, must pass explicitly

**Weaknesses:**
- ⚠️ **Keys still possible** in browser code (user responsibility)
- ⚠️ **No key validation** - Any key accepted

**Browser Security Score: 6/10**

---

**Overall Security Score: (8 + 7 + 7 + 4 + 6 + 7 + 6) / 7 = 6.4/10**

---

## Security Comparison

| Security Aspect | UniInfer | pi-ai | Winner |
|----------------|-----------|--------|--------|
| **API Key Management** | 6/10 | 8/10 | pi-ai |
| **Input Validation** | 2/10 | 7/10 | pi-ai |
| **Error Handling** | 3/10 | 7/10 | pi-ai |
| **Dependencies** | 3/10 | 4/10 | pi-ai (both weak) |
| **Network Security** | 4/10 | 6/10 | pi-ai |
| **Code Security** | 6/10 | 7/10 | pi-ai |
| **Browser Security** | N/A | 6/10 | pi-ai |
| **Proxy Security** | 2/10 | N/A | Tie |
| **Overall** | 3.7/10 | 6.4/10 | **pi-ai** |

---

## Critical Security Findings

### UniInfer Critical Issues
1. **No input validation** - Parameters not validated (CRITICAL)
2. **No testing** - Security bugs unlikely caught
3. **No lockfile** - Supply chain attacks possible
4. **Remote requirements** - Fetching from HTTP URL (risk)
5. **Proxy server insecure** - No rate limiting, no auth validation
6. **Broad exception catching** - May hide security issues

### pi-ai Critical Issues
1. **High-severity vulnerabilities** - AWS SDK needs update (HIGH)
2. **18 production dependencies** - Larger attack surface
3. **No lockfile** - Supply chain risk
4. **No retry logic** - Could cause security-relevant failures

---

## Recommendations

### For UniInfer
1. **URGENT**: Add input validation for all parameters
2. **URGENT**: Implement comprehensive test suite
3. **HIGH**: Fix proxy server security (rate limiting, auth)
4. **HIGH**: Add requirements.lock or use poetry
5. **HIGH**: Stop using remote requirements (pip install credgoo)
6. **MEDIUM**: Add pip-audit to CI/CD
7. **MEDIUM**: Use specific exception types
8. **MEDIUM**: Add audit logging for proxy server

### For pi-ai
1. **URGENT**: Update AWS SDK packages to fix vulnerabilities
2. **HIGH**: Add npm audit to CI/CD
3. **HIGH**: Commit package-lock.json
4. **MEDIUM**: Reduce dependency count where possible
5. **MEDIUM**: Implement retry logic with exponential backoff
6. **MEDIUM**: Add circuit breaker pattern
7. **LOW**: Add model name validation
8. **LOW**: Add parameter range validation

---

## Overall Assessment

**pi-ai** has better security practices with input validation, OAuth support, and browser awareness, but has critical vulnerabilities to address.

**UniInfer** has severe security weaknesses including no input validation, insecure proxy server, and supply chain risks, making it unsuitable for production.

**Winner: pi-ai** (6.4 vs 3.7) - Better security posture despite vulnerabilities. UniInfer needs fundamental security improvements before production use.
