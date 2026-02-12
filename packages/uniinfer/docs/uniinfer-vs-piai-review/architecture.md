# Architecture Review

## UniInfer (Python)

### Code Organization

**Directory Structure:**
```
uniinfer/
├── __init__.py          # Package exports and registration
├── core.py              # Core classes (ChatMessage, ChatCompletionRequest, etc.)
├── factory.py           # Provider factory pattern
├── embedding_factory.py  # Embedding provider factory
├── errors.py            # Error handling
├── strategies.py        # Fallback and cost-based strategies
├── json_utils.py        # JSON utilities
├── tts_stt_endpoints.py # TTS/STT endpoints
├── uniinfer_cli.py      # CLI tool
├── uniioai_proxy.py    # FastAPI proxy server
├── providers/           # 27 provider implementations
│   ├── openai.py
│   ├── anthropic.py
│   ├── mistral.py
│   └── ...
├── examples/           # Example code
└── tests/             # Tests (limited)
```

### Design Patterns

**1. Factory Pattern**
```python
class ProviderFactory:
    _providers = {}

    @classmethod
    def register_provider(cls, name, provider_class):
        cls._providers[name.lower()] = provider_class

    @classmethod
    def get_provider(cls, name, api_key=None):
        provider_class = cls._providers.get(name.lower())
        if not provider_class:
            raise ValueError(f"Provider {name} not found")
        return provider_class(api_key)
```
- ✅ Clean provider registration
- ✅ Runtime provider lookup
- ⚠️ No dependency injection support

**2. Strategy Pattern**
```python
class FallbackStrategy:
    def __init__(self, provider_names, max_retries=1):
        self.provider_names = provider_names
        self.max_retries = max_retries
```
- ✅ Flexible fallback mechanism
- ✅ Latency and error tracking
- ⚠️ CostBasedStrategy is placeholder only
- ⚠️ No circuit breaker pattern

**3. Provider Base Class**
```python
class ChatProvider(ABC):
    def __init__(self, api_key=None):
        self.api_key = api_key

    @abstractmethod
    def complete(self, request, **kwargs):
        pass

    @abstractmethod
    def stream_complete(self, request, **kwargs):
        pass
```
- ✅ Abstract base class enforces interface
- ✅ Consistent API across providers
- ⚠️ No async support (0 async methods found)
- ⚠️ No connection pooling

### Architecture Strengths
- ✅ Clean separation of concerns (core, factory, providers, errors)
- ✅ Provider registration system is extensible
- ✅ Error handling is centralized
- ✅ Fallback strategies for resilience
- ✅ Proxy server for OpenAI compatibility

### Architecture Weaknesses
- ❌ **No async support** - All providers are synchronous
- ❌ **No connection pooling** - Each request creates new connection
- ❌ **No retry logic** - Basic fallback but no exponential backoff
- ❌ **No caching mechanism** - No prompt caching or response caching
- ❌ **No rate limiting** - No built-in rate limit management
- ❌ **No token counting** - No pre-request token estimation
- ❌ **No cost tracking** - No cost calculation or budgeting
- ❌ **No context management** - No session/context persistence
- ❌ **Tight coupling** - Providers directly use `requests` library

**Architecture Score: 5/10**

---

## pi-ai (TypeScript)

### Code Organization

**Directory Structure:**
```
src/
├── index.ts                   # Package exports
├── types.ts                   # All TypeScript types (279 lines)
├── models.ts                  # Model registry and utilities
├── models.generated.ts         # Auto-generated model catalog (317K lines)
├── api-registry.ts            # API registration system
├── env-api-keys.ts           # Environment API key management
├── stream.ts                 # Main streaming API
├── providers/                 # 14 provider implementations
│   ├── anthropic.ts
│   ├── openai-completions.ts
│   ├── openai-responses.ts
│   ├── google-gemini-cli.ts
│   ├── amazon-bedrock.ts
│   └── ...
├── utils/                    # Utility modules
│   ├── event-stream.ts        # Event streaming system
│   ├── json-parse.ts         # JSON parsing utilities
│   ├── oauth/               # OAuth implementation
│   ├── overflow.ts          # Context overflow handling
│   ├── validation.ts         # Tool validation (AJV)
│   └── ...
└── cli.ts                   # CLI tool
```

### Design Patterns

**1. API Registry Pattern**
```typescript
const apiRegistry = new Map<Api, ApiImpl>();

export function registerApiProvider<TApi extends Api>(
  api: TApi,
  impl: ApiImpl<TApi>
): void {
  apiRegistry.set(api, impl);
}
```
- ✅ Runtime API registration
- ✅ Type-safe provider selection
- ✅ Support for custom providers

**2. Event Streaming Pattern**
```typescript
class AssistantMessageEventStream {
  private controller = new AbortController();
  private queue: AssistantMessageEvent[] = [];

  async *[Symbol.asyncIterator](): AsyncIterator<AssistantMessageEvent> {
    while (true) {
      if (this.queue.length > 0) {
        yield this.queue.shift()!;
      } else if (this.done) {
        break;
      }
      await this.waitForEvent();
    }
  }
}
```
- ✅ Async generator pattern
- ✅ Event-driven architecture
- ✅ Abort signal support
- ✅ Proper event ordering

**3. Compatibility Layer Pattern**
```typescript
interface OpenAICompletionsCompat {
  supportsStore?: boolean;
  supportsDeveloperRole?: boolean;
  supportsReasoningEffort?: boolean;
  maxTokensField?: 'max_completion_tokens' | 'max_tokens';
  // ... 10+ compatibility flags
}
```
- ✅ Auto-detection from URLs
- ✅ Manual override capability
- ✅ Handles provider quirks gracefully

**4. Cross-Provider Handoff**
```typescript
function transformMessages(messages: Message[], targetApi: Api): Message[] {
  // Transform thinking blocks to <thinking> tags for cross-provider
  // Preserve tool calls and tool results
  // Handle image content
}
```
- ✅ Seamless provider switching
- ✅ Context preservation
- ✅ Compatibility transformation

### Architecture Strengths
- ✅ **Native async/await** - Fully async throughout (52 async methods)
- ✅ **Type safety** - Comprehensive TypeScript types
- ✅ **Event-driven streaming** - Proper async generators
- ✅ **Abort signal support** - Cancellable requests
- ✅ **Provider compatibility layer** - Handles API differences
- ✅ **Cross-provider handoffs** - Context transformation
- ✅ **Tool validation** - AJV with TypeBox schemas
- ✅ **Token tracking** - Per-request usage with cost calculation
- ✅ **Cost tracking** - Built-in cost estimation per model
- ✅ **Context overflow handling** - Graceful error messages
- ✅ **OAuth support** - Multiple OAuth providers
- ✅ **Browser support** - Works in browser extensions
- ✅ **Partial JSON parsing** - Streaming tool calls
- ✅ **Session-based caching** - Provider prompt caching
- ✅ **Environment-based auth** - Automatic API key detection

### Architecture Weaknesses
- ⚠️ **High complexity** - 279-line types file, complex compat layer
- ⚠️ **Tight coupling to providers** - Direct SDK dependencies
- ⚠️ **Large generated file** - 317K lines in models.generated.ts
- ⚠️ **No connection pooling visible** - Unclear if connections are reused
- ⚠️ **Security vulnerabilities** - AWS SDK dependencies need updates
- ⚠️ **No retry logic** - No exponential backoff visible
- ⚠️ **No circuit breaker** - No protection from failing providers

**Architecture Score: 9/10**

---

## Comparison

| Aspect | UniInfer | pi-ai | Winner |
|--------|----------|-------|--------|
| **Design Patterns** | Factory, Strategy | Registry, Event Stream, Compat | pi-ai |
| **Async Support** | ❌ None | ✅ Full async | pi-ai |
| **Type Safety** | ⚠️ Basic hints | ✅ Full TypeScript | pi-ai |
| **Event System** | ❌ None | ✅ Async generators | pi-ai |
| **Error Handling** | ✅ Centralized | ✅ Event-based | Tie |
| **Streaming** | ⚠️ Iterator-based | ✅ Async generator | pi-ai |
| **Connection Pooling** | ❌ None | ⚠️ Unclear | Tie |
| **Retry Logic** | ⚠️ Basic fallback | ⚠️ Not visible | Tie |
| **Caching** | ❌ None | ✅ Session-based | pi-ai |
| **Token Tracking** | ❌ None | ✅ Detailed | pi-ai |
| **Cost Tracking** | ❌ None | ✅ Full support | pi-ai |
| **Context Management** | ❌ None | ✅ Serializable | pi-ai |
| **Tool Validation** | ❌ None | ✅ AJV + TypeBox | pi-ai |
| **Cross-Provider** | ❌ Basic | ✅ Advanced | pi-ai |
| **Provider Quirks** | ⚠️ Ad-hoc | ✅ Compatibility layer | pi-ai |
| **Browser Support** | ❌ No | ✅ Yes | pi-ai |
| **Code Organization** | ✅ Good | ✅ Excellent | pi-ai |
| **Extensibility** | ✅ Easy to add providers | ✅ Type-safe registration | Tie |
| **Complexity** | ✅ Simple | ⚠️ High | uniinfer |

---

## Architectural Differences

### UniInfer: Simple but Limited
- **Pros**: Easy to understand, quick to add providers, minimal complexity
- **Cons**: Missing critical features (async, caching, token tracking), not production-ready
- **Best for**: Simple prototypes, learning, quick provider unification
- **Not for**: Production use, complex workflows, performance-critical apps

### pi-ai: Complex but Complete
- **Pros**: Full-featured, production-ready, type-safe, async-native
- **Cons**: Steep learning curve, high complexity, security vulnerabilities
- **Best for**: Production applications, agentic workflows, cross-provider scenarios
- **Not for**: Simple prototypes, quick scripts, beginners

---

## Key Architectural Decisions

### UniInfer's Choices
1. **Sync-only design** - Simplifies code but limits performance
2. **Requests library** - Simple but no connection pooling
3. **No state management** - Stateless by design
4. **Factory pattern** - Good for provider registration

### pi-ai's Choices
1. **Async-first** - Better performance and concurrency
2. **Event streaming** - Powerful but complex
3. **Compatibility layer** - Handles provider differences elegantly
4. **TypeBox + AJV** - Type-safe tool validation
5. **Generated models** - Auto-updated but large files

---

## Recommendations

### For UniInfer
1. Add async support with `async def` methods
2. Implement connection pooling (use `httpx` or `aiohttp`)
3. Add token counting and cost tracking
4. Implement context management and serialization
5. Add retry logic with exponential backoff
6. Create compatibility layer for provider quirks
7. Add session-based caching support

### For pi-ai
1. Simplify compatibility layer (reduce 10+ flags)
2. Fix AWS SDK security vulnerabilities
3. Add connection pooling documentation
4. Implement retry logic with exponential backoff
5. Add circuit breaker pattern for failing providers
6. Split large generated files into smaller modules
7. Consider simplifying event system for basic use cases

---

## Overall Assessment

**pi-ai** has a sophisticated, production-ready architecture with all critical features (async, streaming, caching, token tracking), but complexity is high.

**UniInfer** has a simple, clean architecture that's easy to understand, but lacks critical production features and is fundamentally synchronous.

**Winner: pi-ai** (9 vs 5) - Production-ready architecture despite complexity. UniInfer needs fundamental architectural changes (async, caching, token tracking) to be production-ready.
