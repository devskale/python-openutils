# Feature Comparison

## Core Feature Matrix

| Feature | UniInfer (Python) | pi-ai (TypeScript) | Winner |
|---------|-------------------|---------------------|--------|
| **Language** | Python 3.7+ | TypeScript (Node.js 20+) | Context |
| **Chat Completions** | ✅ | ✅ | Tie |
| **Streaming** | ✅ Iterator-based | ✅ Async generator | pi-ai |
| **Non-Streaming** | ✅ | ✅ | Tie |
| **Tool Calling** | ⚠️ Basic support | ✅ Advanced with validation | pi-ai |
| **Embeddings** | ✅ Full support | ❌ Not supported | uniinfer |
| **TTS (Text-to-Speech)** | ✅ Supported | ❌ Not supported | uniinfer |
| **STT (Speech-to-Text)** | ✅ Supported | ❌ Not supported | uniinfer |
| **Image Input** | ⚠️ Limited | ✅ Full support | pi-ai |
| **Reasoning/Thinking** | ❌ No | ✅ Full support (5 levels) | pi-ai |
| **Token Tracking** | ❌ None | ✅ Detailed (input/output/cache) | pi-ai |
| **Cost Tracking** | ❌ None | ✅ Per-request cost calculation | pi-ai |
| **Context Management** | ❌ No | ✅ Serializable context | pi-ai |
| **Cross-Provider Handoffs** | ❌ Basic fallback | ✅ Advanced with transformation | pi-ai |
| **Fallback Strategies** | ✅ 2 strategies | ✅ Not exposed directly | uniinfer |
| **Session-Based Caching** | ❌ None | ✅ Supported | pi-ai |
| **OAuth Support** | ❌ None | ✅ 5+ providers | pi-ai |
| **Custom Providers** | ✅ Easy registration | ✅ Type-safe registration | pi-ai |
| **API Proxy Server** | ✅ OpenAI-compatible FastAPI | ❌ None | uniinfer |
| **CLI Tool** | ✅ Comprehensive | ✅ Simple | Tie |
| **Browser Support** | ❌ No | ✅ Yes (with CSP handling) | pi-ai |
| **Rate Limit Handling** | ⚠️ Basic detection | ⚠️ Basic detection | Tie |
| **Retry Logic** | ⚠️ Fallback only | ⚠️ Not visible | Tie |
| **Circuit Breaker** | ❌ None | ❌ None | Tie |
| **Connection Pooling** | ❌ None | ⚠️ Unclear | Tie |
| **Abort/Cancel Requests** | ❌ None | ✅ AbortSignal support | pi-ai |
| **Error Recovery** | ⚠️ Basic | ⚠️ Event-based | pi-ai |
| **Logging** | ⚠️ Basic | ⚠️ Basic | Tie |

---

## Provider Support

### UniInfer: 27 Providers

**Chat Providers (24):**
1. OpenAI
2. Anthropic
3. Mistral
4. Ollama
5. OpenRouter
6. Arli AI
7. InternLM
8. StepFun
9. Sambanova
10. Upstage
11. NGC
12. Cloudflare
13. Chutes
14. Pollinations
15. Bigmodel (GLM-4)
16. Tu AI
17. HuggingFace (optional)
18. Cohere (optional)
19. Moonshot (optional)
20. Groq (optional)
21. AI21 (optional)
22. Gemini (optional)
23. Mistral (separate)
24. And more...

**Embedding Providers:**
- Ollama Embedding
- Tu AI Embedding

**TTS/STT Providers:**
- Tu AI TTS
- Tu AI STT

### pi-ai: 15 Providers (via 9 APIs)

**Providers:**
1. OpenAI (Responses API + Codex)
2. Anthropic (Messages API)
3. Google (Generative AI + Gemini CLI + Vertex)
4. Azure OpenAI (Responses)
5. xAI (Grok)
6. Groq
7. Cerebras
8. OpenRouter
9. Vercel AI Gateway
10. zAI
11. Mistral
12. MiniMax
13. HuggingFace
14. Amazon Bedrock
15. GitHub Copilot
16. Kimi For Coding (Moonshot)
17. Antigravity (free providers)
18. OpenCode

**APIs:**
- `openai-completions` - Used by Mistral, xAI, Groq, Cerebras, OpenRouter, Vercel, zAI
- `openai-responses` - OpenAI Responses API
- `azure-openai-responses` - Azure OpenAI
- `openai-codex-responses` - OpenAI Codex
- `anthropic-messages` - Anthropic
- `google-generative-ai` - Google
- `google-gemini-cli` - Google Cloud Code Assist
- `google-vertex` - Google Vertex AI
- `bedrock-converse-stream` - Amazon Bedrock

---

## Advanced Feature Deep Dive

### Tool Calling

**UniInfer:**
```python
# Basic tool support via ChatMessage
message = ChatMessage(
    role="assistant",
    content=None,
    tool_calls=[{"id": "...", "function": {...}}]
)
```
- ⚠️ No tool validation
- ⚠️ No tool result handling
- ⚠️ No partial streaming of tool calls
- ⚠️ No tool schemas

**pi-ai:**
```typescript
// Type-safe tools with TypeBox schemas
const tools: Tool[] = [{
  name: 'get_weather',
  description: 'Get current weather',
  parameters: Type.Object({
    location: Type.String(),
    units: StringEnum(['celsius', 'fahrenheit'])
  })
}];

// Streaming tool calls with partial JSON
for await (const event of stream(model, context)) {
  if (event.type === 'toolcall_delta') {
    const partialArgs = event.partial.content[event.contentIndex].arguments;
    // Progressive UI updates
  }
}

// Automatic validation with AJV
const validatedArgs = validateToolCall(tools, toolCall);
```
- ✅ Full tool validation with TypeBox
- ✅ Streaming tool calls with partial JSON
- ✅ Error messages from validation returned to LLM
- ✅ Tool results with images support
- ✅ Browser extension CSP handling

### Embeddings

**UniInfer:**
```python
from uniinfer import EmbeddingProviderFactory, EmbeddingRequest

provider = EmbeddingProviderFactory.get_provider("ollama")
request = EmbeddingRequest(
    input=["Hello world", "How are you?"],
    model="nomic-embed-text:latest"
)
response = provider.embed(request)

for i, embedding in enumerate(response.data):
    print(f"{len(embedding['embedding'])} dimensions")
```
- ✅ Multiple embedding providers
- ✅ Batch embedding support
- ✅ OpenAI-compatible format
- ✅ Easy to use

**pi-ai:**
- ❌ No embedding support
- Could add via custom provider but not built-in

### TTS/STT

**UniInfer:**
```python
# Text-to-Speech
provider = TTSProviderFactory.get_provider("tu")
request = TTSRequest(
    text="Hello world",
    model="kokoro"
)
response = provider.synthesize(request)
# Returns audio data

# Speech-to-Text
provider = STTProviderFactory.get_provider("tu")
request = STTRequest(
    audio_file="speech.mp3",
    model="whisper-large"
)
response = provider.transcribe(request)
# Returns transcribed text
```
- ✅ Integrated TTS/STT
- ✅ Multiple models supported
- ✅ Audio file handling

**pi-ai:**
- ❌ No TTS/STT support
- Not in scope of library

### Reasoning/Thinking

**UniInfer:**
- ❌ No reasoning support
- No way to access model thinking process

**pi-ai:**
```typescript
// Unified reasoning interface
const response = await completeSimple(model, context, {
  reasoning: 'medium'  // 'minimal' | 'low' | 'medium' | 'high' | 'xhigh'
});

// Access thinking blocks
for (const block of response.content) {
  if (block.type === 'thinking') {
    console.log('Thinking:', block.thinking);
  }
}

// Streaming thinking
for await (const event of streamSimple(model, context, { reasoning: 'high' })) {
  if (event.type === 'thinking_delta') {
    process.stdout.write(event.delta);  // Stream thinking
  }
}
```
- ✅ 5 reasoning levels
- ✅ Provider-specific options (OpenAI `reasoning_effort`, Anthropic `thinkingEnabled`)
- ✅ Streaming thinking content
- ✅ Access to reasoning in final response
- ✅ Cross-provider thinking transformation (to `<thinking>` tags)

### Token & Cost Tracking

**UniInfer:**
```python
# Only available in response usage
response.usage
# {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}

# No cost calculation
# No pre-request estimation
```
- ⚠️ Basic usage only
- ❌ No cost tracking
- ❌ No pre-request estimation
- ❌ No budgeting

**pi-ai:**
```typescript
// Detailed token tracking
const message: AssistantMessage = {
  usage: {
    input: 1000,
    output: 500,
    cacheRead: 200,
    cacheWrite: 100,
    totalTokens: 1800,
    cost: {
      input: 0.002,      // $/million tokens
      output: 0.01,
      cacheRead: 0.0003,
      cacheWrite: 0.00375,
      total: 0.01605
    }
  }
};

// Cost available per model in registry
const model = getModel('openai', 'gpt-4o-mini');
console.log(model.cost);
// { input: 0.15, output: 0.60, cacheRead: 0.03, cacheWrite: 0.375 }
```
- ✅ Complete token breakdown
- ✅ Cost calculation per request
- ✅ Cache read/write costs
- ✅ Pre-request cost estimation
- ✅ Budget-aware decisions

### Context Management

**UniInfer:**
```python
# No built-in context management
# Users must manage messages manually
messages = [
    ChatMessage(role="user", content="Hello"),
    ChatMessage(role="assistant", content="Hi there!")
]
# No serialization, no persistence
```
- ❌ No context serialization
- ❌ No session management
- ❌ No context overflow handling

**pi-ai:**
```typescript
// Serializable context
const context: Context = {
  systemPrompt: 'You are helpful.',
  messages: [
    { role: 'user', content: 'Hello', timestamp: Date.now() },
    { role: 'assistant', content: [...], timestamp: Date.now() }
  ],
  tools: [...]
};

// Serialize to JSON
const serialized = JSON.stringify(context);
// Can save to database, localStorage, file

// Deserialize and continue
const restored: Context = JSON.parse(serialized);

// Cross-provider handoff
const claude = getModel('anthropic', 'claude-sonnet-4');
await complete(claude, context);

const gpt = getModel('openai', 'gpt-4o');
// Automatically transforms thinking blocks
await complete(gpt, context);  // Works seamlessly
```
- ✅ Full JSON serialization
- ✅ Timestamps on all messages
- ✅ Cross-provider transformation
- ✅ Context overflow detection
- ✅ Easy persistence

### Fallback Strategies

**UniInfer:**
```python
# FallbackStrategy with latency tracking
strategy = FallbackStrategy(
    providers=["openai", "anthropic", "ollama"],
    max_retries=2
)

response, provider_name = strategy.complete(request)

# Get statistics
stats = strategy.get_stats()
# {
#   'openai': {'avg_latency': 0.5, 'error_count': 0},
#   'anthropic': {'avg_latency': 0.8, 'error_count': 1},
#   ...
# }

# CostBasedStrategy (placeholder)
cost_strategy = CostBasedStrategy({
    'openai': 0.002,
    'anthropic': 0.003,
    'ollama': 0.0001
})
```
- ✅ Two strategies implemented
- ✅ Latency tracking
- ✅ Error counting
- ⚠️ CostBasedStrategy is placeholder

**pi-ai:**
- ✅ Fallback available via custom code
- ⚠️ Not exposed as first-class API
- ⚠️ No built-in strategies
- Can implement manually with `stream()`

---

## Unique Features

### UniInfer Exclusive
1. **Embeddings API** - Multiple embedding providers
2. **TTS/STT** - Integrated audio capabilities
3. **OpenAI-Compatible Proxy Server** - FastAPI server
4. **Fallback Strategies** - Built-in latency/error tracking
5. **Simple CLI** - Easy command-line usage

### pi-ai Exclusive
1. **Reasoning/Thinking** - Access to model thought process
2. **Token & Cost Tracking** - Detailed metrics per request
3. **Context Serialization** - Full conversation persistence
4. **Cross-Provider Handoffs** - Seamless provider switching
5. **OAuth Support** - 5+ OAuth providers
6. **Browser Support** - Works in browser extensions
7. **Tool Validation** - TypeBox + AJV validation
8. **Partial JSON Streaming** - Progressive tool call parsing
9. **Session-Based Caching** - Provider prompt caching
10. **Abort Signal** - Request cancellation

---

## Feature Score

| Category | UniInfer | pi-ai | Weight | Weighted Score |
|----------|-----------|-------|--------|---------------|
| Core LLM Features | 8/10 | 10/10 | 30% | 2.4 vs 3.0 |
| Streaming | 7/10 | 10/10 | 15% | 1.05 vs 1.5 |
| Tool Calling | 5/10 | 10/10 | 15% | 0.75 vs 1.5 |
| Embeddings | 10/10 | 0/10 | 10% | 1.0 vs 0 |
| Audio (TTS/STT) | 10/10 | 0/10 | 5% | 0.5 vs 0 |
| Context Management | 2/10 | 10/10 | 10% | 0.2 vs 1.0 |
| Token/Cost Tracking | 0/10 | 10/10 | 10% | 0 vs 1.0 |
| Provider Support | 9/10 | 8/10 | 5% | 0.45 vs 0.4 |
| **TOTAL** | **6.4/10** | **9.4/10** | 100% | |

---

## Use Case Recommendations

### Choose UniInfer when:
- ✅ Need embedding generation
- ✅ Need TTS/STT capabilities
- ✅ Want an OpenAI-compatible proxy server
- ✅ Prefer Python ecosystem
- ✅ Building simple applications
- ✅ Don't need advanced features (context, tokens, tools)
- ✅ Want maximum provider coverage (27 providers)

### Choose pi-ai when:
- ✅ Building agentic applications with tools
- ✅ Need advanced tool calling with validation
- ✅ Require context serialization and persistence
- ✅ Need cross-provider handoffs
- ✅ Building TypeScript/Node.js applications
- ✅ Need token and cost tracking
- ✅ Want reasoning/thinking capabilities
- ✅ Require browser support
- ✅ Need OAuth authentication

---

## Overall Assessment

**pi-ai** is feature-superior for modern LLM applications with tools, context management, token tracking, and reasoning support.

**UniInfer** has unique features (embeddings, TTS/STT, proxy server) but lacks critical production features (async, caching, token tracking, context).

**Winner: pi-ai** (9.4 vs 6.4) - More comprehensive feature set for agentic AI applications, though uniinfer wins on embeddings and audio.

---

## Feature Gap Analysis

### Missing from UniInfer (Critical)
1. Async support - Major performance limitation
2. Token/cost tracking - Essential for production
3. Context management - Needed for stateful apps
4. Tool validation - Important for agentic workflows
5. Session caching - Important for performance
6. Reasoning support - Growing requirement

### Missing from pi-ai (Nice-to-Have)
1. Embeddings - Could add via custom provider
2. TTS/STT - Out of scope for LLM library
3. Proxy server - Could build separately
4. More providers - Already has most important ones

### Both Missing
1. Retry logic with exponential backoff
2. Circuit breaker pattern
3. Connection pooling (unclear in pi-ai)
4. Rate limiting per user/key
