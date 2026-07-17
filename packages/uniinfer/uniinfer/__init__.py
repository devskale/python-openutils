"""
UniInfer - Unified Inference API for LLM chat completions

A standardized interface for making chat completion requests across multiple
LLM inference providers.
"""

from .core import (
    ChatMessage, ChatCompletionRequest, ChatCompletionResponse, ChatProvider,
    EmbeddingRequest, EmbeddingResponse, EmbeddingProvider,
    TTSRequest, TTSResponse, TTSProvider,
    STTRequest, STTResponse, STTProvider,
    ModelInfo
)
from .factory import ProviderFactory
from .embedding_factory import EmbeddingProviderFactory
from .providers import (
    MistralProvider, AnthropicProvider, MiniMaxProvider, OpenAIProvider, OpenAITTSProvider,
    OllamaProvider, OllamaEmbeddingProvider, OpenRouterProvider, ArliAIProvider,
    InternLMProvider, StepFunProvider, SambanovaProvider,
    UpstageProvider, NGCProvider, CloudflareProvider, ChutesProvider, OpenCodeProvider,
    PollinationsProvider, ZAIProvider, ZAICodeProvider, TUProvider, TUStagingProvider, TuAIEmbeddingProvider,
    TuAITTSProvider, TuAISTTProvider
)
from .errors import (
    UniInferError, ProviderError, AuthenticationError,
    RateLimitError, TimeoutError, InvalidRequestError
)
from .strategies import FallbackStrategy, CostBasedStrategy

# Import optional providers conditionally
try:
    from .providers import HuggingFaceProvider
    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False

try:
    from .providers import CohereProvider
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False

try:
    from .providers import MoonshotProvider
    HAS_MOONSHOT = True
except ImportError:
    HAS_MOONSHOT = False

try:
    from .providers import GroqProvider
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from .providers import AI21Provider
    HAS_AI21 = True
except ImportError:
    HAS_AI21 = False

# Whether google-genai is *installed* (cheap check, no import). The GeminiProvider
# module is heavy (~50s to import on some boxes), so we register it lazily below
# and only import it when actually used.
import importlib.util as _importlib_util
HAS_GENAI = _importlib_util.find_spec("google.genai") is not None

# Register built-in providers
ProviderFactory.register_provider("mistral", MistralProvider)
ProviderFactory.register_provider("anthropic", AnthropicProvider)
ProviderFactory.register_provider("minimax", MiniMaxProvider)
ProviderFactory.register_provider("openai", OpenAIProvider)
ProviderFactory.register_provider("ollama", OllamaProvider)
ProviderFactory.register_provider("openrouter", OpenRouterProvider)
ProviderFactory.register_provider("arli", ArliAIProvider)
ProviderFactory.register_provider("internlm", InternLMProvider)
ProviderFactory.register_provider("stepfun", StepFunProvider)
ProviderFactory.register_provider("sambanova", SambanovaProvider)
ProviderFactory.register_provider("upstage", UpstageProvider)
ProviderFactory.register_provider("ngc", NGCProvider)
ProviderFactory.register_provider("cloudflare", CloudflareProvider)
ProviderFactory.register_provider("chutes", ChutesProvider)
ProviderFactory.register_provider("opencode", OpenCodeProvider)
ProviderFactory.register_provider("pollinations", PollinationsProvider)
ProviderFactory.register_provider("zai", ZAIProvider)
ProviderFactory.register_provider("zai-code", ZAICodeProvider)
ProviderFactory.register_provider("tu", TUProvider)
ProviderFactory.register_provider("tu-staging", TUStagingProvider)

# Register embedding providers
EmbeddingProviderFactory.register_provider("ollama", OllamaEmbeddingProvider)
EmbeddingProviderFactory.register_provider("tu", TuAIEmbeddingProvider)

# Register optional providers if available
if HAS_HUGGINGFACE:
    ProviderFactory.register_provider("huggingface", HuggingFaceProvider)

if HAS_COHERE:
    ProviderFactory.register_provider("cohere", CohereProvider)

if HAS_MOONSHOT:
    ProviderFactory.register_provider("moonshot", MoonshotProvider)

if HAS_GROQ:
    ProviderFactory.register_provider("groq", GroqProvider)

if HAS_AI21:
    ProviderFactory.register_provider("ai21", AI21Provider)

if HAS_GENAI:
    # Lazy: avoid importing the heavy google-genai SDK at startup. The module is
    # only imported on the first gemini request (or direct GeminiProvider access).
    ProviderFactory.register_lazy("gemini", "uniinfer.providers.gemini:GeminiProvider")

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("uniinfer")
except PackageNotFoundError:
    __version__ = "unknown"

# Export commonly used functions and classes
__all__ = [
    'ChatMessage',
    'ChatCompletionRequest',
    'ChatCompletionResponse',
    'ChatProvider',
    'EmbeddingRequest',
    'EmbeddingResponse',
    'EmbeddingProvider',
    'EmbeddingProviderFactory',
    'TTSRequest',
    'TTSResponse',
    'TTSProvider',
    'STTRequest',
    'STTResponse',
    'STTProvider',
    'ModelInfo',
    'MistralProvider',
    'AnthropicProvider',
    'MiniMaxProvider',
    'OpenAIProvider',
    'OpenAITTSProvider',
    'ChutesProvider',
    'OpenCodeProvider',
    'PollinationsProvider',
    'ZAIProvider',
    'ZAICodeProvider',
    'OllamaProvider',
    'OllamaEmbeddingProvider',
    'OpenRouterProvider',
    'ArliAIProvider',
    'InternLMProvider',
    'StepFunProvider',
    'SambanovaProvider',
    'UpstageProvider',
    'NGCProvider',
    'CloudflareProvider',
    'TUProvider',
    'TUStagingProvider',
    'TuAIEmbeddingProvider',
    'TuAITTSProvider',
    'TuAISTTProvider',
    'UniInferError',
    'ProviderError',
    'AuthenticationError',
    'RateLimitError',
    'TimeoutError',
    'InvalidRequestError',
    'FallbackStrategy',
    'CostBasedStrategy'
]

# Add optional providers to exports if available
if HAS_HUGGINGFACE:
    __all__.append('HuggingFaceProvider')

if HAS_COHERE:
    __all__.append('CohereProvider')

if HAS_MOONSHOT:
    __all__.append('MoonshotProvider')

if HAS_GROQ:
    __all__.append('GroqProvider')

if HAS_AI21:
    __all__.append('AI21Provider')

if HAS_GENAI:
    __all__.append('GeminiProvider')


def __getattr__(name):
    """Lazy attribute access (PEP 562).

    GeminiProvider is intentionally not imported at module load (google-genai is
    ~50s to import). It is resolved here on first access, preserving
n    ``from uniinfer import GeminiProvider`` for callers that need the class.
    """
    if name == "GeminiProvider":
        if HAS_GENAI:
            from uniinfer.providers.gemini import GeminiProvider as _G
            return _G
        raise ImportError("GeminiProvider requires the 'gemini' extra (google-genai)")
    raise AttributeError(f"module 'uniinfer' has no attribute {name!r}")
