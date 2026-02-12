"""
Provider implementations for different LLM services.
"""
from .mistral import MistralProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .openai_tts import OpenAITTSProvider
from .ollama import OllamaProvider
from .ollama_embedding import OllamaEmbeddingProvider
from .openrouter import OpenRouterProvider
from .arli import ArliAIProvider
from .internlm import InternLMProvider
from .stepfun import StepFunProvider
from .sambanova import SambanovaProvider
from .upstage import UpstageProvider
from .ngc import NGCProvider
from .cloudflare import CloudflareProvider
from .chutes import ChutesProvider
from .pollinations import PollinationsProvider
from .bigmodel import ZAIProvider, ZAICodeProvider

# Import providers with optional dependencies
try:
    from .gemini import GeminiProvider  # noqa: F401
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    from .tu import TUProvider
    from .tu_embedding import TuAIEmbeddingProvider
    from .tu_tts import TuAITTSProvider
    from .tu_stt import TuAISTTProvider
    HAS_TU = True
except ImportError:
    HAS_TU = False

# Import other providers with optional dependencies
try:
    from .huggingface import HuggingFaceProvider  # noqa: F401
    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False

try:
    from .cohere import CohereProvider  # noqa: F401
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False

try:
    from .moonshot import MoonshotProvider  # noqa: F401
    HAS_MOONSHOT = True
except ImportError:
    HAS_MOONSHOT = False

try:
    from .groq import GroqProvider  # noqa: F401
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

try:
    from .ai21 import AI21Provider  # noqa: F401
    HAS_AI21 = True
except ImportError:
    HAS_AI21 = False

# Import all provider classes here so they can be easily imported from uniinfer.providers
__all__ = [
    'MistralProvider',
    'AnthropicProvider',
    'OpenAIProvider',
    'OpenAITTSProvider',
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
    'ChutesProvider',
    'PollinationsProvider',
    'ZAIProvider',
    'ZAICodeProvider',
    'TUProvider',
    'TuAIEmbeddingProvider',
    'TuAITTSProvider',
    'TuAISTTProvider',
]

# Add optional providers to __all__ if available
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
