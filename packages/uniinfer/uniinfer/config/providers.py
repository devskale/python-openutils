"""Provider configuration registry used by runtime, CLI, and proxy."""

import os

# Check if HuggingFace support is available
try:
    from uniinfer import HuggingFaceProvider  # noqa: F401
    HAS_HUGGINGFACE = True
except ImportError:
    HAS_HUGGINGFACE = False

# Check if Cohere support is available
try:
    from uniinfer import CohereProvider  # noqa: F401
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False

# Check if Moonshot support is available
try:
    from uniinfer import MoonshotProvider  # noqa: F401
    HAS_MOONSHOT = True
except ImportError:
    HAS_MOONSHOT = False

# Check if OpenAI client is available (for StepFun)
try:
    from openai import OpenAI  # noqa: F401
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

# Check if Groq support is available
try:
    from uniinfer import GroqProvider  # noqa: F401
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

# Check if AI21 support is available
try:
    from uniinfer import AI21Provider  # noqa: F401
    HAS_AI21 = True
except ImportError:
    HAS_AI21 = False

# Check if Gemini support is available (cheap spec check — do NOT import
# GeminiProvider here, it pulls in the heavy google-genai SDK at startup).
import importlib.util as _importlib_util
HAS_GENAI = _importlib_util.find_spec("google.genai") is not None


PROVIDER_CONFIGS = {
    'mistral': {
        'name': 'Mistral AI',
        'default_model': 'mistral-small-latest',
        'needs_api_key': True,
    },
    'anthropic': {
        'name': 'Anthropic (Claude)',
        'default_model': 'claude-3-sonnet-20240229',
        'needs_api_key': True,
    },
    'minimax': {
        'name': 'MiniMax',
        'default_model': 'MiniMax-M2.1',
        'needs_api_key': True,
    },
    'openai': {
        'name': 'OpenAI',
        'default_model': 'gpt-3.5-turbo',
        'needs_api_key': True,
    },
    'ollama': {
        'name': 'Ollama',
        'default_model': 'gemma3:4b',
        'needs_api_key': False,
        'extra_params': {
            'base_url': 'https://amp1.mooo.com:11444'
        }
    },
    'arli': {
        'name': 'ArliAI',
        'default_model': 'Qwen3-14B',
        'needs_api_key': True,
    },
    'openrouter': {
        'name': 'OpenRouter',
        'default_model': 'moonshotai/moonlight-16b-a3b-instruct:free',
        'needs_api_key': True,
    },
    'internlm': {
        'name': 'InternLM',
        'default_model': 'internlm3-latest',
        'needs_api_key': True,
        'extra_params': {
            'top_p': 0.9
        }
    },
    'stepfun': {
        'name': 'StepFun AI',
        'default_model': 'step-1-8k',
        'needs_api_key': True,
    },
    'sambanova': {
        'name': 'SambaNova',
        'default_model': 'Meta-Llama-3.1-8B-Instruct',
        'needs_api_key': True,
    },
    'upstage': {
        'name': 'Upstage AI',
        'default_model': 'solar-pro',
        'needs_api_key': True,
    },
    'ngc': {
        'name': 'NVIDIA GPU Cloud (NGC)',
        'default_model': 'nvidia/llama-3.3-nemotron-super-49b-v1',
        'needs_api_key': True,
    },
    'chutes': {
        'name': 'Chutes AI',
        'default_model': 'deepseek-ai/DeepSeek-V3-0324',
        'needs_api_key': True,
    },
    'opencode': {
        'name': 'OpenCode (Zen)',
        'default_model': 'deepseek-v4-flash-free',
        'needs_api_key': True,
    },
    'kilo': {
        'name': 'Kilo Gateway',
        'default_model': 'tencent/hy3:free',
        'needs_api_key': False,
    },
    'zai': {
        'name': 'Z.ai',
        'default_model': 'glm-4.5-flash',
        'needs_api_key': True,
    },
    'zai-code': {
        'name': 'Z.ai Code',
        'default_model': 'glm-4.5',
        'needs_api_key': True,
    },
    'tu': {
        'name': 'tu',
        'default_model': 'qwen-coder-30b',
        'needs_api_key': True,
    },
    'pollinations': {
        'name': 'Pollinations AI',
        'default_model': 'grok',
        'needs_api_key': False,
    },
    'cloudflare': {
        'name': 'Cloudflare Workers AI',
        'default_model': '@cf/meta/llama-4-scout-17b-16e-instruct',
        'needs_api_key': True,
        'extra_params': {
            'account_id': os.getenv('CLOUDFLARE_ACCOUNT_ID', '1ee331dfd225ac49d67c521a73ca7fe8')
        }
    }
}

if HAS_HUGGINGFACE:
    PROVIDER_CONFIGS['huggingface'] = {
        'name': 'HuggingFace Inference',
        'default_model': 'mistralai/Mistral-7B-Instruct-v0.3',
        'needs_api_key': True,
    }

if HAS_COHERE:
    PROVIDER_CONFIGS['cohere'] = {
        'name': 'Cohere',
        'default_model': 'command-r-plus-08-2024',
        'needs_api_key': True,
    }

if HAS_MOONSHOT:
    PROVIDER_CONFIGS['moonshot'] = {
        'name': 'Moonshot AI',
        'default_model': 'moonshot-v1-8k',
        'needs_api_key': True,
    }

if HAS_GROQ:
    PROVIDER_CONFIGS['groq'] = {
        'name': 'Groq',
        'default_model': 'llama-3.1-8b',
        'needs_api_key': True,
    }

if HAS_AI21:
    PROVIDER_CONFIGS['ai21'] = {
        'name': 'AI21 Labs',
        'default_model': 'jamba-mini-1.6-2025-03',
        'needs_api_key': True,
    }

if HAS_GENAI:
    PROVIDER_CONFIGS['gemini'] = {
        'name': 'Google Gemini',
        'default_model': 'gemini-2.5-flash',
        'needs_api_key': True,
    }


def add_provider(provider_id, config):
    """Add or update a provider configuration."""
    PROVIDER_CONFIGS[provider_id] = config


def get_provider_config(provider_id):
    """Get configuration for a specific provider."""
    return PROVIDER_CONFIGS.get(provider_id)


def get_all_providers():
    """Get all available provider configurations."""
    return PROVIDER_CONFIGS


# --------------------------------------------------------------------------- #
# Documented free-tier rate limits (provider_limits.json)
# --------------------------------------------------------------------------- #
import json as _json
from pathlib import Path as _Path

_LIMITS_FILE = _Path(__file__).parent / "provider_limits.json"


def _load_provider_limits() -> dict:
    """Load provider_limits.json, skipping non-provider keys (``_doc``)."""
    try:
        data = _json.loads(_LIMITS_FILE.read_text())
        return {k: v for k, v in data.items() if not k.startswith("_")}
    except (OSError, _json.JSONDecodeError):
        return {}


_PROVIDER_LIMITS = _load_provider_limits()

# Bake documented free-tier limits into each provider config so
# ``get_provider_config('groq')['free_tier_limits']`` works.
for _pid, _cfg in PROVIDER_CONFIGS.items():
    if _pid in _PROVIDER_LIMITS:
        _cfg["free_tier_limits"] = _PROVIDER_LIMITS[_pid]


def get_provider_limits(provider_id: str) -> dict:
    """Documented free-tier rate limits for a provider (from provider_limits.json)."""
    return _PROVIDER_LIMITS.get(provider_id, {})


def get_all_provider_limits() -> dict:
    """All documented free-tier rate limits, keyed by provider id."""
    return _PROVIDER_LIMITS
