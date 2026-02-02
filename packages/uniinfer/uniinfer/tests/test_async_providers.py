"""
Unit tests for async provider methods.
"""
import pytest
from uniinfer import ChatMessage, ChatCompletionRequest
from uniinfer.providers.openai import OpenAIProvider
from uniinfer.providers.anthropic import AnthropicProvider
from uniinfer.providers.mistral import MistralProvider
from uniinfer.providers.ollama import OllamaProvider
from uniinfer.providers.gemini import GeminiProvider
from uniinfer.providers.pollinations import PollinationsProvider


class TestOpenAIAsync:
    """Test async methods for OpenAI provider."""

    def test_acomplete_method_exists(self):
        """Test that acomplete method exists."""
        provider = OpenAIProvider(api_key="test-key")
        assert hasattr(provider, 'acomplete')
        assert callable(provider.acomplete)

    def test_astream_complete_method_exists(self):
        """Test that astream_complete method exists."""
        provider = OpenAIProvider(api_key="test-key")
        assert hasattr(provider, 'astream_complete')
        assert callable(provider.astream_complete)

    def test_acomplete_is_async(self):
        """Test that acomplete is a coroutine function."""
        import inspect
        provider = OpenAIProvider(api_key="test-key")
        assert inspect.iscoroutinefunction(provider.acomplete)

    def test_astream_complete_is_async_generator(self):
        """Test that astream_complete is an async generator function."""
        import inspect
        provider = OpenAIProvider(api_key="test-key")
        assert inspect.isasyncgenfunction(provider.astream_complete)


class TestAnthropicAsync:
    """Test async methods for Anthropic provider."""

    def test_acomplete_method_exists(self):
        """Test that acomplete method exists."""
        provider = AnthropicProvider(api_key="test-key")
        assert hasattr(provider, 'acomplete')
        assert callable(provider.acomplete)

    def test_astream_complete_method_exists(self):
        """Test that astream_complete method exists."""
        provider = AnthropicProvider(api_key="test-key")
        assert hasattr(provider, 'astream_complete')
        assert callable(provider.astream_complete)

    def test_acomplete_is_async(self):
        """Test that acomplete is a coroutine function."""
        import inspect
        provider = AnthropicProvider(api_key="test-key")
        assert inspect.iscoroutinefunction(provider.acomplete)

    def test_astream_complete_is_async_generator(self):
        """Test that astream_complete is an async generator function."""
        import inspect
        provider = AnthropicProvider(api_key="test-key")
        assert inspect.isasyncgenfunction(provider.astream_complete)


class TestMistralAsync:
    """Test async methods for Mistral provider."""

    def test_acomplete_method_exists(self):
        """Test that acomplete method exists."""
        provider = MistralProvider(api_key="test-key")
        assert hasattr(provider, 'acomplete')
        assert callable(provider.acomplete)

    def test_astream_complete_method_exists(self):
        """Test that astream_complete method exists."""
        provider = MistralProvider(api_key="test-key")
        assert hasattr(provider, 'astream_complete')
        assert callable(provider.astream_complete)

    def test_acomplete_is_async(self):
        """Test that acomplete is a coroutine function."""
        import inspect
        provider = MistralProvider(api_key="test-key")
        assert inspect.iscoroutinefunction(provider.acomplete)

    def test_astream_complete_is_async_generator(self):
        """Test that astream_complete is an async generator function."""
        import inspect
        provider = MistralProvider(api_key="test-key")
        assert inspect.isasyncgenfunction(provider.astream_complete)


class TestOllamaAsync:
    """Test async methods for Ollama provider."""

    def test_acomplete_method_exists(self):
        """Test that acomplete method exists."""
        provider = OllamaProvider(base_url="http://localhost:11434")
        assert hasattr(provider, 'acomplete')
        assert callable(provider.acomplete)

    def test_astream_complete_method_exists(self):
        """Test that astream_complete method exists."""
        provider = OllamaProvider(base_url="http://localhost:11434")
        assert hasattr(provider, 'astream_complete')
        assert callable(provider.astream_complete)

    def test_acomplete_is_async(self):
        """Test that acomplete is a coroutine function."""
        import inspect
        provider = OllamaProvider(base_url="http://localhost:11434")
        assert inspect.iscoroutinefunction(provider.acomplete)

    def test_astream_complete_is_async_generator(self):
        """Test that astream_complete is an async generator function."""
        import inspect
        provider = OllamaProvider(base_url="http://localhost:11434")
        assert inspect.isasyncgenfunction(provider.astream_complete)


class TestPollinationsAsync:
    """Test async methods for Pollinations provider."""

    def test_acomplete_method_exists(self):
        """Test that acomplete method exists."""
        provider = PollinationsProvider(api_key="test-key")
        assert hasattr(provider, 'acomplete')
        assert callable(provider.acomplete)

    def test_astream_complete_method_exists(self):
        """Test that astream_complete method exists."""
        provider = PollinationsProvider(api_key="test-key")
        assert hasattr(provider, 'astream_complete')
        assert callable(provider.astream_complete)

    def test_acomplete_is_async(self):
        """Test that acomplete is a coroutine function."""
        import inspect
        provider = PollinationsProvider(api_key="test-key")
        assert inspect.iscoroutinefunction(provider.acomplete)

    def test_astream_complete_is_async_generator(self):
        """Test that astream_complete is an async generator function."""
        import inspect
        provider = PollinationsProvider(api_key="test-key")
        assert inspect.isasyncgenfunction(provider.astream_complete)


class TestGeminiAsync:
    """Test async methods for Gemini provider."""

    def test_acomplete_method_exists(self):
        """Test that acomplete method exists."""
        provider = GeminiProvider(api_key="test-key")
        assert hasattr(provider, 'acomplete')
        assert callable(provider.acomplete)

    def test_astream_complete_method_exists(self):
        """Test that astream_complete method exists."""
        provider = GeminiProvider(api_key="test-key")
        assert hasattr(provider, 'astream_complete')
        assert callable(provider.astream_complete)

    def test_acomplete_is_async(self):
        """Test that acomplete is a coroutine function."""
        import inspect
        provider = GeminiProvider(api_key="test-key")
        assert inspect.iscoroutinefunction(provider.acomplete)

    def test_astream_complete_is_async_generator(self):
        """Test that astream_complete is an async generator function."""
        import inspect
        provider = GeminiProvider(api_key="test-key")
        assert inspect.isasyncgenfunction(provider.astream_complete)


class TestSyncWrappers:
    """Test that sync methods work as wrappers."""

    def test_openai_complete_is_wrapper(self):
        """Test that sync complete method wraps async."""
        provider = OpenAIProvider(api_key="test-key")
        assert hasattr(provider, 'complete')
        assert callable(provider.complete)

    def test_anthropic_complete_is_wrapper(self):
        """Test that sync complete method wraps async."""
        provider = AnthropicProvider(api_key="test-key")
        assert hasattr(provider, 'complete')
        assert callable(provider.complete)

    def test_mistral_complete_is_wrapper(self):
        """Test that sync complete method wraps async."""
        provider = MistralProvider(api_key="test-key")
        assert hasattr(provider, 'complete')
        assert callable(provider.complete)

    def test_ollama_complete_is_wrapper(self):
        """Test that sync complete method wraps async."""
        provider = OllamaProvider(base_url="http://localhost:11434")
        assert hasattr(provider, 'complete')
        assert callable(provider.complete)

    def test_pollinations_complete_is_wrapper(self):
        """Test that sync complete method wraps async."""
        provider = PollinationsProvider(api_key="test-key")
        assert hasattr(provider, 'complete')
        assert callable(provider.complete)

    def test_openai_stream_complete_is_wrapper(self):
        """Test that sync stream_complete wraps async."""
        provider = OpenAIProvider(api_key="test-key")
        assert hasattr(provider, 'stream_complete')
        assert callable(provider.stream_complete)

    def test_anthropic_stream_complete_is_wrapper(self):
        """Test that sync stream_complete wraps async."""
        provider = AnthropicProvider(api_key="test-key")
        assert hasattr(provider, 'stream_complete')
        assert callable(provider.stream_complete)

    def test_mistral_stream_complete_is_wrapper(self):
        """Test that sync stream_complete wraps async."""
        provider = MistralProvider(api_key="test-key")
        assert hasattr(provider, 'stream_complete')
        assert callable(provider.stream_complete)

    def test_ollama_stream_complete_is_wrapper(self):
        """Test that sync stream_complete wraps async."""
        provider = OllamaProvider(base_url="http://localhost:11434")
        assert hasattr(provider, 'stream_complete')
        assert callable(provider.stream_complete)

    def test_pollinations_stream_complete_is_wrapper(self):
        """Test that sync stream_complete wraps async."""
        provider = PollinationsProvider(api_key="test-key")
        assert hasattr(provider, 'stream_complete')
        assert callable(provider.stream_complete)


    def test_anthropic_complete_is_wrapper(self):
        """Test that sync complete method wraps async."""
        provider = AnthropicProvider(api_key="test-key")
        assert hasattr(provider, 'complete')
        assert callable(provider.complete)

    def test_mistral_complete_is_wrapper(self):
        """Test that sync complete method wraps async."""
        provider = MistralProvider(api_key="test-key")
        assert hasattr(provider, 'complete')
        assert callable(provider.complete)

    def test_ollama_complete_is_wrapper(self):
        """Test that sync complete method wraps async."""
        provider = OllamaProvider(base_url="http://localhost:11434")
        assert hasattr(provider, 'complete')
        assert callable(provider.complete)
