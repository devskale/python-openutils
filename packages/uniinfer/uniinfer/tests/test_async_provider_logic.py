"""
Comprehensive tests for refactored async provider logic.
"""
import pytest
import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch
from uniinfer import ChatMessage, ChatCompletionRequest, ChatCompletionResponse
from uniinfer.providers.openai import OpenAIProvider
from uniinfer.providers.groq import GroqProvider
from uniinfer.providers.sambanova import SambanovaProvider
from uniinfer.providers.internlm import InternLMProvider
from uniinfer.providers.moonshot import MoonshotProvider
from uniinfer.providers.stepfun import StepFunProvider
from uniinfer.providers.upstage import UpstageProvider
from uniinfer.providers.ngc import NGCProvider

@pytest.fixture
def chat_request():
    return ChatCompletionRequest(
        model="test-model",
        messages=[ChatMessage(role="user", content="Hello")],
        temperature=0.7
    )

@pytest.mark.asyncio
async def test_openai_provider_acomplete(chat_request):
    provider = OpenAIProvider(api_key="test-key")
    
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "Hi there"}}],
        "usage": {"total_tokens": 15},
        "model": "test-model"
    }

    # Mock the internal httpx client
    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response
    
    with patch.object(provider, '_get_async_client', new_callable=AsyncMock) as mock_get_client:
        mock_get_client.return_value = mock_client
        
        response = await provider.acomplete(chat_request)
        
        assert response.message.content == "Hi there"
        assert response.provider == "openai"
        assert response.usage["total_tokens"] == 15

@pytest.mark.asyncio
async def test_groq_provider_acomplete(chat_request):
    provider = GroqProvider(api_key="test-key")
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.role = "assistant"
    mock_response.choices[0].message.content = "Groq response"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model_dump.return_value = {"id": "test-res"}

    with patch.object(provider.async_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        
        response = await provider.acomplete(chat_request)
        
        assert response.message.content == "Groq response"
        assert response.provider == "groq"
        mock_create.assert_called_once()

@pytest.mark.asyncio
async def test_sambanova_provider_acomplete(chat_request):
    provider = SambanovaProvider(api_key="test-key")
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.role = "assistant"
    mock_response.choices[0].message.content = "Samba response"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model_dump.return_value = {"id": "test-res"}

    with patch.object(provider.async_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        
        response = await provider.acomplete(chat_request)
        
        assert response.message.content == "Samba response"
        assert response.provider == "sambanova"
        mock_create.assert_called_once()

@pytest.mark.asyncio
async def test_moonshot_provider_acomplete(chat_request):
    provider = MoonshotProvider(api_key="test-key")
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.role = "assistant"
    mock_response.choices[0].message.content = "Moonshot response"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model_dump.return_value = {"id": "test-res"}

    with patch.object(provider.async_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        
        response = await provider.acomplete(chat_request)
        
        assert response.message.content == "Moonshot response"
        assert response.provider == "moonshot"
        mock_create.assert_called_once()

@pytest.mark.asyncio
async def test_stepfun_provider_acomplete(chat_request):
    provider = StepFunProvider(api_key="test-key")
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.role = "assistant"
    mock_response.choices[0].message.content = "StepFun response"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model_dump.return_value = {"id": "test-res"}

    with patch.object(provider.async_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        
        response = await provider.acomplete(chat_request)
        
        assert response.message.content == "StepFun response"
        assert response.provider == "stepfun"
        mock_create.assert_called_once()

@pytest.mark.asyncio
async def test_upstage_provider_acomplete(chat_request):
    provider = UpstageProvider(api_key="test-key")
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.role = "assistant"
    mock_response.choices[0].message.content = "Upstage response"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model_dump.return_value = {"id": "test-res"}

    with patch.object(provider.async_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        
        response = await provider.acomplete(chat_request)
        
        assert response.message.content == "Upstage response"
        assert response.provider == "upstage"
        mock_create.assert_called_once()

@pytest.mark.asyncio
async def test_ngc_provider_acomplete(chat_request):
    provider = NGCProvider(api_key="test-key")
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.role = "assistant"
    mock_response.choices[0].message.content = "NGC response"
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15
    mock_response.model_dump.return_value = {"id": "test-res"}

    with patch.object(provider.async_client.chat.completions, 'create', new_callable=AsyncMock) as mock_create:
        mock_create.return_value = mock_response
        
        response = await provider.acomplete(chat_request)
        
        assert response.message.content == "NGC response"
        assert response.provider == "ngc"
        mock_create.assert_called_once()

@pytest.mark.asyncio
async def test_openai_streaming(chat_request):
    provider = OpenAIProvider(api_key="test-key")
    
    async def mock_aiter():
        yield "data: " + json.dumps({"choices": [{"delta": {"content": "Hello"}}]})
        yield "data: " + json.dumps({"choices": [{"delta": {"content": " world"}}]})
        yield "data: [DONE]"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.aiter_lines.return_value = mock_aiter()
    
    mock_context = AsyncMock()
    mock_context.__aenter__.return_value = mock_response

    mock_client = MagicMock()
    mock_client.stream.return_value = mock_context
    
    with patch.object(provider, '_get_async_client', new_callable=AsyncMock) as mock_get_client:
        mock_get_client.return_value = mock_client
        
        chunks = []
        async for chunk in provider.astream_complete(chat_request):
            chunks.append(chunk.message.content)
            
        assert "".join(chunks) == "Hello world"
