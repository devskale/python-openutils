"""
Tests for TuAIProvider async and tool support.
"""
import pytest
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
from uniinfer import ChatMessage, ChatCompletionRequest, ChatCompletionResponse
from uniinfer.providers.tu import TUProvider

class TestTUProviderAsync:
    """Test suite for TUProvider async methods."""

    @pytest.fixture
    def provider(self):
        return TUProvider(api_key="test-key")

    def test_init(self, provider):
        """Test provider initialization."""
        assert provider.api_key == "test-key"
        assert provider.base_url == "https://aqueduct.ai.datalab.tuwien.ac.at/v1"

    @pytest.mark.asyncio
    async def test_acomplete_success(self, provider):
        """Test successful async chat completion."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")]
        )

        mock_response_data = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hi there!"
                },
                "finish_reason": "stop"
            }],
            "model": "test-model",
            "usage": {"total_tokens": 10}
        }

        mock_http_response = MagicMock(spec=httpx.Response)
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response_data
        mock_http_response.text = '{"choices": []}'
        mock_http_response.headers = {"content-type": "application/json"}

        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.post.return_value = mock_http_response
        mock_client.is_closed = False

        provider._async_client = mock_client

        with patch("uniinfer.providers.tu.log_raw_response"):
            response = await provider.acomplete(request)

        assert isinstance(response, ChatCompletionResponse)
        assert response.message.content == "Hi there!"
        assert response.model == "test-model"

    @pytest.mark.asyncio
    async def test_astream_complete_success(self, provider):
        """Test successful async streaming chat completion."""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")]
        )

        # Mock stream response
        chunks = [
            'data: {"choices": [{"delta": {"role": "assistant", "content": "Hi"}}], "model": "test-model"}\n\n',
            'data: {"choices": [{"delta": {"content": " there!"}}], "model": "test-model"}\n\n',
            'data: [DONE]\n\n'
        ]

        async def mock_aiter_lines():
            for chunk in chunks:
                yield chunk.strip()

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines
        # Add raise_for_status to mock_response
        mock_response.raise_for_status = MagicMock()

        # Mock httpx.AsyncClient.stream
        mock_client = AsyncMock(spec=httpx.AsyncClient)
        mock_client.stream.return_value.__aenter__.return_value = mock_response
        
        with patch.object(provider, "_get_async_client", return_value=mock_client):
            responses = []
            async for chunk in provider.astream_complete(request):
                responses.append(chunk)
            
            assert len(responses) == 2
            assert responses[0].message.content == "Hi"
            assert responses[1].message.content == " there!"

    def test_prepare_payload_with_tools(self, provider):
        """Test payload preparation with tool support."""
        tools = [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}}
            }
        }]
        
        request = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="What's the weather?")],
            tools=tools,
            tool_choice="auto"
        )
        
        payload = provider._prepare_payload(request)
        
        assert "tools" in payload
        assert payload["tools"] == tools
        assert payload["tool_choice"] == "auto"
        assert payload["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_throttle(self, provider):
        """_throttle now delegates to the shared adaptive per-model limiter."""
        from uniinfer.providers.tu import _TU_LIMITER
        _TU_LIMITER.reset("test-model")
        info = await provider._throttle("test-model")
        assert isinstance(info, dict)
        assert "rpm" in info
        assert "waited" in info
        assert info["rpm"] >= 0.5

    def test_flatten_messages(self, provider):
        """Test message flattening logic (deprecated in TUProvider, but checking core compatibility)."""
        # TUProvider no longer has _flatten_messages, it uses direct mapping.
        # We can remove these tests or update them to test _prepare_payload.
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi", tool_calls=[{"id": "1", "type": "function", "function": {"name": "f"}}])
            ]
        )
        payload = provider._prepare_payload(request)
        assert len(payload["messages"]) == 2
        assert payload["messages"][1]["tool_calls"][0]["id"] == "1"

    def test_flatten_messages_vlm(self, provider):
        """Test VLM payload preparation."""
        content = [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content=content)]
        )
        payload = provider._prepare_payload(request)
        assert payload["messages"][0]["content"] == content
