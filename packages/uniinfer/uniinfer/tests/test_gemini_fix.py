
import pytest
from unittest.mock import MagicMock, patch
from uniinfer import ChatMessage, ChatCompletionRequest
from uniinfer.providers.gemini import GeminiProvider

class TestGeminiFix:
    def setup_method(self):
        self.provider = GeminiProvider(api_key="test-key")

    def test_prepare_content_and_config_single_message(self):
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            model="gemini-1.5-flash"
        )
        content, config, tools = self.provider._prepare_content_and_config(request)
        
        # In the fixed version, for single user message, it should be a string
        assert content == "Hello"
        assert tools is None

    def test_prepare_content_and_config_system_message(self):
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="system", content="You are a helper"),
                ChatMessage(role="user", content="Hello")
            ],
            model="gemini-1.5-flash"
        )
        content, config, tools = self.provider._prepare_content_and_config(request)
        
        # When system message is present, it should be converted to a list
        assert isinstance(content, list)
        # Check that system message was prepended or correctly handled
        # In current implementation, it's prepended to the first user message
        assert any("You are a helper" in p["text"] for p in content[0]["parts"])

    def test_prepare_api_params(self):
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            model="gemini-1.5-flash"
        )
        content, config, tools = self.provider._prepare_content_and_config(request)
        params = self.provider._prepare_api_params(request, content, config, tools)
        
        assert params["model"] == "gemini-1.5-flash"
        # Content is "Hello" string, and _prepare_api_params should keep it as is (passed to client)
        assert params["contents"] == "Hello"
        assert params["config"] == config

    @patch('google.genai.Client')
    @pytest.mark.asyncio
    async def test_acomplete_impl(self, mock_client_class):
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.text = "Hi there!"
        mock_response.parts = [MagicMock(text="Hi there!")]
        mock_response.prompt_feedback = None
        
        # Mock the async call using AsyncMock
        from unittest.mock import AsyncMock
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
        
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="Hello")],
            model="gemini-1.5-flash"
        )
        
        response = await self.provider.acomplete(request)
        assert response.message.content == "Hi there!"
        assert response.provider == "gemini"
