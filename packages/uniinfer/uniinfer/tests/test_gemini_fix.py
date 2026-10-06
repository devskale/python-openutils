
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


class TestGeminiSchemaSanitizer:
    """additionalProperties in any position makes the live Gemini API 400
    (Unknown name \"additional_properties\") — the provider must strip it
    recursively before building FunctionDeclarations."""

    TOOLS_WITH_ADDITIONAL_PROPERTIES = [
        {
            "type": "function",
            "function": {
                "name": "edit",
                "description": "edit a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "edits": {"type": "object", "additionalProperties": True},
                        "closed": {"type": "object", "additionalProperties": False, "properties": {}},
                        "union": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "object", "additionalProperties": True},
                            ]
                        },
                    },
                    "required": ["path"],
                },
            },
        }
    ]

    def setup_method(self):
        self.provider = GeminiProvider(api_key="test-key")

    def test_additional_properties_stripped_everywhere(self):
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="gemini-2.5-flash",
            tools=self.TOOLS_WITH_ADDITIONAL_PROPERTIES,
        )
        _, _, tools = self.provider._prepare_content_and_config(request)
        params = tools[0]["parameters"]
        assert "additionalProperties" not in params["properties"]["edits"]
        assert "additionalProperties" not in params["properties"]["closed"]
        assert all("additionalProperties" not in alt for alt in params["properties"]["union"]["anyOf"])
        # everything else survives untouched
        assert params["properties"]["edits"]["type"] == "object"
        assert params["properties"]["union"]["anyOf"][0] == {"type": "string"}
        assert params["required"] == ["path"]

    def test_schema_without_additional_properties_untouched(self):
        clean = {
            "type": "object",
            "properties": {"path": {"type": "string", "pattern": "^/"}},
            "required": ["path"],
        }
        request = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="gemini-2.5-flash",
            tools=[{"type": "function", "function": {"name": "edit", "parameters": clean}}],
        )
        _, _, tools = self.provider._prepare_content_and_config(request)
        assert tools[0]["parameters"] == clean
