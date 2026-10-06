
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


class TestGeminiTurnMapping:
    """Newer Gemini models (gemini-3.5-flash-lite) reject requests that end
    with a model turn, and tool calls/results have their own wire shape
    (functionCall on model turns, functionResponse on user turns). The old
    flat mapping dropped tool calls and sent tool results as model text."""

    def setup_method(self):
        self.provider = GeminiProvider(api_key="test-key")

    def _content_for(self, messages):
        request = ChatCompletionRequest(messages=messages, model="gemini-3.5-flash-lite")
        content, _, _ = self.provider._prepare_content_and_config(request)
        return content

    def test_assistant_tool_calls_become_function_call_parts(self):
        content = self._content_for([
            ChatMessage(role="user", content="search"),
            ChatMessage(role="assistant", content="", tool_calls=[
                {"id": "call_1", "type": "function",
                 "function": {"name": "web_search", "arguments": '{"query": "ESP32"}'}},
            ]),
        ])
        model_turn = content[1]
        assert model_turn["role"] == "model"
        fc = model_turn["parts"][0]["functionCall"]
        assert fc["name"] == "web_search"
        assert fc["args"] == {"query": "ESP32"}

    def test_tool_result_is_function_response_on_user_turn(self):
        content = self._content_for([
            ChatMessage(role="user", content="search"),
            ChatMessage(role="assistant", content="", tool_calls=[
                {"id": "call_1", "type": "function",
                 "function": {"name": "web_search", "arguments": '{}'}},
            ]),
            ChatMessage(role="tool", content="10 results", tool_call_id="call_1"),
        ])
        result_turn = content[2]
        assert result_turn["role"] == "user"
        fr = result_turn["parts"][0]["functionResponse"]
        assert fr["name"] == "web_search"
        assert fr["response"] == {"result": "10 results"}

    def test_transcript_ending_with_model_turn_gets_user_close(self):
        content = self._content_for([
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="assistant", content="Hello! How can I help?"),
        ])
        assert content[-1]["role"] == "user"

    def test_transcript_ending_with_user_turn_untouched(self):
        content = self._content_for([ChatMessage(role="user", content="hi") * 1 if False else ChatMessage(role="user", content="hi")])
        assert content == "hi"  # single-message shortcut

    def test_tool_result_without_parent_call_gets_fallback_name(self):
        content = self._content_for([
            ChatMessage(role="user", content="hi"),
            ChatMessage(role="tool", content="orphan result", tool_call_id="missing"),
        ])
        fr = content[1]["parts"][0]["functionResponse"]
        assert fr["name"] == "unknown_function"
        assert fr["response"] == {"result": "orphan result"}


class TestGeminiThoughtSignatures:
    """Gemini 3 requires the original thought signature replayed on every
    functionCall part of a tool round; without it the API 400s ('Function call
    is missing a thought_signature' / 'Corrupted thought signature'). The
    signature rides the OpenAI-compat wire as reasoning.encrypted
    reasoning_details keyed by tool_call id (the shape pi preserves)."""

    def setup_method(self):
        self.provider = GeminiProvider(api_key="test-key")

    def test_response_extracts_signatures_from_parts(self):
        from unittest.mock import MagicMock
        import base64
        part = MagicMock(function_call=MagicMock(name="web_search", args={"q": "x"}),
                         thought_signature=b"\x9a\x04", text=None, thought=None)
        part.function_call.name = "web_search"
        response = MagicMock(parts=[part], prompt_feedback=None)
        details = self.provider._extract_reasoning_details(response)
        assert details == [{
            "type": "reasoning.encrypted",
            "id": "call_web_search",
            "data": base64.b64encode(b"\x9a\x04").decode(),
            "format": "gemini",
            "index": 0,
        }]

    def test_response_without_signatures_yields_none(self):
        from unittest.mock import MagicMock
        part = MagicMock(function_call=None, thought_signature=None, text="hi", thought=None)
        response = MagicMock(parts=[part], prompt_feedback=None)
        assert self.provider._extract_reasoning_details(response) is None

    def test_request_replays_signature_on_function_call_part(self):
        content = self._content_for = None
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="user", content="search"),
                ChatMessage(role="assistant", content="", tool_calls=[
                    {"id": "call_web_search", "type": "function",
                     "function": {"name": "web_search", "arguments": '{"q": "ESP32"}'}},
                ], reasoning_details=[
                    {"type": "reasoning.encrypted", "id": "call_web_search",
                     "data": "mgQ=", "format": "gemini", "index": 0},
                ]),
                ChatMessage(role="tool", content="3 results", tool_call_id="call_web_search"),
            ],
            model="gemini-3.5-flash-lite",
        )
        content, _, _ = self.provider._prepare_content_and_config(request)
        model_part = content[1]["parts"][0]
        assert model_part["functionCall"]["name"] == "web_search"
        assert model_part["thoughtSignature"] == "mgQ="

    def test_request_without_signature_leaves_part_clean(self):
        request = ChatCompletionRequest(
            messages=[
                ChatMessage(role="user", content="search"),
                ChatMessage(role="assistant", content="", tool_calls=[
                    {"id": "call_x", "type": "function",
                     "function": {"name": "x", "arguments": '{}'}},
                ]),
            ],
            model="gemini-3.5-flash-lite",
        )
        content, _, _ = self.provider._prepare_content_and_config(request)
        assert "thoughtSignature" not in content[1]["parts"][0]

    def test_chat_message_reasoning_details_round_trip(self):
        msg = ChatMessage(role="assistant", content=None,
                          reasoning_details=[{"type": "reasoning.encrypted", "id": "a", "data": "b"}])
        d = msg.to_dict()
        assert d["reasoning_details"][0]["data"] == "b"
        assert ChatMessage(**d).reasoning_details == msg.reasoning_details
