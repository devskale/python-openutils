"""
Google Gemini provider implementation.
"""
import json
from typing import Dict, Any, Iterator, Optional, List

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import AuthenticationError, map_provider_error, UniInferError

# Try to import the google-genai package (latest recommended package)
# Note: Install with 'pip install google-genai' if not available
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class GeminiProvider(ChatProvider):
    """
    Provider for Google Gemini API.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the Gemini provider.

        Args:
            api_key (Optional[str]): The Gemini API key.
            **kwargs: Additional provider-specific configuration parameters.
        """
        super().__init__(api_key)

        if not HAS_GENAI:
            raise ImportError(
                "The 'google-genai' package is required to use the Gemini provider. "
                "Install it with 'pip install google-genai'"
            )

        # Create the Gemini client using the new client-based approach
        self.client = genai.Client(api_key=self.api_key)

        # Save any additional configuration
        self.config = kwargs

    def _prepare_content_and_config(self, request: ChatCompletionRequest) -> tuple:
        """
        Prepare content and config for Gemini API from our messages.

        Args:
            request (ChatCompletionRequest): The request to prepare for.

        Returns:
            tuple: (content, config, tools) for the Gemini API.
        """
        # Extract all messages
        messages = request.messages

        def _flatten_text(content: Any) -> str:
            """
            Normalize message content to plain text.
            Supports OpenAI-style content arrays: [{"type":"text","text":"..."}, ...].
            """
            if isinstance(content, list):
                parts: List[str] = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                if parts:
                    return "".join(parts)
                # Fallback: join any string-like items
                return "".join(str(p) for p in content)
            return content if isinstance(content, str) else str(content) if content is not None else ""

        # Look for system message
        system_message = None
        for msg in messages:
            if msg.role == "system":
                system_message = _flatten_text(msg.content)
                break

        # Prepare config with generation parameters
        config_params = {}
        if request.temperature is not None:
            config_params["temperature"] = request.temperature
        if request.max_tokens is not None:
            config_params["max_output_tokens"] = request.max_tokens

        # Create the config object using the new types structure
        config = types.GenerateContentConfig(**config_params)

        # Prepare the content based on non-system messages
        # For simple queries with just one user message, use a simple string
        if len(messages) == 1 and messages[0].role == "user":
            content = _flatten_text(messages[0].content)
        else:
            # For more complex exchanges, format as a conversation
            content = []
            for msg in messages:
                if msg.role == "system":
                    # System messages handled in config
                    continue
                elif msg.role == "user":
                    content.append(
                        {"role": "user", "parts": [{"text": _flatten_text(msg.content)}]})
                elif msg.role == "assistant":
                    # Handle assistant messages with tool_calls
                    parts = []
                    if msg.content:
                        flat = _flatten_text(msg.content)
                        if flat:
                            parts.append({"text": flat})
                    if msg.tool_calls:
                        # Convert OpenAI tool_calls to Gemini function_call format
                        for tool_call in msg.tool_calls:
                            if tool_call.get('type') == 'function':
                                func = tool_call.get('function', {})
                                parts.append({
                                    "function_call": {
                                        "name": func.get('name'),
                                        "args": json.loads(func.get('arguments', '{}'))
                                    }
                                })
                    content.append({"role": "model", "parts": parts})
                elif msg.role == "tool":
                    # Handle tool response messages
                    content.append({
                        "role": "function",
                        "parts": [{
                            "function_response": {
                                "name": msg.tool_call_id,  # Use tool_call_id as name
                                "response": {"content": msg.content}
                            }
                        }]
                    })

        if system_message:
            # Prepend system message to the first user message, or add it as the first message
            if content and content[0]['role'] == 'user':
                if isinstance(content[0]['parts'], list) and content[0]['parts'] and 'text' in content[0]['parts'][0]:
                    content[0]['parts'][0]['text'] = f"{system_message}\n\n{content[0]['parts'][0]['text']}"
                else:
                    # Fallback if structure is not as expected
                    content.insert(0, {"role": "user", "parts": [
                                   {"text": system_message}]})
            else:
                # if no user message, add system message as user message
                content.insert(0, {"role": "user", "parts": [
                               {"text": system_message}]})

        # Convert OpenAI tools format to Gemini function declarations
        gemini_tools = None
        if request.tools:
            gemini_tools = []
            for tool in request.tools:
                if tool.get('type') == 'function':
                    func = tool.get('function', {})
                    gemini_func = {
                        "name": func.get('name'),
                        "description": func.get('description', ''),
                    }
                    if 'parameters' in func:
                        gemini_func["parameters"] = func['parameters']
                    gemini_tools.append(gemini_func)

        return content, config, gemini_tools

    def complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make a chat completion request to Gemini.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Gemini-specific parameters.

        Returns:
            ChatCompletionResponse: The completion response.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("Gemini API key is required")

        try:
            # Get the model from the request or use a default
            model = request.model or "gemini-1.5-pro"

            # Prepare the content and get gemini_tools
            content, _, gemini_tools = self._prepare_content_and_config(request)

            # Prepare config with generation parameters
            config_params = {}
            if request.temperature is not None:
                config_params["temperature"] = request.temperature
            if request.max_tokens is not None:
                config_params["max_output_tokens"] = request.max_tokens
            
            # Add tools to config if provided
            if gemini_tools:
                from google.genai.types import Tool, FunctionDeclaration
                tool_declarations = []
                for func_def in gemini_tools:
                    tool_declarations.append(
                        FunctionDeclaration(
                            name=func_def["name"],
                            description=func_def.get("description", ""),
                            parameters=func_def.get("parameters")
                        )
                    )
                config_params["tools"] = [Tool(function_declarations=tool_declarations)]
            
            # Create the config object
            config = types.GenerateContentConfig(**config_params)

            # Prepare API call parameters
            api_params = {
                "model": model,
                "contents": content,
                "config": config
            }

            # Make the API call using the new client-based approach
            response = self.client.models.generate_content(**api_params)

            # Check for content filtering
            if response.prompt_feedback and response.prompt_feedback.block_reason:
                raise UniInferError(
                    f"Gemini content generation blocked. Reason: {response.prompt_feedback.block_reason}. "
                    f"Safety ratings: {response.prompt_feedback.safety_ratings}"
                )

            # Extract response text and tool calls
            content_text = ""
            tool_calls = None
            
            if response.parts:
                try:
                    content_text = response.text  # Preferred way, handles multi-part correctly if applicable
                # Handles cases where .text might raise ValueError (e.g. no parts with text)
                except ValueError:
                    # Fallback to iterating parts if .text fails or parts exist but .text is empty.
                    # Concatenate text from all text-bearing parts in the response.
                    all_parts_text_list = []
                    for part in response.parts:
                        # Check if the part has a 'text' attribute and if it's not None or empty
                        part_text_content = getattr(part, 'text', None)
                        if part_text_content:  # This ensures part_text_content is not None and not an empty string
                            all_parts_text_list.append(part_text_content)
                    if all_parts_text_list:
                        content_text = "".join(all_parts_text_list)
                
                # Check for function calls in response parts
                function_calls = []
                for part in response.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        # Convert Gemini function_call to OpenAI tool_calls format
                        import json
                        function_calls.append({
                            "id": f"call_{part.function_call.name}",
                            "type": "function",
                            "function": {
                                "name": part.function_call.name,
                                "arguments": json.dumps(dict(part.function_call.args))
                            }
                        })
                if function_calls:
                    tool_calls = function_calls

            if not content_text and not tool_calls and not (response.prompt_feedback and response.prompt_feedback.block_reason):
                # If content is still empty and not blocked, it might be an empty generation
                print(
                    "Warning: Gemini response has no text content and no explicit block reason.")

            # Create a ChatMessage from the response
            message = ChatMessage(
                role="assistant",
                content=content_text if content_text else None,
                tool_calls=tool_calls
            )

            # Create usage information (Gemini doesn't provide detailed token counts)
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

            return ChatCompletionResponse(
                message=message,
                provider='gemini',
                model=model,
                usage=usage,
                raw_response=response
            )

        except Exception as e:
            # Map the error to a standardized format with rich info
            status_code = None
            response_body = str(e)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                response_body = e.response.text
                status_code = e.response.status_code
            elif hasattr(e, 'status_code'):
                status_code = e.status_code
            elif hasattr(e, 'message'):
                response_body = e.message

            mapped_error = map_provider_error("gemini", e, status_code=status_code, response_body=response_body)
            raise mapped_error

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from Gemini.

        Args:
            api_key (Optional[str]): The Gemini API key.

        Returns:
            list: A list of available model names.
        """
        if not HAS_GENAI:
            raise ImportError(
                "The 'google-genai' package is required to use the Gemini provider. "
                "Install it with 'pip install google-genai'"
            )

        # Try to get API key from credgoo if not provided
        if api_key is None:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("gemini")
                if api_key is None:
                    raise ValueError(
                        "Failed to retrieve Gemini API key from credgoo")
            except ImportError:
                raise ValueError(
                    "Gemini API key is required when credgoo is not available")

        # Create a client instance for listing models
        client = genai.Client(api_key=api_key)
        models = []

        # Get models using the new client-based approach
        model_list = client.models.list()
        for model in model_list:
            models.append(model.name)

        return models

    def stream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> Iterator[ChatCompletionResponse]:
        """
        Stream a chat completion response from Gemini.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Gemini-specific parameters.

        Returns:
            Iterator[ChatCompletionResponse]: An iterator of response chunks.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("Gemini API key is required")

        try:
            # Get the model from the request or use a default
            model = request.model or "gemini-1.5-pro"

            # Prepare the content and get gemini_tools
            content, _, gemini_tools = self._prepare_content_and_config(request)

            # Prepare config with generation parameters
            config_params = {}
            if request.temperature is not None:
                config_params["temperature"] = request.temperature
            if request.max_tokens is not None:
                config_params["max_output_tokens"] = request.max_tokens
            
            # Add tools to config if provided
            if gemini_tools:
                from google.genai.types import Tool, FunctionDeclaration, GenerateContentConfig
                tool_declarations = []
                for func_def in gemini_tools:
                    tool_declarations.append(
                        FunctionDeclaration(
                            name=func_def["name"],
                            description=func_def.get("description", ""),
                            parameters=func_def.get("parameters")
                        )
                    )
                config_params["tools"] = [Tool(function_declarations=tool_declarations)]
            
            # Create the config object
            from google.genai import types
            config = types.GenerateContentConfig(**config_params)

            # Prepare API call parameters
            api_params = {
                "model": model,
                "contents": content,
                "config": config
            }
            
            # Make the streaming API call using the new client-based approach
            stream = self.client.models.generate_content_stream(**api_params)

            # Process the streaming response
            for chunk in stream:
                # Check for content filtering in each chunk
                if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                    # Yield a response indicating blockage, or raise an error
                    # For now, let's raise an error to make it explicit
                    raise UniInferError(
                        f"Gemini content generation blocked mid-stream. Reason: {chunk.prompt_feedback.block_reason}. "
                        f"Safety ratings: {chunk.prompt_feedback.safety_ratings}"
                    )

                # Ensure parts exist and are not empty before accessing text
                chunk_text = ""
                chunk_tool_calls = None
                
                if chunk.parts:
                    try:
                        chunk_text = chunk.text  # Preferred way
                    except ValueError:  # Handles cases where .text might raise ValueError
                        # Fallback to iterating parts if .text fails.
                        # Concatenate text from all text-bearing parts in the chunk.
                        all_parts_text_list = []
                        for part in chunk.parts:
                            # Check if the part has a 'text' attribute and if it's not None or empty
                            part_text_content = getattr(part, 'text', None)
                            if part_text_content:  # This ensures part_text_content is not None and not an empty string
                                all_parts_text_list.append(part_text_content)
                        if all_parts_text_list:
                            chunk_text = "".join(all_parts_text_list)
                    
                    # Check for function calls in chunk parts
                    function_calls = []
                    for part in chunk.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            # Convert Gemini function_call to OpenAI tool_calls format
                            import json
                            function_calls.append({
                                "id": f"call_{part.function_call.name}",
                                "type": "function",
                                "function": {
                                    "name": part.function_call.name,
                                    "arguments": json.dumps(dict(part.function_call.args))
                                }
                            })
                    if function_calls:
                        chunk_tool_calls = function_calls

                if chunk_text or chunk_tool_calls:
                    # Create a message for this chunk
                    message = ChatMessage(
                        role="assistant",
                        content=chunk_text if chunk_text else None,
                        tool_calls=chunk_tool_calls
                    )

                    # Create a response for this chunk
                    yield ChatCompletionResponse(
                        message=message,
                        provider='gemini',
                        model=model,
                        usage={},
                        raw_response=chunk
                    )
                elif not (chunk.prompt_feedback and chunk.prompt_feedback.block_reason):
                    # If no text and not blocked, it might be an empty chunk or end of stream signal
                    # Depending on Gemini's stream behavior, this might be normal or an issue
                    # For now, we'll just skip empty, non-blocked chunks
                    # print(f"DEBUG: Gemini stream chunk has no text and no block reason: {chunk}")
                    pass

        except Exception as e:
            # Map the error to a standardized format with rich info
            status_code = None
            response_body = str(e)
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                response_body = e.response.text
                status_code = e.response.status_code
            elif hasattr(e, 'status_code'):
                status_code = e.status_code
            elif hasattr(e, 'message'):
                response_body = e.message

            mapped_error = map_provider_error("gemini", e, status_code=status_code, response_body=response_body)
            raise mapped_error
