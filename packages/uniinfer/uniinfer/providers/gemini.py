"""
Google Gemini provider implementation with async support.
"""
import asyncio
import json
from typing import Dict, Any, Iterator, Optional, List, AsyncIterator

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import AuthenticationError, map_provider_error, UniInferError

# Try to import google-genai package (latest recommended package)
# Note: Install with 'pip install google-genai' if not available
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


class GeminiProvider(ChatProvider):
    """
    Provider for Google Gemini API with async support.
    
    Uses the modern google-genai SDK which is natively async.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Gemini provider.

        Args:
            api_key (Optional[str]): The Gemini API key.
            **kwargs: Additional provider-specific configuration parameters.
        """
        super().__init__(api_key)
        self._client = None

        if not HAS_GENAI:
            raise ImportError(
                "The 'google-genai' package is required to use the Gemini provider. "
                "Install it with 'pip install google-genai'"
            )

        # Store any additional configuration
        self.config = kwargs

    async def _get_client(self):
        """
        Get or create async Gemini client.
        """
        if self._client is None:
            self._client = genai.AsyncClient(api_key=self.api_key)
        return self._client

    async def close(self):
        """
        Close async client.
        """
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from Gemini.

        Args:
            api_key (Optional[str]): The Gemini API key. If not provided,
                                     it attempts to retrieve it using credgoo.

        Returns:
            list: A list of available model IDs.

        Raises:
            ValueError: If no API key is provided or found.
            Exception: If the API request fails.
        """
        # Retrieve API key if not provided
        if not api_key:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key('gemini')
            except (ImportError, Exception):
                pass

        if not api_key:
            raise ValueError("Gemini API key is required for listing models")

        try:
            # Create synchronous client for list_models
            client = genai.Client(api_key=api_key)
            models = []
            
            # Query the actual API for available models
            for model in client.models.list():
                if hasattr(model, 'name'):
                    models.append(model.name)
            
            return models

        except Exception as e:
            status_code = getattr(e, 'status_code', None)
            response_body = str(e)
            raise map_provider_error("Gemini", e, status_code=status_code, response_body=response_body)

    def _prepare_content_and_config(self, request: ChatCompletionRequest) -> tuple:
        """
        Prepare content and config for Gemini API from our messages.

        Args:
            request (ChatCompletionRequest): The request to prepare for.

        Returns:
            tuple: (content, config, tools) for Gemini API.
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

        # Create a config object using new types structure
        config = types.GenerateContentConfig(**config_params)

        # Prepare content based on non-system messages
        # For simple queries with just one user message, use a simple string
        if len(messages) == 1 and messages[0].role == "user":
            content = _flatten_text(messages[0].content)
        else:
            # For more complex exchanges, format as a conversation
            content = []

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

        if system_message:
            # Prepend system message to the first user message, or add it as the first message
            if content and isinstance(content, list) and content and 'role' in content[0] and content[0]['role'] == 'user':
                if isinstance(content[0]['parts'], list) and 'text' in content[0]['parts'][0]:
                    content[0]['parts'][0]['text'] = f"{system_message}\n{content[0]['parts'][0]['text']}"
                else:
                    # Fallback if structure is not as expected
                    content.insert(0, {"role": "user", "parts": [
                                    {"text": system_message}]})
            else:
                # If no user message, add system message as user message
                content.insert(0, {"role": "user", "parts": [
                                {"text": system_message}]})

        return content, config, gemini_tools

    async def acomplete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make an async chat completion request to Gemini.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Gemini-specific parameters.

        Returns:
            ChatCompletionResponse: The completion response.

        Raises:
            Exception: If the request fails.
        """
        client = await self._get_client()

        if self.api_key is None:
            raise ValueError("Gemini API key is required")

        try:
            # Get model from request or use a default
            model = request.model or "gemini-1.5-flash"

            # Prepare content and get gemini_tools
            content, config, gemini_tools = self._prepare_content_and_config(request)

            # Prepare API call parameters
            api_params = {
                "model": model,
                "contents": content if isinstance(content, str) else [content] if isinstance(content, list) else [],
                "config": config
            }

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
                api_params["tools"] = [Tool(function_declarations=tool_declarations)]

            # Add additional provider-specific parameters
            api_params.update(provider_specific_kwargs)

            # Make async API call using the modern client
            response = await client.models.generate_content(**api_params)

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
                    content_text = response.text
                except ValueError:
                    # Fallback to iterating parts if .text fails or parts exist but .text is empty.
                    # Concatenate text from all text-bearing parts in the response.
                    all_parts_text_list = []
                    for part in response.parts:
                        # Check if the part has a 'text' attribute and if it's not None or an empty string.
                        part_text_content = getattr(part, 'text', None)
                        if part_text_content:
                            # This ensures part_text_content is not None and not an empty string
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
                print(
                    "Warning: Gemini response has no text content and no explicit block reason."
                )

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
            # Map error to a standardized format with rich info
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

    async def astream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> AsyncIterator[ChatCompletionResponse]:
        """
        Stream an async chat completion response from Gemini.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Gemini-specific parameters.

        Returns:
            AsyncIterator[ChatCompletionResponse]: An async iterator of response chunks.

        Raises:
            Exception: If the request fails.
        """
        client = await self._get_client()

        if self.api_key is None:
            raise ValueError("Gemini API key is required")

        try:
            # Get model from request or use a default
            model = request.model or "gemini-1.5-flash"

            # Prepare content and get gemini_tools
            content, config, gemini_tools = self._prepare_content_and_config(request)

            # Prepare API call parameters
            api_params = {
                "model": model,
                "contents": content if isinstance(content, str) else [content] if isinstance(content, list) else [],
                "config": config
            }

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
                api_params["tools"] = [Tool(function_declarations=tool_declarations)]

            # Add additional provider-specific parameters
            api_params.update(provider_specific_kwargs)

            # Stream the response using async client
            async for chunk in client.models.generate_content_stream(**api_params):
                # Check for content filtering
                if chunk.prompt_feedback and chunk.prompt_feedback.block_reason:
                    raise UniInferError(
                        f"Gemini content generation blocked. Reason: {chunk.prompt_feedback.block_reason}. "
                        f"Safety ratings: {chunk.prompt_feedback.safety_ratings}"
                    )

                # Extract response text
                content_text = ""
                
                if chunk.parts:
                    try:
                        content_text = chunk.text
                    except ValueError:
                        all_parts_text_list = []
                        for part in chunk.parts:
                            part_text_content = getattr(part, 'text', None)
                            if part_text_content:
                                all_parts_text_list.append(part_text_content)
                        
                        if all_parts_text_list:
                            content_text = "".join(all_parts_text_list)

                if content_text:
                    # Create a ChatMessage for this chunk
                    message = ChatMessage(
                        role="assistant",
                        content=content_text
                    )

                    usage = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }

                    yield ChatCompletionResponse(
                        message=message,
                        provider='gemini',
                        model=model,
                        usage=usage,
                        raw_response=chunk
                    )

        except Exception as e:
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
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None or not loop.is_running():
            return asyncio.run(self.acomplete(request, **provider_specific_kwargs))

        import concurrent.futures
        import threading
        result_container = []
        exception_container = []

        def run_in_thread():
            try:
                result = asyncio.run(self.acomplete(request, **provider_specific_kwargs))
                result_container.append(result)
            except Exception as e:
                exception_container.append(e)

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

        if exception_container:
            raise exception_container[0]
        return result_container[0]

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
        async def _async_gen():
            async for chunk in self.astream_complete(request, **provider_specific_kwargs):
                yield chunk

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None or not loop.is_running():
            async def _collect_chunks():
                chunks = []
                async for chunk in self.astream_complete(request, **provider_specific_kwargs):
                    chunks.append(chunk)
                return chunks
            
            for chunk in asyncio.run(_collect_chunks()):
                yield chunk
            return

        import concurrent.futures
        import threading
        result_container = []
        exception_container = []

        def run_in_thread():
            try:
                for chunk in asyncio.run(_async_gen()):
                    result_container.append(chunk)
            except Exception as e:
                exception_container.append(e)

        thread = threading.Thread(target=run_in_thread)
        thread.start()
        thread.join()

        if exception_container:
            raise exception_container[0]

        while result_container:
            yield result_container.pop(0)
