"""
SambaNova provider implementation.
"""
from typing import Dict, Any, Iterator, Optional, List
import os

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage
from ..errors import map_provider_error

try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class SambanovaProvider(ChatProvider):
    """
    Provider for SambaNova AI API.

    SambaNova is an AI hardware and software provider with a focus on enterprise AI solutions.
    The API uses an OpenAI-compatible interface.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.sambanova.ai/v1", **kwargs):
        """
        Initialize the SambaNova provider.

        Args:
            api_key (Optional[str]): The SambaNova API key.
            base_url (str): The base URL for the SambaNova API.
            **kwargs: Additional configuration options.
        """
        super().__init__(api_key)
        self.base_url = base_url

        if not HAS_OPENAI:
            raise ImportError(
                "openai package is required for the SambanovaProvider. "
                "Install it with: pip install openai"
            )

        # Initialize the OpenAI client for SambaNova
        self.client = openai.OpenAI(
            api_key=self.api_key or os.environ.get("SAMBANOVA_API_KEY"),
            base_url=self.base_url
        )

    @classmethod
    def list_models(cls) -> List[str]:
        """
        List available models from SambaNova API.

        Returns:
            List[str]: List of available model names
        """
        return [
            "sambastudio-7b",
            "sambastudio-13b",
            "sambastudio-20b",
            "sambastudio-70b"
        ]

    def complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make a chat completion request to SambaNova.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional SambaNova-specific parameters.

        Returns:
            ChatCompletionResponse: The completion response.

        Raises:
            Exception: If the request fails.
        """
        # Format messages for SambaNova - using standard format as in the test code
        def _flatten_messages(msgs: List[ChatMessage]) -> List[Dict[str, Any]]:
            out = []
            for m in msgs:
                md = {"role": m.role, "content": m.content}
                content = md.get("content")
                if isinstance(content, list):
                    parts: List[str] = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                    md["content"] = "".join(parts) if parts else "".join(str(p) for p in content)
                out.append(md)
            return out

        messages = _flatten_messages(request.messages)

        # Prepare parameters
        params = {
            "model": request.model or "Meta-Llama-3.1-8B-Instruct",  # Updated default model
            "messages": messages,
            "temperature": request.temperature,
        }

        # Add max_tokens if provided
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens

        # Add tools and tool_choice if provided
        if request.tools:
            params["tools"] = request.tools
        if request.tool_choice:
            params["tool_choice"] = request.tool_choice

        # Add any provider-specific parameters
        params.update(provider_specific_kwargs)

        try:
            # Make the chat completion request
            completion = self.client.chat.completions.create(**params)

            # Extract the response content
            content = ""
            raw_content = completion.choices[0].message.content

            # Handle different content formats
            if isinstance(raw_content, list):
                for item in raw_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        content += item.get("text", "")
                    elif isinstance(item, str):
                        content += item
            else:
                content = raw_content

            # Extract tool calls if present (OpenAI-compatible)
            tool_calls = None
            try:
                tc = getattr(completion.choices[0].message, 'tool_calls', None)
                if tc:
                    tool_calls = []
                    for t in tc:
                        func_name = None
                        func_args = None
                        try:
                            func = getattr(t, 'function', None)
                            if func:
                                func_name = getattr(func, 'name', None)
                                func_args = getattr(func, 'arguments', None)
                        except Exception:
                            pass
                        tool_calls.append({
                            "id": getattr(t, 'id', None),
                            "type": getattr(t, 'type', 'function'),
                            "function": {
                                "name": func_name,
                                "arguments": func_args
                            }
                        })
            except Exception:
                tool_calls = None

            message = ChatMessage(
                role=completion.choices[0].message.role,
                content=content,
                tool_calls=tool_calls
            )

            # Extract usage information
            usage = {}
            if hasattr(completion, 'usage'):
                usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens
                }

            # Create raw response
            try:
                raw_response = completion.model_dump_json()
            except AttributeError:
                # Fallback to a simple dict
                raw_response = {
                    "choices": [{"message": {"role": message.role, "content": message.content}}],
                    "model": params["model"],
                    "usage": usage
                }

            return ChatCompletionResponse(
                message=message,
                provider='sambanova',
                model=params["model"],
                usage=usage,
                raw_response=raw_response
            )
        except Exception as e:
            status_code = getattr(e, 'status_code', None)
            response_body = getattr(e, 'response', None)
            if hasattr(response_body, 'text'):
                response_body = response_body.text
            raise map_provider_error("sambanova", e, status_code=status_code, response_body=str(response_body) if response_body else str(e))

    @classmethod
    def list_models(cls, api_key: Optional[str] = None, base_url: str = "https://api.sambanova.ai/v1") -> List[str]:
        """
        List available models from SambaNova.

        Args:
            api_key (Optional[str]): The SambaNova API key.
            base_url (str): The base URL for the SambaNova API.

        Returns:
            List[str]: A list of available model names.

        Raises:
            Exception: If the request fails.
        """
        if not HAS_OPENAI:
            raise ImportError(
                "The 'openai' package is required to use the SambaNova provider. "
                "Install it with 'pip install openai'"
            )

        # Try to get API key from credgoo if not provided
        if api_key is None:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("sambanova")
                if api_key is None:
                    raise ValueError(
                        "Failed to retrieve SambaNova API key from credgoo")
            except ImportError:
                raise ValueError(
                    "SambaNova API key is required when credgoo is not available")

        # If we still don't have an API key, return default models
        if api_key is None:
            return [
                "Meta-Llama-3.1-8B-Instruct",
                "sambastudio-7b",
                "sambastudio-13b",
                "sambastudio-20b",
                "sambastudio-70b"
            ]

        # Initialize a temporary client
        client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key
        )

        try:
            # Use the OpenAI client to fetch models from SambaNova
            response = client.models.list()

            # Extract model IDs from the response
            models = [model.id for model in response.data]

            return models

        except Exception as e:
            # Map error for visibility
            status_code = getattr(e, 'status_code', None)
            response_body = getattr(e, 'response', None)
            if hasattr(response_body, 'text'):
                response_body = response_body.text
            
            # Map the error
            try:
                mapped_error = map_provider_error("sambanova", e, status_code=status_code, response_body=str(response_body) if response_body else None)
            except:
                mapped_error = e

            # Fallback to default models if API call fails
            print(f"Warning: Failed to fetch SambaNova models: {str(e)}")
            return [
                "Meta-Llama-3.1-8B-Instruct",
                "sambastudio-7b",
                "sambastudio-13b",
                "sambastudio-20b",
                "sambastudio-70b"
            ]

    def stream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> Iterator[ChatCompletionResponse]:
        """
        Stream a chat completion response from SambaNova.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional SambaNova-specific parameters.

        Returns:
            Iterator[ChatCompletionResponse]: An iterator of response chunks.

        Raises:
            Exception: If the request fails.
        """
        # Format messages for SambaNova - using standard format as in the test code
        def _flatten_messages(msgs: List[ChatMessage]) -> List[Dict[str, Any]]:
            out = []
            for m in msgs:
                md = {"role": m.role, "content": m.content}
                content = md.get("content")
                if isinstance(content, list):
                    parts: List[str] = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            parts.append(part.get("text", ""))
                    md["content"] = "".join(parts) if parts else "".join(str(p) for p in content)
                out.append(md)
            return out

        messages = _flatten_messages(request.messages)

        # Prepare parameters
        params = {
            "model": request.model or "Meta-Llama-3.1-8B-Instruct",  # Updated default model
            "messages": messages,
            "temperature": request.temperature,
            "stream": True
        }

        # Add max_tokens if provided
        if request.max_tokens is not None:
            params["max_tokens"] = request.max_tokens

        # Add tools and tool_choice if provided
        if request.tools:
            params["tools"] = request.tools
        if request.tool_choice:
            params["tool_choice"] = request.tool_choice

        # Add any provider-specific parameters
        params.update(provider_specific_kwargs)

        try:
            # Make the streaming request
            stream = self.client.chat.completions.create(**params)

            for chunk in stream:
                # Extract content/tool_calls from the chunk
                content = ""
                chunk_tool_calls = None

                # The way to access content might differ based on the chunk structure
                try:
                    if hasattr(chunk.choices[0], 'delta'):
                        delta = chunk.choices[0].delta

                        # Check if content is available in delta
                        if hasattr(delta, 'content'):
                            content = delta.content or ""
                        try:
                            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                                chunk_tool_calls = []
                                for t in delta.tool_calls:
                                    func_name = None
                                    func_args = None
                                    try:
                                        func = getattr(t, 'function', None)
                                        if func:
                                            func_name = getattr(func, 'name', None)
                                            func_args = getattr(func, 'arguments', None)
                                    except Exception:
                                        pass
                                    chunk_tool_calls.append({
                                        "id": getattr(t, 'id', None),
                                        "type": getattr(t, 'type', 'function'),
                                        "function": {
                                            "name": func_name,
                                            "arguments": func_args
                                        }
                                    })
                        except Exception:
                            chunk_tool_calls = None
                except Exception:
                    # If there was an error accessing the content, try alternate methods
                    try:
                        # Try to get content directly from choices
                        content = chunk.choices[0].text
                    except:
                        pass

                # Skip empty chunks that have neither content nor tool_calls
                if not content and not chunk_tool_calls:
                    continue

                # Create a message for this chunk
                message = ChatMessage(role="assistant", content=content if content else None, tool_calls=chunk_tool_calls)

                # No usage stats in streaming mode
                usage = {}

                yield ChatCompletionResponse(
                    message=message,
                    provider='sambanova',
                    model=params["model"],
                    usage=usage,
                    raw_response={"delta": {"content": content}}
                )
        except Exception as e:
            status_code = getattr(e, 'status_code', None)
            response_body = getattr(e, 'response', None)
            if hasattr(response_body, 'text'):
                response_body = response_body.text
            raise map_provider_error("sambanova", e, status_code=status_code, response_body=str(response_body) if response_body else str(e))
