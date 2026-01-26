"""
Moonshot provider implementation.
"""
import requests
from typing import Dict, Any, Iterator, Optional

from ..core import ChatProvider, ChatCompletionRequest, ChatCompletionResponse, ChatMessage

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


class MoonshotProvider(ChatProvider):
    """
    Provider for Moonshot AI API.

    Moonshot AI is a China-based LLM provider that uses the OpenAI client format.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.moonshot.cn/v1", **kwargs):
        """
        Initialize the Moonshot provider.

        Args:
            api_key (Optional[str]): The Moonshot API key.
            base_url (str): The base URL for the Moonshot API.
            **kwargs: Additional configuration options.
        """
        super().__init__(api_key)
        self.base_url = base_url

        if not HAS_OPENAI:
            raise ImportError(
                "openai package is required for the MoonshotProvider. "
                "Install it with: pip install openai"
            )

        # Initialize the OpenAI client for Moonshot
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    @classmethod
    def list_models(cls, api_key: Optional[str] = None) -> list:
        """
        List available models from Moonshot AI.

        Args:
            api_key (Optional[str]): The Moonshot API key.

        Returns:
            list: A list of available model names.
        """
        if api_key is None:
            try:
                from credgoo.credgoo import get_api_key
                api_key = get_api_key("moonshot")
                if api_key is None:
                    raise ValueError(
                        "Failed to retrieve Moonshot API key from credgoo")
            except ImportError:
                raise ValueError(
                    "Moonshot API key is required when credgoo is not available")

        headers = {
            "Authorization": f"Bearer {api_key}"
        }

        try:
            response = requests.get(
                "https://api.moonshot.cn/v1/models",
                headers=headers
            )
            response.raise_for_status()

            models_data = response.json()
            return [model["id"] for model in models_data.get("data", [])]
        except Exception as e:
            status_code = getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            response_body = getattr(e.response, 'text', None) if hasattr(e, 'response') else None
            raise map_provider_error("moonshot", e, status_code=status_code, response_body=response_body)

        if not HAS_OPENAI:
            raise ImportError(
                "openai package is required for the MoonshotProvider. "
                "Install it with: pip install openai"
            )

        # Initialize the OpenAI client for Moonshot
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> ChatCompletionResponse:
        """
        Make a chat completion request to Moonshot.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Moonshot-specific parameters.

        Returns:
            ChatCompletionResponse: The completion response.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("Moonshot API key is required")

        # Prepare messages in the OpenAI format
        messages = [msg.to_dict() for msg in request.messages]

        # Prepare parameters
        params = {
            "model": request.model or "moonshot-v1-8k",  # Default model
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

            # Extract the response content and tool calls
            response_message = completion.choices[0].message
            
            # Extract tool calls if present
            tool_calls = None
            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response_message.tool_calls
                ]
            
            message = ChatMessage(
                role=response_message.role,
                content=response_message.content,
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
                raw_response = {
                    "choices": [{"message": message.to_dict()}],
                    "model": params["model"],
                    "usage": usage
                }

            return ChatCompletionResponse(
                message=message,
                provider='moonshot',
                model=params["model"],
                usage=usage,
                raw_response=raw_response
            )
        except Exception as e:
            status_code = getattr(e, 'status_code', None)
            response_body = getattr(e, 'response', None)
            if hasattr(response_body, 'text'):
                response_body = response_body.text
            raise map_provider_error("moonshot", e, status_code=status_code, response_body=str(response_body) if response_body else None)

    def stream_complete(
        self,
        request: ChatCompletionRequest,
        **provider_specific_kwargs
    ) -> Iterator[ChatCompletionResponse]:
        """
        Stream a chat completion response from Moonshot.

        Args:
            request (ChatCompletionRequest): The request to make.
            **provider_specific_kwargs: Additional Moonshot-specific parameters.

        Returns:
            Iterator[ChatCompletionResponse]: An iterator of response chunks.

        Raises:
            Exception: If the request fails.
        """
        if self.api_key is None:
            raise ValueError("Moonshot API key is required")

        # Prepare messages in the OpenAI format
        messages = [msg.to_dict() for msg in request.messages]

        # Prepare parameters
        params = {
            "model": request.model or "moonshot-v1-8k",  # Default model
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
                if hasattr(chunk.choices[0], 'delta'):
                    delta = chunk.choices[0].delta
                    
                    # Extract content and tool_calls from delta
                    content = getattr(delta, 'content', None)
                    tool_calls = None
                    
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        tool_calls = [
                            {
                                "id": getattr(tc, 'id', None),
                                "type": getattr(tc, 'type', 'function'),
                                "function": {
                                    "name": tc.function.name if hasattr(tc, 'function') else None,
                                    "arguments": tc.function.arguments if hasattr(tc, 'function') else None
                                }
                            }
                            for tc in delta.tool_calls
                        ]
                    
                    if content or tool_calls:
                        # Create a message for this chunk
                        message = ChatMessage(
                            role="assistant",
                            content=content,
                            tool_calls=tool_calls
                        )

                        # No detailed usage stats in streaming mode
                        usage = {}

                        yield ChatCompletionResponse(
                            message=message,
                            provider='moonshot',
                            model=params["model"],
                            usage=usage,
                            raw_response={"chunk": {"content": content}}
                        )
        except Exception as e:
            status_code = getattr(e, 'status_code', None)
            response_body = getattr(e, 'response', None)
            if hasattr(response_body, 'text'):
                response_body = response_body.text
            raise map_provider_error("moonshot", e, status_code=status_code, response_body=str(response_body) if response_body else None)
